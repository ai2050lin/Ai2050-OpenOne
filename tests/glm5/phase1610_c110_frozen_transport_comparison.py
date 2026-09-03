#!/usr/bin/env python3
"""Phase1610 / C110: compare frozen, energy-matched, and multi-role transports."""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
C108 = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

MODES = ("frozen_support", "wrong_same_k", "wrong_l2_matched", "coordinate_permuted", "whole_query_anchor", "whole_query_anchor_plus_focus_record")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pairs(rows: list[dict], protocol: dict) -> list[dict]:
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_unit[row["unit_id"]].append(row)
    pairs = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        for surface, distractor, code in itertools.product((1, -1), repeat=3):
            def pick(truth: int) -> dict:
                return next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, distractor, code))
            recipient, donor = pick(-1), pick(1)
            q_lengths = {len(row["role_positions"]["query_anchor"]) for row in (recipient, donor)}
            f_lengths = {len(row["role_positions"]["focus_record"]) for row in (recipient, donor)}
            if len(q_lengths) != 1 or len(f_lengths) != 1:
                raise RuntimeError((unit_id, q_lengths, f_lengths))
            pairs.append({
                "pair_id": f"c110-pair-{len(pairs):04d}", "unit_id": unit_id, "family": unit_rows[0]["family"], "partition": unit_rows[0]["partition"],
                "surface_factor": surface, "distractor_factor": distractor, "code": code,
                "query_span": next(iter(q_lengths)), "focus_record_span": next(iter(f_lengths)), "recipient": recipient, "donor": donor,
            })
    return pairs


def fixed_positions(rows: list[dict], role: str, device: torch.device) -> torch.Tensor:
    lengths = {len(row["role_positions"][role]) for row in rows}
    if len(lengths) != 1:
        raise RuntimeError((role, lengths))
    return torch.tensor([row["role_positions"][role] for row in rows], dtype=torch.long, device=device)


def forward_with_roles(model: Any, rows: list[dict], roles: tuple[str, ...], state: int, pad: int, device: torch.device, width: int):
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    role_positions = {role: fixed_positions(rows, role, device) for role in roles}
    batch_indices = torch.arange(len(rows), device=device)[:, None]
    captured = []
    def hook(module: Any, args: tuple[torch.Tensor, ...]):
        captured.append({role: args[0][batch_indices, role_positions[role]].detach().clone() for role in roles})
    handle = model.model.layers[state].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    logits = model.lm_head(output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]).float()
    if len(captured) != 1:
        raise RuntimeError(("capture hook", len(captured)))
    return logits, captured[0]


def forward_patched_roles(model: Any, rows: list[dict], role_values: dict[str, torch.Tensor], state: int, pad: int, device: torch.device, width: int):
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    role_positions = {role: fixed_positions(rows, role, device) for role in role_values}
    batch_indices = torch.arange(len(rows), device=device)[:, None]
    count = []
    def hook(module: Any, args: tuple[torch.Tensor, ...]):
        updated = args[0].clone()
        for role, values in role_values.items():
            updated[batch_indices, role_positions[role]] = values
        count.append(1)
        return (updated, *args[1:])
    handle = model.model.layers[state].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    logits = model.lm_head(output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]).float()
    if len(count) != 1:
        raise RuntimeError(("patch hook", len(count)))
    return logits


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[int(row["candidate_ids"][0][0])] - logits[int(row["candidate_ids"][1][0])])


@torch.inference_mode()
def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    field_report = core.load(OUT / "analysis/field_prediction_adjudication.json")
    field_audit = core.load(OUT / "audit/independent_field_adjudication_audit.json")
    if field_report["authorization"] != "execute_phase1610_c110_frozen_transport_comparison_regardless_of_field_gate" or not field_audit["all_checks_passed"]:
        raise RuntimeError("C110 transport authorization missing")
    c108_protocol = core.load(C108 / "protocol/preregistration.json")
    adapter = {
        "phase": 1610, "campaign": "C110", "created_at_utc": now(), "status": "transport_implementation_adapter_frozen_before_transport_run",
        "contract_sha256": core.sha(OUT / "protocol/preregistration.json"), "field_report_sha256": core.sha(OUT / "analysis/field_prediction_adjudication.json"),
        "modes": list(MODES), "state": 19, "single_role": "query_anchor", "multi_roles": ["query_anchor", "focus_record"],
        "coordinate_permutations": c108_protocol["coordinate_permutations"],
        "permutation_source_sha256": core.sha(C108 / "protocol/preregistration.json"),
        "energy_match": protocol["energy_match"], "energy_match_relative_tolerance_bf16": 0.01,
        "producer_sha256": core.sha(Path(__file__)), "authorization": "execute_frozen_transport_modes",
    }
    core.save(OUT / "protocol/transport_adapter.json", adapter)
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = build_pairs(rows, protocol)
    if len(pairs) != 192:
        raise RuntimeError(len(pairs))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["family"], pair["partition"], pair["code"], pair["query_span"], pair["focus_record_span"])].append(pair)
    model = None
    results = []
    first_repeat = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for (family, partition, code, query_span, focus_span), group in sorted(grouped.items()):
            target_values = protocol["supports"]["attribute_binding_k256" if family == "attribute_binding" else "agent_patient_k128"]
            wrong_values = protocol["supports"]["attribute_wrong_agent_k256" if family == "attribute_binding" else "agent_wrong_attribute_k128"]
            support = torch.tensor(target_values, dtype=torch.long, device=device)
            wrong = torch.tensor(wrong_values, dtype=torch.long, device=device)
            permutation = torch.tensor(adapter["coordinate_permutations"][family], dtype=torch.long, device=device)
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                recipients = [row["recipient"] for row in batch]
                donors = [row["donor"] for row in batch]
                recipient_logits, recipient_states = forward_with_roles(model, recipients, ("query_anchor", "focus_record"), 19, pad, device, 224)
                donor_logits, donor_states = forward_with_roles(model, donors, ("query_anchor", "focus_record"), 19, pad, device, 224)
                if first_repeat is None:
                    first_repeat = (recipients, recipient_logits.detach().clone(), {key: value.detach().clone() for key, value in recipient_states.items()})
                rq, dq = recipient_states["query_anchor"], donor_states["query_anchor"]
                target_delta = dq[..., support] - rq[..., support]
                wrong_delta = dq[..., wrong] - rq[..., wrong]
                target_norm = torch.sqrt(torch.sum(target_delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                wrong_norm = torch.sqrt(torch.sum(wrong_delta.float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                alpha = (target_norm / wrong_norm)[:, None, None].to(rq.dtype)
                patches: dict[str, dict[str, torch.Tensor]] = {}
                value = rq.clone(); value[..., support] = dq[..., support]; patches["frozen_support"] = {"query_anchor": value}
                value = rq.clone(); value[..., wrong] = dq[..., wrong]; patches["wrong_same_k"] = {"query_anchor": value}
                value = rq.clone(); value[..., wrong] = rq[..., wrong] + alpha * wrong_delta; patches["wrong_l2_matched"] = {"query_anchor": value}
                matched_norm = torch.sqrt(torch.sum((value[..., wrong] - rq[..., wrong]).float() ** 2, dim=(1, 2))).clamp_min(1e-12)
                value = rq.clone(); value[..., support] = dq[..., permutation[support]]; patches["coordinate_permuted"] = {"query_anchor": value}
                patches["whole_query_anchor"] = {"query_anchor": dq.clone()}
                patches["whole_query_anchor_plus_focus_record"] = {"query_anchor": dq.clone(), "focus_record": donor_states["focus_record"].clone()}
                patched_logits = {mode: forward_patched_roles(model, recipients, role_values, 19, pad, device, 224) for mode, role_values in patches.items()}
                for local, pair in enumerate(batch):
                    base = margin(recipient_logits[local], recipients[local])
                    donor_margin = margin(donor_logits[local], donors[local])
                    row_result = {
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family, "partition": partition, "code": code,
                        "surface_factor": pair["surface_factor"], "distractor_factor": pair["distractor_factor"], "state": 19,
                        "query_span": query_span, "focus_record_span": focus_span, "k": len(target_values), "recipient_yes_minus_no": base, "donor_yes_minus_no": donor_margin,
                        "target_l2": float(target_norm[local]), "wrong_l2": float(wrong_norm[local]), "modes": {},
                    }
                    for mode, logits in patched_logits.items():
                        patched = margin(logits[local], recipients[local])
                        gain = patched - base
                        if mode == "frozen_support": movement = float(target_norm[local])
                        elif mode == "wrong_same_k": movement = float(wrong_norm[local])
                        elif mode == "wrong_l2_matched": movement = float(matched_norm[local])
                        else: movement = None
                        row_result["modes"][mode] = {
                            "yes_minus_no": patched, "truth_direction_gain": gain, "code_aligned_task_gain": code * gain,
                            "truth_flip": base <= 0.0 < patched, "task_flip": code * base <= 0.0 < code * patched,
                            "l2_movement": movement, "gain_per_l2": None if movement is None else gain / max(movement, 1e-12),
                        }
                    row_result["energy_match_abs_error"] = abs(row_result["modes"]["wrong_l2_matched"]["l2_movement"] - row_result["modes"]["frozen_support"]["l2_movement"])
                    row_result["energy_match_relative_error"] = row_result["energy_match_abs_error"] / max(row_result["modes"]["frozen_support"]["l2_movement"], 1e-12)
                    results.append(row_result)
                print(f"[phase1610] {family}/{partition}/code={code}/q={query_span}/f={focus_span} {start + len(batch)}/{len(group)}", flush=True)
        repeat_rows, old_logits, old_states = first_repeat
        new_logits, new_states = forward_with_roles(model, repeat_rows, ("query_anchor", "focus_record"), 19, pad, device, 224)
        repeat_logits = float(torch.max(torch.abs(new_logits - old_logits)))
        repeat_hidden = max(float(torch.max(torch.abs(new_states[role] - old_states[role]))) for role in old_states)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
    results_path = OUT / "analysis/fresh_transport_results.jsonl"
    core.write_rows(results_path, results)
    summaries = []
    for family in protocol["families"]:
        for partition in protocol["partitions"]:
            for code in (1, -1):
                selected = [row for row in results if row["family"] == family and row["partition"] == partition and row["code"] == code]
                medians = {mode: float(np.median([row["modes"][mode]["truth_direction_gain"] for row in selected])) for mode in MODES}
                efficiencies = {mode: float(np.median([row["modes"][mode]["gain_per_l2"] for row in selected])) for mode in ("frozen_support", "wrong_same_k", "wrong_l2_matched")}
                summaries.append({
                    "family": family, "partition": partition, "code": code, "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}),
                    "median_truth_direction_gain": medians, "median_gain_per_l2": efficiencies,
                    "target_efficiency_exceeds_wrong": efficiencies["frozen_support"] > efficiencies["wrong_same_k"],
                    "target_gain_exceeds_energy_matched_wrong": medians["frozen_support"] > medians["wrong_l2_matched"],
                    "multi_role_exceeds_whole_query": medians["whole_query_anchor_plus_focus_record"] > medians["whole_query_anchor"],
                    "truth_flip_rate": {mode: float(np.mean([row["modes"][mode]["truth_flip"] for row in selected])) for mode in MODES},
                })
    summary_path = OUT / "analysis/fresh_transport_summary.jsonl"
    core.write_rows(summary_path, summaries)
    attr = [row for row in summaries if row["family"] == "attribute_binding"]
    agent = [row for row in summaries if row["family"] == "agent_patient"]
    prediction = {
        "attribute_target_efficiency_gt_wrong_cells": sum(row["target_efficiency_exceeds_wrong"] for row in attr),
        "agent_target_efficiency_lt_wrong_cells": sum(not row["target_efficiency_exceeds_wrong"] for row in agent),
        "attribute_prediction_passed": all(row["target_efficiency_exceeds_wrong"] for row in attr),
        "agent_prediction_passed": all(not row["target_efficiency_exceeds_wrong"] for row in agent),
    }
    checks = {
        "adapter": adapter["authorization"] == "execute_frozen_transport_modes" and core.sha(Path(__file__)) == adapter["producer_sha256"],
        "rows": len(results) == 192, "summary": len(summaries) == 8 and all(row["pairs"] == 24 and row["independent_units"] == 6 for row in summaries),
        "modes": all(set(row["modes"]) == set(MODES) for row in results), "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in results for mode in MODES),
        "energy_match": max(row["energy_match_relative_error"] for row in results) <= adapter["energy_match_relative_tolerance_bf16"],
        "repeat_hidden": repeat_hidden == 0.0, "repeat_logits": repeat_logits == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1610, "campaign": "C110", "created_at_utc": now(), "status": "fresh_frozen_transport_comparison_complete",
        "producer_sha256": core.sha(Path(__file__)), "results_sha256": core.sha(results_path), "summary_sha256": core.sha(summary_path),
        "prediction": prediction, "summaries": summaries, "checks": checks, "runtime": {"placement": placement, "quantization": quant},
        "claim_boundary": "fresh activation transport comparison; readout stability, movement energy, and output leverage are separate; no weight, neuron, or universal semantic claim",
        "authorization": "run_phase1611_c110_synthesis_heatmap_and_major_stage_closure",
    }
    core.save(OUT / "analysis/transport_adjudication.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime" and key != "summaries"}, indent=2))


if __name__ == "__main__":
    main()
