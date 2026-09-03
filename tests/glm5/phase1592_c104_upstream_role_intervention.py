#!/usr/bin/env python3
"""Phase1592 / C104: causal intervention on frozen upstream role-state candidates."""
from __future__ import annotations

import argparse
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
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE = 1592
CAMPAIGN = "C104"
DIM = 2560
FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
MODES = ("correct_role_state", "sign_reversed", "same_truth_role_state", "coordinate_permuted_correct")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pairs(rows: list[dict[str, Any]], authorized: list[str], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] in authorized and row["partition"] in ("confirmation", "lockbox"):
            by_unit[row["unit_id"]].append(row)
    pairs = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        family = unit_rows[0]["family"]
        role = predictions[family]["role"]
        for surface, distractor, code in itertools.product((1, -1), repeat=3):
            recipient = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (-1, surface, distractor, code))
            donor = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (1, surface, distractor, code))
            same_truth = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (-1, surface, -distractor, code))
            lengths = (len(recipient["role_positions"][role]), len(donor["role_positions"][role]), len(same_truth["role_positions"][role]))
            if len(set(lengths)) != 1:
                raise RuntimeError((unit_id, family, role, lengths))
            pairs.append({
                "unit_id": unit_id,
                "family": family,
                "partition": unit_rows[0]["partition"],
                "surface_factor": surface,
                "distractor_factor": distractor,
                "code": code,
                "role": role,
                "span_length": lengths[0],
                "recipient": recipient,
                "donor": donor,
                "same_truth_donor": same_truth,
            })
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"c104-pair-{index:04d}"
    return pairs


def prepare() -> None:
    final = core.load(OUT / "analysis/frozen_candidate_validation_final.json")
    audit = core.load(OUT / "audit/independent_frozen_candidate_validation_final_audit.json")
    contract = core.load(OUT / "protocol/preregistration.json")
    if final["authorization"] != "run_phase1592_c104_upstream_role_intervention" or not audit["all_checks_passed"]:
        raise RuntimeError("C104 intervention authorization missing")
    predictions = {row["family"]: row for row in contract["predictions"]}
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    pairs = build_pairs(rows, final["formal_replication_families"], predictions)
    manifest = [{
        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": pair["family"],
        "partition": pair["partition"], "code": pair["code"], "role": pair["role"],
        "span_length": pair["span_length"], "recipient_case_id": pair["recipient"]["case_id"],
        "donor_case_id": pair["donor"]["case_id"], "same_truth_donor_case_id": pair["same_truth_donor"]["case_id"],
    } for pair in pairs]
    manifest_path = OUT / "protocol/upstream_intervention_pair_manifest.jsonl"
    core.write_rows(manifest_path, manifest)
    permutations = {}
    for family_index, family in enumerate(FAMILIES):
        permutations[family] = np.random.default_rng(15920 + family_index).permutation(DIM).tolist()
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_upstream_full_role_state_intervention_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "validation_final_sha256": core.sha(OUT / "analysis/frozen_candidate_validation_final.json"),
        "pair_manifest_sha256": core.sha(manifest_path),
        "pairs": len(pairs),
        "predictions": contract["predictions"],
        "modes": list(MODES),
        "coordinate_permutations": permutations,
        "partitions": ["confirmation", "lockbox"],
        "code_strata": ["standard", "reversed"],
        "readout": "Yes-minus-No candidate margin; positive gain means movement from false recipient toward true donor",
        "adjudication": "report each family x partition x code stratum; preserve partial regularities, no project-level conjunctive gate",
        "claim_boundary": "whole role-span activation-state sufficiency at a frozen upstream state; no semantic neuron, weight, attention, MLP, or cross-model claim",
        "authorization": "execute_qwen_upstream_role_intervention",
    }
    core.save(OUT / "protocol/upstream_intervention_protocol.json", protocol)
    print(json.dumps({key: value for key, value in protocol.items() if key != "coordinate_permutations"}, indent=2))


def fixed_positions(rows: list[dict[str, Any]], role: str, device: torch.device) -> torch.Tensor:
    lengths = {len(row["role_positions"][role]) for row in rows}
    if len(lengths) != 1:
        raise RuntimeError((role, lengths))
    return torch.tensor([row["role_positions"][role] for row in rows], dtype=torch.long, device=device)


def forward_with_role(model: Any, rows: list[dict[str, Any]], role: str, state_index: int,
                      pad: int, device: torch.device, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    role_positions = fixed_positions(rows, role, device)
    batch_indices = torch.arange(len(rows), device=device)[:, None]
    captured = []
    def hook(module: Any, args: tuple[torch.Tensor, ...]):
        captured.append(args[0][batch_indices, role_positions].detach().clone())
    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                             output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    logits = model.lm_head(output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]).float()
    if len(captured) != 1:
        raise RuntimeError(("capture hook count", len(captured)))
    return logits, captured[0]


def forward_patched(model: Any, rows: list[dict[str, Any]], role: str, state_index: int,
                    patch_values: torch.Tensor, pad: int, device: torch.device, width: int) -> torch.Tensor:
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    role_positions = fixed_positions(rows, role, device)
    batch_indices = torch.arange(len(rows), device=device)[:, None]
    count = []
    def hook(module: Any, args: tuple[torch.Tensor, ...]):
        updated = args[0].clone()
        updated[batch_indices, role_positions] = patch_values
        count.append(1)
        return (updated, *args[1:])
    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
                             output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    logits = model.lm_head(output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]).float()
    if len(count) != 1:
        raise RuntimeError(("patch hook count", len(count)))
    return logits


def yes_margin(logits: torch.Tensor, row: dict[str, Any]) -> float:
    no_token = int(row["candidate_ids"][0][0])
    yes_token = int(row["candidate_ids"][1][0])
    return float(logits[yes_token] - logits[no_token])


@torch.inference_mode()
def execute() -> None:
    protocol = core.load(OUT / "protocol/upstream_intervention_protocol.json")
    final = core.load(OUT / "analysis/frozen_candidate_validation_final.json")
    if protocol["authorization"] != "execute_qwen_upstream_role_intervention" or protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("C104 intervention execution not authorized")
    predictions = {row["family"]: row for row in protocol["predictions"]}
    pairs = build_pairs(core.rows(OUT / "compiled/qwen3.jsonl"), final["formal_replication_families"], predictions)
    if len(pairs) != protocol["pairs"]:
        raise RuntimeError((len(pairs), protocol["pairs"]))
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["family"], pair["partition"], pair["code"], pair["span_length"])].append(pair)
    results = []
    model = None
    first_repeat = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        width = 224
        for (family, partition, code, span_length), group in sorted(grouped.items()):
            role = predictions[family]["role"]
            state_index = int(predictions[family]["state"])
            permutation = torch.tensor(protocol["coordinate_permutations"][family], dtype=torch.long, device=device)
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                same_donors = [pair["same_truth_donor"] for pair in batch]
                recipient_logits, recipient_state = forward_with_role(model, recipients, role, state_index, pad, device, width)
                donor_logits, donor_state = forward_with_role(model, donors, role, state_index, pad, device, width)
                _, same_state = forward_with_role(model, same_donors, role, state_index, pad, device, width)
                if first_repeat is None:
                    first_repeat = (recipients, role, state_index, recipient_logits.detach().clone(), recipient_state.detach().clone())
                patch_targets = {
                    "correct_role_state": donor_state,
                    "sign_reversed": 2.0 * recipient_state - donor_state,
                    "same_truth_role_state": same_state,
                    "coordinate_permuted_correct": donor_state[..., permutation],
                }
                patched_logits = {mode: forward_patched(model, recipients, role, state_index, values, pad, device, width)
                                  for mode, values in patch_targets.items()}
                for local, pair in enumerate(batch):
                    recipient_margin = yes_margin(recipient_logits[local], pair["recipient"])
                    donor_margin = yes_margin(donor_logits[local], pair["donor"])
                    modes = {}
                    for mode, logits in patched_logits.items():
                        margin = yes_margin(logits[local], pair["recipient"])
                        modes[mode] = {"yes_margin": margin, "true_direction_gain": margin - recipient_margin}
                    results.append({
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family,
                        "partition": partition, "code": code, "codebook": pair["recipient"]["codebook"],
                        "role": role, "state": state_index, "span_length": span_length,
                        "recipient_yes_margin": recipient_margin, "donor_yes_margin": donor_margin,
                        "donor_true_direction_gap": donor_margin - recipient_margin,
                        "modes": modes,
                    })
                print(f"[phase1592] {family}/{partition}/code={code}/span={span_length} {start + len(batch)}/{len(group)}", flush=True)
        if first_repeat is None:
            raise RuntimeError("repeat batch missing")
        repeat_rows, repeat_role, repeat_state_index, old_logits, old_state = first_repeat
        logits, state = forward_with_role(model, repeat_rows, repeat_role, repeat_state_index, pad, device, width)
        repeat_hidden = float(torch.max(torch.abs(state - old_state)).item())
        repeat_logits = float(torch.max(torch.abs(logits - old_logits)).item())
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
    results_path = OUT / "analysis/upstream_role_intervention_results.jsonl"
    core.write_rows(results_path, results)
    summaries = []
    for family in FAMILIES:
        for partition in ("confirmation", "lockbox"):
            for code in (1, -1):
                selected = [row for row in results if row["family"] == family and row["partition"] == partition and row["code"] == code]
                medians = {mode: float(np.median([row["modes"][mode]["true_direction_gain"] for row in selected])) for mode in MODES}
                summaries.append({
                    "family": family, "partition": partition, "code": code,
                    "codebook": selected[0]["codebook"], "role": selected[0]["role"], "state": selected[0]["state"],
                    "pairs": len(selected), "median_donor_true_direction_gap": float(np.median([row["donor_true_direction_gap"] for row in selected])),
                    "median_true_direction_gain": medians,
                    "correct_positive": medians["correct_role_state"] > 0.0,
                    "correct_beats_all_controls": all(medians["correct_role_state"] > medians[mode] for mode in MODES if mode != "correct_role_state"),
                })
    summary_path = OUT / "analysis/upstream_role_intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    family_rollup = []
    for family in FAMILIES:
        rows_family = [row for row in summaries if row["family"] == family]
        family_rollup.append({
            "family": family,
            "positive_cells": sum(row["correct_positive"] for row in rows_family),
            "controlled_cells": sum(row["correct_positive"] and row["correct_beats_all_controls"] for row in rows_family),
            "total_cells": len(rows_family),
            "all_partition_code_cells_controlled": all(row["correct_positive"] and row["correct_beats_all_controls"] for row in rows_family),
        })
    rollup_path = OUT / "analysis/upstream_role_intervention_family_rollup.jsonl"
    core.write_rows(rollup_path, family_rollup)
    checks = {
        "pairs": len(results) == protocol["pairs"] == 192,
        "summary": len(summaries) == 16,
        "finite": all(math.isfinite(value) for row in results for value in [row["recipient_yes_margin"], row["donor_yes_margin"], row["donor_true_direction_gap"], *[entry["true_direction_gain"] for entry in row["modes"].values()]]),
        "repeat_hidden": repeat_hidden == 0.0,
        "repeat_logits": repeat_logits == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_upstream_full_role_state_intervention_complete",
        "pairs": len(results),
        "results_sha256": core.sha(results_path),
        "summary_sha256": core.sha(summary_path),
        "rollup_sha256": core.sha(rollup_path),
        "fully_controlled_families": [row["family"] for row in family_rollup if row["all_partition_code_cells_controlled"]],
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits},
        "checks": checks,
        "runtime": {"placement": placement, "quantization": quant},
        "interpretation": "whole upstream role-state transport is a causal sufficiency test stratified by output code; failure does not erase predictive barcode replication",
        "authorization": "export_and_close_c104_major_stage",
    }
    core.save(OUT / "analysis/upstream_role_intervention_final.json", report)
    print(json.dumps({"report": {key: value for key, value in report.items() if key != "runtime"}, "rollup": family_rollup, "summary": summaries}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else execute()


if __name__ == "__main__":
    main()
