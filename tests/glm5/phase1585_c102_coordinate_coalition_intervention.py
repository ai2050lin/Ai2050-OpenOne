#!/usr/bin/env python3
"""Phase1585 / C102: conditional residual-coordinate coalition intervention."""
from __future__ import annotations

import argparse
import gc
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
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE = 1585
CAMPAIGN = "C102"
DIM = 2560
GRAPH_FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pairs(rows: list[dict[str, Any]], authorized: list[str]) -> list[dict[str, Any]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] in authorized and row["partition"] in ("confirmation", "lockbox"):
            by_unit[row["unit_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        first = unit_rows[0]
        if first["arm"] == "graph":
            for x in (1, -1):
                for branch in (1, -1):
                    for code in (1, -1):
                        recipient = next(row for row in unit_rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, -x, branch, code))
                        donor = next(row for row in unit_rows if (row["x"], row["y"], row["branch"], row["code"]) == (x, x, branch, code))
                        same_truth = next(row for row in unit_rows if (row["x"], row["y"], row["branch"], row["code"]) == (-x, x, branch, code))
                        pairs.append({"unit_id": unit_id, "arm": "graph", "family": first["family"], "world": first["world"], "partition": first["partition"], "recipient": recipient, "donor": donor, "same_truth_donor": same_truth})
        else:
            for surface in (1, -1):
                for distractor in (1, -1):
                    for code in (1, -1):
                        recipient = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (-1, surface, distractor, code))
                        donor = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (1, surface, distractor, code))
                        same_truth = next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (-1, surface, -distractor, code))
                        pairs.append({"unit_id": unit_id, "arm": "breadth", "family": first["family"], "world": first["world"], "partition": first["partition"], "recipient": recipient, "donor": donor, "same_truth_donor": same_truth})
    for index, pair in enumerate(pairs):
        pair["pair_id"] = f"c102-pair-{index:04d}"
    return pairs


def prepare() -> None:
    final = core.load(OUT / "analysis/staged_barcode_final.json")
    audit = core.load(OUT / "audit/independent_staged_barcode_final_audit.json")
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    if final["authorization"] != "run_phase1585_conditional_coordinate_intervention" or not audit["all_checks_passed"]:
        raise RuntimeError("coordinate intervention not authorized")
    graph = [{**row, "arm": "graph"} for row in core.rows(OUT / "compiled/qwen3_graph.jsonl")]
    breadth = [{**row, "arm": "breadth"} for row in core.rows(OUT / "compiled/qwen3_breadth.jsonl")]
    pairs = build_pairs([*graph, *breadth], final["authorized_intervention_families"])
    manifest = []
    for pair in pairs:
        manifest.append({key: pair[key] for key in ("pair_id", "unit_id", "arm", "family", "world", "partition")} | {"recipient_case_id": pair["recipient"]["case_id"], "donor_case_id": pair["donor"]["case_id"], "same_truth_donor_case_id": pair["same_truth_donor"]["case_id"]})
    core.write_rows(OUT / "protocol/intervention_pair_manifest.jsonl", manifest)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "conditional_coordinate_coalition_intervention_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "staged_final_sha256": core.sha(OUT / "analysis/staged_barcode_final.json"),
        "selection_sha256": core.sha(OUT / "protocol/response_discovery_selection.json"),
        "pair_manifest_sha256": core.sha(OUT / "protocol/intervention_pair_manifest.jsonl"),
        "pairs": len(pairs),
        "partitions": ["confirmation", "lockbox"],
        "modes": ["correct_frozen_support", "sign_reversed", "same_truth_donor", "wrong_family_support", "whole_state_upper_bound"],
        "selectors": selection["selection"],
        "typed_missingness": {
            "wrong_family_support": "M_IDENTICAL_FULL_SUPPORT when K=2560",
            "whole_state_upper_bound": "M_IDENTICAL_FULL_SUPPORT when K=2560",
            "state36": "direct final-state readout intervention; no downstream propagation remains",
        },
        "adjudication": "report gains and controls per family and partition; do not require one conjunctive project gate",
        "claim_boundary": "residual activation-coordinate causal sufficiency only; no weight, attention or MLP localization",
        "authorization": "execute_qwen_coordinate_coalition_intervention",
    }
    core.save(OUT / "protocol/intervention_protocol.json", protocol)
    print(json.dumps(protocol, indent=2))


def selector_for(family: str) -> tuple[int, int, list[int]]:
    selection = core.load(OUT / "protocol/response_discovery_selection.json")["selection"][family]
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    item = next(row for row in frozen["selectors"] if row["family"] == family)
    return int(selection["state"]), int(selection["k"]), item["coordinate_rank"][: int(selection["k"])]


def wrong_coordinates(family: str, k: int) -> list[int]:
    families = GRAPH_FAMILIES if family in GRAPH_FAMILIES else BREADTH_FAMILIES
    wrong_family = families[(families.index(family) + 1) % len(families)]
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    item = next(row for row in frozen["selectors"] if row["family"] == wrong_family)
    return item["coordinate_rank"][:k]


def forward_with_state(model: Any, rows: list[dict[str, Any]], state_index: int, pad: int, device: torch.device, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    captured: list[torch.Tensor] = []
    handle = None
    if state_index < 36:
        def hook(module: Any, args: tuple[torch.Tensor, ...]):
            hidden = args[0]
            captured.append(hidden[torch.arange(len(rows), device=device), boundaries].detach().clone())
        handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
    final_boundary = output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]
    logits = model.lm_head(final_boundary).float()
    selected_state = final_boundary.detach().clone() if state_index == 36 else captured[0]
    return logits, selected_state


def forward_patched(
    model: Any,
    rows: list[dict[str, Any]],
    state_index: int,
    patch_values: torch.Tensor,
    pad: int,
    device: torch.device,
    width: int,
) -> torch.Tensor:
    if state_index == 36:
        return model.lm_head(patch_values).float()
    ids, mask, positions, lengths = fixed_base.fixed_batch(rows, pad, device, width)
    boundaries = torch.tensor([length - 1 for length in lengths], device=device)
    consistency: list[float] = []
    def hook(module: Any, args: tuple[torch.Tensor, ...]):
        hidden = args[0]
        updated = hidden.clone()
        current = hidden[torch.arange(len(rows), device=device), boundaries]
        consistency.append(float(torch.max(torch.abs(current - patch_values)).item()))
        updated[torch.arange(len(rows), device=device), boundaries] = patch_values
        return (updated, *args[1:])
    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    try:
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    boundary = output.last_hidden_state[torch.arange(len(rows), device=device), boundaries]
    logits = model.lm_head(boundary).float()
    if len(consistency) != 1:
        raise RuntimeError(("patch hook count", len(consistency)))
    return logits


def candidate_margin(logits: torch.Tensor, row: dict[str, Any]) -> float:
    no_token = int(row["candidate_ids"][0][0])
    yes_token = int(row["candidate_ids"][1][0])
    return float(logits[yes_token] - logits[no_token])


@torch.inference_mode()
def execute() -> None:
    protocol = core.load(OUT / "protocol/intervention_protocol.json")
    final = core.load(OUT / "analysis/staged_barcode_final.json")
    if protocol["authorization"] != "execute_qwen_coordinate_coalition_intervention" or protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("intervention execution not authorized")
    graph = [{**row, "arm": "graph"} for row in core.rows(OUT / "compiled/qwen3_graph.jsonl")]
    breadth = [{**row, "arm": "breadth"} for row in core.rows(OUT / "compiled/qwen3_breadth.jsonl")]
    pairs = build_pairs([*graph, *breadth], final["authorized_intervention_families"])
    if len(pairs) != protocol["pairs"]:
        raise RuntimeError((len(pairs), protocol["pairs"]))
    results: list[dict[str, Any]] = []
    model = None
    repeat_hidden = math.inf
    repeat_logits = math.inf
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        width = 224
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            grouped[(pair["family"], pair["partition"])].append(pair)
        first_repeat = None
        for (family, partition), family_pairs in sorted(grouped.items()):
            state_index, k, correct_coordinates = selector_for(family)
            wrong_support = wrong_coordinates(family, k)
            correct_tensor = torch.tensor(correct_coordinates, dtype=torch.long, device=device)
            wrong_tensor = torch.tensor(wrong_support, dtype=torch.long, device=device)
            for start in range(0, len(family_pairs), 8):
                batch = family_pairs[start:start + 8]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                same_donors = [pair["same_truth_donor"] for pair in batch]
                recipient_logits, recipient_state = forward_with_state(model, recipients, state_index, pad, device, width)
                donor_logits, donor_state = forward_with_state(model, donors, state_index, pad, device, width)
                _, same_state = forward_with_state(model, same_donors, state_index, pad, device, width)
                if first_repeat is None:
                    first_repeat = (recipients, state_index, recipient_logits.detach().clone(), recipient_state.detach().clone())
                patch_targets: dict[str, torch.Tensor] = {}
                correct = recipient_state.clone()
                correct[:, correct_tensor] = donor_state[:, correct_tensor]
                patch_targets["correct_frozen_support"] = correct
                reverse = recipient_state.clone()
                reverse[:, correct_tensor] = 2.0 * recipient_state[:, correct_tensor] - donor_state[:, correct_tensor]
                patch_targets["sign_reversed"] = reverse
                same = recipient_state.clone()
                same[:, correct_tensor] = same_state[:, correct_tensor]
                patch_targets["same_truth_donor"] = same
                wrong = recipient_state.clone()
                wrong[:, wrong_tensor] = donor_state[:, wrong_tensor]
                patch_targets["wrong_family_support"] = wrong
                patch_targets["whole_state_upper_bound"] = donor_state.clone()
                patched_logits = {mode: forward_patched(model, recipients, state_index, values, pad, device, width) for mode, values in patch_targets.items()}
                for local, pair in enumerate(batch):
                    recipient_margin = candidate_margin(recipient_logits[local], pair["recipient"])
                    donor_margin = candidate_margin(donor_logits[local], pair["donor"])
                    donor_direction = 1.0 if pair["donor"]["output_yes"] else -1.0
                    modes = {}
                    for mode, logits in patched_logits.items():
                        margin = candidate_margin(logits[local], pair["recipient"])
                        modes[mode] = {"margin": margin, "target_gain": donor_direction * (margin - recipient_margin)}
                    results.append(
                        {
                            "pair_id": pair["pair_id"],
                            "unit_id": pair["unit_id"],
                            "arm": pair["arm"],
                            "family": family,
                            "world": pair["world"],
                            "partition": partition,
                            "state": state_index,
                            "k": k,
                            "propagation_scope": "direct_final_state_readout" if state_index == 36 else f"propagates_through_layers_{state_index}_to_35",
                            "recipient_margin": recipient_margin,
                            "donor_margin": donor_margin,
                            "donor_target_gap": donor_direction * (donor_margin - recipient_margin),
                            "modes": modes,
                            "typed_missing": {"wrong_family_support": k == DIM, "whole_state_upper_bound": k == DIM},
                        }
                    )
                print(f"[phase1585] {family}/{partition} {start + len(batch)}/{len(family_pairs)}", flush=True)
        if first_repeat is None:
            raise RuntimeError("repeat batch missing")
        rows, state_index, old_logits, old_state = first_repeat
        logits, state = forward_with_state(model, rows, state_index, pad, device, width)
        repeat_hidden = float(torch.max(torch.abs(state - old_state)).item())
        repeat_logits = float(torch.max(torch.abs(logits - old_logits)).item())
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
    path = OUT / "analysis/coordinate_coalition_intervention_results.jsonl"
    core.write_rows(path, results)
    summaries = []
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        for partition in ("confirmation", "lockbox"):
            selected = [row for row in results if row["family"] == family and row["partition"] == partition]
            mode_medians = {mode: float(np.median([row["modes"][mode]["target_gain"] for row in selected])) for mode in protocol["modes"]}
            sparse = selected[0]["k"] < DIM
            informative_controls = ["sign_reversed", "same_truth_donor"] + (["wrong_family_support"] if sparse else [])
            summaries.append(
                {
                    "family": family,
                    "partition": partition,
                    "state": selected[0]["state"],
                    "k": selected[0]["k"],
                    "propagation_scope": selected[0]["propagation_scope"],
                    "pairs": len(selected),
                    "median_donor_target_gap": float(np.median([row["donor_target_gap"] for row in selected])),
                    "median_target_gain": mode_medians,
                    "correct_positive": mode_medians["correct_frozen_support"] > 0.0,
                    "correct_beats_informative_controls": all(mode_medians["correct_frozen_support"] > mode_medians[control] for control in informative_controls),
                    "typed_missing_full_support_controls": not sparse,
                }
            )
    core.write_rows(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl", summaries)
    important = []
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        rows_family = [row for row in summaries if row["family"] == family]
        if all(row["correct_positive"] and row["correct_beats_informative_controls"] for row in rows_family):
            important.append(family)
    checks = {
        "pairs": len(results) == protocol["pairs"] == 384,
        "finite": all(math.isfinite(value) for row in results for value in [row["recipient_margin"], row["donor_margin"], row["donor_target_gap"], *[entry["target_gain"] for entry in row["modes"].values()]]),
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
        "status": "conditional_coordinate_coalition_intervention_complete",
        "results_sha256": core.sha(path),
        "summary_sha256": core.sha(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl"),
        "pairs": len(results),
        "important_families": important,
        "important_count": len(important),
        "numeric": {"repeat_hidden_max_abs": repeat_hidden, "repeat_logits_max_abs": repeat_logits},
        "checks": checks,
        "runtime": {"placement": placement, "quantization": quant},
        "interpretation": "positive controlled gains imply sufficiency of a residual-state coordinate coalition at its tested boundary; K=2560 and state36 results are whole-state/direct-readout effects, not sparse semantic-neuron or propagation evidence",
        "authorization": "export_c102_coordinate_and_token_heatmap",
    }
    core.save(OUT / "analysis/coordinate_coalition_intervention_final.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else execute()


if __name__ == "__main__":
    main()
