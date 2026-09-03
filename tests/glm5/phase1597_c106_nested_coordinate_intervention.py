#!/usr/bin/env python3
"""Phase1597 / C106: execute nested activation-coordinate coalition interventions."""
from __future__ import annotations

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
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C105 = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1592_c104_upstream_role_intervention as parent

PHASE = 1597
CAMPAIGN = "C106"


def yes_minus_no(logits: torch.Tensor, row: dict[str, Any]) -> float:
    yes_token = int(row["candidate_ids"][0][0])
    no_token = int(row["candidate_ids"][1][0])
    return float(logits[yes_token] - logits[no_token])


@torch.inference_mode()
def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if protocol["authorization"] != "execute_phase1597_c106_nested_coordinate_interventions" or not audit["all_checks_passed"]:
        raise RuntimeError("C106 execution not authorized")
    predictions = {row["family"]: row for row in protocol["predictions"]}
    pairs = parent.build_pairs(core.rows(C104 / "compiled/qwen3.jsonl"), protocol["families"], predictions)
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
            rank = protocol["rankings"][family]
            permutation = torch.tensor(protocol["coordinate_permutations"][family], dtype=torch.long, device=device)
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                recipients = [pair["recipient"] for pair in batch]
                donors = [pair["donor"] for pair in batch]
                same_donors = [pair["same_truth_donor"] for pair in batch]
                recipient_logits, recipient_state = parent.forward_with_role(model, recipients, role, state_index, pad, device, width)
                donor_logits, donor_state = parent.forward_with_role(model, donors, role, state_index, pad, device, width)
                _, same_state = parent.forward_with_role(model, same_donors, role, state_index, pad, device, width)
                if first_repeat is None:
                    first_repeat = (recipients, role, state_index, recipient_logits.detach().clone(), recipient_state.detach().clone())
                recipient_margins = [yes_minus_no(recipient_logits[local], row) for local, row in enumerate(recipients)]
                donor_margins = [yes_minus_no(donor_logits[local], row) for local, row in enumerate(donors)]
                batch_results = [{
                    "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family,
                    "partition": partition, "code": code, "codebook": pair["recipient"]["codebook"],
                    "role": role, "state": state_index, "span_length": span_length,
                    "recipient_yes_minus_no": recipient_margins[local], "donor_yes_minus_no": donor_margins[local],
                    "donor_true_direction_gap": donor_margins[local] - recipient_margins[local], "nested": {},
                } for local, pair in enumerate(batch)]
                for k in protocol["nested_k"]:
                    coordinates = torch.tensor(rank[:k], dtype=torch.long, device=device)
                    patch_targets = {}
                    correct = recipient_state.clone()
                    correct[..., coordinates] = donor_state[..., coordinates]
                    patch_targets["correct_role_state"] = correct
                    reverse = recipient_state.clone()
                    reverse[..., coordinates] = 2.0 * recipient_state[..., coordinates] - donor_state[..., coordinates]
                    patch_targets["sign_reversed"] = reverse
                    same = recipient_state.clone()
                    same[..., coordinates] = same_state[..., coordinates]
                    patch_targets["same_truth_role_state"] = same
                    permuted = recipient_state.clone()
                    permuted[..., coordinates] = donor_state[..., permutation[coordinates]]
                    patch_targets["coordinate_permuted_correct"] = permuted
                    patched = {mode: parent.forward_patched(model, recipients, role, state_index, values, pad, device, width)
                               for mode, values in patch_targets.items()}
                    for local, row in enumerate(batch_results):
                        row["nested"][str(k)] = {}
                        for mode, logits in patched.items():
                            margin = yes_minus_no(logits[local], recipients[local])
                            row["nested"][str(k)][mode] = {"yes_minus_no": margin, "true_direction_gain": margin - recipient_margins[local]}
                results.extend(batch_results)
                print(f"[phase1597] {family}/{partition}/code={code}/span={span_length} {start + len(batch)}/{len(group)}", flush=True)
        if first_repeat is None:
            raise RuntimeError("repeat batch missing")
        repeat_rows, repeat_role, repeat_state, old_logits, old_state = first_repeat
        logits, state = parent.forward_with_role(model, repeat_rows, repeat_role, repeat_state, pad, device, width)
        repeat_hidden = float(torch.max(torch.abs(state - old_state)).item())
        repeat_logits = float(torch.max(torch.abs(logits - old_logits)).item())
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()

    path = OUT / "analysis/nested_coordinate_intervention_results.jsonl"
    core.write_rows(path, results)
    summaries = []
    for family in protocol["families"]:
        for k in protocol["nested_k"]:
            for partition in protocol["formal_partitions"]:
                for code in (1, -1):
                    selected = [row for row in results if row["family"] == family and row["partition"] == partition and row["code"] == code]
                    medians = {mode: float(np.median([row["nested"][str(k)][mode]["true_direction_gain"] for row in selected])) for mode in protocol["modes"]}
                    summaries.append({
                        "family": family, "k": k, "support_fraction": k / 2560.0, "partition": partition, "code": code,
                        "codebook": selected[0]["codebook"], "role": selected[0]["role"], "state": selected[0]["state"],
                        "pairs": len(selected), "median_true_direction_gain": medians,
                        "correct_positive": medians["correct_role_state"] > 0.0,
                        "correct_beats_all_controls": all(medians["correct_role_state"] > medians[mode] for mode in protocol["modes"] if mode != "correct_role_state"),
                    })
    summary_path = OUT / "analysis/nested_coordinate_intervention_summary.jsonl"
    core.write_rows(summary_path, summaries)
    family_rows = []
    for family in protocol["families"]:
        k_rows = []
        for k in protocol["nested_k"]:
            selected = [row for row in summaries if row["family"] == family and row["k"] == k]
            controlled = sum(row["correct_positive"] and row["correct_beats_all_controls"] for row in selected)
            k_rows.append({"k": k, "controlled_cells": controlled, "total_cells": len(selected), "all_four_controlled": controlled == len(selected) == 4})
        minimal = next((row["k"] for row in k_rows if row["all_four_controlled"]), None)
        family_rows.append({"family": family, "minimal_all_four_controlled_k": minimal, "nested_results": k_rows})
    family_path = OUT / "analysis/minimal_coordinate_coalition_by_family.jsonl"
    core.write_rows(family_path, family_rows)

    c105_rows = {row["pair_id"]: row for row in core.rows(C105 / "analysis/c104_corrected_intervention_results.jsonl") if row["family"] in protocol["families"]}
    positive_control_max_abs = 0.0
    for row in results:
        old = c105_rows[row["pair_id"]]
        for mode in protocol["modes"]:
            positive_control_max_abs = max(positive_control_max_abs, abs(row["nested"]["2560"][mode]["true_direction_gain"] - old["modes"][mode]["true_direction_gain_corrected"]))
    checks = {
        "rows": len(results) == protocol["pairs"] == 96,
        "summary": len(summaries) == 2 * len(protocol["nested_k"]) * 2 * 2,
        "finite": all(math.isfinite(entry["true_direction_gain"]) for row in results for nested in row["nested"].values() for entry in nested.values()),
        "repeat_hidden": repeat_hidden == 0.0,
        "repeat_logits": repeat_logits == 0.0,
        "whole_state_positive_control": positive_control_max_abs == 0.0,
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "positive_control_max_abs": positive_control_max_abs})
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "nested_minimal_coordinate_coalition_intervention_complete",
        "results_sha256": core.sha(path), "summary_sha256": core.sha(summary_path), "family_sha256": core.sha(family_path),
        "minimal_k": {row["family"]: row["minimal_all_four_controlled_k"] for row in family_rows},
        "whole_state_positive_control_max_abs": positive_control_max_abs,
        "checks": checks,
        "runtime": {"placement": placement, "quantization": quant},
        "interpretation": "a finite minimal K establishes sufficiency for the frozen ranked activation-coordinate coalition, not necessity, sparsity, neuron identity, or weight localization",
        "authorization": "audit_export_and_close_c106",
    }
    core.save(OUT / "analysis/final.json", report)
    print(json.dumps({"report": {key: value for key, value in report.items() if key != "runtime"}, "families": family_rows}, indent=2))


if __name__ == "__main__":
    main()
