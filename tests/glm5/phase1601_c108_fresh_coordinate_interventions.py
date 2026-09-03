#!/usr/bin/env python3
"""Phase1601 / C108: fresh write and delete tests for frozen activation-coordinate supports."""
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
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1592_c104_upstream_role_intervention as parent

PHASE = 1601
CAMPAIGN = "C108"
WIDTH = 224
WRITE_MODES = ("frozen_support", "wrong_family_support", "sign_reversed", "same_truth", "coordinate_permuted", "whole_state")
DELETE_MODES = ("frozen_support", "wrong_family_support", "same_truth")


def margin(logits: torch.Tensor, row: dict[str, Any]) -> float:
    return float(logits[int(row["candidate_ids"][0][0])] - logits[int(row["candidate_ids"][1][0])])


def build_pairs(rows: list[dict], protocol: dict) -> list[dict]:
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_unit[row["unit_id"]].append(row)
    predictions = {row["family"]: row for row in protocol["predictions"]}
    pairs = []
    for unit_id, unit_rows in sorted(by_unit.items()):
        family = unit_rows[0]["family"]
        role = predictions[family]["role"]
        for surface, distractor, code in itertools.product((1, -1), repeat=3):
            def pick(truth: int, dis: int) -> dict:
                return next(row for row in unit_rows if (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) == (truth, surface, dis, code))
            recipient = pick(-1, distractor)
            donor = pick(1, distractor)
            recipient_same_truth = pick(-1, -distractor)
            donor_same_truth = pick(1, -distractor)
            lengths = [len(row["role_positions"][role]) for row in (recipient, donor, recipient_same_truth, donor_same_truth)]
            if len(set(lengths)) != 1:
                raise RuntimeError((unit_id, role, lengths))
            pairs.append({
                "pair_id": f"c108-pair-{len(pairs):04d}", "unit_id": unit_id, "family": family,
                "partition": unit_rows[0]["partition"], "surface_factor": surface,
                "distractor_factor": distractor, "code": code, "role": role, "span_length": lengths[0],
                "recipient": recipient, "donor": donor,
                "recipient_same_truth": recipient_same_truth, "donor_same_truth": donor_same_truth,
            })
    return pairs


def enrich_gain(code: int, base: float, target: float) -> dict:
    raw = target - base
    return {
        "yes_minus_no": target,
        "truth_direction_gain": raw,
        "code_aligned_task_gain": code * raw,
        "truth_target_correct": target > 0.0,
        "task_target_correct": code * target > 0.0,
        "truth_target_flip": base <= 0.0 < target,
        "task_target_flip": code * base <= 0.0 < code * target,
    }


def enrich_loss(code: int, donor: float, target: float) -> dict:
    raw = donor - target
    return {
        "yes_minus_no": target,
        "truth_direction_loss": raw,
        "code_aligned_task_loss": code * raw,
        "donor_truth_correct_after_delete": target > 0.0,
        "donor_task_correct_after_delete": code * target > 0.0,
        "truth_target_lost": donor > 0.0 >= target,
        "task_target_lost": code * donor > 0.0 >= code * target,
    }


def summarize(rows: list[dict], protocol: dict) -> tuple[list[dict], list[dict]]:
    summaries = []
    for family in protocol["families"]:
        for partition in protocol["partitions"]:
            for code in (1, -1):
                selected = [row for row in rows if row["family"] == family and row["partition"] == partition and row["code"] == code]
                write_raw = {mode: float(np.median([row["write"][mode]["truth_direction_gain"] for row in selected])) for mode in WRITE_MODES}
                write_task = {mode: float(np.median([row["write"][mode]["code_aligned_task_gain"] for row in selected])) for mode in WRITE_MODES}
                delete_raw = {mode: float(np.median([row["delete"][mode]["truth_direction_loss"] for row in selected])) for mode in DELETE_MODES}
                delete_task = {mode: float(np.median([row["delete"][mode]["code_aligned_task_loss"] for row in selected])) for mode in DELETE_MODES}
                controls = ("wrong_family_support", "sign_reversed", "same_truth", "coordinate_permuted")
                delete_controls = ("wrong_family_support", "same_truth")
                correct_write = [row["write"]["frozen_support"] for row in selected]
                correct_delete = [row["delete"]["frozen_support"] for row in selected]
                recovery = []
                valid_recovery = []
                for row in selected:
                    denominator = row["donor_yes_minus_no"] - row["recipient_yes_minus_no"]
                    ratio = None if abs(denominator) <= 1e-12 else row["write"]["frozen_support"]["truth_direction_gain"] / denominator
                    if ratio is not None:
                        recovery.append(ratio)
                        if code * row["donor_yes_minus_no"] > 0.0:
                            valid_recovery.append(ratio)
                summaries.append({
                    "family": family, "partition": partition, "code": code,
                    "codebook": selected[0]["codebook"], "role": selected[0]["role"], "state": selected[0]["state"],
                    "k": selected[0]["k"], "pairs": len(selected), "independent_units": len({row["unit_id"] for row in selected}),
                    "median_write_truth_direction_gain": write_raw,
                    "median_write_code_aligned_task_gain": write_task,
                    "truth_direction_write_controlled": write_raw["frozen_support"] > 0.0 and all(write_raw["frozen_support"] > write_raw[mode] for mode in controls),
                    "code_aligned_task_write_controlled": write_task["frozen_support"] > 0.0 and all(write_task["frozen_support"] > write_task[mode] for mode in controls),
                    "median_delete_truth_direction_loss": delete_raw,
                    "median_delete_code_aligned_task_loss": delete_task,
                    "truth_direction_delete_controlled": delete_raw["frozen_support"] > 0.0 and all(delete_raw["frozen_support"] > delete_raw[mode] for mode in delete_controls),
                    "code_aligned_task_delete_controlled": delete_task["frozen_support"] > 0.0 and all(delete_task["frozen_support"] > delete_task[mode] for mode in delete_controls),
                    "patched_truth_target_accuracy": float(np.mean([item["truth_target_correct"] for item in correct_write])),
                    "patched_task_target_accuracy": float(np.mean([item["task_target_correct"] for item in correct_write])),
                    "truth_target_flip_rate": float(np.mean([item["truth_target_flip"] for item in correct_write])),
                    "task_target_flip_rate": float(np.mean([item["task_target_flip"] for item in correct_write])),
                    "truth_loss_flip_rate": float(np.mean([item["truth_target_lost"] for item in correct_delete])),
                    "task_loss_flip_rate": float(np.mean([item["task_target_lost"] for item in correct_delete])),
                    "median_donor_trajectory_recovery_ratio": float(np.median(recovery)) if recovery else None,
                    "task_valid_recovery_pairs": len(valid_recovery),
                    "median_task_recovery_ratio_when_donor_valid": float(np.median(valid_recovery)) if valid_recovery else None,
                })
    family_rows = []
    for family in protocol["families"]:
        selected = [row for row in summaries if row["family"] == family]
        family_rows.append({
            "family": family, "k": selected[0]["k"], "cells": len(selected),
            "truth_direction_write_cells": sum(row["truth_direction_write_controlled"] for row in selected),
            "code_aligned_task_write_cells": sum(row["code_aligned_task_write_controlled"] for row in selected),
            "truth_direction_delete_cells": sum(row["truth_direction_delete_controlled"] for row in selected),
            "code_aligned_task_delete_cells": sum(row["code_aligned_task_delete_controlled"] for row in selected),
            "mean_patched_truth_target_accuracy": float(np.mean([row["patched_truth_target_accuracy"] for row in selected])),
            "mean_patched_task_target_accuracy": float(np.mean([row["patched_task_target_accuracy"] for row in selected])),
            "mean_truth_target_flip_rate": float(np.mean([row["truth_target_flip_rate"] for row in selected])),
            "mean_task_target_flip_rate": float(np.mean([row["task_target_flip_rate"] for row in selected])),
        })
    return summaries, family_rows


@torch.inference_mode()
def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/independent_pre_model_audit.json")
    if protocol["authorization"] != "execute_phase1601_c108_fresh_coordinate_interventions" or not audit["all_checks_passed"]:
        raise RuntimeError("C108 execution not authorized")
    pairs = build_pairs(core.rows(OUT / "compiled/qwen3.jsonl"), protocol)
    if len(pairs) != 192:
        raise RuntimeError(len(pairs))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["family"], pair["partition"], pair["code"], pair["span_length"])].append(pair)
    model = None
    results = []
    first_repeat = None
    runtime = {}
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        runtime = {"placement": placement, "quantization": quant}
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for (family, partition, code, span_length), group in sorted(grouped.items()):
            prediction = next(row for row in protocol["predictions"] if row["family"] == family)
            role, state = prediction["role"], int(prediction["state"])
            k = int(protocol["frozen_k"][family])
            support = torch.tensor(protocol["rankings"][family][:k], dtype=torch.long, device=device)
            other = next(value for value in protocol["families"] if value != family)
            wrong = torch.tensor(protocol["rankings"][other][:k], dtype=torch.long, device=device)
            permutation = torch.tensor(protocol["coordinate_permutations"][family], dtype=torch.long, device=device)
            for start in range(0, len(group), 8):
                batch = group[start:start + 8]
                recipients = [row["recipient"] for row in batch]
                donors = [row["donor"] for row in batch]
                recipient_same = [row["recipient_same_truth"] for row in batch]
                donor_same = [row["donor_same_truth"] for row in batch]
                recipient_logits, recipient_state = parent.forward_with_role(model, recipients, role, state, pad, device, WIDTH)
                donor_logits, donor_state = parent.forward_with_role(model, donors, role, state, pad, device, WIDTH)
                _, recipient_same_state = parent.forward_with_role(model, recipient_same, role, state, pad, device, WIDTH)
                _, donor_same_state = parent.forward_with_role(model, donor_same, role, state, pad, device, WIDTH)
                if first_repeat is None:
                    first_repeat = (recipients, role, state, recipient_logits.detach().clone(), recipient_state.detach().clone())
                recipient_margins = [margin(recipient_logits[i], recipients[i]) for i in range(len(batch))]
                donor_margins = [margin(donor_logits[i], donors[i]) for i in range(len(batch))]

                write_values = {}
                value = recipient_state.clone(); value[..., support] = donor_state[..., support]; write_values["frozen_support"] = value
                value = recipient_state.clone(); value[..., wrong] = donor_state[..., wrong]; write_values["wrong_family_support"] = value
                value = recipient_state.clone(); value[..., support] = 2.0 * recipient_state[..., support] - donor_state[..., support]; write_values["sign_reversed"] = value
                value = recipient_state.clone(); value[..., support] = recipient_same_state[..., support]; write_values["same_truth"] = value
                value = recipient_state.clone(); value[..., support] = donor_state[..., permutation[support]]; write_values["coordinate_permuted"] = value
                write_values["whole_state"] = donor_state.clone()
                write_logits = {mode: parent.forward_patched(model, recipients, role, state, values, pad, device, WIDTH) for mode, values in write_values.items()}

                delete_values = {}
                value = donor_state.clone(); value[..., support] = recipient_state[..., support]; delete_values["frozen_support"] = value
                value = donor_state.clone(); value[..., wrong] = recipient_state[..., wrong]; delete_values["wrong_family_support"] = value
                value = donor_state.clone(); value[..., support] = donor_same_state[..., support]; delete_values["same_truth"] = value
                delete_logits = {mode: parent.forward_patched(model, donors, role, state, values, pad, device, WIDTH) for mode, values in delete_values.items()}

                for i, pair in enumerate(batch):
                    row = {
                        "pair_id": pair["pair_id"], "unit_id": pair["unit_id"], "family": family,
                        "partition": partition, "code": code, "codebook": pair["recipient"]["codebook"],
                        "role": role, "state": state, "k": k, "span_length": span_length,
                        "recipient_yes_minus_no": recipient_margins[i], "donor_yes_minus_no": donor_margins[i],
                        "recipient_task_margin": -code * recipient_margins[i], "donor_task_margin": code * donor_margins[i],
                        "write": {}, "delete": {},
                    }
                    for mode, logits in write_logits.items():
                        row["write"][mode] = enrich_gain(code, recipient_margins[i], margin(logits[i], recipients[i]))
                    for mode, logits in delete_logits.items():
                        row["delete"][mode] = enrich_loss(code, donor_margins[i], margin(logits[i], donors[i]))
                    results.append(row)
                print(f"[phase1601] {family}/{partition}/code={code}/span={span_length} {start + len(batch)}/{len(group)}", flush=True)
        if first_repeat is None:
            raise RuntimeError("repeat batch missing")
        repeat_rows, repeat_role, repeat_state_index, old_logits, old_state = first_repeat
        logits, state_values = parent.forward_with_role(model, repeat_rows, repeat_role, repeat_state_index, pad, device, WIDTH)
        repeat_hidden = float(torch.max(torch.abs(state_values - old_state)).item())
        repeat_logits = float(torch.max(torch.abs(logits - old_logits)).item())
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()

    path = OUT / "analysis/fresh_coordinate_intervention_results.jsonl"
    core.write_rows(path, results)
    summaries, family_rows = summarize(results, protocol)
    summary_path = OUT / "analysis/fresh_coordinate_intervention_summary.jsonl"
    family_path = OUT / "analysis/fresh_coordinate_family_rollup.jsonl"
    core.write_rows(summary_path, summaries)
    core.write_rows(family_path, family_rows)
    behavior = {
        "recipient_truth_accuracy": float(np.mean([row["recipient_yes_minus_no"] < 0.0 for row in results])),
        "donor_truth_accuracy": float(np.mean([row["donor_yes_minus_no"] > 0.0 for row in results])),
        "recipient_task_accuracy": float(np.mean([row["recipient_task_margin"] > 0.0 for row in results])),
        "donor_task_accuracy": float(np.mean([row["donor_task_margin"] > 0.0 for row in results])),
        "all_case_task_accuracy": float(np.mean([row[key] > 0.0 for row in results for key in ("recipient_task_margin", "donor_task_margin")])),
        "standard_task_accuracy": float(np.mean([row[key] > 0.0 for row in results if row["code"] == 1 for key in ("recipient_task_margin", "donor_task_margin")])),
        "reversed_task_accuracy": float(np.mean([row[key] > 0.0 for row in results if row["code"] == -1 for key in ("recipient_task_margin", "donor_task_margin")])),
    }
    checks = {
        "rows": len(results) == 192,
        "summary": len(summaries) == 8 and len(family_rows) == 2,
        "independent_units": all(row["independent_units"] == 6 for row in summaries),
        "finite": all(math.isfinite(entry[key]) for row in results for branch in ("write", "delete") for entry in row[branch].values() for key in entry if isinstance(entry[key], (int, float)) and not isinstance(entry[key], bool)),
        "repeat_hidden": repeat_hidden == 0.0,
        "repeat_logits": repeat_logits == 0.0,
        "candidate_order": all(row["candidate_ids"] == [[9834], [902]] for row in core.rows(OUT / "compiled/qwen3.jsonl")),
        "bf16_nonquantized": runtime["quantization"]["has_bf16_parameters"] and not runtime["quantization"]["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_coordinate_write_delete_intervention_complete",
        "results_sha256": core.sha(path), "summary_sha256": core.sha(summary_path), "family_sha256": core.sha(family_path),
        "behavior": behavior, "family_rollup": family_rows, "checks": checks, "runtime": runtime,
        "claim_boundary": "fresh frozen-support activation intervention; reports direction, task alignment, flips, and deletion separately; no minimality or weight claim",
        "authorization": "independent_audit_synthesize_and_close_c108",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({key: value for key, value in final.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
