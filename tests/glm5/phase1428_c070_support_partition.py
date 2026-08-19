#!/usr/bin/env python3
"""Phase1428: one-shot C070 quartet-versus-complement support partition."""
from __future__ import annotations

import inspect
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1392_c062_full_field_camera as batcher
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1428, "C070"
CONTRACT = TESTS / "result/phase1425_c070_quartet_complement_contract"
BEHAVIOR = TESTS / "result/phase1426_c070_roster_behavior"
CAMERA = TESTS / "result/phase1427_c070_partition_camera"
OUT = TESTS / "result/phase1428_c070_support_partition"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")
ARMS = ("self", "quartet_only", "complement_only", "full_state", "wrong_full_state")
SPLITS = ("confirmation", "lockbox")


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[row["candidate_ids"][0][0]].float() - logits[row["candidate_ids"][1][0]].float())


def direction_rows(case: dict, compiled: dict[str, dict], direction: str) -> tuple[dict, dict, dict, float]:
    if direction == "true_to_false":
        return compiled[case["true_recipient"]], compiled[case["false_donor"]], compiled[case["true_donor"]], -1.0
    return compiled[case["false_recipient"]], compiled[case["true_donor"]], compiled[case["false_donor"]], 1.0


@torch.inference_mode()
def run_direction(model, pad: int, device: torch.device, supports: bool, case: dict,
                  direction: str, compiled: dict[str, dict], state_index: int) -> list[dict]:
    recipient, desired, wrong, orientation = direction_rows(case, compiled, direction)
    rows = [recipient, desired, wrong] + [recipient for _ in ARMS]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
    if len({int(mask[index].sum()) for index in range(len(rows))}) != 1:
        raise RuntimeError("non-isomorphic prompt lengths")
    quartet = sorted({
        point
        for role in ROLES
        for point in batcher.points(recipient, offsets[0], role)
    })
    if len(quartet) != 4:
        raise RuntimeError(f"quartet must contain four distinct points: {quartet}")
    complement = [point for point in range(ids.shape[1]) if point not in set(quartet)]

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        first = 3
        value[first] = original[0]
        value[first + 1, quartet] = original[1, quartet]
        value[first + 2, complement] = original[1, complement]
        value[first + 3] = original[1]
        value[first + 4] = original[2]
        return (value,) + args[1:]

    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": position_ids,
        "use_cache": False,
        "return_dict": True,
    }
    if supports:
        kwargs["logits_to_keep"] = 1
    try:
        output = model(**kwargs)
    finally:
        handle.remove()
    baseline = margin(output.logits[0, -1], recipient)
    desired_margin = margin(output.logits[1, -1], desired)
    wrong_margin = margin(output.logits[2, -1], wrong)
    records = []
    for row_index, arm in enumerate(ARMS, start=3):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"],
            "partition": case["partition"],
            "family": case["family"],
            "donor_family": case["donor_family"],
            "direction": direction,
            "state_index": state_index,
            "arm": arm,
            "quartet_points": quartet,
            "complement_count": len(complement),
            "recipient_margin": baseline,
            "desired_donor_margin": desired_margin,
            "wrong_donor_margin": wrong_margin,
            "swap_margin": changed,
            "oriented_gain": orientation * (changed - baseline),
            "desired_sign": changed < 0.0 if direction == "true_to_false" else changed > 0.0,
            "wrong_expected_sign": changed > 0.0 if direction == "true_to_false" else changed < 0.0,
            "desired_donor_relative_deviation": abs(changed - desired_margin) / (abs(desired_margin) + 1e-12),
            "wrong_donor_relative_deviation": abs(changed - wrong_margin) / (abs(wrong_margin) + 1e-12),
            "recipient_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()),
            "desired_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[1, -1].float()).abs().max()),
            "wrong_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[2, -1].float()).abs().max()),
        })
    del output, ids, mask, position_ids
    return records


def group_metrics(rows: list[dict], gate: dict, partition_threshold: float, full_sign_threshold: float) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    if not rows or not all(arms.values()):
        raise RuntimeError("missing arm")
    by_arm = {arm: {row["set_id"]: row for row in values} for arm, values in arms.items()}
    set_ids = sorted(by_arm["self"])
    if not all(set(values) == set(set_ids) for values in by_arm.values()):
        raise RuntimeError("unbalanced arms")
    synergy = []
    residual = []
    for set_id in set_ids:
        q = by_arm["quartet_only"][set_id]["oriented_gain"]
        c = by_arm["complement_only"][set_id]["oriented_gain"]
        f = by_arm["full_state"][set_id]["oriented_gain"]
        synergy.append(f - max(q, c))
        residual.append(f - q - c)
    desired_fraction = {
        arm: sum(row["desired_sign"] for row in values) / len(values)
        for arm, values in arms.items()
    }
    metrics = {
        "count": len(set_ids),
        "self_output_max_abs_diff": max(row["recipient_output_max_abs_diff"] for row in arms["self"]),
        "desired_sign_fraction": desired_fraction,
        "wrong_expected_sign_fraction": sum(row["wrong_expected_sign"] for row in arms["wrong_full_state"]) / len(arms["wrong_full_state"]),
        "full_donor_relative_deviation_median": med([row["desired_donor_relative_deviation"] for row in arms["full_state"]]),
        "wrong_donor_relative_deviation_median": med([row["wrong_donor_relative_deviation"] for row in arms["wrong_full_state"]]),
        "oriented_gain_median": {arm: med([row["oriented_gain"] for row in values]) for arm, values in arms.items()},
        "synergy_advantage_median": med(synergy),
        "synergy_win_fraction": sum(value >= gate["synergy_advantage_median_min"] for value in synergy) / len(synergy),
        "nonlinear_residual_median": med(residual),
    }
    full_checks = {
        "self": metrics["self_output_max_abs_diff"] <= gate["self_max_abs_diff"],
        "desired_sign": desired_fraction["full_state"] >= full_sign_threshold,
        "wrong_sign": metrics["wrong_expected_sign_fraction"] >= full_sign_threshold,
        "full_donor": metrics["full_donor_relative_deviation_median"] <= gate["full_donor_relative_deviation_median_max"],
        "wrong_donor": metrics["wrong_donor_relative_deviation_median"] <= gate["wrong_donor_relative_deviation_median_max"],
        "full_gain": metrics["oriented_gain_median"]["full_state"] >= gate["full_oriented_gain_median_min"],
    }
    support = {
        "quartet": desired_fraction["quartet_only"] >= partition_threshold,
        "complement": desired_fraction["complement_only"] >= partition_threshold,
        "synergy": metrics["synergy_advantage_median"] >= gate["synergy_advantage_median_min"] and metrics["synergy_win_fraction"] >= gate["synergy_win_fraction_min"],
    }
    return {**metrics, "full_checks": full_checks, "full_qualified": all(full_checks.values()), "support_checks": support}


def classify_direction(aggregate: dict, family_metrics: dict, gate: dict) -> dict:
    full_families = [family for family, values in family_metrics.items() if all(values[split]["full_qualified"] for split in SPLITS)]
    quartet_families = [family for family, values in family_metrics.items() if all(values[split]["support_checks"]["quartet"] for split in SPLITS)]
    complement_families = [family for family, values in family_metrics.items() if all(values[split]["support_checks"]["complement"] for split in SPLITS)]
    full_valid = all(aggregate[split]["full_qualified"] for split in SPLITS) and len(full_families) >= gate["minimum_family_breadth"]
    quartet_supported = all(aggregate[split]["support_checks"]["quartet"] for split in SPLITS) and len(quartet_families) >= gate["minimum_family_breadth"]
    complement_supported = all(aggregate[split]["support_checks"]["complement"] for split in SPLITS) and len(complement_families) >= gate["minimum_family_breadth"]
    synergy_supported = all(aggregate[split]["support_checks"]["synergy"] for split in SPLITS)
    if not full_valid:
        classification = "full_transport_failed"
    elif quartet_supported and complement_supported:
        classification = "redundant_dual_support"
    elif quartet_supported:
        classification = "quartet_dominant"
    elif complement_supported:
        classification = "complement_dominant"
    else:
        classification = "joint_only_or_unresolved"
    return {
        "classification": classification,
        "full_valid": full_valid,
        "quartet_supported": quartet_supported,
        "complement_supported": complement_supported,
        "synergy_supported": synergy_supported,
        "full_qualified_families": full_families,
        "quartet_qualified_families": quartet_families,
        "complement_qualified_families": complement_families,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1428 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if camera_final["authorization"] != "run_phase1428_c070_support_partition" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera did not authorize mechanism")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    holdouts = [row for row in selected if row["partition"] in SPLITS]
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    gate = protocol["mechanism"]
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in holdouts:
            for direction in DIRECTIONS:
                records.extend(run_direction(model, pad, device, supports, case, direction, compiled, gate["state_index"]))
        core.write_rows(OUT / "raw/support_partition.jsonl", records)

        aggregate = {
            direction: {
                split: group_metrics(
                    [row for row in records if row["partition"] == split and row["direction"] == direction],
                    gate,
                    gate["aggregate_partition_desired_sign_fraction_min"],
                    gate["full_desired_sign_fraction_min"],
                )
                for split in SPLITS
            }
            for direction in DIRECTIONS
        }
        family_metrics = {
            direction: {
                family: {
                    split: group_metrics(
                        [row for row in records if row["partition"] == split and row["direction"] == direction and row["family"] == family],
                        gate,
                        gate["family_partition_desired_sign_fraction_min"],
                        gate["family_partition_desired_sign_fraction_min"],
                    )
                    for split in SPLITS
                }
                for family in behavior_final["qualified_families"]
            }
            for direction in DIRECTIONS
        }
        direction_results = {
            direction: classify_direction(aggregate[direction], family_metrics[direction], gate)
            for direction in DIRECTIONS
        }
        classes = {result["classification"] for result in direction_results.values()}
        overall_classification = next(iter(classes)) if len(classes) == 1 else "direction_asymmetric"
        checks = {
            "holdout_sets": len(holdouts) == 48,
            "split_balance": all(sum(row["partition"] == split for row in holdouts) == 24 for split in SPLITS),
            "record_count": len(records) == 48 * len(DIRECTIONS) * len(ARMS),
            "arm_balance": all(sum(row["arm"] == arm for row in records) == 96 for arm in ARMS),
            "directions": {row["direction"] for row in records} == set(DIRECTIONS),
            "state16_only": {row["state_index"] for row in records} == {16},
            "holdout_only": {row["partition"] for row in records} == set(SPLITS),
            "partition_complete": all(len(row["quartet_points"]) == 4 and row["complement_count"] == 66 for row in records),
            "finite": all(math.isfinite(row[key]) for row in records for key in (
                "recipient_margin", "desired_donor_margin", "wrong_donor_margin", "swap_margin", "oriented_gain",
                "desired_donor_relative_deviation", "wrong_donor_relative_deviation", "recipient_output_max_abs_diff",
                "desired_output_max_abs_diff", "wrong_output_max_abs_diff",
            )),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_set_count": len(holdouts),
            "record_count": len(records),
            "aggregate_metrics": aggregate,
            "family_metrics": family_metrics,
            "direction_results": direction_results,
            "overall_classification": overall_classification,
            "checks": checks,
            "all_execution_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"],
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        core.save(OUT / "analysis/support_partition_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_execution_checks_passed": summary["all_execution_checks_passed"],
            "overall_classification": overall_classification,
            "direction_results": direction_results,
            "authorization": "run_phase1429_c070_campaign_closure",
        })
        print(json.dumps({key: value for key, value in summary.items() if key != "family_metrics"}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
