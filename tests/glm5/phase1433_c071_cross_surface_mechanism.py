#!/usr/bin/env python3
"""Phase1433: one-shot C071 cross-surface semantic-role quartet transport."""
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

PHASE, CAMPAIGN = 1433, "C071"
CONTRACT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
BEHAVIOR = TESTS / "result/phase1431_c071_cross_surface_behavior"
CAMERA = TESTS / "result/phase1432_c071_role_map_camera"
OUT = TESTS / "result/phase1433_c071_cross_surface_mechanism"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")
SPLITS = ("confirmation", "lockbox")
ARMS = (
    "self",
    "same_surface_quartet",
    "cross_surface_role_mapped",
    "cross_surface_role_permuted",
    "wrong_cross_surface_role_mapped",
)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def margin(logits: torch.Tensor, row: dict) -> float:
    yes = row["candidate_ids"][0][0]
    no = row["candidate_ids"][1][0]
    return float(logits[yes].float() - logits[no].float())


def transfer_surfaces(transfer: str) -> tuple[str, str]:
    if transfer == "belongs_include_to_lists_member":
        return "belongs_include", "lists_member"
    if transfer == "lists_member_to_belongs_include":
        return "lists_member", "belongs_include"
    raise ValueError(transfer)


def direction_rows(case: dict, compiled: dict[str, dict], source_surface: str,
                   target_surface: str, direction: str) -> tuple[dict, dict, dict, dict, float]:
    if direction == "true_to_false":
        return (
            compiled[case[f"{target_surface}_true_recipient"]],
            compiled[case[f"{target_surface}_false_donor"]],
            compiled[case[f"{source_surface}_false_donor"]],
            compiled[case[f"{source_surface}_true_donor"]],
            -1.0,
        )
    return (
        compiled[case[f"{target_surface}_false_recipient"]],
        compiled[case[f"{target_surface}_true_donor"]],
        compiled[case[f"{source_surface}_true_donor"]],
        compiled[case[f"{source_surface}_false_donor"]],
        1.0,
    )


@torch.inference_mode()
def run_case(model, pad: int, device: torch.device, supports: bool, case: dict,
             transfer: str, direction: str, compiled: dict[str, dict],
             state_index: int, permutation: dict[str, str]) -> list[dict]:
    source_surface, target_surface = transfer_surfaces(transfer)
    recipient, same_donor, cross_donor, wrong_donor, orientation = direction_rows(
        case, compiled, source_surface, target_surface, direction
    )
    rows = [recipient, same_donor, cross_donor, wrong_donor] + [recipient for _ in ARMS]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
    measured: dict[str, float | int] = {}

    def role_points(row: dict, batch_index: int) -> dict[str, int]:
        return {role: batcher.points(row, offsets[batch_index], role)[0] for role in ROLES}

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        recipient_points = role_points(recipient, 0)
        same_points = role_points(same_donor, 1)
        cross_points = role_points(cross_donor, 2)
        wrong_points = role_points(wrong_donor, 3)
        arm_points = {arm: role_points(recipient, 4 + index) for index, arm in enumerate(ARMS)}
        for role in ROLES:
            value[4, arm_points["self"][role]] = original[0, recipient_points[role]]
            value[5, arm_points["same_surface_quartet"][role]] = original[1, same_points[role]]
            value[6, arm_points["cross_surface_role_mapped"][role]] = original[2, cross_points[role]]
            value[7, arm_points["cross_surface_role_permuted"][role]] = original[2, cross_points[permutation[role]]]
            value[8, arm_points["wrong_cross_surface_role_mapped"][role]] = original[3, wrong_points[role]]
        target_roles = set(recipient_points.values())
        target_active = list(range(offsets[0], ids.shape[1]))
        complement = [point for point in target_active if point not in target_roles]
        measured["quartet_size"] = len(target_roles)
        measured["target_complement_count"] = len(complement)
        measured["self_role_max_abs_diff"] = max(
            float((value[4, arm_points["self"][role]].float() - original[0, recipient_points[role]].float()).abs().max())
            for role in ROLES
        )
        measured["mapped_role_max_abs_diff"] = max(
            float((value[6, arm_points["cross_surface_role_mapped"][role]].float() - original[2, cross_points[role]].float()).abs().max())
            for role in ROLES
        )
        measured["mapped_complement_max_abs_diff"] = float(
            (value[6, complement].float() - original[0, complement].float()).abs().max()
        )
        measured["permuted_complement_max_abs_diff"] = float(
            (value[7, complement].float() - original[0, complement].float()).abs().max()
        )
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

    recipient_margin = margin(output.logits[0, -1], recipient)
    same_margin = margin(output.logits[1, -1], same_donor)
    cross_margin = margin(output.logits[2, -1], cross_donor)
    wrong_margin = margin(output.logits[3, -1], wrong_donor)
    records = []
    for row_index, arm in enumerate(ARMS, start=4):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"],
            "partition": case["partition"],
            "family": case["family"],
            "donor_family": case["donor_family"],
            "surface_transfer": transfer,
            "source_surface": source_surface,
            "target_surface": target_surface,
            "direction": direction,
            "state_index": state_index,
            "arm": arm,
            "source_length": len(cross_donor["prompt_ids"]),
            "target_length": len(recipient["prompt_ids"]),
            **measured,
            "recipient_margin": recipient_margin,
            "same_donor_margin": same_margin,
            "cross_donor_margin": cross_margin,
            "wrong_donor_margin": wrong_margin,
            "swap_margin": changed,
            "oriented_gain": orientation * (changed - recipient_margin),
            "desired_sign": changed < 0.0 if direction == "true_to_false" else changed > 0.0,
            "wrong_expected_sign": changed > 0.0 if direction == "true_to_false" else changed < 0.0,
            "recipient_output_max_abs_diff": float(
                (output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()
            ),
        })
    del output, ids, mask, position_ids
    return records


def group_metrics(rows: list[dict]) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    if not rows or not all(arms.values()):
        raise RuntimeError("missing arm")
    by_arm = {arm: {row["set_id"]: row for row in values} for arm, values in arms.items()}
    set_ids = sorted(by_arm["self"])
    if not all(set(values) == set(set_ids) for values in by_arm.values()):
        raise RuntimeError("unbalanced arms")
    mapped = arms["cross_surface_role_mapped"]
    permuted = arms["cross_surface_role_permuted"]
    desired = {arm: sum(row["desired_sign"] for row in values) / len(values) for arm, values in arms.items()}
    gain = {arm: median([row["oriented_gain"] for row in values]) for arm, values in arms.items()}
    paired_gain_gaps = [
        by_arm["cross_surface_role_mapped"][set_id]["oriented_gain"]
        - by_arm["cross_surface_role_permuted"][set_id]["oriented_gain"]
        for set_id in set_ids
    ]
    return {
        "count": len(set_ids),
        "self_output_max_abs_diff": max(row["recipient_output_max_abs_diff"] for row in arms["self"]),
        "desired_sign_fraction": desired,
        "wrong_expected_sign_fraction": sum(row["wrong_expected_sign"] for row in arms["wrong_cross_surface_role_mapped"]) / len(arms["wrong_cross_surface_role_mapped"]),
        "oriented_gain_median": gain,
        "mapped_vs_permuted_sign_gap": desired["cross_surface_role_mapped"] - desired["cross_surface_role_permuted"],
        "mapped_vs_permuted_gain_gap_median": median(paired_gain_gaps),
        "mapped_margin_median": median([row["swap_margin"] for row in mapped]),
        "permuted_margin_median": median([row["swap_margin"] for row in permuted]),
    }


def classify_cell(split_metrics: dict[str, dict], family_metrics: dict[str, dict], gate: dict) -> dict:
    self_pass = all(split_metrics[split]["self_output_max_abs_diff"] <= gate["self_max_abs_diff"] for split in SPLITS)
    same_pass = all(split_metrics[split]["desired_sign_fraction"]["same_surface_quartet"] >= gate["same_surface_desired_sign_fraction_min"] for split in SPLITS)
    mapped_pass = all(
        split_metrics[split]["desired_sign_fraction"]["cross_surface_role_mapped"] >= gate["cross_surface_desired_sign_fraction_min"]
        and split_metrics[split]["oriented_gain_median"]["cross_surface_role_mapped"] >= gate["mapped_oriented_gain_median_min"]
        for split in SPLITS
    )
    wrong_pass = all(split_metrics[split]["wrong_expected_sign_fraction"] >= gate["wrong_expected_sign_fraction_min"] for split in SPLITS)
    selective_pass = all(
        split_metrics[split]["mapped_vs_permuted_sign_gap"] >= gate["mapped_vs_permuted_sign_gap_min"]
        and split_metrics[split]["mapped_vs_permuted_gain_gap_median"] >= gate["mapped_vs_permuted_gain_gap_median_min"]
        for split in SPLITS
    )
    same_families = []
    mapped_families = []
    wrong_families = []
    for family, metrics in family_metrics.items():
        if all(metrics[split]["desired_sign_fraction"]["same_surface_quartet"] >= gate["family_desired_sign_fraction_min"] for split in SPLITS):
            same_families.append(family)
        if all(metrics[split]["desired_sign_fraction"]["cross_surface_role_mapped"] >= gate["family_desired_sign_fraction_min"] for split in SPLITS):
            mapped_families.append(family)
        if all(metrics[split]["wrong_expected_sign_fraction"] >= gate["family_desired_sign_fraction_min"] for split in SPLITS):
            wrong_families.append(family)
    same_breadth = len(same_families) >= gate["minimum_family_breadth"]
    mapped_breadth = len(mapped_families) >= gate["minimum_family_breadth"]
    wrong_breadth = len(wrong_families) >= gate["minimum_family_breadth"]
    executor_ok = self_pass and same_pass and wrong_pass and same_breadth and wrong_breadth
    cross_ok = mapped_pass and mapped_breadth
    if not executor_ok:
        classification = "same_surface_executor_failed"
    elif not cross_ok:
        classification = "same_surface_only"
    elif not selective_pass:
        classification = "cross_surface_nonspecific"
    else:
        classification = "role_isomorphic_selective"
    return {
        "classification": classification,
        "self_pass": self_pass,
        "same_surface_pass": same_pass,
        "cross_surface_mapped_pass": mapped_pass,
        "wrong_donor_pass": wrong_pass,
        "selective_pass": selective_pass,
        "same_surface_qualified_families": same_families,
        "cross_surface_qualified_families": mapped_families,
        "wrong_donor_qualified_families": wrong_families,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1433 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    if camera_final["authorization"] != "run_phase1433_c071_cross_surface_mechanism" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera gate missing")
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
            for transfer in gate["surface_transfers"]:
                for direction in gate["directions"]:
                    records.extend(run_case(
                        model, pad, device, supports, case, transfer, direction, compiled,
                        gate["state_index"], protocol["role_map"]["permuted_source"],
                    ))
        core.write_rows(OUT / "raw/cross_surface_mechanism.jsonl", records)

        aggregate = {
            transfer: {
                direction: {
                    split: group_metrics([
                        row for row in records
                        if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == split
                    ])
                    for split in SPLITS
                }
                for direction in DIRECTIONS
            }
            for transfer in gate["surface_transfers"]
        }
        family_metrics = {
            transfer: {
                direction: {
                    family: {
                        split: group_metrics([
                            row for row in records
                            if row["surface_transfer"] == transfer and row["direction"] == direction
                            and row["family"] == family and row["partition"] == split
                        ])
                        for split in SPLITS
                    }
                    for family in behavior_final["qualified_families"]
                }
                for direction in DIRECTIONS
            }
            for transfer in gate["surface_transfers"]
        }
        cell_results = {
            transfer: {
                direction: classify_cell(
                    aggregate[transfer][direction], family_metrics[transfer][direction], gate
                )
                for direction in DIRECTIONS
            }
            for transfer in gate["surface_transfers"]
        }
        classes = {
            cell_results[transfer][direction]["classification"]
            for transfer in gate["surface_transfers"] for direction in DIRECTIONS
        }
        overall = next(iter(classes)) if len(classes) == 1 else "transfer_or_direction_asymmetric"
        checks = {
            "holdout_sets": len(holdouts) == 48,
            "split_balance": all(sum(row["partition"] == split for row in holdouts) == 24 for split in SPLITS),
            "record_count": len(records) == 48 * 2 * 2 * len(ARMS),
            "arm_balance": all(sum(row["arm"] == arm for row in records) == 48 * 2 * 2 for arm in ARMS),
            "transfers": {row["surface_transfer"] for row in records} == set(gate["surface_transfers"]),
            "directions": {row["direction"] for row in records} == set(DIRECTIONS),
            "holdout_only": {row["partition"] for row in records} == set(SPLITS),
            "state16_only": {row["state_index"] for row in records} == {16},
            "different_shapes": all(row["source_length"] != row["target_length"] for row in records),
            "quartet": all(row["quartet_size"] == 4 for row in records),
            "mapped_write": max(row["mapped_role_max_abs_diff"] for row in records) <= 1e-4,
            "complement": max(max(row["mapped_complement_max_abs_diff"], row["permuted_complement_max_abs_diff"]) for row in records) <= 1e-4,
            "finite": all(math.isfinite(row[key]) for row in records for key in (
                "recipient_margin", "same_donor_margin", "cross_donor_margin", "wrong_donor_margin",
                "swap_margin", "oriented_gain", "recipient_output_max_abs_diff",
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
            "cell_results": cell_results,
            "overall_classification": overall,
            "checks": checks,
            "all_execution_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"],
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/mechanism_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_execution_checks_passed": summary["all_execution_checks_passed"],
            "overall_classification": overall,
            "cell_results": cell_results,
            "authorization": "run_phase1434_c071_campaign_closure",
        })
        print(json.dumps({key: value for key, value in summary.items() if key != "family_metrics"}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
