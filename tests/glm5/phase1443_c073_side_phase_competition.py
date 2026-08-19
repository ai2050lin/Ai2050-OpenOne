#!/usr/bin/env python3
"""Phase1443: one-shot C073 semantic-side versus physical-phase competition."""
from __future__ import annotations

import hashlib
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

PHASE, CAMPAIGN = 1443, "C073"
CONTRACT = TESTS / "result/phase1440_c073_side_phase_contract"
BEHAVIOR = TESTS / "result/phase1441_c073_behavior"
CAMERA = TESTS / "result/phase1442_c073_matched_camera"
OUT = TESTS / "result/phase1443_c073_side_phase_competition"
ROLES = ("record_target", "record_family", "query_target", "query_family")
SPLITS = ("confirmation", "lockbox")
DIRECTIONS = ("true_to_false", "false_to_true")


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[row["candidate_ids"][0][0]].float() - logits[row["candidate_ids"][1][0]].float())


def route_surfaces(route: str, protocol: dict) -> tuple[str, str, str]:
    spec = protocol["routes"][route]
    return spec["source"], spec["target"], spec["order"]


def direction_rows(case: dict, compiled: dict[str, dict], source: str, target: str, direction: str) -> tuple[dict, dict, dict, float]:
    if direction == "true_to_false":
        return (
            compiled[case[f"{target}_true_recipient"]],
            compiled[case[f"{source}_false_donor"]],
            compiled[case[f"{source}_true_donor"]],
            -1.0,
        )
    return (
        compiled[case[f"{target}_false_recipient"]],
        compiled[case[f"{source}_true_donor"]],
        compiled[case[f"{source}_false_donor"]],
        1.0,
    )


@torch.inference_mode()
def run_context(model, pad: int, device: torch.device, supports: bool, case: dict, route: str,
                direction: str, compiled: dict[str, dict], permutations: dict[str, dict], protocol: dict) -> list[dict]:
    source, target, order = route_surfaces(route, protocol)
    recipient, correct_donor, wrong_donor, orientation = direction_rows(case, compiled, source, target, direction)
    arms = protocol["mechanism"]["arms"]
    rows = [recipient, correct_donor, wrong_donor] + [recipient for _ in arms]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
    write_errors: dict[str, dict] = {}

    def points(row: dict, batch_index: int) -> dict[str, int]:
        return {role: batcher.points(row, offsets[batch_index], role)[0] for role in ROLES}

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        recipient_points = points(recipient, 0)
        correct_points = points(correct_donor, 1)
        wrong_points = points(wrong_donor, 2)
        for row_index, arm in enumerate(arms, start=3):
            arm_points = points(recipient, row_index)
            if arm == "self":
                donor_index, donor_points, mapping = 0, recipient_points, {role: role for role in ROLES}
            elif arm == "correct_identity":
                donor_index, donor_points, mapping = 1, correct_points, {role: role for role in ROLES}
            elif arm == "wrong_identity":
                donor_index, donor_points, mapping = 2, wrong_points, {role: role for role in ROLES}
            else:
                donor_index, donor_points, mapping = 1, correct_points, permutations[arm]["mapping"]
            for role in ROLES:
                value[row_index, arm_points[role]] = original[donor_index, donor_points[mapping[role]]]
            role_error = max(float((value[row_index, arm_points[role]].float() - original[donor_index, donor_points[mapping[role]]].float()).abs().max()) for role in ROLES)
            active = list(range(offsets[row_index], ids.shape[1]))
            complement = [point for point in active if point not in set(arm_points.values())]
            complement_error = float((value[row_index, complement].float() - original[0, complement].float()).abs().max())
            write_errors[arm] = {"write_max_abs_diff": role_error, "complement_max_abs_diff": complement_error}
        return (value,) + args[1:]

    handle = model.model.layers[protocol["state_index"]].register_forward_pre_hook(hook)
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    try:
        output = model(**kwargs)
    finally:
        handle.remove()
    recipient_margin = margin(output.logits[0, -1], recipient)
    correct_donor_margin = margin(output.logits[1, -1], correct_donor)
    wrong_donor_margin = margin(output.logits[2, -1], wrong_donor)
    records = []
    for row_index, arm in enumerate(arms, start=3):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"],
            "partition": case["partition"],
            "family": case["family"],
            "donor_family": case["donor_family"],
            "route": route,
            "route_order": order,
            "source_surface": source,
            "target_surface": target,
            "direction": direction,
            "state_index": protocol["state_index"],
            "source_length": len(correct_donor["prompt_ids"]),
            "target_length": len(recipient["prompt_ids"]),
            "arm": arm,
            "recipient_margin": recipient_margin,
            "correct_donor_margin": correct_donor_margin,
            "wrong_donor_margin": wrong_donor_margin,
            "swap_margin": changed,
            "oriented_gain": orientation * (changed - recipient_margin),
            "desired_sign": changed < 0 if direction == "true_to_false" else changed > 0,
            "wrong_expected_sign": changed > 0 if direction == "true_to_false" else changed < 0,
            "recipient_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()),
            **write_errors[arm],
        })
    del output, ids, mask, position_ids
    return records


def fraction(rows: list[dict], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def cell_metrics(rows: list[dict], families: list[str], gate: dict) -> dict:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in gate["arms"]}
    family_threshold = gate["family_paired_win_fraction_min"]
    identity_families = [family for family in families if fraction([row for row in by_arm["correct_identity"] if row["family"] == family], "desired_sign") >= family_threshold]
    wrong_families = [family for family in families if fraction([row for row in by_arm["wrong_identity"] if row["family"] == family], "wrong_expected_sign") >= family_threshold]
    controls = {
        "self_output_max_abs_diff": max(row["recipient_output_max_abs_diff"] for row in by_arm["self"]),
        "correct_identity_desired_sign_fraction": fraction(by_arm["correct_identity"], "desired_sign"),
        "wrong_identity_expected_sign_fraction": fraction(by_arm["wrong_identity"], "wrong_expected_sign"),
        "correct_identity_family_breadth": identity_families,
        "wrong_identity_family_breadth": wrong_families,
    }
    executor = (
        controls["self_output_max_abs_diff"] <= gate["self_max_abs_diff"]
        and controls["correct_identity_desired_sign_fraction"] >= gate["identity_desired_sign_fraction_min"]
        and controls["wrong_identity_expected_sign_fraction"] >= gate["wrong_expected_sign_fraction_min"]
        and len(identity_families) >= gate["minimum_family_breadth"]
        and len(wrong_families) >= gate["minimum_family_breadth"]
    )
    arm_metrics = {
        arm: {
            "count": len(values),
            "desired_sign_fraction": fraction(values, "desired_sign"),
            "oriented_gain_median": med([row["oriented_gain"] for row in values]),
            "swap_margin_median": med([row["swap_margin"] for row in values]),
        }
        for arm, values in by_arm.items()
    }
    indexed = {arm: {row["set_id"]: row for row in values} for arm, values in by_arm.items()}
    set_ids = sorted(indexed["p07"])
    gaps = [indexed["p07"][set_id]["oriented_gain"] - indexed["p23"][set_id]["oriented_gain"] for set_id in set_ids]
    semantic_families = []
    physical_families = []
    for family in families:
        family_ids = [set_id for set_id in set_ids if indexed["p07"][set_id]["family"] == family]
        semantic_fraction = sum(indexed["p07"][set_id]["oriented_gain"] > indexed["p23"][set_id]["oriented_gain"] for set_id in family_ids) / len(family_ids)
        physical_fraction = sum(indexed["p23"][set_id]["oriented_gain"] > indexed["p07"][set_id]["oriented_gain"] for set_id in family_ids) / len(family_ids)
        if semantic_fraction >= gate["family_paired_win_fraction_min"]:
            semantic_families.append(family)
        if physical_fraction >= gate["family_paired_win_fraction_min"]:
            physical_families.append(family)
    semantic_win_fraction = sum(value > 0 for value in gaps) / len(gaps)
    physical_win_fraction = sum(value < 0 for value in gaps) / len(gaps)
    gap_median = med(gaps)
    semantic_winner = (
        executor
        and arm_metrics["p07"]["desired_sign_fraction"] >= gate["arm_desired_sign_fraction_min"]
        and semantic_win_fraction >= gate["paired_win_fraction_min"]
        and gap_median >= gate["paired_gain_gap_median_min"]
        and len(semantic_families) >= gate["minimum_family_breadth"]
    )
    physical_winner = (
        executor
        and arm_metrics["p23"]["desired_sign_fraction"] >= gate["arm_desired_sign_fraction_min"]
        and physical_win_fraction >= gate["paired_win_fraction_min"]
        and gap_median <= -gate["paired_gain_gap_median_min"]
        and len(physical_families) >= gate["minimum_family_breadth"]
    )
    interactions = [
        indexed["p07"][set_id]["oriented_gain"]
        - indexed["p01"][set_id]["oriented_gain"]
        - indexed["p06"][set_id]["oriented_gain"]
        + indexed["correct_identity"][set_id]["oriented_gain"]
        for set_id in set_ids
    ]
    return {
        "count_per_arm": len(set_ids),
        "executor_pass": executor,
        "controls": controls,
        "arms": arm_metrics,
        "p07_minus_p23": {
            "semantic_p07_win_fraction": semantic_win_fraction,
            "physical_p23_win_fraction": physical_win_fraction,
            "median_oriented_gain_gap": gap_median,
            "semantic_family_breadth": semantic_families,
            "physical_family_breadth": physical_families,
        },
        "semantic_side_winner": semantic_winner,
        "physical_phase_winner": physical_winner,
        "interaction_descriptive_only_median": med(interactions),
    }


def classify(cells: dict, protocol: dict, execution_ok: bool) -> tuple[str, dict]:
    mechanism = protocol["mechanism"]
    all_cells = [cells[route][direction][split] for route in mechanism["routes"] for direction in DIRECTIONS for split in SPLITS]
    reversed_cells = [cells[route][direction][split] for route in mechanism["reversed_routes"] for direction in DIRECTIONS for split in SPLITS]
    same_cells = [cells[route][direction][split] for route in mechanism["same_order_routes"] for direction in DIRECTIONS for split in SPLITS]
    counts = {
        "total_executor_pass": sum(cell["executor_pass"] for cell in all_cells),
        "total_cells": len(all_cells),
        "reversed_semantic_winners": sum(cell["semantic_side_winner"] for cell in reversed_cells),
        "reversed_physical_winners": sum(cell["physical_phase_winner"] for cell in reversed_cells),
        "same_order_semantic_winners": sum(cell["semantic_side_winner"] for cell in same_cells),
        "same_order_physical_winners": sum(cell["physical_phase_winner"] for cell in same_cells),
    }
    if not execution_ok or counts["total_executor_pass"] != counts["total_cells"]:
        classification = "executor_failed"
    elif counts["reversed_semantic_winners"] == mechanism["strong_required_reversed_cells"] and counts["same_order_semantic_winners"] >= mechanism["strong_required_same_order_cells"]:
        classification = "semantic_side_confirmed"
    elif counts["reversed_physical_winners"] == mechanism["strong_required_reversed_cells"]:
        classification = "physical_phase_confirmed"
    elif counts["reversed_semantic_winners"] >= mechanism["conditional_required_reversed_cells"] and counts["reversed_physical_winners"] == 0:
        classification = "conditional_semantic_side"
    elif counts["reversed_physical_winners"] >= mechanism["conditional_required_reversed_cells"] and counts["reversed_semantic_winners"] == 0:
        classification = "conditional_physical_phase"
    else:
        classification = "mixed_or_no_stable_separation"
    return classification, counts


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1443 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    if camera_final["authorization"] != "run_phase1443_c073_side_phase_competition" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    holdouts = [row for row in selected if row["partition"] in SPLITS]
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    registry = {row["permutation_id"]: row for row in core.rows(CONTRACT / "material/permutation_registry.jsonl")}
    permutation_ids = [arm for arm in protocol["mechanism"]["arms"] if arm.startswith("p")]
    permutations = {permutation_id: registry[permutation_id] for permutation_id in permutation_ids}
    holdout_ids = "\n".join(sorted(row["set_id"] for row in holdouts))
    reveal = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "holdout_set_ids_sha256": hashlib.sha256(holdout_ids.encode("utf-8")).hexdigest(),
        "holdout_count": len(holdouts),
        "routes": protocol["mechanism"]["routes"],
        "directions": protocol["mechanism"]["directions"],
        "arms": protocol["mechanism"]["arms"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_shot": True,
    }
    core.save(OUT / "protocol/reveal_manifest.json", reveal)
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in holdouts:
            for route in protocol["mechanism"]["routes"]:
                for direction in DIRECTIONS:
                    records.extend(run_context(model, pad, device, supports, case, route, direction, compiled, permutations, protocol))
        core.write_rows(OUT / "raw/side_phase_competition.jsonl", records)
        gate = protocol["mechanism"]
        checks = {
            "holdouts": len(holdouts) == gate["holdout_sets"] and all(sum(row["partition"] == split for row in holdouts) == 24 for split in SPLITS),
            "record_count": len(records) == gate["holdout_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(gate["arms"]),
            "balance": all(sum(row["partition"] == split and row["route"] == route and row["direction"] == direction and row["arm"] == arm for row in records) == 24 for split in SPLITS for route in gate["routes"] for direction in DIRECTIONS for arm in gate["arms"]),
            "holdout_only": {row["partition"] for row in records} == set(SPLITS),
            "state16": {row["state_index"] for row in records} == {16},
            "route_shapes": all((row["source_length"] == row["target_length"]) == (row["route_order"] == "same_order") for row in records),
            "write_errors": max(max(row["write_max_abs_diff"], row["complement_max_abs_diff"]) for row in records) <= protocol["camera"]["write_max_abs_diff"],
            "finite": all(math.isfinite(row[key]) for row in records for key in ("recipient_margin", "correct_donor_margin", "wrong_donor_margin", "swap_margin", "oriented_gain", "recipient_output_max_abs_diff", "write_max_abs_diff", "complement_max_abs_diff")),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        cells = {
            route: {
                direction: {
                    split: cell_metrics([row for row in records if row["route"] == route and row["direction"] == direction and row["partition"] == split], behavior_final["qualified_families"], gate)
                    for split in SPLITS
                }
                for direction in DIRECTIONS
            }
            for route in gate["routes"]
        }
        classification, counts = classify(cells, protocol, all(checks.values()))
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_set_count": len(holdouts),
            "record_count": len(records),
            "cell_results": cells,
            "classification_counts": counts,
            "overall_classification": classification,
            "checks": checks,
            "all_execution_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"],
            "reveal_manifest": reveal,
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/side_phase_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_execution_checks_passed": summary["all_execution_checks_passed"],
            "overall_classification": classification,
            "classification_counts": counts,
            "authorization": "run_phase1444_c073_campaign_closure",
        })
        print(json.dumps({
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_set_count": len(holdouts),
            "record_count": len(records),
            "classification_counts": counts,
            "overall_classification": classification,
            "checks": checks,
        }, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
