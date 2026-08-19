#!/usr/bin/env python3
"""Phase1423: one-shot bidirectional four-role composition for C069."""
from __future__ import annotations

import inspect, json, math, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1392_c062_full_field_camera as batcher
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1423, "C069"
CONTRACT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
BEHAVIOR = TESTS / "result/phase1421_c069_catalog_behavior"
CAMERA = TESTS / "result/phase1422_c069_quartet_camera"
OUT = TESTS / "result/phase1423_c069_bidirectional_composition"
RECORD_ROLES = ("record_target", "record_family")
QUERY_ROLES = ("query_target", "query_family")
DIRECTIONS = ("true_recipient", "false_recipient")
ARMS = ("self", "surface", "member", "matched_g", "matched_h", "mismatched_gh", "mismatched_hg", "natural_false_gh", "natural_false_hg")
DONOR_KEYS = ("recipient", "surface", "member", "g_true", "h_true", "g_false_h", "h_false_g")
SOURCE = {
    "self": (0, 0), "surface": (1, 1), "member": (2, 2),
    "matched_g": (3, 3), "matched_h": (4, 4),
    "mismatched_gh": (3, 4), "mismatched_hg": (4, 3),
    "natural_false_gh": (5, 5), "natural_false_hg": (6, 6),
}


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[row["candidate_ids"][0][0]].float() - logits[row["candidate_ids"][1][0]].float())


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def donor_case_keys(case: dict, direction: str) -> dict[str, str]:
    truth = direction == "true_recipient"
    return {
        "recipient": case[direction],
        "surface": case["true_surface" if truth else "false_surface"],
        "member": case["true_member" if truth else "false_member"],
        "g_true": case["g_true"], "h_true": case["h_true"],
        "g_false_h": case["g_false_h"], "h_false_g": case["h_false_g"],
    }


@torch.inference_mode()
def run_direction(model, pad: int, device: torch.device, supports: bool, case: dict,
                  direction: str, compiled: dict[str, dict], state_index: int) -> list[dict]:
    key_map = donor_case_keys(case, direction)
    donors = [compiled[key_map[key]] for key in DONOR_KEYS]
    recipient = donors[0]
    rows = donors + [recipient for _ in ARMS]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        for target_index, arm in enumerate(ARMS, start=len(donors)):
            record_source, query_source = SOURCE[arm]
            for role in RECORD_ROLES:
                source_points = batcher.points(donors[record_source], offsets[record_source], role)
                target_points = batcher.points(recipient, offsets[target_index], role)
                if len(source_points) != 1 or len(target_points) != 1:
                    raise RuntimeError(f"non-singleton role: {role}")
                value[target_index, target_points[0]] = original[record_source, source_points[0]]
            for role in QUERY_ROLES:
                source_points = batcher.points(donors[query_source], offsets[query_source], role)
                target_points = batcher.points(recipient, offsets[target_index], role)
                if len(source_points) != 1 or len(target_points) != 1:
                    raise RuntimeError(f"non-singleton role: {role}")
                value[target_index, target_points[0]] = original[query_source, source_points[0]]
        return (value,) + args[1:]

    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    try:
        output = model(**kwargs)
    finally:
        handle.remove()
    baseline = margin(output.logits[0, -1], recipient)
    records = []
    for row_index, arm in enumerate(ARMS, start=len(donors)):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"], "partition": case["partition"], "family": case["family"],
            "g_family": case["g_family"], "h_family": case["h_family"], "direction": direction,
            "state_index": state_index, "arm": arm, "baseline_margin": baseline, "swap_margin": changed,
            "relative_deviation": abs(changed - baseline) / (abs(baseline) + 1e-12),
            "positive": changed > 0.0, "negative": changed < 0.0,
        })
    del output, ids, mask, position_ids
    return records


def group_metrics(rows: list[dict], gate: dict, direction: str) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    if not rows or not all(arms.values()):
        raise RuntimeError("missing arm")
    by_arm = {arm: {row["set_id"]: row for row in values} for arm, values in arms.items()}
    set_ids = sorted(by_arm["self"])
    if not all(set(values) == set(set_ids) for values in by_arm.values()):
        raise RuntimeError("unbalanced arms")
    true_direction = direction == "true_recipient"
    controls = arms["surface"] + arms["member"]
    control_sign = sum(row["positive"] if true_direction else row["negative"] for row in controls) / len(controls)
    matched = arms["matched_g"] + arms["matched_h"]
    mismatched = arms["mismatched_gh"] + arms["mismatched_hg"]
    natural_false = arms["natural_false_gh"] + arms["natural_false_hg"]
    interactions, rescue_gains = [], []
    for set_id in set_ids:
        gg = by_arm["matched_g"][set_id]["swap_margin"]
        hh = by_arm["matched_h"][set_id]["swap_margin"]
        gh = by_arm["mismatched_gh"][set_id]["swap_margin"]
        hg = by_arm["mismatched_hg"][set_id]["swap_margin"]
        interactions.append((gg + hh - gh - hg) / 2.0)
        rescue_gains.append((gg + hh) / 2.0 - by_arm["self"][set_id]["baseline_margin"])
    metrics = {
        "count": len(set_ids),
        "baseline_margin_median": med([row["baseline_margin"] for row in arms["self"]]),
        "self_max_abs_diff": max(abs(row["baseline_margin"] - row["swap_margin"]) for row in arms["self"]),
        "control_sign_fraction": control_sign,
        "control_relative_deviation_median": med([row["relative_deviation"] for row in controls]),
        "matched_positive_fraction": sum(row["positive"] for row in matched) / len(matched),
        "mismatched_negative_fraction": sum(row["negative"] for row in mismatched) / len(mismatched),
        "natural_false_negative_fraction": sum(row["negative"] for row in natural_false) / len(natural_false),
        "matched_rescue_gain_median": med(rescue_gains),
        "interaction_median": med(interactions),
        "interaction_win_fraction": sum(value > 0.0 for value in interactions) / len(interactions),
    }
    graded_checks = {
        "self": metrics["self_max_abs_diff"] <= gate["self_max_abs_diff"],
        "control_sign": metrics["control_sign_fraction"] >= gate["control_sign_fraction_min"],
        "control_deviation": metrics["control_relative_deviation_median"] <= gate["control_relative_deviation_median_max"],
        "interaction_median": metrics["interaction_median"] >= gate["interaction_median_min"],
        "interaction_win": metrics["interaction_win_fraction"] >= gate["interaction_win_fraction_min"],
        "false_rescue": true_direction or metrics["matched_rescue_gain_median"] >= gate["matched_rescue_gain_false_median_min"],
    }
    discrete_checks = {
        "matched_positive": metrics["matched_positive_fraction"] >= gate["matched_positive_true_min" if true_direction else "matched_positive_false_min"],
        "mismatched_negative": metrics["mismatched_negative_fraction"] >= gate["mismatched_negative_true_min" if true_direction else "mismatched_negative_false_min"],
        "natural_false_negative": metrics["natural_false_negative_fraction"] >= gate["natural_false_negative_true_min" if true_direction else "natural_false_negative_false_min"],
    }
    return {
        **metrics, "graded_checks": graded_checks, "discrete_checks": discrete_checks,
        "graded_qualified": all(graded_checks.values()), "discrete_qualified": all(discrete_checks.values()),
        "strong_qualified": all(graded_checks.values()) and all(discrete_checks.values()),
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1423 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if camera_final["authorization"] != "run_phase1423_c069_bidirectional_composition" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera did not authorize")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    holdouts = [row for row in selected if row["partition"] in ("confirmation", "lockbox")]
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
        core.write_rows(OUT / "raw/bidirectional_composition.jsonl", records)
        splits = ("confirmation", "lockbox")
        aggregate = {
            split: {direction: group_metrics([row for row in records if row["partition"] == split and row["direction"] == direction], gate, direction) for direction in DIRECTIONS}
            for split in splits
        }
        family_metrics = {}
        graded_families, discrete_families, strong_families = [], [], []
        for family in behavior_final["qualified_families"]:
            values = {
                split: {direction: group_metrics([row for row in records if row["partition"] == split and row["direction"] == direction and row["family"] == family], gate, direction) for direction in DIRECTIONS}
                for split in splits
            }
            family_metrics[family] = values
            groups = [values[split][direction] for split in splits for direction in DIRECTIONS]
            if all(group["graded_qualified"] for group in groups): graded_families.append(family)
            if all(group["discrete_qualified"] for group in groups): discrete_families.append(family)
            if all(group["strong_qualified"] for group in groups): strong_families.append(family)
        groups = [aggregate[split][direction] for split in splits for direction in DIRECTIONS]
        graded_confirmed = all(group["graded_qualified"] for group in groups) and len(graded_families) >= gate["minimum_family_breadth"]
        discrete_confirmed = all(group["discrete_qualified"] for group in groups) and len(discrete_families) >= gate["minimum_family_breadth"]
        strong_confirmed = graded_confirmed and discrete_confirmed and len(strong_families) >= gate["minimum_family_breadth"]
        checks = {
            "holdout_sets": len(holdouts) == 48,
            "split_balance": all(sum(row["partition"] == split for row in holdouts) == 24 for split in splits),
            "record_count": len(records) == 48 * len(DIRECTIONS) * len(ARMS),
            "arms": {row["arm"] for row in records} == set(ARMS), "directions": {row["direction"] for row in records} == set(DIRECTIONS),
            "state16_only": {row["state_index"] for row in records} == {16}, "holdout_only": {row["partition"] for row in records} == set(splits),
            "finite": all(math.isfinite(row[key]) for row in records for key in ("baseline_margin", "swap_margin", "relative_deviation")),
            "bf16": quant["has_bf16_parameters"], "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "holdout_set_count": len(holdouts), "record_count": len(records),
            "split_direction_metrics": aggregate, "family_metrics": family_metrics,
            "graded_qualified_families": graded_families, "discrete_qualified_families": discrete_families,
            "strong_qualified_families": strong_families, "graded_confirmed": graded_confirmed,
            "discrete_confirmed": discrete_confirmed, "strong_confirmed": strong_confirmed,
            "checks": checks, "all_checks_passed": all(checks.values()), "contract_sha256": protocol["contract_sha256"],
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/composition_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE, "campaign": CAMPAIGN, "all_checks_passed": summary["all_checks_passed"],
            "graded_confirmed": graded_confirmed, "discrete_confirmed": discrete_confirmed,
            "strong_confirmed": strong_confirmed, "authorization": "run_phase1424_c069_campaign_closure",
        })
        print(json.dumps({key: value for key, value in summary.items() if key != "family_metrics"}, indent=2))
        print(json.dumps({family: {split: {direction: {"graded": result["graded_qualified"], "discrete": result["discrete_qualified"], "strong": result["strong_qualified"]} for direction, result in directions.items()} for split, directions in values.items()} for family, values in family_metrics.items()}, indent=2))
    finally:
        if model is not None: release_bf16(model)


if __name__ == "__main__": main()
