#!/usr/bin/env python3
"""Phase1415: one-shot holdout test of C067 paired whole-state composition."""
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

PHASE, CAMPAIGN = 1415, "C067"
CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
BEHAVIOR = TESTS / "result/phase1413_c067_behavior"
CAMERA = TESTS / "result/phase1414_c067_dual_write_camera"
OUT = TESTS / "result/phase1415_c067_paired_composition"

ARMS = (
    "self_dual",
    "surface_dual",
    "member_dual",
    "record_only_g",
    "query_only_g",
    "matched_dual_g",
    "matched_dual_h",
    "mismatched_dual_gh",
    "mismatched_dual_hg",
)
DONOR_KEYS = ("recipient", "surface_same", "member_same", "g_true", "h_true")
SOURCE = {
    "self_dual": (0, 0),
    "surface_dual": (1, 1),
    "member_dual": (2, 2),
    "record_only_g": (3, 0),
    "query_only_g": (0, 3),
    "matched_dual_g": (3, 3),
    "matched_dual_h": (4, 4),
    "mismatched_dual_gh": (3, 4),
    "mismatched_dual_hg": (4, 3),
}


def margin(logits: torch.Tensor, row: dict) -> float:
    yes_id = row["candidate_ids"][0][0]
    no_id = row["candidate_ids"][1][0]
    return float(logits[yes_id].float() - logits[no_id].float())


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def loss_fraction(base: float, changed: float) -> float:
    return max(0.0, base - changed) / (abs(base) + 1e-12)


@torch.inference_mode()
def run_case(model, pad: int, device: torch.device, supports: bool, case: dict,
             compiled: dict[str, dict], state_index: int) -> list[dict]:
    donors = [compiled[case[key]] for key in DONOR_KEYS]
    recipient = donors[0]
    rows = donors + [recipient for _ in ARMS]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        for arm_offset, arm in enumerate(ARMS, start=len(donors)):
            record_source, query_source = SOURCE[arm]
            for role, source_index in (("record_family", record_source), ("query_family", query_source)):
                source_points = batcher.points(donors[source_index], offsets[source_index], role)
                target_points = batcher.points(recipient, offsets[arm_offset], role)
                if len(source_points) != 1 or len(target_points) != 1:
                    raise RuntimeError(f"non-singleton role: {role}")
                value[arm_offset, target_points[0]] = original[source_index, source_points[0]]
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
    records = []
    for row_index, arm in enumerate(ARMS, start=len(donors)):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"],
            "partition": case["partition"],
            "family": case["family"],
            "g_family": case["g_family"],
            "h_family": case["h_family"],
            "surface": case["surface"],
            "state_index": state_index,
            "arm": arm,
            "baseline_margin": baseline,
            "swap_margin": changed,
            "signed_damage": baseline - changed,
            "loss_fraction": loss_fraction(baseline, changed),
            "positive": changed > 0.0,
            "negative": changed < 0.0,
        })
    del output, ids, mask, position_ids
    return records


def group_metrics(rows: list[dict], gate: dict) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    if not rows or not all(arms.values()):
        raise RuntimeError("missing arm")
    by_arm_set = {arm: {row["set_id"]: row for row in values} for arm, values in arms.items()}
    set_ids = sorted(by_arm_set["self_dual"])
    if not all(set(values) == set(set_ids) for values in by_arm_set.values()):
        raise RuntimeError("unbalanced arms")
    self_error = max(abs(row["baseline_margin"] - row["swap_margin"]) for row in arms["self_dual"])
    surface_loss = median([row["loss_fraction"] for row in arms["surface_dual"]])
    member_loss = median([row["loss_fraction"] for row in arms["member_dual"]])
    record_damage_values = [row["signed_damage"] for row in arms["record_only_g"]]
    query_redirect = sum(row["negative"] for row in arms["query_only_g"]) / len(arms["query_only_g"])
    matched_rows = arms["matched_dual_g"] + arms["matched_dual_h"]
    mismatch_rows = arms["mismatched_dual_gh"] + arms["mismatched_dual_hg"]
    rescue_advantages = []
    matched_mismatch_advantages = []
    for set_id in set_ids:
        matched = (by_arm_set["matched_dual_g"][set_id]["swap_margin"] + by_arm_set["matched_dual_h"][set_id]["swap_margin"]) / 2.0
        single_best = max(by_arm_set["record_only_g"][set_id]["swap_margin"], by_arm_set["query_only_g"][set_id]["swap_margin"])
        mismatch = (by_arm_set["mismatched_dual_gh"][set_id]["swap_margin"] + by_arm_set["mismatched_dual_hg"][set_id]["swap_margin"]) / 2.0
        rescue_advantages.append(matched - single_best)
        matched_mismatch_advantages.append(matched - mismatch)
    metrics = {
        "count": len(set_ids),
        "self_max_abs_diff": self_error,
        "surface_control_loss_fraction_median": surface_loss,
        "member_control_loss_fraction_median": member_loss,
        "record_damage_median": median(record_damage_values),
        "record_damage_win_fraction": sum(value > 0.0 for value in record_damage_values) / len(record_damage_values),
        "query_redirect_fraction": query_redirect,
        "matched_positive_fraction": sum(row["positive"] for row in matched_rows) / len(matched_rows),
        "matched_rescue_advantage_median": median(rescue_advantages),
        "mismatched_negative_fraction": sum(row["negative"] for row in mismatch_rows) / len(mismatch_rows),
        "matched_over_mismatched_median": median(matched_mismatch_advantages),
    }
    checks = {
        "self": metrics["self_max_abs_diff"] <= gate["self_max_abs_diff"],
        "surface_control": metrics["surface_control_loss_fraction_median"] <= gate["surface_control_loss_fraction_max"],
        "member_control": metrics["member_control_loss_fraction_median"] <= gate["member_control_loss_fraction_max"],
        "record_damage": metrics["record_damage_median"] >= gate["record_damage_median_min"],
        "record_win": metrics["record_damage_win_fraction"] >= gate["record_damage_win_fraction_min"],
        "query_redirect": metrics["query_redirect_fraction"] >= gate["query_redirect_fraction_min"],
        "matched_positive": metrics["matched_positive_fraction"] >= gate["matched_positive_fraction_min"],
        "matched_rescue": metrics["matched_rescue_advantage_median"] >= gate["matched_rescue_advantage_median_min"],
        "mismatched_negative": metrics["mismatched_negative_fraction"] >= gate["mismatched_negative_fraction_min"],
        "matched_over_mismatched": metrics["matched_over_mismatched_median"] >= gate["matched_over_mismatched_median_min"],
    }
    return {**metrics, "checks": checks, "qualified": all(checks.values())}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1415 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if camera_final["authorization"] != "run_phase1415_c067_paired_composition" or not camera_audit["all_checks_passed"]:
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
            records.extend(run_case(model, pad, device, supports, case, compiled, gate["state_index"]))
        core.write_rows(OUT / "raw/paired_composition.jsonl", records)
        split_metrics = {
            split: group_metrics([row for row in records if row["partition"] == split], gate)
            for split in ("confirmation", "lockbox")
        }
        family_metrics = {}
        qualified_families = []
        for family in behavior_final["qualified_families"]:
            values = {
                split: group_metrics([row for row in records if row["partition"] == split and row["family"] == family], gate)
                for split in ("confirmation", "lockbox")
            }
            family_metrics[family] = values
            if all(result["qualified"] for result in values.values()):
                qualified_families.append(family)
        composition_confirmed = all(result["qualified"] for result in split_metrics.values()) and len(qualified_families) >= gate["minimum_family_breadth"]
        checks = {
            "holdout_sets": len(holdouts) == 48,
            "split_balance": sum(row["partition"] == "confirmation" for row in holdouts) == 24 and sum(row["partition"] == "lockbox" for row in holdouts) == 24,
            "record_count": len(records) == 48 * len(ARMS),
            "arms": {row["arm"] for row in records} == set(ARMS),
            "state16_only": {row["state_index"] for row in records} == {16},
            "holdout_only": {row["partition"] for row in records} == {"confirmation", "lockbox"},
            "finite": all(math.isfinite(row[key]) for row in records for key in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_set_count": len(holdouts),
            "record_count": len(records),
            "split_metrics": split_metrics,
            "family_metrics": family_metrics,
            "qualified_families": qualified_families,
            "composition_confirmed": composition_confirmed,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"],
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/composition_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_checks_passed": summary["all_checks_passed"],
            "composition_confirmed": composition_confirmed,
            "qualified_families": qualified_families,
            "authorization": "run_phase1416_c067_campaign_closure",
        })
        print(json.dumps({key: value for key, value in summary.items() if key != "family_metrics"}, indent=2))
        print(json.dumps({family: {split: {"qualified": result["qualified"], **{key: value for key, value in result.items() if key not in ("checks", "qualified")}} for split, result in values.items()} for family, values in family_metrics.items()}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
