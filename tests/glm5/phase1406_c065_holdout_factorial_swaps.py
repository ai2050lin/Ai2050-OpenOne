#!/usr/bin/env python3
"""Phase1406: test frozen C065 natural-state candidates on untouched holdouts."""
from __future__ import annotations

import inspect
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1392_c062_full_field_camera as batcher

PHASE, CAMPAIGN = 1406, "C065"
CONTRACT = TESTS / "result/phase1403_c065_active_only_natural_state_contract"
CAMERA = TESTS / "result/phase1404_c065_state_swap_camera"
FIELD = TESTS / "result/phase1405_c065_natural_discovery_field"
MATERIAL = TESTS / "result/phase1400_c064_fixed_answer_factorial_contract"
OUT = TESTS / "result/phase1406_c065_holdout_factorial_swaps"
ARMS = (
    "self",
    "surface_same",
    "member_same",
    "family_same_polarity",
    "polarity_same_family",
    "family_and_polarity",
)
DONOR_KEY = {
    "self": "recipient",
    "surface_same": "surface_same",
    "member_same": "member_same",
    "family_same_polarity": "family_same_polarity",
    "polarity_same_family": "polarity_same_family",
    "family_and_polarity": "family_and_polarity",
}


def margin(logits: torch.Tensor, row: dict) -> float:
    yes_id = row["candidate_ids"][0][0]
    no_id = row["candidate_ids"][1][0]
    return float(logits[yes_id].float() - logits[no_id].float())


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def loss_fraction(base: float, changed: float) -> float:
    return max(0.0, base - changed) / (abs(base) + 1e-12)


@torch.inference_mode()
def run_case(model, pad: int, device: torch.device, supports: bool, case: dict,
             compiled: dict[str, dict], candidates: list[dict]) -> list[dict]:
    donor_rows = [compiled[case[DONOR_KEY[arm]]] for arm in ARMS]
    recipient = donor_rows[0]
    target_specs = [(candidate, arm) for candidate in candidates for arm in ARMS]
    rows = donor_rows + [recipient for _ in target_specs]
    ids, mask, pos, offsets = batcher.make_batch(rows, pad, device)
    handles = []
    by_state: dict[int, list[tuple[int, dict, str]]] = defaultdict(list)
    for target_offset, (candidate, arm) in enumerate(target_specs, start=len(ARMS)):
        by_state[candidate["state_index"]].append((target_offset, candidate, arm))

    for state_index, specs in by_state.items():
        def hook(_module, args, specs=tuple(specs)):
            original = args[0]
            value = original.clone()
            for target_index, candidate, arm in specs:
                donor_index = ARMS.index(arm)
                role = candidate["role"]
                target_points = recipient["role_positions"][role]
                donor_points = donor_rows[donor_index]["role_positions"][role]
                if len(target_points) != 1 or len(donor_points) != 1:
                    raise RuntimeError(f"non-singleton role {role}")
                tp = offsets[target_index] + target_points[0]
                dp = offsets[donor_index] + donor_points[0]
                value[target_index, tp] = original[donor_index, dp]
            return (value,) + args[1:]
        handles.append(model.model.layers[state_index].register_forward_pre_hook(hook))

    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": pos,
        "use_cache": False,
        "return_dict": True,
    }
    if supports:
        kwargs["logits_to_keep"] = 1
    try:
        out = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    baseline = margin(out.logits[0, -1], recipient)
    records = []
    for target_index, (candidate, arm) in enumerate(target_specs, start=len(ARMS)):
        changed = margin(out.logits[target_index, -1], recipient)
        records.append({
            "set_id": case["set_id"],
            "partition": case["partition"],
            "family": case["family"],
            "surface": case["surface"],
            "candidate_id": candidate["candidate_id"],
            "object": candidate["object"],
            "window_index": candidate["window_index"],
            "state_index": candidate["state_index"],
            "role": candidate["role"],
            "arm": arm,
            "baseline_margin": baseline,
            "swap_margin": changed,
            "signed_damage": baseline - changed,
            "loss_fraction": loss_fraction(baseline, changed),
            "redirected_to_no": changed < 0.0,
        })
    del out, ids, mask, pos
    return records


def group_metrics(rows: list[dict], object_name: str, gate: dict) -> dict:
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    if not all(arms.values()):
        raise RuntimeError("missing arm")
    self_error = max(abs(r["baseline_margin"] - r["swap_margin"]) for r in arms["self"])
    surface_loss = med([r["loss_fraction"] for r in arms["surface_same"]])
    member_loss = med([r["loss_fraction"] for r in arms["member_same"]])
    family_damage = med([r["signed_damage"] for r in arms["family_same_polarity"]])
    member_damage = {r["set_id"]: r["signed_damage"] for r in arms["member_same"]}
    advantages = [r["signed_damage"] - member_damage[r["set_id"]]
                  for r in arms["family_same_polarity"]]
    polarity_redirect = sum(r["redirected_to_no"] for r in arms["polarity_same_family"]) / len(arms["polarity_same_family"])
    checks = {
        "self": self_error <= gate["self_max_abs_diff"],
        "surface_control": surface_loss <= gate["surface_control_loss_fraction_max"],
        "member_control": member_loss <= gate["member_control_loss_fraction_max"],
    }
    if object_name == "family_identity":
        checks.update({
            "family_damage": family_damage >= gate["family_damage_median_min"],
            "family_over_member": med(advantages) >= gate["family_over_member_median_min"],
            "family_win": sum(v > 0.0 for v in advantages) / len(advantages) >= gate["family_over_member_win_min"],
        })
    else:
        checks["polarity_redirect"] = polarity_redirect >= gate["polarity_redirect_fraction_min"]
    return {
        "count": len(rows) // len(ARMS),
        "self_max_abs_diff": self_error,
        "surface_control_loss_fraction_median": surface_loss,
        "member_control_loss_fraction_median": member_loss,
        "family_damage_median": family_damage,
        "family_over_member_median": med(advantages),
        "family_over_member_win_fraction": sum(v > 0.0 for v in advantages) / len(advantages),
        "polarity_redirect_fraction": polarity_redirect,
        "checks": checks,
        "qualified": all(checks.values()),
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1406 exists")
    field_final = core.load(FIELD / "analysis/final.json")
    field_audit = core.load(FIELD / "audit/independent_final_audit.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if field_final["authorization"] != "run_phase1406_c065_holdout_factorial_swaps":
        raise RuntimeError("field did not authorize")
    if not field_audit["all_checks_passed"] or not camera_audit["all_checks_passed"]:
        raise RuntimeError("parent audit failed")
    candidates_doc = core.load(FIELD / "protocol/frozen_natural_event_candidates.json")
    candidates = candidates_doc["candidates"]
    expected_field_hash = core.sha(FIELD / "raw/natural_full_field.jsonl")
    if candidates_doc["source_field_sha256"] != expected_field_hash:
        raise RuntimeError("candidate source hash mismatch")
    selected = core.rows(CONTRACT / "material/eligible_factor_sets.jsonl")
    holdouts = [r for r in selected if r["partition"] in ("confirmation", "lockbox")]
    compiled = {r["case_id"]: r for r in core.rows(MATERIAL / "compiled/qwen3_active.jsonl")}
    gate = protocol["factorial_swap"]
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in holdouts:
            surface_candidates = [r for r in candidates if r["surface"] == case["surface"]]
            if len(surface_candidates) != 6:
                raise RuntimeError("candidate surface count")
            records.extend(run_case(model, pad, device, supports, case, compiled, surface_candidates))
        core.write_rows(OUT / "raw/factorial_swaps.jsonl", records)

        candidate_summary = {}
        for candidate in candidates:
            cid = candidate["candidate_id"]
            route_rows = [r for r in records if r["candidate_id"] == cid]
            split_metrics = {
                split: group_metrics([r for r in route_rows if r["partition"] == split], candidate["object"], gate)
                for split in ("confirmation", "lockbox")
            }
            family_metrics = {}
            qualified_families = []
            for family in protocol["material"]["qualified_families"]:
                fm = {
                    split: group_metrics([r for r in route_rows if r["partition"] == split and r["family"] == family], candidate["object"], gate)
                    for split in ("confirmation", "lockbox")
                }
                family_metrics[family] = fm
                if all(v["qualified"] for v in fm.values()):
                    qualified_families.append(family)
            qualified = (
                all(v["qualified"] for v in split_metrics.values())
                and len(qualified_families) >= gate["minimum_family_breadth"]
            )
            candidate_summary[cid] = {
                "candidate": candidate,
                "split_metrics": split_metrics,
                "family_metrics": family_metrics,
                "qualified_families": qualified_families,
                "qualified": qualified,
            }

        family_candidates = [cid for cid, value in candidate_summary.items()
                             if value["candidate"]["object"] == "family_identity" and value["qualified"]]
        polarity_candidates = [cid for cid, value in candidate_summary.items()
                               if value["candidate"]["object"] == "joint_polarity" and value["qualified"]]
        route_status = {
            "family_identity": {"qualified_candidates": family_candidates, "confirmed": bool(family_candidates)},
            "joint_polarity": {"qualified_candidates": polarity_candidates, "confirmed": bool(polarity_candidates)},
        }
        checks = {
            "holdout_count": len(holdouts) == 36,
            "partition_balance": sum(r["partition"] == "confirmation" for r in holdouts) == 18 and sum(r["partition"] == "lockbox" for r in holdouts) == 18,
            "record_count": len(records) == 36 * 6 * 6,
            "candidate_count": len(candidates) == 18,
            "arms": set(r["arm"] for r in records) == set(ARMS),
            "holdout_only": not any(r["partition"] == "response_discovery" for r in records),
            "finite": all(math.isfinite(r[key]) for r in records for key in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
            "self_identity": max(abs(r["signed_damage"]) for r in records if r["arm"] == "self") <= gate["self_max_abs_diff"],
            "bf16": quant["has_bf16_parameters"],
            "not_quantized": not quant["has_quantized_modules"],
        }
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "holdout_count": len(holdouts),
            "record_count": len(records),
            "candidate_summary": candidate_summary,
            "route_status": route_status,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "candidate_sha256": core.sha(FIELD / "protocol/frozen_natural_event_candidates.json"),
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        core.save(OUT / "analysis/factorial_swap_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_checks_passed": summary["all_checks_passed"],
            "route_status": route_status,
            "authorization": "run_phase1407_c065_campaign_closure",
        })
        print(json.dumps({k: v for k, v in summary.items() if k != "candidate_summary"}, indent=2))
        print(json.dumps({
            cid: {
                "object": value["candidate"]["object"],
                "state": value["candidate"]["state_index"],
                "role": value["candidate"]["role"],
                "qualified_families": value["qualified_families"],
                "qualified": value["qualified"],
                "split": {split: metrics["qualified"] for split, metrics in value["split_metrics"].items()},
            }
            for cid, value in candidate_summary.items()
        }, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
