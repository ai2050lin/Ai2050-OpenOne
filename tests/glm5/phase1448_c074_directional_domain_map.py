#!/usr/bin/env python3
"""Phase1448: one-shot C074 identity-only directional transport domain map."""
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

PHASE, CAMPAIGN = 1448, "C074"
CONTRACT = TESTS / "result/phase1445_c074_directional_domain_contract"
BEHAVIOR = TESTS / "result/phase1446_c074_behavior"
CAMERA = TESTS / "result/phase1447_c074_identity_camera"
OUT = TESTS / "result/phase1448_c074_directional_domain_map"
ROLES = ("record_target", "record_family", "query_target", "query_family")
SPLITS = ("confirmation", "lockbox")
DIRECTIONS = ("true_to_false", "false_to_true")


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    weight = index - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def margin(logits: torch.Tensor, row: dict) -> float:
    return float(logits[row["candidate_ids"][0][0]].float() - logits[row["candidate_ids"][1][0]].float())


def direction_rows(case: dict, compiled: dict[str, dict], source: str, target: str, direction: str) -> tuple[dict, dict, dict, float]:
    if direction == "true_to_false":
        return compiled[case[f"{target}_true_recipient"]], compiled[case[f"{source}_false_donor"]], compiled[case[f"{source}_true_donor"]], -1.0
    return compiled[case[f"{target}_false_recipient"]], compiled[case[f"{source}_true_donor"]], compiled[case[f"{source}_false_donor"]], 1.0


@torch.inference_mode()
def run_context(model, pad: int, device: torch.device, supports: bool, case: dict, route: str, direction: str, compiled: dict[str, dict], protocol: dict) -> list[dict]:
    spec = protocol["routes"][route]
    source, target = spec["source"], spec["target"]
    recipient, correct, wrong, orientation = direction_rows(case, compiled, source, target, direction)
    arms = protocol["domain"]["arms"]
    rows = [recipient, correct, wrong] + [recipient for _ in arms]
    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
    measured: dict[str, dict] = {}

    def points(row: dict, batch_index: int) -> dict[str, int]:
        return {role: batcher.points(row, offsets[batch_index], role)[0] for role in ROLES}

    def hook(_module, args):
        original = args[0]
        value = original.clone()
        base_points = [points(recipient, 0), points(correct, 1), points(wrong, 2)]
        for row_index, arm in enumerate(arms, start=3):
            arm_points = points(recipient, row_index)
            donor_index = {"self": 0, "correct_identity": 1, "wrong_identity": 2}[arm]
            donor_points = base_points[donor_index]
            for role in ROLES:
                value[row_index, arm_points[role]] = original[donor_index, donor_points[role]]
            role_error = max(float((value[row_index, arm_points[role]].float() - original[donor_index, donor_points[role]].float()).abs().max()) for role in ROLES)
            active = list(range(offsets[row_index], ids.shape[1]))
            complement = [point for point in active if point not in set(arm_points.values())]
            complement_error = float((value[row_index, complement].float() - original[0, complement].float()).abs().max())
            measured[arm] = {"write_max_abs_diff": role_error, "complement_max_abs_diff": complement_error}
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
    correct_margin = margin(output.logits[1, -1], correct)
    wrong_margin = margin(output.logits[2, -1], wrong)
    records = []
    for row_index, arm in enumerate(arms, start=3):
        changed = margin(output.logits[row_index, -1], recipient)
        records.append({
            "set_id": case["set_id"], "partition": case["partition"], "family": case["family"], "donor_family": case["donor_family"],
            "route": route, "source_surface": source, "target_surface": target,
            "same_surface": spec["same_surface"], "same_frame": spec["same_frame"], "same_order": spec["same_order"],
            "direction": direction, "state_index": protocol["state_index"], "arm": arm,
            "source_length": len((correct if arm == "correct_identity" else wrong if arm == "wrong_identity" else recipient)["prompt_ids"]),
            "target_length": len(recipient["prompt_ids"]),
            "recipient_margin": recipient_margin, "correct_donor_margin": correct_margin, "wrong_donor_margin": wrong_margin,
            "swap_margin": changed, "oriented_gain": orientation * (changed - recipient_margin),
            "desired_sign": changed < 0 if direction == "true_to_false" else changed > 0,
            "wrong_expected_sign": changed > 0 if direction == "true_to_false" else changed < 0,
            "full_logit_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()),
            **measured[arm],
        })
    del output, ids, mask, position_ids
    return records


def fraction(rows: list[dict], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def arm_metrics(rows: list[dict], sign_field: str) -> dict:
    gains = [row["oriented_gain"] for row in rows]
    margins = [row["swap_margin"] for row in rows]
    return {
        "count": len(rows), "sign_fraction": fraction(rows, sign_field),
        "oriented_gain_min": min(gains), "oriented_gain_q25": quantile(gains, 0.25),
        "oriented_gain_median": med(gains), "oriented_gain_q75": quantile(gains, 0.75), "oriented_gain_max": max(gains),
        "swap_margin_min": min(margins), "swap_margin_median": med(margins), "swap_margin_max": max(margins),
    }


def cell_metrics(rows: list[dict], families: list[str], gate: dict) -> dict:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in gate["arms"]}
    identity_families = [family for family in families if fraction([row for row in by_arm["correct_identity"] if row["family"] == family], "desired_sign") >= gate["family_fraction_min"]]
    wrong_families = [family for family in families if fraction([row for row in by_arm["wrong_identity"] if row["family"] == family], "wrong_expected_sign") >= gate["family_fraction_min"]]
    self_max = max(row["full_logit_max_abs_diff"] for row in by_arm["self"])
    identity_fraction = fraction(by_arm["correct_identity"], "desired_sign")
    wrong_fraction = fraction(by_arm["wrong_identity"], "wrong_expected_sign")
    passed = self_max <= gate["self_max_abs_diff"] and identity_fraction >= gate["identity_desired_sign_fraction_min"] and wrong_fraction >= gate["wrong_expected_sign_fraction_min"] and len(identity_families) >= gate["minimum_family_breadth"] and len(wrong_families) >= gate["minimum_family_breadth"]
    return {
        "count_per_arm": len(by_arm["self"]), "pass": passed,
        "controls": {
            "self_output_max_abs_diff": self_max,
            "correct_identity_desired_sign_fraction": identity_fraction,
            "wrong_identity_expected_sign_fraction": wrong_fraction,
            "correct_identity_family_breadth": identity_families,
            "wrong_identity_family_breadth": wrong_families,
        },
        "correct_identity": arm_metrics(by_arm["correct_identity"], "desired_sign"),
        "wrong_identity": arm_metrics(by_arm["wrong_identity"], "wrong_expected_sign"),
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1448 exists")
    camera_final = core.load(CAMERA / "analysis/final.json")
    camera_audit = core.load(CAMERA / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    if camera_final["authorization"] != "run_phase1448_c074_directional_domain_map" or not camera_audit["all_checks_passed"]:
        raise RuntimeError("camera gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    holdouts = [row for row in selected if row["partition"] in SPLITS]
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    holdout_ids = "\n".join(sorted(row["set_id"] for row in holdouts))
    reveal = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "holdout_set_ids_sha256": hashlib.sha256(holdout_ids.encode("utf-8")).hexdigest(),
        "holdout_count": len(holdouts), "routes": protocol["domain"]["routes"],
        "directions": protocol["domain"]["directions"], "arms": protocol["domain"]["arms"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True,
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
            for route in protocol["domain"]["routes"]:
                for direction in DIRECTIONS:
                    records.extend(run_context(model, pad, device, supports, case, route, direction, compiled, protocol))
        core.write_rows(OUT / "raw/directional_domain.jsonl", records)
        gate = protocol["domain"]
        checks = {
            "holdouts": len(holdouts) == gate["holdout_sets"] and all(sum(row["partition"] == split for row in holdouts) == 24 for split in SPLITS),
            "record_count": len(records) == gate["holdout_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(gate["arms"]),
            "balance": all(sum(row["partition"] == split and row["route"] == route and row["direction"] == direction and row["arm"] == arm for row in records) == 24 for split in SPLITS for route in gate["routes"] for direction in DIRECTIONS for arm in gate["arms"]),
            "holdout_only": {row["partition"] for row in records} == set(SPLITS),
            "state16": {row["state_index"] for row in records} == {16},
            "route_metadata": all(protocol["routes"][row["route"]][key] == row[key] for row in records for key in ("same_surface", "same_frame", "same_order")),
            "write_errors": max(max(row["write_max_abs_diff"], row["complement_max_abs_diff"]) for row in records) <= protocol["camera"]["write_max_abs_diff"],
            "finite": all(math.isfinite(row[key]) for row in records for key in ("recipient_margin", "correct_donor_margin", "wrong_donor_margin", "swap_margin", "oriented_gain", "full_logit_max_abs_diff", "write_max_abs_diff", "complement_max_abs_diff")),
            "bf16": quant["has_bf16_parameters"], "not_quantized": not quant["has_quantized_modules"],
        }
        cells = {
            route: {direction: {split: cell_metrics([row for row in records if row["route"] == route and row["direction"] == direction and row["partition"] == split], behavior_final["qualified_families"], gate) for split in SPLITS} for direction in DIRECTIONS}
            for route in gate["routes"]
        }
        edges, robust = {}, []
        for route in gate["routes"]:
            edges[route] = {}
            for direction in DIRECTIONS:
                passed = [split for split in SPLITS if cells[route][direction][split]["pass"]]
                classification = "robust" if len(passed) == 2 else "split_specific" if len(passed) == 1 else "rejected"
                row = {"edge_id": f"{route}::{direction}", "route": route, "direction": direction, "classification": classification, "passed_splits": passed, **protocol["routes"][route]}
                edges[route][direction] = row
                if classification == "robust":
                    robust.append(row)
        core.write_rows(OUT / "analysis/robust_edges.jsonl", robust)
        class_counts = {name: sum(edges[route][direction]["classification"] == name for route in edges for direction in edges[route]) for name in gate["edge_classes"]}
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "holdout_set_count": len(holdouts), "record_count": len(records),
            "cell_count": len(gate["routes"]) * len(DIRECTIONS) * len(SPLITS), "edge_count": len(gate["routes"]) * len(DIRECTIONS),
            "cell_results": cells, "edge_results": edges, "class_counts": class_counts,
            "robust_edge_count": len(robust), "robust_edge_ids": [row["edge_id"] for row in robust],
            "checks": checks, "all_execution_checks_passed": all(checks.values()),
            "contract_sha256": protocol["contract_sha256"], "reveal_manifest": reveal,
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/directional_domain_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE, "campaign": CAMPAIGN, "all_execution_checks_passed": summary["all_execution_checks_passed"],
            "class_counts": class_counts, "robust_edge_count": len(robust), "robust_edge_ids": summary["robust_edge_ids"],
            "authorization": "run_phase1449_c074_campaign_closure",
        })
        print(json.dumps({key: value for key, value in summary.items() if key not in ("cell_results", "edge_results")}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
