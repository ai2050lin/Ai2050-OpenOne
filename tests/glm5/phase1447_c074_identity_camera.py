#!/usr/bin/env python3
"""Phase1447: calibrate the C074 identity-only directional transport camera."""
from __future__ import annotations

import inspect
import json
import math
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

PHASE, CAMPAIGN = 1447, "C074"
CONTRACT = TESTS / "result/phase1445_c074_directional_domain_contract"
BEHAVIOR = TESTS / "result/phase1446_c074_behavior"
OUT = TESTS / "result/phase1447_c074_identity_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")


def direction_rows(case: dict, compiled: dict[str, dict], source: str, target: str, direction: str) -> tuple[dict, dict, dict]:
    if direction == "true_to_false":
        return compiled[case[f"{target}_true_recipient"]], compiled[case[f"{source}_false_donor"]], compiled[case[f"{source}_true_donor"]]
    return compiled[case[f"{target}_false_recipient"]], compiled[case[f"{source}_true_donor"]], compiled[case[f"{source}_false_donor"]]


def known_truth(count: int, arms: list[str]) -> list[dict]:
    records = []
    source_positions = {role: point for role, point in zip(ROLES, (1, 4, 8, 11))}
    target_positions = {role: point for role, point in zip(ROLES, (0, 3, 6, 9))}
    target_role_points = set(target_positions.values())
    complement = [point for point in range(11) if point not in target_role_points]
    for seed in range(count):
        generator = torch.Generator().manual_seed(740000 + seed)
        correct = torch.randn(13, 32, generator=generator)
        wrong = torch.randn(13, 32, generator=generator)
        target = torch.randn(11, 32, generator=generator)
        for arm in arms:
            donor = target if arm == "self" else correct if arm == "correct_identity" else wrong
            points = target_positions if arm == "self" else source_positions
            changed = target.clone()
            for role in ROLES:
                changed[target_positions[role]] = donor[points[role]]
            records.append({
                "seed": seed, "arm": arm,
                "roles_exact": all(torch.equal(changed[target_positions[role]], donor[points[role]]) for role in ROLES),
                "complement_exact": torch.equal(changed[complement], target[complement]),
            })
    return records


@torch.inference_mode()
def run_qwen(cases: list[dict], compiled: dict[str, dict], protocol: dict) -> tuple[list[dict], dict]:
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in cases:
            for route in protocol["camera"]["routes"]:
                spec = protocol["routes"][route]
                source, target = spec["source"], spec["target"]
                for direction in DIRECTIONS:
                    recipient, correct, wrong = direction_rows(case, compiled, source, target, direction)
                    arms = protocol["camera"]["arms"]
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
                            measured[arm] = {"write_max_abs_diff": role_error, "complement_max_abs_diff": complement_error, "quartet_size": len(set(arm_points.values())), "complement_count": len(complement)}
                        return (value,) + args[1:]

                    handle = model.model.layers[protocol["state_index"]].register_forward_pre_hook(hook)
                    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    try:
                        output = model(**kwargs)
                    finally:
                        handle.remove()
                    for row_index, arm in enumerate(arms, start=3):
                        records.append({
                            "set_id": case["set_id"], "family": case["family"], "partition": case["partition"],
                            "route": route, "source_surface": source, "target_surface": target,
                            "same_surface": spec["same_surface"], "same_frame": spec["same_frame"], "same_order": spec["same_order"],
                            "direction": direction, "state_index": protocol["state_index"],
                            "source_length": len((correct if arm == "correct_identity" else wrong if arm == "wrong_identity" else recipient)["prompt_ids"]),
                            "target_length": len(recipient["prompt_ids"]), "arm": arm,
                            "self_output_max_abs_diff": float((output.logits[row_index, -1].float() - output.logits[0, -1].float()).abs().max()) if arm == "self" else 0.0,
                            **measured[arm],
                        })
                    del output, ids, mask, position_ids
        return records, {"placement": placement, "quantization": quant}
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1447 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if behavior_final["authorization"] != "run_phase1447_c074_identity_camera" or not behavior_audit["all_checks_passed"]:
        raise RuntimeError("behavior gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    cases = []
    for family in behavior_final["qualified_families"]:
        values = sorted([row for row in selected if row["partition"] == "response_discovery" and row["family"] == family], key=lambda row: row["set_id"])
        cases.extend(values[:2])
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    arms = protocol["camera"]["arms"]
    known = known_truth(protocol["camera"]["known_truth_systems"], arms)
    qwen, runtime = run_qwen(cases, compiled, protocol)
    core.write_rows(OUT / "raw/known_truth_identity.jsonl", known)
    core.write_rows(OUT / "raw/qwen_identity_camera.jsonl", qwen)
    gate = protocol["camera"]
    checks = {
        "known_count": len(known) == gate["known_truth_systems"] * len(arms),
        "known_exact": all(row["roles_exact"] and row["complement_exact"] for row in known),
        "arms": {row["arm"] for row in qwen} == set(arms),
        "qwen_sets": len(cases) == gate["qwen_discovery_sets"] and len({row["family"] for row in cases}) == 6,
        "qwen_count": len(qwen) == gate["qwen_discovery_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(arms),
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "routes": {row["route"] for row in qwen} == set(gate["routes"]),
        "directions": {row["direction"] for row in qwen} == set(gate["directions"]),
        "state16": {row["state_index"] for row in qwen} == {16},
        "quartet": all(row["quartet_size"] == 4 for row in qwen),
        "writes": max(row["write_max_abs_diff"] for row in qwen) <= gate["write_max_abs_diff"],
        "complement": max(row["complement_max_abs_diff"] for row in qwen) <= gate["untouched_complement_max_abs_diff"],
        "self": max(row["self_output_max_abs_diff"] for row in qwen if row["arm"] == "self") <= gate["self_output_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_output_max_abs_diff")),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "known_truth_count": len(known), "qwen_set_count": len(cases), "qwen_record_count": len(qwen),
        "max_errors": {key: max(row[key] for row in qwen) for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_output_max_abs_diff")},
        "checks": checks, "camera_qualified": all(checks.values()), "runtime": runtime, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/camera_summary.json", summary)
    authorization = "run_phase1448_c074_directional_domain_map" if summary["camera_qualified"] else "close_c074_at_camera_gate"
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": summary["camera_qualified"], "authorization": authorization})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
