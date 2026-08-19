#!/usr/bin/env python3
"""Phase1432: calibrate the C071 cross-surface semantic-role map camera."""
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

PHASE, CAMPAIGN = 1432, "C071"
CONTRACT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
BEHAVIOR = TESTS / "result/phase1431_c071_cross_surface_behavior"
OUT = TESTS / "result/phase1432_c071_role_map_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")


def known_truth(count: int, permutation: dict[str, str]) -> list[dict]:
    records = []
    source_positions = {role: point for role, point in zip(ROLES, (1, 4, 8, 11))}
    target_positions = {role: point for role, point in zip(ROLES, (0, 3, 6, 9))}
    target_roles = set(target_positions.values())
    target_complement = [point for point in range(11) if point not in target_roles]
    for seed in range(count):
        generator = torch.Generator().manual_seed(710000 + seed)
        source = torch.randn(13, 32, generator=generator)
        target = torch.randn(11, 32, generator=generator)
        self_write = target.clone()
        mapped = target.clone()
        permuted = target.clone()
        for role in ROLES:
            self_write[target_positions[role]] = target[target_positions[role]]
            mapped[target_positions[role]] = source[source_positions[role]]
            permuted[target_positions[role]] = source[source_positions[permutation[role]]]
        records.append({
            "seed": seed,
            "source_roles_distinct": len(set(source_positions.values())) == 4,
            "target_roles_distinct": len(set(target_positions.values())) == 4,
            "permutation_deranged": all(permutation[role] != role for role in ROLES) and set(permutation.values()) == set(ROLES),
            "self_exact": torch.equal(self_write, target),
            "mapped_roles_exact": all(torch.equal(mapped[target_positions[role]], source[source_positions[role]]) for role in ROLES),
            "permuted_roles_exact": all(torch.equal(permuted[target_positions[role]], source[source_positions[permutation[role]]]) for role in ROLES),
            "mapped_complement_exact": torch.equal(mapped[target_complement], target[target_complement]),
            "permuted_complement_exact": torch.equal(permuted[target_complement], target[target_complement]),
        })
    return records


def transfer_surfaces(transfer: str) -> tuple[str, str]:
    if transfer == "belongs_include_to_lists_member":
        return "belongs_include", "lists_member"
    if transfer == "lists_member_to_belongs_include":
        return "lists_member", "belongs_include"
    raise ValueError(transfer)


def direction_rows(case: dict, compiled: dict[str, dict], source_surface: str, target_surface: str, direction: str) -> tuple[dict, dict]:
    if direction == "true_to_false":
        return compiled[case[f"{target_surface}_true_recipient"]], compiled[case[f"{source_surface}_false_donor"]]
    return compiled[case[f"{target_surface}_false_recipient"]], compiled[case[f"{source_surface}_true_donor"]]


@torch.inference_mode()
def run_qwen(cases: list[dict], compiled: dict[str, dict], protocol: dict) -> tuple[list[dict], dict]:
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        state_index = protocol["camera"]["state_index"]
        permutation = protocol["role_map"]["permuted_source"]
        records = []
        for case in cases:
            for transfer in protocol["camera"]["surface_transfers"]:
                source_surface, target_surface = transfer_surfaces(transfer)
                for direction in DIRECTIONS:
                    recipient, donor = direction_rows(case, compiled, source_surface, target_surface, direction)
                    rows = [recipient, donor, recipient, recipient, recipient]
                    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
                    measured = {}

                    def hook(_module, args):
                        original = args[0]
                        value = original.clone()
                        source_points = {role: batcher.points(donor, offsets[1], role)[0] for role in ROLES}
                        target_points = {role: batcher.points(recipient, offsets[0], role)[0] for role in ROLES}
                        self_points = {role: batcher.points(recipient, offsets[2], role)[0] for role in ROLES}
                        mapped_points = {role: batcher.points(recipient, offsets[3], role)[0] for role in ROLES}
                        permuted_points = {role: batcher.points(recipient, offsets[4], role)[0] for role in ROLES}
                        active_target = list(range(offsets[0], ids.shape[1]))
                        complement = [point for point in active_target if point not in set(target_points.values())]
                        for role in ROLES:
                            value[2, self_points[role]] = original[0, target_points[role]]
                            value[3, mapped_points[role]] = original[1, source_points[role]]
                            value[4, permuted_points[role]] = original[1, source_points[permutation[role]]]
                        measured["self_role_max_abs_diff"] = max(float((value[2, self_points[role]].float() - original[0, target_points[role]].float()).abs().max()) for role in ROLES)
                        measured["mapped_role_max_abs_diff"] = max(float((value[3, mapped_points[role]].float() - original[1, source_points[role]].float()).abs().max()) for role in ROLES)
                        measured["permuted_role_max_abs_diff"] = max(float((value[4, permuted_points[role]].float() - original[1, source_points[permutation[role]]].float()).abs().max()) for role in ROLES)
                        measured["mapped_complement_max_abs_diff"] = float((value[3, complement].float() - original[0, complement].float()).abs().max())
                        measured["permuted_complement_max_abs_diff"] = float((value[4, complement].float() - original[0, complement].float()).abs().max())
                        measured["source_role_points"] = source_points
                        measured["target_role_points"] = target_points
                        measured["target_complement_count"] = len(complement)
                        return (value,) + args[1:]

                    handle = model.model.layers[state_index].register_forward_pre_hook(hook)
                    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    try:
                        output = model(**kwargs)
                    finally:
                        handle.remove()
                    self_output_diff = float((output.logits[2, -1].float() - output.logits[0, -1].float()).abs().max())
                    records.append({
                        "set_id": case["set_id"], "family": case["family"], "partition": case["partition"],
                        "surface_transfer": transfer, "source_surface": source_surface, "target_surface": target_surface,
                        "direction": direction, "state_index": state_index,
                        "source_length": len(donor["prompt_ids"]), "target_length": len(recipient["prompt_ids"]),
                        **measured, "self_output_max_abs_diff": self_output_diff,
                    })
                    del output, ids, mask, position_ids
        return records, {"placement": placement, "quantization": quant}
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1432 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if behavior_final["authorization"] != "run_phase1432_c071_role_map_camera" or not behavior_audit["all_checks_passed"]:
        raise RuntimeError("behavior gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    cases = [row for row in selected if row["partition"] == "response_discovery"]
    if len(cases) != protocol["camera"]["qwen_discovery_sets"]:
        raise RuntimeError("camera set count")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    known = known_truth(protocol["camera"]["known_truth_systems"], protocol["role_map"]["permuted_source"])
    qwen, runtime = run_qwen(cases, compiled, protocol)
    core.write_rows(OUT / "raw/known_truth_systems.jsonl", known)
    core.write_rows(OUT / "raw/qwen_role_map_camera.jsonl", qwen)
    gate = protocol["camera"]
    checks = {
        "known_count": len(known) == gate["known_truth_systems"],
        "known_maps": all(all(row[key] for key in ("source_roles_distinct", "target_roles_distinct", "permutation_deranged", "self_exact", "mapped_roles_exact", "permuted_roles_exact", "mapped_complement_exact", "permuted_complement_exact")) for row in known),
        "qwen_count": len(qwen) == gate["qwen_discovery_sets"] * len(gate["surface_transfers"]) * len(gate["directions"]),
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "transfers": {row["surface_transfer"] for row in qwen} == set(gate["surface_transfers"]),
        "directions": {row["direction"] for row in qwen} == set(gate["directions"]),
        "state16": {row["state_index"] for row in qwen} == {16},
        "different_shapes": all(row["source_length"] != row["target_length"] for row in qwen),
        "role_points": all(len(set(row["source_role_points"].values())) == 4 and len(set(row["target_role_points"].values())) == 4 for row in qwen),
        "self_roles": max(row["self_role_max_abs_diff"] for row in qwen) <= gate["self_role_max_abs_diff"],
        "mapped_roles": max(row["mapped_role_max_abs_diff"] for row in qwen) <= gate["mapped_role_max_abs_diff"],
        "mapped_complement": max(row["mapped_complement_max_abs_diff"] for row in qwen) <= gate["untouched_complement_max_abs_diff"],
        "permuted_complement": max(row["permuted_complement_max_abs_diff"] for row in qwen) <= gate["untouched_complement_max_abs_diff"],
        "self_output": max(row["self_output_max_abs_diff"] for row in qwen) <= gate["self_role_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("self_role_max_abs_diff", "mapped_role_max_abs_diff", "permuted_role_max_abs_diff", "mapped_complement_max_abs_diff", "permuted_complement_max_abs_diff", "self_output_max_abs_diff")),
        "bf16": runtime["quantization"]["has_bf16_parameters"], "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "known_truth_count": len(known),
        "qwen_set_count": len(cases), "qwen_case_count": len(qwen),
        "max_errors": {key: max(row[key] for row in qwen) for key in ("self_role_max_abs_diff", "mapped_role_max_abs_diff", "permuted_role_max_abs_diff", "mapped_complement_max_abs_diff", "permuted_complement_max_abs_diff", "self_output_max_abs_diff")},
        "checks": checks, "camera_qualified": all(checks.values()), "runtime": runtime,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/camera_summary.json", summary)
    authorization = "run_phase1433_c071_cross_surface_mechanism" if summary["camera_qualified"] else "close_c071_at_camera_gate"
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": summary["camera_qualified"], "authorization": authorization})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
