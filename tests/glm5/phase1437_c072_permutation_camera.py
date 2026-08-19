#!/usr/bin/env python3
"""Phase1437: calibrate all 24 C072 semantic-role permutation writes."""
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

PHASE, CAMPAIGN = 1437, "C072"
CONTRACT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
BEHAVIOR = TESTS / "result/phase1436_c072_behavior"
OUT = TESTS / "result/phase1437_c072_permutation_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")


def transfer_surfaces(transfer: str) -> tuple[str, str]:
    if transfer == "memo_contains_to_circle_roll":
        return "memo_contains", "circle_roll"
    if transfer == "circle_roll_to_memo_contains":
        return "circle_roll", "memo_contains"
    raise ValueError(transfer)


def direction_rows(case: dict, compiled: dict[str, dict], source_surface: str, target_surface: str, direction: str) -> tuple[dict, dict]:
    if direction == "true_to_false":
        return compiled[case[f"{target_surface}_true_recipient"]], compiled[case[f"{source_surface}_false_donor"]]
    return compiled[case[f"{target_surface}_false_recipient"]], compiled[case[f"{source_surface}_true_donor"]]


def known_truth(count: int, registry: list[dict]) -> list[dict]:
    records = []
    source_positions = {role: point for role, point in zip(ROLES, (1, 4, 8, 11))}
    target_positions = {role: point for role, point in zip(ROLES, (0, 3, 6, 9))}
    target_role_points = set(target_positions.values())
    complement = [point for point in range(11) if point not in target_role_points]
    for seed in range(count):
        generator = torch.Generator().manual_seed(720000 + seed)
        source = torch.randn(13, 32, generator=generator)
        target = torch.randn(11, 32, generator=generator)
        for permutation in registry:
            changed = target.clone()
            for role in ROLES:
                changed[target_positions[role]] = source[source_positions[permutation["mapping"][role]]]
            records.append({
                "seed": seed, "permutation_id": permutation["permutation_id"],
                "roles_exact": all(torch.equal(changed[target_positions[role]], source[source_positions[permutation["mapping"][role]]]) for role in ROLES),
                "complement_exact": torch.equal(changed[complement], target[complement]),
            })
    return records


@torch.inference_mode()
def run_qwen(cases: list[dict], compiled: dict[str, dict], registry: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in cases:
            for transfer in protocol["camera"]["surface_transfers"]:
                source_surface, target_surface = transfer_surfaces(transfer)
                for direction in DIRECTIONS:
                    recipient, donor = direction_rows(case, compiled, source_surface, target_surface, direction)
                    rows = [recipient, donor, recipient] + [recipient for _ in registry]
                    ids, mask, position_ids, offsets = batcher.make_batch(rows, pad, device)
                    measured: dict[str, dict] = {}

                    def points(row: dict, batch_index: int) -> dict[str, int]:
                        return {role: batcher.points(row, offsets[batch_index], role)[0] for role in ROLES}

                    def hook(_module, args):
                        original = args[0]
                        value = original.clone()
                        recipient_points = points(recipient, 0)
                        donor_points = points(donor, 1)
                        self_points = points(recipient, 2)
                        for role in ROLES:
                            value[2, self_points[role]] = original[0, recipient_points[role]]
                        self_error = max(float((value[2, self_points[role]].float() - original[0, recipient_points[role]].float()).abs().max()) for role in ROLES)
                        for index, permutation in enumerate(registry, start=3):
                            arm_points = points(recipient, index)
                            for role in ROLES:
                                value[index, arm_points[role]] = original[1, donor_points[permutation["mapping"][role]]]
                            role_error = max(float((value[index, arm_points[role]].float() - original[1, donor_points[permutation["mapping"][role]]].float()).abs().max()) for role in ROLES)
                            active = list(range(offsets[index], ids.shape[1]))
                            complement = [point for point in active if point not in set(arm_points.values())]
                            complement_error = float((value[index, complement].float() - original[0, complement].float()).abs().max())
                            measured[permutation["permutation_id"]] = {"write_max_abs_diff": role_error, "complement_max_abs_diff": complement_error, "self_role_max_abs_diff": self_error, "quartet_size": len(set(arm_points.values())), "complement_count": len(complement)}
                        return (value,) + args[1:]

                    handle = model.model.layers[protocol["state_index"]].register_forward_pre_hook(hook)
                    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "use_cache": False, "return_dict": True}
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    try:
                        output = model(**kwargs)
                    finally:
                        handle.remove()
                    self_output_diff = float((output.logits[2, -1].float() - output.logits[0, -1].float()).abs().max())
                    for permutation in registry:
                        records.append({
                            "set_id": case["set_id"], "family": case["family"], "partition": case["partition"],
                            "surface_transfer": transfer, "source_surface": source_surface, "target_surface": target_surface,
                            "direction": direction, "state_index": protocol["state_index"],
                            "source_length": len(donor["prompt_ids"]), "target_length": len(recipient["prompt_ids"]),
                            "permutation_id": permutation["permutation_id"], **measured[permutation["permutation_id"]],
                            "self_output_max_abs_diff": self_output_diff,
                        })
                    del output, ids, mask, position_ids
        return records, {"placement": placement, "quantization": quant}
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1437 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if behavior_final["authorization"] != "run_phase1437_c072_permutation_camera" or not behavior_audit["all_checks_passed"]:
        raise RuntimeError("behavior gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    cases = []
    for family in behavior_final["qualified_families"]:
        values = sorted([row for row in selected if row["partition"] == "response_discovery" and row["family"] == family], key=lambda row: row["set_id"])
        cases.extend(values[:2])
    registry = core.rows(CONTRACT / "material/permutation_registry.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    known = known_truth(protocol["camera"]["known_truth_systems"], registry)
    qwen, runtime = run_qwen(cases, compiled, registry, protocol)
    core.write_rows(OUT / "raw/known_truth_permutations.jsonl", known)
    core.write_rows(OUT / "raw/qwen_permutation_camera.jsonl", qwen)
    gate = protocol["camera"]
    checks = {
        "known_count": len(known) == gate["known_truth_systems"] * gate["all_permutations"],
        "known_exact": all(row["roles_exact"] and row["complement_exact"] for row in known),
        "qwen_sets": len(cases) == gate["qwen_discovery_sets"] and len({row["family"] for row in cases}) == 6,
        "qwen_count": len(qwen) == gate["qwen_discovery_sets"] * 2 * 2 * 24,
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "permutations": len({row["permutation_id"] for row in qwen}) == 24 and all(sum(row["permutation_id"] == permutation["permutation_id"] for row in qwen) == gate["qwen_discovery_sets"] * 4 for permutation in registry),
        "transfers": {row["surface_transfer"] for row in qwen} == set(gate["surface_transfers"]),
        "directions": {row["direction"] for row in qwen} == set(gate["directions"]),
        "state16": {row["state_index"] for row in qwen} == {16},
        "different_shapes": all(row["source_length"] != row["target_length"] for row in qwen),
        "quartet": all(row["quartet_size"] == 4 for row in qwen),
        "writes": max(row["write_max_abs_diff"] for row in qwen) <= gate["write_max_abs_diff"],
        "complement": max(row["complement_max_abs_diff"] for row in qwen) <= gate["untouched_complement_max_abs_diff"],
        "self": max(max(row["self_role_max_abs_diff"], row["self_output_max_abs_diff"]) for row in qwen) <= gate["self_output_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_role_max_abs_diff", "self_output_max_abs_diff")),
        "bf16": runtime["quantization"]["has_bf16_parameters"], "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "known_truth_count": len(known), "qwen_set_count": len(cases), "qwen_case_count": len(qwen),
        "max_errors": {key: max(row[key] for row in qwen) for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_role_max_abs_diff", "self_output_max_abs_diff")},
        "checks": checks, "camera_qualified": all(checks.values()), "runtime": runtime, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/camera_summary.json", summary)
    authorization = "run_phase1438_c072_permutation_spectrum" if summary["camera_qualified"] else "close_c072_at_camera_gate"
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": summary["camera_qualified"], "authorization": authorization})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
