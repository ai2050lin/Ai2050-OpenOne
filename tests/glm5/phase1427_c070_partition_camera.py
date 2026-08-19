#!/usr/bin/env python3
"""Phase1427: calibrate the frozen C070 quartet/complement partition camera."""
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
import phase1392_c062_full_field_camera as field
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1427, "C070"
CONTRACT = TESTS / "result/phase1425_c070_quartet_complement_contract"
BEHAVIOR = TESTS / "result/phase1426_c070_roster_behavior"
OUT = TESTS / "result/phase1427_c070_partition_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")
DIRECTIONS = ("true_to_false", "false_to_true")


def known_truth(count: int) -> list[dict]:
    records = []
    quartet = torch.tensor([2, 3, 7, 8])
    complement = torch.tensor([0, 1, 4, 5, 6, 9, 10])
    for seed in range(count):
        generator = torch.Generator().manual_seed(700000 + seed)
        source = torch.randn(11, 32, generator=generator)
        target = torch.randn(11, 32, generator=generator)
        quartet_write = target.clone(); quartet_write[quartet] = source[quartet]
        complement_write = target.clone(); complement_write[complement] = source[complement]
        full_write = target.clone(); full_write[:] = source
        self_write = target.clone(); self_write[:] = target
        records.append({
            "seed": seed,
            "partition_disjoint": not set(quartet.tolist()) & set(complement.tolist()),
            "partition_complete": set(quartet.tolist()) | set(complement.tolist()) == set(range(11)),
            "quartet_exact": torch.equal(quartet_write[quartet], source[quartet]) and torch.equal(quartet_write[complement], target[complement]),
            "complement_exact": torch.equal(complement_write[complement], source[complement]) and torch.equal(complement_write[quartet], target[quartet]),
            "full_exact": torch.equal(full_write, source),
            "self_exact": torch.equal(self_write, target),
        })
    return records


def direction_rows(case: dict, compiled: dict[str, dict], direction: str) -> tuple[dict, dict]:
    if direction == "true_to_false":
        return compiled[case["true_recipient"]], compiled[case["false_donor"]]
    return compiled[case["false_recipient"]], compiled[case["true_donor"]]


@torch.inference_mode()
def run_qwen(cases: list[dict], compiled: dict[str, dict], state_index: int) -> tuple[list[dict], dict]:
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case in cases:
            for direction in DIRECTIONS:
                recipient, donor = direction_rows(case, compiled, direction)
                rows = [recipient, donor, recipient, recipient]
                ids, mask, position_ids, offsets = field.make_batch(rows, pad, device)
                if len({int(mask[index].sum()) for index in range(len(rows))}) != 1:
                    raise RuntimeError("non-isomorphic prompt lengths")

                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    value[2] = original[0]
                    value[3] = original[1]
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
                self_diff = float((output.logits[2, -1].float() - output.logits[0, -1].float()).abs().max())
                donor_diff = float((output.logits[3, -1].float() - output.logits[1, -1].float()).abs().max())
                records.append({
                    "set_id": case["set_id"],
                    "family": case["family"],
                    "partition": case["partition"],
                    "direction": direction,
                    "state_index": state_index,
                    "sequence_length": int(mask[0].sum()),
                    "role_points": {role: field.points(recipient, offsets[0], role) for role in ROLES},
                    "self_full_max_abs_diff": self_diff,
                    "donor_full_max_abs_diff": donor_diff,
                })
                del output, ids, mask, position_ids
        return records, {"placement": placement, "quantization": quant}
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1427 exists")
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if final["authorization"] != "run_phase1427_c070_partition_camera" or not audit["all_checks_passed"]:
        raise RuntimeError("behavior gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    cases = [row for row in selected if row["partition"] == "response_discovery"]
    if len(cases) != protocol["camera"]["qwen_discovery_sets"]:
        raise RuntimeError("camera set count")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    known = known_truth(protocol["camera"]["known_truth_systems"])
    qwen, runtime = run_qwen(cases, compiled, protocol["camera"]["state_index"])
    core.write_rows(OUT / "raw/known_truth_systems.jsonl", known)
    core.write_rows(OUT / "raw/qwen_full_state_transport.jsonl", qwen)
    checks = {
        "known_count": len(known) == protocol["camera"]["known_truth_systems"],
        "known_partition": all(row["partition_disjoint"] and row["partition_complete"] for row in known),
        "known_writes": all(all(row[key] for key in ("quartet_exact", "complement_exact", "full_exact", "self_exact")) for row in known),
        "qwen_count": len(qwen) == protocol["camera"]["qwen_discovery_sets"] * len(DIRECTIONS),
        "qwen_state": {row["state_index"] for row in qwen} == {protocol["camera"]["state_index"]},
        "qwen_roles": all(all(len(row["role_points"][role]) == 1 for role in ROLES) for row in qwen),
        "qwen_directions": {row["direction"] for row in qwen} == set(DIRECTIONS),
        "qwen_self": max(row["self_full_max_abs_diff"] for row in qwen) <= protocol["camera"]["self_full_max_abs_diff"],
        "qwen_donor": max(row["donor_full_max_abs_diff"] for row in qwen) <= protocol["camera"]["donor_full_transport_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("self_full_max_abs_diff", "donor_full_max_abs_diff")),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "known_truth_count": len(known),
        "qwen_set_count": len(cases),
        "qwen_case_count": len(qwen),
        "qwen_self_max_abs_diff": max(row["self_full_max_abs_diff"] for row in qwen),
        "qwen_donor_max_abs_diff": max(row["donor_full_max_abs_diff"] for row in qwen),
        "checks": checks,
        "camera_qualified": all(checks.values()),
        "runtime": runtime,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/camera_summary.json", summary)
    authorization = "run_phase1428_c070_support_partition" if summary["camera_qualified"] else "close_c070_at_camera_gate"
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "camera_qualified": summary["camera_qualified"],
        "authorization": authorization,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
