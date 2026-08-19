#!/usr/bin/env python3
"""Phase1414: calibrate the frozen C067 dual-role whole-state write camera."""
from __future__ import annotations

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

PHASE, CAMPAIGN = 1414, "C067"
CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
BEHAVIOR = TESTS / "result/phase1413_c067_behavior"
OUT = TESTS / "result/phase1414_c067_dual_write_camera"


def known_truth(count: int) -> list[dict]:
    records = []
    for seed in range(count):
        generator = torch.Generator().manual_seed(670000 + seed)
        source = torch.randn(11, 32, generator=generator)
        target = torch.randn(11, 32, generator=generator)
        record_points = torch.tensor([2, 3])
        query_points = torch.tensor([7, 8])
        written = target.clone()
        written[record_points] = source[record_points]
        written[query_points] = source[query_points]
        expected = target.clone()
        expected[record_points] = source[record_points]
        expected[query_points] = source[query_points]
        self_write = source.clone()
        self_write[record_points] = source[record_points]
        self_write[query_points] = source[query_points]
        records.append({
            "seed": seed,
            "dual_write_exact": torch.equal(written, expected),
            "self_dual_exact": torch.equal(self_write, source),
            "unwritten_exact": torch.equal(written[[0, 1, 4, 5, 6, 9, 10]], target[[0, 1, 4, 5, 6, 9, 10]]),
            "shape_exact": list(written.shape) == [11, 32],
        })
    return records


@torch.inference_mode()
def run_qwen(cases: list[dict], compiled: dict[str, dict], state_index: int) -> tuple[list[dict], dict]:
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        records = []
        for case in cases:
            row = compiled[case["recipient"]]
            rows = [row, row]
            ids, mask, position_ids, offsets = field.make_batch(rows, pad, device)

            def hook(_module, args):
                original = args[0]
                value = original.clone()
                for role in ("record_family", "query_family"):
                    source_points = field.points(rows[0], offsets[0], role)
                    target_points = field.points(rows[1], offsets[1], role)
                    if len(source_points) != len(target_points):
                        raise RuntimeError(f"role length mismatch: {role}")
                    value[1, target_points] = original[0, source_points]
                return (value,) + args[1:]

            handle = model.model.layers[state_index].register_forward_pre_hook(hook)
            try:
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                handle.remove()
            max_abs_diff = float((output.logits[0, -1].float() - output.logits[1, -1].float()).abs().max())
            records.append({
                "set_id": case["set_id"],
                "family": case["family"],
                "partition": case["partition"],
                "surface": case["surface"],
                "state_index": state_index,
                "record_points": field.points(row, offsets[0], "record_family"),
                "query_points": field.points(row, offsets[0], "query_family"),
                "output_max_abs_diff": max_abs_diff,
            })
            del output, ids, mask, position_ids
        return records, {"placement": placement, "quantization": quant}
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1414 exists")
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if final["authorization"] != "run_phase1414_c067_dual_write_camera" or not audit["all_checks_passed"]:
        raise RuntimeError("behavior gate missing")
    selected = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    cases = [row for row in selected if row["partition"] == "response_discovery"]
    if len(cases) != protocol["camera"]["qwen_discovery_sets"]:
        raise RuntimeError("camera case count")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    known = known_truth(protocol["camera"]["known_truth_systems"])
    qwen, runtime = run_qwen(cases, compiled, protocol["camera"]["state_index"])
    core.write_rows(OUT / "raw/known_truth_systems.jsonl", known)
    core.write_rows(OUT / "raw/qwen_dual_write_identity.jsonl", qwen)
    checks = {
        "known_count": len(known) == protocol["camera"]["known_truth_systems"],
        "known_dual": all(row["dual_write_exact"] for row in known),
        "known_self": all(row["self_dual_exact"] for row in known),
        "known_unwritten": all(row["unwritten_exact"] for row in known),
        "known_shape": all(row["shape_exact"] for row in known),
        "qwen_count": len(qwen) == protocol["camera"]["qwen_discovery_sets"],
        "qwen_state": {row["state_index"] for row in qwen} == {protocol["camera"]["state_index"]},
        "qwen_roles": all(len(row["record_points"]) == 1 and len(row["query_points"]) == 1 for row in qwen),
        "qwen_identity": max(row["output_max_abs_diff"] for row in qwen) <= protocol["camera"]["self_dual_max_abs_diff"],
        "finite": all(math.isfinite(row["output_max_abs_diff"]) for row in qwen),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "known_truth_count": len(known),
        "qwen_case_count": len(qwen),
        "qwen_output_max_abs_diff": max(row["output_max_abs_diff"] for row in qwen),
        "checks": checks,
        "camera_qualified": all(checks.values()),
        "runtime": runtime,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/camera_summary.json", summary)
    authorization = "run_phase1415_c067_paired_composition" if summary["camera_qualified"] else "close_c067_at_camera_gate"
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "camera_qualified": summary["camera_qualified"],
        "authorization": authorization,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
