#!/usr/bin/env python3
"""Phase1527: calibrate singleton no-padding execution against the invalid left-padded C089 capture."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
OLD = RESULT / "phase1522_c089_unified_forward_capture"
PARENT = RESULT / "phase1526_c089_full_dimensional_diagnostics"
OUT = RESULT / "phase1527_c090_singleton_numeric_calibration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16


@torch.inference_mode()
def one(model, row, device, supports):
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1], device=device)[None]
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "output_hidden_states": True, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    out = model(**kwargs)
    pooled = np.empty((37, 4, int(model.config.hidden_size)), dtype=np.float32)
    roles = ("source_word", "target_word", "relation_anchor", "boundary")
    for state, hidden in enumerate(out.hidden_states):
        for ri, role in enumerate(roles):
            points = torch.tensor(row["role_positions"][role], dtype=torch.long, device=device)
            pooled[state, ri] = hidden[0, points].float().mean(dim=0).cpu().numpy()
    logits = out.logits[0, -1].float()
    scores = np.asarray([float(logits[ids_[0]]) for ids_ in row["candidate_ids"]], dtype=np.float64)
    del out, ids, mask, pos, logits
    return pooled, scores


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1527 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    calibration = core.load(PARENT / "protocol/singleton_calibration_protocol.json")
    selected = core.rows(PARENT / "protocol/singleton_calibration_cases.jsonl")
    if parent["authorization"] != "run_phase1527_c090_singleton_numeric_calibration" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1526 authorization missing")
    if core.sha(PARENT / "protocol/singleton_calibration_cases.jsonl") != calibration["case_sha256"]:
        raise RuntimeError("calibration selection mismatch")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    old_index = {row["case_id"]: row for row in core.rows(OLD / "raw/all_role_field_index.jsonl")}
    old_field = np.load(OLD / "raw/all_role_field.float16.npy", mmap_mode="r")
    rows = [compiled[row["case_id"]] for row in selected]
    field_path = OUT / "raw/singleton_calibration_field.float32.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float32, shape=(72, 37, 4, 2560))
    scores, model = [], None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        first = []
        for i, row in enumerate(rows):
            pooled, score = one(model, row, device, supports)
            field[i] = pooled
            scores.append(score.tolist())
            if i < 12:
                first.append((pooled, score))
        field.flush()
        repeat_hidden, repeat_logits = 0.0, 0.0
        for i, row in enumerate(rows[:12]):
            pooled, score = one(model, row, device, supports)
            repeat_hidden = max(repeat_hidden, float(np.max(np.abs(first[i][0] - pooled))))
            repeat_logits = max(repeat_logits, float(np.max(np.abs(first[i][1] - score))))
    finally:
        if model is not None:
            release_bf16(model)
    del field
    singleton = np.load(field_path, mmap_mode="r")
    lookup = {(row["set_id"], row["surface"], row["cell"]): i for i, row in enumerate(rows)}
    causal_pairs = []
    for set_id in sorted({row["set_id"] for row in rows}):
        for surface in ("a_question", "b_question"):
            for left, right in (("aa", "ab"), ("bb", "ba")):
                a, b = lookup[(set_id, surface, left)], lookup[(set_id, surface, right)]
                relative = []
                max_abs = []
                for state in range(37):
                    x, y = np.asarray(singleton[a, state, 0], dtype=np.float64), np.asarray(singleton[b, state, 0], dtype=np.float64)
                    relative.append(float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)))
                    max_abs.append(float(np.max(np.abs(x - y))))
                causal_pairs.append({"set_id": set_id, "surface": surface, "pair": f"{left}__{right}", "max_relative_l2": max(relative), "max_abs": max(max_abs), "state35_relative_l2": relative[35]})
    batch_comparison = []
    for i, row in enumerate(rows):
        old_row = old_index[row["case_id"]]
        old = np.asarray(old_field[old_row["row_index"]], dtype=np.float64)
        new = np.asarray(singleton[i], dtype=np.float64)
        batch_comparison.append({
            "case_id": row["case_id"],
            "field_relative_l2": float(np.linalg.norm(old - new) / (np.linalg.norm(new) + 1e-12)),
            "field_max_abs": float(np.max(np.abs(old - new))),
            "logit_max_abs": float(np.max(np.abs(np.asarray(old_row["scores"], dtype=np.float64) - np.asarray(scores[i], dtype=np.float64)))),
        })
    max_causal_relative = max(row["max_relative_l2"] for row in causal_pairs)
    canonical = repeat_hidden <= calibration["singleton_repeat_max_abs"] and repeat_logits <= calibration["singleton_repeat_max_abs"] and max_causal_relative <= calibration["singleton_causal_prefix_relative_l2"]
    summary = {
        "phase": 1527, "campaign": "C090", "case_count": len(rows),
        "singleton_repeat_hidden_max_abs": repeat_hidden, "singleton_repeat_logit_max_abs": repeat_logits,
        "singleton_causal_prefix_max_relative_l2": max_causal_relative,
        "singleton_causal_prefix_max_abs": max(row["max_abs"] for row in causal_pairs),
        "left_batch_vs_singleton": {
            "field_relative_l2_max": max(row["field_relative_l2"] for row in batch_comparison),
            "field_relative_l2_mean": float(np.mean([row["field_relative_l2"] for row in batch_comparison])),
            "field_max_abs": max(row["field_max_abs"] for row in batch_comparison),
            "logit_max_abs": max(row["logit_max_abs"] for row in batch_comparison),
        },
        "thresholds": calibration,
        "canonical_singleton_engine_pass": canonical,
        "runtime": {"placement": placement, "quantization": quant},
    }
    checks = {
        "count": len(rows) == len(selected) == calibration["case_count"] == 72,
        "shape": list(singleton.shape) == [72, 37, 4, 2560],
        "finite": bool(np.isfinite(np.asarray(singleton)).all()) and all(np.isfinite(value) for score in scores for value in score),
        "repeat": repeat_hidden <= calibration["singleton_repeat_max_abs"] and repeat_logits <= calibration["singleton_repeat_max_abs"],
        "causal_prefix": max_causal_relative <= calibration["singleton_causal_prefix_relative_l2"],
        "batch_difference_detected": summary["left_batch_vs_singleton"]["field_max_abs"] > 1e-2,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "canonical": canonical,
    }
    summary["checks"] = checks
    summary["files"] = {"field": {"sha256": core.sha(field_path), "bytes": field_path.stat().st_size}}
    core.write_rows(OUT / "analysis/singleton_causal_prefix_pairs.jsonl", causal_pairs)
    core.write_rows(OUT / "analysis/left_batch_vs_singleton.jsonl", batch_comparison)
    core.save(OUT / "analysis/singleton_numeric_calibration.json", summary)
    status = "singleton_no_padding_engine_calibrated" if canonical else "singleton_engine_failed_causal_prefix_gate"
    authorization = "run_phase1528_c090_singleton_full_recapture" if canonical else "preregister_phase1528_c090_right_padded_group_calibration"
    core.save(OUT / "analysis/final.json", {"phase": 1527, "campaign": "C090", "status": status, "authorization": authorization})
    print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
