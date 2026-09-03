#!/usr/bin/env python3
"""Phase1529: run the frozen right-padded same-shape quartet calibration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
PARENT = RESULT / "phase1528_c090_right_padded_group_calibration_contract"
OUT = RESULT / "phase1529_c090_right_padded_group_calibration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

ROLES = ("source_word", "target_word", "relation_anchor", "boundary")


def make_right_batch(rows, pad, device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, :len(value)] = value
        mask[i, :len(value)] = 1
        lengths.append(len(value))
    pos = mask.cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    return ids, mask, pos, lengths


@torch.inference_mode()
def run_batch(model, rows, pad, device):
    ids, mask, pos, lengths = make_right_batch(rows, pad, device)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, output_hidden_states=True, return_dict=True)
    pooled = np.empty((len(rows), 37, 4, int(model.config.hidden_size)), dtype=np.float32)
    for state, hidden in enumerate(out.hidden_states):
        for i, row in enumerate(rows):
            for ri, role in enumerate(ROLES):
                points = torch.tensor(row["role_positions"][role], dtype=torch.long, device=device)
                pooled[i, state, ri] = hidden[i, points].float().mean(dim=0).cpu().numpy()
    scores = []
    for i, row in enumerate(rows):
        logits = out.logits[i, lengths[i] - 1].float()
        scores.append([float(logits[ids_[0]]) for ids_ in row["candidate_ids"]])
    del out, ids, mask, pos
    return pooled, np.asarray(scores, dtype=np.float64)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1529 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(PARENT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1529_c090_right_padded_group_calibration" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1528 authorization missing")
    batches = core.rows(PARENT / "protocol/right_padded_calibration_batches.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    ordered = [case_id for batch in batches for case_id in batch["case_ids"]]
    field_path = OUT / "raw/right_padded_calibration_field.float32.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float32, shape=(72, 37, 4, 2560))
    scores, cursor, first = {}, 0, []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for bi, batch in enumerate(batches):
            rows = [compiled[case_id] for case_id in batch["case_ids"]]
            pooled, values = run_batch(model, rows, pad, device)
            field[cursor:cursor + 4] = pooled
            for i, case_id in enumerate(batch["case_ids"]):
                scores[case_id] = values[i].tolist()
            if bi < protocol["engine"]["repeat_batches"]:
                first.append((pooled, values, rows))
            cursor += 4
        field.flush()
        repeat_hidden, repeat_logits = 0.0, 0.0
        for pooled, values, rows in first:
            repeat_pool, repeat_scores = run_batch(model, rows, pad, device)
            repeat_hidden = max(repeat_hidden, float(np.max(np.abs(pooled - repeat_pool))))
            repeat_logits = max(repeat_logits, float(np.max(np.abs(values - repeat_scores))))
    finally:
        if model is not None:
            release_bf16(model)
    del field
    field = np.load(field_path, mmap_mode="r")
    row_index = {case_id: i for i, case_id in enumerate(ordered)}
    causal_pairs = []
    for batch in batches:
        by_cell = {compiled[case_id]["cell"]: case_id for case_id in batch["case_ids"]}
        for left, right in (("aa", "ab"), ("bb", "ba")):
            a, b = row_index[by_cell[left]], row_index[by_cell[right]]
            relative, max_abs = [], []
            for state in range(37):
                x, y = np.asarray(field[a, state, 0], dtype=np.float64), np.asarray(field[b, state, 0], dtype=np.float64)
                relative.append(float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)))
                max_abs.append(float(np.max(np.abs(x - y))))
            causal_pairs.append({"batch_id": batch["batch_id"], "pair": f"{left}__{right}", "max_relative_l2": max(relative), "max_abs": max(max_abs)})
    gates = protocol["gates"]
    causal_max = max(row["max_relative_l2"] for row in causal_pairs)
    passed = repeat_hidden <= gates["repeat_hidden_max_abs"] and repeat_logits <= gates["repeat_logit_max_abs"] and causal_max <= gates["causal_prefix_relative_l2"]
    summary = {
        "phase": 1529, "campaign": "C090", "case_count": len(ordered), "batch_count": len(batches),
        "repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits,
        "causal_prefix_max_relative_l2": causal_max, "causal_prefix_max_abs": max(row["max_abs"] for row in causal_pairs),
        "gates": gates, "canonical_right_padded_engine_pass": passed,
        "runtime": {"placement": placement, "quantization": quant},
    }
    checks = {
        "shape": list(field.shape) == [72, 37, 4, 2560], "finite": bool(np.isfinite(np.asarray(field)).all()),
        "coverage": len(scores) == len(ordered) == 72,
        "repeat": repeat_hidden <= gates["repeat_hidden_max_abs"] and repeat_logits <= gates["repeat_logit_max_abs"],
        "causal_prefix": causal_max <= gates["causal_prefix_relative_l2"],
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "canonical": passed,
    }
    summary["checks"] = checks
    summary["files"] = {"field": {"sha256": core.sha(field_path), "bytes": field_path.stat().st_size}}
    core.write_rows(OUT / "analysis/right_padded_causal_prefix_pairs.jsonl", causal_pairs)
    core.save(OUT / "analysis/right_padded_group_calibration.json", summary)
    status = "right_padded_group_engine_calibrated" if passed else "right_padded_group_engine_failed"
    authorization = "run_phase1530_c090_canonical_full_recapture" if passed else "close_c090_numeric_engine_search"
    core.save(OUT / "analysis/final.json", {"phase": 1529, "campaign": "C090", "status": status, "authorization": authorization})
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "summary": summary})
    print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
