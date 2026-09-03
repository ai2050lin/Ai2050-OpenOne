#!/usr/bin/env python3
"""Phase1522: one authoritative Qwen forward for C089 behavior and full field."""
from __future__ import annotations

import inspect
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
OUT = RESULT / "phase1522_c089_unified_forward_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1392_c062_full_field_camera import make_batch

BATCH = 12


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def balanced_accuracy(rows: list[dict]) -> float:
    return sum(accuracy([row for row in rows if row["gold_label"] == label]) for label in ("yes", "no")) / 2


def role_weights(batch: list[dict], offsets: list[int], width: int, roles: list[str], device) -> torch.Tensor:
    weights = torch.zeros((len(batch), len(roles), width), dtype=torch.float32, device=device)
    for i, row in enumerate(batch):
        for ri, role in enumerate(roles):
            points = [offsets[i] + point for point in row["role_positions"][role]]
            weights[i, ri, points] = 1.0 / len(points)
    return weights


@torch.inference_mode()
def run(cases: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    raw = OUT / "raw/all_role_field.float16.npy"
    raw.parent.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        roles = protocol["roles"]
        dim = int(model.config.hidden_size)
        shape = (len(cases), 37, len(roles), dim)
        field = np.lib.format.open_memmap(raw, mode="w+", dtype=np.float16, shape=shape)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        index, finite, first_scores = [], True, None
        for start in range(0, len(cases), BATCH):
            batch = cases[start:start + BATCH]
            ids, mask, pos, offsets = make_batch(batch, pad, device)
            kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports:
                kwargs["logits_to_keep"] = 1
            out = model(**kwargs)
            if len(out.hidden_states) != 37:
                raise RuntimeError(("state_count", len(out.hidden_states)))
            weights = role_weights(batch, offsets, ids.shape[1], roles, device)
            block = np.empty((len(batch), 37, len(roles), dim), dtype=np.float16)
            for state, hidden in enumerate(out.hidden_states):
                pooled = torch.einsum("brs,bsd->brd", weights, hidden.float())
                finite = finite and bool(torch.isfinite(pooled).all())
                block[:, state] = pooled.to(dtype=torch.float16, device="cpu").numpy()
            field[start:start + len(batch)] = block
            logits = out.logits[:, -1].float()
            batch_scores = []
            for i, row in enumerate(batch):
                scores = [float(logits[i, ids_[0]]) for ids_ in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                batch_scores.append(scores)
                index.append({
                    "row_index": start + i, "case_id": row["case_id"], "set_id": row["set_id"],
                    "family": row["family"], "partition": row["partition"], "surface": row["surface"],
                    "cell": row["cell"], "source": row["source"], "target": row["target"],
                    "truth": row["truth"], "truth_sign": row["truth_sign"], "gold_label": row["gold_label"],
                    "candidates": row["candidates"], "gold_position": row["gold_position"], "scores": scores,
                    "prediction": prediction, "predicted_label": row["candidates"][prediction],
                    "correct": prediction == row["gold_position"], "role_positions": row["role_positions"],
                })
            if start == 0:
                first_scores = batch_scores
            del out, ids, mask, pos, weights, block, logits
        field.flush()
        del field

        batch = cases[:BATCH]
        ids, mask, pos, _ = make_batch(batch, pad, device)
        kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "output_hidden_states": True, "return_dict": True}
        if supports:
            kwargs["logits_to_keep"] = 1
        repeat_out = model(**kwargs)
        repeat_logits = repeat_out.logits[:, -1].float()
        repeat = max(abs(first_scores[i][j] - float(repeat_logits[i, batch[i]["candidate_ids"][j][0]])) for i in range(BATCH) for j in range(2))
        return index, {
            "shape": list(shape), "hidden_dim": dim, "finite_during_capture": finite,
            "numeric_repeat_max_abs_diff": repeat, "placement": placement, "quantization": quant,
        }
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1522 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1522_c089_unified_forward_capture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1521 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(CONTRACT / "material/relation_composition_sets.jsonl")
    rows, runtime = run(compiled, protocol)
    by = {row["case_id"]: row for row in rows}
    cells = ("aa", "ab", "ba", "bb")
    stratified = []
    for group in groups:
        keys = [next(row["case_id"] for row in rows if row["set_id"] == group["set_id"] and row["surface"] == surface and row["cell"] == cell) for surface in ("a_question", "b_question") for cell in cells]
        correct = sum(by[key]["correct"] for key in keys)
        stratum = "success" if correct == 8 else "failed" if correct == 0 else "mixed"
        stratified.append({"set_id": group["set_id"], "family": group["family"], "partition": group["partition"], "correct_count": correct, "case_count": 8, "stratum": stratum})
    family_summary = {}
    qualified = []
    for family in protocol["families"]:
        family_rows = [row for row in rows if row["family"] == family]
        discovery = [row for row in family_rows if row["partition"] == "response_discovery"]
        surface_accuracy = {surface: accuracy([row for row in discovery if row["surface"] == surface]) for surface in ("a_question", "b_question")}
        qualifies = balanced_accuracy(discovery) >= protocol["behavior_qualification"]["discovery_family_balanced_accuracy"] and all(value >= protocol["behavior_qualification"]["discovery_each_surface_accuracy"] for value in surface_accuracy.values())
        family_summary[family] = {
            "count": len(family_rows), "accuracy": accuracy(family_rows), "balanced_accuracy": balanced_accuracy(family_rows),
            "partition": {partition: {"accuracy": accuracy(cell := [row for row in family_rows if row["partition"] == partition]), "balanced_accuracy": balanced_accuracy(cell)} for partition in protocol["partitions"]},
            "discovery_surface_accuracy": surface_accuracy, "behavior_qualified": qualifies,
        }
        if qualifies:
            qualified.append(family)
    summary = {
        "phase": 1522, "campaign": "C089", "global_accuracy": accuracy(rows), "global_balanced_accuracy": balanced_accuracy(rows),
        "family": family_summary, "behavior_qualified_families": qualified,
        "partition": {partition: {"accuracy": accuracy(cell := [row for row in rows if row["partition"] == partition]), "balanced_accuracy": balanced_accuracy(cell)} for partition in protocol["partitions"]},
        "surface": {surface: {"accuracy": accuracy(cell := [row for row in rows if row["surface"] == surface]), "balanced_accuracy": balanced_accuracy(cell)} for surface in ("a_question", "b_question")},
        "stratum_counts": dict(Counter(row["stratum"] for row in stratified)), "error_count": sum(not row["correct"] for row in rows),
        "runtime": runtime, "single_authoritative_forward": True,
    }
    raw = OUT / "raw/all_role_field.float16.npy"
    index_path = OUT / "raw/all_role_field_index.jsonl"
    strata_path = OUT / "material/stratified_relation_sets.jsonl"
    core.write_rows(index_path, rows)
    core.write_rows(strata_path, stratified)
    arr = np.load(raw, mmap_mode="r")
    checks = {
        "count": len(rows) == 360, "shape": list(arr.shape) == [360, 37, 4, 2560], "dtype": arr.dtype == np.float16,
        "finite": runtime["finite_during_capture"] and all(math.isfinite(value) for row in rows for value in row["scores"]),
        "repeat": runtime["numeric_repeat_max_abs_diff"] <= 1e-6,
        "strata": len(stratified) == 45 and sum(Counter(row["stratum"] for row in stratified).values()) == 45,
        "bf16": runtime["quantization"]["has_bf16_parameters"], "not_quantized": not runtime["quantization"]["has_quantized_modules"],
        "single_forward": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    summary["files"] = {
        "field": {"bytes": raw.stat().st_size, "sha256": core.sha(raw)},
        "index": {"sha256": core.sha(index_path)}, "strata": {"sha256": core.sha(strata_path)},
    }
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(OUT / "analysis/unified_behavior_and_capture_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1522, "campaign": "C089", "status": "authoritative_behavior_and_field_capture_complete", "behavior_qualified_families": qualified, "authorization": "run_phase1523_c089_truth_contrast_atlas"})
    print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
