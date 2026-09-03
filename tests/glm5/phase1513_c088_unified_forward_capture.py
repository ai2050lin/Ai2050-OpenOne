#!/usr/bin/env python3
"""Phase1513: one authoritative Qwen forward for C088 behavior and full field."""
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
CONTRACT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
OUT = RESULT / "phase1513_c088_unified_forward_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1392_c062_full_field_camera import make_batch

BATCH = 12


def accuracy(rows):
    return sum(row["correct"] for row in rows) / len(rows)


def balanced_accuracy(rows):
    return sum(accuracy([row for row in rows if row["gold_label"] == label]) for label in ("yes", "no")) / 2.0


@torch.inference_mode()
def run(cases, protocol):
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
            role_index = torch.tensor(
                [[offsets[i] + batch[i]["role_positions"][role][0] for role in roles] for i in range(len(batch))],
                dtype=torch.long,
                device=device,
            )
            bi = torch.arange(len(batch), device=device)[:, None]
            block = np.empty((len(batch), 37, len(roles), dim), dtype=np.float16)
            for state, hidden in enumerate(out.hidden_states):
                gathered = hidden[bi, role_index]
                finite = finite and bool(torch.isfinite(gathered).all())
                block[:, state] = gathered.to(dtype=torch.float16, device="cpu").numpy()
            field[start:start + len(batch)] = block
            logits = out.logits[:, -1].float()
            batch_scores = []
            for i, row in enumerate(batch):
                scores = [float(logits[i, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                batch_scores.append(scores)
                index.append({
                    "row_index": start + i,
                    "case_id": row["case_id"],
                    "set_id": row["set_id"],
                    "partition": row["partition"],
                    "material_source": row["material_source"],
                    "item": row["item"],
                    "source_instance_id": row["source_instance_id"],
                    "surface": row["surface"],
                    "codebook": row["codebook"],
                    "code_sign": row["code_sign"],
                    "semantic_match": row["semantic_match"],
                    "semantic_label": row["semantic_label"],
                    "semantic_sign": row["semantic_sign"],
                    "candidate": row["candidate"],
                    "gold_label": row["gold_label"],
                    "candidates": row["candidates"],
                    "gold_position": row["gold_position"],
                    "scores": scores,
                    "prediction": prediction,
                    "predicted_label": row["candidates"][prediction],
                    "correct": prediction == row["gold_position"],
                    "role_positions": row["role_positions"],
                })
            if start == 0:
                first_scores = batch_scores
            del out, ids, mask, pos, role_index, bi, block, logits
        field.flush()
        del field

        batch = cases[:BATCH]
        ids, mask, pos, _ = make_batch(batch, pad, device)
        kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "output_hidden_states": True, "return_dict": True}
        if supports:
            kwargs["logits_to_keep"] = 1
        repeat_out = model(**kwargs)
        repeat_logits = repeat_out.logits[:, -1].float()
        repeat = max(
            abs(first_scores[i][j] - float(repeat_logits[i, batch[i]["candidate_ids"][j][0]]))
            for i in range(BATCH) for j in range(2)
        )
        return index, {
            "shape": list(shape),
            "hidden_dim": dim,
            "finite_during_capture": finite,
            "numeric_repeat_max_abs_diff": repeat,
            "placement": placement,
            "quantization": quant,
        }
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1513 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1513_c088_unified_forward_capture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1512 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(CONTRACT / "material/composition_sets.jsonl")
    rows, runtime = run(compiled, protocol)
    keys = tuple(f"{surface}_{codebook}_{semantic}" for surface in protocol["surfaces"] for codebook in protocol["codebooks"] for semantic in ("same", "different"))
    by = {row["case_id"]: row for row in rows}
    stratified = []
    for group in groups:
        correct = sum(by[group[key]]["correct"] for key in keys)
        stratum = "success" if correct == 8 else "failed" if correct == 0 else "mixed"
        stratified.append({**group, "correct_count": correct, "case_count": 8, "stratum": stratum})
    strata = Counter(row["stratum"] for row in stratified)
    truth_code = {
        f"{semantic}_{codebook}": {
            "count": len(cell := [row for row in rows if row["semantic_label"] == semantic and row["codebook"] == codebook]),
            "accuracy": accuracy(cell),
            "predicted_yes_rate": sum(row["predicted_label"] == "yes" for row in cell) / len(cell),
        }
        for semantic in ("same", "different") for codebook in protocol["codebooks"]
    }
    summary = {
        "phase": 1513,
        "campaign": "C088",
        "global_accuracy": accuracy(rows),
        "global_balanced_accuracy": balanced_accuracy(rows),
        "partition": {partition: {"count": len(cell := [row for row in rows if row["partition"] == partition]), "accuracy": accuracy(cell), "balanced_accuracy": balanced_accuracy(cell)} for partition in protocol["partitions"]},
        "surface": {surface: {"accuracy": accuracy(cell := [row for row in rows if row["surface"] == surface]), "balanced_accuracy": balanced_accuracy(cell)} for surface in protocol["surfaces"]},
        "codebook": {codebook: {"accuracy": accuracy(cell := [row for row in rows if row["codebook"] == codebook]), "balanced_accuracy": balanced_accuracy(cell)} for codebook in protocol["codebooks"]},
        "semantic": {semantic: accuracy([row for row in rows if row["semantic_label"] == semantic]) for semantic in ("same", "different")},
        "truth_code": truth_code,
        "stratum_counts": dict(strata),
        "stratum_partition_counts": {stratum: {partition: sum(row["stratum"] == stratum and row["partition"] == partition for row in stratified) for partition in protocol["partitions"]} for stratum in ("success", "mixed", "failed")},
        "error_count": sum(not row["correct"] for row in rows),
        "runtime": runtime,
        "single_authoritative_forward": True,
    }
    raw = OUT / "raw/all_role_field.float16.npy"
    index_path = OUT / "raw/all_role_field_index.jsonl"
    strata_path = OUT / "material/stratified_composition_sets.jsonl"
    core.write_rows(index_path, rows)
    core.write_rows(strata_path, stratified)
    arr = np.load(raw, mmap_mode="r")
    checks = {
        "count": len(rows) == 1984,
        "shape": list(arr.shape) == [1984, 37, 4, 2560],
        "dtype": arr.dtype == np.float16,
        "finite": runtime["finite_during_capture"] and all(math.isfinite(value) for row in rows for value in row["scores"]),
        "repeat": runtime["numeric_repeat_max_abs_diff"] <= 1e-6,
        "strata": len(stratified) == 248 and sum(strata.values()) == 248,
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
        "single_forward": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    summary["files"] = {
        "field": {"bytes": raw.stat().st_size, "sha256": core.sha(raw)},
        "index": {"sha256": core.sha(index_path)},
        "strata": {"sha256": core.sha(strata_path)},
    }
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(OUT / "analysis/unified_behavior_and_capture_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1513, "campaign": "C088", "status": "authoritative_behavior_and_field_capture_complete", "stratum_counts": dict(strata), "authorization": "run_phase1514_c088_factorial_field_atlas"})
    print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
