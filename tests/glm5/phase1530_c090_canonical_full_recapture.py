#!/usr/bin/env python3
"""Phase1530: canonical right-padded full recapture of the frozen C089 materials."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
PARENT = RESULT / "phase1529_c090_right_padded_group_calibration"
OUT = RESULT / "phase1530_c090_canonical_full_recapture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1529_c090_right_padded_group_calibration import run_batch

CELLS = ("aa", "ab", "ba", "bb")
SURFACES = ("a_question", "b_question")


def accuracy(rows):
    return sum(row["correct"] for row in rows) / len(rows)


def balanced_accuracy(rows):
    return sum(accuracy([row for row in rows if row["gold_label"] == label]) for label in ("yes", "no")) / 2


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1530 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1530_c090_canonical_full_recapture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1529 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(CONTRACT / "material/relation_composition_sets.jsonl")
    case_lookup = {row["case_id"]: row for row in compiled}
    tuple_lookup = {(row["set_id"], row["surface"], row["cell"]): row for row in compiled}
    row_index = {row["case_id"]: i for i, row in enumerate(compiled)}
    field_path = OUT / "raw/canonical_all_role_field.float16.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(360, 37, 4, 2560))
    scores, repeat_blocks, model = {}, [], None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        batch_counter = 0
        for group in groups:
            for surface in SURFACES:
                rows = [tuple_lookup[(group["set_id"], surface, cell)] for cell in CELLS]
                pooled, values = run_batch(model, rows, pad, device)
                for i, row in enumerate(rows):
                    field[row_index[row["case_id"]]] = pooled[i].astype(np.float16)
                    scores[row["case_id"]] = values[i].tolist()
                if batch_counter < 3:
                    repeat_blocks.append((pooled, values, rows))
                batch_counter += 1
        field.flush()
        repeat_hidden, repeat_logits = 0.0, 0.0
        for pooled, values, rows in repeat_blocks:
            again, again_scores = run_batch(model, rows, pad, device)
            repeat_hidden = max(repeat_hidden, float(np.max(np.abs(pooled - again))))
            repeat_logits = max(repeat_logits, float(np.max(np.abs(values - again_scores))))
    finally:
        if model is not None:
            release_bf16(model)
    del field
    field = np.load(field_path, mmap_mode="r")
    index = []
    for i, row in enumerate(compiled):
        values = scores[row["case_id"]]
        prediction = int(values[1] > values[0])
        index.append({
            "row_index": i, "case_id": row["case_id"], "set_id": row["set_id"], "family": row["family"],
            "partition": row["partition"], "surface": row["surface"], "cell": row["cell"],
            "source": row["source"], "target": row["target"], "truth": row["truth"], "truth_sign": row["truth_sign"],
            "gold_label": row["gold_label"], "candidates": row["candidates"], "gold_position": row["gold_position"],
            "scores": values, "prediction": prediction, "predicted_label": row["candidates"][prediction],
            "correct": prediction == row["gold_position"], "role_positions": row["role_positions"],
        })
    stratified = []
    for group in groups:
        cell = [row for row in index if row["set_id"] == group["set_id"]]
        correct = sum(row["correct"] for row in cell)
        stratified.append({"set_id": group["set_id"], "family": group["family"], "partition": group["partition"], "correct_count": correct, "case_count": 8, "stratum": "success" if correct == 8 else "failed" if correct == 0 else "mixed"})
    family_summary, qualified = {}, []
    for family in protocol["families"]:
        family_rows = [row for row in index if row["family"] == family]
        discovery = [row for row in family_rows if row["partition"] == "response_discovery"]
        surface_accuracy = {surface: accuracy([row for row in discovery if row["surface"] == surface]) for surface in SURFACES}
        qualifies = balanced_accuracy(discovery) >= protocol["behavior_qualification"]["discovery_family_balanced_accuracy"] and all(value >= protocol["behavior_qualification"]["discovery_each_surface_accuracy"] for value in surface_accuracy.values())
        family_summary[family] = {
            "accuracy": accuracy(family_rows), "balanced_accuracy": balanced_accuracy(family_rows),
            "partition": {partition: {"accuracy": accuracy(cell := [row for row in family_rows if row["partition"] == partition]), "balanced_accuracy": balanced_accuracy(cell)} for partition in protocol["partitions"]},
            "discovery_surface_accuracy": surface_accuracy, "behavior_qualified": qualifies,
        }
        if qualifies:
            qualified.append(family)
    source_role = protocol["roles"].index("source_word")
    causal_max = 0.0
    for group in groups:
        for surface in SURFACES:
            rows = {row["cell"]: row for row in index if row["set_id"] == group["set_id"] and row["surface"] == surface}
            for left, right in (("aa", "ab"), ("bb", "ba")):
                x = np.asarray(field[rows[left]["row_index"], :, source_role], dtype=np.float32)
                y = np.asarray(field[rows[right]["row_index"], :, source_role], dtype=np.float32)
                causal_max = max(causal_max, float(np.max(np.abs(x - y))))
    index_path = OUT / "raw/canonical_all_role_field_index.jsonl"
    strata_path = OUT / "material/canonical_stratified_relation_sets.jsonl"
    core.write_rows(index_path, index)
    core.write_rows(strata_path, stratified)
    summary = {
        "phase": 1530, "campaign": "C090", "engine": "right-padded same-set same-surface quartet batches",
        "global_accuracy": accuracy(index), "global_balanced_accuracy": balanced_accuracy(index),
        "family": family_summary, "behavior_qualified_families": qualified,
        "partition": {partition: {"accuracy": accuracy(cell := [row for row in index if row["partition"] == partition]), "balanced_accuracy": balanced_accuracy(cell)} for partition in protocol["partitions"]},
        "surface": {surface: {"accuracy": accuracy(cell := [row for row in index if row["surface"] == surface]), "balanced_accuracy": balanced_accuracy(cell)} for surface in SURFACES},
        "stratum_counts": dict(Counter(row["stratum"] for row in stratified)),
        "repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits,
        "source_causal_prefix_max_abs": causal_max,
        "runtime": {"placement": placement, "quantization": quant},
    }
    checks = {
        "shape": list(field.shape) == [360, 37, 4, 2560], "finite": bool(np.isfinite(np.asarray(field)).all()),
        "coverage": len(index) == 360 and len(scores) == 360, "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "causal_prefix": causal_max == 0.0, "strata": len(stratified) == 45,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    summary["files"] = {
        "field": {"sha256": core.sha(field_path), "bytes": field_path.stat().st_size},
        "index": {"sha256": core.sha(index_path)}, "strata": {"sha256": core.sha(strata_path)},
    }
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(OUT / "analysis/canonical_behavior_and_capture_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1530, "campaign": "C090", "status": "canonical_behavior_and_full_state_recapture_complete", "behavior_qualified_families": qualified, "authorization": "run_phase1531_c090_canonical_truth_contrast_atlas"})
    print(json.dumps({key: value for key, value in summary.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
