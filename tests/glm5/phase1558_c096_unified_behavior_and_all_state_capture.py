#!/usr/bin/env python3
"""Phase1558: one-load C096 behavior and complete all-state CUDA capture."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
OUT = RESULT / "phase1558_c096_unified_behavior_and_all_state_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1537_c091_behavior_only_qualification import make_fixed_batch

ROLES = ("source_word", "target_word", "relation_anchor", "boundary")


@torch.inference_mode()
def run_batch(model, rows: list[dict], pad: int, device, fixed_length: int):
    ids, mask, position_ids, lengths = make_fixed_batch(rows, pad, device, fixed_length)
    out = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    states = len(out.hidden_states)
    hidden_size = int(model.config.hidden_size)
    pooled = np.empty((len(rows), states, len(ROLES), hidden_size), dtype=np.float32)
    values = np.empty((len(rows), 2), dtype=np.float32)
    for state, hidden in enumerate(out.hidden_states):
        for i, row in enumerate(rows):
            for role_index, role in enumerate(ROLES):
                points = torch.tensor(row["role_positions"][role], dtype=torch.long, device=device)
                pooled[i, state, role_index] = hidden[i, points].float().mean(dim=0).cpu().numpy()
    for i, row in enumerate(rows):
        for j, candidate in enumerate(row["candidate_ids"]):
            values[i, j] = float(out.logits[i, lengths[i] - 1, candidate[0]].float().cpu())
    return pooled, values


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def recall_truth(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["truth"] is truth]
    return sum(row["correct"] for row in subset) / len(subset)


def balanced_accuracy(rows: list[dict]) -> float:
    return 0.5 * (recall_truth(rows, True) + recall_truth(rows, False))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1558 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1558_c096_unified_behavior_and_all_state_capture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1557 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_fresh_pairs.jsonl")
    row_index = {row["case_id"]: i for i, row in enumerate(compiled)}
    grouped = []
    for pair in pairs:
        rows = [row for row in compiled if row["pair_id"] == pair["pair_id"]]
        rows.sort(key=lambda row: (protocol["surfaces"].index(row["surface"]), protocol["families"].index(row["query_family"])))
        if len(rows) != 6:
            raise RuntimeError((pair["pair_id"], len(rows)))
        grouped.append(rows)

    field_path = OUT / "raw/c096_all_role_field.float16.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(540, 37, 4, 2560))
    scores: dict[str, list[float]] = {}
    repeat_blocks = []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        fixed_length = int(protocol["execution"]["fixed_global_sequence_length"])
        for batch_index, rows in enumerate(grouped):
            pooled, values = run_batch(model, rows, pad, device, fixed_length)
            for i, row in enumerate(rows):
                field[row_index[row["case_id"]]] = pooled[i].astype(np.float16)
                scores[row["case_id"]] = values[i].tolist()
            if batch_index < 3:
                repeat_blocks.append((rows, pooled, values))
            if (batch_index + 1) % 15 == 0:
                print(f"[phase1558] completed {batch_index + 1}/{len(grouped)} pair batches", flush=True)
        field.flush()
        repeat_hidden_max_abs = 0.0
        repeat_logit_max_abs = 0.0
        for rows, pooled, values in repeat_blocks:
            again_hidden, again_logits = run_batch(model, rows, pad, device, fixed_length)
            repeat_hidden_max_abs = max(repeat_hidden_max_abs, float(np.max(np.abs(pooled - again_hidden))))
            repeat_logit_max_abs = max(repeat_logit_max_abs, float(np.max(np.abs(values - again_logits))))
    finally:
        if model is not None:
            release_bf16(model)
    del field
    field = np.load(field_path, mmap_mode="r")

    index = []
    behavior_rows = []
    for i, row in enumerate(compiled):
        values = scores[row["case_id"]]
        prediction_index = int(values[1] > values[0])
        predicted_label = row["candidates"][prediction_index]
        label_logits = {row["candidates"][j]: float(values[j]) for j in range(2)}
        true_label = row["gold_label"] if row["truth"] else next(label for label in row["candidates"] if label != row["gold_label"])
        false_label = next(label for label in row["candidates"] if label != true_label)
        behavior_item = {
            "case_id": row["case_id"],
            "pair_id": row["pair_id"],
            "pair_family": row["pair_family"],
            "query_family": row["query_family"],
            "partition": row["partition"],
            "partition_rank": row["partition_rank"],
            "concreteness": row["concreteness"],
            "surface": row["surface"],
            "source": row["source"],
            "target": row["target"],
            "truth": bool(row["truth"]),
            "gold_label": row["gold_label"],
            "candidates": row["candidates"],
            "candidate_logits": values,
            "label_logits": label_logits,
            "truth_margin": label_logits[true_label] - label_logits[false_label],
            "predicted_label": predicted_label,
            "correct": predicted_label == row["gold_label"],
        }
        behavior_rows.append(behavior_item)
        index.append({
            "row_index": i,
            "case_id": row["case_id"],
            "pair_id": row["pair_id"],
            "pair_family": row["pair_family"],
            "query_family": row["query_family"],
            "partition": row["partition"],
            "partition_rank": row["partition_rank"],
            "concreteness": row["concreteness"],
            "surface": row["surface"],
            "source": row["source"],
            "target": row["target"],
            "truth": bool(row["truth"]),
            "gold_label": row["gold_label"],
            "candidate_logits": values,
            "role_positions": row["role_positions"],
        })

    three_way = []
    for pair in pairs:
        for surface in protocol["surfaces"]:
            rows = [row for row in behavior_rows if row["pair_id"] == pair["pair_id"] and row["surface"] == surface]
            winner = max(rows, key=lambda row: row["truth_margin"])["query_family"]
            three_way.append({"pair_id": pair["pair_id"], "pair_family": pair["family"], "partition": pair["partition"], "concreteness": pair["concreteness"], "surface": surface, "predicted_family": winner, "correct": winner == pair["family"]})

    family_summary = {}
    qualified = []
    thresholds = protocol["behavior_typing"]["thresholds"]
    for family in protocol["families"]:
        family_rows = [row for row in behavior_rows if row["query_family"] == family]
        discovery = [row for row in family_rows if row["partition"] == "response_discovery"]
        surface_ba = {surface: balanced_accuracy([row for row in discovery if row["surface"] == surface]) for surface in protocol["surfaces"]}
        family_three_way = [row for row in three_way if row["pair_family"] == family and row["partition"] == "response_discovery"]
        three_way_accuracy = accuracy(family_three_way)
        metrics = {
            "all_accuracy": accuracy(family_rows),
            "all_balanced_accuracy": balanced_accuracy(family_rows),
            "discovery_accuracy": accuracy(discovery),
            "discovery_balanced_accuracy": balanced_accuracy(discovery),
            "discovery_true_recall": recall_truth(discovery, True),
            "discovery_false_recall": recall_truth(discovery, False),
            "discovery_surface_balanced_accuracy": surface_ba,
            "discovery_three_way_pair_selection_accuracy": three_way_accuracy,
        }
        qualifies = (
            metrics["discovery_balanced_accuracy"] >= thresholds["discovery_balanced_accuracy"]
            and all(value >= thresholds["discovery_each_surface_balanced_accuracy"] for value in surface_ba.values())
            and metrics["discovery_true_recall"] >= thresholds["discovery_true_recall"]
            and metrics["discovery_false_recall"] >= thresholds["discovery_false_recall"]
            and three_way_accuracy >= thresholds["discovery_three_way_accuracy"]
        )
        metrics["behavior_qualified"] = qualifies
        metrics["missingness"] = None if qualifies else "M_BEHAVIOR"
        family_summary[family] = metrics
        if qualifies:
            qualified.append(family)

    postquery_causal_max_abs = 0.0
    for pair in pairs:
        rows = [row for row in index if row["pair_id"] == pair["pair_id"] and row["surface"] == "postquery"]
        base = rows[0]
        for row in rows[1:]:
            for role_index in (0, 1):
                left = np.asarray(field[base["row_index"], :, role_index], dtype=np.float32)
                right = np.asarray(field[row["row_index"], :, role_index], dtype=np.float32)
                postquery_causal_max_abs = max(postquery_causal_max_abs, float(np.max(np.abs(left - right))))
    prequery_anchor_max_abs = 0.0
    for query_family in protocol["families"]:
        rows = [row for row in index if row["surface"] == "prequery" and row["query_family"] == query_family]
        base = rows[0]
        left = np.asarray(field[base["row_index"], :, 2], dtype=np.float32)
        for row in rows[1:]:
            right = np.asarray(field[row["row_index"], :, 2], dtype=np.float32)
            prequery_anchor_max_abs = max(prequery_anchor_max_abs, float(np.max(np.abs(left - right))))
    finite = all(np.isfinite(np.asarray(field[start:start + 30])).all() for start in range(0, len(field), 30))

    index_path = OUT / "raw/c096_all_role_field_index.jsonl"
    behavior_path = OUT / "raw/c096_behavior_logits.jsonl"
    three_way_path = OUT / "analysis/c096_three_way_pair_selection.jsonl"
    core.write_rows(index_path, index)
    core.write_rows(behavior_path, behavior_rows)
    core.write_rows(three_way_path, three_way)
    gate = protocol["numeric_integrity_gate"]
    checks = {
        "shape": list(field.shape) == [540, 37, 4, 2560],
        "finite": finite,
        "coverage": len(index) == len(behavior_rows) == 540 and len(three_way) == 180,
        "repeat_hidden": repeat_hidden_max_abs <= gate["repeat_hidden_max_abs"],
        "repeat_logits": repeat_logit_max_abs <= gate["repeat_logit_max_abs"],
        "postquery_causal_identity": postquery_causal_max_abs <= gate["postquery_source_target_causal_max_abs"],
        "prequery_causal_identity": prequery_anchor_max_abs <= gate["prequery_relation_anchor_causal_max_abs"],
        "bf16_nonquantized": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "behavior_typed": set(family_summary) == set(protocol["families"]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1558,
        "campaign": "C096",
        "status": "unified_behavior_and_all_state_capture_numeric_gate_pass",
        "global_behavior": {"accuracy": accuracy(behavior_rows), "balanced_accuracy": balanced_accuracy(behavior_rows), "three_way_accuracy": accuracy(three_way)},
        "family_behavior": family_summary,
        "behavior_qualified_families": qualified,
        "behavior_strata": {"|".join(map(str, key)): value for key, value in Counter((row["partition"], row["surface"], row["query_family"], row["truth"], row["correct"]) for row in behavior_rows).items()},
        "field_shape": list(field.shape),
        "repeat_hidden_max_abs": repeat_hidden_max_abs,
        "repeat_logit_max_abs": repeat_logit_max_abs,
        "postquery_source_target_causal_max_abs": postquery_causal_max_abs,
        "prequery_relation_anchor_causal_max_abs": prequery_anchor_max_abs,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "files": {
            "field": {"path": str(field_path.relative_to(ROOT)), "sha256": core.sha(field_path), "bytes": field_path.stat().st_size, "shape": [540, 37, 4, 2560]},
            "index": {"path": str(index_path.relative_to(ROOT)), "sha256": core.sha(index_path), "rows": len(index)},
            "behavior": {"path": str(behavior_path.relative_to(ROOT)), "sha256": core.sha(behavior_path), "rows": len(behavior_rows)},
            "three_way": {"path": str(three_way_path.relative_to(ROOT)), "sha256": core.sha(three_way_path), "rows": len(three_way)},
        },
        "claim_boundary": protocol["claim_boundary"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1559_c096_fresh_prediction_atlas_and_adjudication",
    }
    core.save(OUT / "analysis/c096_capture_and_behavior_summary.json", report)
    core.save(OUT / "analysis/final.json", {"phase": 1558, "campaign": "C096", "status": report["status"], "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key not in {"runtime", "behavior_strata"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
