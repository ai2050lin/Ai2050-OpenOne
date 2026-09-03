#!/usr/bin/env python3
"""Phase1537: behavior-only Qwen3 qualification for the frozen C091 contract."""
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
CONTRACT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
OUT = RESULT / "phase1537_c091_behavior_only_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16


def make_fixed_batch(rows: list[dict], pad: int, device, fixed_length: int):
    ids = torch.full((len(rows), fixed_length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        length = int(values.numel())
        if length > fixed_length:
            raise RuntimeError((row["case_id"], length, fixed_length))
        ids[i, :length] = values
        mask[i, :length] = 1
        lengths.append(length)
    position_ids = mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask == 0, 0)
    return ids, mask, position_ids, lengths


@torch.inference_mode()
def run_behavior_batch(model, rows: list[dict], pad: int, device, fixed_length: int) -> np.ndarray:
    ids, mask, position_ids, lengths = make_fixed_batch(rows, pad, device, fixed_length)
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    values = np.empty((len(rows), 2), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, candidate in enumerate(row["candidate_ids"]):
            if len(candidate) != 1:
                raise RuntimeError((row["case_id"], candidate))
            values[i, j] = float(out.logits[i, lengths[i] - 1, candidate[0]].float().cpu())
    return values


def accuracy(rows: list[dict]) -> float:
    return sum(row["correct"] for row in rows) / len(rows)


def recall(rows: list[dict], label: str) -> float:
    subset = [row for row in rows if row["gold_label"] == label]
    return sum(row["correct"] for row in subset) / len(subset)


def balanced_accuracy(rows: list[dict]) -> float:
    return (recall(rows, "是") + recall(rows, "否")) / 2


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1537 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1537_c091_behavior_only_qualification" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1536 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_pairs.jsonl")
    lookup = {row["case_id"]: row for row in compiled}
    grouped = []
    for pair in pairs:
        rows = [row for row in compiled if row["pair_id"] == pair["pair_id"]]
        rows.sort(key=lambda row: (protocol["surfaces"].index(row["surface"]), protocol["families"].index(row["query_family"])))
        if len(rows) != 6:
            raise RuntimeError((pair["pair_id"], len(rows)))
        grouped.append(rows)
    scores = {}
    repeats = []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        fixed_length = int(protocol["execution"]["fixed_global_sequence_length"])
        for batch_index, rows in enumerate(grouped):
            values = run_behavior_batch(model, rows, pad, device, fixed_length)
            for i, row in enumerate(rows):
                scores[row["case_id"]] = values[i].tolist()
            if batch_index < 3:
                repeats.append((rows, values))
            if (batch_index + 1) % 15 == 0:
                print(f"[phase1537] completed {batch_index + 1}/{len(grouped)} pair batches", flush=True)
        repeat_logit_max_abs = 0.0
        for rows, values in repeats:
            again = run_behavior_batch(model, rows, pad, device, fixed_length)
            repeat_logit_max_abs = max(repeat_logit_max_abs, float(np.max(np.abs(values - again))))
    finally:
        if model is not None:
            release_bf16(model)

    index = []
    for row in compiled:
        values = scores[row["case_id"]]
        prediction = int(values[1] > values[0])
        label_logits = {row["candidates"][i]: float(values[i]) for i in range(2)}
        predicted_label = row["candidates"][prediction]
        index.append({
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
            "gold_label": row["gold_label"],
            "candidates": row["candidates"],
            "candidate_logits": values,
            "label_logits": label_logits,
            "yes_no_margin": label_logits["是"] - label_logits["否"],
            "predicted_label": predicted_label,
            "correct": predicted_label == row["gold_label"],
        })
    three_way = []
    for pair in pairs:
        for surface in protocol["surfaces"]:
            rows = [row for row in index if row["pair_id"] == pair["pair_id"] and row["surface"] == surface]
            winner = max(rows, key=lambda row: row["yes_no_margin"])["query_family"]
            three_way.append({
                "pair_id": pair["pair_id"],
                "pair_family": pair["family"],
                "partition": pair["partition"],
                "concreteness": pair["concreteness"],
                "surface": surface,
                "predicted_family": winner,
                "correct": winner == pair["family"],
            })
    family_summary = {}
    preview_qualified = []
    gate = protocol["behavior_gate"]
    for family in protocol["families"]:
        query_rows = [row for row in index if row["query_family"] == family]
        discovery = [row for row in query_rows if row["partition"] == "response_discovery"]
        surface_ba = {
            surface: balanced_accuracy([row for row in discovery if row["surface"] == surface])
            for surface in protocol["surfaces"]
        }
        family_three_way = [row for row in three_way if row["pair_family"] == family and row["partition"] == "response_discovery"]
        three_way_accuracy = sum(row["correct"] for row in family_three_way) / len(family_three_way)
        metrics = {
            "all_accuracy": accuracy(query_rows),
            "all_balanced_accuracy": balanced_accuracy(query_rows),
            "discovery_accuracy": accuracy(discovery),
            "discovery_balanced_accuracy": balanced_accuracy(discovery),
            "discovery_true_recall": recall(discovery, "是"),
            "discovery_false_recall": recall(discovery, "否"),
            "discovery_surface_balanced_accuracy": surface_ba,
            "discovery_three_way_pair_selection_accuracy": three_way_accuracy,
        }
        qualifies = (
            metrics["discovery_balanced_accuracy"] >= gate["discovery_query_family_balanced_accuracy"]
            and all(value >= gate["discovery_each_surface_balanced_accuracy"] for value in surface_ba.values())
            and metrics["discovery_true_recall"] >= gate["discovery_true_recall"]
            and metrics["discovery_false_recall"] >= gate["discovery_false_recall"]
            and three_way_accuracy >= gate["discovery_three_way_pair_selection_accuracy"]
        )
        metrics["preview_behavior_qualified"] = qualifies
        family_summary[family] = metrics
        if qualifies:
            preview_qualified.append(family)
    index_path = OUT / "raw/behavior_logits.jsonl"
    three_way_path = OUT / "analysis/three_way_pair_selection.jsonl"
    core.write_rows(index_path, index)
    core.write_rows(three_way_path, three_way)
    checks = {
        "coverage": len(index) == 540 and len(scores) == 540,
        "three_way_coverage": len(three_way) == 180,
        "finite": all(np.isfinite(row["candidate_logits"]).all() for row in index),
        "repeat_logits": repeat_logit_max_abs <= protocol["numeric_gate_before_hidden_use"]["repeat_logit_max_abs"],
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "hidden_not_requested": True,
        "family_metrics": set(family_summary) == set(protocol["families"]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1537,
        "campaign": "C091",
        "status": "behavior_only_qualification_run_complete",
        "global": {
            "accuracy": accuracy(index),
            "balanced_accuracy": balanced_accuracy(index),
            "three_way_pair_selection_accuracy": sum(row["correct"] for row in three_way) / len(three_way),
        },
        "family": family_summary,
        "preview_behavior_qualified_families": preview_qualified,
        "strata": {
            "|".join(map(str, key)): value
            for key, value in Counter((row["partition"], row["surface"], row["gold_label"], row["correct"]) for row in index).items()
        },
        "repeat_logit_max_abs": repeat_logit_max_abs,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "files": {
            "behavior_logits": {"sha256": core.sha(index_path), "rows": len(index)},
            "three_way": {"sha256": core.sha(three_way_path), "rows": len(three_way)},
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/behavior_summary.json", report)
    core.save(OUT / "analysis/final.json", {
        "phase": 1537,
        "campaign": "C091",
        "status": report["status"],
        "preview_behavior_qualified_families": preview_qualified,
        "authorization": "run_phase1538_c091_behavior_gate_adjudication",
    })
    print(json.dumps({key: value for key, value in report.items() if key not in ("runtime", "strata")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
