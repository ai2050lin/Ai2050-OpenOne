#!/usr/bin/env python3
"""Phase474 failure audit for Phase473 label-mapping behavior."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase472_label_mapping_reversal_protocol" / "phase472_label_mapping_reversal_samples.jsonl"
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase473_glm4_label_mapping_behavior" / "phase473_glm4_label_mapping_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase474_label_mapping_failure_audit"
OUT_PATH = OUT_DIR / "phase474_label_mapping_failure_audit.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def enrich(gens: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = {sample["sample_id"]: sample for sample in samples}
    out = []
    for row in gens:
        sample = meta[row["sample_id"]]
        true_label = "A" if sample["label_mapping"] == "mu_ab" else "B"
        false_label = "B" if sample["label_mapping"] == "mu_ab" else "A"
        item = dict(row)
        item.update({
            "label_mapping": sample["label_mapping"],
            "expected_label": sample["canonical_answer"],
            "truth_value": sample["truth_value"],
            "true_label_for_mapping": true_label,
            "false_label_for_mapping": false_label,
            "output_is_true_label": row["normalized_generated"] == true_label,
            "output_is_false_label": row["normalized_generated"] == false_label,
            "target_position": sample["role_nodes"]["target_position"],
            "query_position": sample["role_nodes"]["query_position"],
            "label_role": f"{sample['canonical_answer']}/{sample['pair_role']}",
        })
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, items in sorted(buckets.items()):
        counts = Counter(row["classification"] for row in items)
        outputs = Counter(row["normalized_generated"] or "<empty>" for row in items)
        true_label_outputs = sum(1 for row in items if row["output_is_true_label"])
        false_label_outputs = sum(1 for row in items if row["output_is_false_label"])
        n = len(items)
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update({
            "n": n,
            "semantic": counts["semantic"],
            "wrong": counts["wrong"],
            "other": counts["other"],
            "semantic_rate": counts["semantic"] / n if n else 0.0,
            "output_distribution": dict(outputs),
            "true_label_outputs": true_label_outputs,
            "true_label_output_rate": true_label_outputs / n if n else 0.0,
            "false_label_outputs": false_label_outputs,
            "false_label_output_rate": false_label_outputs / n if n else 0.0,
        })
        out.append(item)
    return out


def mapping_flip_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_sample_id"], row["transform"])].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        by_mu = {row["label_mapping"]: row for row in items}
        if set(by_mu) != {"mu_ab", "mu_ba"}:
            continue
        out.append({
            "source_sample_id": key[0],
            "transform": key[1],
            "truth_value": by_mu["mu_ab"]["truth_value"],
            "mu_ab_output": by_mu["mu_ab"]["normalized_generated"],
            "mu_ba_output": by_mu["mu_ba"]["normalized_generated"],
            "mu_ab_classification": by_mu["mu_ab"]["classification"],
            "mu_ba_classification": by_mu["mu_ba"]["classification"],
            "both_correct": by_mu["mu_ab"]["classification"] == "semantic" and by_mu["mu_ba"]["classification"] == "semantic",
            "output_flipped": by_mu["mu_ab"]["normalized_generated"] != by_mu["mu_ba"]["normalized_generated"],
            "both_true_label": by_mu["mu_ab"]["output_is_true_label"] and by_mu["mu_ba"]["output_is_true_label"],
        })
    return out


def summarize_flips(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, items in sorted(buckets.items()):
        n = len(items)
        both_correct = sum(1 for row in items if row["both_correct"])
        output_flipped = sum(1 for row in items if row["output_flipped"])
        both_true_label = sum(1 for row in items if row["both_true_label"])
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update({
            "n": n,
            "both_correct": both_correct,
            "both_correct_rate": both_correct / n if n else 0.0,
            "output_flipped": output_flipped,
            "output_flipped_rate": output_flipped / n if n else 0.0,
            "both_true_label": both_true_label,
            "both_true_label_rate": both_true_label / n if n else 0.0,
        })
        out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = enrich(load_jsonl(GEN_PATH), load_jsonl(SAMPLES_PATH))
    flips = mapping_flip_table(rows)
    out = {
        "schema_version": "phase474_label_mapping_failure_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "overall_by_transform": summarize(rows, ["transform"]),
        "by_transform_mapping_truth": summarize(rows, ["transform", "label_mapping", "truth_value"]),
        "by_transform_truth": summarize(rows, ["transform", "truth_value"]),
        "by_transform_mapping_expected": summarize(rows, ["transform", "label_mapping", "expected_label"]),
        "mapping_flip_summary": summarize_flips(flips, ["transform", "truth_value"]),
        "interpretation": {
            "claim_first_is_fixed_ab_label": False,
            "claim_first_main_failure": "Claim-first follows the mapping's true label almost always; false claims collapse.",
            "evidence_first_mapping_reversal_supported": "Partial: evidence-first stays strong under mu_ab and remains mostly functional under mu_ba, but mu_ba false claims are weaker.",
            "physical_trace_authorized": False,
            "next_step": "If continuing, run a small position-resolved scalar precheck on evidence-first only across label mappings to test label-state versus relation-state separation.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
