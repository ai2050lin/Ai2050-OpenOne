#!/usr/bin/env python3
"""Phase457 failure audit for Phase456 independent GLM4 replicate.

No model run, no CUDA, no physical trace.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase455_independent_core_protocol" / "phase455_independent_core_samples.jsonl"
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase456_glm4_independent_core_behavior" / "phase456_glm4_independent_core_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase457_independent_failure_audit"
OUT_PATH = OUT_DIR / "phase457_independent_failure_audit.json"

CORE_TRANSFORMS = {
    "core_catalog_frame",
    "core_numbered_records",
    "core_evidence_claim",
    "core_question_sync",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    outputs: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(row[field] for field in fields)
        buckets[key][row["classification"]] += 1
        outputs[key][row["normalized_generated"] or "<empty>"] += 1
    out = []
    for key, counts in sorted(buckets.items()):
        n = sum(counts.values())
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update({
            "n": n,
            "semantic": counts["semantic"],
            "wrong": counts["wrong"],
            "other": counts["other"],
            "semantic_rate": counts["semantic"] / n if n else 0.0,
            "output_distribution": dict(outputs[key]),
        })
        out.append(item)
    return out


def attach_sample_metadata(gens: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = {sample["sample_id"]: sample for sample in samples}
    out = []
    for row in gens:
        sample = meta[row["sample_id"]]
        nodes = sample["role_nodes"]
        item = dict(row)
        item.update({
            "track": "core" if row["transform"] in CORE_TRANSFORMS else "stress",
            "target_side": "right" if nodes["target_item"] == nodes["right_item"] else "left",
            "query_signal_side": "right" if nodes["query_signal"] == nodes["right_signal"] else "left",
            "ledger": nodes["ledger"],
            "expected_label": sample["canonical_answer"],
            "pair_index_mod_4": sample["pair_index"] % 4,
        })
        out.append(item)
    return out


def pair_failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core_rows = [row for row in rows if row["track"] == "core"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in core_rows:
        grouped[row["source_pair_id"]].append(row)
    all_ok = partial = none = 0
    worst = []
    for pair_id, pair_rows in grouped.items():
        ok = sum(row["classification"] == "semantic" for row in pair_rows)
        if ok == len(pair_rows):
            all_ok += 1
        elif ok == 0:
            none += 1
        else:
            partial += 1
        if ok < len(pair_rows):
            sample = pair_rows[0]
            worst.append({
                "source_pair_id": pair_id,
                "ok": ok,
                "n": len(pair_rows),
                "pair_index": sample["pair_index"],
                "target_side": sample["target_side"],
                "ledger": sample["ledger"],
                "outputs": dict(Counter(row["normalized_generated"] for row in pair_rows)),
            })
    worst.sort(key=lambda item: (item["ok"], item["pair_index"]))
    return {
        "n_pairs": len(grouped),
        "all_core_outputs_correct_pairs": all_ok,
        "partial_pairs": partial,
        "zero_correct_pairs": none,
        "worst_examples": worst[:20],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = attach_sample_metadata(load_jsonl(GEN_PATH), load_jsonl(SAMPLES_PATH))
    core_rows = [row for row in rows if row["track"] == "core"]
    stress_rows = [row for row in rows if row["track"] == "stress"]
    out = {
        "schema_version": "phase457_independent_failure_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "overall_by_track": summarize(rows, ["track"]),
        "core_by_transform_and_label": summarize(core_rows, ["transform", "expected_label"]),
        "core_by_pair_role": summarize(core_rows, ["pair_role"]),
        "core_by_target_side": summarize(core_rows, ["target_side"]),
        "core_by_query_signal_side": summarize(core_rows, ["query_signal_side"]),
        "core_by_ledger": summarize(core_rows, ["ledger"]),
        "core_by_pair_index_mod_4": summarize(core_rows, ["pair_index_mod_4"]),
        "stress_by_transform_and_label": summarize(stress_rows, ["transform", "expected_label"]),
        "core_pair_failure_summary": pair_failure_summary(rows),
        "interpretation": {
            "s3_core_confirmed": False,
            "dominant_pattern": "glm4 shows strong B-output tendency on the independent generator; A/base cases are the main failure mode.",
            "next_step": "redesign independent generator to reduce lexical/template novelty or test whether relation direction is too indirect before physical tracing.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
