#!/usr/bin/env python3
"""Phase467 failure audit for Phase466 template-factor behavior."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase465_template_factor_protocol" / "phase465_template_factor_samples.jsonl"
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase466_glm4_template_factor_behavior" / "phase466_glm4_template_factor_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase467_template_factor_failure_audit"
OUT_PATH = OUT_DIR / "phase467_template_factor_failure_audit.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def enrich(gens: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = {sample["sample_id"]: sample for sample in samples}
    out = []
    for row in gens:
        sample = meta[row["sample_id"]]
        nodes = sample["role_nodes"]
        item = dict(row)
        item.update({
            "expected_label": sample["canonical_answer"],
            "target_position": nodes["target_position"],
            "query_position": nodes["query_position"],
            "query_matches_target": nodes["query_position"] == nodes["target_position"],
            "label_role": f"{sample['canonical_answer']}/{sample['pair_role']}",
            "pair_index_mod4": sample["pair_index"] % 4,
        })
        out.append(item)
    return out


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


def transform_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_transform = summarize(rows, ["transform"])
    rates = {row["transform"]: row["semantic_rate"] for row in by_transform}
    return {
        "rates": rates,
        "range": max(rates.values()) - min(rates.values()) if rates else 0.0,
        "best": max(rates, key=rates.get) if rates else None,
        "worst": min(rates, key=rates.get) if rates else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = enrich(load_jsonl(GEN_PATH), load_jsonl(SAMPLES_PATH))
    out = {
        "schema_version": "phase467_template_factor_failure_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "overall": summarize(rows, []),
        "by_transform": summarize(rows, ["transform"]),
        "by_transform_label": summarize(rows, ["transform", "expected_label"]),
        "by_transform_label_role": summarize(rows, ["transform", "label_role"]),
        "by_transform_target_query": summarize(rows, ["transform", "target_position", "query_position"]),
        "by_label_role": summarize(rows, ["label_role"]),
        "by_mod4": summarize(rows, ["pair_index_mod4"]),
        "template_range": transform_range(rows),
        "interpretation": {
            "physical_trace_authorized": False,
            "main_question": "Whether template factors perturb label mapping, evidence parsing, or late A/B competition.",
            "next_step": "If one stable success and one stable failure template are confirmed, run a small diagnostic physical precheck only on those two frozen templates.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
