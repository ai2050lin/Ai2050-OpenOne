#!/usr/bin/env python3
"""Phase464 failure audit for Phase463 bridge behavior."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase462_bridge_protocol" / "phase462_bridge_samples.jsonl"
GEN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase463_glm4_bridge_behavior" / "phase463_glm4_bridge_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase464_bridge_failure_audit"
OUT_PATH = OUT_DIR / "phase464_bridge_failure_audit.json"

CORE_TRANSFORMS = {
    "bridge_lexical_frame",
    "bridge_numbered_facts",
    "bridge_evidence_claim",
    "bridge_claim_sync",
}


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
            "track": "core" if row["transform"] in CORE_TRANSFORMS else "stress",
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = enrich(load_jsonl(GEN_PATH), load_jsonl(SAMPLES_PATH))
    core_rows = [row for row in rows if row["track"] == "core"]
    stress_rows = [row for row in rows if row["track"] == "stress"]
    out = {
        "schema_version": "phase464_bridge_failure_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "overall_by_track": summarize(rows, ["track"]),
        "core_by_transform_label": summarize(core_rows, ["transform", "expected_label"]),
        "core_by_label_role": summarize(core_rows, ["label_role"]),
        "core_by_transform_label_role": summarize(core_rows, ["transform", "label_role"]),
        "core_by_target_query": summarize(core_rows, ["target_position", "query_position"]),
        "core_by_mod4": summarize(core_rows, ["pair_index_mod4"]),
        "stress_by_transform_label": summarize(stress_rows, ["transform", "expected_label"]),
        "interpretation": {
            "bridge_restored_s3_core": False,
            "main_failure": "Breaking base/A binding did not restore orbit stability; numbered facts and claim-first templates are especially weak.",
            "important_positive": "bridge_lexical_frame and bridge_semicolon_compact remain high, suggesting a narrow template-conditioned route still exists.",
            "next_step": "factor bridge template effects one at a time instead of changing generator family again.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
