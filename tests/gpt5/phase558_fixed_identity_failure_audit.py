#!/usr/bin/env python3
"""Audit Phase558 behavior failures without changing the frozen gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase558_fixed_identity_color"
SUMMARY_PATH = OUT_DIR / "phase558_behavior_summary.json"
OUTPUT_PATH = OUT_DIR / "phase558_failure_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalized_prefix(text: str) -> str:
    return " ".join(text.strip().split()).casefold()[:80]


def analyze_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(OUT_DIR / f"phase558_{model}_behavior_rows.jsonl")
    failures = [row for row in rows if not row["semantic_correct"]]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        by_split[row["split"]].append(row)
    return {
        "model": model,
        "row_count": len(rows),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(rows),
        "failure_event_counts": dict(Counter(row["semantic_event"] for row in failures)),
        "failure_split_counts": dict(Counter(row["split"] for row in failures)),
        "failure_surface_counts": dict(Counter(str(row["surface_id"]) for row in failures)),
        "failure_fact_order_counts": dict(Counter(str(row["fact_order"]) for row in failures)),
        "failure_binding_counts": dict(Counter(str(row["binding"]) for row in failures)),
        "failure_query_object_counts": dict(Counter(str(row["query_object_index"]) for row in failures)),
        "failure_target_color_counts": dict(Counter(row["target"] for row in failures)),
        "generated_prefix_counts": dict(Counter(normalized_prefix(row["generated_text"]) for row in failures).most_common(20)),
        "all_failures_are_field_name_color": bool(
            failures and all(normalized_prefix(row["generated_text"]) == "color" for row in failures)
        ),
        "registered_distractor_first_count": sum(
            row["semantic_event"] == "registered_distractor" for row in failures
        ),
        "split_details": {
            split: {
                "failure_count": len(group),
                "unrecoverable_count": sum(
                    row["semantic_event"] == "unrecoverable" for row in group
                ),
                "registered_distractor_count": sum(
                    row["semantic_event"] == "registered_distractor" for row in group
                ),
            }
            for split, group in sorted(by_split.items())
        },
    }


def audit() -> dict[str, Any]:
    summary = read_json(SUMMARY_PATH)
    reports = [analyze_model(model) for model in MODELS]
    payload = {
        "schema_version": "phase558_failure_audit.v1",
        "phase_id": "Phase558",
        "created_at": now(),
        "authorized_models": summary["authorized_models"],
        "model_reports": reports,
        "interpretation": {
            "qwen3": (
                "Near-perfect binding behavior, but the frozen discovery cell confidence gate failed; "
                "all nine open failures returned the field name 'color' on the table surface."
            ),
            "glm4": (
                "High row accuracy but unstable world closure; failures often emitted the nontarget "
                "color before the target, so the first semantic event followed fact/order bias."
            ),
            "deepseek7b": (
                "Broad behavior and interface failure with high unrecoverable rates; no internal claim."
            ),
        },
        "gate_changed_after_result": False,
        "internal_collection_authorized": False,
        "sealed_split_read": False,
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    audit()
