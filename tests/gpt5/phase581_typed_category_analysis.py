#!/usr/bin/env python3
"""Aggregate Phase581 typed-category behavior decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase581_typed_category_behavior as behavior
import phase581_typed_category_protocol as protocol


OUTPUT = protocol.OUT_DIR / "phase581_typed_category_behavior_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    model_results = {}
    authorized = {}
    for model in protocol.MODELS:
        paths = behavior.paths(model)
        summary = read_json(paths["summary"])
        registry = read_json(paths["registry"])
        if summary["rows_sha256"] != sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase581 {model} rows hash drift")
        if summary["sealed_split_read"] or registry["sealed_split_read"]:
            raise RuntimeError(f"Phase581 {model} sealed access")
        relations = list(summary["internal_trace_authorized_relations"])
        if relations != list(registry["internal_trace_authorized_relations"]):
            raise RuntimeError(f"Phase581 {model} registry drift")
        if relations:
            authorized[model] = relations
        model_results[model] = {
            "internal_trace_authorized_relations": relations,
            "unit_metrics": summary["unit_metrics"],
            "qualified_objects_by_split_relation": registry[
                "qualified_objects_by_split_relation"
            ],
            "rows_sha256": summary["rows_sha256"],
            "summary_sha256": sha256_file(paths["summary"]),
        }
    payload = {
        "schema_version": "phase581_typed_category_behavior_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_behavior_only",
        "model_results": model_results,
        "internal_trace_authorized_model_relations": authorized,
        "causal_intervention_authorized": False,
        "sealed_validation_authorized": False,
        "sealed_split_read": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "internal_trace_authorized_model_relations": authorized,
                "sealed_validation_authorized": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
