#!/usr/bin/env python3
"""Aggregate Phase587 counterbalanced continuation decisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import phase587_counterbalanced_continuation_observer as observer
import phase587_counterbalanced_continuation_protocol as protocol


OUTPUT = protocol.OUT_DIR / "phase587_counterbalanced_continuation_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    model_results: dict[str, Any] = {}
    authorized: dict[str, list[str]] = {}
    for model in protocol.MODELS:
        paths = observer.paths(model)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
        if summary["rows_sha256"] != protocol.sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase587 {model} rows drift")
        if summary["sealed_split_read"] or registry["sealed_split_read"]:
            raise RuntimeError(f"Phase587 {model} sealed access")
        relations = list(summary["open_hidden_capture_authorized_relations"])
        if relations != list(registry["open_hidden_capture_authorized_relations"]):
            raise RuntimeError(f"Phase587 {model} registry drift")
        if relations:
            authorized[model] = relations
        model_results[model] = {
            "open_hidden_capture_authorized_relations": relations,
            "unit_metrics": summary["unit_metrics"],
            "qualified_objects_by_unit": registry["qualified_objects_by_unit"],
            "rows_sha256": summary["rows_sha256"],
            "summary_sha256": protocol.sha256_file(paths["summary"]),
        }
    payload = {
        "schema_version": "phase587_counterbalanced_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_external_counterbalanced_observer",
        "model_results": model_results,
        "open_hidden_capture_authorized_model_relations": authorized,
        "natural_generation_qualified_models": [],
        "causal_intervention_authorized": False,
        "sealed_validation_authorized": False,
        "sealed_split_read": False,
    }
    protocol.write_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "open_hidden_capture_authorized_model_relations": authorized,
                "causal_intervention_authorized": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
