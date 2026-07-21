#!/usr/bin/env python3
"""Aggregate the three prospective Phase589 model decisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import phase589_food_attribute_observer as observer
import phase589_food_attribute_protocol as protocol


OUTPUT = protocol.OUT_DIR / "phase589_food_attribute_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    model_results = {}
    authorized = []
    for model in protocol.MODELS:
        paths = observer.paths(model)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary["rows_sha256"] != protocol.sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase589 {model} rows drift")
        if summary["sealed_split_read"]:
            raise RuntimeError(f"Phase589 {model} sealed access")
        if summary["open_hidden_capture_authorized"]:
            authorized.append(model)
        model_results[model] = {
            "split_metrics": summary["split_metrics"],
            "open_hidden_capture_authorized": summary[
                "open_hidden_capture_authorized"
            ],
            "rows_sha256": summary["rows_sha256"],
        }
    payload = {
        "schema_version": "phase589_food_attribute_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_prospective_food_attribute_observer",
        "model_results": model_results,
        "open_hidden_capture_authorized_models": authorized,
        "cross_model_food_attribute_observer_pass": len(authorized) == len(protocol.MODELS),
        "natural_generation_qualified_models": [],
        "causal_intervention_authorized": False,
        "sealed_validation_authorized": False,
        "sealed_split_read": False,
    }
    protocol.write_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "open_hidden_capture_authorized_models": authorized,
                "cross_model_food_attribute_observer_pass": payload[
                    "cross_model_food_attribute_observer_pass"
                ],
                "causal_intervention_authorized": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
