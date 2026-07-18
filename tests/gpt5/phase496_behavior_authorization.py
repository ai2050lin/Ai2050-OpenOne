#!/usr/bin/env python3
"""Aggregate Phase495 without loading a model or reading physical splits."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "tests" / "gpt5" / "result" / "phase495_cross_family_behavior_gate"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase496_behavior_authorization"
MODELS = ("qwen3", "glm4", "deepseek7b")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def concise(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": summary["model"],
        "row_count": summary["row_count"],
        "fit_families_behavior_pass": summary["gates"]["fit_families_behavior_pass"],
        "passed_unseen_families": summary["gates"]["passed_unseen_families"],
        "all_unseen_families_behavior_pass": summary["gates"]["all_unseen_families_behavior_pass"],
        "unseen_native_intersection": summary["overall"]["unseen_native_intersection"],
        "families": {
            family: {
                "native_intersection": payload["native_intersection"],
                "paired_world_all_correct": payload["paired_world_all_correct"],
                "behavior_gate_pass": payload["behavior_gate_pass"],
            }
            for family, payload in summary["families"].items()
        },
    }


def main() -> None:
    summaries = {}
    for model in MODELS:
        path = IN_DIR / f"phase495_{model}_summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing required Phase495 result: {path}")
        summary = load_json(path)
        if summary["model"] != model or not summary["full_frozen_split"] or not summary["cuda_used"]:
            raise RuntimeError(f"Invalid full CUDA behavior result for {model}")
        summaries[model] = summary

    physical_models = [
        model for model in MODELS
        if summaries[model]["authorization"]["formation_fit_authorized"]
        and summaries[model]["authorization"]["physical_prediction_families"]
    ]
    cross_family_candidates = [
        model for model in MODELS
        if summaries[model]["authorization"]["cross_family_physical_candidate"]
    ]
    authorization = {
        "schema_version": "phase496_behavior_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_authorization_frozen",
        "models_in_required_execution_order": list(MODELS),
        "physical_models_in_required_execution_order": physical_models,
        "cross_family_candidate_models": cross_family_candidates,
        "models": {model: concise(summary) for model, summary in summaries.items()},
        "authorization": {
            "open_formation_fit_and_unseen_prediction": bool(physical_models),
            "at_least_two_behavior_cross_family_candidates": len(cross_family_candidates) >= 2,
            "sealed_read": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
        },
        "stopping_rule": "Only behavior-qualified families may enter physical collection; failures remain in the denominator.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "phase496_open_physical_authorization.json"
    path.write_text(json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
