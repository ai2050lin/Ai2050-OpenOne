#!/usr/bin/env python3
"""Summarize Phase443 behavior qualification results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase443_behavior_qualification"
OUT_PATH = OUT_DIR / "phase443_behavior_aggregate_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def main() -> None:
    models = {}
    stopped_at = None
    for model in MODELS:
        path = OUT_DIR / f"phase443_{model}_summary.json"
        if not path.exists():
            models[model] = {"status": "not_run"}
            if stopped_at is None:
                stopped_at = model
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        models[model] = {
            "status": data["status"],
            "selected_tasks": data["selected_tasks"],
            "final_by_ability": data["final_by_ability"],
        }
        if data["status"] != "pass" and stopped_at is None:
            stopped_at = model

    out = {
        "schema_version": "phase443_behavior_aggregate_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stop_after_failure" if any(item["status"] == "fail" for item in models.values()) else "pass_or_incomplete",
        "stop_rule": "do_not_continue_after_model_or_ability_surface_orbit_failure",
        "stopped_at": stopped_at,
        "models": models,
        "next_authorized": "analysis_only_no_physical_trace",
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
