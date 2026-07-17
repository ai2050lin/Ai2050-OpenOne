#!/usr/bin/env python3
"""Aggregate Phase446 anti-shortcut static and behavior results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract"
BEHAVIOR_DIR = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_behavior"
OUT_PATH = BEHAVIOR_DIR / "phase446_antishortcut_aggregate_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def main() -> None:
    static_audit = json.loads((STATIC_DIR / "phase446_static_audit_report.json").read_text(encoding="utf-8"))
    models = {}
    candidates = []
    for model in MODELS:
        path = BEHAVIOR_DIR / f"phase446_{model}_summary.json"
        if not path.exists():
            models[model] = {"status": "not_run"}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        models[model] = {"status": data["status"], "by_task": data["by_task"]}
        for task, item in data["by_task"].items():
            if item["qualified_for_minimal_physical"]:
                candidates.append({"model": model, "task": task})
    out = {
        "schema_version": "phase446_antishortcut_aggregate_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "static_status": static_audit["status"],
        "status": "physical_candidate_found" if candidates else "no_physical_candidate",
        "physical_candidates": candidates,
        "models": models,
        "physical_collection_performed": False,
        "next_authorized": "redesign_or_interface_analysis_only" if not candidates else "minimal_physical_window_freeze",
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
