#!/usr/bin/env python3
"""Audit and retire the Phase386 eight-token behavior pilot as a whole cohort."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
BEHAVIOR = PHASE_ROOT / "behavior"
PILOT = PHASE_ROOT / "behavior_pilot_8_tokens"
MODELS = ("qwen3", "glm4", "deepseek7b")
PILOT_MAX_NEW_TOKENS = 8
REPAIR_MAX_NEW_TOKENS = 24


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    model_rows = []
    all_files_present = True
    for model in MODELS:
        path = BEHAVIOR / "private/models" / model / "phase386_behavior_rows.jsonl"
        all_files_present = all_files_present and path.is_file()
        rows = read_jsonl(path)
        failures = [row for row in rows if not row["strict_behavior_correct"]]
        capped_failures = [
            row
            for row in failures
            if row["generated_token_count"] == PILOT_MAX_NEW_TOKENS
        ]
        capped = Counter(
            row["mechanism_id_private"] for row in capped_failures
        )
        model_rows.append(
            {
                "model": model,
                "case_count": len(rows),
                "failure_count": len(failures),
                "capped_failure_count": len(capped_failures),
                "capped_failure_fraction": len(capped_failures)
                / max(len(failures), 1),
                "capped_failures_by_mechanism": dict(capped),
            }
        )
    deepseek = next(row for row in model_rows if row["model"] == "deepseek7b")
    output_budget_failure = (
        all_files_present
        and deepseek["capped_failure_count"] >= 300
        and deepseek["capped_failure_fraction"] >= 0.75
    )
    if not output_budget_failure:
        raise RuntimeError(f"Eight-token pilot retirement not authorized: {model_rows}")
    if PILOT.exists():
        shutil.rmtree(PILOT)
    shutil.copytree(BEHAVIOR, PILOT)
    amendment = {
        "schema_version": "60.1.1",
        "phase_id": "Phase386-BehaviorBudgetAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_max_new_tokens": PILOT_MAX_NEW_TOKENS,
        "model_rows": model_rows,
        "output_budget_failure": True,
        "pilot_cohort_retired": True,
        "failed_cases_selectively_replaced": False,
        "all_models_must_rerun": True,
        "replacement_max_new_tokens": REPAIR_MAX_NEW_TOKENS,
        "internal_collection_started": False,
        "physical_holdout_opened": False,
        "claim_boundary": {
            "pilot_failure_is_model_language_failure": False,
            "replacement_budget_may_be_retuned_again": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_behavior_budget_amendment.json", amendment)
    print(json.dumps(amendment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
