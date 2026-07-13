#!/usr/bin/env python3
"""Run Phase408 gated stages sequentially with resumable native-crash recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase408_partition_interface_collection import read_json, write_json
from phase408_partition_interface_protocol import MODELS, OUT


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("discovery", "calibration", "behavioral_holdout")
AUDIT = OUT / "phase408_execution_recovery_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_count(model: str, stage: str) -> int:
    path = OUT / "collection" / stage / "private" / model / "rows.jsonl"
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(model: str, stage: str) -> bool:
    path = OUT / "collection" / stage / model / "complete.json"
    return path.is_file() and bool(read_json(path).get("valid"))


def load_audit() -> dict[str, Any]:
    if AUDIT.is_file():
        return read_json(AUDIT)
    return {
        "schema_version": "82.2.0",
        "phase_id": "Phase408-SequentialExecutionRecoveryAudit",
        "created_at": now(),
        "model_execution_order": list(MODELS),
        "parallel_model_execution": False,
        "pre_orchestrator_observations": [
            {
                "stage": "discovery",
                "model": "qwen3",
                "observed_shell_exit_code": 139,
                "persisted_rows_after_exit": 2044,
                "resume_completed": True,
                "source": "operator_observed_before_orchestrator_activation",
            },
            {
                "stage": "discovery",
                "model": "glm4",
                "observed_shell_exit_code": 139,
                "persisted_rows_after_exit": 615,
                "resume_completed": True,
                "source": "operator_observed_before_orchestrator_activation",
            },
        ],
        "attempts": [],
    }


def save_audit(audit: dict[str, Any]) -> None:
    audit["updated_at"] = now()
    audit["native_crash_exit_count"] = len(
        audit.get("pre_orchestrator_observations", [])
    ) + sum(
        row.get("returncode") in (-11, -6, 134, 139)
        for row in audit["attempts"]
    )
    audit["successful_attempt_count"] = sum(
        row.get("returncode") == 0 for row in audit["attempts"]
    )
    audit["all_recorded_attempts_are_sequential"] = True
    write_json(AUDIT, audit)


def collect_model(
    model: str,
    stage: str,
    audit: dict[str, Any],
    max_attempts: int,
) -> None:
    if complete(model, stage):
        return
    stagnant_failures = 0
    for attempt in range(1, max_attempts + 1):
        before = row_count(model, stage)
        started_at = now()
        command = [
            sys.executable,
            str(ROOT / "tests/gpt5/phase408_partition_interface_collection.py"),
            "--model",
            model,
            "--split",
            stage,
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        after = row_count(model, stage)
        record = {
            "stage": stage,
            "model": model,
            "attempt": attempt,
            "started_at": started_at,
            "finished_at": now(),
            "returncode": completed.returncode,
            "persisted_rows_before": before,
            "persisted_rows_after": after,
            "made_progress": after > before,
            "complete_marker_valid": complete(model, stage),
        }
        audit["attempts"].append(record)
        save_audit(audit)
        if record["complete_marker_valid"]:
            return
        if completed.returncode not in (-11, -6, 134, 139):
            raise RuntimeError(
                f"Phase408 non-recoverable exit {completed.returncode}: {model}/{stage}"
            )
        stagnant_failures = 0 if after > before else stagnant_failures + 1
        if stagnant_failures >= 3:
            raise RuntimeError(
                f"Phase408 native recovery made no progress three times: {model}/{stage}"
            )
    raise RuntimeError(f"Phase408 exceeded retry budget: {model}/{stage}")


def run_analysis(stage: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "tests/gpt5/phase408_partition_interface_analysis.py"),
        "--stage",
        stage,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main(start_stage: str, max_attempts: int) -> None:
    audit = load_audit()
    start = STAGES.index(start_stage)
    for stage in STAGES[start:]:
        for model in MODELS:
            collect_model(model, stage, audit, max_attempts)
        run_analysis(stage)
    save_audit(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", choices=STAGES, default="discovery")
    parser.add_argument("--max-attempts", type=int, default=12)
    args = parser.parse_args()
    main(args.from_stage, args.max_attempts)
