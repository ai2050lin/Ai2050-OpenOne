#!/usr/bin/env python3
"""Execute every Phase1073 stage with sequential model loading."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1073_late_query_protocol as protocol


MANIFEST = protocol.OUT_ROOT / "analysis" / "run_manifest.json"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(payload: dict[str, Any]) -> None:
    protocol.write_json(MANIFEST, payload)


def execute(
    payload: dict[str, Any],
    label: str,
    script: str,
    *args: str,
) -> None:
    stage = {
        "label": label,
        "script": script,
        "args": list(args),
        "started_at_utc": timestamp(),
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "return_code": None,
    }
    payload["stages"].append(stage)
    write_manifest(payload)
    started = time.time()
    result = subprocess.run(
        [sys.executable, str(TEST_ROOT / script), *args],
        cwd=str(ROOT),
        check=False,
    )
    stage["elapsed_seconds"] = time.time() - started
    stage["completed_at_utc"] = timestamp()
    stage["return_code"] = int(result.returncode)
    write_manifest(payload)
    if result.returncode != 0:
        payload["status"] = "failed"
        payload["failed_stage"] = label
        write_manifest(payload)
        raise SystemExit(result.returncode)


def run_formal(payload: dict[str, Any]) -> None:
    execute(
        payload,
        "formal_protocol",
        "phase1073_late_query_protocol.py",
    )
    for model in protocol.MODELS:
        execute(
            payload,
            f"formal_scan_{model}",
            "phase1073_late_query_scan.py",
            model,
        )
    execute(payload, "formal_finalize", "phase1073_finalize.py")
    execute(
        payload,
        "posthoc_diagnostics",
        "phase1073_posthoc_diagnostics.py",
    )
    payload["status"] = "complete"
    payload["completed_at_utc"] = timestamp()
    write_manifest(payload)
    execute(payload, "result_audit", "phase1073_result_audit.py")
    payload["status"] = "complete"
    payload["completed_at_utc"] = timestamp()
    write_manifest(payload)


def new_manifest() -> dict[str, Any]:
    return {
        "schema_version": "phase1073_run_manifest.v1",
        "phase": protocol.PHASE,
        "python": sys.executable,
        "started_at_utc": timestamp(),
        "completed_at_utc": None,
        "status": "running",
        "model_order": list(protocol.MODELS),
        "precision": "fp16",
        "quantization": "none",
        "strictly_sequential": True,
        "stages": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-formal",
        action="store_true",
        help="reuse complete calibration and restart formal stages",
    )
    args = parser.parse_args()
    if args.resume_formal:
        payload = protocol.read_json(MANIFEST)
        payload["status"] = "running"
        payload["failed_stage"] = None
        payload["resume_started_at_utc"] = timestamp()
        write_manifest(payload)
        run_formal(payload)
        print("Phase1073 resumed formal run complete")
        return

    payload = new_manifest()
    write_manifest(payload)
    execute(
        payload,
        "calibration_protocol",
        "phase1073_behavior_calibration_protocol.py",
    )
    for model in protocol.MODELS:
        execute(
            payload,
            f"calibration_scan_{model}",
            "phase1073_behavior_calibration_scan.py",
            model,
        )
    execute(
        payload,
        "calibration_finalize",
        "phase1073_behavior_calibration_finalize.py",
    )
    run_formal(payload)
    print("Phase1073 sequential run complete")


if __name__ == "__main__":
    main()
