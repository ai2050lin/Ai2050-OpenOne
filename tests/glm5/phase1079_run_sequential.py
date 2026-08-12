#!/usr/bin/env python3
"""Run Phase1079 sequentially across the three local FP16 models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1079_output_orthogonal_pattern_protocol as protocol


def run_command(command: list[str]) -> dict[str, Any]:
    started = time.time()
    print(json.dumps({
        "event": "command_start",
        "command": command,
    }), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps({
        "event": "command_end",
        **result,
    }), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {result}")
    return result


def completed_model(
    model_name: str,
    protocol_digest: str,
    case_count: int,
    unit_count: int,
) -> dict[str, Any] | None:
    path = protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    if not path.exists():
        return None
    summary = protocol.read_json(path)
    if (
        summary.get("protocol_digest") == protocol_digest
        and summary.get("case_count") == case_count
        and summary.get("unit_count") == unit_count
    ):
        return summary
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    started = time.time()
    commands = []

    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1079_output_orthogonal_pattern_protocol.py"
        ),
    ]))
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_digest = str(prereg["protocol_digest"])
    case_count = int(prereg["case_count_per_model"])
    unit_count = int(prereg["unit_count_per_model"])

    model_runs = []
    for model_name in protocol.MODELS:
        existing = completed_model(
            model_name, protocol_digest, case_count, unit_count
        )
        if not args.force and existing is not None:
            row = {
                "model": model_name,
                "status": "reused_matching_complete_result",
                "elapsed_seconds": 0.0,
                "result_elapsed_seconds": existing["elapsed_seconds"],
                "result_summary_digest": existing["summary_digest"],
            }
            model_runs.append(row)
            print(json.dumps(row), flush=True)
            continue
        result = run_command([
            python,
            str(
                ROOT
                / "tests"
                / "glm5"
                / "phase1079_output_orthogonal_pattern_scan.py"
            ),
            model_name,
        ])
        summary = protocol.read_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        model_runs.append({
            "model": model_name,
            "status": "completed",
            "elapsed_seconds": result["elapsed_seconds"],
            "result_elapsed_seconds": summary["elapsed_seconds"],
            "result_summary_digest": summary["summary_digest"],
        })
        commands.append(result)

    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1079_output_orthogonal_pattern_finalize.py"
        ),
    ]))
    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1079_result_audit.py"
        ),
    ]))
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    manifest = {
        "schema_version": "phase1079_run_manifest.v1",
        "phase": protocol.PHASE,
        "protocol_digest": protocol_digest,
        "python": python,
        "model_order": list(protocol.MODELS),
        "precision": protocol.PRECISION,
        "quantization": protocol.QUANTIZATION,
        "sequential_execution": True,
        "model_runs": model_runs,
        "commands": commands,
        "automatic_next": automatic,
        "elapsed_seconds": time.time() - started,
    }
    manifest["manifest_digest"] = protocol.digest(manifest)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "run_manifest.json",
        manifest,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "status": "complete",
        "model_order": list(protocol.MODELS),
        "automatic_continue": automatic["continue"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "manifest_digest": manifest["manifest_digest"],
    }), flush=True)


if __name__ == "__main__":
    main()
