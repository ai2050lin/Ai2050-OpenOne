#!/usr/bin/env python3
"""Run Phase1077 models sequentially, then finalize and audit."""

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

import phase1077_nonblocking_pattern_atlas_protocol as protocol


def run_command(command: list[str]) -> dict[str, Any]:
    started = time.time()
    print(json.dumps({
        "event": "command_start",
        "command": command,
    }), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )
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
    digest: str,
) -> dict[str, Any] | None:
    summary_path = (
        protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    )
    if not summary_path.exists():
        return None
    summary = protocol.read_json(summary_path)
    complete = (
        summary.get("protocol_digest") == digest
        and summary.get("case_count") == (
            len(protocol.FAMILIES) * 2 * 15 * 2 * 4
        )
    )
    return summary if complete else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    commands = []
    started = time.time()

    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1077_nonblocking_pattern_atlas_protocol.py"
        ),
    ]))
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    digest = str(prereg["protocol_digest"])

    model_runs = []
    for model_name in protocol.MODELS:
        existing = completed_model(model_name, digest)
        if not args.force and existing is not None:
            row = {
                "model": model_name,
                "status": "reused_matching_complete_result",
                "elapsed_seconds": 0.0,
                "result_elapsed_seconds": existing[
                    "elapsed_seconds"
                ],
                "result_summary_digest": existing[
                    "summary_digest"
                ],
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
                / "phase1077_nonblocking_pattern_atlas_scan.py"
            ),
            model_name,
        ])
        model_runs.append({
            "model": model_name,
            "status": "completed",
            "elapsed_seconds": result["elapsed_seconds"],
            "result_elapsed_seconds": protocol.read_json(
                protocol.OUT_ROOT
                / "atlas"
                / model_name
                / "summary.json"
            )["elapsed_seconds"],
        })
        commands.append(result)

    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1077_nonblocking_pattern_atlas_finalize.py"
        ),
    ]))
    commands.append(run_command([
        python,
        str(
            ROOT
            / "tests"
            / "glm5"
            / "phase1077_result_audit.py"
        ),
    ]))
    manifest = {
        "schema_version": "phase1077_run_manifest.v1",
        "phase": protocol.PHASE,
        "protocol_digest": digest,
        "python": python,
        "model_order": list(protocol.MODELS),
        "precision": protocol.PRECISION,
        "quantization": protocol.QUANTIZATION,
        "sequential_execution": True,
        "model_runs": model_runs,
        "commands": commands,
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
        "elapsed_seconds": manifest["elapsed_seconds"],
        "manifest_digest": manifest["manifest_digest"],
    }), flush=True)


if __name__ == "__main__":
    main()
