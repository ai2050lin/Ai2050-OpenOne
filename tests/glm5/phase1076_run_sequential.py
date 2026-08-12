#!/usr/bin/env python3
"""Run Phase1076 stages sequentially with one FP16 model at a time."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1076_polarity_head_causal_protocol as protocol


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(arguments: list[str]) -> float:
    started = time.time()
    subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
    )
    return float(time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a model stage whose summary already exists.",
    )
    args = parser.parse_args()
    analysis_dir = protocol.OUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phase1076_run_manifest.v1",
        "phase": protocol.PHASE,
        "python_executable": sys.executable,
        "started_at_utc": timestamp(),
        "sequential_model_order": list(protocol.MODELS),
        "precision": protocol.PRECISION,
        "quantization": protocol.QUANTIZATION,
        "model_stages": [],
        "concurrent_model_processes": False,
    }
    manifest["protocol_elapsed_seconds"] = run_command([
        "tests/glm5/phase1076_polarity_head_causal_protocol.py"
    ])
    protocol.write_json(
        analysis_dir / "run_manifest.json", manifest
    )

    for model in protocol.MODELS:
        summary_path = (
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        stage = {
            "model": model,
            "stage": "behavior",
            "started_at_utc": timestamp(),
            "skipped_existing": bool(
                args.resume and summary_path.exists()
            ),
        }
        if not stage["skipped_existing"]:
            stage["elapsed_seconds"] = run_command([
                "tests/glm5/phase1076_behavior_scan.py",
                "--model",
                model,
            ])
        stage["completed_at_utc"] = timestamp()
        manifest["model_stages"].append(stage)
        protocol.write_json(
            analysis_dir / "run_manifest.json", manifest
        )

    manifest["behavior_finalize_elapsed_seconds"] = run_command([
        "tests/glm5/phase1076_behavior_finalize.py"
    ])
    decision = protocol.read_json(
        analysis_dir / "behavior_decision.json"
    )
    manifest["behavior_decision"] = decision
    protocol.write_json(
        analysis_dir / "run_manifest.json", manifest
    )

    if decision["should_run_causal_validation"]:
        for model in protocol.MODELS:
            summary_path = (
                protocol.OUT_ROOT
                / "causal"
                / model
                / "summary.json"
            )
            stage = {
                "model": model,
                "stage": "causal",
                "started_at_utc": timestamp(),
                "skipped_existing": bool(
                    args.resume and summary_path.exists()
                ),
            }
            if not stage["skipped_existing"]:
                stage["elapsed_seconds"] = run_command([
                    "tests/glm5/phase1076_causal_scan.py",
                    "--model",
                    model,
                ])
            stage["completed_at_utc"] = timestamp()
            manifest["model_stages"].append(stage)
            protocol.write_json(
                analysis_dir / "run_manifest.json", manifest
            )

    manifest["finalize_elapsed_seconds"] = run_command([
        "tests/glm5/phase1076_finalize.py"
    ])
    manifest["completed_at_utc"] = timestamp()
    manifest["completed"] = True
    protocol.write_json(
        analysis_dir / "run_manifest.json", manifest
    )
    manifest["audit_elapsed_seconds"] = run_command([
        "tests/glm5/phase1076_result_audit.py"
    ])
    protocol.write_json(
        analysis_dir / "run_manifest.json", manifest
    )


if __name__ == "__main__":
    main()
