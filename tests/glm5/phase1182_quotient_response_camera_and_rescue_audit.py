#!/usr/bin/env python3
"""Independent integrity/recompute audit for Phase1182."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1182_quotient_response_camera_and_rescue as phase  # noqa: E402


AUDIT_PATH = phase.OUT_ROOT / "audit/independent_audit.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-split", type=int, default=4)
    args = parser.parse_args()
    protocol = phase.validate_protocol()
    checks: list[dict[str, Any]] = []
    add(checks, "final_exists", phase.FINAL_PATH.exists())
    add(checks, "camera_seal_exists", phase.CAMERA_SEAL.exists())
    add(checks, "camera_metadata_exists", phase.CAMERA_METADATA.exists())
    metadata = phase.read_json(phase.CAMERA_METADATA)
    add(checks, "camera_seal_hash", phase.file_sha256(phase.CAMERA_SEAL) == metadata["npz_sha256"])
    seals = phase.load_camera_seal()
    for split, rows_path, summary_path, rescue_path, expected_count in (
        ("discovery", phase.DISCOVERY_ROWS, phase.DISCOVERY_SUMMARY, phase.DISCOVERY_RESCUE, 64),
        ("confirmation", phase.CONFIRMATION_ROWS, phase.CONFIRMATION_SUMMARY, phase.CONFIRMATION_RESCUE, 64),
    ):
        rows = phase.read_jsonl(rows_path)
        summary = phase.read_json(summary_path)
        rescue = phase.read_json(rescue_path)
        add(checks, f"{split}_row_count", len(rows) == expected_count, len(rows))
        add(checks, f"{split}_unique_checkpoints", len({row["checkpoint"] for row in rows}) == expected_count)
        add(checks, f"{split}_rows_digest", phase.digest(rows) == summary["rows_digest"])
        add(checks, f"{split}_camera_hash", summary["camera_seal_sha256"] == metadata["npz_sha256"])
        add(checks, f"{split}_rescue_summary_present", rescue["summary"] == summary["rescue"])
        manifest = protocol["checkpoint_manifests"][split]
        add(
            checks,
            f"{split}_endpoint_hashes",
            all(
                row["checkpoint_sha256"]
                == manifest[next(key for key in manifest if key.endswith(row["checkpoint"]))]
                for row in rows
            ),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        device = torch.device("cuda")
        selected = np.linspace(0, len(rows) - 1, num=min(args.sample_per_split, len(rows)), dtype=int)
        path_map = {path.name: path for path in phase.endpoint_paths(split)}
        maximum_target_error = 0.0
        maximum_endpoint_internal_error = 0.0
        for index in selected:
            row = rows[int(index)]
            payload = torch.load(path_map[row["checkpoint"]], map_location="cpu", weights_only=False)
            panel = phase.load_panel(payload, split)
            model = phase.p1181.load_model(payload, device)
            target = phase.p1181.response_spectrum(model, panel, device)["ordered"]
            internal = phase.internal_features(model, panel, device)
            maximum_target_error = max(
                maximum_target_error,
                float(np.max(np.abs(np.asarray(target) - np.asarray(row["target"])))),
            )
            maximum_endpoint_internal_error = max(
                maximum_endpoint_internal_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(internal) - np.asarray(row["endpoint_internal"])
                        )
                    )
                ),
            )
            del model
            torch.cuda.empty_cache()
        add(checks, f"{split}_sample_target_recompute", maximum_target_error <= 1e-7, maximum_target_error)
        add(
            checks,
            f"{split}_sample_endpoint_internal_recompute",
            maximum_endpoint_internal_error <= 1e-7,
            maximum_endpoint_internal_error,
        )
        if split == "confirmation":
            thresholds = protocol["thresholds"]
            task_names = phase.qualifying_task_names(rows, thresholds)
            score_rows = [
                row
                for row in rows
                if row["task_name"] in task_names and phase.qualified(row, thresholds)
            ]
            endpoint = phase.score_stage(score_rows, "endpoint", seals["endpoint"])
            prefix = phase.score_stage(score_rows, "prefix", seals["prefix"])
            for stage, recomputed in (("endpoint", endpoint), ("prefix", prefix)):
                stored = summary[stage]
                add(
                    checks,
                    f"confirmation_{stage}_cosine",
                    abs(recomputed["joint"]["mean_cosine"] - stored["joint"]["mean_cosine"]) <= 1e-12,
                )
                add(
                    checks,
                    f"confirmation_{stage}_increment",
                    abs(recomputed["residual_cosine_improvement"] - stored["residual_cosine_improvement"]) <= 1e-12,
                )
    final = phase.read_json(phase.FINAL_PATH)
    confirmation = phase.read_json(phase.CONFIRMATION_SUMMARY)
    add(checks, "primary_decision", final["primary_pass"] == confirmation["primary_pass"])
    add(
        checks,
        "component_decisions",
        final["component_decisions"]
        == {
            "endpoint_increment": confirmation["endpoint"]["gate_pass"],
            "prefix_increment": confirmation["prefix"]["gate_pass"],
            "donor_future_response_rescue": confirmation["rescue"]["gate_pass"],
        },
    )
    integrity_pass = all(check["passed"] for check in checks)
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "integrity_and_recompute_pass": integrity_pass,
        "scientific_primary_pass": final["primary_pass"],
        "check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "checks": checks,
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not integrity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
