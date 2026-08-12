#!/usr/bin/env python3
"""Audit Phase1018 protocol, result completeness, and artifact integrity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


ATTACHMENTS = (
    Path(
        r"C:\Users\Admin\.codex\attachments"
        r"\788a9952-af70-4a4f-9d15-59fce719940f\pasted-text.txt"
    ),
    Path(
        r"C:\Users\Admin\.codex\attachments"
        r"\3d0ec9c5-0ca8-4bf0-9971-99ae06cb5314\pasted-text.txt"
    ),
)
SCRIPT_NAMES = (
    "phase1018_language_pattern_protocol.py",
    "phase1018_language_pattern_behavior.py",
    "phase1018_language_pattern_scan.py",
    "phase1018_language_pattern_finalize.py",
    "phase1018_language_pattern_audit.py",
)
FORBIDDEN_RESULT_SUFFIXES = (
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".npy",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact_manifest() -> list[dict[str, Any]]:
    paths = [
        ROOT / "tests" / "glm5" / name for name in SCRIPT_NAMES
    ]
    paths.extend(
        path
        for path in OUT_ROOT.rglob("*")
        if path.is_file() and "audit" not in path.relative_to(OUT_ROOT).parts
    )
    paths.extend(ATTACHMENTS)
    rows = []
    for path in sorted(set(paths), key=lambda value: str(value).lower()):
        rows.append({
            "path": (
                str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else str(path)
            ),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def audit() -> dict[str, Any]:
    checks = []
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    checks.append({
        "name": "protocol_revision",
        "passed": int(prereg["protocol_revision"]) == PROTOCOL_REVISION,
        "observed": prereg["protocol_revision"],
        "expected": PROTOCOL_REVISION,
    })
    checks.append({
        "name": "attachments_present",
        "passed": all(path.exists() for path in ATTACHMENTS),
        "observed": [path.exists() for path in ATTACHMENTS],
        "expected": [True, True],
    })

    behavior_counts = {}
    scan_counts = {}
    panel_file_checks = {}
    for model in MODELS:
        behavior_rows = read_jsonl(
            OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        behavior_summary = read_json(
            OUT_ROOT / "behavior" / model / "formal.summary.json"
        )
        behavior_counts[model] = {
            "row_count": len(behavior_rows),
            "summary_count": int(behavior_summary["count"]),
            "protocol_digest": behavior_summary["protocol_digest"],
        }
        checks.append({
            "name": f"{model}_behavior_complete",
            "passed": (
                len(behavior_rows) == 3072
                and int(behavior_summary["count"]) == 3072
                and behavior_summary["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "observed": behavior_counts[model],
            "expected": {
                "row_count": 3072,
                "summary_count": 3072,
                "protocol_digest": prereg["protocol_digest"],
            },
        })
        scan_summary = read_json(
            OUT_ROOT / "formal_scan" / model / "summary.json"
        )
        events = read_jsonl(
            OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        expected_events = (
            1
            + 3 * int(scan_summary["model_info"]["n_layers"])
            + int(scan_summary["model_info"]["n_layers"])
            * int(scan_summary["model_info"]["head_count"])
        )
        scan_counts[model] = {
            "panel_count": int(scan_summary["panel_count"]),
            "unit_count": int(scan_summary["unit_count"]),
            "state_case_count": int(scan_summary["state_case_count"]),
            "event_count": len(events),
            "identity_maximum": scan_summary["identity_maximum"],
            "prefix_branch_maximum": scan_summary[
                "prefix_branch_maximum"
            ],
        }
        checks.append({
            "name": f"{model}_scan_complete",
            "passed": (
                int(scan_summary["panel_count"]) == 96
                and int(scan_summary["unit_count"]) == 768
                and int(scan_summary["state_case_count"]) == 3840
                and len(events) == expected_events
                and float(scan_summary["identity_maximum"]) == 0.0
                and float(scan_summary["prefix_branch_maximum"]) == 0.0
                and scan_summary["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "observed": scan_counts[model],
            "expected": {
                "panel_count": 96,
                "unit_count": 768,
                "state_case_count": 3840,
                "event_count": expected_events,
                "identity_maximum": 0.0,
                "prefix_branch_maximum": 0.0,
            },
        })

        required_count = 0
        array_shape_count = 0
        for family in FAMILIES:
            for panel_summary_path in (
                OUT_ROOT / "formal_scan" / model / family
            ).glob("*/*/summary.json"):
                panel_root = panel_summary_path.parent
                required = (
                    panel_root / "summary.json",
                    panel_root / "units.jsonl",
                    panel_root / "response_scalars.npz",
                    panel_root / "direction_metrics.npz",
                    panel_root / "directions.npz",
                )
                required_count += int(all(path.exists() for path in required))
                with np.load(
                    panel_root / "response_scalars.npz"
                ) as response:
                    array_shape_count += int(
                        response["normalized_magnitude"].shape
                        == (8, 6, 7, expected_events)
                    )
        panel_file_checks[model] = {
            "complete_panel_file_sets": required_count,
            "valid_response_array_shapes": array_shape_count,
        }
        checks.append({
            "name": f"{model}_panel_artifacts",
            "passed": required_count == 96 and array_shape_count == 96,
            "observed": panel_file_checks[model],
            "expected": {
                "complete_panel_file_sets": 96,
                "valid_response_array_shapes": 96,
            },
        })

    analysis = read_json(OUT_ROOT / "analysis" / "summary.json")
    automatic = read_json(
        OUT_ROOT / "analysis" / "automatic_continuation.json"
    )
    checks.append({
        "name": "analysis_complete",
        "passed": (
            int(analysis["phase"]) == PHASE
            and analysis["protocol_digest"] == prereg["protocol_digest"]
            and int(analysis["counts"]["item_summary_rows"]) == 144
            and int(analysis["counts"]["family_model_summary_rows"]) == 12
            and int(
                analysis["counts"]["threshold_sensitivity_rows"]
            ) == 108
        ),
        "observed": analysis["counts"],
        "expected": {
            "item_summary_rows": 144,
            "family_model_summary_rows": 12,
            "threshold_sensitivity_rows": 108,
        },
    })
    checks.append({
        "name": "automatic_gate_recorded",
        "passed": (
            automatic["any_targeted_causal_test_started"]
            is False
            and len(automatic["by_family"]) == len(FAMILIES)
        ),
        "observed": automatic,
        "expected": {
            "any_targeted_causal_test_started": False,
            "family_count": len(FAMILIES),
        },
    })
    forbidden = [
        str(path.relative_to(OUT_ROOT))
        for path in OUT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in FORBIDDEN_RESULT_SUFFIXES
    ]
    markdown = [
        str(path.relative_to(OUT_ROOT))
        for path in OUT_ROOT.rglob("*.md")
    ]
    checks.append({
        "name": "no_model_weights_in_results",
        "passed": not forbidden,
        "observed": forbidden,
        "expected": [],
    })
    checks.append({
        "name": "no_markdown_in_results",
        "passed": not markdown,
        "observed": markdown,
        "expected": [],
    })

    manifest = artifact_manifest()
    audit_root = OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(audit_root / "artifact_manifest.jsonl", manifest)
    result = {
        "schema_version": "phase1018_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "behavior_counts": behavior_counts,
        "scan_counts": scan_counts,
        "panel_file_checks": panel_file_checks,
        "manifest_file_count": len(manifest),
        "attachment_hashes": {
            str(path): sha256(path) for path in ATTACHMENTS
        },
    }
    write_json(audit_root / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1018 audit failed")
    return result


if __name__ == "__main__":
    audit()
