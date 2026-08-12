#!/usr/bin/env python3
"""Audit Phase1019 corrected held-out atlas artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1019_corrected_pattern_protocol as protocol


SCRIPT_NAMES = (
    "phase1019_corrected_pattern_protocol.py",
    "phase1019_corrected_pattern_behavior.py",
    "phase1019_corrected_pattern_scan.py",
    "phase1019_corrected_pattern_finalize.py",
    "phase1019_corrected_pattern_audit.py",
)
FORBIDDEN_SUFFIXES = (
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


def carrier_prefix_maximum(model: str, family: str) -> float:
    maximum = 0.0
    for path in (
        protocol.OUT_ROOT / "formal_scan" / model / family
    ).glob("*/*/response_scalars.npz"):
        with np.load(path) as data:
            roles = data["role_names"].tolist()
            contrasts = data["contrast_names"].tolist()
            values = data["normalized_magnitude"]
            d_index = contrasts.index("D")
            for role in ("carrier_start", "carrier_end"):
                maximum = max(
                    maximum,
                    float(np.nanmax(
                        values[:, d_index, roles.index(role), :]
                    )),
                )
    return maximum


def artifact_manifest() -> list[dict[str, Any]]:
    paths = [
        ROOT / "tests" / "glm5" / name for name in SCRIPT_NAMES
    ]
    paths.extend(
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
        and "audit" not in path.relative_to(protocol.OUT_ROOT).parts
    )
    rows = []
    for path in sorted(paths, key=lambda value: str(value).lower()):
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def audit() -> dict[str, Any]:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    checks = [{
        "name": "protocol_identity",
        "passed": (
            int(prereg["phase"]) == protocol.PHASE
            and int(prereg["protocol_revision"])
            == protocol.PROTOCOL_REVISION
        ),
        "observed": {
            "phase": prereg["phase"],
            "revision": prereg["protocol_revision"],
        },
        "expected": {
            "phase": protocol.PHASE,
            "revision": protocol.PROTOCOL_REVISION,
        },
    }]
    behavior_counts = {}
    scan_counts = {}
    carrier_checks = {}
    for model in protocol.MODELS:
        behavior_rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        behavior_summary = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "formal.summary.json"
        )
        behavior_counts[model] = {
            "row_count": len(behavior_rows),
            "summary_count": int(behavior_summary["count"]),
        }
        checks.append({
            "name": f"{model}_behavior_complete",
            "passed": (
                len(behavior_rows) == 3840
                and int(behavior_summary["count"]) == 3840
                and behavior_summary["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "observed": behavior_counts[model],
            "expected": {
                "row_count": 3840,
                "summary_count": 3840,
            },
        })
        scan_summary = protocol.read_json(
            protocol.OUT_ROOT / "formal_scan" / model / "summary.json"
        )
        events = protocol.read_jsonl(
            protocol.OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        n_layers = int(scan_summary["model_info"]["n_layers"])
        n_heads = int(scan_summary["model_info"]["head_count"])
        expected_events = 1 + 3 * n_layers + n_layers * n_heads
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
                int(scan_summary["panel_count"]) == 120
                and int(scan_summary["unit_count"]) == 960
                and int(scan_summary["state_case_count"]) == 4800
                and len(events) == expected_events
                and float(scan_summary["identity_maximum"]) == 0.0
                and float(scan_summary["prefix_branch_maximum"]) == 0.0
                and scan_summary["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "observed": scan_counts[model],
            "expected": {
                "panel_count": 120,
                "unit_count": 960,
                "state_case_count": 4800,
                "event_count": expected_events,
                "identity_maximum": 0.0,
                "prefix_branch_maximum": 0.0,
            },
        })
        complete_panels = 0
        valid_shapes = 0
        for family in protocol.FAMILIES:
            for summary_path in (
                protocol.OUT_ROOT
                / "formal_scan"
                / model
                / family
            ).glob("*/*/summary.json"):
                panel_root = summary_path.parent
                required = (
                    panel_root / "summary.json",
                    panel_root / "units.jsonl",
                    panel_root / "response_scalars.npz",
                    panel_root / "direction_metrics.npz",
                    panel_root / "directions.npz",
                )
                complete_panels += int(
                    all(path.exists() for path in required)
                )
                with np.load(
                    panel_root / "response_scalars.npz"
                ) as data:
                    valid_shapes += int(
                        data["normalized_magnitude"].shape
                        == (8, 6, 7, expected_events)
                    )
        checks.append({
            "name": f"{model}_panel_artifacts",
            "passed": complete_panels == 120 and valid_shapes == 120,
            "observed": {
                "complete_panels": complete_panels,
                "valid_shapes": valid_shapes,
            },
            "expected": {
                "complete_panels": 120,
                "valid_shapes": 120,
            },
        })
        carrier_checks[model] = {
            family: carrier_prefix_maximum(model, family)
            for family in ("punctuation", "translation", "contrast")
        }
        checks.append({
            "name": f"{model}_post_carrier_causal_order",
            "passed": all(
                value == 0.0
                for value in carrier_checks[model].values()
            ),
            "observed": carrier_checks[model],
            "expected": {
                family: 0.0
                for family in ("punctuation", "translation", "contrast")
            },
        })

    split_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "split_overlap_audit.jsonl"
    )
    analysis = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "summary.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "automatic_continuation.json"
    )
    checks.append({
        "name": "independent_confirmation",
        "passed": (
            len(split_rows) == 180
            and all(
                row["exact_overlap_count"] == 0
                and row["independent_confirmation"]
                for row in split_rows
            )
        ),
        "observed": {
            "row_count": len(split_rows),
            "overlap_total": sum(
                row["exact_overlap_count"] for row in split_rows
            ),
        },
        "expected": {"row_count": 180, "overlap_total": 0},
    })
    checks.append({
        "name": "analysis_complete",
        "passed": (
            int(analysis["counts"]["item_summary_rows"]) == 180
            and int(analysis["counts"]["family_model_summary_rows"]) == 12
            and int(
                analysis["counts"]["threshold_sensitivity_rows"]
            ) == 108
            and int(
                analysis["counts"]["rare_lexical_necessity_rows"]
            ) == 36
            and int(analysis["counts"]["rare_depth_summary_rows"]) > 0
        ),
        "observed": analysis["counts"],
        "expected": {
            "item_summary_rows": 180,
            "family_model_summary_rows": 12,
            "threshold_sensitivity_rows": 108,
            "rare_lexical_necessity_rows": 36,
        },
    })
    checks.append({
        "name": "automatic_stop_recorded",
        "passed": (
            automatic["any_targeted_causal_test_started"] is False
        ),
        "observed": automatic[
            "any_targeted_causal_test_started"
        ],
        "expected": False,
    })
    forbidden = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    markdown = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*.md")
    ]
    checks.append({
        "name": "result_hygiene",
        "passed": not forbidden and not markdown,
        "observed": {
            "forbidden_artifacts": forbidden,
            "markdown_files": markdown,
        },
        "expected": {
            "forbidden_artifacts": [],
            "markdown_files": [],
        },
    })

    manifest = artifact_manifest()
    audit_root = protocol.OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(
        audit_root / "artifact_manifest.jsonl", manifest
    )
    result = {
        "schema_version": "phase1019_audit.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "behavior_counts": behavior_counts,
        "scan_counts": scan_counts,
        "carrier_prefix_maximum": carrier_checks,
        "manifest_file_count": len(manifest),
    }
    protocol.write_json(audit_root / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1019 audit failed")
    return result


if __name__ == "__main__":
    audit()
