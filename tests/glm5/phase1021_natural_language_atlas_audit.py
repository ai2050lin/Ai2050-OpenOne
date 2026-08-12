#!/usr/bin/env python3
"""Audit Phase1021 protocol, behavior, scans, and analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1021_natural_language_atlas_protocol as protocol


SCRIPTS = (
    "phase1021_natural_language_atlas_protocol.py",
    "phase1021_natural_language_atlas_behavior.py",
    "phase1021_natural_language_atlas_scan.py",
    "phase1021_natural_language_atlas_finalize.py",
    "phase1021_natural_language_atlas_audit.py",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(
    rows: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
) -> None:
    rows.append({
        "name": name,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
    })


def main() -> None:
    checks = []
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_summary = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "summary.json"
    )
    check(
        checks,
        "protocol_identity",
        {
            "phase": prereg["phase"],
            "revision": prereg["protocol_revision"],
            "digest": prereg["protocol_digest"],
        },
        {
            "phase": protocol.PHASE,
            "revision": protocol.PROTOCOL_REVISION,
            "digest": protocol_summary["protocol_digest"],
        },
    )

    unit_count = int(protocol_summary["unit_count"])
    for model in protocol.MODELS:
        for mode in protocol.PROMPT_MODES:
            audit = protocol.read_json(
                protocol.OUT_ROOT
                / "protocol"
                / f"audit.{model}.{mode}.json"
            )
            check(
                checks,
                f"{model}_{mode}_protocol",
                {
                    "units": audit["unit_count"],
                    "cases": audit["case_count"],
                    "identity": audit["identity_exact"],
                    "prefix": audit["prefix_branch_unchanged"],
                    "answers_hidden": (
                        audit["candidate_answers_not_displayed"]
                    ),
                    "overlap": (
                        audit[
                            "discovery_confirmation_rendered_overlap"
                        ]
                    ),
                },
                {
                    "units": unit_count,
                    "cases": unit_count * len(protocol.STATES),
                    "identity": True,
                    "prefix": True,
                    "answers_hidden": True,
                    "overlap": 0,
                },
            )

    behavior_counts = {}
    expected_formal = unit_count * len(protocol.FACTORIAL_STATES)
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "formal.summary.json"
        )
        selection = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "selection.json"
        )
        behavior_counts[model] = len(rows)
        check(
            checks,
            f"{model}_behavior_complete",
            {
                "rows": len(rows),
                "summary": summary["case_count"],
                "digest": summary["protocol_digest"],
                "selection_digest": selection["protocol_digest"],
                "nonempty_outputs": all(
                    isinstance(row["generated_text"], str)
                    for row in rows
                ),
            },
            {
                "rows": expected_formal,
                "summary": expected_formal,
                "digest": prereg["protocol_digest"],
                "selection_digest": prereg["protocol_digest"],
                "nonempty_outputs": True,
            },
        )

    scan_counts = {}
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT
            / "formal_scan"
            / model
            / "summary.json"
        )
        scan_counts[model] = {
            "panel_count": summary["panel_count"],
            "unit_count": summary["unit_count"],
            "state_case_count": summary["state_case_count"],
        }
        check(
            checks,
            f"{model}_scan_arithmetic",
            {
                "state_cases": summary["state_case_count"],
                "five_times_units": summary["unit_count"] * 5,
                "identity": summary["identity_maximum"],
                "prefix": summary["prefix_branch_maximum"],
                "digest": summary["protocol_digest"],
            },
            {
                "state_cases": summary["unit_count"] * 5,
                "five_times_units": summary["unit_count"] * 5,
                "identity": 0.0,
                "prefix": 0.0,
                "digest": prereg["protocol_digest"],
            },
        )

    analysis = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "summary.json"
    )
    check(
        checks,
        "analysis_identity",
        {
            "phase": analysis["phase"],
            "revision": analysis["protocol_revision"],
            "digest": analysis["protocol_digest"],
            "repeated": analysis["counts"][
                "repeated_physical_event_rows"
            ],
        },
        {
            "phase": protocol.PHASE,
            "revision": protocol.PROTOCOL_REVISION,
            "digest": prereg["protocol_digest"],
            "repeated": len(protocol.read_jsonl(
                protocol.OUT_ROOT
                / "analysis"
                / "repeated_physical_events.jsonl"
            )),
        },
    )

    markdown = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*.md")
    ]
    temporary = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*.tmp")
    ]
    forbidden = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
        and path.name.lower() in {
            "thumbs.db",
            ".ds_store",
        }
    ]
    check(
        checks,
        "result_hygiene",
        {
            "markdown": markdown,
            "temporary": temporary,
            "forbidden": forbidden,
        },
        {
            "markdown": [],
            "temporary": [],
            "forbidden": [],
        },
    )

    script_manifest = []
    for name in SCRIPTS:
        path = ROOT / "tests" / "glm5" / name
        script_manifest.append({
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    artifact_manifest = [
        {
            "path": str(path.relative_to(protocol.OUT_ROOT)),
            "size": path.stat().st_size,
        }
        for path in sorted(protocol.OUT_ROOT.rglob("*"))
        if path.is_file()
    ]
    result = {
        "schema_version": "phase1021_audit.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "behavior_counts": behavior_counts,
        "scan_counts": scan_counts,
        "script_manifest": script_manifest,
        "artifact_manifest_file_count": len(artifact_manifest),
        "artifact_manifest": artifact_manifest,
    }
    protocol.write_json(
        protocol.OUT_ROOT / "audit" / "audit.json", result
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
