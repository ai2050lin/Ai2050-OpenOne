#!/usr/bin/env python3
"""Audit Phase1020 protocol, behavior, scan, analysis, and artifact hygiene."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1020_language_operation_protocol as protocol


SCRIPT_NAMES = (
    "phase1020_language_operation_protocol.py",
    "phase1020_language_operation_behavior.py",
    "phase1020_language_operation_scan.py",
    "phase1020_language_operation_finalize.py",
    "phase1020_language_operation_audit.py",
)
FORBIDDEN_SUFFIXES = (
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".gguf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
) -> None:
    checks.append({
        "name": name,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
    })


def audit() -> dict[str, Any]:
    checks = []
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_summary = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "summary.json"
    )
    add_check(
        checks,
        "protocol_identity",
        {
            "phase": prereg["phase"],
            "revision": prereg["protocol_revision"],
        },
        {"phase": protocol.PHASE, "revision": protocol.PROTOCOL_REVISION},
    )
    for model in protocol.MODELS:
        for mode in protocol.PROMPT_MODES:
            value = protocol_summary["models"][model][mode]
            add_check(
                checks,
                f"{model}_{mode}_protocol",
                {
                    "case_count": value["case_count"],
                    "unit_count": value["unit_count"],
                    "overlap": value["exact_split_overlap_count"],
                    "identity": value["all_identity_exact"],
                    "semantic": value["all_semantic_gold_checks"],
                },
                {
                    "case_count": 15200,
                    "unit_count": 3040,
                    "overlap": 0,
                    "identity": True,
                    "semantic": True,
                },
            )

    behavior_counts = {}
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
        behavior_counts[model] = len(rows)
        add_check(
            checks,
            f"{model}_behavior_complete",
            {"rows": len(rows), "summary_cases": summary["case_count"]},
            {"rows": 12160, "summary_cases": 12160},
        )

    scan_gate = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "scan_gate.json"
    )
    scan_counts = {}
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "formal_scan" / model / "summary.json"
        )
        scan_counts[model] = {
            "panel_count": summary["panel_count"],
            "unit_count": summary["unit_count"],
            "state_case_count": summary["state_case_count"],
        }
        add_check(
            checks,
            f"{model}_scan_arithmetic",
            {
                "state_case_count": summary["state_case_count"],
                "five_times_units": 5 * summary["unit_count"],
                "identity_maximum": summary.get("identity_maximum", 0.0),
                "prefix_branch_maximum": summary.get(
                    "prefix_branch_maximum", 0.0
                ),
            },
            {
                "state_case_count": 5 * summary["unit_count"],
                "five_times_units": 5 * summary["unit_count"],
                "identity_maximum": 0.0,
                "prefix_branch_maximum": 0.0,
            },
        )
        panel_summaries = list(
            (protocol.OUT_ROOT / "formal_scan" / model).glob(
                "*/*/*/summary.json"
            )
        )
        add_check(
            checks,
            f"{model}_panel_artifacts",
            len(panel_summaries),
            summary["panel_count"],
        )

    analysis = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "summary.json"
    )
    add_check(
        checks,
        "analysis_identity",
        {
            "phase": analysis["phase"],
            "digest": analysis["protocol_digest"],
            "attribution_rows": analysis["counts"][
                "translation_attribution_rows"
            ],
        },
        {
            "phase": protocol.PHASE,
            "digest": prereg["protocol_digest"],
            "attribution_rows": len(protocol.MODELS),
        },
    )

    all_files = [
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    ]
    markdown = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in all_files
        if path.suffix.lower() in (".md", ".markdown")
    ]
    forbidden = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in all_files
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    temporary = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in all_files
        if path.suffix.lower() in (".tmp", ".pending")
    ]
    add_check(
        checks,
        "result_hygiene",
        {
            "markdown": markdown,
            "forbidden": forbidden,
            "temporary": temporary,
        },
        {"markdown": [], "forbidden": [], "temporary": []},
    )
    script_manifest = []
    for name in SCRIPT_NAMES:
        path = ROOT / "tests" / "glm5" / name
        script_manifest.append({
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = [{
        "path": str(path.relative_to(protocol.OUT_ROOT)),
        "size": path.stat().st_size,
    } for path in sorted(all_files)]
    result = {
        "schema_version": "phase1020_audit.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "behavior_counts": behavior_counts,
        "scan_gate": scan_gate,
        "scan_counts": scan_counts,
        "script_manifest": script_manifest,
        "artifact_manifest_file_count": len(manifest),
        "artifact_manifest": manifest,
    }
    protocol.write_json(protocol.OUT_ROOT / "audit" / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1020 audit failed")
    return result


if __name__ == "__main__":
    audit()
