#!/usr/bin/env python3
"""Independent audit for C426-C433 / Phase1960-1967."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1960_c426_c433_axis_lockbox_campaign as campaign


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    finals = {
        name: load(path / "analysis/final.json")
        for name, path in campaign.OUTS.items()
    }
    checks = {}
    checks["phase_sequence"] = [
        finals[f"C{i}"]["phase"] for i in range(426, 434)
    ] == list(range(1960, 1968))
    checks["all_closed"] = all(
        value["status"] == "closed" and value["all_checks_passed"]
        for value in finals.values()
    )
    c427 = finals["C427"]["headline"]
    checks["material"] = (
        c427["rows"] == 3456
        and c427["yes_frequency"] == 0.5
        and c427["first_position_frequency"] == 0.5
        and c427["human_naturalness_review"] is False
    )
    c428 = finals["C428"]["headline"]
    checks["axis_accounting"] = set(c428["axis_results"]) == set(campaign.AXES)
    c429 = finals["C429"]["headline"]
    checks["field_branch"] = c429["field_ran"] == bool(c428["eligible_axes"])
    if c429["field_ran"]:
        checks["field_axis"] = c429["role_shape"][1:] == [38, 6, 2560]
    else:
        checks["field_axis"] = True
    c430 = finals["C430"]["headline"]
    checks["mobius_branch"] = c430["prediction_ran"] == c429["field_ran"]
    c431 = finals["C431"]["headline"]
    checks["dynamic_branch"] = c431["prediction_ran"] == c430["prediction_ran"]
    c432 = finals["C432"]["headline"]
    joint = bool(
        set(c430["candidate_families"]) & set(c431["candidate_families"])
    )
    checks["writer_branch"] = (
        (not joint and not c432["writer_ran"])
        or (joint and "result" in c432)
    )
    c433 = finals["C433"]["headline"]
    visual = load(campaign.VISUAL)
    checks["visual"] = (
        visual["schema"] == "c433.axis_lockbox_field.v1"
        and len(visual["rows"]) == c433["visual_rows"]
        and (
            (not c429["field_ran"] and not visual["rows"])
            or (
                bool(visual["rows"])
                and all(len(row["values"]) == 2560 for row in visual["rows"])
            )
        )
    )
    cleanup = load(campaign.OUTS["C433"] / "audit/cleanup.json")
    checks["cleanup"] = (
        len(cleanup) == c433["cleanup_files"]
        and all(row["sha256"] and row["removed"] for row in cleanup)
        and all(not (ROOT / row["path"]).exists() for row in cleanup)
    )
    checks["new_math_closed"] = c433["new_math_gate_passed"] is False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(failed)
    result = {
        "phase": 1967,
        "campaign": "C433",
        "audit": "independent",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": True,
        "strict_conclusion": (
            "Axis behavior qualification, all-row Mobius observation, dynamic-H000 "
            "forecast, and state writing are separately accounted. No fixed "
            "coordinate dictionary or new mathematics is inferred."
        ),
    }
    path = campaign.OUTS["C433"] / "audit/independent_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
