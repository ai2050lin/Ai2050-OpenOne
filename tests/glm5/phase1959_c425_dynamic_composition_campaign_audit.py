#!/usr/bin/env python3
"""Independent audit for C415-C425 / Phase1949-1959."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1949_c415_c425_dynamic_composition_campaign as campaign


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    finals = {
        name: load(out / "analysis/final.json")
        for name, out in campaign.OUTS.items()
    }
    checks = {}
    checks["phase_sequence"] = [
        finals[f"C{i}"]["phase"] for i in range(415, 426)
    ] == list(range(1949, 1960))
    checks["all_closed"] = all(
        value["status"] == "closed" and value["all_checks_passed"]
        for value in finals.values()
    )
    c416 = finals["C416"]["headline"]
    checks["balanced_material"] = (
        c416["rows"] == 4608
        and max(abs(v - 0.5) for v in c416["zero_model_accuracies"].values())
        < 1e-12
        and max(abs(v - 0.5) for v in c416["per_axis_yes_frequency"].values())
        < 1e-12
    )
    checks["naturalness_boundary"] = c416["human_naturalness_review"] is False
    c417 = finals["C417"]["headline"]
    checks["behavior_accounted"] = c417["rows"] == 4608
    c418 = finals["C418"]["headline"]
    checks["field_branch"] = c418["field_ran"] == c417["field_eligible"]
    if c418["field_ran"]:
        checks["field_full_axis"] = c418["role_shape"][1:] == [38, 6, 2560]
    else:
        checks["field_full_axis"] = True
    c419 = finals["C419"]["headline"]
    checks["mobius_branch"] = c419["prediction_ran"] == c418["field_ran"]
    checks["mobius_accounted"] = (
        not c419["prediction_ran"] or c419["cells"] == 12
    )
    c420 = finals["C420"]["headline"]
    checks["dynamic_branch"] = c420["prediction_ran"] == c419["prediction_ran"]
    checks["dynamic_accounted"] = (
        not c420["prediction_ran"] or c420["cells"] == 12
    )
    c421 = finals["C421"]["headline"]
    checks["full_token_branch"] = c421["observation_ran"] == c418["field_ran"]
    if c421["observation_ran"]:
        checks["full_token_axis"] = (
            c421["pairs"] == 48 and c421["shape"][-1] == 2560
        )
    else:
        checks["full_token_axis"] = True
    c422 = finals["C422"]["headline"]
    checks["graph_rows"] = c422["rows"] == 3584
    c423 = finals["C423"]["headline"]
    checks["graph_branch"] = c423["field_ran"] == c422["graph_field_eligible"]
    c424 = finals["C424"]["headline"]
    joint = bool(
        set(c419["candidate_families"]) & set(c420["candidate_families"])
    )
    checks["writer_branch"] = (
        (not joint and not c424["writer_ran"])
        or (joint and "result" in c424)
    )
    c425 = finals["C425"]["headline"]
    visual = load(campaign.VISUAL)
    checks["visual_full_coordinates"] = (
        visual["schema"] == "c425.dynamic_composition_field.v1"
        and len(visual["rows"]) == c425["visual_rows"]
        and (
            (not c418["field_ran"] and not visual["rows"])
            or (
                bool(visual["rows"])
                and all(len(row["values"]) == 2560 for row in visual["rows"])
            )
        )
    )
    cleanup = load(campaign.OUTS["C425"] / "audit/cleanup.json")
    checks["cleanup"] = (
        len(cleanup) == c425["cleanup_files"]
        and all(row["sha256"] and row["removed"] for row in cleanup)
        and all(not (ROOT / row["path"]).exists() for row in cleanup)
    )
    checks["new_math_closed"] = c425["new_math_gate_passed"] is False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(failed)
    result = {
        "phase": 1959,
        "campaign": "C425",
        "audit": "independent",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": True,
        "strict_conclusion": (
            "Balanced factor behavior, Mobius prediction, dynamic-H000 prediction, "
            "token response, graph qualification, and writer evidence remain separately "
            "accounted. No fixed coordinate dictionary or new mathematics is inferred."
        ),
    }
    path = campaign.OUTS["C425"] / "audit/independent_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
