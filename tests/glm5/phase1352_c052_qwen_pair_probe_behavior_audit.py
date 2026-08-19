#!/usr/bin/env python3
"""Independent result audit for Phase1352/C052."""
from __future__ import annotations

import json
import math
import py_compile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
PARENT = TESTS / "result/phase1351_c052_qwen_pair_probe_contract"
OUT = TESTS / "result/phase1352_c052_qwen_pair_probe_behavior"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def grouped_accuracy(data, key):
    return {
        value: sum(r["correct"] for r in data if str(r[key]) == value)
        / sum(str(r[key]) == value for r in data)
        for value in sorted({str(r[key]) for r in data})
    }


def main():
    protocol = load(PARENT / "protocol/preregistration.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_summary.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_behavior.jsonl")
    executor = load(OUT / "raw/qwen3_executor.json")
    checks = {
        "contract": manifest["contract_sha256"] == protocol["contract_sha256"],
        "model": manifest["model"] == "qwen3" and summary["model"] == "qwen3",
        "count": len(data) == 4608,
        "unique": len({r["case_id"] for r in data}) == 4608,
        "finite": executor["finite"] and all(math.isfinite(x) for r in data for x in r["scores"]),
        "executor": executor["qualified"] and executor["rank_agreement"] == 1.0
        and executor["max_abs_diff"] <= protocol["behavior_gate"]["executor_max_abs_diff_max"],
    }
    recomputed_ok = executor["qualified"]
    gate = protocol["behavior_gate"]
    for panel in protocol["material"]["panels"]:
        selected = [r for r in data if r["panel"] == panel]
        reported = summary["panels"][panel]
        quartets = defaultdict(list)
        for row in selected:
            quartets[row["quartet_key"]].append(row)
        quartet_score = sum(
            len(q) == 4 and all(x["correct"] for x in q) for q in quartets.values()
        ) / len(quartets)
        part = grouped_accuracy(selected, "partition")
        surface = grouped_accuracy(selected, "surface")
        truth = grouped_accuracy(selected, "truth")
        accuracy = sum(r["correct"] for r in selected) / len(selected)
        checks[f"{panel}_count"] = len(selected) == 1536 and len(quartets) == 384
        checks[f"{panel}_reported"] = (
            abs(accuracy - reported["accuracy"]) <= 1e-12
            and abs(quartet_score - reported["quartet_all_correct_fraction"]) <= 1e-12
            and part == reported["partition_accuracy"]
            and surface == reported["surface_accuracy"]
            and truth == reported["truth_accuracy"]
        )
        if panel == "core_membership":
            family = grouped_accuracy(selected, "target_family")
            panel_ok = (
                accuracy >= gate["core_accuracy_min"]
                and min(part.values()) >= gate["core_partition_min"]
                and min(surface.values()) >= gate["core_surface_min"]
                and min(family.values()) >= gate["core_family_min"]
                and min(truth.values()) >= gate["core_truth_min"]
                and quartet_score >= gate["core_quartet_all_min"]
            )
            checks["core_family_reported"] = family == reported["family_accuracy"]
        else:
            panel_ok = (
                accuracy >= gate["control_accuracy_min"]
                and min(part.values()) >= gate["control_partition_min"]
                and min(surface.values()) >= gate["control_surface_min"]
                and min(truth.values()) >= gate["control_truth_min"]
                and quartet_score >= gate["control_quartet_all_min"]
            )
        checks[f"{panel}_gate"] = panel_ok == reported["qualified"]
        recomputed_ok = recomputed_ok and panel_ok
    checks["qualification"] = recomputed_ok == summary["qualified"] == final["behavior_gate_passed"]
    expected = "run_phase1353_c052_qwen_full_probe" if recomputed_ok else "close_c052_behavior"
    checks["authorization"] = final["authorization"] == expected
    checks["compiled"] = True
    try:
        py_compile.compile(str(TESTS / "phase1352_c052_qwen_pair_probe_behavior.py"), doraise=True)
    except Exception:
        checks["compiled"] = False
    result = {
        "phase": 1352,
        "campaign": "C052",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
