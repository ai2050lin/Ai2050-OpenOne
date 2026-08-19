#!/usr/bin/env python3
"""Independent audit for Phase1365 C056 known-truth path camera."""
from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1365_c056_planted_hidden_path_camera"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    summary = load(OUT / "analysis/known_truth_summary.json")
    final = load(OUT / "analysis/final.json")
    records = rows(OUT / "raw/known_truth_path_records.jsonl")
    paths = sorted(summary["metrics"])
    checks = {
        "record_count": len(records) == 1280,
        "split_count": {row["split"] for row in records} == {"discovery", "confirmation"},
        "gauge_count": {row["gauge"] for row in records} == {0, 1, 2, 3},
        "path_coverage": all(sum(row["path"] == path for row in records) == 256 for path in paths),
        "finite": all(math.isfinite(value) for row in records for value in row["output_gain"].values()),
        "identity": all(values["identity_passed"] for values in summary["metrics"].values()),
        "exact_topology": summary["predicted_positive"] == summary["expected_positive"],
        "positive_set": set(summary["predicted_positive"]) == {"family_early", "family_mid", "query_late"},
        "negative_set": all(not summary["metrics"][name]["qualified"] for name in ("family_late", "query_mid")),
        "authorization": final["authorization"] == "run_phase1366_c056_qwen_path_observation",
    }
    py_compile.compile(str(T / "phase1365_c056_planted_hidden_path_camera.py"), doraise=True)
    checks["script_compiles"] = True
    result = {
        "phase": 1365, "campaign": "C056", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
