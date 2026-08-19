#!/usr/bin/env python3
"""Independent result audit for Phase1356/C053."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1356_c053_typed_relation_causal"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_summary.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_typed_causal.jsonl")
    correct = [x["gains"]["same_family_true_donor"] for x in data]
    different = [x["gains"]["different_family_true_donor"] for x in data]
    false = [x["gains"]["same_family_false_donor"] for x in data]
    advantage = [c - max(d, f) for c, d, f in zip(correct, different, false)]
    metrics = summary["metrics"]
    checks = {
        "count": len(data) == manifest["recipient_count"],
        "finite": all(math.isfinite(v) for x in data for d in (x["margins"], x["gains"]) for v in d.values()),
        "gain": abs(statistics.median(correct) - metrics["correct_gain_median"]) <= 1e-12,
        "direction": abs(sum(x > 0 for x in correct) / len(correct) - metrics["correct_direction_fraction"]) <= 1e-12,
        "advantage": abs(statistics.median(advantage) - metrics["correct_over_wrong_median"]) <= 1e-12,
        "self": abs(max(abs(x["gains"]["self"]) for x in data) - metrics["self_max_abs_diff"]) <= 1e-12,
        "qualification": summary["qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == ("close_c053_with_typed_causal_candidate"
                                                      if summary["qualified"]
                                                      else "close_c053_at_causal_selectivity_boundary"),
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1356_c053_typed_relation_causal.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1356, "campaign": "C053", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
