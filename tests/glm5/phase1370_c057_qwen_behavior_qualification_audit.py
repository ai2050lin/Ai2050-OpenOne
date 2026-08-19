#!/usr/bin/env python3
"""Independent audit for Phase1370 C057 behavior qualification."""
from __future__ import annotations

import json
import math
import py_compile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
OUT = TESTS / "result/phase1370_c057_qwen_behavior_qualification"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(CONTRACT / "protocol/preregistration.json")
    summary = load(OUT / "analysis/qwen3_behavior_summary.json")
    final = load(OUT / "analysis/final.json")
    active = rows(OUT / "raw/active_behavior.jsonl")
    status = rows(OUT / "raw/status_behavior.jsonl")
    selected = rows(OUT / "material/eligible_pairs.jsonl")
    quartets = defaultdict(list)
    for row in active:
        quartets[row["quartet_key"]].append(row["correct"])
    cells = defaultdict(list)
    for row in selected:
        cells[(row["target_family"], row["partition"], row["surface"])].append(row)
    checks = {
        "record_counts": len(active) == 864 and len(status) == 288,
        "finite": all(math.isfinite(row["margin"]) for row in active + status),
        "active_accuracy": abs(sum(row["correct"] for row in active) / len(active) - summary["active"]["accuracy"]) < 1e-12,
        "quartet": abs(sum(all(v) for v in quartets.values()) / len(quartets) - summary["active"]["quartet_all_fraction"]) < 1e-12,
        "selected": len(selected) == 288 and len(cells) == 48 and set(map(len, cells.values())) == {6},
        "numeric": summary["numeric_same_shape_max_abs_diff"] <= protocol["behavior"]["same_shape_repeat_max_abs_diff"],
        "checks_consistent": summary["behavior_qualified"] == all(summary["checks"].values()),
        "final_consistent": final["behavior_qualified"] == summary["behavior_qualified"],
        "authorization": final["authorization"] == ("run_phase1371_c057_instrument_calibration"
                                                       if summary["behavior_qualified"] else
                                                       "close_c057_behavior_unqualified_before_hidden_access"),
    }
    py_compile.compile(str(TESTS / "phase1370_c057_qwen_behavior_qualification.py"), doraise=True)
    py_compile.compile(str(TESTS / "phase1370_c057_qwen_behavior_qualification_audit.py"), doraise=True)
    checks["scripts_compile"] = True
    result = {"phase": 1370, "campaign": "C057", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
