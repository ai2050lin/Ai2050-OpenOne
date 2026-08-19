#!/usr/bin/env python3
"""Independent result audit for Phase1354/C053."""
from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1354_c053_behavior_route_competition"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    protocol = load(CONTRACT / "protocol/preregistration.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_summary.json")
    final = load(OUT / "analysis/final.json")
    executor = load(OUT / "raw/qwen3_executor.json")
    expected_counts = {"B1_binary": 1728, "B3_choice": 1728, "N_status": 576}
    checks = {
        "contract": manifest["contract_sha256"] == protocol["contract_sha256"],
        "model": manifest["model"] == "qwen3" and summary["model"] == "qwen3",
        "executor": executor["qualified"] and all(x["finite"] and x["rank_agreement"] == 1.0
                                                   and x["max_abs_diff"] <= 1e-6
                                                   for x in executor["routes"].values()),
    }
    for route, count in expected_counts.items():
        data = rows(OUT / f"raw/{route}_behavior.jsonl")
        checks[f"{route}_count"] = len(data) == count and len({x["case_id"] for x in data}) == count
        checks[f"{route}_finite"] = all(math.isfinite(v) for x in data for v in x["scores"])
    reported = summary["summaries"]
    checks["B1_recomputed"] = abs(reported["B1_absolute"]["accuracy"] -
                                  sum(x["correct"] for x in rows(OUT / "raw/B1_binary_behavior.jsonl")) / 1728) <= 1e-12
    checks["B3_recomputed"] = abs(reported["B3_choice"]["accuracy"] -
                                  sum(x["correct"] for x in rows(OUT / "raw/B3_choice_behavior.jsonl")) / 1728) <= 1e-12
    checks["status_recomputed"] = abs(reported["N_status"]["accuracy"] -
                                      sum(x["correct"] for x in rows(OUT / "raw/N_status_behavior.jsonl")) / 576) <= 1e-12
    qualified = [route for route, passed in summary["route_qualified"].items() if passed]
    checks["qualified_routes"] = qualified == final["qualified_routes"]
    fields = (["quartet_interaction_field"] if "B2_relative" in qualified else []) + \
             (["choice_order_invariance_field"] if "B3_choice" in qualified else [])
    checks["authorized_fields"] = fields == final["authorized_fields"]
    checks["authorization"] = final["authorization"] == (
        "run_phase1355_c053_fields" if fields else "close_c053_after_behavior_routes"
    )
    checks["script_compiles"] = True
    try:
        py_compile.compile(str(TESTS / "phase1354_c053_behavior_route_competition.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1354, "campaign": "C053", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
