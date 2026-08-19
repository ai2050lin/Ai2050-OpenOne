#!/usr/bin/env python3
"""Independent frozen-contract audit for Phase1353/C053."""
from __future__ import annotations

import json
import py_compile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1353_c053_route_portfolio_contract"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    protocol = load(OUT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    pre = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    graph = load(OUT / "material/frozen_concept_graph.json")
    b1 = rows(OUT / "material/b1_binary_cases.jsonl")
    b3 = rows(OUT / "material/b3_choice_cases.jsonl")
    status = rows(OUT / "material/status_null_cases.jsonl")
    q1 = defaultdict(list)
    for row in b1:
        q1[row["quartet_key"]].append(row)
    q3 = defaultdict(list)
    for row in b3:
        q3[row["choice_group"]].append(row)
    checks = {
        "authorization": final["authorization"] == "run_phase1354_c053_behavior_routes",
        "hash": final["contract_sha256"] == protocol["contract_sha256"],
        "preaudit": pre["all_checks_passed"],
        "qwen_only": protocol["model"] == "qwen3",
        "concepts": len(graph["concepts"]) == 96 and len({x["word"] for x in graph["concepts"]}) == 96,
        "family_holdout": set(graph["seen_families"]).isdisjoint(graph["held_families"]),
        "b1": len(b1) == 1728 and len(q1) == 432 and all(len(q) == 4 for q in q1.values()),
        "b1_balance": Counter(x["truth"] for x in b1) == {True: 864, False: 864},
        "b3": len(b3) == 1728 and len(q3) == 864 and all({x["gold_position"] for x in q} == {0, 1} for q in q3.values()),
        "status": len(status) == 576 and Counter(x["truth"] for x in status) == {True: 288, False: 288},
        "compiled": all(len(rows(OUT / f"compiled/qwen3_{route}.jsonl")) == count
                        for route, count in (("B1_binary", 1728), ("B3_choice", 1728), ("N_status", 576))),
        "finite_routes": set(protocol["routes"]) == {"B1_absolute", "B2_relative", "B3_choice"},
        "route_level_stop": protocol["route_logic"]["route_fail"] == "eliminate only that route",
        "all_fail_stop": protocol["route_logic"]["all_behavior_routes_fail"] == "close C053 before hidden state",
        "hidden_condition": protocol["causal_gate"]["authorized_only_if"] == "B2 and shared relation field pass",
        "no_reduction": "No PCA" in protocol["parameter_boundary"],
        "human_scope": pre["independent_human_blind_review"] is False,
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1353_c053_route_portfolio_contract.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1353, "campaign": "C053", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
