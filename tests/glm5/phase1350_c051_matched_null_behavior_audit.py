#!/usr/bin/env python3
"""Independent result audit for Phase1350/C051."""
from __future__ import annotations

import json
import py_compile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1350_c051_matched_null_behavior"
PARENT = TESTS / "result/phase1349_c051_matched_null_library_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    protocol = load(PARENT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    checks = {"contract": load(OUT / "protocol/execution_manifest.json")["contract_sha256"] == protocol["contract_sha256"]}
    recomputed = {}
    for model in MODELS:
        summary = load(OUT / f"analysis/{model}_summary.json")
        data = rows(OUT / f"raw/{model}_behavior.jsonl")
        executor = load(OUT / f"raw/{model}_executor.json")
        checks[f"{model}_count"] = len(data) == 3072
        checks[f"{model}_finite"] = executor["finite"] and executor["qualified"]
        model_ok = executor["qualified"]
        for panel in protocol["panels"]:
            selected = [r for r in data if r["panel"] == panel]
            quartets = defaultdict(list)
            for row in selected:
                quartets[row["quartet_key"]].append(row)
            acc = sum(r["correct"] for r in selected) / len(selected)
            panel_summary = summary["panels"][panel]
            checks[f"{model}_{panel}_accuracy"] = abs(acc - panel_summary["accuracy"]) <= 1e-12
            checks[f"{model}_{panel}_quartets"] = abs(
                sum(all(x["correct"] for x in q) for q in quartets.values()) / len(quartets)
                - panel_summary["quartet_all_correct_fraction"]
            ) <= 1e-12
            model_ok = model_ok and panel_summary["qualified"]
        checks[f"{model}_qualification"] = model_ok == summary["qualified"]
        recomputed[model] = model_ok
    common = recomputed["qwen3"] and recomputed["glm4"]
    checks["common_gate"] = common == final["required_common_models_passed"]
    checks["authorization"] = final["authorization"] == (
        "freeze_phase1351_c052_formation_contract" if common else "close_c051_null_library"
    )
    checks["compiled"] = True
    try:
        py_compile.compile(str(TESTS / "phase1350_c051_matched_null_behavior.py"), doraise=True)
    except Exception:
        checks["compiled"] = False
    result = {"phase": 1350, "campaign": "C051", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
