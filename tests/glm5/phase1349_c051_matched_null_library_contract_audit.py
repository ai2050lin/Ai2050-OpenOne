#!/usr/bin/env python3
"""Independent artifact audit for Phase1349/C051."""
from __future__ import annotations

import json
import py_compile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1349_c051_matched_null_library_contract"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    final = load(OUT / "analysis/final.json")
    protocol = load(OUT / "protocol/preregistration.json")
    pre = load(OUT / "audit/pre_model_material_zero_audit.json")
    material = rows(OUT / "material/frozen_cases.jsonl")
    grouped = defaultdict(list)
    for row in material:
        grouped[row["quartet_key"]].append(row)
    counts = Counter(row["panel"] for row in material)
    cell_balance = Counter((row["panel"], row["partition"], row["surface"], row["truth"]) for row in material)
    role = [row for row in material if row["panel"] == "role_bound_lexical"]
    checks = {
        "final_authorization": final.get("authorization") == "run_phase1350_c051_null_behavior",
        "contract_match": final.get("contract_sha256") == protocol.get("contract_sha256"),
        "preaudit": pre.get("all_checks_passed") and pre.get("passed") == pre.get("total"),
        "models": protocol.get("models") == ["qwen3", "glm4", "deepseek7b"],
        "required_models": protocol.get("required_common_models") == ["qwen3", "glm4"],
        "cases": len(material) == 3072,
        "panels": counts == {"role_bound_lexical": 1536, "explicit_status": 1536},
        "quartets": len(grouped) == 768 and all(len(v) == 4 for v in grouped.values()),
        "truth_balance": len(cell_balance) == 32 and all(v == 96 for v in cell_balance.values()),
        "role_presence": all(row["query_item"] in row["prompt"] and row["query_label"] in row["prompt"] for row in role),
        "role_negatives": Counter(row["mismatch_type"] for row in role)
            == {"both_match": 768, "category_mismatch": 384, "item_mismatch": 384},
        "constant_chance": pre["zero_models"]["always_yes"] == 0.5
            and pre["zero_models"]["always_no"] == 0.5,
        "single_role_below_gate": pre["zero_models"]["role_item_only"] == 0.75
            and pre["zero_models"]["role_category_only"] == 0.75,
        "no_hidden": protocol.get("hidden_state_boundary", "").startswith("No hidden state"),
        "stop_rule": "do not change" in protocol.get("stop_rule", ""),
        "compiled_counts": all(len(rows(OUT / f"compiled/{m}_cases.jsonl")) == 3072 for m in protocol["models"]),
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1349_c051_matched_null_library_contract.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1349, "campaign": "C051", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
