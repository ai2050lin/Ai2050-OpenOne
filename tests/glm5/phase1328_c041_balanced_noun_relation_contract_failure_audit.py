#!/usr/bin/env python3
"""Adjudicate the Phase1328 scaffold implementation failure without rerunning it."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1328_c041_balanced_noun_relation_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    final = load(OUT / "analysis/final.json")
    machine = load(OUT / "audit/tokenizer_zero_model_audit.json")
    original = load(OUT / "audit/independent_final_audit.json")
    source = rows(OUT / "material/frozen_behavior_cases.jsonl")
    sets = defaultdict(list)
    for row in source:
        sets[row["semantic_set"]].append(row)
    independent_exact = all(
        len(values) == 4
        and Counter(row["gold_position"] for row in values) == Counter({0: 2, 1: 2})
        and Counter(row["surface"] for row in values)
        == Counter({"reference_family": 2, "vocabulary_kind": 2})
        for values in sets.values()
    )
    scaffold = (ROOT / "tests/glm5/phase1327_c040_cross_model_noun_relation_contract.py").read_text(encoding="utf-8")
    checks = {
        "formal_stop_unchanged": final["all_gates_passed"] is False
                                 and final["authorization"] == "stop_c040_before_model",
        "original_audit_only_final_failed": {name for name, passed in original["checks"].items() if not passed}
                                             == {"final"},
        "material_actually_exact": independent_exact and len(sets) == 144,
        "recorded_scaffold_value_wrong": machine["exact_mirrored_pairing"] is False,
        "scaffold_bug_present": "for values in defaultdict(list," in scaffold and ").items()" in scaffold,
        "all_zero_models_pass": machine["zero_models"]["maximum_nonsemantic_accuracy"] <= 0.60
                                and machine["zero_models"]["candidate_identity_majority"] == 0.5,
        "no_model_or_hidden": not (OUT / "raw").exists() and not (OUT / "field").exists(),
    }
    output = {"phase": 1328, "campaign": "C041", "audit_type": "pre_model_scaffold_erratum",
              "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()),
              "finding": "The inherited exact-pairing implementation iterated dict.items() as one value.",
              "formal_result_changed": False,
              "authorization": "close_c041_and_permit_fresh_non_scaffold_contract" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_failure_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
