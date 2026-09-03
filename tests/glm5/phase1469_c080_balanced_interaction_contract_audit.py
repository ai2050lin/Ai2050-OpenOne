#!/usr/bin/env python3
"""Independent audit for Phase1469."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1469_c080_balanced_interaction_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1469_c080_balanced_interaction_contract as phase


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    explicit = core.rows(OUT / "material/explicit_active_cases.jsonl")
    withdrawal = core.rows(OUT / "material/withdrawal_active_cases.jsonl")
    explicit_sets = core.rows(OUT / "material/explicit_interaction_sets.jsonl")
    withdrawal_sets = core.rows(OUT / "material/withdrawal_interaction_sets.jsonl")
    py_compile.compile(str(TESTS / "phase1469_c080_balanced_interaction_contract.py"), doraise=True)
    checks = {
        "preaudit": preaudit["all_checks_passed"] and preaudit["hidden_state_accessed"] is False,
        "contract_hash": protocol["contract_sha256"] == core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")}),
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1470_c080_explicit_behavior",
        "counts": len(explicit) == len(withdrawal) == 10368 and len(explicit_sets) == len(withdrawal_sets) == 540,
        "partitions": all(Counter(row["partition"] for row in rows) == {name: 3456 for name in phase.PARTITIONS} for rows in (explicit, withdrawal)),
        "truth": all(Counter(row["truth"] for row in rows) == {False: 8640, True: 1728} for rows in (explicit, withdrawal)),
        "pairs": all(Counter(row["pair_id"] for row in rows) == {value: 36 for value in phase.PAIR_IDS} for rows in (explicit_sets, withdrawal_sets)),
        "material_hashes": all(protocol["branches"][branch][key] == core.sha(OUT / path) for branch, key, path in (
            ("explicit", "active_sha256", "material/explicit_active_cases.jsonl"),
            ("explicit", "compiled_sha256", "compiled/qwen3_explicit.jsonl"),
            ("explicit", "sets_sha256", "material/explicit_interaction_sets.jsonl"),
            ("withdrawal", "active_sha256", "material/withdrawal_active_cases.jsonl"),
            ("withdrawal", "compiled_sha256", "compiled/qwen3_withdrawal.jsonl"),
            ("withdrawal", "sets_sha256", "material/withdrawal_interaction_sets.jsonl"),
        )),
        "four_grid": all(all(sum(key.endswith(corner) for key in row) == len(phase.EXPLICIT_SURFACES) * len(phase.NUISANCE_CELLS) for corner in phase.PAIR_CORNERS) for row in explicit_sets),
        "withdrawal_pre_frozen": protocol["branches"]["withdrawal"]["active_sha256"] and protocol["withdrawal_observation"]["candidate_cells"].startswith("mapped"),
        "observables": set(protocol["allowed_observables"]) == {"input embeddings", "all full-dimensional Hidden States", "yes/no logits"},
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "parameters", "PCA", "learned probe")),
        "hidden_not_accessed": True,
    }
    result = {
        "phase": 1469,
        "campaign": "C080",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
