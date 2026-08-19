#!/usr/bin/env python3
"""Independent audit for Phase1347 C050 formation-clock contract."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1347_c050_formation_clock_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    material_audit = core.load(OUT / "audit/pre_model_material_zero_power_audit.json")
    concepts = core.load(OUT / "material/frozen_concept_graph.json")["concepts"]
    rows = core.rows(OUT / "material/frozen_cases.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[row["quartet_key"]].append(row)
    digest_object = dict(protocol)
    authorization = digest_object.pop("authorization")
    stored_digest = digest_object.pop("contract_sha256")
    checks = {
        "authorization": authorization == "run_phase1348_c050_behavior",
        "digest": stored_digest == core.digest(digest_object) == final["contract_sha256"],
        "hashes": protocol["material"]["graph_sha256"] == core.sha(OUT / "material/frozen_concept_graph.json")
        and protocol["material"]["cases_sha256"] == core.sha(OUT / "material/frozen_cases.jsonl"),
        "material_audit": material_audit["all_checks_passed"],
        "human_scope": material_audit["independent_human_blind_review"].startswith("not available"),
        "concepts": len(concepts) == 64 and len({row["word"] for row in concepts}) == 64,
        "cases": len(rows) == 3072 and len({row["case_id"] for row in rows}) == 3072,
        "panels": Counter(row["panel"] for row in rows)
        == {"core_membership": 1536, "label_only": 768, "generic_equality": 768},
        "quartets": len(groups) == 768
        and all([row["cell"] for row in group] == ["aa", "ab", "ba", "bb"] for group in groups.values()),
        "nested_partitions": protocol["material"]["partitions"]
        == ["prototype_discovery", "clock_selection", "confirmation", "holdout"],
        "selection_is_cross_word": protocol["formation_gate"]["prototype_partition"]
        != protocol["formation_gate"]["clock_partition"],
        "persistence": protocol["formation_gate"]["persistence_layers"] == 3,
        "nulls": protocol["formation_gate"]["clock_label_only_top1_max"] == 0.30
        and protocol["formation_gate"]["clock_generic_equality_top1_max"] == 0.30,
        "no_confirmation_selection": "never used for selection" in protocol["formation_gate"]["selection"],
        "stopping": "do not change" in protocol["stop_rule"],
        "compiled": all(len(core.rows(OUT / f"compiled/{model}_cases.jsonl")) == len(rows) for model in MODELS),
    }
    result = {
        "phase": 1347,
        "campaign": "C050",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
