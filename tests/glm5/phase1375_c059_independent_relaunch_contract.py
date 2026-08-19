#!/usr/bin/env python3
"""Phase1375: independent C059 relaunch after C058's over-demanded behavior quota."""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1373_c058_dose_distance_group_campaign_contract as base

PHASE, CAMPAIGN = 1375, "C059"
PARENT = TESTS / "result/phase1374_c058_qwen_behavior_qualification"
C058 = TESTS / "result/phase1373_c058_dose_distance_group_campaign_contract"
C057 = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
OUT = TESTS / "result/phase1375_c059_independent_relaunch_contract"

FAMILIES = {
    "insect": ("ant", "bee", "beetle", "butterfly", "moth", "wasp",
               "termite", "mosquito", "grasshopper", "cricket", "dragonfly", "cockroach"),
    "fruit": ("apple", "banana", "orange", "grape", "mango", "peach",
              "pear", "cherry", "lemon", "plum", "melon", "apricot"),
    "furniture": ("chair", "table", "sofa", "desk", "bed", "cabinet",
                  "stool", "dresser", "bookshelf", "wardrobe", "bench", "cupboard"),
    "profession": ("doctor", "teacher", "lawyer", "nurse", "engineer", "farmer",
                   "chef", "dentist", "plumber", "architect", "pilot", "carpenter"),
}


def relabel(rows: list[dict], old: str, new: str) -> list[dict]:
    for row in rows:
        row["case_id"] = row["case_id"].replace(old, new)
    return rows


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1375 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c058_behavior_unqualified_before_hidden_access" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C058 negative behavior audit missing")

    base.FAMILIES = FAMILIES
    tok = base.tokenizer()
    concept_rows = base.concepts()
    active = relabel(base.active_cases(), "c058", "c059")
    status = relabel(base.status_cases(), "c058", "c059")
    compiled_active = base.compile_rows(tok, active, base.SYSTEM_ACTIVE)
    compiled_status = base.compile_rows(tok, status, base.SYSTEM_STATUS)
    pairs = base.candidate_pairs(active, status)
    old_words = set()
    for source in (C057, C058):
        old_words.update(row["word"] for row in core.load(source / "material/frozen_concept_graph.json")["concepts"])
    family_tokens = {family: tok.encode(" " + family, add_special_tokens=False) for family in FAMILIES}
    checks = {
        "parent_closed_audited": parent_audit.get("all_checks_passed"),
        "concept_count": len(concept_rows) == 48 and len({r["word"] for r in concept_rows}) == 48,
        "independent_of_c057_c058": not ({r["word"] for r in concept_rows} & old_words),
        "family_balance": all(len(words) == 12 for words in FAMILIES.values()),
        "partition_balance": all(sum(r["partition"] == p for r in concept_rows) == 16 for p in base.PARTITIONS),
        "semantic_uniqueness": all(r["sense"] and r["adjudication"] for r in concept_rows),
        "family_single_token": all(len(ids) == 1 for ids in family_tokens.values()),
        "active_balance": len(active) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status_balance": len(status) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "candidate_pairs": len(pairs) == 432,
        "controlled_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.")
                                      for r in active + status),
        "compiled_counts": len(compiled_active) == 864 and len(compiled_status) == 288,
        "family_spans_single": all(len(r["role_positions"]["family"]) == 1
                                   for r in compiled_active + compiled_status),
        "candidate_single_tokens": all(len(ids) == 1 for r in compiled_active + compiled_status
                                       for ids in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"target", "family", "query", "boundary"}
                           for r in compiled_active + compiled_status),
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c059.independent_concepts.v1", "families": FAMILIES,
        "partitions": {k: list(v) for k, v in base.PARTITIONS.items()}, "concepts": concept_rows,
    })
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5,
                        "target_only_quartet_upper_bound": 0.5, "family_only_quartet_upper_bound": 0.5,
                        "status_always_yes": 0.5, "status_always_no": 0.5},
        "independent_human_blind_review": False,
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = copy.deepcopy(core.load(C058 / "protocol/preregistration.json"))
    protocol.update({
        "phase": PHASE, "campaign": CAMPAIGN,
        "schema": "c059.independent_branched_dose_distance_groups_mediation.v1",
        "research_object": "independent-material same-recipient dose, geometry, coordinate-group, and mediation relaunch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    protocol["material"].update({
        "families": list(FAMILIES), "eligible_cases_per_cell": 6, "eligible_case_target": 216,
        "discovery_target": 72, "confirmation_target": 72, "lockbox_target": 72,
        "active_sha256": core.sha(OUT / "material/active_membership_cases.jsonl"),
        "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
        "pair_sha256": core.sha(OUT / "material/candidate_pairs.jsonl"),
    })
    protocol["branching"] = {
        "phase1376": "behavior qualification and 216-case balanced freeze",
        "phase1377": "known-truth and exact-shape calibration",
        "phase1378": "run both whole-state dose paths and full response geometry; save discovery field",
        "phase1379": "run every raw-coordinate route on confirmation plus lockbox independent of reverse result",
        "phase1380": "run early mediation if lambda1 positive sufficiency qualifies",
        "finish": "close after every independently eligible branch completes; no replacement route, layer, material, dose, or gate",
    }
    protocol["claim_boundary"]["allowed"] = "Qwen-specific independent-material dose response, response geometry, coordinate-group curve, and mediation boundary"
    protocol.pop("contract_sha256", None)
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1376_c059_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
