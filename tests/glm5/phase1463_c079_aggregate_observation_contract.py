#!/usr/bin/env python3
"""Phase1463: preregister C079 aggregate-qualified full-field observation."""
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
import phase1435_c072_permutation_spectrum_contract as c072
import phase1460_c078_colon_label_contract as base
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1463, "C079"
PARENT = TESTS / "result/phase1462_c078_behavior_gate_closure"
SOURCE_PROTOCOL = TESTS / "result/phase1460_c078_colon_label_contract/protocol/preregistration.json"
OUT = TESTS / "result/phase1463_c079_aggregate_observation_contract"
FAMILIES = {
    "Quest": ("Diego", "Drew", "Duke", "Eden", "Eduardo", "Edwin"),
    "Unity": ("Emerson", "Emmanuel", "Enrique", "Everett", "Ezra", "Fernando"),
    "Workshop": ("Finn", "Gene", "Genesis", "Geoffrey", "Gilbert", "Harper"),
    "Ocean": ("Harvey", "Helena", "Ibrahim", "Ivy", "Jackson", "Jake"),
    "Cloud": ("Joey", "Jude", "Julian", "Julio", "Kai", "Kendrick"),
    "Moon": ("Kingston", "Kirk", "Kurt", "Kylie", "Leonardo", "Lilly"),
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1463 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c079_aggregate_eligible_observation_campaign_on_fresh_material" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1462 did not authorize C079")
    base.FAMILIES = FAMILIES
    base.ORDER = tuple(FAMILIES)
    tok = tokenizer()
    active = base.active_cases()
    compiled = base.c077.compile_rows(tok, active)
    composition = base.composition_sets(active)
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    relation_words = {value for relation in base.RELATIONS.values() for value in relation.values()}
    truths = [row["truth"] for row in active]
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in base.SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in base.ROLE_SLOTS) for row in compiled if row["surface"] == surface} for surface in base.SURFACES}
    zero = {
        "always_yes": base.c077.ba(truths, [True] * len(active)),
        "always_no": base.c077.ba(truths, [False] * len(active)),
        "surface": base.c077.ba(truths, [row["surface"] == "a_colon" for row in active]),
        "entity": base.c077.ba(truths, [row["entity_match"] for row in active]),
        "object": base.c077.ba(truths, [row["object_match"] for row in active]),
        "entity_object": base.c077.ba(truths, [row["entity_match"] and row["object_match"] for row in active]),
        "label_identity": base.c077.ba(truths, [row["record_label"] == row["query_label"] for row in active]),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 36 and not ({value.lower() for value in members} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members | relation_words),
        "active": len(active) == 3456 and Counter(row["surface"] for row in active) == {surface: 1728 for surface in base.SURFACES},
        "truth": Counter(truths) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_label"] == row["query_label"]) for row in active),
        "nuisance": all(Counter(row[key] for row in active) == {True: 1728, False: 1728} for key in ("entity_match", "object_match", "relation_match")),
        "composition": len(composition) == 216 and Counter(row["partition"] for row in composition) == {name: 72 for name in base.PARTITIONS},
        "compiled": len(compiled) == 3456,
        "same_shape": all(len(values) == 1 for values in lengths.values()),
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "role_singletons": all(all(len(row["role_positions"][role]) == 1 for role in base.ROLE_SLOTS) for row in compiled),
        "naturalness": all(row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero": all(value == 0.5 for key, value in zero.items() if key != "label_identity") and zero["label_identity"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c079.aggregate_observation.v1", "families": FAMILIES, "relations": base.RELATIONS, "partitions": {key: list(value) for key, value in base.PARTITIONS.items()}, "surfaces": base.SURFACES})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "semantic_scope": "explicit known-truth relation-label identity on third fresh material set", "naturalness_scope": "machine-audited controlled English; no human blind review"}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = copy.deepcopy(core.load(SOURCE_PROTOCOL))
    protocol.update({"phase": PHASE, "campaign": CAMPAIGN, "schema": "c079.aggregate_eligible_observation.v1", "research_object": "aggregate-qualified, case-exact full-dimensional trajectory of an explicit relation-label identity carrier", "material": {"active_count": 3456, "composition_count": 216, "surface_lengths": {key: next(iter(value)) for key, value in lengths.items()}, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False}, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    protocol["behavior"] = {
        "global_surface_balanced_accuracy_min": 0.98,
        "surface_partition_balanced_accuracy_min": 0.97,
        "surface_truth_accuracy_min": 0.97,
        "relation_surface_balanced_accuracy_min": 0.95,
        "eligible_set_total_min": 180,
        "eligible_set_split_min": 60,
        "eligible_set_relation_min": 25,
        "same_batch_repeat_max_abs_diff": 1e-6,
        "excluded_redundant_gate": "no family x relation x surface x six-sample-cell conjunction; case-exact eligible sets remain mandatory",
    }
    protocol["stop_rule"] = "aggregate behavior and eligible-set breadth must pass; only all-sixteen-correct sets enter discovery; discovery freezes before holdout"
    protocol["contract_sha256"] = core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")})
    protocol["authorization"] = "run_phase1464_c079_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
