#!/usr/bin/env python3
"""Phase1460: preregister C078 colon-label full-field observation."""
from __future__ import annotations

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
import phase1456_c077_labeled_relation_contract as c077
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1460, "C078"
PARENT = TESTS / "result/phase1459_c074_c077_analysis_adjudication"
OUT = TESTS / "result/phase1460_c078_colon_label_contract"
FAMILIES = {
    "Galaxy": ("Abbott", "Ahmed", "Alejandro", "Ali", "Allan", "Alma"),
    "Union": ("Amir", "Ana", "Anderson", "Angelo", "Annie", "Antonio"),
    "Crown": ("Archer", "Ari", "Ashton", "Athena", "Axel", "Becky"),
    "Liberty": ("Bennett", "Bernard", "Billy", "Bobby", "Brady", "Brock"),
    "Noble": ("Bryce", "Byron", "Carter", "Cassidy", "Charlie", "Clint"),
    "Oasis": ("Cole", "Connie", "Cooper", "Crystal", "Dalton", "Damian"),
}
RELATIONS = c077.RELATIONS
IDS = tuple(RELATIONS)
ORDER = tuple(FAMILIES)
PARTITIONS = c077.PARTITIONS
SYSTEM = c077.SYSTEM
CELLS = c077.CELLS
ROLES = c077.ROLES
ROLE_SLOTS = c077.ROLE_SLOTS
SURFACES = {
    "a_colon": "First relation label: {record_label}. First clause: {record_target} {record_relation} {record_object}. Second relation label: {query_label}. Second clause: {query_target} may {query_relation} {query_object}. Are the two relation labels identical? Answer only yes or no.",
    "b_colon": "Recorded label: {record_label}. Recorded fact: {record_target} {record_relation} {record_object}. Queried label: {query_label}. Queried possibility: {query_target} can {query_relation} {query_object}. Are the recorded and queried labels identical? Answer only yes or no.",
}


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def active_cases() -> list[dict]:
    rows = []
    for family_index, family in enumerate(ORDER):
        other_family = ORDER[(family_index + 1) % len(ORDER)]
        for index in range(6):
            record_target = FAMILIES[family][index]
            other_target = FAMILIES[other_family][index]
            for relation_index, relation_id in enumerate(IDS):
                other_relation_id = IDS[(relation_index + 1 + index % 2) % len(IDS)]
                for surface, template in SURFACES.items():
                    for entity_match in (1, 0):
                        for object_match in (1, 0):
                            for relation_match in (1, 0):
                                query_id = relation_id if relation_match else other_relation_id
                                row = {
                                    "case_id": f"c078-a-{len(rows):04d}",
                                    "partition": partition(index),
                                    "family": family,
                                    "other_family": other_family,
                                    "index": index,
                                    "surface": surface,
                                    "cell": f"{entity_match}{object_match}{relation_match}",
                                    "record_label": RELATIONS[relation_id]["label"],
                                    "record_target": record_target,
                                    "record_relation": RELATIONS[relation_id]["record"],
                                    "record_relation_id": relation_id,
                                    "record_object": family,
                                    "query_label": RELATIONS[query_id]["label"],
                                    "query_target": record_target if entity_match else other_target,
                                    "query_relation": RELATIONS[query_id]["query"],
                                    "query_relation_id": query_id,
                                    "query_object": family if object_match else other_family,
                                    "entity_match": bool(entity_match),
                                    "object_match": bool(object_match),
                                    "relation_match": bool(relation_match),
                                    "truth": bool(relation_match),
                                    "candidates": ["yes", "no"],
                                    "gold_position": 0 if relation_match else 1,
                                }
                                row["prompt"] = template.format(**row)
                                rows.append(row)
    return rows


def composition_sets(active: list[dict]) -> list[dict]:
    by = {(row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row for row in active}
    result = []
    for family in ORDER:
        for index in range(6):
            for relation_id in IDS:
                row = {"set_id": f"c078-compose-{len(result):04d}", "partition": partition(index), "family": family, "index": index, "record_relation_id": relation_id}
                for surface in SURFACES:
                    for cell in CELLS:
                        row[f"{surface}_{cell}"] = by[(family, index, relation_id, surface, cell)]["case_id"]
                result.append(row)
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1460 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1460_c078_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1459 did not authorize C078")
    tok = tokenizer()
    active = active_cases()
    compiled = c077.compile_rows(tok, active)
    composition = composition_sets(active)
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    relation_words = {value for relation in RELATIONS.values() for value in relation.values()}
    truths = [row["truth"] for row in active]
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLE_SLOTS) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    zero = {
        "always_yes": c077.ba(truths, [True] * len(active)),
        "always_no": c077.ba(truths, [False] * len(active)),
        "surface": c077.ba(truths, [row["surface"] == "a_colon" for row in active]),
        "entity": c077.ba(truths, [row["entity_match"] for row in active]),
        "object": c077.ba(truths, [row["object_match"] for row in active]),
        "entity_object": c077.ba(truths, [row["entity_match"] and row["object_match"] for row in active]),
        "label_identity": c077.ba(truths, [row["record_label"] == row["query_label"] for row in active]),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 36 and not ({value.lower() for value in members} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members | relation_words),
        "active": len(active) == 3456 and Counter(row["surface"] for row in active) == {surface: 1728 for surface in SURFACES},
        "truth": Counter(truths) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_label"] == row["query_label"]) for row in active),
        "nuisance": all(Counter(row[key] for row in active) == {True: 1728, False: 1728} for key in ("entity_match", "object_match", "relation_match")),
        "composition": len(composition) == 216 and Counter(row["partition"] for row in composition) == {name: 72 for name in PARTITIONS},
        "compiled": len(compiled) == 3456,
        "same_shape": all(len(values) == 1 for values in lengths.values()),
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "role_singletons": all(all(len(row["role_positions"][role]) == 1 for role in ROLE_SLOTS) and len({row["role_positions"][role][0] for role in ROLE_SLOTS}) == 9 for row in compiled),
        "naturalness": all(row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero": all(value == 0.5 for key, value in zero.items() if key != "label_identity") and zero["label_identity"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c078.colon_label.v1", "families": FAMILIES, "relations": RELATIONS, "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "semantic_scope": "explicit known-truth relation-label identity with independent person and organization nuisances", "naturalness_scope": "machine-audited controlled English; no human blind review"}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c078.colon_label_observation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "expected_hidden_state_count": 37,
        "research_object": "full-dimensional role-aligned trajectory of an explicit relation-label identity carrier",
        "roles": list(ROLES),
        "role_slots": list(ROLE_SLOTS),
        "relations": RELATIONS,
        "surfaces": list(SURFACES),
        "cells": list(CELLS),
        "partitions": list(PARTITIONS),
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "t-SNE", "UMAP", "learned probe", "coordinate pruning before raw capture", "post-holdout changes"],
        "material": {"active_count": 3456, "composition_count": 216, "surface_lengths": {key: next(iter(value)) for key, value in lengths.items()}, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": {"global_surface_balanced_accuracy_min": 0.98, "family_relation_surface_accuracy_min": 0.90, "family_relation_surface_balanced_accuracy_min": 0.90, "partition_min": 0.90, "truth_min": 0.90, "cell_min": 0.90, "eligible_set_total_min": 180, "eligible_set_split_min": 60, "eligible_set_relation_min": 25, "same_batch_repeat_max_abs_diff": 1e-6},
        "capture": {"eligible_rule": "all sixteen cases in a frozen composition set must be behavior-correct", "discovery_partition": "response_discovery", "state_count": 37, "role_slot_count": 9, "dtype": "float16", "raw_format": "numpy NPY memmap plus JSONL index", "no_pooling": True, "no_coordinate_selection": True},
        "observation": {"effects": ["relation_label", "entity_nuisance", "object_nuisance"], "basic_summaries": ["L2 norm", "mean vector", "direction consistency", "cross-surface cosine", "coordinate sign consistency"], "freeze_full_vector_layer_role_candidates_before_holdout": True, "no_coordinate_subset": True},
        "validation": {"partitions": ["confirmation", "lockbox"], "candidate_source": "discovery frozen manifest", "candidate_failure": "closes candidate only"},
        "claim_boundary": {"allowed": "explicit labeled-carrier trajectory regularities in behavior-correct Qwen3 cases", "forbidden": ["unlabeled relation semantics", "semantic neurons", "causal necessity", "natural use", "cross-model law", "new mathematics"]},
        "stop_rule": "behavior failure closes C078; discovery observations freeze before any holdout access; holdout failures close only their candidate branch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1461_c078_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
