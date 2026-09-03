#!/usr/bin/env python3
"""Phase1450: preregister C075 observation-first full-layer relation atlas."""
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
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1450, "C075"
PARENT = TESTS / "result/phase1449_c074_campaign_closure"
C074 = TESTS / "result/phase1448_c074_directional_domain_map"
OUT = TESTS / "result/phase1450_c075_full_field_atlas_contract"
FAMILIES = {
    "Aurora": ("Wade", "Wallace", "Wyatt", "Zoe", "Vanessa", "Veronica"),
    "Crest": ("Victoria", "Vincent", "Virginia", "Walter", "Warren", "Wayne"),
    "Flint": ("Wesley", "Whitney", "William", "Alec", "Alexa", "Alison"),
    "Haven": ("Andre", "Arnold", "Bailey", "Bernie", "Brett", "Cassandra"),
    "Ivory": ("Chase", "Chelsea", "Cindy", "Clark", "Cliff", "Damon"),
    "Keystone": ("Dana", "Devin", "Edmund", "Emanuel", "Erica", "Ernest"),
}
RELATIONS = ("joined", "supported", "visited", "contacted", "selected", "praised")
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 2), "confirmation": range(2, 4), "lockbox": range(4, 6)}
SYSTEM = "Compare the two supplied clauses. Answer yes exactly when the person, relation, and organization are all identical. Output exactly yes or no."
SURFACES = {
    "a_evidence_first": "Evidence statement: {record_target} {record_relation} {record_object}. Comparison statement: {query_target} {query_relation} {query_object}. Do the person, relation, and organization all match? Answer only yes or no.",
    "b_evidence_first": "Registry record: {record_target} {record_relation} {record_object}. Check this claim against it: {query_target} {query_relation} {query_object}. Are all three fields identical? Reply only yes or no.",
}
ROLES = ("record_target", "record_relation", "record_object", "query_target", "query_relation", "query_object")
ROLE_SLOTS = ROLES + ("boundary",)
CELLS = tuple(f"{e}{o}{r}" for e in (1, 0) for o in (1, 0) for r in (1, 0))


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def active_cases() -> list[dict]:
    rows = []
    for fi, family in enumerate(ORDER):
        other_family = ORDER[(fi + 1) % len(ORDER)]
        for index in range(6):
            record_target = FAMILIES[family][index]
            other_target = FAMILIES[other_family][index]
            for ri, record_relation in enumerate(RELATIONS):
                other_relation = RELATIONS[(ri + 1 + index % 2) % len(RELATIONS)]
                for surface, template in SURFACES.items():
                    for entity_match in (1, 0):
                        for object_match in (1, 0):
                            for relation_match in (1, 0):
                                cell = f"{entity_match}{object_match}{relation_match}"
                                query_target = record_target if entity_match else other_target
                                query_object = family if object_match else other_family
                                query_relation = record_relation if relation_match else other_relation
                                truth = bool(entity_match and object_match and relation_match)
                                rows.append({
                                    "case_id": f"c075-a-{len(rows):04d}", "partition": partition(index),
                                    "family": family, "other_family": other_family, "index": index,
                                    "surface": surface, "cell": cell,
                                    "record_target": record_target, "record_relation": record_relation, "record_object": family,
                                    "query_target": query_target, "query_relation": query_relation, "query_object": query_object,
                                    "entity_match": bool(entity_match), "object_match": bool(object_match), "relation_match": bool(relation_match),
                                    "truth": truth,
                                    "prompt": template.format(record_target=record_target, record_relation=record_relation, record_object=family, query_target=query_target, query_relation=query_relation, query_object=query_object),
                                    "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                                })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        spans = {role: c072.all_spans(tok, ids, row[role]) for role in ROLES}
        if not all(spans.values()):
            raise RuntimeError((row["case_id"], spans))
        positions = {
            "record_target": spans["record_target"][0], "record_relation": spans["record_relation"][0], "record_object": spans["record_object"][0],
            "query_target": spans["query_target"][-1], "query_relation": spans["query_relation"][-1], "query_object": spans["query_object"][-1],
            "boundary": [len(ids) - 1],
        }
        compiled.append({
            **row, "prompt_ids": ids, "role_positions": positions,
            "candidate_ids": [list(map(int, tok.encode(" " + value, add_special_tokens=False))) for value in row["candidates"]],
        })
    return compiled


def composition_sets(active: list[dict]) -> list[dict]:
    by = {(row["family"], row["index"], row["record_relation"], row["surface"], row["cell"]): row for row in active}
    result = []
    for family in ORDER:
        for index in range(6):
            for relation in RELATIONS:
                row = {"set_id": f"c075-compose-{len(result):04d}", "partition": partition(index), "family": family, "index": index, "record_relation": relation}
                for surface in SURFACES:
                    for cell in CELLS:
                        row[f"{surface}_{cell}"] = by[(family, index, relation, surface, cell)]["case_id"]
                result.append(row)
    return result


def balanced_accuracy(truths: list[bool], predictions: list[bool]) -> float:
    return c072.balanced_accuracy(truths, predictions)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1450 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c075_full_hiddenstate_observation_atlas_on_c074_robust_edges" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C074 closure missing")
    robust = core.rows(C074 / "analysis/robust_edges.jsonl")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    truths = [row["truth"] for row in active]
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLE_SLOTS) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    zero_models = {
        "always_yes_balanced_accuracy": balanced_accuracy(truths, [True] * len(active)),
        "always_no_balanced_accuracy": balanced_accuracy(truths, [False] * len(active)),
        "surface_only_balanced_accuracy": balanced_accuracy(truths, [row["surface"] == "a_evidence_first" for row in active]),
        "entity_only_balanced_accuracy": balanced_accuracy(truths, [row["entity_match"] for row in active]),
        "object_only_balanced_accuracy": balanced_accuracy(truths, [row["object_match"] for row in active]),
        "relation_only_balanced_accuracy": balanced_accuracy(truths, [row["relation_match"] for row in active]),
        "entity_object_balanced_accuracy": balanced_accuracy(truths, [row["entity_match"] and row["object_match"] for row in active]),
        "entity_relation_balanced_accuracy": balanced_accuracy(truths, [row["entity_match"] and row["relation_match"] for row in active]),
        "object_relation_balanced_accuracy": balanced_accuracy(truths, [row["object_match"] and row["relation_match"] for row in active]),
        "exact_conjunction_balanced_accuracy": balanced_accuracy(truths, [row["entity_match"] and row["object_match"] and row["relation_match"] for row in active]),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "robust_anchor": len(robust) == 10 and all(row["classification"] == "robust" for row in robust),
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 36 and not ({value.lower() for value in members} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members | set(RELATIONS)),
        "active": len(active) == 3456 and Counter(row["surface"] for row in active) == {surface: 1728 for surface in SURFACES},
        "cells": Counter(row["cell"] for row in active) == {cell: 432 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 432, False: 3024},
        "semantic_unique": all(row["truth"] == (row["entity_match"] and row["object_match"] and row["relation_match"]) for row in active),
        "composition": len(composition) == 216 and Counter(row["partition"] for row in composition) == {name: 72 for name in PARTITIONS},
        "compiled": len(compiled) == len(active),
        "same_shape_per_surface": all(len(values) == 1 for values in lengths.values()),
        "stable_role_map": all(len(values) == 1 for values in signatures.values()),
        "role_singletons": all(all(len(row["role_positions"][role]) == 1 for role in ROLE_SLOTS) and len({row["role_positions"][role][0] for role in ROLE_SLOTS}) == len(ROLE_SLOTS) for row in compiled),
        "role_order": all(max(row["role_positions"][role][0] for role in ROLES[:3]) < min(row["role_positions"][role][0] for role in ROLES[3:]) < row["role_positions"]["boundary"][0] for row in compiled),
        "answer_singletons": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero_models": max(value for key, value in zero_models.items() if key != "exact_conjunction_balanced_accuracy") == 13 / 14 and zero_models["exact_conjunction_balanced_accuracy"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c075.relation_atlas_material.v1", "families": FAMILIES, "relations": list(RELATIONS),
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES,
        "concepts": [{"word": word, "family": family, "index": index, "partition": partition(index)} for family, values in FAMILIES.items() for index, word in enumerate(values)],
    })
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/c074_robust_edge_anchor.jsonl", robust)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "zero_models": zero_models,
        "semantic_scope": "controlled three-factor equality over person, relation verb, and organization",
        "naturalness_scope": "machine-audited controlled English; no independent human blind review",
        "observation_rule": "discovery raw full-dimensional embedding and every-layer role states first; no candidate hypothesis may access confirmation or lockbox before Phase1453 freeze",
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c075.observation_first_full_field_atlas.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "expected_hidden_state_count": 37,
        "research_object": "raw full-dimensional embedding-to-every-layer role-aligned response field for six relation patterns",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at every model state", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "heatmap hotspot discovery", "PCA", "t-SNE", "UMAP", "learned probe", "coordinate pruning before raw capture", "post-holdout candidate or threshold changes"],
        "roles": list(ROLES), "role_slots": list(ROLE_SLOTS), "relations": list(RELATIONS), "surfaces": list(SURFACES), "cells": list(CELLS), "partitions": list(PARTITIONS),
        "material": {
            "active_count": 3456, "composition_count": 216, "minimum_families": 4, "minimum_relations": 5,
            "surface_lengths": {key: next(iter(values)) for key, values in lengths.items()},
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"),
            "c074_robust_edge_sha256": core.sha(OUT / "material/c074_robust_edge_anchor.jsonl"), "human_naturalness_lock": False,
        },
        "zero_model_gate": {"maximum_incomplete_balanced_accuracy": 13 / 14, "required_model_balanced_accuracy_min": 0.97},
        "behavior": {
            "family_relation_surface_accuracy_min": 0.95, "family_relation_surface_balanced_accuracy_min": 0.95,
            "partition_min": 0.90, "truth_min": 0.90, "cell_min": 0.90,
            "relation_set_all_min": 0.85, "same_batch_repeat_max_abs_diff": 1e-6,
        },
        "discovery_capture": {
            "partition": "response_discovery", "expected_case_count": 1152,
            "state_count": 37, "role_slot_count": 7, "dtype": "float16",
            "raw_format": "numpy memmap N x state x role_slot x hidden_dimension plus JSON index",
            "no_pooling": True, "no_coordinate_selection": True, "no_holdout_access": True,
        },
        "discovery_description": {
            "factor_effects": ["entity", "object", "relation"],
            "operation": "full-factorial paired first differences at every raw coordinate, layer, role, relation, and surface",
            "allowed_summaries": ["L2 norm", "mean vector", "direction consistency", "cross-surface cosine", "coordinate sign consistency"],
            "candidate_freeze_before_holdout": True,
        },
        "holdout_validation": {
            "partitions": ["confirmation", "lockbox"],
            "candidate_source": "Phase1453 frozen discovery manifest only",
            "branch_failure": "closes the candidate branch but does not stop other frozen candidates",
        },
        "stop_rule": "behavior failure blocks Hidden State; discovery capture failure closes acquisition; all discovery candidates are frozen before any holdout state is read; candidate failures are route-level, not campaign-level",
        "claim_boundary": {
            "allowed": "descriptive and then held-out predictive regularities in raw role-aligned embedding-HiddenState responses for one Qwen3 controlled relation task",
            "forbidden": ["semantic neuron group discovered from discovery alone", "necessity or natural use", "complete language manifold", "relative encoding proven", "cross-model law", "new mathematics established"],
        },
        "branching": {"phase1451": "behavior", "phase1452": "discovery raw capture", "phase1453": "description and candidate freeze", "phase1454": "confirmation and lockbox validation", "phase1455": "closure"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1451_c075_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
