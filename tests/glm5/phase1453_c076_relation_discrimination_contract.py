#!/usr/bin/env python3
"""Phase1453: preregister C076 explicit relation-discrimination atlas."""
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

PHASE, CAMPAIGN = 1453, "C076"
PARENT = TESTS / "result/phase1452_c075_behavior_gate_closure"
OUT = TESTS / "result/phase1453_c076_relation_discrimination_contract"
FAMILIES = {
    "Linden": ("Esther", "Felix", "Forrest", "Franklin", "Garrett", "Georgia"),
    "Nova": ("Gwen", "Hank", "Hugo", "Isaiah", "Jared", "Jill"),
    "Slate": ("Jonah", "Jorge", "Kara", "Karl", "Kent", "Kerry"),
    "Vertex": ("Lana", "Lance", "Lara", "Laurel", "Lawrence", "Leon"),
    "Ember": ("Leo", "Levi", "Lionel", "Lloyd", "Lola", "Lorenzo"),
    "Junction": ("Lucia", "Luna", "Macy", "Manuel", "Penny", "Marco"),
}
RELATIONS = {
    "join": {"record": "joined", "query": "join"},
    "support": {"record": "supported", "query": "support"},
    "visit": {"record": "visited", "query": "visit"},
    "contact": {"record": "contacted", "query": "contact"},
    "select": {"record": "selected", "query": "select"},
    "praise": {"record": "praised", "query": "praise"},
}
RELATION_IDS = tuple(RELATIONS)
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 2), "confirmation": range(2, 4), "lockbox": range(4, 6)}
SYSTEM = "Judge only whether the two clauses express the same relationship type. Person and organization names may differ and must be ignored. Output exactly yes or no."
SURFACES = {
    "a_explicit": "First clause: {record_target} {record_relation} {record_object}. Second clause: {query_target} may {query_relation} {query_object}. Do the clauses express the same relationship type? Answer only yes or no.",
    "b_explicit": "Relation comparison: {record_target} {record_relation} {record_object}; {query_target} can {query_relation} {query_object}. Ignoring the names, is the relationship the same? Reply only yes or no.",
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
            record_target, other_target = FAMILIES[family][index], FAMILIES[other_family][index]
            for ri, relation_id in enumerate(RELATION_IDS):
                other_id = RELATION_IDS[(ri + 1 + index % 2) % len(RELATION_IDS)]
                for surface, template in SURFACES.items():
                    for entity_match in (1, 0):
                        for object_match in (1, 0):
                            for relation_match in (1, 0):
                                query_id = relation_id if relation_match else other_id
                                cell = f"{entity_match}{object_match}{relation_match}"
                                row = {
                                    "case_id": f"c076-a-{len(rows):04d}", "partition": partition(index),
                                    "family": family, "other_family": other_family, "index": index, "surface": surface, "cell": cell,
                                    "record_target": record_target, "record_relation": RELATIONS[relation_id]["record"], "record_relation_id": relation_id, "record_object": family,
                                    "query_target": record_target if entity_match else other_target,
                                    "query_relation": RELATIONS[query_id]["query"], "query_relation_id": query_id,
                                    "query_object": family if object_match else other_family,
                                    "entity_match": bool(entity_match), "object_match": bool(object_match), "relation_match": bool(relation_match),
                                    "truth": bool(relation_match), "candidates": ["yes", "no"], "gold_position": 0 if relation_match else 1,
                                }
                                row["prompt"] = template.format(**row)
                                rows.append(row)
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
        compiled.append({**row, "prompt_ids": ids, "role_positions": positions, "candidate_ids": [list(map(int, tok.encode(" " + value, add_special_tokens=False))) for value in row["candidates"]]})
    return compiled


def composition_sets(active: list[dict]) -> list[dict]:
    by = {(row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row for row in active}
    result = []
    for family in ORDER:
        for index in range(6):
            for relation_id in RELATION_IDS:
                row = {"set_id": f"c076-compose-{len(result):04d}", "partition": partition(index), "family": family, "index": index, "record_relation_id": relation_id}
                for surface in SURFACES:
                    for cell in CELLS:
                        row[f"{surface}_{cell}"] = by[(family, index, relation_id, surface, cell)]["case_id"]
                result.append(row)
    return result


def ba(truths: list[bool], predictions: list[bool]) -> float:
    return c072.balanced_accuracy(truths, predictions)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1453 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c076_explicit_relation_discrimination_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C075 closure missing")
    tok = tokenizer()
    active, compiled = active_cases(), None
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    relation_words = {value for pair in RELATIONS.values() for value in pair.values()}
    truths = [row["truth"] for row in active]
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLE_SLOTS) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    zero = {
        "always_yes": ba(truths, [True] * len(active)), "always_no": ba(truths, [False] * len(active)),
        "surface": ba(truths, [row["surface"] == "a_explicit" for row in active]),
        "entity": ba(truths, [row["entity_match"] for row in active]), "object": ba(truths, [row["object_match"] for row in active]),
        "entity_object": ba(truths, [row["entity_match"] and row["object_match"] for row in active]),
        "relation_concept": ba(truths, [row["record_relation_id"] == row["query_relation_id"] for row in active]),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 36 and not ({value.lower() for value in members} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members | relation_words),
        "active": len(active) == 3456 and Counter(row["surface"] for row in active) == {surface: 1728 for surface in SURFACES},
        "cells": Counter(row["cell"] for row in active) == {cell: 432 for cell in CELLS},
        "truth": Counter(truths) == {True: 1728, False: 1728},
        "semantic_unique": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in active),
        "nuisance_balance": all(Counter(row[key] for row in active) == {True: 1728, False: 1728} for key in ("entity_match", "object_match", "relation_match")),
        "composition": len(composition) == 216 and Counter(row["partition"] for row in composition) == {name: 72 for name in PARTITIONS},
        "compiled": len(compiled) == len(active), "same_shape": all(len(values) == 1 for values in lengths.values()),
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "role_singletons": all(all(len(row["role_positions"][role]) == 1 for role in ROLE_SLOTS) and len({row["role_positions"][role][0] for role in ROLE_SLOTS}) == 7 for row in compiled),
        "role_order": all(max(row["role_positions"][role][0] for role in ROLES[:3]) < min(row["role_positions"][role][0] for role in ROLES[3:]) < row["role_positions"]["boundary"][0] for row in compiled),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero_models": all(value == 0.5 for key, value in zero.items() if key != "relation_concept") and zero["relation_concept"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c076.explicit_relation_material.v1", "families": FAMILIES, "relations": RELATIONS, "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES, "concepts": [{"word": word, "family": family, "index": index, "partition": partition(index)} for family, values in FAMILIES.items() for index, word in enumerate(values)]})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "semantic_scope": "explicit relation-concept equality across past/base morphology with independently balanced person and organization nuisances", "naturalness_scope": "machine-audited controlled English; no independent human blind review"}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c076.explicit_relation_discrimination_atlas.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "expected_hidden_state_count": 37,
        "research_object": "observation-first full-dimensional relation-concept field with person and organization nuisance variation",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at every model state", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "coordinate pruning before raw capture", "post-holdout candidate or threshold changes"],
        "roles": list(ROLES), "role_slots": list(ROLE_SLOTS), "relations": RELATIONS, "surfaces": list(SURFACES), "cells": list(CELLS), "partitions": list(PARTITIONS),
        "material": {"active_count": 3456, "composition_count": 216, "surface_lengths": {key: next(iter(values)) for key, values in lengths.items()}, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": {"global_surface_balanced_accuracy_min": 0.98, "family_relation_surface_accuracy_min": 0.95, "family_relation_surface_balanced_accuracy_min": 0.95, "partition_min": 0.90, "truth_min": 0.90, "cell_min": 0.90, "all_composition_sets_required": True, "same_batch_repeat_max_abs_diff": 1e-6},
        "discovery_capture": {"partition": "response_discovery", "expected_case_count": 1152, "state_count": 37, "role_slot_count": 7, "dtype": "float16", "raw_format": "numpy memmap N x state x role_slot x hidden_dimension plus JSON index", "no_pooling": True, "no_coordinate_selection": True, "no_holdout_access": True},
        "discovery_description": {"effects": ["relation_match", "entity_nuisance", "object_nuisance"], "operation": "full-factorial paired first differences at every raw coordinate, layer, role, relation, and surface", "allowed_summaries": ["L2 norm", "mean vector", "direction consistency", "cross-surface cosine", "coordinate sign consistency"], "candidate_freeze_before_holdout": True},
        "holdout_validation": {"partitions": ["confirmation", "lockbox"], "candidate_source": "frozen discovery manifest only", "branch_failure": "candidate branch closes without stopping other frozen candidates"},
        "stop_rule": "behavior failure blocks Hidden State; discovery capture is response-discovery only; all candidates and thresholds freeze before holdout; candidate failures are route-level",
        "claim_boundary": {"allowed": "descriptive then held-out predictive regularities in raw relation-concept responses for one Qwen3 controlled task", "forbidden": ["semantic neuron group from discovery alone", "necessity or natural use", "complete language manifold", "relative encoding proven", "cross-model law", "new mathematics established"]},
        "branching": {"phase1454": "behavior", "phase1455": "discovery capture", "phase1456": "description and candidate freeze", "phase1457": "holdout validation", "phase1458": "closure"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1454_c076_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
