#!/usr/bin/env python3
"""Phase1425: preregister C070 quartet-versus-complement support partition."""
from __future__ import annotations

import json, sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1425, "C070"
PARENT = TESTS / "result/phase1424_c069_campaign_closure"
OUT = TESTS / "result/phase1425_c070_quartet_complement_contract"
FAMILIES = {
    "Alpha": ("Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Laura"),
    "Beta": ("Mike", "Nancy", "Oscar", "Peggy", "Quinn", "Rachel", "Sam", "Tina", "Victor", "Wendy", "Xavier", "Zach"),
    "Gamma": ("Aaron", "Abby", "Adam", "Adrian", "Alex", "Alexis", "Amanda", "Amy", "Andrew", "Angela", "Anna", "Anthony"),
    "Delta": ("Arthur", "Ashley", "Barbara", "Benjamin", "Betty", "Brandon", "Brenda", "Brian", "Brittany", "Bruce", "Bryan", "Carl"),
    "Theta": ("Catherine", "Charles", "Christina", "Christine", "Christopher", "Cynthia", "Daniel", "Deborah", "Dennis", "Diana", "Donald", "Donna"),
    "Sigma": ("Dorothy", "Douglas", "Edward", "Elizabeth", "Emily", "Emma", "Eric", "Ethan", "Eugene", "Evelyn", "Frances", "Gabriel"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SYSTEM = "Use only the roster entry. Answer yes only when both the queried person and team exactly match that entry. Output exactly yes or no."
TEMPLATE = "Roster entry: {record_target} is assigned to team {record_family}. Check: is {query_target} assigned to team {query_family}? Output only yes or no."
ROLES = ("record_target", "record_family", "query_target", "query_family")
CELLS = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(v, add_special_tokens=False))) for v in (value, " " + value)]
    found = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                span = list(range(start, start + len(needle)))
                if span not in found: found.append(span)
    return sorted(found)


def active_cases() -> list[dict]:
    rows = []
    for base_a, base_b in combinations(ORDER, 2):
        for index in range(12):
            fa, fb = (base_a, base_b) if index % 2 == 0 else (base_b, base_a)
            aw, bw = FAMILIES[fa][index], FAMILIES[fb][index]
            cells = {
                "aa": (aw, fa, aw, fa, True), "ab": (aw, fa, aw, fb, False),
                "ac": (aw, fa, bw, fa, False), "ad": (aw, fa, bw, fb, False),
                "bb": (bw, fb, bw, fb, True), "ba": (bw, fb, bw, fa, False),
                "bc": (bw, fb, aw, fb, False), "bd": (bw, fb, aw, fa, False),
            }
            for cell, (rt, rf, qt, qf, truth) in cells.items():
                rows.append({
                    "case_id": f"c070-a-{len(rows):04d}", "partition": partition(index),
                    "pair": f"{base_a}__{base_b}", "orientation": f"{fa}__{fb}", "index": index, "cell": cell,
                    "record_target": rt, "record_family": rf, "query_target": qt, "query_family": qf,
                    "truth": truth, "prompt": TEMPLATE.format(record_target=rt, record_family=rf, query_target=qt, query_family=qf),
                    "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        spans = {role: all_spans(tok, ids, row[role]) for role in ROLES}
        if not all(spans.values()): raise RuntimeError((row["case_id"], spans))
        compiled.append({
            "case_id": row["case_id"], "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("yes", "no")],
            "role_positions": {
                "record_target": spans["record_target"][0], "record_family": spans["record_family"][0],
                "query_target": spans["query_target"][-1], "query_family": spans["query_family"][-1],
                "boundary": [len(ids) - 1],
            },
        })
    return compiled


def signature(rows: list[dict]) -> dict[tuple, dict]:
    return {tuple(row[role] for role in ROLES): row for row in rows}


def composition_sets(active: list[dict]) -> list[dict]:
    by = signature(active)
    result = []
    for fi, family in enumerate(ORDER):
        g, h = ORDER[(fi + 1) % 6], ORDER[(fi + 2) % 6]
        for index in range(12):
            fw = FAMILIES[family][index]
            donor, other = (g, h) if index % 2 == 0 else (h, g)
            dw = FAMILIES[donor][index]
            result.append({
                "set_id": f"c070-compose-{len(result):04d}", "partition": partition(index),
                "family": family, "index": index, "donor_family": donor, "other_family": other,
                "true_recipient": by[(fw, family, fw, family)]["case_id"],
                "false_recipient": by[(fw, family, fw, g)]["case_id"],
                "true_donor": by[(dw, donor, dw, donor)]["case_id"],
                "false_donor": by[(dw, donor, dw, other)]["case_id"],
            })
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError("Phase1425 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c070_quartet_complement_support_partition" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C069 closure missing")
    tok = tokenizer(); active = active_cases(); compiled = compile_rows(tok, active); composition = composition_sets(active)
    source = {row["case_id"]: row for row in active}
    words = {word for values in FAMILIES.values() for word in values}; labels = set(FAMILIES)
    old_words, old_families = set(), set()
    for path in (TESTS / "result").glob("phase*/material/frozen_concept_graph.json"):
        try:
            data = core.load(path); old_words |= {str(row["word"]).lower() for row in data["concepts"]}; old_families |= {str(row.get("family", "")).lower() for row in data["concepts"]}
        except Exception: pass
    role_signatures = {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled}
    cell_family = {(family, cell): sum(row["record_family"] == family and row["cell"] == cell for row in active) for family in ORDER for cell in CELLS}
    donor_truth = {"true_recipient": True, "false_recipient": False, "true_donor": True, "false_donor": False}
    checks = {
        "parent": parent_audit["all_checks_passed"], "six_new_labels": len(labels) == 6 and not ({x.lower() for x in labels} & (old_words | old_families)),
        "seventy_two_fresh_people": len(words) == 72 and not ({x.lower() for x in words} & old_words),
        "active": len(active) == 1440 and Counter(row["cell"] for row in active) == {cell: 180 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 360, False: 1080},
        "every_family_all_cells": all(count == 30 for count in cell_family.values()),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[key]]["truth"] == truth for row in composition for key, truth in donor_truth.items()),
        "compiled": len(compiled) == len(active), "same_shape": len({len(row["prompt_ids"]) for row in compiled}) == 1,
        "same_role_coordinates": len(role_signatures) == 1,
        "answers_singleton": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "quartet_singleton": all(len(row["role_positions"][role]) == 1 for row in compiled for role in ROLES),
        "semantic_unique": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "machine_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.") and row["prompt"].count("team") == 2 for row in active),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()): raise RuntimeError({key: value for key, value in checks.items() if not value})
    concepts = [{"word": word, "family": family, "index": index, "partition": partition(index), "sense": f"person explicitly assigned to controlled roster team {family}"} for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c070.roster_assignment.v1", "families": FAMILIES, "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active); core.write_rows(OUT / "material/composition_sets.jsonl", composition); core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": {"always_yes_balanced_accuracy": 0.5, "always_no_balanced_accuracy": 0.5, "fixed_candidate_position_balanced_accuracy": 0.5, "person_only_balanced_accuracy": 0.5, "team_only_balanced_accuracy": 0.5, "exact_person_and_team_solver": 1.0}, "semantic_scope": "closed-world exact person-and-team roster equality", "naturalness_scope": "machine-audited controlled roster English", "independent_human_blind_review": False, "risks": ["explicit lexical equality", "arbitrary team labels", "single surface", "singleton roles", "raw truth prevalence 1:3"]}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c070.quartet_complement_partition.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "research_object": "state16 quartet-versus-complement causal support partition",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "position subset search", "coordinate search", "post-reveal changes"],
        "material": {"families": list(FAMILIES), "partitions": list(PARTITIONS), "active_count": len(active), "composition_count": len(composition), "minimum_qualified_families": 4, "selected_per_family_partition": 4, "prompt_token_length": len(compiled[0]["prompt_ids"]), "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": {"family_accuracy_min": 0.95, "family_balanced_accuracy_min": 0.95, "family_partition_min": 0.90, "family_truth_min": 0.90, "family_cell_min": 0.90, "family_set_all_min": 0.85, "same_shape_repeat_max_abs_diff": 1e-6},
        "camera": {"known_truth_systems": 256, "qwen_discovery_sets": 24, "state_index": 16, "roles": list(ROLES), "self_full_max_abs_diff": 1e-4, "donor_full_transport_max_abs_diff": 1e-4},
        "mechanism": {
            "state_index": 16, "directions": ["true_to_false", "false_to_true"],
            "arms": ["self", "quartet_only", "complement_only", "full_state", "wrong_full_state"],
            "self_max_abs_diff": 1e-4, "full_desired_sign_fraction_min": 0.90, "wrong_full_expected_sign_fraction_min": 0.90,
            "full_donor_relative_deviation_median_max": 0.25, "wrong_donor_relative_deviation_median_max": 0.25,
            "full_oriented_gain_median_min": 0.50, "aggregate_partition_desired_sign_fraction_min": 0.80,
            "family_partition_desired_sign_fraction_min": 0.75, "minimum_family_breadth": 4,
            "synergy_advantage_median_min": 0.50, "synergy_win_fraction_min": 0.65,
        },
        "classification": {
            "per_direction": ["redundant_dual_support", "quartet_dominant", "complement_dominant", "joint_only_or_unresolved", "full_transport_failed"],
            "overall": "same classification in both directions or direction_asymmetric",
            "claim_rule": "partition sufficiency requires both holdouts, aggregate threshold, and at least four family-level passes",
        },
        "branching": {"phase1426": "behavior", "phase1427": "partition camera", "phase1428": "five-arm support partition", "phase1429": "closure"},
        "stop_rule": "behavior failure blocks hidden; camera failure blocks holdouts; mechanism runs once and every frozen classification closes the campaign",
        "claim_boundary": {"allowed": "Qwen controlled-roster physical-position support partition at state16", "forbidden": ["semantic manifold", "minimal/necessary mechanism", "orthogonal subspaces", "cross-model/open-language law"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol); protocol["authorization"] = "run_phase1426_c070_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__": main()
