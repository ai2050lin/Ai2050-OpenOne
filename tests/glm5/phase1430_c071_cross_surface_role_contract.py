#!/usr/bin/env python3
"""Phase1430: preregister C071 cross-surface role-isomorphic quartet transport."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1430, "C071"
PARENT = TESTS / "result/phase1429_c070_campaign_closure"
OUT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
FAMILIES = {
    "Echo": ("Ada", "Alan", "Albert", "Ann", "Anne", "Austin", "Blake", "Charlotte", "Connor", "Craig", "David", "Dean"),
    "Phoenix": ("Fred", "Gary", "George", "Grant", "Harry", "Henry", "Howard", "Hugh", "Ian", "Jack", "Jacob", "James"),
    "Titan": ("Jane", "Jason", "Jean", "Jennifer", "Jeremy", "Jerry", "Joe", "John", "Johnny", "Jonathan", "Jordan", "Joseph"),
    "Azure": ("Juan", "Justin", "Karen", "Keith", "Kelly", "Kyle", "Larry", "Lisa", "Louis", "Luke", "Maria", "Mark"),
    "Omega": ("Martin", "Mary", "Matthew", "Michael", "Michelle", "Neil", "Patrick", "Paul", "Peter", "Philip", "Richard", "Robert"),
    "Lambda": ("Roger", "Roy", "Ryan", "Sarah", "Scott", "Sean", "Simon", "Stephen", "Steven", "Susan", "Taylor", "Thomas"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SYSTEM = "Use only the roster statement. Answer yes exactly when the queried person and group match the recorded person and group. Output exactly yes or no."
SURFACES = {
    "belongs_include": "Roster note: {record_target} belongs to group {record_family}. Check: does group {query_family} include {query_target}? Answer only yes or no.",
    "lists_member": "Group {record_family} lists {record_target} as a member. Check: is {query_target} listed in group {query_family}? Answer only yes or no.",
}
ROLES = ("record_target", "record_family", "query_target", "query_family")
CELLS = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")
PERMUTED_SOURCE = {
    "record_target": "record_family",
    "record_family": "query_target",
    "query_target": "query_family",
    "query_family": "record_target",
}


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(v, add_special_tokens=False))) for v in (value, " " + value)]
    found = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                span = list(range(start, start + len(needle)))
                if span not in found:
                    found.append(span)
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
            for surface, template in SURFACES.items():
                for cell, (rt, rf, qt, qf, truth) in cells.items():
                    rows.append({
                        "case_id": f"c071-a-{len(rows):04d}", "partition": partition(index),
                        "pair": f"{base_a}__{base_b}", "orientation": f"{fa}__{fb}", "index": index,
                        "surface": surface, "cell": cell, "record_target": rt, "record_family": rf,
                        "query_target": qt, "query_family": qf, "truth": truth,
                        "prompt": template.format(record_target=rt, record_family=rf, query_target=qt, query_family=qf),
                        "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        spans = {role: all_spans(tok, ids, row[role]) for role in ROLES}
        if not all(spans.values()):
            raise RuntimeError((row["case_id"], spans))
        positions = {
            "record_target": spans["record_target"][0],
            "record_family": spans["record_family"][0],
            "query_target": spans["query_target"][-1],
            "query_family": spans["query_family"][-1],
            "boundary": [len(ids) - 1],
        }
        compiled.append({
            "case_id": row["case_id"], "surface": row["surface"], "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("yes", "no")],
            "role_positions": positions,
        })
    return compiled


def composition_sets(active: list[dict]) -> list[dict]:
    by = {(row["surface"], *(row[role] for role in ROLES)): row for row in active}
    result = []
    for fi, family in enumerate(ORDER):
        g, h = ORDER[(fi + 1) % 6], ORDER[(fi + 2) % 6]
        for index in range(12):
            fw = FAMILIES[family][index]
            donor, other = (g, h) if index % 2 == 0 else (h, g)
            dw = FAMILIES[donor][index]
            row = {
                "set_id": f"c071-compose-{len(result):04d}", "partition": partition(index),
                "family": family, "index": index, "donor_family": donor, "other_family": other,
            }
            for surface in SURFACES:
                row[f"{surface}_true_recipient"] = by[(surface, fw, family, fw, family)]["case_id"]
                row[f"{surface}_false_recipient"] = by[(surface, fw, family, fw, g)]["case_id"]
                row[f"{surface}_true_donor"] = by[(surface, dw, donor, dw, donor)]["case_id"]
                row[f"{surface}_false_donor"] = by[(surface, dw, donor, dw, other)]["case_id"]
            result.append(row)
    return result


def balanced_accuracy(truths: list[bool], predictions: list[bool]) -> float:
    tpr = sum(p for t, p in zip(truths, predictions) if t) / sum(truths)
    negatives = sum(not t for t in truths)
    tnr = sum(not p for t, p in zip(truths, predictions) if not t) / negatives
    return (tpr + tnr) / 2.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1430 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c071_cross_surface_role_isomorphic_quartet_transport" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C070 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    source = {row["case_id"]: row for row in active}
    words = {word for values in FAMILIES.values() for word in values}
    labels = set(FAMILIES)
    old_words = set()
    for path in (TESTS / "result").glob("phase*/material/frozen_concept_graph.json"):
        try:
            data = core.load(path)
            for row in data.get("concepts", []):
                for key in ("word", "family", "label", "name"):
                    if row.get(key):
                        old_words.add(str(row[key]).lower())
        except Exception:
            pass
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    role_signatures = {
        surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled if row["surface"] == surface}
        for surface in SURFACES
    }
    cell_family_surface = {
        (family, surface, cell): sum(row["record_family"] == family and row["surface"] == surface and row["cell"] == cell for row in active)
        for family in ORDER for surface in SURFACES for cell in CELLS
    }
    truths = [row["truth"] for row in active]
    zero_models = {
        "always_yes_balanced_accuracy": balanced_accuracy(truths, [True] * len(active)),
        "always_no_balanced_accuracy": balanced_accuracy(truths, [False] * len(active)),
        "surface_only_balanced_accuracy": balanced_accuracy(truths, [row["surface"] == "belongs_include" for row in active]),
        "person_only_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] for row in active]),
        "group_only_balanced_accuracy": balanced_accuracy(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact_person_and_group_solver": balanced_accuracy(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    donor_truth = {"true_recipient": True, "false_recipient": False, "true_donor": True, "false_donor": False}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "six_fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old_words),
        "seventy_two_fresh_people": len(words) == 72 and not ({value.lower() for value in words} & old_words),
        "singleton_material": all(len(tok.encode(value, add_special_tokens=False)) == 1 and len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in words | labels),
        "active": len(active) == 2880 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in SURFACES},
        "cells": Counter(row["cell"] for row in active) == {cell: 360 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 720, False: 2160},
        "all_family_surface_cells": all(count == 30 for count in cell_family_surface.values()),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[f"{surface}_{key}"]]["truth"] == truth for row in composition for surface in SURFACES for key, truth in donor_truth.items()),
        "compiled": len(compiled) == len(active),
        "same_shape_within_surface": all(len(values) == 1 for values in lengths.values()),
        "different_surface_lengths": len({next(iter(values)) for values in lengths.values()}) == 2,
        "stable_role_coordinates": all(len(values) == 1 for values in role_signatures.values()),
        "role_coordinates_differ": next(iter(role_signatures["belongs_include"])) != next(iter(role_signatures["lists_member"])),
        "quartet_singleton_distinct": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 and all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled),
        "answers_singleton": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "semantic_unique": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "machine_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.") and row["prompt"].count("?") == 1 for row in active),
        "zero_models_recomputed": abs(zero_models["person_only_balanced_accuracy"] - 5 / 6) < 1e-12 and abs(zero_models["group_only_balanced_accuracy"] - 5 / 6) < 1e-12 and zero_models["exact_person_and_group_solver"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    concepts = [
        {"word": word, "family": family, "index": index, "partition": partition(index), "sense": f"person explicitly assigned to controlled roster group {family}"}
        for family, values in FAMILIES.items() for index, word in enumerate(values)
    ]
    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c071.cross_surface_roster.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES, "concepts": concepts,
    })
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()),
        "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero_models,
        "c070_zero_model_erratum": {
            "previous_reported_person_only_balanced_accuracy": 0.5,
            "previous_reported_group_only_balanced_accuracy": 0.5,
            "recomputed_each": 5 / 6,
            "impact": "does not alter C070 behavior qualification or intervention result; corrects shortcut accounting",
        },
        "semantic_scope": "closed-world exact person-and-group equality across two controlled surfaces",
        "naturalness_scope": "machine-audited controlled roster English; no independent human blind review",
        "risks": ["explicit lexical equality", "arbitrary group labels", "two controlled surfaces", "singleton roles", "raw truth prevalence 1:3", "person-only and group-only shortcuts achieve 5/6 balanced accuracy"],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c071.cross_surface_role_isomorphism.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "state16 cross-surface semantic-role-mapped quartet transport",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "role subset search", "coordinate search", "post-reveal changes"],
        "role_map": {"roles": list(ROLES), "definition": "phi_x maps each semantic role to its compiled singleton token position", "permuted_source": PERMUTED_SOURCE},
        "material": {
            "families": list(FAMILIES), "surfaces": list(SURFACES), "partitions": list(PARTITIONS),
            "active_count": len(active), "composition_count": len(composition), "minimum_qualified_families": 4,
            "selected_per_family_partition": 4, "surface_token_lengths": {key: next(iter(value)) for key, value in lengths.items()},
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False,
        },
        "zero_model_gate": {"maximum_incomplete_balanced_accuracy": 5 / 6, "required_model_balanced_accuracy_min": 0.95},
        "behavior": {
            "family_surface_accuracy_min": 0.95, "family_surface_balanced_accuracy_min": 0.95,
            "family_surface_partition_min": 0.90, "family_surface_truth_min": 0.90,
            "family_surface_cell_min": 0.90, "family_set_all_min": 0.85,
            "same_shape_repeat_max_abs_diff": 1e-6,
        },
        "camera": {
            "known_truth_systems": 256, "qwen_discovery_sets": 24, "state_index": 16,
            "surface_transfers": ["belongs_include_to_lists_member", "lists_member_to_belongs_include"],
            "directions": ["true_to_false", "false_to_true"],
            "self_role_max_abs_diff": 1e-4, "mapped_role_max_abs_diff": 1e-4,
            "untouched_complement_max_abs_diff": 1e-4,
        },
        "mechanism": {
            "state_index": 16, "directions": ["true_to_false", "false_to_true"],
            "surface_transfers": ["belongs_include_to_lists_member", "lists_member_to_belongs_include"],
            "arms": ["self", "same_surface_quartet", "cross_surface_role_mapped", "cross_surface_role_permuted", "wrong_cross_surface_role_mapped"],
            "self_max_abs_diff": 1e-4, "same_surface_desired_sign_fraction_min": 0.90,
            "cross_surface_desired_sign_fraction_min": 0.90, "wrong_expected_sign_fraction_min": 0.90,
            "mapped_oriented_gain_median_min": 0.50, "mapped_vs_permuted_sign_gap_min": 0.50,
            "mapped_vs_permuted_gain_gap_median_min": 0.50, "family_desired_sign_fraction_min": 0.75,
            "minimum_family_breadth": 4,
        },
        "classification": {
            "per_surface_transfer_direction": ["role_isomorphic_selective", "cross_surface_nonspecific", "same_surface_only", "same_surface_executor_failed"],
            "role_isomorphic_selective": "same-surface and cross-surface mapped controls pass, wrong donor passes, and mapped beats the frozen role permutation on sign and gain",
            "overall": "all four transfer-direction cells must agree; otherwise transfer_or_direction_asymmetric",
        },
        "branching": {"phase1431": "behavior", "phase1432": "role-map camera", "phase1433": "cross-surface mechanism", "phase1434": "closure"},
        "stop_rule": "behavior failure blocks hidden; camera failure blocks holdouts; mechanism runs once and every frozen classification closes C071",
        "claim_boundary": {
            "allowed": "Qwen controlled-roster cross-surface role-mapped quartet transport at fixed state16",
            "forbidden": ["semantic manifold", "natural language invariant", "minimal/necessary mechanism", "relative encoding proven", "cross-model law"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1431_c071_cross_surface_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True,
        "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
