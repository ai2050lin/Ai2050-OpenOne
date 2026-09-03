#!/usr/bin/env python3
"""Phase1445: preregister C074 directional whole-state transport domain map."""
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
import phase1435_c072_permutation_spectrum_contract as c072
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1445, "C074"
PARENT = TESTS / "result/phase1444_c073_campaign_closure"
OUT = TESTS / "result/phase1445_c074_directional_domain_contract"
FAMILIES = {
    "Eclipse": ("Avery", "Calvin", "Carla", "Carlos", "Carolyn", "Cheryl", "Christian", "Courtney", "Danielle", "Denise", "Diane", "Elaine"),
    "Forge": ("Ella", "Faith", "Hayden", "Helen", "Hudson", "Hunter", "Irene", "Jackie", "Jacqueline", "Jamie", "Janet", "Jeffrey"),
    "Nexus": ("Jenna", "Jesse", "Jessica", "Joan", "Joanna", "Joel", "Joshua", "Joyce", "Judith", "Julia", "Julie", "Katherine"),
    "Orchard": ("Kathleen", "Kathryn", "Katie", "Leah", "Leonard", "Liam", "Lori", "Lucy", "Lydia", "Madison", "Malcolm", "Marie"),
    "Signal": ("Marilyn", "Max", "Maxwell", "Miguel", "Mitchell", "Nelson", "Pedro", "Phillip", "Preston", "Ralph", "Randy", "Raymond"),
    "Timber": ("Ronald", "Samuel", "Seth", "Shawn", "Shirley", "Sophia", "Stanley", "Suzanne", "Sydney", "Tony", "Travis", "Troy"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SYSTEM = "Use only the supplied record. Answer yes exactly when both the queried person and group match the recorded person and group. Output exactly yes or no."
SURFACES = {
    "a_evidence_first": "Evidence statement: {record_target} belongs to group {record_family}. Verification question: Does {query_target} belong to group {query_family}? Answer only yes or no.",
    "a_question_first": "Verification question: Does {query_target} belong to group {query_family}? Evidence statement: {record_target} belongs to group {record_family}. Answer only yes or no.",
    "b_evidence_first": "Registry entry: {record_target} is listed under {record_family}. Is {query_target} listed under {query_family}? Reply only yes or no.",
    "b_question_first": "Is {query_target} listed under {query_family}? Consult this registry entry: {record_target} is listed under {record_family}. Reply only yes or no.",
}
SURFACE_META = {
    "a_evidence_first": {"frame": "a", "order": "evidence_first"},
    "a_question_first": {"frame": "a", "order": "question_first"},
    "b_evidence_first": {"frame": "b", "order": "evidence_first"},
    "b_question_first": {"frame": "b", "order": "question_first"},
}
ROLES = ("record_target", "record_family", "query_target", "query_family")
CELLS = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")
ROUTES = {
    f"{source}__to__{target}": {
        "source": source,
        "target": target,
        "same_surface": source == target,
        "same_frame": SURFACE_META[source]["frame"] == SURFACE_META[target]["frame"],
        "same_order": SURFACE_META[source]["order"] == SURFACE_META[target]["order"],
    }
    for source in SURFACES for target in SURFACES
}


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


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
                        "case_id": f"c074-a-{len(rows):04d}", "partition": partition(index),
                        "pair": f"{base_a}__{base_b}", "orientation": f"{fa}__{fb}", "index": index,
                        "surface": surface, "frame": SURFACE_META[surface]["frame"],
                        "surface_order": SURFACE_META[surface]["order"], "cell": cell,
                        "record_target": rt, "record_family": rf, "query_target": qt, "query_family": qf,
                        "truth": truth,
                        "prompt": template.format(record_target=rt, record_family=rf, query_target=qt, query_family=qf),
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
        if row["surface_order"] == "evidence_first":
            positions = {
                "record_target": spans["record_target"][0], "record_family": spans["record_family"][0],
                "query_target": spans["query_target"][-1], "query_family": spans["query_family"][-1],
            }
        else:
            positions = {
                "query_target": spans["query_target"][0], "query_family": spans["query_family"][0],
                "record_target": spans["record_target"][-1], "record_family": spans["record_family"][-1],
            }
        positions["boundary"] = [len(ids) - 1]
        compiled.append({
            **row, "prompt_ids": ids, "role_positions": positions,
            "candidate_ids": [list(map(int, tok.encode(" " + value, add_special_tokens=False))) for value in row["candidates"]],
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
            row = {"set_id": f"c074-compose-{len(result):04d}", "partition": partition(index), "family": family, "index": index, "donor_family": donor, "other_family": other}
            for surface in SURFACES:
                row[f"{surface}_true_recipient"] = by[(surface, fw, family, fw, family)]["case_id"]
                row[f"{surface}_false_recipient"] = by[(surface, fw, family, fw, g)]["case_id"]
                row[f"{surface}_true_donor"] = by[(surface, dw, donor, dw, donor)]["case_id"]
                row[f"{surface}_false_donor"] = by[(surface, dw, donor, dw, other)]["case_id"]
            result.append(row)
    return result


def balanced_accuracy(truths: list[bool], predictions: list[bool]) -> float:
    return c072.balanced_accuracy(truths, predictions)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1445 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c074_directional_transport_domain_test" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C073 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    source = {row["case_id"]: row for row in active}
    truths = [row["truth"] for row in active]
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    zero_models = {
        "always_yes_balanced_accuracy": balanced_accuracy(truths, [True] * len(active)),
        "always_no_balanced_accuracy": balanced_accuracy(truths, [False] * len(active)),
        "surface_only_balanced_accuracy": balanced_accuracy(truths, [row["surface"] == "a_evidence_first" for row in active]),
        "person_only_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] for row in active]),
        "group_only_balanced_accuracy": balanced_accuracy(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact_conjunction_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    donor_truth = {"true_recipient": True, "false_recipient": False, "true_donor": True, "false_donor": False}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 72 and not ({value.lower() for value in members} & old),
        "context_singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members),
        "active": len(active) == 5760 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in SURFACES},
        "cells": Counter(row["cell"] for row in active) == {cell: 720 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 1440, False: 4320},
        "semantic_unique": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[f"{surface}_{key}"]]["truth"] == truth for row in composition for surface in SURFACES for key, truth in donor_truth.items()),
        "compiled": len(compiled) == len(active),
        "same_shape_per_surface": all(len(values) == 1 for values in lengths.values()),
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "role_order": all(
            (max(row["role_positions"][role][0] for role in ROLES[:2]) < min(row["role_positions"][role][0] for role in ROLES[2:]))
            if row["surface_order"] == "evidence_first" else
            (max(row["role_positions"][role][0] for role in ROLES[2:]) < min(row["role_positions"][role][0] for role in ROLES[:2]))
            for row in compiled
        ),
        "quartet_singletons": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 and all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled),
        "answer_singletons": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero_models": zero_models["always_yes_balanced_accuracy"] == 0.5 and zero_models["always_no_balanced_accuracy"] == 0.5 and zero_models["surface_only_balanced_accuracy"] == 0.5 and abs(zero_models["person_only_balanced_accuracy"] - 5 / 6) < 1e-12 and abs(zero_models["group_only_balanced_accuracy"] - 5 / 6) < 1e-12 and zero_models["exact_conjunction_balanced_accuracy"] == 1.0,
        "routes": len(ROUTES) == 16 and len({(value["source"], value["target"]) for value in ROUTES.values()}) == 16,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c074.directional_domain_material.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES,
        "surface_meta": SURFACE_META,
        "concepts": [{"word": word, "family": family, "index": index, "partition": partition(index)} for family, values in FAMILIES.items() for index, word in enumerate(values)],
    })
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "zero_models": zero_models,
        "semantic_scope": "closed-world exact person-and-group conjunction across a two-frame by two-order surface factorial",
        "naturalness_scope": "machine-audited controlled English; no independent human blind review",
        "external_review_corrections": [
            "C073 established a conditional transport domain problem, not semantic-side or physical-phase invariance",
            "behavioral equivalence does not imply whole-state transplantability",
            "a failed directed edge is undefined for mechanism comparison, not a negative result about relative encoding",
            "C074 maps identity-only admissibility and does not search for permutations, coordinates, roles, layers, attention, or MLP mechanisms",
        ],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c074.directional_transport_domain.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "state_index": 16,
        "research_object": "identity-only directed whole-state quartet transport domain over a four-surface factorial",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "heatmap hotspot discovery", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "role subset search", "coordinate search", "semantic permutation competition", "post-reveal threshold or class changes"],
        "roles": list(ROLES), "surfaces": list(SURFACES), "surface_meta": SURFACE_META, "routes": ROUTES,
        "partitions": list(PARTITIONS),
        "material": {
            "active_count": 5760, "composition_count": 72, "minimum_families": 4, "human_naturalness_lock": False,
            "surface_lengths": {key: next(iter(values)) for key, values in lengths.items()},
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"),
        },
        "zero_model_gate": {"maximum_incomplete_balanced_accuracy": 5 / 6, "required_model_balanced_accuracy_min": 0.95},
        "behavior": {
            "family_surface_accuracy_min": 0.95, "family_surface_balanced_accuracy_min": 0.95,
            "family_surface_partition_min": 0.90, "family_surface_truth_min": 0.90,
            "family_surface_cell_min": 0.90, "family_set_all_min": 0.85,
            "same_shape_repeat_max_abs_diff": 1e-6,
        },
        "camera": {
            "known_truth_systems": 256, "qwen_discovery_sets": 12,
            "arms": ["self", "correct_identity", "wrong_identity"],
            "write_max_abs_diff": 1e-4, "untouched_complement_max_abs_diff": 1e-4,
            "self_output_max_abs_diff": 1e-4, "routes": list(ROUTES),
            "directions": ["true_to_false", "false_to_true"],
        },
        "domain": {
            "holdout_sets": 48, "routes": list(ROUTES), "directions": ["true_to_false", "false_to_true"],
            "arms": ["self", "correct_identity", "wrong_identity"],
            "self_max_abs_diff": 1e-4, "identity_desired_sign_fraction_min": 0.90,
            "wrong_expected_sign_fraction_min": 0.90, "family_fraction_min": 0.75,
            "minimum_family_breadth": 4,
            "edge_classes": {
                "robust": "confirmation and lockbox cells both pass independently",
                "split_specific": "exactly one holdout split passes",
                "rejected": "neither holdout split passes",
            },
        },
        "stop_rule": "behavior failure blocks Hidden State; camera failure blocks holdout; each directed edge is classified independently; failed edges do not stop other preregistered edges; all edge outcomes close C074",
        "claim_boundary": {
            "allowed": "directed applicability map for identity whole-state quartet transport at Qwen3 state16 in one controlled task",
            "forbidden": ["semantic-side mechanism", "neuron mechanism", "natural semantic manifold", "necessity or natural use", "relative encoding proven", "cross-model law", "new mathematics established"],
        },
        "branching": {
            "robust_edges_present": "preregister C075 observation atlas on the frozen robust edge set",
            "no_robust_edges": "preregister C075 natural-state-only observation atlas without transport claims",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1446_c074_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
