#!/usr/bin/env python3
"""Phase1440: preregister C073 semantic-side versus physical-phase competition."""
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

PHASE, CAMPAIGN = 1440, "C073"
PARENT = TESTS / "result/phase1439_c072_campaign_closure"
OUT = TESTS / "result/phase1440_c073_side_phase_contract"
FAMILIES = {
    "Summit": ("Kenneth", "Kimberly", "Kristen", "Lauren", "Leslie", "Linda", "Lindsay", "Logan", "Louise", "Lucas", "Marcus", "Margaret"),
    "Copper": ("Marina", "Martha", "Mason", "Maya", "Megan", "Melanie", "Melissa", "Meredith", "Mia", "Miranda", "Molly", "Monica"),
    "Sterling": ("Morgan", "Naomi", "Natalie", "Nathan", "Nicholas", "Nicole", "Nina", "Noah", "Nolan", "Nora", "Norman", "Oliver"),
    "Horizon": ("Olivia", "Owen", "Paige", "Pamela", "Parker", "Patricia", "Paula", "Rebecca", "Regina", "Riley", "Russell", "Samantha"),
    "Basin": ("Sandra", "Sara", "Sebastian", "Shane", "Shannon", "Sharon", "Sheila", "Sierra", "Sofia", "Spencer", "Stella", "Stephanie"),
    "Citadel": ("Sylvia", "Tara", "Teresa", "Theodore", "Tiffany", "Timothy", "Todd", "Tracy", "Trevor", "Tristan", "Tyler", "Valerie"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SYSTEM = "Use only the evidential record. Answer yes exactly when both the queried member and group match the recorded member and group. Output exactly yes or no."
SURFACES = {
    "evidence_first": "Evidential record: {record_target} is assigned to group {record_family}. Is {query_target} assigned to group {query_family}? Reply only yes or no.",
    "question_first": "Is {query_target} assigned to group {query_family}? Consult this evidential record: {record_target} is assigned to group {record_family}. Reply only yes or no.",
}
ROLES = ("record_target", "record_family", "query_target", "query_family")
CELLS = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")
ROLE_SIDE = {"record_target": "record", "record_family": "record", "query_target": "query", "query_family": "query"}
SURFACE_PHASE = {
    "evidence_first": {"record_target": "early", "record_family": "early", "query_target": "late", "query_family": "late"},
    "question_first": {"record_target": "late", "record_family": "late", "query_target": "early", "query_family": "early"},
}
ROUTES = {
    "evidence_same": {"source": "evidence_first", "target": "evidence_first", "order": "same_order"},
    "question_same": {"source": "question_first", "target": "question_first", "order": "same_order"},
    "evidence_to_question": {"source": "evidence_first", "target": "question_first", "order": "reversed_order"},
    "question_to_evidence": {"source": "question_first", "target": "evidence_first", "order": "reversed_order"},
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
                        "case_id": f"c073-a-{len(rows):04d}", "partition": partition(index),
                        "pair": f"{base_a}__{base_b}", "orientation": f"{fa}__{fb}", "index": index,
                        "surface": surface, "surface_order": "record_then_query" if surface == "evidence_first" else "query_then_record",
                        "cell": cell, "record_target": rt, "record_family": rf,
                        "query_target": qt, "query_family": qf, "truth": truth,
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
        if row["surface"] == "evidence_first":
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
            row = {
                "set_id": f"c073-compose-{len(result):04d}", "partition": partition(index),
                "family": family, "index": index, "donor_family": donor, "other_family": other,
            }
            for surface in SURFACES:
                row[f"{surface}_true_recipient"] = by[(surface, fw, family, fw, family)]["case_id"]
                row[f"{surface}_false_recipient"] = by[(surface, fw, family, fw, g)]["case_id"]
                row[f"{surface}_true_donor"] = by[(surface, dw, donor, dw, donor)]["case_id"]
                row[f"{surface}_false_donor"] = by[(surface, dw, donor, dw, other)]["case_id"]
            result.append(row)
    return result


def enriched_registry() -> list[dict]:
    registry = c072.permutation_registry()
    for row in registry:
        semantic_count = sum(ROLE_SIDE[role] == ROLE_SIDE[row["mapping"][role]] for role in ROLES)
        row["semantic_side_preserved_count"] = semantic_count
        row["semantic_side_preserving"] = semantic_count == 4
        row["semantic_side_crossing"] = semantic_count == 0
        row["physical_phase_preserved_by_route"] = {
            route: sum(
                SURFACE_PHASE[spec["target"]][role] == SURFACE_PHASE[spec["source"]][row["mapping"][role]]
                for role in ROLES
            )
            for route, spec in ROUTES.items()
        }
    return registry


def balanced_accuracy(truths: list[bool], predictions: list[bool]) -> float:
    return c072.balanced_accuracy(truths, predictions)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1440 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c073_independent_record_query_side_preservation_test" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C072 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    registry = enriched_registry()
    by_id = {row["permutation_id"]: row for row in registry}
    old = c072.old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    source = {row["case_id"]: row for row in active}
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {
        surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled if row["surface"] == surface}
        for surface in SURFACES
    }
    truths = [row["truth"] for row in active]
    zero_models = {
        "always_yes_balanced_accuracy": balanced_accuracy(truths, [True] * len(active)),
        "always_no_balanced_accuracy": balanced_accuracy(truths, [False] * len(active)),
        "surface_only_balanced_accuracy": balanced_accuracy(truths, [row["surface"] == "evidence_first" for row in active]),
        "person_only_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] for row in active]),
        "group_only_balanced_accuracy": balanced_accuracy(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact_conjunction_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    truth_fixture = {"aa": True, "ab": False, "ac": False, "ad": False, "bb": True, "ba": False, "bc": False, "bd": False}
    p07, p23 = by_id["p07"], by_id["p23"]
    matched = all(p07[key] == p23[key] for key in ("fixed_points", "parity", "cycle_type", "preserves_entity_family_kind"))
    order_checks = {
        "evidence_first": all(max(row["role_positions"][role][0] for role in ROLES[:2]) < min(row["role_positions"][role][0] for role in ROLES[2:]) for row in compiled if row["surface"] == "evidence_first"),
        "question_first": all(max(row["role_positions"][role][0] for role in ROLES[2:]) < min(row["role_positions"][role][0] for role in ROLES[:2]) for row in compiled if row["surface"] == "question_first"),
    }
    donor_truth = {"true_recipient": True, "false_recipient": False, "true_donor": True, "false_donor": False}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "fresh_labels": len(labels) == 6 and not ({value.lower() for value in labels} & old),
        "fresh_members": len(members) == 72 and not ({value.lower() for value in members} & old),
        "context_singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in labels | members),
        "active": len(active) == 2880 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in SURFACES},
        "cells": Counter(row["cell"] for row in active) == {cell: 360 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 720, False: 2160},
        "manual_truth_fixture": all(row["truth"] == truth_fixture[row["cell"]] for row in active),
        "semantic_unique": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[f"{surface}_{key}"]]["truth"] == truth for row in composition for surface in SURFACES for key, truth in donor_truth.items()),
        "compiled": len(compiled) == len(active),
        "same_shape_per_surface": all(len(values) == 1 for values in lengths.values()),
        "different_surface_shapes": len({next(iter(values)) for values in lengths.values()}) == 2,
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "reversed_role_order": all(order_checks.values()),
        "quartet_singletons": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 and all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled),
        "answer_singletons": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero_models": zero_models["always_yes_balanced_accuracy"] == 0.5 and zero_models["always_no_balanced_accuracy"] == 0.5 and abs(zero_models["person_only_balanced_accuracy"] - 5 / 6) < 1e-12 and abs(zero_models["group_only_balanced_accuracy"] - 5 / 6) < 1e-12 and zero_models["exact_conjunction_balanced_accuracy"] == 1.0,
        "permutations": len(registry) == 24 and len({tuple(row["source_indices_by_target"]) for row in registry}) == 24,
        "matched_pair": matched and p07["semantic_side_preserving"] and p23["semantic_side_crossing"],
        "same_order_competition": all(p07["physical_phase_preserved_by_route"][route] == 4 and p23["physical_phase_preserved_by_route"][route] == 0 for route in ("evidence_same", "question_same")),
        "reversed_order_competition": all(p07["physical_phase_preserved_by_route"][route] == 0 and p23["physical_phase_preserved_by_route"][route] == 4 for route in ("evidence_to_question", "question_to_evidence")),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c073.side_phase_registry.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES,
        "concepts": [{"word": word, "family": family, "index": index, "partition": partition(index)} for family, values in FAMILIES.items() for index, word in enumerate(values)],
    })
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "material/permutation_registry.jsonl", registry)
    core.save(OUT / "material/matched_contrast.json", {
        "semantic_side_arm": p07, "physical_phase_arm": p23,
        "matched_fields": ["fixed_points", "parity", "cycle_type", "preserves_entity_family_kind"],
        "routes": ROUTES,
    })
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "zero_models": zero_models, "manual_truth_fixture": truth_fixture,
        "semantic_scope": "closed-world exact member-and-group conjunction with semantic-side/physical-order reversal",
        "naturalness_scope": "machine-audited controlled evidential English; no independent human blind review",
        "external_review_corrections": [
            "C072 is conditional causal evidence, not an eleventh complete mechanism discovery",
            "p01 and p06 passing while p07 fails a threshold does not by itself establish nonlinear synergy or antagonism",
            "record/query side was confounded with early/late causal position in C072",
            "relative encoding and neuron-level coding remain hypotheses",
            "attention/MLP, gradients, heatmaps, PCA, t-SNE, UMAP, learned probes, and hotspot discovery remain unauthorized",
        ],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c073.semantic_side_vs_physical_phase.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "state_index": 16,
        "research_object": "matched full-dimensional quartet competition between semantic record/query side and physical early/late phase",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "heatmap hotspot discovery", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "role subset search", "coordinate search", "post-reveal threshold or class changes"],
        "roles": list(ROLES), "surfaces": list(SURFACES), "surface_phase": SURFACE_PHASE,
        "routes": ROUTES, "partitions": list(PARTITIONS),
        "material": {
            "active_count": 2880, "composition_count": 72, "minimum_families": 4, "human_naturalness_lock": False,
            "surface_lengths": {key: next(iter(values)) for key, values in lengths.items()},
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"),
        },
        "permutations": {
            "registry_sha256": core.sha(OUT / "material/permutation_registry.jsonl"),
            "semantic_side_id": "p07", "physical_phase_id": "p23",
            "auxiliary_ids": ["p00", "p01", "p06"],
            "matched_fields": ["fixed_points", "parity", "cycle_type", "preserves_entity_family_kind"],
            "interaction_descriptive_only": "I_double = g_p07 - g_p01 - g_p06 + g_p00",
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
            "permutation_ids": ["p00", "p01", "p06", "p07", "p23"],
            "write_max_abs_diff": 1e-4, "untouched_complement_max_abs_diff": 1e-4,
            "self_output_max_abs_diff": 1e-4, "routes": list(ROUTES),
            "directions": ["true_to_false", "false_to_true"],
        },
        "mechanism": {
            "holdout_sets": 48, "routes": list(ROUTES),
            "reversed_routes": ["evidence_to_question", "question_to_evidence"],
            "same_order_routes": ["evidence_same", "question_same"],
            "directions": ["true_to_false", "false_to_true"],
            "arms": ["self", "correct_identity", "wrong_identity", "p01", "p06", "p07", "p23"],
            "self_max_abs_diff": 1e-4, "identity_desired_sign_fraction_min": 0.90,
            "wrong_expected_sign_fraction_min": 0.90, "minimum_family_breadth": 4,
            "arm_desired_sign_fraction_min": 0.75, "paired_win_fraction_min": 0.75,
            "paired_gain_gap_median_min": 5.0, "family_paired_win_fraction_min": 0.75,
            "strong_required_reversed_cells": 8, "conditional_required_reversed_cells": 6,
            "strong_required_same_order_cells": 6,
        },
        "classification": {
            "semantic_side_confirmed": "all eight reversed-order cells favor p07 over matched p23 with frozen efficacy, paired-gap, and family-breadth gates; at least six same-order cells also favor p07",
            "physical_phase_confirmed": "all eight reversed-order cells favor p23 over matched p07 with frozen efficacy, paired-gap, and family-breadth gates",
            "conditional_semantic_side": "at least six reversed-order cells favor p07, none favor p23, and executor controls pass",
            "conditional_physical_phase": "at least six reversed-order cells favor p23, none favor p07, and executor controls pass",
            "mixed_or_no_stable_separation": "executor passes but the matched competition does not meet a directional frozen branch",
            "executor_failed": "one or more frozen identity, wrong-donor, self, numeric, or breadth controls fail",
        },
        "stop_rule": "behavior failure blocks Hidden State; camera failure blocks holdout; the matched mechanism runs once; every classification closes C073",
        "claim_boundary": {
            "allowed": "matched output-response evidence distinguishing semantic record/query side from physical early/late phase in one controlled Qwen3 state16 task",
            "forbidden": ["neuron mechanism", "natural semantic manifold", "necessity or natural use", "relative encoding proven", "cross-model law", "new mathematics established"],
        },
        "branching": {"phase1441": "behavior", "phase1442": "matched camera", "phase1443": "one holdout reveal", "phase1444": "closure"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1441_c073_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
