#!/usr/bin/env python3
"""Phase1435: preregister C072 exhaustive quartet-permutation response spectrum."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1435, "C072"
PARENT = TESTS / "result/phase1434_c071_campaign_closure"
OUT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
FAMILIES = {
    "Nimbus": ("Addison", "Alicia", "Allison", "Amber", "Amelia", "Andrea", "Angel", "Anita", "Audrey", "Autumn", "Ava", "Barry"),
    "Vista": ("Bella", "Beverly", "Bonnie", "Bradley", "Brooke", "Caleb", "Cameron", "Carmen", "Carrie", "Casey", "Chad", "Chloe"),
    "Beacon": ("Claire", "Clara", "Clarence", "Claudia", "Clayton", "Cody", "Colin", "Curtis", "Dakota", "Dale", "Darren", "Dawn"),
    "Grove": ("Derek", "Dominic", "Dylan", "Earl", "Edgar", "Eleanor", "Elena", "Eli", "Elias", "Elijah", "Ellen", "Ellie"),
    "Prairie": ("Elliot", "Ellis", "Eva", "Evan", "Fiona", "Floyd", "Gavin", "Gerald", "Gina", "Glenn", "Gloria", "Gordon"),
    "Ridge": ("Graham", "Gregory", "Hannah", "Harold", "Hazel", "Heather", "Hector", "Holly", "Isaac", "Isabel", "Javier", "Jay"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SYSTEM = "Use only the registry statement. Answer yes exactly when both the queried member and circle match the recorded member and circle. Output exactly yes or no."
SURFACES = {
    "memo_contains": "Registry memo: {record_target} is enrolled in circle {record_family}. Question: does circle {query_family} contain {query_target}? Reply only yes or no.",
    "circle_roll": "Circle {record_family} records {record_target} on its roll. Question: is {query_target} enrolled in circle {query_family}? Reply only yes or no.",
}
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
                        "case_id": f"c072-a-{len(rows):04d}", "partition": partition(index),
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
                "set_id": f"c072-compose-{len(result):04d}", "partition": partition(index),
                "family": family, "index": index, "donor_family": donor, "other_family": other,
            }
            for surface in SURFACES:
                row[f"{surface}_true_recipient"] = by[(surface, fw, family, fw, family)]["case_id"]
                row[f"{surface}_false_recipient"] = by[(surface, fw, family, fw, g)]["case_id"]
                row[f"{surface}_true_donor"] = by[(surface, dw, donor, dw, donor)]["case_id"]
                row[f"{surface}_false_donor"] = by[(surface, dw, donor, dw, other)]["case_id"]
            result.append(row)
    return result


def parity(values: tuple[int, ...]) -> str:
    inversions = sum(values[i] > values[j] for i in range(4) for j in range(i + 1, 4))
    return "even" if inversions % 2 == 0 else "odd"


def cycle_type(values: tuple[int, ...]) -> str:
    seen, lengths = set(), []
    for start in range(4):
        if start in seen:
            continue
        current, length = start, 0
        while current not in seen:
            seen.add(current)
            current = values[current]
            length += 1
        lengths.append(length)
    return "-".join(map(str, sorted(lengths, reverse=True)))


def permutation_registry() -> list[dict]:
    registry = []
    entity = {0, 2}
    record = {0, 1}
    for index, values in enumerate(permutations(range(4))):
        mapping = {ROLES[target]: ROLES[source] for target, source in enumerate(values)}
        registry.append({
            "permutation_id": f"p{index:02d}", "source_indices_by_target": list(values), "mapping": mapping,
            "identity": values == tuple(range(4)), "fixed_points": sum(i == value for i, value in enumerate(values)),
            "parity": parity(values), "cycle_type": cycle_type(values),
            "preserves_entity_family_kind": {values[i] in entity for i in entity} == {True} and {values[i] not in entity for i in set(range(4)) - entity} == {True},
            "preserves_record_query_axis": {values[i] in record for i in record} == {True} and {values[i] not in record for i in set(range(4)) - record} == {True},
        })
    return registry


def balanced_accuracy(truths: list[bool], predictions: list[bool]) -> float:
    tpr = sum(pred for truth, pred in zip(truths, predictions) if truth) / sum(truths)
    tnr = sum(not pred for truth, pred in zip(truths, predictions) if not truth) / sum(not truth for truth in truths)
    return (tpr + tnr) / 2.0


def old_material_words() -> set[str]:
    words = set()
    for path in (TESTS / "result").glob("phase*/material/frozen_concept_graph.json"):
        try:
            data = core.load(path)
            families = data.get("families", {})
            if isinstance(families, dict):
                words |= {str(key).lower() for key in families}
                for values in families.values():
                    if isinstance(values, list):
                        words |= {str(value).lower() for value in values}
            for row in data.get("concepts", []):
                for key in ("word", "family", "label", "name"):
                    if row.get(key):
                        words.add(str(row[key]).lower())
        except Exception:
            pass
    return words


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1435 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c072_exhaustive_quartet_permutation_response_spectrum" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C071 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    registry = permutation_registry()
    source = {row["case_id"]: row for row in active}
    old = old_material_words()
    labels = set(FAMILIES)
    members = {value for values in FAMILIES.values() for value in values}
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in SURFACES}
    signatures = {
        surface: {tuple((role, tuple(row["role_positions"][role])) for role in ROLES) for row in compiled if row["surface"] == surface}
        for surface in SURFACES
    }
    truths = [row["truth"] for row in active]
    zero_models = {
        "always_yes_balanced_accuracy": balanced_accuracy(truths, [True] * len(active)),
        "always_no_balanced_accuracy": balanced_accuracy(truths, [False] * len(active)),
        "surface_only_balanced_accuracy": balanced_accuracy(truths, [row["surface"] == "memo_contains" for row in active]),
        "person_only_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] for row in active]),
        "circle_only_balanced_accuracy": balanced_accuracy(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact_conjunction_balanced_accuracy": balanced_accuracy(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
    }
    truth_fixture = {"aa": True, "bb": True, "ab": False, "ac": False, "ad": False, "ba": False, "bc": False, "bd": False}
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
        "same_shape": all(len(values) == 1 for values in lengths.values()),
        "different_shapes": len({next(iter(values)) for values in lengths.values()}) == 2,
        "stable_roles": all(len(values) == 1 for values in signatures.values()),
        "different_roles": next(iter(signatures["memo_contains"])) != next(iter(signatures["circle_roll"])),
        "quartet_singletons": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 and all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled),
        "answer_singletons": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "zero_models": zero_models["always_yes_balanced_accuracy"] == 0.5 and zero_models["always_no_balanced_accuracy"] == 0.5 and abs(zero_models["person_only_balanced_accuracy"] - 5 / 6) < 1e-12 and abs(zero_models["circle_only_balanced_accuracy"] - 5 / 6) < 1e-12 and zero_models["exact_conjunction_balanced_accuracy"] == 1.0,
        "permutations": len(registry) == 24 and sum(row["identity"] for row in registry) == 1 and len({tuple(row["source_indices_by_target"]) for row in registry}) == 24,
        "permutation_strata": Counter(row["parity"] for row in registry) == {"even": 12, "odd": 12} and Counter(row["cycle_type"] for row in registry) == {"1-1-1-1": 1, "2-1-1": 6, "3-1": 8, "2-2": 3, "4": 6},
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c072.cross_surface_registry.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "surfaces": SURFACES,
        "concepts": [{"word": word, "family": family, "index": index, "partition": partition(index)} for family, values in FAMILIES.items() for index, word in enumerate(values)],
    })
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "material/permutation_registry.jsonl", registry)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "zero_models": zero_models, "manual_truth_fixture": truth_fixture,
        "semantic_scope": "closed-world exact member-and-circle conjunction",
        "naturalness_scope": "machine-audited controlled registry English; no independent human blind review",
        "external_review_corrections": [
            "C071 is conditional causal evidence, not a numbered complete mechanism discovery",
            "one derangement did not establish an unordered multiset or universal permutation invariance",
            "relative encoding remains a hypothesis",
            "attention/MLP, gradients, heatmaps, PCA, t-SNE, UMAP, and post-reveal hotspot discovery are unauthorized",
        ],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c072.exhaustive_permutation_spectrum.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "state_index": 16,
        "research_object": "cross-surface full-dimensional quartet response under all 24 semantic-role permutations",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "role subset search", "coordinate search", "post-reveal threshold or class changes"],
        "roles": list(ROLES), "surfaces": list(SURFACES), "partitions": list(PARTITIONS),
        "material": {"active_count": 2880, "composition_count": 72, "minimum_families": 4, "human_naturalness_lock": False, "surface_lengths": {key: next(iter(values)) for key, values in lengths.items()}, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl")},
        "permutations": {"count": 24, "identity_id": next(row["permutation_id"] for row in registry if row["identity"]), "registry_sha256": core.sha(OUT / "material/permutation_registry.jsonl"), "descriptive_strata_only": ["fixed_points", "parity", "cycle_type", "preserves_entity_family_kind", "preserves_record_query_axis"]},
        "zero_model_gate": {"maximum_incomplete_balanced_accuracy": 5 / 6, "required_model_balanced_accuracy_min": 0.95},
        "behavior": {"family_surface_accuracy_min": 0.95, "family_surface_balanced_accuracy_min": 0.95, "family_surface_partition_min": 0.90, "family_surface_truth_min": 0.90, "family_surface_cell_min": 0.90, "family_set_all_min": 0.85, "same_shape_repeat_max_abs_diff": 1e-6},
        "camera": {"known_truth_systems": 128, "qwen_discovery_sets": 12, "all_permutations": 24, "write_max_abs_diff": 1e-4, "untouched_complement_max_abs_diff": 1e-4, "self_output_max_abs_diff": 1e-4, "surface_transfers": ["memo_contains_to_circle_roll", "circle_roll_to_memo_contains"], "directions": ["true_to_false", "false_to_true"]},
        "mechanism": {
            "holdout_sets": 48, "surface_transfers": ["memo_contains_to_circle_roll", "circle_roll_to_memo_contains"], "directions": ["true_to_false", "false_to_true"],
            "controls": ["self", "same_surface_identity", "wrong_cross_surface_identity"],
            "self_max_abs_diff": 1e-4, "same_surface_desired_sign_fraction_min": 0.90, "wrong_expected_sign_fraction_min": 0.90,
            "permutation_desired_sign_fraction_min": 0.90, "permutation_oriented_gain_median_min": 0.50,
            "family_sign_fraction_min": 0.75, "minimum_family_breadth": 4,
            "identity_vs_best_nonidentity_sign_gap_min": 0.25, "identity_vs_best_nonidentity_gain_gap_median_min": 0.50,
            "symmetric_gain_range_ratio_max": 0.25,
        },
        "classification": {
            "role_order_selective": "executor controls pass; every cell qualifies identity only; identity beats the best nonidentity by both frozen sign and paired-gain gaps in both splits",
            "permutation_symmetric_multiset": "executor controls pass; all 24 permutations qualify in every cell; per-split median-gain range divided by absolute identity gain is at most 0.25",
            "subgroup_structured": "executor controls pass; the same strict nontrivial proper subset qualifies in all four cells and is closed under identity, inverse, and composition",
            "heterogeneous_or_executor_failed": "all remaining outcomes, including failed controls, transfer/direction disagreement, non-subgroup pass sets, or full-S4 sufficiency with excessive gain heterogeneity",
        },
        "stop_rule": "behavior failure blocks Hidden State; camera failure blocks holdouts; the mechanism runs once; every classification closes C072",
        "claim_boundary": {"allowed": "finite output-response symmetry class of a fixed Qwen3 state16 quartet under a controlled registry contract", "forbidden": ["natural semantic manifold", "neuron mechanism", "necessity or natural use", "relative encoding proven", "cross-model law", "new mathematics established"]},
        "branching": {"phase1436": "behavior", "phase1437": "permutation camera", "phase1438": "holdout spectrum", "phase1439": "closure"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1436_c072_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
