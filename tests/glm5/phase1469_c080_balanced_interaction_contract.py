#!/usr/bin/env python3
"""Phase1469: preregister the complete C080 balanced interaction campaign."""
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
import phase1456_c077_labeled_relation_contract as c077
import phase1460_c078_colon_label_contract as c078
import phase1463_c079_aggregate_observation_contract as c079
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1469, "C080"
PARENT = TESTS / "result/phase1468_c079_campaign_closure"
OUT = TESTS / "result/phase1469_c080_balanced_interaction_contract"

FAMILIES = {
    "Academy": ("Abel", "Antoine", "Ariel", "Augustine", "Basil", "Beau"),
    "Alliance": ("Belle", "Ben", "Bo", "Boris", "Bowen", "Bruno"),
    "Bureau": ("Burton", "Carolina", "Carson", "Chance", "Chandler", "Clement"),
    "Council": ("Conrad", "Donovan", "Duncan", "Eddie", "Elsa", "Emil"),
    "Guild": ("Flynn", "Griffin", "Holden", "Jed", "Jeremiah", "Jonas"),
    "Society": ("Lincoln", "Marcel", "Milo", "Nico", "Omar", "Orlando"),
}
RELATIONS = c077.RELATIONS
IDS = tuple(RELATIONS)
ORDER = tuple(FAMILIES)
PAIRS = tuple(combinations(IDS, 2))
PAIR_IDS = tuple(f"{left}__{right}" for left, right in PAIRS)
PARTITIONS = c077.PARTITIONS
NUISANCE_CELLS = tuple(f"{entity}{obj}" for entity in (1, 0) for obj in (1, 0))
PAIR_CORNERS = ("AA", "AB", "BA", "BB")
EXPLICIT_SYSTEM = (
    "Compare only the two explicit labels. Answer yes exactly when the labels are identical. "
    "The two notes are context. Output exactly yes or no."
)
WITHDRAWAL_SYSTEM = (
    "Compare the semantic relation expressed by the two action verbs. Inflected forms of the same "
    "relation count as identical. Answer exactly yes or no."
)
EXPLICIT_SURFACES = {
    "a_explicit": (
        "First label: {record_label}. First note: {record_target} saw {record_object}. "
        "Second label: {query_label}. Second note: {query_target} may see {query_object}. "
        "Are the two labels identical? Answer only yes or no."
    ),
    "b_explicit": (
        "Recorded label: {record_label}. Recorded note: {record_target} saw {record_object}. "
        "Queried label: {query_label}. Queried note: {query_target} can see {query_object}. "
        "Do the labels match exactly? Reply only yes or no."
    ),
}
WITHDRAWAL_SURFACES = {
    "a_natural": (
        "First fact: {record_target} {record_relation} {record_object}. Second possibility: "
        "{query_target} may {query_relation} {query_object}. Do the two actions express the same "
        "relation? Answer only yes or no."
    ),
    "b_natural": (
        "Recorded action: {record_target} {record_relation} {record_object}. Queried action: "
        "{query_target} can {query_relation} {query_object}. Do the recorded and queried actions "
        "belong to the same relation type? Reply only yes or no."
    ),
}
EXPLICIT_ROLES = (
    "record_label", "record_target", "record_relation", "record_object",
    "query_label", "query_target", "query_relation", "query_object", "boundary",
)
WITHDRAWAL_ROLES = (
    "record_target", "record_relation", "record_object",
    "query_target", "query_relation", "query_object", "boundary",
)


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def pair_id(left: str, right: str) -> str:
    order = {value: index for index, value in enumerate(IDS)}
    a, b = sorted((left, right), key=order.__getitem__)
    return f"{a}__{b}"


def active_cases(branch: str) -> list[dict]:
    explicit = branch == "explicit"
    surfaces = EXPLICIT_SURFACES if explicit else WITHDRAWAL_SURFACES
    rows: list[dict] = []
    for family_index, family in enumerate(ORDER):
        other_family = ORDER[(family_index + 1) % len(ORDER)]
        for index in range(6):
            record_target = FAMILIES[family][index]
            other_target = FAMILIES[other_family][index]
            for record_id in IDS:
                for query_id in IDS:
                    for surface, template in surfaces.items():
                        for entity_match in (1, 0):
                            for object_match in (1, 0):
                                row = {
                                    "case_id": f"c080-{branch[0]}-{len(rows):05d}",
                                    "branch": branch,
                                    "partition": partition(index),
                                    "family": family,
                                    "other_family": other_family,
                                    "index": index,
                                    "surface": surface,
                                    "nuisance_cell": f"{entity_match}{object_match}",
                                    "record_relation_id": record_id,
                                    "query_relation_id": query_id,
                                    "record_label": RELATIONS[record_id]["label"],
                                    "query_label": RELATIONS[query_id]["label"],
                                    "record_target": record_target,
                                    "query_target": record_target if entity_match else other_target,
                                    "record_object": family,
                                    "query_object": family if object_match else other_family,
                                    "record_relation": "saw" if explicit else RELATIONS[record_id]["record"],
                                    "query_relation": "see" if explicit else RELATIONS[query_id]["query"],
                                    "entity_match": bool(entity_match),
                                    "object_match": bool(object_match),
                                    "truth": record_id == query_id,
                                    "candidates": ["yes", "no"],
                                    "gold_position": 0 if record_id == query_id else 1,
                                }
                                row["prompt"] = template.format(**row)
                                rows.append(row)
    return rows


def compile_rows(tok, rows: list[dict], system: str, roles: tuple[str, ...]) -> list[dict]:
    result = []
    content_roles = tuple(role for role in roles if role != "boundary")
    for row in rows:
        ids = core.chat_ids(tok, system, row["prompt"])
        spans = {role: c072.all_spans(tok, ids, row[role]) for role in content_roles}
        if not all(spans.values()):
            raise RuntimeError((row["case_id"], spans))
        positions = {}
        for role in content_roles:
            positions[role] = spans[role][0] if role.startswith("record_") else spans[role][-1]
        positions["boundary"] = [len(ids) - 1]
        result.append({
            **row,
            "prompt_ids": ids,
            "role_positions": positions,
            "candidate_ids": [list(map(int, tok.encode(" " + value, add_special_tokens=False))) for value in row["candidates"]],
        })
    return result


def interaction_sets(rows: list[dict], branch: str) -> list[dict]:
    lookup = {
        (row["family"], row["index"], row["surface"], row["nuisance_cell"], row["record_relation_id"], row["query_relation_id"]): row["case_id"]
        for row in rows
    }
    surfaces = EXPLICIT_SURFACES if branch == "explicit" else WITHDRAWAL_SURFACES
    result = []
    for family in ORDER:
        for index in range(6):
            for left, right in PAIRS:
                row = {
                    "set_id": f"c080-{branch[0]}-interaction-{len(result):04d}",
                    "branch": branch,
                    "partition": partition(index),
                    "family": family,
                    "index": index,
                    "pair_id": pair_id(left, right),
                    "label_a": left,
                    "label_b": right,
                }
                corners = {"AA": (left, left), "AB": (left, right), "BA": (right, left), "BB": (right, right)}
                for surface in surfaces:
                    for nuisance in NUISANCE_CELLS:
                        for corner, (record_id, query_id) in corners.items():
                            row[f"{surface}_{nuisance}_{corner}"] = lookup[(family, index, surface, nuisance, record_id, query_id)]
                result.append(row)
    return result


def zero_models(rows: list[dict]) -> dict[str, float]:
    truth = [row["truth"] for row in rows]
    first_label = IDS[0]
    return {
        "always_yes": c077.ba(truth, [True] * len(rows)),
        "always_no": c077.ba(truth, [False] * len(rows)),
        "surface": c077.ba(truth, [row["surface"].startswith("a_") for row in rows]),
        "entity": c077.ba(truth, [row["entity_match"] for row in rows]),
        "object": c077.ba(truth, [row["object_match"] for row in rows]),
        "record_label_only": c077.ba(truth, [row["record_relation_id"] == first_label for row in rows]),
        "query_label_only": c077.ba(truth, [row["query_relation_id"] == first_label for row in rows]),
        "identity_oracle": c077.ba(truth, [row["record_relation_id"] == row["query_relation_id"] for row in rows]),
    }


def branch_checks(tok, rows, compiled, sets, surfaces, roles) -> dict[str, bool]:
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in surfaces}
    signatures = {
        surface: {tuple((role, tuple(row["role_positions"][role])) for role in roles) for row in compiled if row["surface"] == surface}
        for surface in surfaces
    }
    truth = Counter(row["truth"] for row in rows)
    ordered_pairs = Counter((row["record_relation_id"], row["query_relation_id"]) for row in rows)
    zero = zero_models(rows)
    return {
        "active_count": len(rows) == 10368,
        "surface_balance": Counter(row["surface"] for row in rows) == {surface: 5184 for surface in surfaces},
        "truth_counts": truth == {False: 8640, True: 1728},
        "ordered_pair_balance": len(ordered_pairs) == 36 and len(set(ordered_pairs.values())) == 1,
        "nuisance_balance": all(Counter(row[key] for row in rows) == {True: 5184, False: 5184} for key in ("entity_match", "object_match")),
        "semantic_unique": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in rows),
        "compiled_count": len(compiled) == len(rows),
        "same_length_per_surface": all(len(values) == 1 for values in lengths.values()),
        "stable_role_positions": all(len(values) == 1 for values in signatures.values()),
        "singleton_role_spans": all(all(len(row["role_positions"][role]) == 1 for role in roles) for row in compiled),
        "distinct_role_positions": all(len({row["role_positions"][role][0] for role in roles}) == len(roles) for row in compiled),
        "candidate_singletons": all(all(len(values) == 1 for values in row["candidate_ids"]) for row in compiled),
        "interaction_sets": len(sets) == 540 and Counter(row["partition"] for row in sets) == {name: 180 for name in PARTITIONS},
        "pair_set_balance": Counter(row["pair_id"] for row in sets) == {value: 36 for value in PAIR_IDS},
        "complete_four_grids": all(len([key for key in row if key.endswith(tuple(PAIR_CORNERS))]) == len(surfaces) * len(NUISANCE_CELLS) * 4 for row in sets),
        "zero_models": all(value == 0.5 for key, value in zero.items() if key != "identity_oracle") and zero["identity_oracle"] == 1.0,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1469 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c080_balanced_equality_interaction_and_label_withdrawal_campaign" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1468 did not authorize C080")
    tok = tokenizer()
    prior_words = set(c072.old_material_words())
    prior_words |= {value.lower() for source in (c078.FAMILIES, c079.FAMILIES) for value in source}
    prior_words |= {value.lower() for source in (c078.FAMILIES, c079.FAMILIES) for values in source.values() for value in values}
    new_groups = set(FAMILIES)
    new_people = {value for values in FAMILIES.values() for value in values}
    lexical_values = new_groups | new_people | {value["label"] for value in RELATIONS.values()} | {"saw", "see"} | {value[key] for value in RELATIONS.values() for key in ("record", "query")}
    explicit = active_cases("explicit")
    withdrawal = active_cases("withdrawal")
    explicit_compiled = compile_rows(tok, explicit, EXPLICIT_SYSTEM, EXPLICIT_ROLES)
    withdrawal_compiled = compile_rows(tok, withdrawal, WITHDRAWAL_SYSTEM, WITHDRAWAL_ROLES)
    explicit_sets = interaction_sets(explicit, "explicit")
    withdrawal_sets = interaction_sets(withdrawal, "withdrawal")
    explicit_checks = branch_checks(tok, explicit, explicit_compiled, explicit_sets, EXPLICIT_SURFACES, EXPLICIT_ROLES)
    withdrawal_checks = branch_checks(tok, withdrawal, withdrawal_compiled, withdrawal_sets, WITHDRAWAL_SURFACES, WITHDRAWAL_ROLES)
    checks = {
        "parent_authorized": parent_audit["all_checks_passed"],
        "fresh_groups": len(new_groups) == 6 and not ({value.lower() for value in new_groups} & prior_words),
        "fresh_people": len(new_people) == 36 and not ({value.lower() for value in new_people} & prior_words),
        "single_token_lexicon": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in lexical_values),
        "pair_count": len(PAIRS) == 15 and len(PAIR_IDS) == 15,
        "explicit": all(explicit_checks.values()),
        "withdrawal": all(withdrawal_checks.values()),
        "hidden_not_accessed": True,
        "human_naturalness_not_claimed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    for branch, rows, compiled, sets in (
        ("explicit", explicit, explicit_compiled, explicit_sets),
        ("withdrawal", withdrawal, withdrawal_compiled, withdrawal_sets),
    ):
        core.write_rows(OUT / f"material/{branch}_active_cases.jsonl", rows)
        core.write_rows(OUT / f"compiled/qwen3_{branch}.jsonl", compiled)
        core.write_rows(OUT / f"material/{branch}_interaction_sets.jsonl", sets)
    core.save(OUT / "material/frozen_language_graph.json", {
        "schema": "c080.balanced_interaction.v1",
        "families": FAMILIES,
        "relations": RELATIONS,
        "pair_ids": list(PAIR_IDS),
        "partitions": {key: list(value) for key, value in PARTITIONS.items()},
        "explicit_surfaces": EXPLICIT_SURFACES,
        "withdrawal_surfaces": WITHDRAWAL_SURFACES,
    })
    preaudit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "branch_checks": {"explicit": explicit_checks, "withdrawal": withdrawal_checks},
        "zero_models": {"explicit": zero_models(explicit), "withdrawal": zero_models(withdrawal)},
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "semantic_scope": "known-truth equality of six relation identities over the complete ordered 6x6 matrix",
        "naturalness_scope": "machine-audited controlled English; no independent human naturalness lock",
        "hidden_state_accessed": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    behavior_common = {
        "surface_partition_balanced_accuracy_min": 0.97,
        "surface_truth_accuracy_min": 0.97,
        "nuisance_surface_balanced_accuracy_min": 0.97,
        "equal_label_surface_accuracy_min": 0.95,
        "unequal_pair_surface_accuracy_min": 0.95,
        "eligible_set_total_min": 480,
        "eligible_set_split_min": 160,
        "eligible_set_pair_min": 30,
        "same_batch_repeat_max_abs_diff": 1e-6,
    }
    withdrawal_behavior = dict(behavior_common)
    withdrawal_behavior.update({
        "global_surface_balanced_accuracy_min": 0.95,
        "surface_partition_balanced_accuracy_min": 0.90,
        "surface_truth_accuracy_min": 0.90,
        "nuisance_surface_balanced_accuracy_min": 0.90,
        "equal_label_surface_accuracy_min": 0.90,
        "unequal_pair_surface_accuracy_min": 0.90,
        "eligible_set_total_min": 420,
        "eligible_set_split_min": 140,
        "eligible_set_pair_min": 24,
    })
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c080.balanced_equality_interaction_and_label_withdrawal.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "full-dimensional second-order equality interaction after additive record/query identity main effects are canceled",
        "interaction_formula": "I_AB = 0.5 * (H_AA + H_BB - H_AB - H_BA)",
        "off_diagonal_null_formula": "J_AB;CD = 0.5 * (H_AC + H_BD - H_AD - H_BC), with A,B,C,D distinct",
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "t-SNE", "UMAP", "learned probe", "coordinate pruning", "post-unblind changes"],
        "partitions": list(PARTITIONS),
        "pair_ids": list(PAIR_IDS),
        "nuisance_cells": list(NUISANCE_CELLS),
        "pair_corners": list(PAIR_CORNERS),
        "branches": {
            "explicit": {
                "roles": list(EXPLICIT_ROLES),
                "surfaces": list(EXPLICIT_SURFACES),
                "active_count": len(explicit),
                "interaction_set_count": len(explicit_sets),
                "active_sha256": core.sha(OUT / "material/explicit_active_cases.jsonl"),
                "compiled_sha256": core.sha(OUT / "compiled/qwen3_explicit.jsonl"),
                "sets_sha256": core.sha(OUT / "material/explicit_interaction_sets.jsonl"),
                "behavior": {"global_surface_balanced_accuracy_min": 0.98, **behavior_common},
            },
            "withdrawal": {
                "roles": list(WITHDRAWAL_ROLES),
                "surfaces": list(WITHDRAWAL_SURFACES),
                "active_count": len(withdrawal),
                "interaction_set_count": len(withdrawal_sets),
                "active_sha256": core.sha(OUT / "material/withdrawal_active_cases.jsonl"),
                "compiled_sha256": core.sha(OUT / "compiled/qwen3_withdrawal.jsonl"),
                "sets_sha256": core.sha(OUT / "material/withdrawal_interaction_sets.jsonl"),
                "behavior": withdrawal_behavior,
            },
        },
        "capture": {
            "eligible_rule": "only unique cases referenced by behavior-perfect 32-case interaction sets",
            "discovery_partition": "response_discovery",
            "validation_partitions": ["confirmation", "lockbox"],
            "state_count": 37,
            "dtype": "float16",
            "raw_format": "NPY memmap plus JSONL index",
            "no_pooling": True,
            "no_coordinate_selection": True,
        },
        "explicit_observation": {
            "candidate_roles": ["query_label", "query_target", "query_relation", "query_object", "boundary"],
            "selection": "one highest discovery score state per pair and candidate role; full vector only",
            "score": "max(cross_surface_cosine,0) * max(direction_consistency,0) * min(max(equality_to_offdiagonal_null_norm_ratio,0),1)",
            "validation_thresholds": {
                "cosine_to_discovery_each_surface_min": 0.70,
                "holdout_cross_surface_cosine_min": 0.70,
                "direction_to_discovery_min": 0.40,
                "equality_to_offdiagonal_null_norm_ratio_min": 1.20,
                "both_confirmation_and_lockbox_required": True,
            },
            "withdrawal_open_gate": "at least 12 of 15 frozen boundary candidates pass both holdouts",
        },
        "withdrawal_observation": {
            "role_map": {"query_label": "query_relation", "query_target": "query_target", "query_relation": "query_relation", "query_object": "query_object", "boundary": "boundary"},
            "candidate_cells": "mapped layer-role cells frozen by the explicit branch; no new natural-branch hotspot selection",
            "validation_thresholds": {
                "natural_discovery_cross_surface_cosine_min": 0.50,
                "natural_holdout_cross_surface_cosine_min": 0.50,
                "natural_holdout_to_natural_discovery_cosine_min": 0.50,
                "natural_direction_consistency_min": 0.20,
                "equality_to_offdiagonal_null_norm_ratio_min": 1.00,
                "explicit_vector_transfer_is_descriptive_not_required": True,
            },
        },
        "stop_rules": [
            "explicit behavior failure closes hidden access but not the already audited contract",
            "explicit candidate failures close candidates individually",
            "withdrawal behavior is opened only by the frozen explicit boundary breadth gate",
            "withdrawal behavior failure closes withdrawal hidden access",
            "no causal intervention is authorized in C080",
        ],
        "claim_boundary": {
            "allowed": "Qwen3 full-vector equality-interaction trajectory regularities in behavior-correct controlled tasks",
            "forbidden": ["natural language mechanism closure", "semantic neurons", "causal necessity or sufficiency", "cross-model law", "attention/MLP mechanism", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1470_c080_explicit_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "all_gates_passed": True,
        "contract_sha256": protocol["contract_sha256"],
        "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
