#!/usr/bin/env python3
"""Freeze Phase437 position-factor decomposition before model execution."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_protocol as p435  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase437_position_factor"
P435_OUT = ROOT / "tests/gpt5/result/phase435_natural_relation"
P436_OUT = ROOT / "tests/gpt5/result/phase436_observer_decomposition"
PHASE_ID = "Phase437-PositionFactorProtocol-v1"
SCHEMA_VERSION = "phase437_position_factor.v2"

MODELS = p435.MODELS
DTYPES = p435.DTYPES
CONTRACTS = p435.CONTRACTS
RELATION_FAMILIES = p435.RELATION_FAMILIES

OBSERVER_SPLIT = "observer_calibration"
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_holdout")
PHYSICAL_SPLIT = "physical_calibration"
SEALED_SPLIT = "sealed_physical_holdout"

GROUPS_PER_CONTRACT = {
    OBSERVER_SPLIT: 384,
    "behavior_discovery": 192,
    "behavior_holdout": 384,
    PHYSICAL_SPLIT: 384,
    SEALED_SPLIT: 384,
}

FROZEN_INTERFACES = {
    "qwen3": {
        "field_extract": "direct_value",
        "natural_qa": "answer_field",
        "relation_rewrite": "natural_sentence",
    },
    "glm4": {
        "field_extract": "natural_sentence",
        "natural_qa": "natural_sentence",
        "relation_rewrite": "answer_field",
    },
    "deepseek7b": {
        "field_extract": "answer_field",
        "natural_qa": "answer_field",
        "relation_rewrite": "result_field",
    },
}

BOUNDARIES = ("period", "semicolon", "newline", "field_delimiter")
CONNECTORS = ("parallel", "separate", "none")
RECORD_LENGTHS = ("short", "long")
LABEL_ORDERS = ("entity_first", "relation_first")
POST_GAPS = ("near", "far")

PRIMARY_VARIANTS = (
    "first_natural_near",
    "second_natural_near",
    "first_natural_far",
    "second_natural_far",
)
MATCHED_VARIANTS = (
    "second_matched_near",
    "second_matched_far",
)
BEHAVIOR_VARIANTS = PRIMARY_VARIANTS + MATCHED_VARIANTS
PHYSICAL_VARIANTS = (
    "first_natural_near",
    "second_natural_near",
    "second_matched_near",
)

CONTROL_TYPES = (
    "no_relation",
    "wrong_relation",
    "wrong_query_entity",
    "wrong_query_relation",
    "wrong_value_mapping",
    "order_swap_control",
    "distance_swap_control",
    "boundary_swap_control",
)

SPLIT_LEXEME_OFFSETS = {
    OBSERVER_SPLIT: 0,
    "behavior_discovery": 2304,
    "behavior_holdout": 3456,
    PHYSICAL_SPLIT: 5760,
    SEALED_SPLIT: 8064,
}
PSEUDO_SYLLABLES = (
    "ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "du", "fa", "fe",
    "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu", "ka", "ke", "ki", "ko",
)
KIND_SUFFIX = {
    "person": "",
    "city": " City",
    "object": " Artifact",
    "color": " Shade",
    "animal": " Creature",
    "habitat": " Habitat",
    "country": " Country",
    "capital": " Capital",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def outer_factors(index: int) -> dict[str, str]:
    return {
        "boundary": BOUNDARIES[index % 4],
        "connector": CONNECTORS[(index // 4) % 3],
        "record_length": RECORD_LENGTHS[(index // 12) % 2],
        "label_order": LABEL_ORDERS[(index // 24) % 2],
    }


def query_mapping(combo: int) -> tuple[str, str]:
    return (
        "a" if combo % 2 == 0 else "b",
        "direct" if combo // 2 == 0 else "swapped",
    )


def phase437_lexeme(kind: str, index: int, split: str) -> str:
    value = SPLIT_LEXEME_OFFSETS[split] + index
    if not 0 <= value < len(PSEUDO_SYLLABLES) ** 3:
        raise ValueError(value)
    first = PSEUDO_SYLLABLES[value % 24]
    second = PSEUDO_SYLLABLES[(value // 24) % 24]
    third = PSEUDO_SYLLABLES[(value // (24 * 24)) % 24]
    return f"{(first + second + third).capitalize()}{KIND_SUFFIX[kind]}"


def group_assignment(split: str, index: int) -> dict[str, Any]:
    if split == OBSERVER_SPLIT:
        variant_index = index % len(PRIMARY_VARIANTS)
        outer_index = (index // len(PRIMARY_VARIANTS)) % 48
        replicate = index // (len(PRIMARY_VARIANTS) * 48)
        observer_variant = PRIMARY_VARIANTS[variant_index]
        factors = outer_factors(outer_index)
        label_index = LABEL_ORDERS.index(factors["label_order"])
        combo = (label_index + variant_index + 2 * replicate) % 4
    else:
        outer_index = index % 48
        replicate = index // 48
        if split in {PHYSICAL_SPLIT, SEALED_SPLIT}:
            fold_index = replicate // 4
            pair_index = (replicate % 4) // 2
            pair_side = replicate % 2
            combo = (outer_index + pair_index + fold_index) % 4
        else:
            fold_index = None
            pair_index = None
            pair_side = None
            combo = replicate % 4
        observer_variant = None
    query_role, mapping = query_mapping(combo)
    return {
        "outer_index": outer_index,
        "replicate": replicate,
        "query_role": query_role,
        "mapping": mapping,
        "observer_variant": observer_variant,
        "geometry_pair_index": pair_index if split != OBSERVER_SPLIT else None,
        "geometry_pair_side": pair_side if split != OBSERVER_SPLIT else None,
        **outer_factors(outer_index),
    }


def build_group(split: str, contract: str, index: int, contract_index: int) -> dict[str, Any]:
    assignment = group_assignment(split, index)
    if split == OBSERVER_SPLIT:
        variant_index = PRIMARY_VARIANTS.index(str(assignment["observer_variant"]))
        length_index = RECORD_LENGTHS.index(str(assignment["record_length"]))
        label_index = LABEL_ORDERS.index(str(assignment["label_order"]))
        family_index = (
            length_index + label_index + 2 * variant_index
            + 2 * assignment["replicate"] + contract_index
        ) % len(RELATION_FAMILIES)
    elif split in {PHYSICAL_SPLIT, SEALED_SPLIT}:
        family_index = (
            assignment["outer_index"] // 4
            + int(assignment["geometry_pair_index"])
            + int(assignment["geometry_pair_side"])
            + contract_index
        ) % len(RELATION_FAMILIES)
    else:
        family_index = (
            assignment["outer_index"] // 4
            + assignment["replicate"]
            + contract_index
        ) % len(RELATION_FAMILIES)
    family = RELATION_FAMILIES[family_index]
    config = p435.FAMILY_CONFIG[family]
    serial = contract_index * GROUPS_PER_CONTRACT[split] + index
    base = serial * 2
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "controlled_natural_language": True,
        "natural_corpus_sample": False,
        "contract": contract,
        "group_index": index,
        "semantic_group_id": f"phase437__{split}__{contract}__group_{index:04d}",
        "paired_group_id": f"phase437__{split}__{contract}__pair_{index:04d}",
        "geometry_pair_id": (
            f"phase437__{split}__{contract}__geometry_"
            f"{assignment['outer_index']:02d}_{assignment['replicate'] // 4}_"
            f"{assignment['geometry_pair_index']}"
            if split in {PHYSICAL_SPLIT, SEALED_SPLIT}
            else None
        ),
        "family_id": "language_action",
        "mechanism_id": "position_factorized_relation_lookup",
        "relation_family": family,
        "entity_label": config["entity_label"],
        "value_label": config["value_label"],
        "relation_label": config["relation_label"],
        "entity_a": phase437_lexeme(config["entity_kind"], base, split),
        "entity_b": phase437_lexeme(config["entity_kind"], base + 1, split),
        "value_1": phase437_lexeme(config["value_kind"], base, split),
        "value_2": phase437_lexeme(config["value_kind"], base + 1, split),
        "physical_fold": (
            "discovery" if split == PHYSICAL_SPLIT and assignment["replicate"] < 4
            else "holdout" if split == PHYSICAL_SPLIT
            else "sealed" if split == SEALED_SPLIT
            else None
        ),
        "control_type": CONTROL_TYPES[index % len(CONTROL_TYPES)],
        **assignment,
    }


def build_groups() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, per_contract in GROUPS_PER_CONTRACT.items():
        output[split] = [
            build_group(split, contract, index, contract_index)
            for contract_index, contract in enumerate(CONTRACTS)
            for index in range(per_contract)
        ]
    return output


def vocabulary(rows: list[dict[str, Any]]) -> set[str]:
    return {
        value.lower()
        for row in rows
        for value in (row["entity_a"], row["entity_b"], row["value_1"], row["value_2"])
    }


def previous_vocabulary() -> set[str]:
    paths = list(P435_OUT.glob("phase435_groups_*.jsonl")) + [
        P435_OUT / "sealed/phase435_groups_sealed.jsonl",
        P436_OUT / "phase436_groups_interface_calibration.jsonl",
    ]
    return {
        value.lower()
        for path in paths
        for row in read_jsonl(path)
        for value in (row["entity_a"], row["entity_b"], row["value_1"], row["value_2"])
    }


def factor_balance(rows: list[dict[str, Any]], split: str, contract: str) -> dict[str, Any]:
    selected = [row for row in rows if row["contract"] == contract]
    payload = {
        factor: dict(Counter(row[factor] for row in selected))
        for factor in ("relation_family", "boundary", "connector", "record_length", "label_order", "query_role", "mapping")
    }
    if split == OBSERVER_SPLIT:
        payload["observer_variant"] = dict(Counter(row["observer_variant"] for row in selected))
    return payload


def pairwise_balance(rows: list[dict[str, Any]], factors: tuple[str, ...]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    valid = True
    for index, left in enumerate(factors):
        for right in factors[index + 1 :]:
            counts = Counter((str(row[left]), str(row[right])) for row in rows)
            expected_cells = len({str(row[left]) for row in rows}) * len(
                {str(row[right]) for row in rows}
            )
            passed = len(counts) == expected_cells and len(set(counts.values())) == 1
            pairs[f"{left}__x__{right}"] = {
                "cell_count": len(counts),
                "expected_cell_count": expected_cells,
                "minimum": min(counts.values()),
                "maximum": max(counts.values()),
                "balanced": passed,
            }
            valid = valid and passed
    return {"valid": valid, "pairs": pairs}


def denominator_audit(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected_counts = {
        split: GROUPS_PER_CONTRACT[split] * len(CONTRACTS)
        for split in GROUPS_PER_CONTRACT
    }
    counts = {split: len(rows) for split, rows in groups.items()}
    vocab = {split: vocabulary(rows) for split, rows in groups.items()}
    split_disjoint = all(
        not vocab[left].intersection(vocab[right])
        for index, left in enumerate(vocab)
        for right in list(vocab)[index + 1 :]
    )
    prior_disjoint = not set().union(*vocab.values()).intersection(previous_vocabulary())
    balances = {
        split: {
            contract: factor_balance(rows, split, contract)
            for contract in CONTRACTS
        }
        for split, rows in groups.items()
    }
    pairwise_factors = (
        "boundary", "connector", "record_length", "label_order",
        "observer_variant", "relation_family", "query_role", "mapping",
    )
    observer_pairwise = {
        contract: pairwise_balance(
            [row for row in groups[OBSERVER_SPLIT] if row["contract"] == contract],
            pairwise_factors,
        )
        for contract in CONTRACTS
    }
    group_ids = [row["semantic_group_id"] for rows in groups.values() for row in rows]
    geometry_pair_audit: dict[str, Any] = {}
    geometry_pairs_valid = True
    for split in (PHYSICAL_SPLIT, SEALED_SPLIT):
        pair_counts: Counter[str] = Counter(
            str(row["geometry_pair_id"]) for row in groups[split]
        )
        indexed = {
            pair_id: [row for row in groups[split] if row["geometry_pair_id"] == pair_id]
            for pair_id in pair_counts
        }
        valid_pairs = all(
            len(rows) == 2
            and len({row["relation_family"] for row in rows}) == 2
            and len(
                {
                    (
                        row["contract"], row["outer_index"], row["query_role"],
                        row["mapping"], row["boundary"], row["connector"],
                        row["record_length"], row["label_order"], row["physical_fold"],
                        row["control_type"],
                    )
                    for row in rows
                }
            ) == 1
            for rows in indexed.values()
        )
        geometry_pairs_valid = geometry_pairs_valid and valid_pairs
        geometry_pair_audit[split] = {
            "pair_count": len(indexed),
            "all_pairs_have_two_groups": all(value == 2 for value in pair_counts.values()),
            "different_relation_matched_factor_pairs": valid_pairs,
        }
    balanced = True
    for split in ("behavior_discovery", "behavior_holdout", PHYSICAL_SPLIT, SEALED_SPLIT):
        per_contract = GROUPS_PER_CONTRACT[split]
        for contract in CONTRACTS:
            values = balances[split][contract]
            balanced = balanced and all(
                len(set(values[factor].values())) == 1
                for factor in ("boundary", "connector", "record_length", "label_order", "query_role", "mapping")
            )
            if sum(values["query_role"].values()) != per_contract or sum(values["mapping"].values()) != per_contract:
                raise RuntimeError("Phase437 query/mapping balance count mismatch")
    valid = bool(
        counts == expected_counts
        and split_disjoint
        and prior_disjoint
        and len(group_ids) == len(set(group_ids))
        and all(row["pipeline_sealed"] for row in groups[SEALED_SPLIT])
        and all(
            not row["pipeline_sealed"]
            for split, rows in groups.items()
            if split != SEALED_SPLIT
            for row in rows
        )
        and balanced
        and geometry_pairs_valid
        and all(value["valid"] for value in observer_pairwise.values())
    )
    return {
        "valid": valid,
        "groups_by_split": counts,
        "groups_per_contract": GROUPS_PER_CONTRACT,
        "observer_conditions_per_model": expected_counts[OBSERVER_SPLIT],
        "maximum_behavior_conditions_per_model": {
            split: expected_counts[split] * len(BEHAVIOR_VARIANTS)
            for split in BEHAVIOR_SPLITS
        },
        "maximum_three_model_behavior_conditions": sum(
            expected_counts[split] * len(BEHAVIOR_VARIANTS) * len(MODELS)
            for split in BEHAVIOR_SPLITS
        ),
        "split_vocabulary_disjoint": split_disjoint,
        "vocabulary_disjoint_from_phase435_436": prior_disjoint,
        "factor_balance": balances,
        "observer_pairwise_balance": observer_pairwise,
        "geometry_pair_audit": geometry_pair_audit,
    }


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase437_position_factor_protocol.py",
        "phase437_position_factor_collect.py",
        "phase437_position_factor_analysis.py",
        "test_phase437_position_factor.py",
    )
    return {
        name: sha256_file(ROOT / "tests/gpt5" / name)
        if (ROOT / "tests/gpt5" / name).exists()
        else None
        for name in names
    }


def freeze() -> dict[str, Any]:
    groups = build_groups()
    audit = denominator_audit(groups)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    for split, rows in groups.items():
        path = (
            OUT / "sealed/phase437_groups_sealed.jsonl"
            if split == SEALED_SPLIT
            else OUT / f"phase437_groups_{split}.jsonl"
        )
        write_jsonl(path, rows)

    phase436_protocol = read_json(P436_OUT / "phase436_protocol.json")
    phase436_observer = read_json(P436_OUT / "phase436_observer_freeze.json")
    observed_interfaces = {
        model: {
            contract: phase436_observer["models"][model]["contracts"][contract]["selected_interface"]
            for contract in CONTRACTS
        }
        for model in MODELS
    }
    if observed_interfaces != FROZEN_INTERFACES:
        raise RuntimeError("Phase437 frozen interfaces disagree with Phase436 evidence")

    hashes = {
        split: digest_rows(rows)
        for split, rows in groups.items()
    }
    sealed_path = OUT / "sealed/phase437_groups_sealed.jsonl"
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_group_count": len(groups[SEALED_SPLIT]),
        "sealed_rows_sha256": sha256_file(sealed_path),
        "read_requires_all_open_physical_gates": True,
        "causal_and_single_neuron_forbidden": True,
    }
    write_json(OUT / "phase437_sealed_commitment.json", commitment)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models_in_execution_order": list(MODELS),
        "execution_dtypes": DTYPES,
        "contracts": list(CONTRACTS),
        "frozen_interfaces_from_phase436": FROZEN_INTERFACES,
        "phase436_threshold_audit": {
            "interface_other_ucb": phase436_protocol["interface_semantic_gate"]["other_wilson_ucb"],
            "behavior_other_ucb": phase436_protocol["behavior_gate"]["holdout_other_wilson_ucb"],
            "historical_summary_consistent_with_code": True,
            "phase437_unified_other_ucb": 0.05,
            "phase436_results_not_rewritten": True,
        },
        "phase437_v0_protocol_correction": {
            "status": "quarantined_before_formal_inference",
            "row_count": 3456,
            "defects": [
                "observer label_order and relation_family were not pairwise balanced",
                "observer boundary and mapping were not pairwise balanced",
                "split identity was appended as a multiword target suffix",
            ],
            "formal_thresholds_changed": False,
            "frozen_interfaces_changed": False,
            "quarantine_directory": "tests/gpt5/result/phase437_position_factor_v0_quarantine_unbalanced_assignment",
        },
        "distance_design": {
            "structural_limit": "an earlier record cannot be closer than a later record to a future query without moving or duplicating content",
            "primary_natural_conditions": list(PRIMARY_VARIANTS),
            "conditional_second_record_match": list(MATCHED_VARIANTS),
            "actual_model_token_distances_are_mandatory": True,
            "matched_distance_overlap_required": True,
        },
        "factor_design": {
            "outer_full_factorial": {
                "boundary": list(BOUNDARIES),
                "connector": list(CONNECTORS),
                "record_length": list(RECORD_LENGTHS),
                "label_order": list(LABEL_ORDERS),
            },
            "within_group_variants": list(BEHAVIOR_VARIANTS),
            "query_and_mapping_fixed_within_semantic_group": True,
            "direct_matched_contrasts_only": True,
            "regression_is_not_used_as_a_mechanism_model": True,
        },
        "denominator_audit": audit,
        "group_hashes": hashes,
        "observer_gate": {
            "per_position_wilson_lcb": 0.80,
            "maximum_natural_position_gap": 0.05,
            "other_wilson_ucb": 0.05,
            "teacher_per_position_wilson_lcb": 0.80,
            "format_and_stop_reported_separately": True,
        },
        "behavior_gate": {
            "discovery_and_holdout_per_position_wilson_lcb": 0.80,
            "maximum_natural_position_gap": 0.05,
            "maximum_matched_position_gap": 0.05,
            "maximum_post_gap_effect": 0.05,
            "maximum_outer_factor_range": 0.05,
            "maximum_matched_token_distance_median_error": 4.0,
            "maximum_matched_token_distance_p95_error": 8.0,
            "other_wilson_ucb": 0.05,
            "teacher_per_position_wilson_lcb": 0.80,
            "both_splits_required": True,
        },
        "physical_stage": {
            "variants": list(PHYSICAL_VARIANTS),
            "paired_control_types": list(CONTROL_TYPES),
            "label_blind_geometry": True,
            "semantic_transport_must_exceed_physical_position_transport": True,
            "prediction_must_exceed_position_distance_boundary_contract_baselines": True,
            "causal_and_single_neuron_forbidden": True,
        },
        "physical_numeric_gates": {
            "component_reconstruction_relative_error_max": 0.001,
            "attention_replay_relative_error_max": 0.001,
            "label_blind_geometry_effect_min": 0.05,
            "geometry_positive_relation_families_min": 3,
            "geometry_candidate_over_control_min": 0.02,
            "transport_balanced_accuracy_min": 0.70,
            "transport_per_source_wilson_lcb_min": 0.65,
            "transport_maximum_position_gap": 0.10,
            "transport_over_best_surface_baseline_min": 0.05,
            "transport_candidate_over_control_accuracy_min": 0.10,
        },
        "gate_order": [
            "G0_observer_schema_and_semantic_gate",
            "G1_factorized_behavior_discovery_and_holdout",
            "G2_actual_token_distance_registration",
            "G3_label_blind_relation_geometry",
            "G4_semantic_source_transport",
            "G5_frozen_holdout_prediction",
            "G6_control_specificity",
            "G7_sealed_physical_replication",
        ],
        "sealed_commitment": commitment,
        "implementation_hashes": implementation_hashes(),
    }
    write_json(OUT / "phase437_protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
