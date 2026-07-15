#!/usr/bin/env python3
"""Freeze the Phase435 controlled-natural relation denominator.

The prompts are controlled natural language, not a natural-corpus sample.  Output
interfaces are selected once on an independent calibration split.  Record order,
value mapping, and queried entity are fully crossed in every formal split.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase435_natural_relation"
PHASE_ID = "Phase435-NaturalRelationProtocol"
SCHEMA_VERSION = "phase435_natural_relation.v1"
TRACE_SCHEMA_VERSION = "phase435_natural_relation_trace.v1"

MODELS = ("qwen3", "glm4", "deepseek7b")
DTYPES = {"qwen3": "float16", "glm4": "bfloat16", "deepseek7b": "bfloat16"}
CONTRACTS = ("field_extract", "natural_qa", "relation_rewrite")
GENERIC_CONTROL = "generic_pairing_control"
INTERFACES = ("direct_value", "answer_field", "result_field", "natural_sentence")
INTERFACE_SIMPLICITY = INTERFACES
RELATION_FAMILIES = (
    "person_city",
    "object_color",
    "animal_habitat",
    "country_capital",
)
RECORD_ORDERS = ("ab", "ba")
MAPPINGS = ("direct", "swapped")
QUERY_ROLES = ("a", "b")

INTERFACE_SPLIT = "interface_calibration"
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_holdout")
PHYSICAL_SPLIT = "physical_calibration"
SEALED_SPLIT = "sealed_physical_holdout"
GROUPS_BY_SPLIT = {
    INTERFACE_SPLIT: 96 * len(CONTRACTS),
    "behavior_discovery": 96,
    "behavior_holdout": 192,
    PHYSICAL_SPLIT: 192,
    SEALED_SPLIT: 96,
}
SPLIT_OFFSETS = {
    INTERFACE_SPLIT: 0,
    "behavior_discovery": 700,
    "behavior_holdout": 1000,
    PHYSICAL_SPLIT: 1500,
    SEALED_SPLIT: 2000,
}

WORDS_A = (
    "Alder", "Amber", "Arden", "Ashen", "Azure", "Birch", "Brisk", "Bronze", "Cedar", "Cinder",
    "Clear", "Coral", "Crimson", "Dawn", "Delta", "Ember", "Fern", "Flint", "Frost", "Golden",
)
WORDS_B = (
    "Anchor", "Basin", "Beacon", "Brook", "Canyon", "Crest", "Field", "Garden", "Glen", "Grove",
    "Harbor", "Haven", "Hill", "Island", "Lake", "Meadow", "Mesa", "Orchard", "Ridge", "Valley",
)
WORDS_C = (
    "Arbor", "Atlas", "Briar", "Cobalt", "Dorian", "Elara", "Fable", "Galen", "Helio", "Iris",
    "Juno", "Kestrel", "Lumen", "Marlow", "Neris", "Orion", "Pallas", "Quill", "Rhea", "Sylvan",
)

FAMILY_CONFIG = {
    "person_city": {
        "entity_kind": "person",
        "value_kind": "city",
        "entity_label": "Person",
        "value_label": "Assigned city",
        "relation_label": "assigned city",
        "statement_relation_surface": "assigned to the city",
        "question_relation_surface": "assigned to",
        "statement": "{entity} is assigned to the city {value}.",
        "question": "Which city is assigned to {entity}?",
    },
    "object_color": {
        "entity_kind": "object",
        "value_kind": "color",
        "entity_label": "Object",
        "value_label": "Registered color",
        "relation_label": "registered color",
        "statement_relation_surface": "registered color",
        "question_relation_surface": "registered color",
        "statement": "The registered color of {entity} is {value}.",
        "question": "What is the registered color of {entity}?",
    },
    "animal_habitat": {
        "entity_kind": "animal",
        "value_kind": "habitat",
        "entity_label": "Animal",
        "value_label": "Observed habitat",
        "relation_label": "observed habitat",
        "statement_relation_surface": "observed habitat",
        "question_relation_surface": "habitat",
        "statement": "The observed habitat of {entity} is {value}.",
        "question": "What habitat is recorded for {entity}?",
    },
    "country_capital": {
        "entity_kind": "country",
        "value_kind": "capital",
        "entity_label": "Fictional country",
        "value_label": "Administrative center",
        "relation_label": "administrative center",
        "statement_relation_surface": "administrative center",
        "question_relation_surface": "administrative center",
        "statement": "In this fictional registry, the administrative center of {entity} is {value}.",
        "question": "What administrative center is recorded for {entity}?",
    },
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


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def lexeme(kind: str, index: int) -> str:
    """Create deterministic, natural-looking, split-disjoint lexical items."""

    if not 0 <= index < 8000:
        raise ValueError(index)
    first = WORDS_A[index % 20]
    second = WORDS_B[(index // 20) % 20]
    third = WORDS_C[(index // 400) % 20]
    suffix = {
        "person": "",
        "city": "City",
        "object": "Artifact",
        "color": "Shade",
        "animal": "Creature",
        "habitat": "Habitat",
        "country": "Republic",
        "capital": "Center",
    }[kind]
    words = (
        (first, third, second)
        if kind == "person"
        else (first, second, third, suffix)
    )
    return " ".join(word for word in words if word)


def factor_assignment(index: int) -> dict[str, str]:
    return {
        "baseline_record_order": RECORD_ORDERS[index % 2],
        "baseline_mapping": MAPPINGS[(index // 2) % 2],
        "baseline_query_role": QUERY_ROLES[(index // 4) % 2],
    }


def build_group(
    split: str,
    index: int,
    *,
    forced_contract: str | None = None,
    lexical_index: int | None = None,
) -> dict[str, Any]:
    family = RELATION_FAMILIES[index % len(RELATION_FAMILIES)]
    config = FAMILY_CONFIG[family]
    local_lexical = index if lexical_index is None else lexical_index
    base = SPLIT_OFFSETS[split] + 2 * local_lexical
    contract_variants = [forced_contract] if forced_contract else list(CONTRACTS)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "controlled_natural_language": True,
        "natural_corpus_sample": False,
        "group_index": index,
        "semantic_group_id": f"phase435__{split}__group_{index:04d}",
        "paired_group_id": f"phase435__{split}__pair_{index:04d}",
        "family_id": "language_action",
        "mechanism_id": "natural_relation_lookup",
        "relation_family": family,
        "entity_label": config["entity_label"],
        "value_label": config["value_label"],
        "relation_label": config["relation_label"],
        "entity_a": lexeme(config["entity_kind"], base),
        "entity_b": lexeme(config["entity_kind"], base + 1),
        "value_1": lexeme(config["value_kind"], base),
        "value_2": lexeme(config["value_kind"], base + 1),
        "contract_variants": contract_variants,
        "physical_fold": (
            "discovery" if split == PHYSICAL_SPLIT and index < 96
            else "holdout" if split == PHYSICAL_SPLIT
            else "sealed" if split == SEALED_SPLIT
            else None
        ),
        **factor_assignment(index),
    }


def build_groups() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    calibration: list[dict[str, Any]] = []
    serial = 0
    for contract in CONTRACTS:
        for contract_index in range(96):
            calibration.append(
                build_group(
                    INTERFACE_SPLIT,
                    serial,
                    forced_contract=contract,
                    lexical_index=serial,
                )
            )
            serial += 1
    rows[INTERFACE_SPLIT] = calibration
    for split in (*BEHAVIOR_SPLITS, PHYSICAL_SPLIT, SEALED_SPLIT):
        rows[split] = [
            build_group(split, index) for index in range(GROUPS_BY_SPLIT[split])
        ]
    return rows


def behavior_conditions_per_group(split: str) -> int:
    if split == INTERFACE_SPLIT:
        return len(INTERFACES)
    return len(CONTRACTS) * len(RECORD_ORDERS) * len(MAPPINGS) * len(QUERY_ROLES)


def maximum_physical_conditions_per_group() -> int:
    return (len(CONTRACTS) + 1) * len(RECORD_ORDERS) * len(MAPPINGS) * len(QUERY_ROLES)


def denominator_audit(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    group_counts = {split: len(values) for split, values in rows.items()}
    condition_counts = {
        split: len(values) * behavior_conditions_per_group(split)
        for split, values in rows.items()
        if split not in {PHYSICAL_SPLIT, SEALED_SPLIT}
    }
    vocab = {
        split: {
            token.lower()
            for row in values
            for token in (row["entity_a"], row["entity_b"], row["value_1"], row["value_2"])
        }
        for split, values in rows.items()
    }
    disjoint = all(
        not vocab[left].intersection(vocab[right])
        for left_index, left in enumerate(vocab)
        for right in list(vocab)[left_index + 1 :]
    )
    calibration_contracts = Counter(
        row["contract_variants"][0] for row in rows[INTERFACE_SPLIT]
    )
    formal_balance = {}
    for split in (*BEHAVIOR_SPLITS, PHYSICAL_SPLIT, SEALED_SPLIT):
        values = rows[split]
        formal_balance[split] = {
            "relation_family": dict(Counter(row["relation_family"] for row in values)),
            "physical_fold": dict(Counter(str(row["physical_fold"]) for row in values)),
        }
    valid = bool(
        group_counts == GROUPS_BY_SPLIT
        and condition_counts[INTERFACE_SPLIT] == 1152
        and condition_counts["behavior_discovery"] == 2304
        and condition_counts["behavior_holdout"] == 4608
        and calibration_contracts == Counter({contract: 96 for contract in CONTRACTS})
        and disjoint
        and len({row["semantic_group_id"] for values in rows.values() for row in values})
        == sum(group_counts.values())
        and all(row["pipeline_sealed"] for row in rows[SEALED_SPLIT])
        and all(
            not row["pipeline_sealed"]
            for split, values in rows.items()
            if split != SEALED_SPLIT
            for row in values
        )
    )
    return {
        "valid": valid,
        "language_scope": "controlled natural-language contextual relations; not a natural-corpus sample",
        "groups_by_split": group_counts,
        "calibration_groups_per_contract": dict(calibration_contracts),
        "conditions_by_split_per_model": condition_counts,
        "open_behavior_conditions_per_model": sum(condition_counts.values()),
        "three_model_open_behavior_conditions": sum(condition_counts.values()) * len(MODELS),
        "maximum_open_physical_conditions_per_model": GROUPS_BY_SPLIT[PHYSICAL_SPLIT]
        * maximum_physical_conditions_per_group(),
        "maximum_sealed_conditions_per_model": GROUPS_BY_SPLIT[SEALED_SPLIT]
        * maximum_physical_conditions_per_group(),
        "vocabulary_disjoint_across_splits": disjoint,
        "formal_balance": formal_balance,
    }


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase435_natural_relation_protocol.py",
        "phase435_natural_relation_collect.py",
        "phase435_natural_relation_analysis.py",
        "test_phase435_natural_relation.py",
    )
    return {
        name: (
            hashlib.sha256((ROOT / "tests/gpt5" / name).read_bytes()).hexdigest()
            if (ROOT / "tests/gpt5" / name).exists()
            else None
        )
        for name in names
    }


def freeze() -> dict[str, Any]:
    rows = build_groups()
    audit = denominator_audit(rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    for split, values in rows.items():
        path = (
            OUT / "sealed/phase435_groups_sealed.jsonl"
            if split == SEALED_SPLIT
            else OUT / f"phase435_groups_{split}.jsonl"
        )
        write_jsonl(path, values)
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": SEALED_SPLIT,
        "sealed_group_count": len(rows[SEALED_SPLIT]),
        "maximum_sealed_condition_count_per_model": audit["maximum_sealed_conditions_per_model"],
        "sealed_group_rows_sha256": digest_rows(rows[SEALED_SPLIT]),
        "read_requires_open_gate": True,
        "open_analysis_must_not_import_sealed_rows": True,
    }
    write_json(OUT / "phase435_sealed_commitment.json", commitment)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models_in_execution_order": list(MODELS),
        "execution_dtypes": DTYPES,
        "contracts": list(CONTRACTS),
        "generic_physical_control": GENERIC_CONTROL,
        "interfaces": list(INTERFACES),
        "interface_simplicity_order": list(INTERFACE_SIMPLICITY),
        "relation_families": list(RELATION_FAMILIES),
        "record_orders": list(RECORD_ORDERS),
        "mappings": list(MAPPINGS),
        "query_roles": list(QUERY_ROLES),
        "behavior_splits": list(BEHAVIOR_SPLITS),
        "physical_split": PHYSICAL_SPLIT,
        "sealed_split": SEALED_SPLIT,
        "denominator_audit": audit,
        "rows_sha256": {
            split: digest_rows(values)
            for split, values in rows.items()
            if split != SEALED_SPLIT
        },
        "sealed_commitment": commitment,
        "interface_calibration_gate": {
            "per_contract_wilson_lcb": 0.75,
            "per_position_wilson_lcb": 0.75,
            "maximum_position_gap": 0.10,
            "other_wilson_ucb": 0.10,
            "selection": "first qualified interface in frozen simplicity order; otherwise deterministic best interface remains behavior-ineligible",
        },
        "natural_behavior_gate": {
            "discovery_and_holdout_per_position_wilson_lcb": 0.80,
            "maximum_position_gap": 0.05,
            "holdout_other_wilson_ucb": 0.05,
            "teacher_event_per_position_wilson_lcb": 0.80,
            "interface_parse_per_position_wilson_lcb": 0.80,
            "stop_is_separate_from_content_gate": True,
        },
        "physical_contract": {
            "runs_only_for_behavior_eligible_model_contracts": True,
            "open_window_freeze_groups": 96,
            "open_holdout_groups": 96,
            "generic_pairing_control_included": True,
            "full_layer_component_ledger": True,
            "source_to_query_attention_write": True,
            "single_head_channel_neuron_scan": False,
        },
        "numeric_gates": {
            "component_reconstruction_relative_error_max": 0.001,
            "attention_replay_relative_error_max": 0.001,
            "label_blind_geometry_effect_min": 0.05,
            "geometry_positive_relation_families_min": 3,
            "geometry_candidate_over_generic_min": 0.02,
            "transport_balanced_accuracy_min": 0.70,
            "transport_per_source_wilson_lcb_min": 0.65,
            "transport_maximum_position_gap": 0.10,
            "transport_candidate_over_generic_accuracy_min": 0.10,
        },
        "gate_order": [
            "G0_interface_calibration",
            "G1_natural_content_and_order_balance",
            "G2_component_identity_and_position_registration",
            "G3_label_blind_order_geometry",
            "G4_semantic_source_transport",
            "G5_frozen_holdout_prediction",
            "G6_generic_pairing_specificity",
            "G7_sealed_physical_replication",
        ],
        "causal_and_single_neuron_forbidden_in_phase435": True,
        "source_transport_is_not_inferred_from_state_geometry": True,
        "implementation_hashes": implementation_hashes(),
    }
    write_json(OUT / "phase435_protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
