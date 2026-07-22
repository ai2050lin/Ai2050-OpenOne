#!/usr/bin/env python3
"""Pure-CPU contracts and seeded extension builder for Phase991.

This module does not import torch or transformers.  It extends the sealed
Phase990 delayed-two-hop task without treating the task graph as a discovered
model-internal graph.  The extension is generated from an actual integer seed
and is rejected unless its abstract states, visible semantics, and normalized
prompts are disjoint from all 320 Phase990 worlds.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import phase990_binding_core as p990_core
import phase990_binding_dataset as p990_data


PHASE = 991
SCHEMA_VERSION = 1
EXPERIMENT = "delayed_two_hop_gpu_admission_package"
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
RESULT_ROOT = GLM5 / "result"
OUT = RESULT_ROOT / "phase991_delayed_binding_gpu_admission"

PHASE990_OUT = RESULT_ROOT / "phase990_delayed_binding_protocol"
PHASE990_DATASET = PHASE990_OUT / "dataset.json"
PHASE990_AUDIT = PHASE990_OUT / "dataset_audit.json"
PHASE990_PROTOCOL = PHASE990_OUT / "protocol_preregistration.json"
PHASE990_ERRATUM = PHASE990_OUT / "protocol_audit_erratum_v2.json"

EXTENSION_SEED = 0x0000_0000_03C7_0080
EXTENSION_WORLD_COUNT = 128
EXTENSION_SPLIT = "expanded_confirmation"
EXTENSION_VARIANTS = tuple(p990_core.VARIANTS)
EXTENSION_RECORD_COUNT = EXTENSION_WORLD_COUNT * len(EXTENSION_VARIANTS)
MODEL_ORDER = tuple(p990_core.MODEL_ORDER)
VALUES = tuple(p990_core.VALUES)

# The extension deliberately retains the Phase990 vocabulary.  It tests new
# graphs and prompts, not lexical generalization.  That limitation is frozen.
ENTITY_LEXICON = tuple(p990_core.PEOPLE[:8])
OBJECT_LEXICON = tuple(p990_core.OBJECTS[:4])

SOURCE_PATHS = {
    "core": "tests/glm5/phase991_gpu_admission_core.py",
    "protocol": "tests/glm5/phase991_gpu_admission_protocol.py",
    "resolver": "tests/glm5/phase991_reference_resolver.py",
    "independent_audit": "tests/glm5/phase991_gpu_admission_audit.py",
}
PHASE990_SOURCE_PATHS = {
    "core": "tests/glm5/phase990_binding_core.py",
    "dataset": "tests/glm5/phase990_binding_dataset.py",
    "protocol": "tests/glm5/phase990_protocol_freeze.py",
    "audit": "tests/glm5/phase990_binding_audit.py",
    "audit_erratum_v2": "tests/glm5/phase990_binding_audit_erratum_v2.py",
}

PUBLIC_SPLITS = ("discovery", "confirmation", "adversarial")
SEALED_SPLIT = "sealed_holdout"
ALL_RUNTIME_SPLITS = (*PUBLIC_SPLITS, SEALED_SPLIT, EXTENSION_SPLIT)

CANDIDATE_SET = {
    "semantic_values_in_fixed_order": list(VALUES),
    "teacher_forced_continuations": {value: f" {value}" for value in VALUES},
    "natural_generation_max_new_tokens": 24,
    "natural_generation_is_primary": True,
    "teacher_forced_logits_are_diagnostic_only": True,
}

EQUIVALENCE_RULE = {
    "normalization": "Unicode text is casefolded; ASCII word-boundary marker matches only",
    "marker_regex": r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])",
    "primary_prediction": "first complete marker match in generated text before EOS",
    "no_marker": "unparsed",
    "multiple_distinct_markers": "first marker scores semantics; ambiguity is separately reported",
    "strict_reference_sentence": "The retrieved marker is {value}.",
    "strict_format_is_secondary": True,
    "eos_and_budget_are_secondary": True,
}

THRESHOLDS = {
    "overall_each_major_condition_min_percent": 90,
    "each_preregistered_variant_min_percent": 85,
    "counterfactual_pair_sensitivity_min_percent": 85,
    "shortcut_baseline_must_be_reported": True,
    "behavior_above_shortcut_is_not_mechanism_evidence": True,
    "public_splits_must_pass_before_sealed_access": True,
    "all_three_primary_models_must_pass_before_extension_model_access": True,
}

SEARCH_CANDIDATE_SET = {
    "position_groups": [
        "fact_source", "distractor", "query", "fixed_answer_or_current_generation"
    ],
    "relative_layer_blocks": 6,
    "component": "residual_state",
    "discovery_common_worlds_max": 16,
    "frozen_candidate_blocks_max": 4,
    "no_preselected_head_neuron_direction_or_formula": True,
    "internal_search_not_authorized_by_phase991": True,
}

_SPACE_RE = re.compile(r"\s+")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased JSON: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"JSON root is not an object: {path}")
    return result


def sealed_document(
    payload: Mapping[str, Any], hash_field: str, created_at_utc: str
) -> dict[str, Any]:
    body = deepcopy(dict(payload))
    require(hash_field not in body, f"hash field already present: {hash_field}")
    body["created_at_utc"] = created_at_utc
    body[hash_field] = sha256_json(body)
    return body


def verify_self_hash(document: Mapping[str, Any], hash_field: str) -> None:
    reported = document.get(hash_field)
    require(isinstance(reported, str) and len(reported) == 64, f"bad {hash_field}")
    body = {key: value for key, value in document.items() if key != hash_field}
    require(sha256_json(body) == reported, f"{hash_field} mismatch")


def source_seals(paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, relative in paths.items():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
        output[key] = {
            "path": relative.replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def artifact_seal(path: Path, base: Path = ROOT) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _opaque(prefix: str, *parts: Any) -> str:
    return prefix + sha256_json([PHASE, *parts])[:32]


def normalized_prompt(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip().casefold()


def phase990_bridge() -> dict[str, Any]:
    for path in (
        PHASE990_DATASET,
        PHASE990_AUDIT,
        PHASE990_PROTOCOL,
        PHASE990_ERRATUM,
    ):
        require(path.is_file() and not path.is_symlink(), f"Phase990 bridge missing: {path}")
    corpus = load_json(PHASE990_DATASET)
    audit = load_json(PHASE990_AUDIT)
    protocol = load_json(PHASE990_PROTOCOL)
    erratum = load_json(PHASE990_ERRATUM)
    p990_core.verify_self_hash(corpus, "dataset_sha256", "Phase990 dataset")
    p990_core.verify_self_hash(audit, "dataset_audit_sha256", "Phase990 dataset audit")
    p990_core.verify_self_hash(protocol, "protocol_sha256", "Phase990 protocol")
    p990_core.verify_self_hash(
        erratum, "protocol_audit_erratum_v2_sha256", "Phase990 erratum"
    )
    require(audit.get("passed") is True, "Phase990 dataset audit not passed")
    require(erratum.get("passed") is True, "Phase990 erratum not passed")
    require(len(corpus.get("worlds", [])) == 320, "Phase990 world count drift")
    require(len(corpus.get("records", [])) == 10240, "Phase990 record count drift")
    require(
        protocol.get("phase990_decision", {}).get("gpu_generation_admission")
        == "not_tested",
        "Phase990 admission status drift",
    )
    return {
        "dataset_sha256": corpus["dataset_sha256"],
        "dataset_file_sha256": sha256_file(PHASE990_DATASET),
        "dataset_audit_sha256": audit["dataset_audit_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "erratum_sha256": erratum["protocol_audit_erratum_v2_sha256"],
        "erratum_file_sha256": sha256_file(PHASE990_ERRATUM),
        "world_count": len(corpus["worlds"]),
        "record_count": len(corpus["records"]),
        "cpu_protocol_qualified": True,
        "gpu_behavior_pre_phase991": "not_tested",
    }


def _existing_sets(corpus: Mapping[str, Any]) -> dict[str, set[str]]:
    records = corpus["records"]
    return {
        "abstract": {
            str(record["slot_canonical_semantic_sha256"]) for record in records
        },
        "observable": {
            str(record["observable_semantic_variant_sha256"]) for record in records
        },
        "prompt": {
            str(record["normalized_surface_sha256"]) for record in records
        },
        "record_id": {str(record["record_id"]) for record in records},
        "world_id": {str(world["semantic_world_id"]) for world in corpus["worlds"]},
    }


def _random_owner_permutation(
    rng: random.Random, query_entity_slot: int, target_object_slot: int
) -> list[int]:
    remaining_entities = [slot for slot in range(4) if slot != query_entity_slot]
    remaining_objects = [slot for slot in range(4) if slot != target_object_slot]
    rng.shuffle(remaining_objects)
    owner = [-1] * 4
    owner[query_entity_slot] = target_object_slot
    for entity_slot, object_slot in zip(remaining_entities, remaining_objects):
        owner[entity_slot] = object_slot
    require(sorted(owner) == list(range(4)), "owner permutation construction failed")
    return owner


def _random_attribute_slots(
    rng: random.Random,
    query_relation_slot: int,
    answer_value_slot: int,
    target_object_slot: int,
) -> list[list[int]]:
    rows: list[list[int]] = [[-1] * 4 for _ in range(2)]
    primary_remaining_positions = [
        slot for slot in range(4) if slot != target_object_slot
    ]
    primary_remaining_values = [
        slot for slot in range(4) if slot != answer_value_slot
    ]
    rng.shuffle(primary_remaining_values)
    rows[query_relation_slot][target_object_slot] = answer_value_slot
    for position, value in zip(primary_remaining_positions, primary_remaining_values):
        rows[query_relation_slot][position] = value

    other_relation = 1 - query_relation_slot
    value_partner_position = (target_object_slot + 2) % 4
    binding_partner_position = (target_object_slot + 3) % 4
    reserved = {
        rows[query_relation_slot][target_object_slot],
        rows[query_relation_slot][value_partner_position],
        rows[query_relation_slot][binding_partner_position],
    }
    relation_swap_value = next(value for value in range(4) if value not in reserved)
    other_values = [-1] * 4
    other_values[target_object_slot] = relation_swap_value
    other_positions = [slot for slot in range(4) if slot != target_object_slot]
    other_remaining = [value for value in range(4) if value != relation_swap_value]
    while True:
        rng.shuffle(other_remaining)
        for position, value in zip(other_positions, other_remaining):
            other_values[position] = value
        if all(
            other_values[position] != rows[query_relation_slot][position]
            for position in range(4)
        ):
            break
    rows[other_relation] = list(other_values)
    require(all(sorted(row) == list(range(4)) for row in rows), "attribute rows")
    return rows


def _abstract_hashes_from_slots(
    query_entity_slot: int,
    query_relation_slot: int,
    owner_permutation: Sequence[int],
    attribute_slots: Sequence[Sequence[int]],
    target_object_slot: int,
) -> frozenset[str]:
    # Reuse only the already sealed transform semantics, not its constrained
    # Phase990 schedule.  This extension explores unused permutation states.
    base = {
        "entities": [f"e{index}" for index in range(4)],
        "objects": [f"o{index}" for index in range(4)],
        "values": list(VALUES),
        "owner_edges": [
            {
                "fact_id": f"owner_e{entity}",
                "entity_slot": entity,
                "entity": f"e{entity}",
                "relation": p990_core.OWNER_RELATION,
                "object_slot": int(owner_permutation[entity]),
                "object": f"o{int(owner_permutation[entity])}",
            }
            for entity in range(4)
        ],
        "attribute_edges": [
            {
                "fact_id": f"attribute_r{relation}_o{obj}",
                "object_slot": obj,
                "object": f"o{obj}",
                "relation_slot": relation,
                "relation": p990_core.ATTRIBUTE_RELATIONS[relation],
                "value_slot": int(attribute_slots[relation][obj]),
                "value": VALUES[int(attribute_slots[relation][obj])],
            }
            for relation in range(2)
            for obj in range(4)
        ],
        "query": {
            "entity_slot": query_entity_slot,
            "entity": f"e{query_entity_slot}",
            "relation_slot": query_relation_slot,
            "relation": p990_core.ATTRIBUTE_RELATIONS[query_relation_slot],
        },
        "counterfactual_partners": {
            "value_partner_object_slot": (target_object_slot + 2) % 4,
            "binding_partner_object_slot": (target_object_slot + 3) % 4,
            "binding_partner_entity_slot": list(owner_permutation).index(
                (target_object_slot + 3) % 4
            ),
        },
    }
    p990_data._validate_graph(base)
    hashes = {
        p990_core.sha256_json(
            p990_data._slot_semantics(p990_data.transform_state(base, semantic))
        )
        for semantic in p990_core.SEMANTIC_TRANSFORMS
    }
    require(len(hashes) == 4, "extension counterfactual closure collapsed")
    return frozenset(hashes)


def _lexical_entities(
    rng: random.Random,
    query_entity_slot: int,
    query_relation_slot: int,
    answer_value_slot: int,
    repetition: int,
) -> list[str]:
    query_name_index = (
        query_entity_slot
        + 4 * ((answer_value_slot + query_relation_slot + repetition) % 2)
    )
    remaining = [index for index in range(8) if index != query_name_index]
    rng.shuffle(remaining)
    selected = remaining[:3]
    output: list[str | None] = [None] * 4
    output[query_entity_slot] = ENTITY_LEXICON[query_name_index]
    for slot, name_index in zip(
        [slot for slot in range(4) if slot != query_entity_slot], selected
    ):
        output[slot] = ENTITY_LEXICON[name_index]
    require(len(set(output)) == 4 and all(isinstance(x, str) for x in output), "entities")
    return [str(x) for x in output]


def _lexical_objects(
    target_object_slot: int, answer_value_slot: int, repetition: int
) -> list[str]:
    target_name_index = (answer_value_slot + repetition) % 4
    output: list[str | None] = [None] * 4
    for relative in range(4):
        output[(target_object_slot + relative) % 4] = OBJECT_LEXICON[
            (target_name_index + relative) % 4
        ]
    require(len(set(output)) == 4 and all(isinstance(x, str) for x in output), "objects")
    return [str(x) for x in output]


def _build_world(
    seed: int,
    ordinal: int,
    query_entity_slot: int,
    query_relation_slot: int,
    answer_value_slot: int,
    repetition: int,
    target_object_slot: int,
    owner_permutation: Sequence[int],
    attribute_slots: Sequence[Sequence[int]],
    lexical_rng: random.Random,
) -> dict[str, Any]:
    entities = _lexical_entities(
        lexical_rng,
        query_entity_slot,
        query_relation_slot,
        answer_value_slot,
        repetition,
    )
    objects = _lexical_objects(target_object_slot, answer_value_slot, repetition)
    owner_edges = [
        {
            "fact_id": f"owner_e{entity_slot}",
            "entity_slot": entity_slot,
            "entity": entities[entity_slot],
            "relation": p990_core.OWNER_RELATION,
            "object_slot": int(owner_permutation[entity_slot]),
            "object": objects[int(owner_permutation[entity_slot])],
        }
        for entity_slot in range(4)
    ]
    attribute_edges = [
        {
            "fact_id": f"attribute_r{relation_slot}_o{object_slot}",
            "object_slot": object_slot,
            "object": objects[object_slot],
            "relation_slot": relation_slot,
            "relation": p990_core.ATTRIBUTE_RELATIONS[relation_slot],
            "value_slot": int(attribute_slots[relation_slot][object_slot]),
            "value": VALUES[int(attribute_slots[relation_slot][object_slot])],
        }
        for relation_slot in range(2)
        for object_slot in range(4)
    ]
    binding_object = (target_object_slot + 3) % 4
    world_id = _opaque("p991_w_", seed, ordinal, "world")
    family_id = _opaque("p991_f_", seed, ordinal, "family")
    world = {
        "root_family_id": family_id,
        "semantic_world_id": world_id,
        "split": EXTENSION_SPLIT,
        "split_ordinal": ordinal,
        "global_ordinal": 320 + ordinal,
        "local_rep": repetition,
        "structural_rep": 10_000 + repetition,
        "structural_component_id": sha256_json({
            "owner": list(owner_permutation),
            "attributes": [list(row) for row in attribute_slots],
        }),
        "seed_key": f"phase991:{seed}:{ordinal}:seeded-extension-v1",
        "generator_seed": seed,
        "entities": entities,
        "objects": objects,
        "values": list(VALUES),
        "owner_edges": owner_edges,
        "attribute_edges": attribute_edges,
        "query": {
            "entity_slot": query_entity_slot,
            "entity": entities[query_entity_slot],
            "relation_slot": query_relation_slot,
            "relation": p990_core.ATTRIBUTE_RELATIONS[query_relation_slot],
        },
        "counterfactual_partners": {
            "value_partner_object_slot": (target_object_slot + 2) % 4,
            "binding_partner_entity_slot": list(owner_permutation).index(binding_object),
            "binding_partner_object_slot": binding_object,
        },
        "base_target_object_slot": target_object_slot,
        "base_query_entity_slot": query_entity_slot,
        "base_query_relation_slot": query_relation_slot,
        "base_answer_value_slot": answer_value_slot,
        "independent_unit": True,
        "factor_grid_rows_are_paired": True,
    }
    p990_data._validate_graph(world)
    solved = p990_data.solve_state(world)
    require(solved["answer_value_slot"] == answer_value_slot, "gold construction")
    return world


def _rewrite_item_identity(item: dict[str, Any], world: Mapping[str, Any]) -> None:
    variant = str(item["variant_id"])
    item["record_id"] = _opaque("p991_i_", world["semantic_world_id"], variant)
    factors = p990_data.parse_variant(variant)
    original_variant = p990_data.variant_id(
        "original",
        factors["paraphrase_id"],
        factors["fact_order_id"],
        factors["horizon_id"],
    )
    item["pair_links"]["original_surface_peer_record_id"] = _opaque(
        "p991_i_", world["semantic_world_id"], original_variant
    )
    item["pair_links"]["semantic_peer_record_ids"] = {
        semantic: _opaque(
            "p991_i_",
            world["semantic_world_id"],
            p990_data.variant_id(
                semantic,
                factors["paraphrase_id"],
                factors["fact_order_id"],
                factors["horizon_id"],
            ),
        )
        for semantic in p990_core.SEMANTIC_TRANSFORMS
    }
    item["phase"] = PHASE
    item["source_renderer"] = "sealed_phase990_renderer_with_phase991_identity_rewrite"


def _build_extension_item(world: Mapping[str, Any], variant: str) -> dict[str, Any]:
    # Phase990's renderer has a sealed finite order table keyed by its four
    # original split names.  The textual renderer itself is reusable, but the
    # table has no Phase991 key.  A deterministic proxy index selects an
    # already-audited order pattern; split identity is restored immediately
    # afterwards and is not part of any semantic hash.
    proxy = deepcopy(dict(world))
    proxy["split"] = "discovery"
    proxy["split_ordinal"] = int(world["split_ordinal"]) % 96
    item = p990_data.build_item(proxy, variant)
    item["split"] = EXTENSION_SPLIT
    item["split_ordinal"] = int(world["split_ordinal"])
    state = item["semantic_state"]
    state["split"] = EXTENSION_SPLIT
    state["split_ordinal"] = int(world["split_ordinal"])
    state["global_ordinal"] = int(world["global_ordinal"])
    state["seed_key"] = str(world["seed_key"])
    state["generator_seed"] = int(world["generator_seed"])
    return item


def generate_extension(
    seed: int = EXTENSION_SEED,
    phase990_corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be int")
    require(0 <= seed < 2**64, "seed outside frozen uint64 domain")
    corpus = (
        dict(phase990_corpus)
        if phase990_corpus is not None
        else load_json(PHASE990_DATASET)
    )
    existing = _existing_sets(corpus)
    rng = random.Random(seed)
    lexical_rng = random.Random(seed ^ 0xA5A5_A5A5_5A5A_5A5A)
    used_abstract = set(existing["abstract"])
    extension_abstract: set[str] = set()
    worlds: list[dict[str, Any]] = []
    attempt_counts: list[int] = []

    # 4 query slots x 2 relations x 4 answers x 4 repetitions = 128.
    cells = [
        (query, relation, answer, repetition)
        for repetition in range(4)
        for query in range(4)
        for relation in range(2)
        for answer in range(4)
    ]
    require(len(cells) == EXTENSION_WORLD_COUNT, "cell schedule size")
    for ordinal, (query, relation, answer, repetition) in enumerate(cells):
        accepted: tuple[int, list[int], list[list[int]], frozenset[str]] | None = None
        for attempt in range(1, 100_001):
            target = rng.randrange(4)
            owner = _random_owner_permutation(rng, query, target)
            attributes = _random_attribute_slots(rng, relation, answer, target)
            try:
                closure = _abstract_hashes_from_slots(
                    query, relation, owner, attributes, target
                )
            except RuntimeError:
                # Some otherwise valid base graphs cease to satisfy the
                # Phase990 cross-relation derangement after the registered
                # value-swap.  They are outside this extension domain.
                continue
            if closure & used_abstract or closure & extension_abstract:
                continue
            accepted = (target, owner, attributes, closure)
            attempt_counts.append(attempt)
            extension_abstract.update(closure)
            break
        require(accepted is not None, f"no unused closure for extension cell {ordinal}")
        target, owner, attributes, _closure = accepted
        worlds.append(_build_world(
            seed,
            ordinal,
            query,
            relation,
            answer,
            repetition,
            target,
            owner,
            attributes,
            lexical_rng,
        ))

    records: list[dict[str, Any]] = []
    for world in worlds:
        for variant in EXTENSION_VARIANTS:
            item = _build_extension_item(world, variant)
            _rewrite_item_identity(item, world)
            records.append(item)

    result = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "seeded_expanded_confirmation_dataset_no_model_results",
        "generator": {
            "algorithm": "Python random.Random MT19937 with frozen ordered cell schedule",
            "seed": seed,
            "seed_domain": "unsigned 64-bit integer",
            "cell_order": "repetition,query_entity_slot,query_relation_slot,answer_value_slot",
            "rejection_rule": "reject any four-transform closure overlapping old or earlier extension abstract states",
            "maximum_attempts_per_world": 100_000,
            "attempt_count_min": min(attempt_counts),
            "attempt_count_max": max(attempt_counts),
            "attempt_count_total": sum(attempt_counts),
            "lexical_generalization_tested": False,
        },
        "counts": {
            "worlds": len(worlds),
            "records": len(records),
            "variants_per_world": len(EXTENSION_VARIANTS),
            "abstract_semantic_states": len(extension_abstract),
        },
        "independent_unit": "semantic_world_id",
        "factor_grid_rows_are_paired_not_independent": True,
        "worlds": worlds,
        "records": records,
    }
    result["extension_payload_sha256"] = sha256_json(result)
    return result


def _majority_baseline(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    denominator = 0
    for row in rows:
        state = row["semantic_state"]
        key = (
            str(state["query"]["entity"]),
            str(row["gold"]["answer_object"]),
            str(state["query"]["relation"]),
        )
        cells[key][str(row["gold"]["answer_value"])] += 1
        denominator += 1
    majority = sum(max(counter.values()) for counter in cells.values())
    return {
        "feature": "query_entity_name+selected_object_name+query_relation",
        "denominator": denominator,
        "nonempty_feature_cells": len(cells),
        "majority_correct": majority,
        "accuracy_percent": 100.0 * majority / denominator if denominator else 0.0,
        "cell_count_sha256": sha256_json({
            "|".join(key): dict(sorted(value.items()))
            for key, value in sorted(cells.items())
        }),
    }


def _representative_semantic_rows(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in records
        if row["paraphrase_id"] == "standard"
        and row["fact_order_id"] == "order_a"
        and row["horizon_id"] == "near"
    ]


def audit_extension(
    extension: Mapping[str, Any],
    phase990_corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    corpus = (
        dict(phase990_corpus)
        if phase990_corpus is not None
        else load_json(PHASE990_DATASET)
    )
    existing = _existing_sets(corpus)
    worlds = list(extension.get("worlds", []))
    records = list(extension.get("records", []))
    errors: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    check(len(worlds) == EXTENSION_WORLD_COUNT, "WORLD_COUNT")
    check(len(records) == EXTENSION_RECORD_COUNT, "RECORD_COUNT")
    check(len({w.get("semantic_world_id") for w in worlds}) == len(worlds), "WORLD_IDS")
    check(len({r.get("record_id") for r in records}) == len(records), "RECORD_IDS")
    check(not ({w.get("semantic_world_id") for w in worlds} & existing["world_id"]), "OLD_WORLD_ID_OVERLAP")
    check(not ({r.get("record_id") for r in records} & existing["record_id"]), "OLD_RECORD_ID_OVERLAP")

    abstract = {str(row.get("slot_canonical_semantic_sha256")) for row in records}
    observable = {str(row.get("observable_semantic_variant_sha256")) for row in records}
    prompts = {str(row.get("normalized_surface_sha256")) for row in records}
    check(len(abstract) == 512, "ABSTRACT_UNIQUE")
    check(len(observable) == 512, "OBSERVABLE_UNIQUE")
    check(len(prompts) == EXTENSION_RECORD_COUNT, "PROMPT_UNIQUE")
    check(not (abstract & existing["abstract"]), "ABSTRACT_OVERLAP")
    check(not (observable & existing["observable"]), "OBSERVABLE_OVERLAP")
    check(not (prompts & existing["prompt"]), "PROMPT_OVERLAP")

    cell_counts = Counter(
        (
            int(world["base_query_entity_slot"]),
            int(world["base_query_relation_slot"]),
            int(world["base_answer_value_slot"]),
        )
        for world in worlds
    )
    expected_cells = Counter({
        (query, relation, answer): 4
        for query in range(4)
        for relation in range(2)
        for answer in range(4)
    })
    check(cell_counts == expected_cells, "QRA_GRID")

    world_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        world_groups[str(row.get("semantic_world_id"))].append(row)
    check(all(len(rows) == 32 for rows in world_groups.values()), "VARIANTS_PER_WORLD")
    check(all(
        {str(row.get("variant_id")) for row in rows} == set(EXTENSION_VARIANTS)
        for rows in world_groups.values()
    ), "VARIANT_GRID")

    mechanical_gold = 0
    counterfactual_distinct = 0
    involutions = 0
    for world in worlds:
        p990_data._validate_graph(world)
        solution = p990_data.solve_state(world)
        if int(solution["answer_value_slot"]) == int(world["base_answer_value_slot"]):
            mechanical_gold += 1
        transformed_answers = []
        for semantic in p990_core.SEMANTIC_TRANSFORMS:
            transformed = p990_data.transform_state(world, semantic)
            transformed_answers.append(p990_data.solve_state(transformed)["answer_value"])
            if p990_data._apply_transform_twice(world, semantic):
                involutions += 1
        if len(set(transformed_answers)) == 4:
            counterfactual_distinct += 1
    check(mechanical_gold == EXTENSION_WORLD_COUNT, "MECHANICAL_GOLD")
    check(counterfactual_distinct == EXTENSION_WORLD_COUNT, "COUNTERFACTUAL_DISTINCT")
    check(involutions == EXTENSION_WORLD_COUNT * 4, "INVOLUTIONS")

    lexical_query_answer = Counter(
        (
            str(world["query"]["entity"]),
            VALUES[int(world["base_answer_value_slot"])],
        )
        for world in worlds
    )
    lexical_query_relation = Counter(
        (str(world["query"]["entity"]), str(world["query"]["relation"]))
        for world in worlds
    )
    check(
        set(lexical_query_answer.values()) == {4}
        and len(lexical_query_answer) == 32,
        "QUERY_NAME_ANSWER_GRID",
    )
    check(
        set(lexical_query_relation.values()) == {8}
        and len(lexical_query_relation) == 16,
        "QUERY_NAME_RELATION_GRID",
    )

    extension_baseline = _majority_baseline(_representative_semantic_rows(records))
    primary_baselines: dict[str, Any] = {}
    for split in (*PUBLIC_SPLITS, SEALED_SPLIT):
        primary_baselines[split] = _majority_baseline(
            row for row in _representative_semantic_rows(corpus["records"])
            if row["split"] == split
        )

    # Seed must matter.  A small deterministic fingerprint of the first eight
    # worlds is sufficient for the self-test; full second generation is done
    # by the protocol and independent audit.
    seed_fingerprint = sha256_json([
        {
            "owner": [edge["object_slot"] for edge in world["owner_edges"]],
            "attributes": [edge["value_slot"] for edge in world["attribute_edges"]],
            "entities": world["entities"],
        }
        for world in worlds[:8]
    ])

    result = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "role": "cpu_only_seeded_extension_audit",
        "passed": not errors,
        "errors": errors,
        "counts": {
            "worlds": len(worlds),
            "records": len(records),
            "abstract_semantic_states": len(abstract),
            "observable_semantic_states": len(observable),
            "normalized_prompts": len(prompts),
            "mechanical_gold_worlds": mechanical_gold,
            "four_distinct_counterfactual_answers_worlds": counterfactual_distinct,
            "involution_checks": involutions,
        },
        "three_level_overlap": {
            "abstract_semantic": len(abstract & existing["abstract"]),
            "observable_semantic": len(observable & existing["observable"]),
            "normalized_prompt": len(prompts & existing["prompt"]),
        },
        "balance": {
            "q_relation_answer_cells": len(cell_counts),
            "count_per_cell": 4,
            "query_name_answer_cells": len(lexical_query_answer),
            "query_name_answer_count_per_cell": 4,
            "query_name_relation_cells": len(lexical_query_relation),
            "query_name_relation_count_per_cell": 8,
        },
        "shortcut_contract": {
            "chosen_option": "B_freeze_matched_baseline_and_required_interventions",
            "shortcut_claimed_eliminated": False,
            "primary_selected_owner_conjunction_baselines": primary_baselines,
            "extension_selected_owner_conjunction_baseline": extension_baseline,
            "chance_percent": 25.0,
            "behavior_above_baseline_is_not_second_hop_proof": True,
            "future_mechanism_requires_relation_value_binding_and_wrong_donor_controls": True,
        },
        "lexical_generalization_tested": False,
        "seed_fingerprint_first_eight_worlds": seed_fingerprint,
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    require(result["passed"], f"extension audit failed: {errors}")
    return result


def extension_document(created_at_utc: str, seed: int = EXTENSION_SEED) -> dict[str, Any]:
    extension = generate_extension(seed)
    body = dict(extension)
    body.pop("extension_payload_sha256", None)
    return sealed_document(body, "extension_dataset_sha256", created_at_utc)


def self_test() -> dict[str, Any]:
    bridge = phase990_bridge()
    corpus = load_json(PHASE990_DATASET)
    first = generate_extension(EXTENSION_SEED, corpus)
    second = generate_extension(EXTENSION_SEED, corpus)
    third = generate_extension(EXTENSION_SEED + 1, corpus)
    first_audit = audit_extension(first, corpus)
    second_audit = audit_extension(second, corpus)
    third_audit = audit_extension(third, corpus)
    checks = {
        "phase990_bridge_qualified": bridge["cpu_protocol_qualified"],
        "deterministic_same_seed": first == second,
        "seed_changes_output": first["extension_payload_sha256"] != third["extension_payload_sha256"],
        "first_audit_passed": first_audit["passed"],
        "second_audit_passed": second_audit["passed"],
        "alternate_seed_audit_passed": third_audit["passed"],
        "world_count": first["counts"]["worlds"] == EXTENSION_WORLD_COUNT,
        "record_count": first["counts"]["records"] == EXTENSION_RECORD_COUNT,
        "abstract_overlap_zero": first_audit["three_level_overlap"]["abstract_semantic"] == 0,
        "observable_overlap_zero": first_audit["three_level_overlap"]["observable_semantic"] == 0,
        "prompt_overlap_zero": first_audit["three_level_overlap"]["normalized_prompt"] == 0,
        "shortcut_not_falsely_eliminated": first_audit["shortcut_contract"]["shortcut_claimed_eliminated"] is False,
        "no_model_weights": first_audit["model_weights_loaded"] is False,
        "no_cuda": first_audit["cuda_used"] is False,
    }
    require(all(checks.values()), f"Phase991 core self-test failed: {checks}")
    return {"passed": True, "checks": checks}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    require(arguments.self_test, "only --self-test is supported by the core module")
    print(json.dumps(self_test(), ensure_ascii=False, indent=2, sort_keys=True))
