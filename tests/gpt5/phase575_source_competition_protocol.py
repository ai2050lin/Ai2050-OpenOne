#!/usr/bin/env python3
"""Freeze Phase575 worlds for data-first source-competition discovery."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase569_relation_competition_protocol as p569  # noqa: E402
import phase573_natural_transition_protocol as p573  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase575"
MODELS = ("qwen3", "glm4", "deepseek7b")
AXES = ("relation", "object", "relation_object", "order")
VARIANTS = (
    "base",
    "relation_swap",
    "object_swap",
    "relation_object_swap",
    "order_swap",
)
STRUCTURE_SPLITS = (
    "structure_discovery",
    "structure_confirmation",
    "heldout_recombination",
)
CAUSAL_SPLITS = ("causal_discovery", "causal_confirmation")
OPEN_SPLITS = (*STRUCTURE_SPLITS, *CAUSAL_SPLITS)
ALL_SPLITS = (*OPEN_SPLITS, "sealed")
SPLIT_SPECS = {
    "structure_discovery": ("path_discovery", 1_000_000),
    "structure_confirmation": ("path_discovery", 1_050_000),
    "causal_discovery": ("phenotype_discovery", 1_100_000),
    "causal_confirmation": ("phenotype_confirmation", 1_150_000),
    "heldout_recombination": ("path_confirmation", 1_200_000),
    "sealed": ("sealed", 1_250_000),
}
CANDIDATE_WORLDS_PER_SPLIT = 1024
CONTROL_SCREEN_CAP_PER_SPLIT_MODEL = 384
FINAL_WORLDS_PER_SPLIT_MODEL = 128
FIXED_BATCH_SIZE = 8
NOOP_REPEATS = 2

OUT_DIR = ROOT / "tests/gpt5/result/phase575_source_competition"
OPEN_CASES_PATH = OUT_DIR / "phase575_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase575_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase575_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase575_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase575_static_audit.json"
PHASE574_DECISION = (
    ROOT / "tests/gpt5/result/phase574_query_source/"
    "phase574_query_source_causal_decision.json"
)
PRIOR_OPEN_CASE_BANKS = (
    ROOT / "tests/gpt5/result/phase571_relation_block/phase571_open_cases.jsonl.gz",
    ROOT / "tests/gpt5/result/phase572_relation_joint/phase572_open_cases.jsonl.gz",
    ROOT / "tests/gpt5/result/phase573_natural_transition/phase573_open_cases.jsonl.gz",
    ROOT / "tests/gpt5/result/phase574_query_source/phase574_open_cases.jsonl.gz",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def normalized_words(text: str) -> list[str]:
    return sorted(text.lower().replace("\n", " ").split())


def object_counterfactual_index(binding: int, query_index: int) -> int:
    # These shifts make base, relation, object, and joint targets all distinct.
    shift = {0: 2, 1: 1, 2: 2}[binding]
    return (query_index + shift) % 4


def materialize_variant(
    tokenizers: dict[str, Any],
    row: dict[str, Any],
    split: str,
    world_rank: int,
    variant: str,
    base_case_id: str,
    anchor_fragments: dict[str, str],
) -> dict[str, Any]:
    candidate_ids_by_model = {}
    prompt_counts = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, row["raw_prompt"])
        prompt_counts[model] = len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
        candidate_ids_by_model[model] = {
            value: [
                int(token)
                for token in tokenizer(value, add_special_tokens=False)["input_ids"]
            ]
            for value in row["all_candidates"]
        }
    axis = {
        "base": "base",
        "relation_swap": "relation",
        "object_swap": "object",
        "relation_object_swap": "relation_object",
        "order_swap": "order",
    }[variant]
    return {
        **row,
        "schema_version": "phase575_source_competition_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "split": split,
        "world_rank": world_rank,
        "variant": variant,
        "counterfactual_axis": axis,
        "base_case_id": base_case_id,
        "case_id": f"{base_case_id}_{variant}",
        "physical_anchor_fragments": anchor_fragments,
        "prompt_token_count_by_model": prompt_counts,
        "candidate_token_ids_by_model": candidate_ids_by_model,
        "query_object_count": 4,
        "sealed": split == "sealed",
    }


def build_world(
    tokenizers: dict[str, Any], split: str, world_rank: int
) -> list[dict[str, Any]]:
    source_split, source_base = SPLIT_SPECS[split]
    source_index = source_base + world_rank
    binding, query_index, query_relation, surface, fact_order = p573.factor_grid()[
        world_rank % len(p573.factor_grid())
    ]
    base = p569.controlled_case(
        source_split,
        source_index,
        binding,
        query_index,
        query_relation,
        surface,
        fact_order,
    )
    other_relation = "tag" if query_relation == "body" else "body"
    other_object_index = object_counterfactual_index(binding, query_index)
    relation = p573.update_query(base, query_index, other_relation)
    obj = p573.update_query(base, other_object_index, query_relation)
    joint = p573.update_query(base, other_object_index, other_relation)
    order = p573.order_variant(base)
    base_case_id = f"phase575_{split}_world{world_rank:04d}"
    anchor_fragments = dict(base["semantic_fragments"])
    return [
        materialize_variant(
            tokenizers, row, split, world_rank, variant, base_case_id,
            anchor_fragments,
        )
        for variant, row in (
            ("base", base),
            ("relation_swap", relation),
            ("object_swap", obj),
            ("relation_object_swap", joint),
            ("order_swap", order),
        )
    ]


def freeze() -> dict[str, Any]:
    phase574 = read_json(PHASE574_DECISION)
    if phase574["open_query_source_causal_gate_pass"]:
        raise RuntimeError("Phase575 assumes the Phase574 coarse bridge was closed")
    if phase574["sealed_split_read"]:
        raise RuntimeError("Phase575 requires the Phase574 seal to remain unread")

    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for split in ALL_SPLITS:
        target = sealed_rows if split == "sealed" else open_rows
        for world_rank in range(CANDIDATE_WORLDS_PER_SPLIT):
            target.extend(build_world(tokenizers, split, world_rank))

    failures = []
    expected_open = len(OPEN_SPLITS) * CANDIDATE_WORLDS_PER_SPLIT * len(VARIANTS)
    expected_sealed = CANDIDATE_WORLDS_PER_SPLIT * len(VARIANTS)
    if len(open_rows) != expected_open:
        failures.append("open_case_count")
    if len(sealed_rows) != expected_sealed:
        failures.append("sealed_case_count")
    all_rows = open_rows + sealed_rows
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        failures.append("case_id_collision")
    if any(
        len(ids) != 1
        for row in all_rows
        for model_ids in row["candidate_token_ids_by_model"].values()
        for ids in model_ids.values()
    ):
        failures.append("candidate_not_single_token")

    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_rows:
        by_base.setdefault(row["base_case_id"], {})[row["variant"]] = row
    invariant_failures = 0
    for group in by_base.values():
        if set(group) != set(VARIANTS):
            invariant_failures += 1
            continue
        base = group["base"]
        relation = group["relation_swap"]
        obj = group["object_swap"]
        joint = group["relation_object_swap"]
        order = group["order_swap"]
        checks = (
            base["context"] == relation["context"] == obj["context"] == joint["context"],
            len({base["target"], relation["target"], obj["target"], joint["target"]}) == 4,
            base["query_relation"] == obj["query_relation"],
            relation["query_relation"] == joint["query_relation"],
            base["query_object"] == relation["query_object"],
            obj["query_object"] == joint["query_object"],
            base["query_object"] != obj["query_object"],
            base["target"] == order["target"],
            stable_hash(normalized_words(base["context"]))
            == stable_hash(normalized_words(order["context"])),
        )
        if not all(checks):
            invariant_failures += 1
    if invariant_failures:
        failures.append("counterfactual_invariants")

    prior_open_objects = {
        obj
        for path in PRIOR_OPEN_CASE_BANKS
        for row in iter_jsonl(path)
        for obj in row["objects"]
    }
    new_objects = {obj for row in all_rows for obj in row["objects"]}
    prior_open_overlap = len(prior_open_objects & new_objects)
    if prior_open_overlap:
        failures.append("prior_open_object_overlap")
    if failures:
        raise RuntimeError(
            f"Phase575 static audit failed: {failures}; invariants={invariant_failures}"
        )

    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase575_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_behavior_executed": False,
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    frozen = {
        "schema_version": "phase575_source_competition_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models_in_required_execution_order": list(MODELS),
        "axes": list(AXES),
        "variants": list(VARIANTS),
        "structure_splits": list(STRUCTURE_SPLITS),
        "causal_splits": list(CAUSAL_SPLITS),
        "candidate_worlds_per_split": CANDIDATE_WORLDS_PER_SPLIT,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "noop_repeats": NOOP_REPEATS,
        "control_screen_cap_per_split_model": CONTROL_SCREEN_CAP_PER_SPLIT_MODEL,
        "final_worlds_per_split_model": FINAL_WORLDS_PER_SPLIT_MODEL,
        "behavior_gate": {
            "minimum_relation_qualified_worlds_each_structure_split": 192,
            "minimum_five_variant_qualified_worlds_each_structure_split": 128,
            "minimum_five_variant_qualified_worlds_heldout": 128,
            "both_noop_repeats_exact_and_semantic_stability_required": True,
        },
        "discovery_policy": {
            "natural_trace_must_precede_causal_candidate_freeze": True,
            "record_every_model_layer": True,
            "record_architecture_components_without_mechanism_labels": [
                "post_rotary_query",
                "post_rotary_key",
                "pre_softmax_score",
                "post_softmax_weight",
                "value",
                "projected_value_message",
                "attention_output",
                "residual_state",
            ],
            "full_vectors_only_in_open_structure_discovery": True,
            "confirmation_and_heldout_use_frozen_discovered_events": True,
            "causal_candidate_must_repeat_in_discovery_confirmation_and_heldout": True,
            "no_output_embedding_direction_for_discovery": True,
            "right_padding_and_explicit_position_ids_required": True,
            "causal_prefix_max_relative_delta": 1e-5,
        },
        "causal_policy": {
            "not_frozen_until_natural_replication_passes": True,
            "direct_overwrite_restore_required": True,
            "value_preserving_selection_controls_required": True,
            "full_short_generation_required_after_open_confirmation": True,
            "world_level_pipeline_resamples": 1024,
        },
        "head_channel_parameter_neuron_scan_allowed": False,
        "phase574_decision_sha256": sha256_file(PHASE574_DECISION),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "phase574_sealed_file_not_read": True,
        "sealed_object_disjointness_basis": (
            "injective alpha_code source indices use non-overlapping frozen ranges"
        ),
        "sealed_split_read_for_analysis": False,
    }
    write_json(PROTOCOL_PATH, frozen)
    audit = {
        "schema_version": "phase575_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": True,
        "failures": [],
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "world_group_count": len(by_base),
        "counterfactual_invariant_failure_count": 0,
        "prior_open_object_overlap_count": prior_open_overlap,
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model_execution_performed": False,
        "phase574_sealed_file_read": False,
        "sealed_split_read": False,
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "open_cases": len(open_rows),
        "sealed_cases": len(sealed_rows),
        "world_groups": len(by_base),
        "prior_open_object_overlap": prior_open_overlap,
        "five_variant_target_collision_count": 0,
        "valid": True,
    }, ensure_ascii=False, indent=2))
    return frozen


if __name__ == "__main__":
    freeze()
