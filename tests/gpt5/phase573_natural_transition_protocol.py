#!/usr/bin/env python3
"""Freeze Phase573 natural query-relation, object, and order counterfactual worlds."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase569_relation_competition_protocol as p569  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase573"
MODELS = ("qwen3", "glm4", "deepseek7b")
AXES = ("relation", "object", "order")
VARIANTS = ("base", "relation_swap", "object_swap", "order_swap")
STRUCTURE_SPLITS = (
    "structure_discovery",
    "structure_confirmation",
    "heldout_recombination",
)
CAUSAL_SPLITS = ("causal_discovery", "causal_confirmation")
OPEN_SPLITS = (*STRUCTURE_SPLITS, *CAUSAL_SPLITS)
ALL_SPLITS = (*OPEN_SPLITS, "sealed")
SPLIT_SPECS = {
    "structure_discovery": ("path_discovery", 240000),
    "structure_confirmation": ("path_discovery", 280000),
    "causal_discovery": ("phenotype_discovery", 320000),
    "causal_confirmation": ("phenotype_confirmation", 360000),
    "heldout_recombination": ("path_confirmation", 400000),
    "sealed": ("sealed", 440000),
}
CANDIDATE_WORLDS_PER_SPLIT = 1024
CONTROL_SCREEN_CAP_PER_SPLIT_MODEL = 384
FINAL_WORLDS_PER_AXIS_SPLIT_MODEL = 128
FIXED_BATCH_SIZE = 8
NOOP_REPEATS = 2

OUT_DIR = ROOT / "tests/gpt5/result/phase573_natural_transition"
OPEN_CASES_PATH = OUT_DIR / "phase573_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase573_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase573_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase573_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase573_static_audit.json"
PHASE572_DECISION = (
    ROOT / "tests/gpt5/result/phase572_relation_joint/phase572_stage_decision.json"
)
PRIOR_OPEN_CASES = (
    ROOT / "tests/gpt5/result/phase571_relation_block/phase571_open_cases.jsonl.gz",
    ROOT / "tests/gpt5/result/phase572_relation_joint/phase572_open_cases.jsonl.gz",
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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
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


def factor_grid() -> list[tuple[int, int, str, int, int]]:
    return list(
        itertools.product(
            p569.BINDINGS,
            p569.QUERY_OBJECTS,
            p569.QUERY_RELATIONS,
            p569.SURFACES,
            p569.FACT_ORDERS,
        )
    )


def build_prompt(row: dict[str, Any], context: str, question: str) -> str:
    return f"{context}\nQuery: {question}\nInstruction: {row['instruction']}"


def fragments_for(
    records: list[dict[str, Any]], query_index: int, query_relation: str,
    query_label: str, query_object: str,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    target = next(
        record for record in records
        if record["object_index"] == query_index and record["relation"] == query_relation
    )
    other_relation = "tag" if query_relation == "body" else "body"
    other = next(
        record for record in records
        if record["object_index"] == query_index and record["relation"] == other_relation
    )
    fragments = {
        "target_fact": target["text"],
        "target_fact_object": target["object"],
        "target_fact_relation": target["relation_label"],
        "target_fact_value": target["value"],
        "other_fact": other["text"],
        "other_fact_object": other["object"],
        "other_fact_relation": other["relation_label"],
        "other_fact_value": other["value"],
        "query_relation": query_label,
        "query_object": query_object,
    }
    return fragments, target, other


def update_query(
    base: dict[str, Any], query_index: int, query_relation: str,
) -> dict[str, Any]:
    query_object = base["objects"][query_index]
    question, query_label = p569.render_question(
        int(base["surface_id"]), query_object, query_relation
    )
    fragments, target_record, other_record = fragments_for(
        base["fact_records"], query_index, query_relation, query_label, query_object
    )
    other_relation = "tag" if query_relation == "body" else "body"
    return {
        **base,
        "question": question,
        "raw_prompt": build_prompt(base, base["context"], question),
        "query_object_index": query_index,
        "query_object": query_object,
        "query_relation": query_relation,
        "other_relation": other_relation,
        "query_relation_label": query_label,
        "target": target_record["value"],
        "other_relation_target": other_record["value"],
        "target_aliases": [target_record["value"]],
        "distractors": [
            value for value in base["values"] if value != target_record["value"]
        ],
        "semantic_fragments": fragments,
    }


def reordered_records(records: list[dict[str, Any]], new_order: int) -> list[dict[str, Any]]:
    bank = {(row["relation"], row["object_index"]): row for row in records}
    if new_order == 0:
        return [
            bank[(relation, object_index)]
            for relation in ("tag", "body")
            for object_index in range(4)
        ]
    return [
        bank[(relation, object_index)]
        for object_index in (3, 2, 1, 0)
        for relation in ("tag", "body")
    ]


def order_variant(base: dict[str, Any]) -> dict[str, Any]:
    new_order = 1 - int(base["fact_order"])
    records = reordered_records(base["fact_records"], new_order)
    context = p569.render_context(int(base["surface_id"]), records)
    fragments, target_record, other_record = fragments_for(
        records,
        int(base["query_object_index"]),
        base["query_relation"],
        base["query_relation_label"],
        base["query_object"],
    )
    return {
        **base,
        "fact_records": records,
        "context": context,
        "raw_prompt": build_prompt(base, context, base["question"]),
        "fact_order": new_order,
        "target": target_record["value"],
        "other_relation_target": other_record["value"],
        "semantic_fragments": fragments,
    }


def materialize_variant(
    tokenizers: dict[str, Any], row: dict[str, Any], split: str,
    world_rank: int, variant: str, base_case_id: str,
    anchor_fragments: dict[str, str],
) -> dict[str, Any]:
    candidate_ids_by_model = {}
    prompt_counts = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, row["raw_prompt"])
        prompt_counts[model] = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        candidate_ids_by_model[model] = {
            value: [
                int(token)
                for token in tokenizer(value, add_special_tokens=False)["input_ids"]
            ]
            for value in row["all_candidates"]
        }
    case_id = f"{base_case_id}_{variant}"
    axis = {
        "base": "base",
        "relation_swap": "relation",
        "object_swap": "object",
        "order_swap": "order",
    }[variant]
    return {
        **row,
        "schema_version": "phase573_natural_counterfactual_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "split": split,
        "world_rank": world_rank,
        "variant": variant,
        "counterfactual_axis": axis,
        "base_case_id": base_case_id,
        "case_id": case_id,
        "pair_ids": {
            name: f"{base_case_id}_{name}" for name in AXES
        },
        "physical_anchor_fragments": anchor_fragments,
        "prompt_token_count_by_model": prompt_counts,
        "candidate_token_ids_by_model": candidate_ids_by_model,
        "sealed": split == "sealed",
    }


def build_world(
    tokenizers: dict[str, Any], split: str, world_rank: int,
) -> list[dict[str, Any]]:
    source_split, source_base = SPLIT_SPECS[split]
    source_index = source_base + world_rank
    binding, query_index, query_relation, surface, fact_order = factor_grid()[
        world_rank % len(factor_grid())
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
    base_case_id = f"phase573_{split}_world{world_rank:04d}"
    relation = update_query(
        base, query_index, "tag" if query_relation == "body" else "body"
    )
    obj = update_query(base, (query_index + 1) % 3, query_relation)
    order = order_variant(base)
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
            ("order_swap", order),
        )
    ]


def freeze() -> dict[str, Any]:
    prior = read_json(PHASE572_DECISION)
    if not prior["late_static_joint_role_route_closed"]:
        raise RuntimeError("Phase573 requires the Phase572 late static route to be closed")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows = []
    sealed_rows = []
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
    if any(row["target"] == row["other_relation_target"] for row in all_rows):
        failures.append("target_other_collision")
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
        order = group["order_swap"]
        checks = (
            base["context"] == relation["context"] == obj["context"],
            base["target"] == relation["other_relation_target"],
            base["other_relation_target"] == relation["target"],
            base["target"] != obj["target"],
            base["target"] == order["target"],
            base["other_relation_target"] == order["other_relation_target"],
            stable_hash(normalized_words(base["context"]))
            == stable_hash(normalized_words(order["context"])),
            base["raw_prompt"] != relation["raw_prompt"],
            base["raw_prompt"] != obj["raw_prompt"],
            base["raw_prompt"] != order["raw_prompt"],
        )
        if not all(checks):
            invariant_failures += 1
    if invariant_failures:
        failures.append("counterfactual_invariants")
    prior_objects = set()
    for path in PRIOR_OPEN_CASES:
        prior_objects.update(
            obj for row in iter_jsonl(path) for obj in row["objects"]
        )
    new_objects = {obj for row in all_rows for obj in row["objects"]}
    prior_overlap = len(prior_objects & new_objects)
    if prior_overlap:
        failures.append("prior_open_object_overlap")
    if failures:
        raise RuntimeError(
            f"Phase573 static audit failed: {failures}; invariants={invariant_failures}"
        )

    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase573_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_behavior_executed": False,
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    protocol = {
        "schema_version": "phase573_natural_transition_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "axes": list(AXES),
        "variants": list(VARIANTS),
        "structure_splits": list(STRUCTURE_SPLITS),
        "causal_splits": list(CAUSAL_SPLITS),
        "candidate_worlds_per_split": CANDIDATE_WORLDS_PER_SPLIT,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "noop_repeats": NOOP_REPEATS,
        "relation_screen_variants": ["base", "relation_swap"],
        "control_variants": ["object_swap", "order_swap"],
        "control_screen_cap_per_split_model": CONTROL_SCREEN_CAP_PER_SPLIT_MODEL,
        "final_worlds_per_axis_split_model": FINAL_WORLDS_PER_AXIS_SPLIT_MODEL,
        "behavior_gate": {
            "minimum_relation_qualified_worlds_each_structure_split": 192,
            "minimum_all_axis_qualified_worlds_each_structure_split": 128,
            "minimum_all_axis_qualified_worlds_heldout": 128,
            "both_noop_repeats_must_have_expected_semantic_target": True,
            "order_swap_output_must_equal_base_target": True,
        },
        "trace_gate": {
            "minimum_worlds_per_axis_split": 128,
            "discovery_confirmation_same_event_order_required": True,
            "heldout_recombination_required": True,
            "upstream_primary_axis_must_not_use_output_embedding_direction": True,
            "fact_reading_must_be_measured_at_later_receiver_positions": True,
        },
        "causal_mask_constraint": (
            "Facts precede the query, so a changed query cannot alter earlier fact-token "
            "states. Fact reading must be measured as source routing/messages into query "
            "terminal or answer-boundary receiver positions."
        ),
        "stage_order": [
            "relation_behavior_screen",
            "object_and_order_controls_on_relation_qualified_worlds",
            "heldout_recombination_behavior",
            "coordinate_free_natural_trace",
            "coarse_message_edge_only_if_trace_confirms",
        ],
        "head_channel_parameter_neuron_scan_allowed": False,
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "phase572_decision_sha256": sha256_file(PHASE572_DECISION),
        "sealed_split_read_for_analysis": False,
    }
    write_json(PROTOCOL_PATH, protocol)
    audit = {
        "schema_version": "phase573_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": True,
        "failures": [],
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "world_group_count": len(by_base),
        "counterfactual_invariant_failure_count": 0,
        "prior_open_object_overlap_count": prior_overlap,
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    write_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "open_cases": len(open_rows),
                "sealed_cases": len(sealed_rows),
                "world_groups": len(by_base),
                "prior_open_overlap": prior_overlap,
                "valid": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return protocol


if __name__ == "__main__":
    freeze()
