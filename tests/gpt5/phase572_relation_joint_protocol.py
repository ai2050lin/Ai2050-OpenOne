#!/usr/bin/env python3
"""Freeze fresh Qwen3 cases and gates for Phase572 joint-role interventions."""

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
import phase570_answer_bridge_protocol as p570  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase572"
MODEL = "qwen3"
SOURCE_SPLIT = "path_discovery"
SOURCE_BASE = 90000
WORLDS_PER_CELL = 128
CELL_COUNT = 8
FINAL_PAIRS = 128
ENTRY_LAYER = 23
EXIT_LAYER = 26
CONDITIONS = (
    "baseline",
    "self_qfa_entry_restore",
    "matched_answer_entry",
    "matched_query_entry",
    "matched_fact_entry",
    "matched_query_answer_entry",
    "matched_fact_answer_entry",
    "matched_query_fact_entry",
    "matched_query_fact_answer_entry",
    "wrong_target_query_fact_answer_entry",
    "random_query_fact_answer_entry",
)

OUT_DIR = ROOT / "tests/gpt5/result/phase572_relation_joint"
CASES_PATH = OUT_DIR / "phase572_open_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase572_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase572_static_audit.json"
PHASE571_PROTOCOL = (
    ROOT / "tests/gpt5/result/phase571_relation_block/phase571_frozen_protocol.json"
)
PHASE571_PERMUTATION = (
    ROOT
    / "tests/gpt5/result/phase571_relation_block/phase571_max_block_permutation_audit.json"
)
PHASE571_CAUSAL = (
    ROOT
    / "tests/gpt5/result/phase571_relation_block/phase571_coarse_block_causal_analysis.json"
)
PHASE571_DONOR = (
    ROOT
    / "tests/gpt5/result/phase571_relation_block/phase571_relation_donor_analysis.json"
)
PHASE571_OPEN_CASES = (
    ROOT / "tests/gpt5/result/phase571_relation_block/phase571_open_cases.jsonl.gz"
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


def materialize_case(
    tokenizer: Any,
    cell: str,
    cell_rank: int,
    world_rank: int,
) -> dict[str, Any]:
    factors = p570.parse_cell(cell)
    source_world_index = SOURCE_BASE + cell_rank * 1000 + world_rank
    row = p569.controlled_case(
        SOURCE_SPLIT,
        source_world_index,
        factors["binding"],
        factors["query"],
        factors["relation"],
        factors["surface"],
        factors["order"],
    )
    prompt = render_chat(tokenizer, MODEL, row["raw_prompt"])
    candidate_ids = {
        value: [int(token) for token in tokenizer(value, add_special_tokens=False)["input_ids"]]
        for value in row["all_candidates"]
    }
    return {
        **row,
        "schema_version": "phase572_relation_joint_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": MODEL,
        "pool": "joint_fresh",
        "intended_phenotype": "unassigned_mixed_cell",
        "source_factorial_cell": cell,
        "cell_rank": cell_rank,
        "world_rank": world_rank,
        "source_generation_split": SOURCE_SPLIT,
        "source_generation_world_index": source_world_index,
        "case_id": f"phase572_qwen3_joint_mixedcell{cell_rank}_world{world_rank:03d}",
        "prompt_token_count": len(tokenizer(prompt, add_special_tokens=True)["input_ids"]),
        "candidate_token_ids": candidate_ids,
        "sealed": False,
    }


def freeze() -> dict[str, Any]:
    phase571 = read_json(PHASE571_PROTOCOL)
    permutation = read_json(PHASE571_PERMUTATION)
    causal = read_json(PHASE571_CAUSAL)
    donor = read_json(PHASE571_DONOR)
    if MODEL not in permutation["passed_models"]:
        raise RuntimeError("Phase572 requires the Phase571 Qwen3 permutation gate")
    if MODEL not in causal["passed_models"]:
        raise RuntimeError("Phase572 requires the Phase571 Qwen3 coarse causal gate")
    if donor["relation_selection_donor_gate_pass"]:
        raise RuntimeError("Phase572 is only valid after the single-role donor gate fails")
    cells = [
        row["cell"] for row in phase571["selected_mixed_cells_by_model"][MODEL]
    ]
    if len(cells) != CELL_COUNT:
        raise RuntimeError("Phase572 mixed-cell denominator drift")
    tokenizer = tokenizer_for(MODEL)
    rows = [
        materialize_case(tokenizer, cell, cell_rank, world_rank)
        for cell_rank, cell in enumerate(cells)
        for world_rank in range(WORLDS_PER_CELL)
    ]
    failures = []
    if len(rows) != CELL_COUNT * WORLDS_PER_CELL:
        failures.append("case_count")
    if len({row["case_id"] for row in rows}) != len(rows):
        failures.append("case_id_collision")
    if any(row["target"] == row["other_relation_target"] for row in rows):
        failures.append("target_other_collision")
    if any(
        len(ids) != 1 or len({tuple(v) for v in row["candidate_token_ids"].values()}) != 4
        for row in rows
        for ids in row["candidate_token_ids"].values()
    ):
        failures.append("candidate_token_identity")
    pair_count = len({(row["target"], row["other_relation_target"]) for row in rows})
    if pair_count < 8:
        failures.append("target_other_pair_diversity")
    old_objects = {
        obj for row in iter_jsonl(PHASE571_OPEN_CASES) if row["model"] == MODEL
        for obj in row["objects"]
    }
    new_objects = {obj for row in rows for obj in row["objects"]}
    open_overlap = len(old_objects & new_objects)
    if open_overlap:
        failures.append("phase571_open_object_overlap")
    if failures:
        raise RuntimeError(f"Phase572 static audit failed: {failures}")
    write_jsonl(CASES_PATH, rows)
    protocol = {
        "schema_version": "phase572_relation_joint_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": MODEL,
        "source_split": SOURCE_SPLIT,
        "source_world_base": SOURCE_BASE,
        "mixed_cells": cells,
        "mixed_cell_count": CELL_COUNT,
        "candidate_worlds_per_cell": WORLDS_PER_CELL,
        "candidate_case_count": len(rows),
        "minimum_candidate_pairs": 160,
        "final_pair_count": FINAL_PAIRS,
        "fixed_batch_size": 8,
        "noop_repeats": 2,
        "entry_layer": ENTRY_LAYER,
        "exit_layer": EXIT_LAYER,
        "roles": ["query_relation", "target_fact_value", "answer_boundary"],
        "conditions": list(CONDITIONS),
        "joint_gate": {
            "minimum_self_restore_semantic_match": 0.95,
            "minimum_joint_confusion_repair": 0.10,
            "minimum_joint_correct_preservation": 0.90,
            "minimum_joint_specificity_over_each_control": 0.10,
            "minimum_joint_gain_over_best_single_role": 0.05,
            "minimum_joint_gain_over_best_two_role_subset": 0.05,
            "minimum_two_roles_with_positive_leave_one_out_contribution": 2,
            "minimum_leave_one_out_contribution": 0.05,
        },
        "classification_rule": {
            "joint_gate_pass": "candidate_distributed_relation_state",
            "only_answer_containing_sets_repair": "late_answer_content_transport",
            "controls_equal_or_exceed_joint": "general_state_replacement",
            "no_joint_repair": "late_joint_role_route_closed",
        },
        "model_scope_reason": (
            "Qwen3 alone passed the Phase571 permutation and coarse causal gates; GLM4 "
            "had no confirmed block and DS7B failed confusion repair specificity."
        ),
        "fresh_worlds_required": True,
        "head_channel_parameter_neuron_scan_allowed": False,
        "sealed_split_defined": False,
        "phase571_protocol_sha256": sha256_file(PHASE571_PROTOCOL),
        "phase571_permutation_sha256": sha256_file(PHASE571_PERMUTATION),
        "phase571_causal_sha256": sha256_file(PHASE571_CAUSAL),
        "phase571_donor_sha256": sha256_file(PHASE571_DONOR),
    }
    write_json(PROTOCOL_PATH, protocol)
    audit = {
        "schema_version": "phase572_relation_joint_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": True,
        "failures": [],
        "case_count": len(rows),
        "target_other_pair_count": pair_count,
        "phase571_open_object_overlap_count": open_overlap,
        "cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    write_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "cells": cells,
                "target_other_pair_count": pair_count,
                "phase571_open_object_overlap_count": open_overlap,
                "conditions": list(CONDITIONS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return protocol


if __name__ == "__main__":
    freeze()
