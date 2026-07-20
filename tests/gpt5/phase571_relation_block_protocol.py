#!/usr/bin/env python3
"""Freeze the staged Phase571 relation-block protocol and independent cases."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase569_relation_competition_protocol as p569  # noqa: E402
import phase570_answer_bridge_protocol as p570  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402
from phase569_role_position_utils import ROLE_GROUPS  # noqa: E402


PHASE = "Phase571"
MODELS = p569.MODELS
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
OPEN_POOLS = ("block_discovery", "block_confirmation", "block_causal")
POOL_SPECS = {
    "block_discovery": ("path_discovery", 10000),
    # Keep objects disjoint while retaining enough label-pair diversity for the
    # preregistered eight-pair confirmation denominator.
    "block_confirmation": ("path_discovery", 30000),
    "block_causal": ("phenotype_confirmation", 50000),
    "block_sealed": ("path_discovery", 70000),
}
CANDIDATE_WORLDS_PER_CELL = 128
MIXED_CELLS_PER_MODEL = 8
TRACE_SELECTION_PER_PHENOTYPE = 128
CAUSAL_SELECTION_PER_PHENOTYPE = 160
MINIMUM_CASES_PER_PHENOTYPE = 128
DEPTH_BANDS = 8
TRACE_ROLES = ROLE_GROUPS
CAUSAL_ROLE_ORDER = (
    "query_relation",
    "query_object",
    "target_fact_relation",
    "target_fact_value",
    "other_fact_relation",
    "other_fact_value",
    "answer_boundary",
)

OUT_DIR = ROOT / "tests/gpt5/result/phase571_relation_block"
OPEN_CASES_PATH = OUT_DIR / "phase571_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase571_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase571_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase571_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase571_static_audit.json"
PHASE569_SUMMARY = ROOT / "tests/gpt5/result/phase569_relation_competition/phase569_behavior_summary.json"
PHASE570_SUMMARY = ROOT / "tests/gpt5/result/phase570_answer_bridge_causal/phase570_causal_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def query_only_counterfactual(row: dict[str, Any]) -> dict[str, Any]:
    relation = row["other_relation"]
    question, relation_label = p569.render_question(
        int(row["surface_id"]), row["query_object"], relation
    )
    raw_prompt = f"{row['context']}\nQuery: {question}\nInstruction: {row['instruction']}"
    return {
        "raw_prompt": raw_prompt,
        "question": question,
        "query_relation": relation,
        "query_relation_label": relation_label,
        "target": row["other_relation_target"],
        "other_relation_target": row["target"],
    }


def materialize_case(
    model: str,
    tokenizer: Any,
    pool: str,
    cell: str,
    cell_rank: int,
    world_rank: int,
) -> dict[str, Any]:
    source_split, base_index = POOL_SPECS[pool]
    factors = p570.parse_cell(cell)
    source_world_index = base_index + cell_rank * 1000 + world_rank
    row = p569.controlled_case(
        source_split,
        source_world_index,
        factors["binding"],
        factors["query"],
        factors["relation"],
        factors["surface"],
        factors["order"],
    )
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    counterfactual = query_only_counterfactual(row)
    counterfactual_prompt = render_chat(tokenizer, model, counterfactual["raw_prompt"])
    candidate_ids = {
        value: [int(token) for token in tokenizer(value, add_special_tokens=False)["input_ids"]]
        for value in row["all_candidates"]
    }
    case_id = f"phase571_{model}_{pool}_mixedcell{cell_rank}_world{world_rank:03d}"
    return {
        **row,
        "schema_version": "phase571_relation_block_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "pool": pool,
        "intended_phenotype": "unassigned_mixed_cell",
        "source_factorial_cell": cell,
        "cell_rank": cell_rank,
        "world_rank": world_rank,
        "source_generation_split": source_split,
        "source_generation_world_index": source_world_index,
        "case_id": case_id,
        "prompt_token_count": len(tokenizer(prompt, add_special_tokens=True)["input_ids"]),
        "candidate_token_ids": candidate_ids,
        "query_only_counterfactual": {
            **counterfactual,
            "prompt_token_count": len(
                tokenizer(counterfactual_prompt, add_special_tokens=True)["input_ids"]
            ),
        },
        "sealed": pool == "block_sealed",
    }


def build_rows(
    selected_by_model: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    for model in MODELS:
        for pool in (*OPEN_POOLS, "block_sealed"):
            for cell_rank, cell_report in enumerate(selected_by_model[model]):
                for world_rank in range(CANDIDATE_WORLDS_PER_CELL):
                    row = materialize_case(
                        model,
                        tokenizers[model],
                        pool,
                        cell_report["cell"],
                        cell_rank,
                        world_rank,
                    )
                    (sealed_rows if row["sealed"] else open_rows).append(row)
    return open_rows, sealed_rows


def select_mixed_cells(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report["phenotype_cell_metrics"]
    discovery = metrics["phenotype_discovery"]
    confirmation = metrics["phenotype_confirmation"]
    candidates = []
    for cell in sorted(discovery):
        rates = {
            "discovery_correct": discovery[cell]["accuracy"],
            "discovery_confusion": discovery[cell]["relation_confusion_rate_all_rows"],
            "confirmation_correct": confirmation[cell]["accuracy"],
            "confirmation_confusion": confirmation[cell]["relation_confusion_rate_all_rows"],
        }
        floor = min(rates.values())
        if floor >= 0.15:
            candidates.append({"cell": cell, "historical_rate_floor": floor, "rates": rates})
    candidates.sort(key=lambda row: (-row["historical_rate_floor"], row["cell"]))
    selected = candidates[:MIXED_CELLS_PER_MODEL]
    if len(selected) != MIXED_CELLS_PER_MODEL:
        raise RuntimeError(f"Phase571 lacks eight mixed cells for {report['model']}")
    return selected


def freeze() -> dict[str, Any]:
    behavior = read_json(PHASE569_SUMMARY)
    selected_by_model = {
        report["model"]: select_mixed_cells(report) for report in behavior["model_reports"]
    }
    open_rows, sealed_rows = build_rows(selected_by_model)
    expected_per_pool_model = MIXED_CELLS_PER_MODEL * CANDIDATE_WORLDS_PER_CELL
    expected_open = expected_per_pool_model * len(OPEN_POOLS) * len(MODELS)
    expected_sealed = expected_per_pool_model * len(MODELS)
    failures: list[str] = []
    all_rows = open_rows + sealed_rows
    if len(open_rows) != expected_open:
        failures.append("open_case_count")
    if len(sealed_rows) != expected_sealed:
        failures.append("sealed_case_count")
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        failures.append("case_id_collision")
    if any(row["target"] == row["other_relation_target"] for row in all_rows):
        failures.append("target_other_collision")
    if any(
        len(ids) != 1 for row in all_rows for ids in row["candidate_token_ids"].values()
    ):
        failures.append("candidate_not_single_token")
    if any(
        len({tuple(ids) for ids in row["candidate_token_ids"].values()}) != 4
        for row in all_rows
    ):
        failures.append("candidate_token_collision")
    registered_pair_counts = {
        model: {
            pool: len({
                (row["target"], row["other_relation_target"])
                for row in all_rows
                if row["model"] == model and row["pool"] == pool
            })
            for pool in (*OPEN_POOLS, "block_sealed")
        }
        for model in MODELS
    }
    if any(
        registered_pair_counts[model][pool] < 8
        for model in MODELS
        for pool in (*OPEN_POOLS, "block_sealed")
    ):
        failures.append("registered_target_other_pair_diversity")
    pool_objects = {
        pool: {
            obj
            for row in all_rows
            if row["pool"] == pool
            for obj in row["objects"]
        }
        for pool in (*OPEN_POOLS, "block_sealed")
    }
    overlap_count = 0
    overlap_by_pool_pair = {}
    pools = list(pool_objects)
    for left_index, left in enumerate(pools):
        for right in pools[left_index + 1:]:
            pair_overlap = len(pool_objects[left] & pool_objects[right])
            overlap_by_pool_pair[f"{left}|{right}"] = pair_overlap
            overlap_count += pair_overlap
    if overlap_count:
        failures.append("pool_object_overlap")
    if any(not row["sealed"] for row in sealed_rows) or any(row["sealed"] for row in open_rows):
        failures.append("seal_identity")
    if failures:
        raise RuntimeError(
            f"Phase571 static audit failed: {failures}; pair_counts={registered_pair_counts}; "
            f"pool_overlap={overlap_by_pool_pair}"
        )

    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase571_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_behavior_executed": False,
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    protocol = {
        "schema_version": "phase571_frozen_protocol.v2",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "phenotypes": list(PHENOTYPES),
        "open_pools": list(OPEN_POOLS),
        "candidate_worlds_per_cell": CANDIDATE_WORLDS_PER_CELL,
        "mixed_cells_per_model": MIXED_CELLS_PER_MODEL,
        "candidate_cases_per_pool_model": expected_per_pool_model,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "selected_mixed_cells_by_model": selected_by_model,
        "phenotypes_are_assigned_only_after_natural_output_within_the_same_cells": True,
        "registered_target_other_pair_counts": registered_pair_counts,
        "trace_selection_per_phenotype_pool": TRACE_SELECTION_PER_PHENOTYPE,
        "causal_selection_per_phenotype": CAUSAL_SELECTION_PER_PHENOTYPE,
        "minimum_cases_per_phenotype_pool": MINIMUM_CASES_PER_PHENOTYPE,
        "fixed_execution_batch_size": 8,
        "causal_noop_repeats": 2,
        "depth_band_count": DEPTH_BANDS,
        "trace_components": ["attention_output", "mlp_output"],
        "trace_roles": list(TRACE_ROLES),
        "causal_role_priority": list(CAUSAL_ROLE_ORDER),
        "block_discovery_rule": {
            "all_contiguous_depth_band_intervals": True,
            "maximum_interval_width_in_bands": 4,
            "minimum_absolute_relative_gap_each_split": 0.10,
            "minimum_absolute_positive_rate_gap_each_split": 0.10,
            "confirmation_to_discovery_absolute_gap_ratio_min": 0.50,
            "same_nonzero_gap_sign_required": True,
            "selection_order": [
                "earliest_end_band",
                "shortest_interval",
                "earliest_start_band",
                "largest_minimum_relative_gap",
                "causal_role_priority",
            ],
        },
        "coarse_block_conditions": [
            "baseline",
            "signed_block_remove",
            "full_block_remove",
            "full_block_remove_restore",
            "wrong_depth_full_remove",
            "wrong_role_full_remove",
            "random_matched_replace",
        ],
        "coarse_block_gate": {
            "minimum_paired_cases_per_phenotype": 128,
            "minimum_correct_accuracy_damage": 0.10,
            "minimum_confusion_accuracy_repair": 0.10,
            "maximum_restore_rate_loss": 0.05,
            "minimum_behavior_specificity_advantage": 0.05,
            "signed_projection_is_diagnostic_only": True,
            "all_checks_required_before_donor_stage": True,
        },
        "stage_order": [
            "behavior_replication",
            "signed_write_trace",
            "continuous_block_confirmation",
            "coarse_block_delete_restore_controls",
            "conditional_donor_stage_only_if_coarse_gate_passes",
        ],
        "head_channel_parameter_neuron_scan_allowed": False,
        "phase569_behavior_summary_sha256": sha256_file(PHASE569_SUMMARY),
        "phase570_causal_summary_sha256": sha256_file(PHASE570_SUMMARY),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "sealed_split_read_for_analysis": False,
    }
    write_json(PROTOCOL_PATH, protocol)
    audit = {
        "schema_version": "phase571_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": True,
        "failures": [],
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "target_other_collision_count": 0,
        "candidate_non_single_token_count": 0,
        "candidate_token_collision_count": 0,
        "registered_target_other_pair_counts": registered_pair_counts,
        "pool_object_overlap_count": overlap_count,
        "pool_object_overlap_by_pair": overlap_by_pool_pair,
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model_execution_performed": False,
        "sealed_split_read_for_analysis": False,
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "open_cases": len(open_rows),
        "sealed_cases": len(sealed_rows),
        "per_pool_model": expected_per_pool_model,
        "selected_cells": selected_by_model,
        "valid": True,
    }, ensure_ascii=False, indent=2))
    return protocol


if __name__ == "__main__":
    freeze()
