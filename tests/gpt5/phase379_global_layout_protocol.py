#!/usr/bin/env python3
"""Freeze the decision-aligned cross-family layout denominator for Phase379."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE = "Phase379"
SCHEMA = "52.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "relation_binding",
    "entity_recency",
    "number_agreement",
    "target_vs_wrong",
)
P369 = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
P330 = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_target_step(
    tokenizer: Any, token_ids: list[int], aliases: list[str]
) -> int | None:
    lowered = [alias.strip().casefold() for alias in aliases if alias.strip()]
    for index in range(len(token_ids)):
        text = tokenizer.decode(token_ids[: index + 1]).casefold()
        if any(alias in text for alias in lowered):
            return index
    return None


def phase330_decision_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for behavior_path in sorted(P330.glob("survey/*/*/behavior.parquet")):
        model = behavior_path.parent.parent.name
        family = behavior_path.parent.name
        readout_path = behavior_path.with_name("readout.parquet")
        behavior = {
            row["case_id"]: row
            for row in pq.read_table(behavior_path).to_pylist()
        }
        for readout in pq.read_table(readout_path).to_pylist():
            base = behavior[readout["case_id"]]
            strict = bool(
                base["behavior_success"]
                and base["candidate_winner_is_target"]
                and int(readout["target_full_vocabulary_rank"]) == 1
            )
            rows.append(
                {
                    "model": model,
                    "family_id": family,
                    "case_id": readout["case_id"],
                    "strict_current_decision_aligned": strict,
                }
            )
    by_model = Counter(row["model"] for row in rows)
    strict_by_model = Counter(
        row["model"] for row in rows if row["strict_current_decision_aligned"]
    )
    case_models: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["strict_current_decision_aligned"]:
            case_models[row["case_id"]].add(row["model"])
    common = [case_id for case_id, models in case_models.items() if models == set(MODELS)]
    return {
        "schema_version": SCHEMA,
        "phase_id": "Phase379-Phase330DecisionAudit",
        "created_at": now(),
        "objective": "test_whether_phase330_prompt_end_events_form_a_strict_common_three_model_decision_denominator",
        "denominator": {
            "phase330_model_case_count": len(rows),
            "model_case_counts": dict(by_model),
        },
        "strict_current_decision": {
            "model_counts": dict(strict_by_model),
            "common_three_model_case_count": len(common),
            "common_three_model_case_ids": common,
        },
        "result": {
            "phase330_crossmodel_global_layout_reusable": False,
            "reason": "only_one_case_is_behavior_correct_and_full_vocabulary_rank_one_at_the_same_prompt_end_decision_across_all_models",
            "engineering_coverage_remains_valid": True,
            "scientific_crossmodel_layout_claims_remain_invalid": True,
        },
    }


def load_execution_cases() -> dict[str, dict[str, Any]]:
    paths = (
        P369 / "raw_topology_preregister/private/phase369_execution_cases.jsonl",
        P369
        / "raw_topology_preregister_number_agreement_expansion/private/phase369_number_agreement_expansion_execution_cases.jsonl",
    )
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            result[row["blind_case_id"]] = row
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = phase330_decision_audit()
    write_json(OUT / "phase379_phase330_decision_audit.json", audit)

    behavior_path = (
        P369
        / "behavior_qualification_final_v2/private/phase369_qualified_behavior_rows.jsonl"
    )
    behavior_rows = read_jsonl(behavior_path)
    execution = load_execution_cases()
    tokenizers: dict[str, Any] = {}
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizers[model] = AutoTokenizer.from_pretrained(
                str(spec.local_dir),
                trust_remote_code=spec.trust_remote_code,
                local_files_only=True,
                use_fast=False,
            )
        for behavior in behavior_rows:
            case_id = behavior["blind_case_id"]
            base = execution.get(case_id)
            if base is None:
                raise RuntimeError(f"Missing execution case for {case_id}")
            model = behavior["model"]
            step = first_target_step(
                tokenizers[model],
                behavior["generated_token_ids"],
                behavior["target_aliases"],
            )
            if step is None:
                raise RuntimeError(f"Missing target decision step for {model}/{case_id}")
            split = behavior["phase369_split"]
            if split not in {"fresh_discovery", "fresh_calibration"}:
                raise RuntimeError(f"Unexpected open split: {split}")
            common = {
                "schema_version": SCHEMA,
                "phase_id": PHASE,
                "created_at": now(),
                "blind_case_id": case_id,
                "anonymous_model_id": base["anonymous_model_id"],
                "anonymous_parallel_group_id": base["anonymous_parallel_group_id"],
                "anonymous_group_id": base["anonymous_group_id"],
                "anonymous_condition_slot": base["anonymous_condition_slot"],
                "phase379_split": split,
                "prompt": base["prompt"],
                "raw_prompt": base["raw_prompt"],
                "source_fragment": base["source_fragment"],
                "query_fragment": base["query_fragment"],
                "tokenization_add_special_tokens": base[
                    "tokenization_add_special_tokens"
                ],
                "prompt_token_count": base["prompt_token_count"],
                "interface": base["interface"],
                "answer_phase": base["answer_phase"],
                "generated_token_ids": behavior["generated_token_ids"],
                "target_decision_step": step,
            }
            execution_rows.append(
                {
                    **common,
                    "private_execution_model": model,
                    "family_id": behavior["family_id"],
                    "mechanism_id": behavior["mechanism_id"],
                    "semantic_group_id": behavior["semantic_group_id"],
                    "contrast_condition": behavior["contrast_condition"],
                    "operation_demanded": base["operation_demanded"],
                    "target": behavior["target"],
                    "target_aliases": behavior["target_aliases"],
                    "distractors": behavior["distractors"],
                    "strict_behavior_correct": behavior["strict_behavior_correct"],
                    "semantic_labels_available_to_trace": False,
                    "target_specific_competition_available_to_trace": False,
                }
            )
            blind_rows.append(
                {
                    **common,
                    "generated_token_ids": None,
                    "target_decision_step": None,
                    "decision_context_available_to_executor_only": True,
                    "semantic_label_used_for_discovery": False,
                    "target_or_distractor_exported": False,
                }
            )
            label_rows.append(
                {
                    "blind_case_id": case_id,
                    "model": model,
                    "family_id": behavior["family_id"],
                    "mechanism_id": behavior["mechanism_id"],
                    "semantic_group_id": behavior["semantic_group_id"],
                    "contrast_condition": behavior["contrast_condition"],
                    "phase379_split": split,
                    "target": behavior["target"],
                    "target_aliases": behavior["target_aliases"],
                    "distractors": behavior["distractors"],
                }
            )
    finally:
        tokenizers.clear()

    execution_rows.sort(
        key=lambda row: (
            MODELS.index(row["private_execution_model"]),
            row["phase379_split"],
            row["anonymous_parallel_group_id"],
            row["contrast_condition"],
        )
    )
    blind_rows.sort(key=lambda row: row["blind_case_id"])
    label_rows.sort(key=lambda row: row["blind_case_id"])

    split_counts = Counter(row["phase379_split"] for row in execution_rows)
    mechanism_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_counts = Counter()
    for row in execution_rows:
        mechanism_groups[(row["phase379_split"], row["mechanism_id"])].add(
            row["anonymous_parallel_group_id"]
        )
        group_counts[(row["private_execution_model"], row["anonymous_group_id"])] += 1
    if len(execution_rows) != 516:
        raise RuntimeError(f"Expected 516 cases, got {len(execution_rows)}")
    if set(group_counts.values()) != {4}:
        raise RuntimeError("Every model group must retain all four conditions")
    if set(row["mechanism_id"] for row in execution_rows) != set(MECHANISMS):
        raise RuntimeError("Phase379 mechanism denominator changed")

    private_path = OUT / "private/phase379_execution_cases.jsonl"
    blind_path = OUT / "phase379_blind_case_registry.jsonl"
    label_path = OUT / "private/phase379_label_key.jsonl"
    write_jsonl(private_path, execution_rows)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(label_path, label_rows)

    summary = {
        "schema_version": SCHEMA,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "freeze_behavior_qualified_decision_aligned_cross_family_layout_denominator",
        "denominator": {
            "model_count": 3,
            "representative_family_count": 4,
            "mechanism_count": 4,
            "parallel_group_count": len(
                {row["anonymous_parallel_group_id"] for row in execution_rows}
            ),
            "model_group_count": len(group_counts),
            "case_count": len(execution_rows),
            "fresh_discovery_case_count": split_counts["fresh_discovery"],
            "fresh_calibration_case_count": split_counts["fresh_calibration"],
            "groups_by_split_mechanism": {
                f"{split}:{mechanism}": len(groups)
                for (split, mechanism), groups in sorted(mechanism_groups.items())
            },
        },
        "quality": {
            "all_cases_strict_behavior_correct": all(
                row["strict_behavior_correct"] for row in execution_rows
            ),
            "all_target_decisions_located": all(
                row["target_decision_step"] is not None for row in execution_rows
            ),
            "every_model_group_has_four_conditions": True,
            "old_phase330_common_strict_decision_cases": audit[
                "strict_current_decision"
            ]["common_three_model_case_count"],
            "phase330_reused_as_scientific_layout": False,
            "physical_holdout_opened": False,
        },
        "input_hashes": {
            "qualified_behavior_rows": sha256(behavior_path),
            "execution_cases": sha256(private_path),
            "blind_registry": sha256(blind_path),
            "label_key": sha256(label_path),
        },
        "authorization": {
            "run_fresh_discovery_trace_sequentially": True,
            "open_calibration_before_discovery_mapping_freeze": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
            "run_single_neuron_scan": False,
        },
    }
    protocol = {
        "schema_version": SCHEMA,
        "phase_id": "Phase379-Protocol",
        "created_at": now(),
        "scientific_target": "global_reuse_differentiation_branch_merge_layout_before_local_closure",
        "representative_functions": {
            "relation_binding": "relational_content_operation",
            "entity_recency": "direct_content_retrieval",
            "number_agreement": "grammar_constraint",
            "target_vs_wrong": "readout_competition",
        },
        "semantic_time": "immediately_before_the_token_that_completes_the_observed_target_alias",
        "trace_contract": {
            "all_layers": True,
            "component_boundaries": [
                "layer_input",
                "attention_output",
                "mlp_output",
                "layer_output",
            ],
            "position_roles": ["source", "query", "current"],
            "full_vocabulary_logits": True,
            "exact_vectors_retained": True,
            "top_k_selection": False,
            "fixed_relative_depth_bins_for_mapping": 5,
        },
        "blind_discovery": {
            "all_six_unordered_condition_pairs_per_group": True,
            "semantic_labels_available": False,
            "target_tokens_available": False,
            "candidate_selection_allowed": False,
            "weighted_single_score_used_for_claims": False,
        },
        "post_freeze_axes": [
            "content_change_same_operation",
            "operation_change_same_content",
            "joint_content_operation_change",
        ],
        "calibration_gate": {
            "minimum_profile_cosine": 0.60,
            "minimum_heterogeneous_models": 2,
            "glm4_required_for_heterogeneous_replication": True,
            "threshold_retuning_after_calibration": False,
        },
        "claim_boundary": {
            "descriptive_reuse_is_causal_reuse": False,
            "activation_overlap_is_mechanism_identity": False,
            "soft_functional_territory_is_contiguous_neuron_region": False,
            "terminal_carrier_is_upstream_rule": False,
            "language_encoding_mechanism_closed": False,
        },
        "execution_order": list(MODELS),
        "stage_order": [
            "discovery_exact_trace",
            "blind_all_pair_extraction",
            "discovery_hash_freeze",
            "semantic_layout_mapping",
            "calibration_exact_trace",
            "calibration_without_retuning",
            "causal_scan_only_if_crossmodel_profile_replication_passes",
        ],
        "stop_rules": [
            "do_not_reuse_phase330_prompt_end_events_as_crossmodel_layout",
            "do_not_open_physical_cases_to_repair_layout_profiles",
            "do_not_select_single_neurons_from_activation_magnitude",
            "do_not_call_profile_overlap_a_shared_causal_mechanism",
        ],
    }
    write_json(OUT / "phase379_case_bank_summary.json", summary)
    write_json(OUT / "phase379_protocol.json", protocol)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
