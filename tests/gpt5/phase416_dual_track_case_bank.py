#!/usr/bin/env python3
"""Freeze the Phase416 formal-world collector and physical-atlas denominator.

Phase414 incorrectly coupled three independent permissions: collecting raw
physical traces, assigning functional labels with an observer, and claiming
natural-language external validity.  Phase416 keeps those tracks separate.

The formal cases are inherited from the already frozen Phase347 cross-model
case bank.  This file only selects a deterministic, cross-model aligned 55-case
instrument denominator; it does not relabel model behavior as correct.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase416_formal_world_physical_atlas"
SOURCE_CASES = (
    ROOT
    / "tests/gpt5/result/phase347_three_core_natural_trace"
    / "three_core_natural_physical_trace"
    / "phase347_registered_cases.jsonl"
)
PHASE_ID = "Phase416-FormalWorldPhysicalAtlas"
SCHEMA_VERSION = "89.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")

# 55 cases per model.  The uneven five/six allocation keeps all ten frozen
# mechanisms while giving every major task family at least one six-case cell.
MECHANISM_QUOTAS = {
    "context_relation_binding": 6,
    "parameter_knowledge_retrieval": 6,
    "explicit_copy_control": 5,
    "two_hop_entailment": 6,
    "direct_fact_control": 5,
    "sentence_past_tense": 6,
    "no_morphology_control": 5,
    "answer_only_protocol": 6,
    "contiguous_multi_token_answer": 5,
    "simple_no_source_answer": 5,
}
SIX_CASE_PLAN = ((0, "format_a"), (0, "format_b"), (12, "format_a"),
                 (12, "format_b"), (17, "format_a"), (17, "format_b"))
FIVE_CASE_PLAN = ((0, "format_a"), (0, "format_b"), (17, "format_a"),
                  (17, "format_b"), (23, "format_c"))
ANSWER_ONLY_EVALUATION_PLAN = ((6, "format_a"), (6, "format_b"), (17, "format_a"),
                               (17, "format_b"), (23, "format_a"), (23, "format_b"))
EXCLUDED_PUBLIC_CALIBRATION_CELL = ("answer_only_protocol", 0, "format_a")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_semantic_cases(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    qwen_rows = [row for row in rows if row["model"] == "qwen3"]
    by_mechanism: dict[str, dict[tuple[int, str], dict[str, Any]]] = defaultdict(dict)
    for row in qwen_rows:
        by_mechanism[row["mechanism_id"]][(int(row["item_index"]), row["template_id"])] = row

    selected: dict[str, list[str]] = {}
    for mechanism, quota in MECHANISM_QUOTAS.items():
        if mechanism == "answer_only_protocol":
            plan = ANSWER_ONLY_EVALUATION_PLAN
        else:
            plan = SIX_CASE_PLAN if quota == 6 else FIVE_CASE_PLAN
        available = by_mechanism[mechanism]
        missing = [cell for cell in plan if cell not in available]
        if missing:
            raise RuntimeError(f"Missing frozen Phase347 cells for {mechanism}: {missing}")
        selected[mechanism] = [available[cell]["semantic_case_id"] for cell in plan]
    return selected


def freeze_case_bank() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_rows = read_jsonl(SOURCE_CASES)
    selected = select_semantic_cases(source_rows)
    selected_ids = {value for values in selected.values() for value in values}
    source_index = {(row["model"], row["semantic_case_id"]): row for row in source_rows}
    created_at = now()
    cases: list[dict[str, Any]] = []
    for model in MODELS:
        for mechanism, semantic_ids in selected.items():
            for semantic_id in semantic_ids:
                source = source_index.get((model, semantic_id))
                if source is None:
                    raise RuntimeError(f"Missing cross-model case: {model}/{semantic_id}")
                case_id = f"phase416_{model}_{semantic_id.removeprefix('phase345_').removeprefix('phase346_')}"
                cases.append(
                    {
                        **source,
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "created_at": created_at,
                        "case_id": case_id,
                        "phase416_source_case_id": source["case_id"],
                        "track": "formal_world",
                        "formal_semantics_executable": True,
                        "human_naturalness_review_required": False,
                        "collector_qualification_case": True,
                        "raw_physical_collection_eligible_after_instrument_gate": True,
                        "functional_observer_label_authorized": False,
                        "causal_intervention_authorized": False,
                        "single_neuron_scan_authorized": False,
                        "semantic_event": {
                            "event_id": f"formal_answer::{source['semantic_case_id']}",
                            "target_text": source["target"],
                            "target_aliases": source["target_aliases"],
                            "distractor_texts": source["distractors"],
                            "response_contract": "exact_answer_field_then_optional_eos",
                        },
                        "prompt_sha256": sha256_text(source["prompt"]),
                    }
                )

    cases.sort(key=lambda row: (MODELS.index(row["model"]), row["mechanism_id"], row["semantic_case_id"]))
    per_model = Counter(row["model"] for row in cases)
    per_mechanism = Counter((row["model"], row["mechanism_id"]) for row in cases)
    per_family = Counter((row["model"], row["family_id"]) for row in cases)
    semantic_alignment = {
        semantic_id: {row["model"] for row in cases if row["semantic_case_id"] == semantic_id}
        for semantic_id in selected_ids
    }
    valid = bool(
        len(cases) == 165
        and all(per_model[model] == 55 for model in MODELS)
        and all(models == set(MODELS) for models in semantic_alignment.values())
        and len({row["case_id"] for row in cases}) == len(cases)
        and all(
            per_mechanism[(model, mechanism)] == quota
            for model in MODELS
            for mechanism, quota in MECHANISM_QUOTAS.items()
        )
    )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "qualify_real_collectors_then_map_observer_independent_formal_world_physical_distributions",
        "source_case_bank": str(SOURCE_CASES.relative_to(ROOT)),
        "model_order": list(MODELS),
        "case_count": len(cases),
        "case_count_per_model": 55,
        "mechanism_quotas_per_model": MECHANISM_QUOTAS,
        "track_contract": {
            "formal_world": {
                "semantics": "executable_frozen_target_and_distractor_contract",
                "external_human_review_required_for_raw_collection": False,
                "observer_required_for_raw_collection": False,
                "collector_equivalence_required": True,
                "functional_label_requires_qualified_observer": True,
                "claim_scope": "formal_world_only",
            },
            "natural_language_externalization": {
                "external_human_review_required": True,
                "completed_reviewer_count": 0,
                "authorized": False,
            },
        },
        "collector_gates": {
            "direct_vs_hook_terminal_logit_max_abs": 0.02,
            "direct_vs_hook_terminal_js": 1e-6,
            "layer_output_max_abs": 0.02,
            "component_ledger_relative_error": 0.01,
            "chunked_cache_terminal_logit_max_abs": 0.125,
            "chunked_cache_terminal_js": 1e-6,
            "chunked_cache_top1_exact_required": True,
            "chunked_cache_relative_error": 0.01,
            "checkpoint_replay_terminal_logit_max_abs": 0.02,
            "checkpoint_replay_terminal_js": 1e-6,
            "greedy_generation_token_exact_required": True,
            "greedy_generation_score_max_abs": 0.03,
            "required_case_pass_count": 165,
        },
        "public_calibration": {
            "excluded_from_evaluation_denominator": True,
            "excluded_cell": {
                "mechanism_id": EXCLUDED_PUBLIC_CALIBRATION_CELL[0],
                "item_index": EXCLUDED_PUBLIC_CALIBRATION_CELL[1],
                "template_id": EXCLUDED_PUBLIC_CALIBRATION_CELL[2],
            },
            "observed_chunked_cache_terminal_logit_max_abs": 0.07421875,
            "observed_chunked_cache_relative_error": 0.005421875510364771,
            "observed_terminal_token_exact": True,
            "frozen_fp16_logit_tolerance_after_calibration": 0.125,
            "evaluation_cases_are_disjoint_from_calibration_cell": True,
        },
        "behavior_gate": {
            "minimum_family_case_count": 5,
            "minimum_target_event_match_rate": 0.75,
            "scope": "model_by_formal_family",
            "failure_effect": "withhold_functional_family_label_but_keep_instrument_rows",
        },
        "physical_contract": {
            "all_layers": True,
            "all_prompt_positions_reduced": True,
            "position_roles": ["source", "query", "answer_start"],
            "core_components": [
                "layer_input",
                "attention_output",
                "mlp_output",
                "residual_increment",
                "layer_output",
            ],
            "subcomponents": [
                "q_projection",
                "k_projection",
                "v_projection",
                "mlp_gate",
                "mlp_up",
                "mlp_product",
            ],
            "lossless_anchor_vectors_per_model": 4,
            "observer_index": None,
            "causal": False,
        },
        "stop_rules": [
            "collector_case_failure_blocks_all_new_physical_collection_for_that_model",
            "behavior_family_failure_withholds_family_functional_interpretation",
            "high_event_panel_outside_mass_blocks_panel_summary",
            "no_observer_label_before_frozen_holdout_qualification",
            "no_causal_or_neuron_stage_from_descriptive_physical_results",
        ],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase416-DualTrackCaseBankQualification",
        "created_at": created_at,
        "valid": valid,
        "case_count": len(cases),
        "model_case_count": dict(per_model),
        "model_family_case_count": {
            f"{model}:{family}": count for (model, family), count in sorted(per_family.items())
        },
        "semantic_case_count": len(selected_ids),
        "cross_model_semantic_alignment_count": sum(models == set(MODELS) for models in semantic_alignment.values()),
        "mechanism_count": len(MECHANISM_QUOTAS),
        "external_review_blocks_formal_collector": False,
        "qualified_observer_blocks_raw_physical_collection": False,
        "external_review_blocks_natural_language_generalization": True,
        "qualified_observer_required_for_functional_label": True,
        "model_execution_authorized": valid,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
        "claim_boundary": "case_bank_and_permission_split_only_no_model_result",
    }
    return cases, protocol, summary


def main() -> None:
    cases, protocol, summary = freeze_case_bank()
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase416_registered_cases.jsonl", cases)
    write_json(OUT / "phase416_dual_track_protocol.json", protocol)
    write_json(OUT / "phase416_case_bank_qualification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
