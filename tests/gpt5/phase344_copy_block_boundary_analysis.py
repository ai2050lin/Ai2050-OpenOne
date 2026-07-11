#!/usr/bin/env python3
"""Aggregate Phase344 copy-boundary gates and enforce stop rules."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase344_copy_block_boundary"
PHASE338 = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"
PHASE = "Phase344"
SCHEMA_VERSION = "20.0.0"
ROUND_DEFAULT = "copy_block_heldout_boundary"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("heldout", "private_heldout")
CONTROLS = ("wrong_depth_zero", "wrong_position_zero")
EXPLICIT = (
    "random_label_copy", "digit_copy", "arbitrary_symbol_relay",
    "cross_sentence_pointer", "multi_token_phrase_copy", "delayed_copy",
)
NEIGHBORS = ("key_value_read", "object_name_relay", "field_extraction")
NONCOPY = ("material_relation_binding", "singular_agreement", "direct_entailment", "token_transformation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def avg(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [
        float(row[field]) for row in rows
        if row["condition"] == condition and row.get(field) is not None
        and math.isfinite(float(row[field]))
    ]
    return mean(values) if values else 0.0


def rate(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [bool(row[field]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values) if values else 0.0


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase344_registered_protocol.json")
    registered = read_jsonl(root / "phase344_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    prior = {row["model"]: row for row in read_jsonl(PHASE338 / "phase338_model_gate_summary.jsonl")}
    task_rows = []
    model_rows = []
    completions = []
    total_phrase = total_rollout = 0
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase344_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase344_rollout_rows.jsonl")
        complete = read_json(model_root / "complete.json")
        completions.append(complete)
        total_phrase += len(phrase)
        total_rollout += len(rollout)
        passes = {}
        for task_id in protocol["selected_tasks"]:
            task_class = next(row["task_class"] for row in registered if row["model"] == model and row["mechanism_id"] == task_id)
            summary: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model, "task_id": task_id,
                "task_class": task_class, "frozen_block_id": complete["frozen_block_id"],
                "phase338_full_model_gate_pass": prior[model]["full_model_gate_pass"],
            }
            gates = []
            for split in SPLITS:
                p = [row for row in phrase if row["mechanism_id"] == task_id and row["split"] == split]
                r = [row for row in rollout if row["mechanism_id"] == task_id and row["split"] == split]
                baseline = rate(r, "baseline", "answer_head_semantic_correct")
                correct_behavior = rate(r, "correct_zero", "behavior_lost_vs_baseline")
                control_behavior = max(rate(r, condition, "behavior_lost_vs_baseline") for condition in CONTROLS)
                required = [row for row in p if row["condition"] in ("baseline", "correct_zero", *CONTROLS)]
                valid_rate = sum(row["score_valid"] for row in required) / len(required)
                correct_phrase = avg(p, "correct_zero", "phrase_margin_loss_vs_baseline")
                half_phrase = avg(p, "correct_half", "phrase_margin_loss_vs_baseline")
                permutation_phrase = avg(p, "correct_permutation", "phrase_margin_loss_vs_baseline")
                control_phrase = max(avg(p, condition, "phrase_margin_loss_vs_baseline") for condition in CONTROLS)
                gate = bool(
                    baseline >= thresholds["baseline_capability_rate_min"]
                    and valid_rate >= thresholds["required_phrase_valid_rate_min"]
                    and correct_behavior >= thresholds["correct_behavior_loss_rate_min"]
                    and control_behavior <= thresholds["wrong_control_behavior_loss_rate_max"]
                    and correct_phrase - control_phrase >= thresholds["phrase_control_superiority_min"]
                )
                gates.append(gate)
                summary.update({
                    f"{split}_case_count": len(r) // 4,
                    f"{split}_baseline_capability_rate": round(baseline, 7),
                    f"{split}_required_phrase_valid_rate": round(valid_rate, 7),
                    f"{split}_correct_behavior_loss_rate": round(correct_behavior, 7),
                    f"{split}_max_control_behavior_loss_rate": round(control_behavior, 7),
                    f"{split}_correct_phrase_loss": round(correct_phrase, 7),
                    f"{split}_half_phrase_loss": round(half_phrase, 7),
                    f"{split}_permutation_phrase_loss": round(permutation_phrase, 7),
                    f"{split}_max_control_phrase_loss": round(control_phrase, 7),
                    f"{split}_correct_control_phrase_superiority": round(correct_phrase - control_phrase, 7),
                    f"{split}_gate_pass": gate,
                })
            summary["full_task_gate_pass"] = all(gates)
            summary["single_unit_causal"] = False
            passes[task_id] = summary["full_task_gate_pass"]
            task_rows.append(summary)
        explicit_count = sum(passes[task] for task in EXPLICIT)
        neighbor_count = sum(passes[task] for task in NEIGHBORS)
        noncopy_count = sum(passes[task] for task in NONCOPY)
        lexical_gate = all(passes[task] for task in protocol["lexical_generalization_required_tasks"])
        prior_gate = bool(prior[model]["full_model_gate_pass"])
        copy_specific = bool(
            prior_gate
            and explicit_count >= thresholds["glm4_explicit_copy_task_pass_min"]
            and noncopy_count <= thresholds["glm4_noncopy_task_pass_max"]
            and lexical_gate
        )
        if not prior_gate:
            scope, claim = "phase338_candidate_not_qualified", "descriptive_only"
        elif copy_specific:
            scope, claim = "explicit_copy_boundary_candidate", "model_specific_copy_candidate"
        elif explicit_count and noncopy_count:
            scope, claim = "broad_copy_and_noncopy_sensitive", "candidate_rejected_as_copy_specific"
        elif explicit_count:
            scope = "general_copy_hypothesis_rejected_partial_task_effect"
            claim = "candidate_rejected_as_general_copy"
        else:
            scope, claim = "copy_effect_not_replicated", "candidate_rejected"
        model_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "frozen_block_id": complete["frozen_block_id"],
            "phase338_full_model_gate_pass": prior_gate,
            "explicit_copy_task_gate_count": explicit_count,
            "copy_neighbor_task_gate_count": neighbor_count,
            "noncopy_control_task_gate_count": noncopy_count,
            "lexical_generalization_gate_pass": lexical_gate,
            "copy_specificity_gate_pass": copy_specific,
            "natural_state_content_test_entry_gate_open": copy_specific,
            "scope_classification": scope, "claim_qualification": claim,
            "layer_shrink_executed": False, "single_unit_causal": False,
        })
    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase344:{row['model']}:{row['task_id']}",
            "model": row["model"], "family_id": row["task_class"],
            "mechanism_id": row["task_id"], "frozen_block_id": row["frozen_block_id"],
            "full_task_gate_pass": row["full_task_gate_pass"],
            "phase338_candidate_qualified": row["phase338_full_model_gate_pass"],
            "mapping_status": (
                "qualified_frozen_block_task_effect"
                if row["full_task_gate_pass"] and row["phase338_full_model_gate_pass"]
                else "descriptive_or_failed_task_effect"
            ),
            "single_unit_causal": False,
        }
        for row in task_rows
    ]
    content_models = [row["model"] for row in model_rows if row["natural_state_content_test_entry_gate_open"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered), "registered_task_count": len(protocol["selected_tasks"]),
            "phrase_row_count": total_phrase, "rollout_row_count": total_rollout,
            "invalid_phrase_row_count": sum(row["invalid_phrase_row_count"] for row in completions),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "execution_mode": "b1_left_cache0",
        },
        "results": {
            "full_task_gate_count": sum(row["full_task_gate_pass"] for row in task_rows),
            "copy_specificity_gate_models": content_models,
            "natural_state_content_test_entry_gate_open": bool(content_models),
            "layer_shrink_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9", "mechanism_census_coverage": "72/72",
            "deep_causal_mechanisms_audited": "1/72",
            "copy_operation_boundary_tasks": "13 tasks x 3 models",
            "behavior_mechanism_closure": "0/72", "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase344_copy_specificity",
            "claim": "A Phase338-qualified coarse block has a lexical-generalized copy-specific boundary.",
            "status": "model_specific_candidate" if content_models else "not_supported",
            "evidence_level": "L4_single_case_copy_boundary",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase344_glm4_multitoken_task_effect",
            "claim": "The GLM4 frozen block has a model-specific heldout effect on multi-token phrase copying.",
            "status": (
                "model_specific_task_effect"
                if any(
                    row["model"] == "glm4"
                    and row["task_id"] == "multi_token_phrase_copy"
                    and row["full_task_gate_pass"]
                    for row in task_rows
                )
                else "not_supported"
            ),
            "evidence_level": "L4_single_task_coarse_block_effect",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase344_copy_state_content", "claim": "The block state carries transferable copied content.",
            "status": "not_tested", "evidence_level": "natural_state_replacement_not_run",
        },
    ]
    write_jsonl(root / "phase344_task_gate_summary.jsonl", task_rows)
    write_jsonl(root / "phase344_model_scope_summary.jsonl", model_rows)
    write_jsonl(root / "phase344_copy_boundary_nodes.jsonl", nodes)
    write_jsonl(root / "phase344_claim_registry.jsonl", claims)
    write_json(root / "phase344_global_summary.json", summary)
    report = [
        "# Phase344 Copy-Block Heldout Boundary", "",
        f"- Registered cases: {len(registered)}", f"- Phrase rows: {total_phrase}",
        f"- Single-case rollout rows: {total_rollout}", "",
    ]
    for row in model_rows:
        report.append(
            f"- {row['model']}: explicit={row['explicit_copy_task_gate_count']}/6, "
            f"neighbor={row['copy_neighbor_task_gate_count']}/3, noncopy={row['noncopy_control_task_gate_count']}/4, "
            f"lexical={row['lexical_generalization_gate_pass']}, scope={row['scope_classification']}"
        )
    report.extend(["", f"Natural-state content entry: {bool(content_models)}", "No layer or neuron shrinking was executed."])
    (root / "phase344_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
