#!/usr/bin/env python3
"""Aggregate Phase341 qualified-task causal boundary evidence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase341_fresh_causal_boundary"
PHASE338 = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"
PHASE = "Phase341"
SCHEMA_VERSION = "17.0.0"
ROUND_DEFAULT = "qualified_six_task_causal_boundary"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("heldout", "private_heldout")
CONTROLS = ("wrong_depth_zero", "wrong_position_zero")


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
    protocol = read_json(root / "phase341_registered_protocol.json")
    registered = read_jsonl(root / "phase341_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    prior = {
        row["model"]: row for row in read_jsonl(PHASE338 / "phase338_model_gate_summary.jsonl")
    }
    task_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    completions = []
    total_phrase = total_rollout = 0
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase341_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase341_rollout_rows.jsonl")
        complete = read_json(model_root / "complete.json")
        completions.append(complete)
        total_phrase += len(phrase)
        total_rollout += len(rollout)
        passes: dict[str, bool] = {}
        for task_id in protocol["selected_tasks"]:
            task_class = next(row["task_class"] for row in registered if row["model"] == model and row["mechanism_id"] == task_id)
            summary: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "model": model, "task_id": task_id,
                "task_class": task_class, "frozen_block_id": complete["frozen_block_id"],
                "phase338_full_model_gate_pass": prior[model]["full_model_gate_pass"],
            }
            split_passes = []
            for split in SPLITS:
                p = [row for row in phrase if row["mechanism_id"] == task_id and row["split"] == split]
                r = [row for row in rollout if row["mechanism_id"] == task_id and row["split"] == split]
                baseline = rate(r, "baseline", "answer_head_semantic_correct")
                correct_behavior = rate(r, "correct_zero", "behavior_lost_vs_baseline")
                control_behavior = max(rate(r, condition, "behavior_lost_vs_baseline") for condition in CONTROLS)
                valid_rows = [row for row in p if row["condition"] in ("baseline", "correct_zero", *CONTROLS)]
                valid_rate = sum(row["score_valid"] for row in valid_rows) / len(valid_rows)
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
                split_passes.append(gate)
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
            summary["full_task_gate_pass"] = all(split_passes)
            summary["single_unit_causal"] = False
            passes[task_id] = summary["full_task_gate_pass"]
            task_rows.append(summary)

        relation_count = sum(passes[task] for task in ("material_relation_binding", "part_relation_binding", "location_relation_binding"))
        source_count = int(passes["identity_copy"])
        cross_count = sum(passes[task] for task in ("direct_entailment", "answer_only_protocol"))
        prior_gate = bool(prior[model]["full_model_gate_pass"])
        if not prior_gate:
            scope, shrink, claim = "phase338_candidate_not_qualified", False, "descriptive_only"
        elif not passes["material_relation_binding"]:
            if passes["identity_copy"] and passes["answer_only_protocol"]:
                scope = "explicit_source_token_copy_effect_material_candidate_rejected"
                shrink = False
                claim = "model_specific_reclassified_candidate"
            else:
                scope, shrink, claim = "material_effect_not_replicated", False, "candidate_rejected"
        elif cross_count:
            scope, shrink, claim = "broad_cross_family_sensitive", False, "broad_function_candidate"
        elif source_count:
            scope, shrink, claim = "general_source_operation_sensitive", False, "broad_function_candidate"
        elif relation_count >= 2:
            scope, shrink, claim = "relation_binding_reuse_candidate", True, "task_selective_candidate"
        else:
            scope, shrink, claim = "material_specific_candidate", True, "task_selective_candidate"
        model_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "frozen_block_id": complete["frozen_block_id"],
            "phase338_full_model_gate_pass": prior_gate,
            "full_task_gate_count": sum(passes.values()),
            "relation_task_gate_count": relation_count,
            "source_control_gate_count": source_count,
            "cross_family_control_gate_count": cross_count,
            "material_relation_gate_pass": passes["material_relation_binding"],
            "scope_classification": scope, "claim_qualification": claim,
            "task_selective_layer_shrink_gate_open": shrink,
            "layer_shrink_executed": False, "single_unit_causal": False,
        })

    nodes = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase341:{row['model']}:{row['task_id']}",
            "model": row["model"], "family_id": row["task_class"],
            "mechanism_id": row["task_id"], "frozen_block_id": row["frozen_block_id"],
            "full_task_gate_pass": row["full_task_gate_pass"],
            "phase338_candidate_qualified": row["phase338_full_model_gate_pass"],
            "mapping_status": (
                "qualified_frozen_block_task_effect"
                if row["full_task_gate_pass"] and row["phase338_full_model_gate_pass"]
                else "descriptive_or_failed_task_effect"
            ),
            "rollout_batch_size": 1, "single_unit_causal": False,
        }
        for row in task_rows
    ]
    shrink_models = [row["model"] for row in model_rows if row["task_selective_layer_shrink_gate_open"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered), "registered_task_count": 6,
            "phrase_row_count": total_phrase, "rollout_row_count": total_rollout,
            "invalid_phrase_row_count": sum(row["invalid_phrase_row_count"] for row in completions),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "rollout_batch_size": 1,
        },
        "results": {
            "full_task_gate_count": sum(row["full_task_gate_pass"] for row in task_rows),
            "task_selective_layer_shrink_gate_models": shrink_models,
            "task_selective_layer_shrink_gate_model_count": len(shrink_models),
            "layer_shrink_executed_count": 0,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9", "mechanism_census_coverage": "72/72",
            "deep_causal_mechanisms_audited": "1/72",
            "qualified_cross_task_causal_boundary": "6 tasks x 3 models",
            "behavior_mechanism_closure": "0/72", "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase341_task_selective_layer_entry",
            "claim": "A Phase338-qualified block has a replicated task-selective boundary suitable for layer shrinking.",
            "status": "supported" if shrink_models else "not_supported",
            "evidence_level": "L4_single_case_cross_task_boundary",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase341_glm4_explicit_copy_reclassification",
            "claim": (
                "The GLM4 frozen early-source MLP block is reclassified as an explicit "
                "source-token copy candidate rather than a material-relation block."
            ),
            "status": (
                "model_specific_candidate"
                if any(
                    row["model"] == "glm4"
                    and row["scope_classification"]
                    == "explicit_source_token_copy_effect_material_candidate_rejected"
                    for row in model_rows
                )
                else "not_supported"
            ),
            "evidence_level": "L4_model_specific_reclassified_block_effect",
        },
        {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "claim_id": "phase341_mechanism_closure", "claim": "A language mechanism is closed.",
            "status": "not_supported", "evidence_level": "closure_gate_closed",
        },
    ]
    write_jsonl(root / "phase341_task_gate_summary.jsonl", task_rows)
    write_jsonl(root / "phase341_model_scope_summary.jsonl", model_rows)
    write_jsonl(root / "phase341_task_boundary_nodes.jsonl", nodes)
    write_jsonl(root / "phase341_claim_registry.jsonl", claims)
    write_json(root / "phase341_global_summary.json", summary)
    report = [
        "# Phase341 Fresh Qualified-Task Causal Boundary", "",
        f"- Registered cases: {len(registered)}",
        f"- Phrase rows: {total_phrase}", f"- Single-case rollout rows: {total_rollout}",
        f"- Full task gates: {summary['results']['full_task_gate_count']}/18", "",
        "## Model scope", "",
    ]
    for row in model_rows:
        report.append(
            f"- {row['model']}: prior={row['phase338_full_model_gate_pass']}, "
            f"relation={row['relation_task_gate_count']}/3, source={row['source_control_gate_count']}/1, "
            f"cross={row['cross_family_control_gate_count']}/2, scope={row['scope_classification']}, "
            f"shrink={row['task_selective_layer_shrink_gate_open']}"
        )
    report.extend(["", "No layer or neuron shrinking was executed; closure remains 0/72."])
    (root / "phase341_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
