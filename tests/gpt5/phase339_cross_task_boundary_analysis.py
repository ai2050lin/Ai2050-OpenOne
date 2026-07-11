#!/usr/bin/env python3
"""Aggregate Phase339 without turning cross-task sensitivity into closure."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase339_cross_task_boundary"
PHASE338 = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"
PHASE = "Phase339"
SCHEMA_VERSION = "15.0.0"
ROUND_DEFAULT = "early_source_cross_task_boundary"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("heldout", "private_heldout")
CONTROLS = ("wrong_depth_zero", "wrong_position_zero")
TASK_CLASS_ORDER = ("relation_binding", "source_operation", "cross_family")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def average(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [
        float(row[field]) for row in rows
        if row["condition"] == condition and row.get(field) is not None
        and math.isfinite(float(row[field]))
    ]
    return mean(values) if values else 0.0


def rate(rows: list[dict[str, Any]], condition: str, field: str) -> float:
    values = [bool(row[field]) for row in rows if row["condition"] == condition]
    return sum(values) / len(values) if values else 0.0


def phrase_valid_rate(rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> float:
    values = [
        bool(row.get("score_valid", False))
        for row in rows if row["condition"] in conditions
    ]
    return sum(values) / len(values) if values else 0.0


def classify_scope(
    task_pass: dict[str, bool], task_eligible: dict[str, bool], prior_gate: bool,
) -> tuple[str, bool, str]:
    """Return descriptive scope, shrink gate, and claim qualification."""
    material = task_pass.get("material_relation_binding", False)
    relation_count = sum(
        task_pass.get(task, False)
        for task in (
            "material_relation_binding", "attribute_relation_binding",
            "part_relation_binding", "location_relation_binding",
        )
    )
    source_count = sum(
        task_pass.get(task, False)
        for task in ("identity_copy", "source_span_extraction")
    )
    cross_count = sum(
        task_pass.get(task, False)
        for task in ("singular_agreement", "direct_entailment", "answer_only_protocol")
    )
    if not prior_gate:
        return "phase338_candidate_not_qualified", False, "descriptive_only"
    if not task_eligible.get("material_relation_binding", False):
        return "cross_task_denominator_ineligible", False, "scope_unresolved"
    if not material:
        return "material_effect_not_replicated", False, "candidate_rejected"
    if cross_count:
        return "broad_cross_family_source_sensitive", False, "broad_function_candidate"
    if source_count and relation_count >= 3:
        return "general_source_operation_sensitive", False, "broad_function_candidate"
    if source_count:
        return "mixed_relation_and_source_sensitive", False, "mixed_scope_candidate"
    if relation_count >= 3:
        return "relation_binding_reuse_candidate", True, "task_selective_candidate"
    if relation_count == 1:
        return "material_specific_candidate", True, "task_selective_candidate"
    return "partial_relation_scope_unresolved", False, "mixed_scope_candidate"


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase339_registered_protocol.json")
    registered = read_jsonl(root / "phase339_registered_cases.jsonl")
    thresholds = protocol["thresholds"]
    prior_rows = {
        row["model"]: row
        for row in read_jsonl(PHASE338 / "phase338_model_gate_summary.jsonl")
    }
    task_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    all_phrase: list[dict[str, Any]] = []
    all_rollout: list[dict[str, Any]] = []

    task_meta = {row["task_id"]: row["task_class"] for row in protocol["tasks"]}
    for model in MODELS:
        model_root = root / "models" / model
        phrase = read_jsonl(model_root / "phase339_phrase_rows.jsonl")
        rollout = read_jsonl(model_root / "phase339_rollout_rows.jsonl")
        complete = read_json(model_root / "complete.json")
        completions.append(complete)
        all_phrase.extend(phrase)
        all_rollout.extend(rollout)
        model_task_pass: dict[str, bool] = {}
        model_task_eligible: dict[str, bool] = {}

        for task_id, task_class in task_meta.items():
            split_results: dict[str, bool] = {}
            split_eligibility: dict[str, bool] = {}
            summary: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": model,
                "task_id": task_id,
                "task_class": task_class,
                "frozen_block_id": complete["frozen_block_id"],
                "phase338_full_model_gate_pass": bool(
                    prior_rows.get(model, {}).get("full_model_gate_pass", False)
                ),
            }
            for split in SPLITS:
                p_rows = [
                    row for row in phrase
                    if row["mechanism_id"] == task_id and row["split"] == split
                ]
                r_rows = [
                    row for row in rollout
                    if row["mechanism_id"] == task_id and row["split"] == split
                ]
                baseline_rate = rate(
                    r_rows, "baseline", "answer_head_semantic_correct"
                )
                correct_behavior = rate(
                    r_rows, "correct_zero", "behavior_lost_vs_baseline"
                )
                control_behavior = {
                    condition: rate(r_rows, condition, "behavior_lost_vs_baseline")
                    for condition in CONTROLS
                }
                correct_phrase = average(
                    p_rows, "correct_zero", "phrase_margin_loss_vs_baseline"
                )
                half_phrase = average(
                    p_rows, "correct_half", "phrase_margin_loss_vs_baseline"
                )
                permutation_phrase = average(
                    p_rows, "correct_permutation", "phrase_margin_loss_vs_baseline"
                )
                control_phrase = {
                    condition: average(
                        p_rows, condition, "phrase_margin_loss_vs_baseline"
                    )
                    for condition in CONTROLS
                }
                max_control_behavior = max(control_behavior.values(), default=0.0)
                max_control_phrase = max(control_phrase.values(), default=0.0)
                required_phrase_valid_rate = phrase_valid_rate(
                    p_rows, ("baseline", "correct_zero", *CONTROLS)
                )
                split_pass = bool(
                    baseline_rate >= thresholds["task_baseline_capability_rate_min"]
                    and required_phrase_valid_rate
                    >= thresholds["task_phrase_score_valid_rate_min"]
                    and correct_behavior >= thresholds["task_correct_behavior_loss_rate_min"]
                    and max_control_behavior
                    <= thresholds["task_wrong_control_behavior_loss_rate_max"]
                    and correct_phrase - max_control_phrase
                    >= thresholds["task_phrase_control_superiority_min"]
                )
                split_eligible = bool(
                    baseline_rate >= thresholds["task_baseline_capability_rate_min"]
                    and required_phrase_valid_rate
                    >= thresholds["task_phrase_score_valid_rate_min"]
                )
                split_results[split] = split_pass
                split_eligibility[split] = split_eligible
                prefix = split
                summary.update({
                    f"{prefix}_case_count": len(r_rows) // 4,
                    f"{prefix}_baseline_capability_rate": round(baseline_rate, 7),
                    f"{prefix}_required_phrase_valid_rate": round(
                        required_phrase_valid_rate, 7
                    ),
                    f"{prefix}_correct_behavior_loss_rate": round(correct_behavior, 7),
                    f"{prefix}_max_control_behavior_loss_rate": round(
                        max_control_behavior, 7
                    ),
                    f"{prefix}_correct_phrase_loss": round(correct_phrase, 7),
                    f"{prefix}_half_phrase_loss": round(half_phrase, 7),
                    f"{prefix}_permutation_phrase_loss": round(permutation_phrase, 7),
                    f"{prefix}_max_control_phrase_loss": round(max_control_phrase, 7),
                    f"{prefix}_correct_control_phrase_superiority": round(
                        correct_phrase - max_control_phrase, 7
                    ),
                    f"{prefix}_gate_pass": split_pass,
                    f"{prefix}_denominator_eligible": split_eligible,
                })
            summary["full_task_gate_pass"] = all(split_results.values())
            summary["full_task_denominator_eligible"] = all(
                split_eligibility.values()
            )
            summary["single_unit_causal"] = False
            summary["evidence_level"] = (
                "L4_cross_task_block_effect"
                if summary["full_task_gate_pass"]
                else "L3_cross_task_block_effect_not_confirmed"
            )
            model_task_pass[task_id] = summary["full_task_gate_pass"]
            model_task_eligible[task_id] = summary["full_task_denominator_eligible"]
            task_rows.append(summary)

        prior_gate = bool(prior_rows.get(model, {}).get("full_model_gate_pass", False))
        scope, shrink_gate, qualification = classify_scope(
            model_task_pass, model_task_eligible, prior_gate
        )
        class_counts = {
            task_class: sum(
                model_task_pass[task_id]
                for task_id, klass in task_meta.items() if klass == task_class
            )
            for task_class in TASK_CLASS_ORDER
        }
        model_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "frozen_block_id": complete["frozen_block_id"],
            "phase338_full_model_gate_pass": prior_gate,
            "full_task_gate_count": sum(model_task_pass.values()),
            "eligible_task_count": sum(model_task_eligible.values()),
            "relation_binding_task_gate_count": class_counts["relation_binding"],
            "source_operation_task_gate_count": class_counts["source_operation"],
            "cross_family_task_gate_count": class_counts["cross_family"],
            "material_relation_gate_pass": model_task_pass["material_relation_binding"],
            "scope_classification": scope,
            "claim_qualification": qualification,
            "task_selective_shrink_gate_open": shrink_gate,
            "layer_shrink_executed": False,
            "single_unit_causal": False,
        })

    task_lookup = {(row["model"], row["task_id"]): row for row in task_rows}
    nodes = []
    for model_row in model_rows:
        for task_id, task_class in task_meta.items():
            row = task_lookup[(model_row["model"], task_id)]
            nodes.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "node_id": f"phase339:{model_row['model']}:{task_id}",
                "model": model_row["model"],
                "family_id": task_class,
                "mechanism_id": task_id,
                "frozen_block_id": model_row["frozen_block_id"],
                "heldout_gate_pass": row["heldout_gate_pass"],
                "private_heldout_gate_pass": row["private_heldout_gate_pass"],
                "full_task_gate_pass": row["full_task_gate_pass"],
                "full_task_denominator_eligible": row["full_task_denominator_eligible"],
                "phase338_candidate_qualified": model_row["phase338_full_model_gate_pass"],
                "scope_classification": model_row["scope_classification"],
                "mapping_status": (
                    "qualified_candidate_task_effect"
                    if row["full_task_gate_pass"]
                    and model_row["phase338_full_model_gate_pass"]
                    else "descriptive_frozen_block_observation"
                ),
                "single_unit_causal": False,
            })

    task_cross_model = {
        task_id: sum(
            task_lookup[(model, task_id)]["full_task_gate_pass"] for model in MODELS
        )
        for task_id in task_meta
    }
    selective_models = [
        row["model"] for row in model_rows if row["task_selective_shrink_gate_open"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered),
            "registered_task_count": len(task_meta),
            "model_count": len(MODELS),
            "phrase_row_count": len(all_phrase),
            "rollout_row_count": len(all_rollout),
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "invalid_phrase_row_count": sum(
                row.get("invalid_phrase_row_count", 0) for row in completions
            ),
            "all_phrase_measurements_finite": all(
                row.get("phrase_measurements_all_finite", False)
                for row in completions
            ),
        },
        "results": {
            "full_task_gate_count": sum(row["full_task_gate_pass"] for row in task_rows),
            "full_task_denominator_eligible_count": sum(
                row["full_task_denominator_eligible"] for row in task_rows
            ),
            "task_cross_model_gate_counts": task_cross_model,
            "phase338_qualified_model_count": sum(
                row["phase338_full_model_gate_pass"] for row in model_rows
            ),
            "task_selective_shrink_gate_model_count": len(selective_models),
            "task_selective_shrink_gate_models": selective_models,
            "layer_shrink_executed_count": 0,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "deep_causal_mechanisms_audited": "1/72",
            "frozen_block_cross_task_boundary": "9 tasks x 3 models",
            "behavior_mechanism_closure": "0/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }

    claims = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "claim_id": "phase339_frozen_cross_task_denominator",
            "claim": "Phase338 frozen blocks were audited on nine preregistered tasks without reselection.",
            "status": "supported" if summary["denominator"]["all_model_completions_valid"] else "incomplete",
            "evidence_level": "L2_protocol_complete",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "claim_id": "phase339_task_selective_scope",
            "claim": "A Phase338-qualified block has a task-selective functional boundary suitable for layer shrinking.",
            "status": "supported" if selective_models else "not_supported",
            "evidence_level": "L4_cross_task_scope_audit",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "claim_id": "phase339_encoding_closure",
            "claim": "The language encoding mechanism is closed.",
            "status": "not_supported",
            "evidence_level": "closure_gate_closed",
        },
    ]

    write_jsonl(root / "phase339_task_gate_summary.jsonl", task_rows)
    write_jsonl(root / "phase339_model_scope_summary.jsonl", model_rows)
    write_jsonl(root / "phase339_task_boundary_nodes.jsonl", nodes)
    write_jsonl(root / "phase339_claim_registry.jsonl", claims)
    write_json(root / "phase339_global_summary.json", summary)

    report = [
        "# Phase339 Frozen-Block Cross-Task Boundary", "",
        f"- Registered cases: {len(registered)}",
        f"- Phrase rows: {len(all_phrase)}",
        f"- Rollout rows: {len(all_rollout)}",
        f"- Eligible model-task denominators: {summary['results']['full_task_denominator_eligible_count']}/27",
        f"- Full task gates: {summary['results']['full_task_gate_count']}/27", "",
        "## Model scope", "",
    ]
    for row in model_rows:
        report.append(
            f"- {row['model']}: prior={row['phase338_full_model_gate_pass']}, "
            f"eligible={row['eligible_task_count']}/9, "
            f"relation={row['relation_binding_task_gate_count']}/4, "
            f"source={row['source_operation_task_gate_count']}/2, "
            f"cross={row['cross_family_task_gate_count']}/3, "
            f"scope={row['scope_classification']}, "
            f"shrink_gate={row['task_selective_shrink_gate_open']}"
        )
    report.extend([
        "", "## Boundaries", "",
        "- Wrong-depth and wrong-position blocks are null-location controls.",
        "- Same-block permutation is recorded as structure sensitivity, not a null control.",
        "- Qwen3 and DeepSeek7B remain descriptive because their Phase338 gates failed.",
        f"- Non-finite phrase rows: {summary['denominator']['invalid_phrase_row_count']}; any affected split is ineligible.",
        "- GLM4 has no fully eligible task denominator in this matrix, so its scope remains unresolved.",
        "- No layer, channel, or neuron shrinking was performed.",
        "- No behavior mechanism or intelligent theory was closed.",
    ])
    (root / "phase339_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
