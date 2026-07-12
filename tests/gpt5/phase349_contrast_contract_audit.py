#!/usr/bin/env python3
"""Audit whether the 72 Phase330 tasks can form strict A/B/C/D contrast contracts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase349"
SCHEMA_VERSION = "25.0.0"
ROUND_DEFAULT = "orthogonal_contrast_contract_audit"
OUT = ROOT / "tests/gpt5/result/phase349_contrast_contract_audit"
SOURCE = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas/phase330_case_bank.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
OPERATIONS = (
    "read", "copy", "relation_bind", "role_bind", "compare", "negate",
    "condition_check", "reason_compose", "morph_transform", "content_transform",
    "continue", "stop",
)


OPERATION_BY_FAMILY: dict[str, dict[str, str]] = {
    "content_knowledge": {
        "category": "read", "attribute": "read", "function": "read", "part": "relation_bind",
        "material": "read", "habitat": "read", "comparison_relation": "compare",
        "negated_attribute": "negate",
    },
    "output_protocol": {
        "answer_only": "stop", "single_sentence": "stop", "single_item_list": "content_transform",
        "json": "content_transform", "quote_closure": "stop", "newline_closure": "stop",
        "format_template": "content_transform", "no_explanation": "stop",
    },
    "reasoning_constraint": {
        "direct_entailment": "condition_check", "direct_contradiction": "negate",
        "two_hop_entailment": "reason_compose", "two_hop_blocked": "reason_compose",
        "transitive_order": "compare", "reversed_order_control": "compare",
        "conjunction_rule": "condition_check", "missing_condition_control": "condition_check",
    },
    "syntax_structure": {
        "subject_role": "role_bind", "object_role": "role_bind",
        "singular_agreement": "morph_transform", "plural_agreement": "morph_transform",
        "past_tense": "morph_transform", "pronoun_number": "morph_transform",
        "adjective_attachment": "role_bind", "relative_clause_role": "role_bind",
    },
    "language_action": {
        "answer": "read", "classify": "read", "extract": "read", "transform": "content_transform",
        "translate": "content_transform", "rewrite": "content_transform", "summarize": "content_transform",
        "refuse_or_comply": "condition_check",
    },
    "cross_lingual": {
        "semantic_equivalence": "compare", "translation": "content_transform", "negation": "negate",
        "question": "read", "role_binding": "role_bind", "number_agreement": "morph_transform",
        "protocol_preservation": "content_transform", "mixed_language_routing": "role_bind",
    },
    "readout_competition": {
        "target_vs_wrong": "read", "target_vs_continue": "stop", "target_vs_echo": "copy",
        "target_vs_protocol": "stop", "target_vs_punctuation": "stop", "answer_alias": "read",
        "multi_token_answer": "continue", "full_vocabulary_blockers": "read",
    },
    "state_drift": {
        "entity_drift": "role_bind", "attribute_drift": "relation_bind", "role_drift": "role_bind",
        "language_drift": "role_bind", "format_drift": "stop", "reasoning_drift": "reason_compose",
        "repetition_drift": "stop", "long_context_drift": "read",
    },
    "closure": {
        "semantic_completion": "stop", "protocol_completion": "stop", "stop_wins": "stop",
        "continue_suppression": "stop", "multi_token_completion": "continue", "alias_completion": "stop",
        "generation_stability": "stop", "client_visible_closure": "stop",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))


def overlap(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def contract_metrics(target_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, float]:
    target = {(row["item_index"], row["template_id"]): row for row in target_rows}
    control = {(row["item_index"], row["template_id"]): row for row in control_rows}
    keys = sorted(set(target) & set(control))
    lexical = [overlap(target[key]["context"] + " " + target[key]["question"], control[key]["context"] + " " + control[key]["question"]) for key in keys]
    return {
        "paired_case_count": float(len(keys)),
        "mean_lexical_token_overlap": mean(lexical) if lexical else 0.0,
        "language_match_rate": mean(target[key]["language"] == control[key]["language"] for key in keys) if keys else 0.0,
        "protocol_match_rate": mean(target[key]["protocol"] == control[key]["protocol"] for key in keys) if keys else 0.0,
        "target_bucket_match_rate": mean(target[key]["target_bucket"] == control[key]["target_bucket"] for key in keys) if keys else 0.0,
        "target_identity_rate": mean(target[key]["target"] == control[key]["target"] for key in keys) if keys else 0.0,
    }


def audit(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows = [row for row in read_jsonl(SOURCE) if row["template_id"] in {"template_a", "template_b"}]
    by_mechanism: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_mechanism.setdefault((row["family_id"], row["mechanism_id"]), []).append(row)
    expected = {(family, mechanism) for family, values in OPERATION_BY_FAMILY.items() for mechanism in values}
    if set(by_mechanism) != expected:
        raise RuntimeError("Phase349 operation registry does not match the frozen 72-mechanism census")
    if any(operation not in OPERATIONS for values in OPERATION_BY_FAMILY.values() for operation in values.values()):
        raise RuntimeError("Unknown candidate operation label")

    contracts = []
    for family, mechanism in sorted(expected):
        operation = OPERATION_BY_FAMILY[family][mechanism]
        candidates = []
        for candidate in OPERATION_BY_FAMILY[family]:
            if candidate == mechanism:
                continue
            metrics = contract_metrics(by_mechanism[(family, mechanism)], by_mechanism[(family, candidate)])
            operation_distinct = OPERATION_BY_FAMILY[family][candidate] != operation
            rank = (
                int(operation_distinct),
                metrics["language_match_rate"] + metrics["protocol_match_rate"] + metrics["target_bucket_match_rate"],
                metrics["mean_lexical_token_overlap"],
            )
            candidates.append((rank, candidate, metrics, operation_distinct))
        _rank, control, metrics, operation_distinct = max(candidates, key=lambda value: value[0])
        ready = bool(
            operation_distinct
            and metrics["paired_case_count"] == 48
            and metrics["mean_lexical_token_overlap"] >= 0.60
            and metrics["language_match_rate"] == 1.0
            and metrics["protocol_match_rate"] == 1.0
            and metrics["target_bucket_match_rate"] >= 0.75
        )
        contracts.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "contract_id": f"phase349:{family}:{mechanism}",
            "family_id": family, "mechanism_id": mechanism,
            "candidate_primary_operation": operation,
            "matched_control_mechanism_id": control,
            "control_candidate_operation": OPERATION_BY_FAMILY[family][control],
            "operation_labels_distinct": operation_distinct,
            **{key: round(value, 7) for key, value in metrics.items()},
            "strict_quadruplet_ready": ready,
            "mapping_status": "strict_pair_ready" if ready else "contrast_contract_repair_required",
            "selection_used_model_effects": False,
            "single_unit_causal": False,
        })

    family_summary = []
    for family in sorted(OPERATION_BY_FAMILY):
        values = [row for row in contracts if row["family_id"] == family]
        family_summary.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family, "mechanism_count": len(values),
            "strict_pair_ready_count": sum(row["strict_quadruplet_ready"] for row in values),
            "mean_lexical_token_overlap": round(mean(row["mean_lexical_token_overlap"] for row in values), 7),
        })
    ready_count = sum(row["strict_quadruplet_ready"] for row in contracts)
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "family_count": len(OPERATION_BY_FAMILY), "mechanism_count": len(contracts),
            "candidate_operation_count": len(OPERATIONS), "source_case_count": len(rows),
            "proposed_prompt_model_case_count": 72 * 12 * 2 * 3 * 4,
        },
        "results": {
            "strict_quadruplet_ready_mechanism_count": ready_count,
            "repair_required_mechanism_count": len(contracts) - ready_count,
            "all_72_contract_gate_pass": ready_count == 72,
            "model_execution_started": False,
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
        },
        "thresholds": {
            "paired_case_count": 48, "mean_lexical_token_overlap_min": 0.60,
            "language_match_rate": 1.0, "protocol_match_rate": 1.0,
            "target_bucket_match_rate_min": 0.75, "operation_labels_must_differ": True,
        },
        "entry_decision": "generate_full_20736_denominator" if ready_count == 72 else "repair_contrast_contracts_before_model_execution",
        "claim_boundaries": {
            "candidate_operation_labels_are_validated_theory": False,
            "text_overlap_proves_operation_orthogonality": False,
            "strict_pair_ready_is_a_causal_result": False,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    root = OUT / round_name
    write_jsonl(root / "phase349_contrast_contract_registry.jsonl", contracts)
    write_jsonl(root / "phase349_family_contract_summary.jsonl", family_summary)
    write_json(root / "phase349_global_summary.json", summary)
    report = [
        "# Phase349 Orthogonal Contrast Contract Audit", "",
        f"- Frozen mechanisms: {len(contracts)}", f"- Candidate operations: {len(OPERATIONS)}",
        f"- Strict A/B pair-ready mechanisms: {ready_count}/72",
        f"- Entry decision: {summary['entry_decision']}", "",
        "No model forward, intervention, physical candidate selection, or closure test was executed.",
    ]
    (root / "phase349_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(audit(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
