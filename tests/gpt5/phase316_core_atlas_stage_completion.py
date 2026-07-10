#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase316"
SCHEMA_VERSION = "3.5.0"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/gpt5/result/phase316_core_atlas_stage_completion"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    p311 = read_json(V2 / "phase311_core_language_physical_atlas_summary.json")
    p312 = read_json(V2 / "phase312_matched_path_feature_summary.json")
    p313 = read_json(V2 / "phase313_heldout_component_interaction_summary.json")
    p314 = read_json(V2 / "phase314_core_mechanism_atlas_summary.json")
    p315 = read_json(V2 / "phase315_template_heldout_summary.json")
    if not all([p311, p312, p313, p314, p315]):
        raise SystemExit("Phase311-315 summaries are required")
    claims = read_jsonl(V2 / "phase314_mechanism_claim_rows.jsonl")
    claims.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "claim_id": "claim:phase316:template_heldout_path_prediction",
            "statement": "Frozen Phase311 path prototypes retain partial family and mechanism predictive value after rule-based prompt paraphrasing.",
            "scope": "72 template-and-item-heldout cases across three small models",
            "evidence_level": "L4_predictive_observation",
            "positive_evidence_files": ["phase315_template_heldout_prediction_rows.jsonl"],
            "negative_evidence_files": [],
            "counterexamples_or_limits": [
                "Mechanism accuracy drops materially from same-template heldout to template-heldout.",
                "Paraphrases are rule-based and not open-set prompts.",
            ],
            "next_test": "Use independently authored templates, new domains, and open-set unknown-family prompts.",
            "status": "registered_not_global_theory",
        }
    )
    template_family_acc = safe_float(p315["template_heldout_family_accuracy"])
    template_mech_acc = safe_float(p315["template_heldout_mechanism_accuracy"])
    template_mech_base = safe_float(p315["mechanism_random_baseline"])
    template_quality = max(0.0, min(1.0, (template_mech_acc - template_mech_base) / (1.0 - template_mech_base)))
    scientific_progress = {
        "controlled_core_independent_case_coverage": 1.0,
        "controlled_core_three_position_event_coverage": 1.0,
        "matched_control_analysis_coverage": 1.0,
        "same_template_heldout_prediction_coverage": 1.0,
        "template_heldout_prediction_coverage": 1.0,
        "template_heldout_prediction_quality_above_random": round(template_quality, 6),
        "heldout_causal_case_coverage": safe_float(p314["scientific_progress"]["heldout_causal_case_coverage"]),
        "heldout_causal_quality_proxy": 0.0,
        "natural_gate_coverage": 0.0,
        "strict_clean_closure": 0.0,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "stage_complete_data_not_mechanism",
        "title": "Core language physical atlas stage completion",
        "phase_range": "Phase311-316",
        "independent_controlled_model_cases": int(p311["valid_independent_model_cases"]),
        "independent_template_heldout_model_cases": int(p315["valid_template_heldout_model_cases"]),
        "total_new_independent_model_cases": int(p311["valid_independent_model_cases"]) + int(p315["valid_template_heldout_model_cases"]),
        "layer_component_rows": int(p311["layer_component_rows"]) + int(p315["component_rows"]),
        "path_event_rows": int(p312["path_event_rows"]),
        "matched_similarity_rows": int(p312["matched_similarity_rows"]),
        "same_template_heldout_family_accuracy": safe_float(p312["heldout_family_accuracy"]),
        "same_template_heldout_mechanism_accuracy": safe_float(p312["heldout_mechanism_accuracy"]),
        "same_template_target_conditioned_baseline": safe_float(p312["mechanism_target_conditioned_baseline"]),
        "template_heldout_family_accuracy": template_family_acc,
        "template_heldout_mechanism_accuracy": template_mech_acc,
        "template_heldout_mechanism_random_baseline": template_mech_base,
        "template_heldout_accuracy_drop": {
            "family": round(template_family_acc - safe_float(p312["heldout_family_accuracy"]), 6),
            "mechanism": round(template_mech_acc - safe_float(p312["heldout_mechanism_accuracy"]), 6),
        },
        "mean_adjusted_reuse_score": safe_float(p312["mean_adjusted_reuse_score"]),
        "adjusted_reuse_by_family": p312["adjusted_reuse_by_family"],
        "heldout_causal_cases": int(p313["selected_heldout_cases"]),
        "heldout_winner_changes": int(p313["winner_changed_count"]),
        "strong_nonlinear_interaction_cases": int(p313["nonlinear_interaction_count"]),
        "full_vocab_top1_changed_cases": int(p313["full_vocab_top1_changed_count"]),
        "claim_rows": len(claims),
        "scientific_progress": scientific_progress,
        "objective_conclusions": [
            "Controlled knowledge, syntax, and reasoning tasks have complete three-position component observations for the frozen denominator.",
            "Matched-control adjusted reuse is positive but small overall; reasoning is the strongest of the three controlled families.",
            "Path prototypes predict heldout items above simple baselines, but accuracy drops under template paraphrasing.",
            "Heldout attention/MLP half-scaling produced no target-vs-distractor winner flips and no strong interaction above the preset threshold.",
            "Current evidence remains L4 observational/predictive; no natural gate or strict-clean closure was established.",
        ],
        "hard_limits": [
            "Reasoning prompts use balanced yes/no candidate vocabularies and can share answer-route structure.",
            "Knowledge and syntax adjusted reuse are close to zero after matched controls, so a universal family backbone is not established.",
            "Template paraphrases are rule-based and still preserve much of the task wording.",
            "Causal audit coverage is only five percent of the controlled denominator.",
            "All models are small and the tested prompts are English controlled tasks.",
        ],
        "next_stage": {
            "name": "natural source-to-boundary causal edge mapping",
            "tasks": [
                "Independently authored template and domain holdouts.",
                "Source-token attention to MLP gate/up/product/down tracing.",
                "State-matched replacement rather than only half scaling.",
                "Full-vocabulary and natural rollout boundary validation.",
                "Open-set family discovery and unknown-family rate measurement.",
            ],
        },
    }
    progress = {
        "schema_version": SCHEMA_VERSION,
        "last_phase": PHASE,
        "updated_at": now(),
        "engineering_progress": {
            "atlas_data_system": 0.92,
            "core_case_schema": 1.0,
            "provenance_fields": 1.0,
            "frontend_data_sync": 0.9,
        },
        "scientific_progress": scientific_progress,
        "legacy_management_estimates_are_not_mechanism_completion": True,
    }
    report_lines = [
        "# Phase316 Core Language Physical Atlas Stage Completion",
        "",
        f"- total_new_independent_model_cases: {payload['total_new_independent_model_cases']}",
        f"- layer_component_rows: {payload['layer_component_rows']}",
        f"- same_template_mechanism_accuracy: {payload['same_template_heldout_mechanism_accuracy']}",
        f"- template_heldout_mechanism_accuracy: {payload['template_heldout_mechanism_accuracy']}",
        f"- heldout_winner_changes: {payload['heldout_winner_changes']}",
        f"- natural_gate_coverage: {scientific_progress['natural_gate_coverage']}",
        "",
        "## Conclusion",
        "",
        "The controlled physical atlas and its validation protocol are complete for this frozen denominator. The language mechanism is not closed: matched reuse is weak outside reasoning, template transfer is partial, and the heldout intervention audit is negative.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "phase316_core_atlas_stage_summary.json", payload)
    write_jsonl(OUT / "phase316_mechanism_claim_rows.jsonl", claims)
    (OUT / "phase316_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase316_core_atlas_stage_summary.json", payload)
        write_jsonl(base / "phase316_mechanism_claim_rows.jsonl", claims)
        write_json(base / "phase316_evidence_progress.json", progress)
        write_json(base / "progress.json", progress)
        (base / "phase316_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
