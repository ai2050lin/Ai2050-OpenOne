#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase314"
SCHEMA_VERSION = "3.3.0"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/gpt5/result/phase314_core_mechanism_atlas_synthesis"


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


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def claim(claim_id: str, statement: str, scope: str, level: str, positives: list[str], negatives: list[str], limits: list[str], next_test: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "claim_id": claim_id,
        "statement": statement,
        "scope": scope,
        "evidence_level": level,
        "positive_evidence_files": positives,
        "negative_evidence_files": negatives,
        "counterexamples_or_limits": limits,
        "next_test": next_test,
        "status": "registered_not_global_theory",
    }


def build_claims(p311: dict[str, Any], p312: dict[str, Any], p313: dict[str, Any]) -> list[dict[str, Any]]:
    claims = [
        claim(
            "claim:phase314:core_three_position_observable",
            "Knowledge, syntax, and reasoning controlled tasks have measurable source/query/last component trajectories.",
            "three small models; controlled English prompts",
            "L4_component_attribution" if p311.get("status") == "complete" else "L3_partial_path",
            ["phase311_core_language_component_rows.jsonl", "phase311_core_language_position_summary_rows.jsonl"],
            ["phase311_core_language_missing_rows.jsonl"],
            ["Functional writer/router labels are not yet causal edges."],
            "Trace source-token attention and natural MLP gate activation into downstream boundaries.",
        ),
        claim(
            "claim:phase314:matched_path_reuse",
            "Within-mechanism path reuse can be compared against a same-family matched mechanism control.",
            "Phase311 controlled case bank",
            "L4_observational_matched_path",
            ["phase312_matched_similarity_rows.jsonl", "phase312_path_aggregate_rows.jsonl"],
            [],
            ["Adjusted reuse can still contain prompt-family and target-vocabulary leakage."],
            "Add token-count, frequency, surface-template, and negative-family matched controls.",
        ),
        claim(
            "claim:phase314:heldout_path_prediction",
            "Simple path prototypes have measurable heldout predictive value for family and mechanism labels.",
            "item_index=4 heldout cases",
            "L4_predictive_observation" if safe_float(p312.get("heldout_mechanism_accuracy")) > safe_float(p312.get("mechanism_random_baseline")) else "L3_prediction_not_above_baseline",
            ["phase312_heldout_prediction_rows.jsonl"],
            [],
            ["Only one heldout lexical/rule item per mechanism; template leakage remains possible."],
            "Freeze new templates and domains before prediction, then evaluate without threshold changes.",
        ),
        claim(
            "claim:phase314:component_interaction",
            "Attention and MLP interventions can interact non-additively on heldout target-vs-distractor margins.",
            "Two selected mechanisms per core family and model",
            "L5_candidate" if int(p313.get("winner_changed_count", 0)) > 0 else "L4_intervention_effect",
            ["phase313_nonlinear_interaction_rows.jsonl"],
            ["phase313_missing_rows.jsonl"],
            [
                "Half scaling and feature permutation are diagnostic, not natural gate reconstruction.",
                "Target-vs-distractor winner change is not strict full-vocabulary closure.",
            ],
            "Audit source-to-target necessity with state-matched replacement and full-vocabulary rollout controls.",
        ),
        claim(
            "claim:phase314:semantic_pair_reuse",
            "Shared-backbone semantic pairs show higher three-position path reuse than delta controls in all three models.",
            "Phase309 fruit-centered pair bank",
            "L4_cross_model_observation",
            ["phase309_reuse_delta_path_summary.json", "phase309_three_position_pair_matrix_rows.jsonl"],
            [],
            ["Narrow object domains; raw cosine may contain global layer-shape baseline."],
            "Repeat with expanded domains and matched random baselines.",
        ),
    ]
    return claims


def build_graph(aggregates: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, label: str) -> None:
        nodes[node_id] = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
        }

    for row in aggregates:
        family = str(row["family_id"])
        position = str(row["position_role"])
        component = str(row["component"])
        model = str(row["model"])
        family_node = f"family:{family}"
        event_node = f"event:{model}:{family}:{position}:{component}"
        add_node(family_node, "pattern_family", family)
        add_node(event_node, "component_event", f"{model} {position} {component}")
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "edge_id": f"phase314:observed:{model}:{family}:{position}:{component}",
                "source": family_node,
                "target": event_node,
                "edge_type": "observed_matched_path",
                "model": model,
                "adjusted_reuse_score": row["mean_adjusted_reuse_score"],
                "peak_normalized_depth": row["mean_peak_normalized_depth"],
                "evidence_level": "L4",
            }
        )
    for row in interactions:
        family = str(row["family_id"])
        model = str(row["model"])
        event_node = f"event:{model}:{family}:{row['selected_position_role']}:attention_mlp"
        boundary_node = f"boundary:{model}:{family}:target_distractor"
        add_node(event_node, "joint_component_event", f"{model} attention+MLP")
        add_node(boundary_node, "candidate_boundary", f"{family} target/distractor")
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "edge_id": f"phase314:intervention:{model}:{row['case_id']}",
                "source": event_node,
                "target": boundary_node,
                "edge_type": "heldout_joint_intervention",
                "model": model,
                "case_id": row["case_id"],
                "interaction_value": row["interaction_value"],
                "winner_changed": row["winner_changed"],
                "evidence_level": row["evidence_level"],
            }
        )
    return list(nodes.values()), edges


def report(path: Path, summary: dict[str, Any], claims: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase314 Core Mechanism Atlas Synthesis",
        "",
        "## Frozen Denominators",
        "",
        f"- planned_independent_model_cases: {summary['planned_independent_model_cases']}",
        f"- valid_independent_model_cases: {summary['valid_independent_model_cases']}",
        f"- expected_path_events: {summary['expected_path_events']}",
        f"- valid_path_events: {summary['valid_path_events']}",
        f"- heldout_predictions: {summary['heldout_predictions']}",
        f"- heldout_causal_cases: {summary['heldout_causal_cases']}",
        "",
        "## Scientific Coverage",
        "",
    ]
    for key, value in summary["scientific_progress"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Claims", ""]
    for row in claims:
        lines.append(f"- {row['claim_id']} [{row['evidence_level']}]: {row['statement']}")
    lines += [
        "",
        "## Strict Limit",
        "",
        "This atlas stage improves independent-case accounting, matched baselines, heldout prediction, and component interaction evidence. It does not establish a natural gate or strict clean language closure.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p311 = read_json(V2 / "phase311_core_language_physical_atlas_summary.json")
    p312 = read_json(V2 / "phase312_matched_path_feature_summary.json")
    p313 = read_json(V2 / "phase313_heldout_component_interaction_summary.json")
    cases = read_jsonl(V2 / "phase311_core_language_case_result_rows.jsonl")
    events = read_jsonl(V2 / "phase312_path_event_rows.jsonl")
    aggregates = read_jsonl(V2 / "phase312_path_aggregate_rows.jsonl")
    predictions = read_jsonl(V2 / "phase312_heldout_prediction_rows.jsonl")
    interactions = read_jsonl(V2 / "phase313_nonlinear_interaction_rows.jsonl")
    if not p311 or not p312 or not p313:
        raise SystemExit("Phase311-313 summaries are required")
    planned = int(p311.get("planned_independent_model_cases", 360))
    expected_events = planned * 3 * 3
    valid_cases = len(cases)
    observational_coverage = len(events) / expected_events if expected_events else 0.0
    heldout_expected = 3 * 3 * 8
    prediction_coverage = len(predictions) / heldout_expected
    causal_case_coverage = len(interactions) / planned if planned else 0.0
    family_accuracy = safe_float(p312.get("heldout_family_accuracy"))
    mechanism_accuracy = safe_float(p312.get("heldout_mechanism_accuracy"))
    mechanism_baseline = safe_float(
        p312.get("mechanism_target_conditioned_baseline", p312.get("mechanism_random_baseline"))
    )
    prediction_quality = max(0.0, min(1.0, (mechanism_accuracy - mechanism_baseline) / max(1e-9, 1.0 - mechanism_baseline)))
    causal_quality = min(1.0, (int(p313.get("winner_changed_count", 0)) + 0.25 * int(p313.get("nonlinear_interaction_count", 0))) / max(1, len(interactions)))
    claims = build_claims(p311, p312, p313)
    nodes, edges = build_graph(aggregates, interactions)
    scientific_progress = {
        "frozen_denominator_coverage": round(valid_cases / planned, 6) if planned else 0.0,
        "core_three_position_observational_coverage": round(observational_coverage, 6),
        "matched_baseline_coverage": round(len(read_jsonl(V2 / "phase312_matched_similarity_rows.jsonl")) / max(1, planned * 9), 6),
        "heldout_prediction_coverage": round(prediction_coverage, 6),
        "heldout_prediction_quality_above_baseline": round(prediction_quality, 6),
        "heldout_causal_case_coverage": round(causal_case_coverage, 6),
        "heldout_causal_quality_proxy": round(causal_quality, 6),
        "natural_gate_coverage": 0.0,
        "strict_clean_closure": 0.0,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "title": "Core language physical mechanism atlas synthesis",
        "planned_independent_model_cases": planned,
        "valid_independent_model_cases": valid_cases,
        "missing_independent_model_cases": int(p311.get("missing_independent_model_cases", 0)),
        "layer_component_rows": int(p311.get("layer_component_rows", 0)),
        "expected_path_events": expected_events,
        "valid_path_events": len(events),
        "matched_aggregate_rows": len(aggregates),
        "heldout_predictions": len(predictions),
        "heldout_family_accuracy": family_accuracy,
        "heldout_mechanism_accuracy": mechanism_accuracy,
        "heldout_mechanism_random_baseline": mechanism_baseline,
        "heldout_mechanism_unconditioned_random_baseline": safe_float(p312.get("mechanism_random_baseline")),
        "heldout_causal_cases": len(interactions),
        "heldout_winner_changes": int(p313.get("winner_changed_count", 0)),
        "nonlinear_interaction_cases": int(p313.get("nonlinear_interaction_count", 0)),
        "claim_rows": len(claims),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "evidence_level_counts": dict(Counter(str(r["evidence_level"]) for r in claims)),
        "scientific_progress": scientific_progress,
        "hard_limits": [
            "Core case bank uses controlled English prompts and does not cover open-set language tasks.",
            "Heldout set has one lexical/rule item per mechanism; template-heldout validation is still required.",
            "Half-scaling and permutation interventions do not reconstruct natural gates.",
            "Target-vs-distractor winner changes are not full-vocabulary or strict-clean closure.",
            "All tested models are small and can deviate materially from larger-model mechanisms.",
        ],
    }
    report(OUT / "phase314_report.md", payload, claims)
    write_json(OUT / "phase314_core_mechanism_atlas_summary.json", payload)
    write_jsonl(OUT / "phase314_mechanism_claim_rows.jsonl", claims)
    write_jsonl(OUT / "phase314_graph_nodes.jsonl", nodes)
    write_jsonl(OUT / "phase314_graph_edges.jsonl", edges)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "last_phase": PHASE,
        "updated_at": now(),
        "engineering_progress": {
            "atlas_data_system": 0.9,
            "core_case_schema": 1.0,
            "frontend_data_sync": 0.9,
        },
        "scientific_progress": scientific_progress,
        "legacy_management_estimates_are_not_mechanism_completion": True,
    }
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase314_core_mechanism_atlas_summary.json", payload)
        write_jsonl(base / "phase314_mechanism_claim_rows.jsonl", claims)
        write_jsonl(base / "phase314_graph_nodes.jsonl", nodes)
        write_jsonl(base / "phase314_graph_edges.jsonl", edges)
        write_json(base / "phase314_evidence_progress.json", progress)
        write_json(base / "progress.json", progress)
        report(base / "phase314_report.md", payload, claims)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
