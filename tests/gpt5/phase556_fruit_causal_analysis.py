#!/usr/bin/env python3
"""Qualify Phase556 held-out interventions and emit evidence-preserving edges."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
MODELS = ("qwen3", "glm4")
SCENARIOS = ("matched_factor_delta", "wrong_depth_delta", "channel_roll_delta")
SUMMARY_PATH = OUT_DIR / "phase556_causal_analysis_summary.json"
EDGE_PATH = OUT_DIR / "phase556_qualified_physical_edges.jsonl"
EXPECTED_READOUT_CONTRACT = "first_non_whitespace_candidate_content_token_v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def median(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def parameter_parent_gates() -> dict[tuple[str, str], bool]:
    paths = (
        OUT_DIR / "phase556_direct_parent_analysis.json",
        OUT_DIR / "phase556_glm4_direct_parent_analysis.json",
    )
    gates: dict[tuple[str, str], bool] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if not model:
            continue
        gates.update({
            (model, mechanism): bool(report.get("parameter_localization_authorized", False))
            for mechanism, report in payload.get("mechanism_reports", {}).items()
        })
    return gates


def analyze() -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = OUT_DIR / "causal_intervention" / model / "phase556_causal_rows.jsonl"
        all_rows.extend(read_jsonl(path))
    readout_contracts = {row.get("restricted_readout_contract") for row in all_rows}
    if readout_contracts != {EXPECTED_READOUT_CONTRACT}:
        raise RuntimeError(f"Phase556 stale or mixed causal readout contracts: {readout_contracts}")
    torch_dtypes = {row.get("torch_dtype") for row in all_rows}
    if torch_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"Phase556 causal dtype drift: {torch_dtypes}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["candidate_id"]].append(row)

    candidate_reports: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for cid, rows in sorted(grouped.items()):
        scenario_reports: dict[str, Any] = {}
        for scenario in SCENARIOS:
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            valid_rows = [row for row in scenario_rows if row["numerical_valid"]]
            target = [row for row in valid_rows if row["pair_role"] == "target"]
            control = [row for row in valid_rows if row["pair_role"] == "specificity_control"]
            scenario_reports[scenario] = {
                "total_n": len(scenario_rows),
                "numerical_valid_n": len(valid_rows),
                "numerical_invalid_n": len(scenario_rows) - len(valid_rows),
                "numerical_valid_rate": len(valid_rows) / len(scenario_rows) if scenario_rows else 0.0,
                "target_n": len(target),
                "specificity_control_n": len(control),
                "target_donor_selection_rate": mean_bool(target, "patched_donor_selected"),
                "target_recipient_preservation_rate": mean_bool(target, "patched_recipient_preserved"),
                "specificity_control_recipient_preservation_rate": mean_bool(
                    control, "patched_recipient_preserved"
                ),
                "median_target_transfer_fraction": median([
                    row["transfer_fraction"] for row in target
                ]),
                "median_patch_delta_norm": median([row["patch_delta_norm"] for row in scenario_rows]),
            }
        matched = scenario_reports["matched_factor_delta"]
        wrong_depth = scenario_reports["wrong_depth_delta"]
        channel_roll = scenario_reports["channel_roll_delta"]
        control_max = max(
            wrong_depth["target_donor_selection_rate"],
            channel_roll["target_donor_selection_rate"],
        )
        donor_selection_margin = matched["target_donor_selection_rate"] - control_max
        transfer = matched["median_target_transfer_fraction"]
        natural_rows = [row for row in rows if row["scenario"] == "matched_factor_delta"]
        natural_rows = [row for row in natural_rows if row["numerical_valid"]]
        baseline_correct = mean_bool(natural_rows, "baseline_semantic_correct_restricted")
        donor_correct = mean_bool(natural_rows, "natural_donor_semantic_correct_restricted")
        criteria = {
            "target_n_at_least_96": matched["target_n"] >= 96,
            "specificity_control_n_at_least_96": matched["specificity_control_n"] >= 96,
            "natural_endpoints_at_least_95pct": min(baseline_correct, donor_correct) >= 0.95,
            "all_scenarios_numerically_valid_at_least_95pct": min(
                report["numerical_valid_rate"] for report in scenario_reports.values()
            ) >= 0.95,
            "matched_donor_selection_at_least_50pct": matched["target_donor_selection_rate"] >= 0.50,
            "matched_over_controls_at_least_15pp": donor_selection_margin >= 0.15,
            "median_transfer_at_least_0_20": transfer is not None and transfer >= 0.20,
            "irrelevant_query_preservation_at_least_90pct": (
                matched["specificity_control_recipient_preservation_rate"] >= 0.90
            ),
        }
        qualified = all(criteria.values())
        exemplar = rows[0]
        report = {
            "candidate_id": cid,
            "model": exemplar["model"],
            "torch_dtype": exemplar.get("torch_dtype", "unspecified"),
            "mechanism": exemplar["mechanism"],
            "component": exemplar["component"],
            "layer": exemplar["layer"],
            "relative_depth": exemplar["relative_depth"],
            "component_rank": exemplar["component_rank"],
            "anchor_count": len({row["anchor_id"] for row in rows}),
            "baseline_restricted_accuracy": baseline_correct,
            "donor_restricted_accuracy": donor_correct,
            "matched_over_control_donor_selection_margin": donor_selection_margin,
            "criteria": criteria,
            "causal_qualified": qualified,
            "causal_state_carrier": qualified and exemplar["component"] == "layer_input",
            "causal_component_update": qualified and exemplar["component"] != "layer_input",
            "scenario_reports": scenario_reports,
            "sealed": False,
        }
        candidate_reports.append(report)
        if qualified:
            mechanism_label = (
                "category_assignment_reuse" if exemplar["mechanism"] == "category_reuse"
                else "entity_attribute_binding_difference"
            )
            node = f"{exemplar['model']}:{exemplar['component']}:L{exemplar['layer']}:query_end"
            edges.extend([
                {
                    "schema_version": "phase556_physical_edge.v1",
                    "phase_id": "Phase556",
                    "created_at": now(),
                    "edge_id": f"{cid}__factor_to_state",
                    "model": exemplar["model"],
                    "source": mechanism_label,
                    "target": node,
                    "relation": (
                        "matched_factor_difference_is_sufficient_at_state"
                        if exemplar["component"] == "layer_input"
                        else "matched_factor_difference_in_component_changes_state"
                    ),
                    "evidence_level": (
                        "held_out_causal_state_intervention"
                        if exemplar["component"] == "layer_input"
                        else "held_out_causal_component_intervention"
                    ),
                    "compute_edge": exemplar["component"] != "layer_input",
                    "causal": True,
                    "candidate_id": cid,
                    "sealed": False,
                },
                {
                    "schema_version": "phase556_physical_edge.v1",
                    "phase_id": "Phase556",
                    "created_at": now(),
                    "edge_id": f"{cid}__state_to_answer",
                    "model": exemplar["model"],
                    "source": node,
                    "target": f"{exemplar['mechanism']}:restricted_answer_boundary",
                    "relation": "state_difference_changes_corresponding_answer",
                    "evidence_level": (
                        "held_out_causal_state_intervention"
                        if exemplar["component"] == "layer_input"
                        else "held_out_causal_component_intervention"
                    ),
                    "compute_edge": exemplar["component"] != "layer_input",
                    "causal": True,
                    "candidate_id": cid,
                    "sealed": False,
                },
            ])

    qualified = [row for row in candidate_reports if row["causal_qualified"]]
    parent_gates = parameter_parent_gates()
    component_update_sufficiency = [
        row for row in qualified if row["component"] in ("attention_output", "mlp_output")
    ]
    parameter_qualified = [
        row for row in component_update_sufficiency
        if parent_gates.get((row["model"], row["mechanism"]), False)
    ]
    mechanism_model_support: dict[str, list[str]] = defaultdict(list)
    for row in qualified:
        mechanism_model_support[row["mechanism"]].append(row["model"])
    replicated_mechanisms = sorted(
        mechanism for mechanism, models in mechanism_model_support.items()
        if set(models) == set(MODELS)
    )
    write_jsonl(EDGE_PATH, edges)
    summary = {
        "schema_version": "phase556_causal_analysis_summary.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "tested_model_count": len(MODELS),
        "candidate_count": len(candidate_reports),
        "qualified_candidate_count": len(qualified),
        "qualified_edge_count": len(edges),
        "replicated_mechanisms": replicated_mechanisms,
        "component_update_sufficiency_count": len(component_update_sufficiency),
        "parameter_localization_authorized": bool(parameter_qualified),
        "parameter_localization_requires_independent_parent_writer_gate": True,
        "parameter_parent_gates": {
            f"{model}:{mechanism}": passed
            for (model, mechanism), passed in sorted(parent_gates.items())
        },
        "closure_claim_authorized": False,
        "sealed_split_read": False,
        "qualification_thresholds_are_preregistered_in_code": True,
        "restricted_readout_contract": EXPECTED_READOUT_CONTRACT,
        "torch_dtypes": sorted(torch_dtypes),
        "candidate_reports": candidate_reports,
        "physical_edges_path": str(EDGE_PATH.relative_to(ROOT)),
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "candidate_count": len(candidate_reports),
        "qualified_candidate_count": len(qualified),
        "qualified_edge_count": len(edges),
        "replicated_mechanisms": replicated_mechanisms,
        "parameter_localization_authorized": bool(parameter_qualified),
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
