#!/usr/bin/env python3
"""Build a fixed-node natural physical atlas from Phase347 traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase347_three_core_natural_trace"
PHASE = "Phase347"
SCHEMA_VERSION = "23.0.0"
ROUND_DEFAULT = "three_core_natural_physical_trace"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    registered = read_jsonl(root / "phase347_registered_cases.jsonl")
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    case_rows = [
        row for model in MODELS
        for row in read_jsonl(root / "models" / model / "phase347_case_rows.jsonl")
    ]
    traces = [
        row for model in MODELS
        for row in read_jsonl(root / "models" / model / "phase347_trace_rows.jsonl")
    ]

    buckets: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"case_count": 0.0, "finite_count": 0.0, "norm_sum": 0.0,
                 "projection_sum": 0.0, "abs_projection_sum": 0.0,
                 "abs_cosine_sum": 0.0}
    )
    for row in traces:
        key = (
            row["model"], row["mechanism_id"], row["task_class"], row["component"],
            row["depth_bin"], row["position_role"],
        )
        bucket = buckets[key]
        count = row["finite_count"]
        bucket["case_count"] += row["case_count"]
        bucket["finite_count"] += count
        if count:
            bucket["norm_sum"] += row["mean_component_l2_norm"] * count
            bucket["projection_sum"] += row["mean_target_first_token_projection"] * count
            bucket["abs_projection_sum"] += row["mean_abs_target_first_token_projection"] * count
            bucket["abs_cosine_sum"] += row["mean_abs_target_first_token_cosine"] * count

    nodes = []
    for key, bucket in buckets.items():
        model, task, task_class, component, depth, role = key
        finite = bucket["finite_count"]
        nodes.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "node_id": f"phase347:{model}:{task}:{component}:{depth}:{role}",
            "model": model, "mechanism_id": task, "family_id": task_class,
            "component": component, "depth_bin": depth, "position_role": role,
            "case_count": int(bucket["case_count"]), "finite_count": int(finite),
            "finite_rate": round(finite / bucket["case_count"], 7),
            "mean_component_l2_norm": round(bucket["norm_sum"] / finite, 7) if finite else None,
            "mean_target_first_token_projection": round(bucket["projection_sum"] / finite, 7) if finite else None,
            "mean_abs_target_first_token_projection": round(bucket["abs_projection_sum"] / finite, 7) if finite else None,
            "mean_abs_target_first_token_cosine": round(bucket["abs_cosine_sum"] / finite, 7) if finite else None,
            "mapping_status": "natural_trace_observed", "causal_status": "not_tested",
            "single_unit_causal": False,
        })
    nodes.sort(key=lambda row: row["node_id"])

    # Remove the architecture-wide physical-node baseline before ranking tasks.
    # This prevents late residual scale from becoming a false task mechanism.
    for row in nodes:
        peers = [
            value["mean_abs_target_first_token_cosine"] for value in nodes
            if value["model"] == row["model"]
            and value["component"] == row["component"]
            and value["depth_bin"] == row["depth_bin"]
            and value["position_role"] == row["position_role"]
            and value["mechanism_id"] != row["mechanism_id"]
        ]
        common = sum(peers) / len(peers)
        alignment = row["mean_abs_target_first_token_cosine"]
        row["cross_task_common_alignment"] = round(common, 7)
        row["task_specific_alignment_excess"] = round(alignment - common, 7)

    dominant = []
    for model in MODELS:
        tasks = sorted({row["mechanism_id"] for row in nodes if row["model"] == model})
        for task in tasks:
            candidates = [row for row in nodes if row["model"] == model and row["mechanism_id"] == task]
            winner = max(candidates, key=lambda row: row["task_specific_alignment_excess"])
            dominant.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "mechanism_id": task, "family_id": winner["family_id"],
                "component": winner["component"], "depth_bin": winner["depth_bin"],
                "position_role": winner["position_role"],
                "mean_abs_target_first_token_projection": winner["mean_abs_target_first_token_projection"],
                "mean_abs_target_first_token_cosine": winner["mean_abs_target_first_token_cosine"],
                "cross_task_common_alignment": winner["cross_task_common_alignment"],
                "task_specific_alignment_excess": winner["task_specific_alignment_excess"],
                "selection_scope": "cross_task_baseline_adjusted_natural_trace", "causal_status": "not_tested",
            })

    convergence = []
    for task in sorted({row["mechanism_id"] for row in dominant}):
        rows = [row for row in dominant if row["mechanism_id"] == task]
        stage_roles = {f"{row['depth_bin']}:{row['position_role']}" for row in rows}
        exact_nodes = {f"{row['component']}:{row['depth_bin']}:{row['position_role']}" for row in rows}
        convergence.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "mechanism_id": task, "family_id": rows[0]["family_id"],
            "model_count": len(rows), "stage_role_agreement": len(stage_roles) == 1,
            "exact_node_agreement": len(exact_nodes) == 1,
            "stage_role_values": sorted(stage_roles), "exact_node_values": sorted(exact_nodes),
            "causal_status": "not_tested",
        })

    expected_nodes = 3 * 10 * 3 * 3 * 3
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": len(registered), "case_row_count": len(case_rows),
            "raw_trace_row_count": len(traces), "fixed_physical_node_count": len(nodes),
            "expected_fixed_physical_node_count": expected_nodes,
            "model_completion_valid_count": sum(row["valid"] for row in completions),
            "nonfinite_case_count": sum(row["finite_capture_count"] != row["capture_count"] for row in case_rows),
        },
        "results": {
            "natural_trace_task_model_count": len(dominant),
            "cross_model_stage_role_agreement_task_count": sum(row["stage_role_agreement"] for row in convergence),
            "cross_model_exact_node_agreement_task_count": sum(row["exact_node_agreement"] for row in convergence),
            "coarse_causal_screen_candidate_tasks": [row["mechanism_id"] for row in convergence if row["stage_role_agreement"]],
            "internal_intervention_executed_count": 0,
            "single_unit_causal_count": 0,
            "behavior_mechanism_closed_count": 0,
        },
        "claim_boundary": {
            "natural_trace_is_causal": False,
            "unembedding_projection_is_mechanism": False,
            "dominant_node_is_effective_neuron_set": False,
            "cross_task_adjusted_alignment_is_causal": False,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    valid = bool(
        len(registered) == 720 and len(case_rows) == 720 and len(nodes) == expected_nodes
        and all(row["valid"] for row in completions)
    )
    summary["denominator"]["atlas_valid"] = valid
    write_jsonl(root / "phase347_natural_physical_nodes.jsonl", nodes)
    write_jsonl(root / "phase347_dominant_natural_regions.jsonl", dominant)
    write_jsonl(root / "phase347_cross_model_convergence.jsonl", convergence)
    write_json(root / "phase347_global_summary.json", summary)

    report = [
        "# Phase347 Natural Physical Trace Atlas", "",
        f"- Registered/case rows: {len(registered)}/{len(case_rows)}",
        f"- Raw trace rows: {len(traces)}",
        f"- Fixed physical nodes: {len(nodes)}/{expected_nodes}",
        f"- Non-finite cases: {summary['denominator']['nonfinite_case_count']}",
        f"- Cross-model stage-role agreement: {summary['results']['cross_model_stage_role_agreement_task_count']}/10",
        f"- Cross-model exact-node agreement: {summary['results']['cross_model_exact_node_agreement_task_count']}/10",
        "", "Natural dominance is a search prior only. No causal or neuron-level intervention was executed.",
    ]
    (root / "phase347_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
