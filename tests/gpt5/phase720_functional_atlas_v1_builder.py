#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "results/glm5_phase712_qkv_factor_atlas_audit/phase712_atlas_units_with_qkv.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("results/glm5_phase720_functional_atlas_v1")


FUNCTION_FAMILIES = [
    {
        "function_family": "object_relation_value_short_answer",
        "current_status": "measured_micro_atlas",
        "current_evidence": "Phase 698-712: source contribution, channel split, phrase likelihood, partial natural generation, QK/V factor backfill",
        "scope": "object relation value route at answer_start",
    },
    {
        "function_family": "fruit_identity_reuse_difference",
        "current_status": "not_measured_yet",
        "current_evidence": "requires new global functional atlas run",
        "scope": "apple/banana/orange/grape/etc. shared and differential route map",
    },
    {
        "function_family": "color_value_reuse_difference",
        "current_status": "not_measured_yet",
        "current_evidence": "requires new global functional atlas run",
        "scope": "red/blue/green/yellow/etc. shared and differential route map",
    },
    {
        "function_family": "translation_language_route",
        "current_status": "not_measured_yet",
        "current_evidence": "requires new global functional atlas run",
        "scope": "source language, target language, lexical value, grammar/format route split",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def unit_graph_level(row: dict[str, Any]) -> str:
    if row.get("unit_type") == "attention_channel":
        return "channel_bridge"
    if row.get("unit_type") == "attention_head":
        return "head_route"
    return "unknown_unit"


def next_drilldown(row: dict[str, Any]) -> str:
    if row.get("unit_type") == "attention_head":
        return "decompose into source-restricted channels, QK addressing, V content, W_O readout direction, downstream MLP response"
    if row.get("unit_type") == "attention_channel":
        return "test channel necessity, channel sufficiency, downstream MLP/nonlinear gate, and token-level transfer"
    return "inspect unit definition"


def atlas_node(row: dict[str, Any]) -> dict[str, Any]:
    qkv = row.get("phase712_qkv_factor") or {}
    role = row.get("role_scores") or {}
    return {
        "phase": 720,
        "source_phase": "711+712",
        "function_family": "object_relation_value_short_answer",
        "capability_axis": "semantic_value_route",
        "model": row.get("model"),
        "unit_id": row.get("unit_id"),
        "unit_type": row.get("unit_type"),
        "graph_level": unit_graph_level(row),
        "layer": row.get("layer"),
        "head": row.get("head"),
        "channel": row.get("channel"),
        "source_group": row.get("source_group"),
        "target_position": row.get("target_position"),
        "route_role": row.get("status"),
        "cross_model_status": row.get("cross_model_status"),
        "evidence_level": row.get("evidence_level"),
        "qkv_dominant_factor": qkv.get("qkv_dominant_factor"),
        "qkv_abs_qk_share": as_float(qkv.get("qkv_abs_qk_share")),
        "qkv_abs_v_share": as_float(qkv.get("qkv_abs_v_share")),
        "qkv_abs_interaction_share": as_float(qkv.get("qkv_abs_interaction_share")),
        "qkv_sum_total_direct": as_float(qkv.get("qkv_sum_total_direct")),
        "route_gain_score": as_float(role.get("route_gain_score")),
        "identity_score": as_float(role.get("identity_score")),
        "format_or_prose_score": as_float(role.get("format_or_prose_score")),
        "donor_residue_score": as_float(role.get("donor_residue_score")),
        "phrase_target_minus_donor": as_float(role.get("phrase_target_minus_donor")),
        "phrase_target_minus_prose": as_float(role.get("phrase_target_minus_prose")),
        "closure_status": "partial_natural_generation" if row.get("evidence_level") == "level6_partial_natural_generation" else "non_generation_or_weak",
        "interpretation": "current evidence supports a route/carrier role, not a complete semantic neuron identity claim",
        "next_drilldown": next_drilldown(row),
    }


def top_units(nodes: list[dict[str, Any]], *, unit_type: str, n: int = 12) -> list[dict[str, Any]]:
    rows = [r for r in nodes if r.get("unit_type") == unit_type]
    rows.sort(
        key=lambda r: (
            as_float(r.get("route_gain_score")),
            abs(as_float(r.get("qkv_sum_total_direct"))),
            as_float(r.get("identity_score")),
        ),
        reverse=True,
    )
    keep = []
    for r in rows[:n]:
        keep.append(
            {
                "model": r.get("model"),
                "unit_id": r.get("unit_id"),
                "route_role": r.get("route_role"),
                "evidence_level": r.get("evidence_level"),
                "qkv_dominant_factor": r.get("qkv_dominant_factor"),
                "qkv_abs_qk_share": r.get("qkv_abs_qk_share"),
                "qkv_abs_v_share": r.get("qkv_abs_v_share"),
                "qkv_sum_total_direct": r.get("qkv_sum_total_direct"),
                "route_gain_score": r.get("route_gain_score"),
                "identity_score": r.get("identity_score"),
                "format_or_prose_score": r.get("format_or_prose_score"),
            }
        )
    return keep


def top_units_by_model(nodes: list[dict[str, Any]], *, unit_type: str, n: int = 6) -> dict[str, list[dict[str, Any]]]:
    models = sorted({str(r.get("model")) for r in nodes if r.get("model")})
    return {
        model: top_units(
            [r for r in nodes if str(r.get("model")) == model],
            unit_type=unit_type,
            n=n,
        )
        for model in models
    }


def summarize(nodes: list[dict[str, Any]], input_path: Path) -> dict[str, Any]:
    by_model = Counter(r.get("model") for r in nodes)
    by_unit_type = Counter(r.get("unit_type") for r in nodes)
    by_graph_level = Counter(r.get("graph_level") for r in nodes)
    by_route_role = Counter(r.get("route_role") for r in nodes)
    by_factor = Counter(r.get("qkv_dominant_factor") for r in nodes)
    by_model_factor: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_role: dict[str, Counter[str]] = defaultdict(Counter)
    for r in nodes:
        by_model_factor[str(r.get("model"))][str(r.get("qkv_dominant_factor"))] += 1
        by_model_role[str(r.get("model"))][str(r.get("route_role"))] += 1

    measured_models = sorted(k for k in by_model if k)
    return {
        "phase": 720,
        "title": "Functional Atlas v1 Readiness and Head-to-Neuron Bridge",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_path),
        "n_nodes": len(nodes),
        "by_model": dict(by_model),
        "by_unit_type": dict(by_unit_type),
        "by_graph_level": dict(by_graph_level),
        "by_route_role": dict(by_route_role),
        "by_qkv_dominant_factor": dict(by_factor),
        "by_model_qkv_dominant_factor": {k: dict(v) for k, v in by_model_factor.items()},
        "by_model_route_role": {k: dict(v) for k, v in by_model_role.items()},
        "measured_function_families": ["object_relation_value_short_answer"],
        "unmeasured_function_families": [
            f["function_family"] for f in FUNCTION_FAMILIES if f["current_status"] == "not_measured_yet"
        ],
        "readiness": {
            "head_level_global_atlas": {
                "status": "ready_to_start_v1",
                "reason": "cross-model head candidates, route roles, and QK/V factor labels already exist for one function family",
            },
            "channel_level_bridge": {
                "status": "ready_for_targeted_drilldown",
                "reason": "source-restricted channel units exist, but only inside selected high-value routes",
            },
            "neuron_level_global_atlas": {
                "status": "not_ready_as_full_global_project",
                "reason": "neuron/channel identity is not yet separable from head QK addressing, V content, W_O readout, and downstream MLP effects",
            },
        },
        "small_model_caution": "all conclusions are local to tested small models and may contain architectural or scale-specific bias",
        "top_attention_heads": top_units(nodes, unit_type="attention_head", n=12),
        "top_attention_channels": top_units(nodes, unit_type="attention_channel", n=12),
        "top_attention_heads_by_model": top_units_by_model(nodes, unit_type="attention_head", n=6),
        "top_attention_channels_by_model": top_units_by_model(nodes, unit_type="attention_channel", n=6),
        "function_families": FUNCTION_FAMILIES,
        "next_phase": {
            "phase": 721,
            "title": "Global Functional Head Atlas Data Expansion",
            "same_stage_as_phase720": True,
            "objective": "expand from one measured value-route family to multiple function families before neuron-level global atlas",
            "suggested_function_sets": [
                "fruit identity and category reuse/difference",
                "color value reuse/difference",
                "translation source/target language route",
                "simple grammar protocol route",
            ],
            "minimum_evidence_per_unit": [
                "observational source contribution",
                "head/channel route score",
                "QK/V factor split",
                "causal patch on top units",
                "phrase likelihood or natural generation closure check",
            ],
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 720 Functional Atlas v1 Readiness")
    lines.append("")
    lines.append("## Core Judgment")
    lines.append("")
    lines.append(
        "The uploaded theory direction is basically correct: cracking the encoding mechanism should move from isolated patch tests to a functional atlas. "
        "However, the current evidence supports a head/channel-level route atlas first, not a full neuron-level global atlas."
    )
    lines.append("")
    lines.append("## Objective Results")
    lines.append("")
    lines.append(f"- Nodes built: `{summary['n_nodes']}`")
    lines.append(f"- By model: `{summary['by_model']}`")
    lines.append(f"- By unit type: `{summary['by_unit_type']}`")
    lines.append(f"- QK/V dominant factors: `{summary['by_qkv_dominant_factor']}`")
    lines.append(f"- Measured function families: `{summary['measured_function_families']}`")
    lines.append(f"- Not yet measured: `{summary['unmeasured_function_families']}`")
    lines.append("")
    lines.append("## Feasibility")
    lines.append("")
    for key, item in summary["readiness"].items():
        lines.append(f"- `{key}`: `{item['status']}`. {item['reason']}")
    lines.append("")
    lines.append("## Top Attention Heads")
    lines.append("")
    lines.append("| model | unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in summary["top_attention_heads"]:
        lines.append(
            "| {model} | {unit_id} | {route_role} | {qkv_dominant_factor} | {qkv_abs_qk_share:.3f} | {qkv_abs_v_share:.3f} | {qkv_sum_total_direct:.3f} | {route_gain_score:.3f} | {identity_score:.3f} | {format_or_prose_score:.3f} |".format(
                **r
            )
        )
    lines.append("")
    lines.append("## Top Attention Heads By Model")
    lines.append("")
    for model, rows in summary["top_attention_heads_by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| unit | role | factor | qk_share | v_share | total_direct | route_gain | identity | format/prose |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                "| {unit_id} | {route_role} | {qkv_dominant_factor} | {qkv_abs_qk_share:.3f} | {qkv_abs_v_share:.3f} | {qkv_sum_total_direct:.3f} | {route_gain_score:.3f} | {identity_score:.3f} | {format_or_prose_score:.3f} |".format(
                    **r
                )
            )
        lines.append("")
    lines.append("## Strict Limits")
    lines.append("")
    lines.append("- A head is not a semantic unit. It mixes QK addressing, V content, output projection, residual trajectory, and downstream nonlinear effects.")
    lines.append("- The present atlas covers one measured micro-family: object relation value short-answer route. It does not yet prove apple/red/translation mechanisms.")
    lines.append("- Current models are small; architecture-specific bias and scale effects must be treated as unresolved risks.")
    lines.append("- Neuron-level global atlas should begin only after repeated head/channel patterns are stable across multiple function families.")
    lines.append("")
    lines.append("## Next Phase")
    lines.append("")
    lines.append(
        "Phase 721 should stay in the same atlas-building stage and expand function families before drilling globally to neurons: fruit/category, color, translation, and grammar protocol. "
        "For each family, require observational contribution, QK/V split, causal patch on top units, and generation or phrase-likelihood closure."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_jsonl(args.atlas_path)
    nodes = [atlas_node(row) for row in rows]
    summary = summarize(nodes, args.atlas_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "phase720_functional_atlas_nodes.jsonl", nodes)
    write_json(args.output_dir / "phase720_functional_atlas_summary.json", summary)
    (args.output_dir / "phase720_functional_atlas_report.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
