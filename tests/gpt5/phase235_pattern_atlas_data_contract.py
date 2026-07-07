#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = 235
SOURCE_PHASE = 234
SCHEMA_VERSION = "1.0.0"
INPUT_ROOT = Path("tests/result/phase234_pattern_family_atlas_matrix/pattern_family_atlas_matrix")
RESULT_ROOT = Path("tests/result/pattern_family_atlas")


STATUS_ORDER = [
    "not_started",
    "behavior_tested",
    "trigger_mapped",
    "state_mapped",
    "readout_mapped",
    "source_candidate",
    "hook_supported",
    "closure_candidate",
    "closed",
    "failed_or_deprioritized",
]

EVIDENCE_WEIGHTS = {
    "behavior_consistency": 0.15,
    "state_consistency": 0.15,
    "readout_consistency": 0.20,
    "hook_causal_support": 0.25,
    "closure_support": 0.15,
    "cross_model_consistency": 0.10,
}

KNOWN_EVIDENCE = [
    {
        "edge_id": "edge_glm4_no_answer_anchor_for_continuation",
        "source": "mode:output_protocol:answer_boundary",
        "target": "mode:readout_competition:for_continuation",
        "edge_type": "prompt_anchor_to_regime_switch",
        "family_id": "readout_competition",
        "mode_id": "for_continuation",
        "model": "glm4",
        "evidence_type": "hook_level_causal_contribution",
        "effect_direction": "positive",
        "effect_size": 4.4529,
        "confidence": 0.72,
        "supporting_phases": ["Phase229", "Phase230", "Phase232", "Phase233"],
        "status": "hook_supported",
    },
    {
        "edge_id": "edge_glm4_explain_instruction_because",
        "source": "mode:output_protocol:explain_answer",
        "target": "mode:readout_competition:because_reason",
        "edge_type": "instruction_to_competitor_pressure",
        "family_id": "readout_competition",
        "mode_id": "because_reason",
        "model": "glm4",
        "evidence_type": "hook_level_causal_contribution",
        "effect_direction": "positive",
        "effect_size": 1.9688,
        "confidence": 0.60,
        "supporting_phases": ["Phase232", "Phase233"],
        "status": "hook_supported",
    },
    {
        "edge_id": "edge_qwen3_no_answer_anchor_because_period",
        "source": "mode:output_protocol:answer_boundary",
        "target": "mode:readout_competition:because_reason",
        "edge_type": "prompt_anchor_to_competitor_pressure",
        "family_id": "readout_competition",
        "mode_id": "because_reason",
        "model": "qwen3",
        "evidence_type": "hook_level_causal_contribution",
        "effect_direction": "positive",
        "effect_size": 0.6844,
        "confidence": 0.56,
        "supporting_phases": ["Phase231", "Phase232", "Phase233"],
        "status": "hook_supported",
    },
    {
        "edge_id": "edge_qwen3_period_second_takeover",
        "source": "mode:readout_competition:period_stop",
        "target": "mode:readout_competition:second_competitor_takeover",
        "edge_type": "suppression_to_takeover",
        "family_id": "readout_competition",
        "mode_id": "second_competitor_takeover",
        "model": "qwen3",
        "evidence_type": "observed_after_source_suppression",
        "effect_direction": "mixed",
        "effect_size": 0.5,
        "confidence": 0.50,
        "supporting_phases": ["Phase233", "Phase234"],
        "status": "readout_mapped",
    },
    {
        "edge_id": "edge_deepseek7b_be_continuation_candidate",
        "source": "mode:output_protocol:short_answer",
        "target": "mode:readout_competition:be_continuation",
        "edge_type": "weak_product_coupling",
        "family_id": "readout_competition",
        "mode_id": "be_continuation",
        "model": "deepseek7b",
        "evidence_type": "weak_hook_signal",
        "effect_direction": "positive",
        "effect_size": 0.1836,
        "confidence": 0.32,
        "supporting_phases": ["Phase232", "Phase233"],
        "status": "source_candidate",
    },
]


def utc_now() -> str:
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evidence_score(parts: dict[str, float]) -> float:
    score = 0.0
    for key, weight in EVIDENCE_WEIGHTS.items():
        score += weight * max(0.0, min(1.0, float(parts.get(key, 0.0))))
    return round(score, 4)


def schema_payload() -> dict[str, Any]:
    common_required = ["schema_version", "phase_id", "created_at"]
    return {
        "schema_version": SCHEMA_VERSION,
        "program_id": "pattern_family_atlas_v1",
        "record_types": {
            "family": {
                "required": common_required + ["family_id", "family_name", "priority", "status", "progress"],
            },
            "mode": {
                "required": common_required + ["mode_id", "family_id", "mode_name", "status", "progress", "expected_internal_path"],
            },
            "test_case": {
                "required": common_required + ["case_id", "family_id", "mode_id", "prompt", "target", "expected_pattern"],
            },
            "run": {
                "required": common_required + ["run_id", "model", "status", "started_at"],
            },
            "observation": {
                "required": common_required
                + [
                    "observation_id",
                    "run_id",
                    "case_id",
                    "model",
                    "family_id",
                    "mode_id",
                    "level",
                    "metric_name",
                    "metric_value",
                ],
            },
            "metric": {
                "required": common_required + ["metric_id", "scope", "metric_name", "metric_value"],
            },
            "graph_node": {
                "required": common_required + ["node_id", "node_type", "label", "status", "evidence_score", "progress"],
            },
            "graph_edge": {
                "required": common_required + ["edge_id", "source", "target", "edge_type", "evidence_type", "confidence"],
            },
        },
        "status_order": STATUS_ORDER,
        "evidence_weights": EVIDENCE_WEIGHTS,
    }


def normalize_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    progress_by_family = {
        "content_knowledge": 0.45,
        "output_protocol": 0.50,
        "reasoning_constraint": 0.12,
        "syntax_structure": 0.15,
        "language_action": 0.10,
        "cross_lingual": 0.10,
        "readout_competition": 0.32,
        "state_drift": 0.18,
        "closure": 0.12,
    }
    out = []
    for row in rows:
        family_id = row["family_id"]
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase235",
                "created_at": utc_now(),
                "family_id": family_id,
                "family_name": row.get("family_name", family_id),
                "priority": "high" if int(row.get("priority") or 3) == 1 else "medium" if int(row.get("priority") or 3) == 2 else "low",
                "description": row.get("core_question", ""),
                "core_questions": [row.get("core_question", "")],
                "mechanism_hypothesis": row.get("mechanism_hypothesis", ""),
                "mode_count": int(row.get("mode_count") or 0),
                "status": "active",
                "progress": progress_by_family.get(family_id, 0.05),
                "related_phases": ["Phase234", "Phase235"],
            }
        )
    return out


def normalize_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {
        "because_reason": ("hook_supported", 0.28),
        "period_stop": ("readout_mapped", 0.22),
        "for_continuation": ("hook_supported", 0.35),
        "be_continuation": ("source_candidate", 0.16),
        "second_competitor_takeover": ("readout_mapped", 0.18),
        "object_relation_value": ("behavior_tested", 0.30),
        "explain_answer": ("hook_supported", 0.38),
        "repeat_answer": ("hook_supported", 0.38),
        "model_close": ("not_started", 0.08),
    }
    out = []
    for row in rows:
        mode_id = row["mode_id"]
        status, progress = known.get(mode_id, ("not_started", 0.02))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase235",
                "created_at": utc_now(),
                "mode_id": mode_id,
                "family_id": row["family_id"],
                "mode_name": mode_id,
                "mode_type": "mechanism_mode",
                "hypothesis": row.get("mechanism_hypothesis", ""),
                "expected_internal_path": [
                    "PromptPattern",
                    "GateUpProduct",
                    "ResidualWrite",
                    "ReadoutCompetition",
                    "RolloutPattern",
                ],
                "known_evidence": {},
                "status": status,
                "progress": progress,
                "required_levels": row.get("required_levels", []),
            }
        )
    return out


def normalize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase235",
                "created_at": utc_now(),
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "object": row.get("object", ""),
                "relation": row.get("relation", ""),
                "target": row.get("target", ""),
                "output_protocol": row.get("protocol_id", ""),
                "prompt_template_id": f"{row.get('mode_id')}_{row.get('protocol_id')}_v1",
                "prompt": row.get("prompt", ""),
                "expected_pattern": row.get("expected_pattern", ""),
                "expected_answer": row.get("target", ""),
                "tags": [row.get("family_id", ""), row.get("mode_id", ""), row.get("protocol_id", "")],
                "difficulty": "seed",
                "test_levels": row.get("test_levels", []),
            }
        )
    return out


def graph_nodes(families: list[dict[str, Any]], modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for family in families:
        score = evidence_score(
            {
                "behavior_consistency": family["progress"],
                "state_consistency": 0.2 if family["family_id"] in {"content_knowledge", "output_protocol", "readout_competition"} else 0.05,
                "readout_consistency": family["progress"],
                "hook_causal_support": 0.35 if family["family_id"] in {"output_protocol", "readout_competition"} else 0.0,
                "closure_support": 0.0,
                "cross_model_consistency": 0.25,
            }
        )
        nodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase235",
                "created_at": utc_now(),
                "node_id": f"family:{family['family_id']}",
                "node_type": "pattern_family",
                "label": family["family_name"],
                "family_id": family["family_id"],
                "mode_id": "",
                "model": "",
                "layer": None,
                "component": "",
                "status": family["status"],
                "evidence_score": score,
                "progress": family["progress"],
            }
        )
    for mode in modes:
        score = evidence_score(
            {
                "behavior_consistency": mode["progress"],
                "state_consistency": mode["progress"] if mode["status"] in {"source_candidate", "hook_supported"} else 0.0,
                "readout_consistency": mode["progress"] if mode["family_id"] == "readout_competition" else 0.05,
                "hook_causal_support": 0.7 if mode["status"] == "hook_supported" else 0.2 if mode["status"] == "source_candidate" else 0.0,
                "closure_support": 0.0,
                "cross_model_consistency": 0.25,
            }
        )
        nodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase235",
                "created_at": utc_now(),
                "node_id": f"mode:{mode['family_id']}:{mode['mode_id']}",
                "node_type": "pattern_mode",
                "label": mode["mode_name"],
                "family_id": mode["family_id"],
                "mode_id": mode["mode_id"],
                "model": "",
                "layer": None,
                "component": "",
                "status": mode["status"],
                "evidence_score": score,
                "progress": mode["progress"],
            }
        )
    return nodes


def graph_edges() -> list[dict[str, Any]]:
    rows = []
    for edge in KNOWN_EVIDENCE:
        item = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            **edge,
        }
        rows.append(item)
    return rows


def progress_payload(families: list[dict[str, Any]], modes: list[dict[str, Any]]) -> dict[str, Any]:
    levels = {
        "behavior": 0.30,
        "prompt_trigger": 0.22,
        "gate_up_product": 0.20,
        "residual_state": 0.25,
        "readout_competition": 0.32,
        "competitor_source": 0.18,
        "rollout": 0.05,
        "closure": 0.10,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase235",
        "created_at": utc_now(),
        "global_progress": {
            "pattern_family_atlas": 0.34,
            "model_internal_closure": 0.46,
            "general_language_mechanism_confidence": 0.43,
        },
        "families": {row["family_id"]: row["progress"] for row in families},
        "levels": levels,
        "modes": {row["mode_id"]: row["progress"] for row in modes},
        "next_phase": "Phase236_behavior_family_benchmark",
    }


def seed_runs() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            "run_id": "run_phase235_contract_seed",
            "model": "none",
            "status": "completed",
            "started_at": utc_now(),
            "completed_at": utc_now(),
            "description": "Data contract and client index seed package. No model inference.",
        }
    ]


def seed_metrics(families: list[dict[str, Any]], modes: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            "metric_id": "metric_family_count",
            "scope": "program",
            "metric_name": "family_count",
            "metric_value": len(families),
        },
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            "metric_id": "metric_mode_count",
            "scope": "program",
            "metric_name": "mode_count",
            "metric_value": len(modes),
        },
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            "metric_id": "metric_seed_case_count",
            "scope": "program",
            "metric_name": "seed_test_case_count",
            "metric_value": len(cases),
        },
    ]


def seed_observations() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase235",
            "created_at": utc_now(),
            "observation_id": "obs_contract_placeholder_0001",
            "run_id": "run_phase235_contract_seed",
            "case_id": "",
            "model": "none",
            "family_id": "readout_competition",
            "mode_id": "second_competitor_takeover",
            "level": "client_contract",
            "step": None,
            "layer": None,
            "component": "",
            "prompt_variant": "",
            "metric_name": "schema_ready",
            "metric_value": 1.0,
            "winner_regime": "",
            "second_competitor": "",
            "target_rank": None,
            "target_logit": None,
            "top_token": "",
            "behavior_pattern": "",
            "closure_flags": {
                "answer_correct": False,
                "pattern_matched": False,
                "boundary_stable": False,
                "done_state_stable": False,
                "model_stop_executed": False,
                "no_drift": False,
            },
        }
    ]


def manifest_payload(families: list[dict[str, Any]], modes: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "program_id": "pattern_family_atlas_v1",
        "phase": "Phase235",
        "created_at": utc_now(),
        "models": ["qwen3", "glm4", "deepseek7b"],
        "families": len(families),
        "modes": len(modes),
        "test_cases": len(cases),
        "levels": 8,
        "schema_version": SCHEMA_VERSION,
        "files": {
            "schema": "schema.json",
            "client_index": "client_index.json",
            "families": "families.jsonl",
            "modes": "modes.jsonl",
            "test_cases": "test_cases.jsonl",
            "runs": "runs.jsonl",
            "observations": "observations.jsonl",
            "metrics": "metrics.jsonl",
            "graph_nodes": "graph_nodes.jsonl",
            "graph_edges": "graph_edges.jsonl",
            "progress": "progress.json",
            "summary": "summary.md",
        },
    }


def client_index_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase235",
        "created_at": utc_now(),
        "entrypoint": "manifest.json",
        "views": [
            {
                "view_id": "pattern_atlas_overview",
                "title": "模式图谱总览",
                "required_files": ["manifest.json", "progress.json", "families.jsonl", "graph_nodes.jsonl", "graph_edges.jsonl"],
                "widgets": ["global_progress", "family_cards", "priority_queue", "recent_phase_updates"],
            },
            {
                "view_id": "family_detail",
                "title": "模式族详情",
                "required_files": ["families.jsonl", "modes.jsonl", "metrics.jsonl", "graph_nodes.jsonl", "graph_edges.jsonl"],
                "widgets": ["mode_table", "evidence_summary", "failure_summary", "next_phase"],
            },
            {
                "view_id": "mode_detail",
                "title": "单模式详情",
                "required_files": ["modes.jsonl", "observations.jsonl", "metrics.jsonl", "graph_edges.jsonl"],
                "widgets": ["behavior_distribution", "readout_margin", "source_evidence", "rollout_trace", "closure_flags"],
            },
            {
                "view_id": "mechanism_graph",
                "title": "机制链路图",
                "required_files": ["graph_nodes.jsonl", "graph_edges.jsonl"],
                "widgets": ["node_graph", "edge_table", "evidence_filter"],
            },
            {
                "view_id": "model_compare",
                "title": "模型对比",
                "required_files": ["observations.jsonl", "metrics.jsonl", "graph_edges.jsonl"],
                "widgets": ["model_matrix", "competitor_distribution", "small_model_bias_warning"],
            },
        ],
    }


def summary_markdown(manifest: dict[str, Any], progress: dict[str, Any], edges: list[dict[str, Any]]) -> str:
    lines = ["# Pattern Family Atlas Data Contract", ""]
    lines.append(f"schema_version: {SCHEMA_VERSION}")
    lines.append(f"phase: {manifest['phase']}")
    lines.append(f"families: {manifest['families']}")
    lines.append(f"modes: {manifest['modes']}")
    lines.append(f"test_cases: {manifest['test_cases']}")
    lines.extend(["", "## Required Files", ""])
    for name, file_name in manifest["files"].items():
        lines.append(f"- {name}: `{file_name}`")
    lines.extend(["", "## Progress", ""])
    for key, value in progress["global_progress"].items():
        lines.append(f"- {key}: {value:.2f}")
    lines.extend(["", "## Known Evidence Edges", "", "| edge | model | type | status | confidence |", "| --- | --- | --- | --- | ---: |"])
    for edge in edges:
        lines.append(f"| {edge['edge_id']} | {edge['model']} | {edge['edge_type']} | {edge['status']} | {edge['confidence']:.2f} |")
    return "\n".join(lines) + "\n"


def build(round_name: str) -> dict[str, Any]:
    src_families = read_jsonl(INPUT_ROOT / "phase234_pattern_family_rows.jsonl")
    src_modes = read_jsonl(INPUT_ROOT / "phase234_pattern_mode_rows.jsonl")
    src_cases = read_jsonl(INPUT_ROOT / "phase234_seed_test_case_rows.jsonl")
    if not src_families or not src_modes or not src_cases:
        raise SystemExit(f"missing Phase234 matrix files under {INPUT_ROOT}")

    out_dir = RESULT_ROOT / round_name
    families = normalize_families(src_families)
    modes = normalize_modes(src_modes)
    cases = normalize_cases(src_cases)
    nodes = graph_nodes(families, modes)
    edges = graph_edges()
    progress = progress_payload(families, modes)
    manifest = manifest_payload(families, modes, cases)
    client_index = client_index_payload()
    schema = schema_payload()
    runs = seed_runs()
    observations = seed_observations()
    metrics = seed_metrics(families, modes, cases)

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "schema.json", schema)
    write_json(out_dir / "client_index.json", client_index)
    write_jsonl(out_dir / "families.jsonl", families)
    write_jsonl(out_dir / "modes.jsonl", modes)
    write_jsonl(out_dir / "test_cases.jsonl", cases)
    write_jsonl(out_dir / "runs.jsonl", runs)
    write_jsonl(out_dir / "observations.jsonl", observations)
    write_jsonl(out_dir / "metrics.jsonl", metrics)
    write_jsonl(out_dir / "graph_nodes.jsonl", nodes)
    write_jsonl(out_dir / "graph_edges.jsonl", edges)
    write_json(out_dir / "progress.json", progress)
    (out_dir / "summary.md").write_text(summary_markdown(manifest, progress, edges), encoding="utf-8")

    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Pattern atlas data contract and client index",
        "status": "complete",
        "timestamp": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "families": len(families),
        "modes": len(modes),
        "test_cases": len(cases),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "phase235_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase235 pattern atlas data contract")
    parser.add_argument("--round-name", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.round_name)


if __name__ == "__main__":
    main()
