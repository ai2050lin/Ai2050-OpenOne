#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 264
SOURCE_PHASES = [261, 262, 263]
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase264_language_pattern_physical_path_atlas")
ROUND_DEFAULT = "language_pattern_physical_path_atlas"

P261_ROOT = Path("tests/result/phase261_stop_continuation_competition_atlas/stop_continuation_competition_atlas")
P262_ROOT = Path("tests/result/phase262_continuation_regime_decomposition_atlas/continuation_regime_decomposition_atlas")
P263_ROOT = Path("tests/result/phase263_continuation_suppression_candidate_causal_audit/continuation_suppression_candidate_causal_audit")


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


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def path_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("case_id", "")),
        str(row.get("variant_id", "")),
        str(row.get("mode_id", "")),
        str(row.get("condition", "")),
        str(row.get("regime_id", "plain")),
    )


def family_from_mode(mode_id: str) -> str:
    mode = str(mode_id)
    if "json" in mode or "list" in mode or "answer" in mode or "protocol" in mode:
        return "output_protocol"
    if "explain" in mode or "reason" in mode:
        return "reasoning_constraint"
    return "readout_competition"


def state_factor_from_condition(condition: str, regime_id: str, source: str) -> dict[str, float]:
    condition = str(condition)
    regime_id = str(regime_id)
    source = str(source)
    return {
        "S_template": 1.0 if "template_complete" in condition else 0.0,
        "S_semantic": 1.0 if "semantic_correct" in condition else 0.0,
        "S_boundary": 1.0 if "boundary_complete" in condition or "boundary" in regime_id else 0.0,
        "S_protocol": 1.0 if regime_id in {"answer_anchor", "json_structure", "list_item"} else 0.0,
        "S_structure": 1.0 if regime_id in {"json_structure", "list_item"} or "structured" in source else 0.0,
        "S_continue": 1.0 if source else 0.0,
        "S_stop": 0.0,
        "S_done": 1.0 if ("complete" in condition or regime_id in {"period_boundary", "newline_boundary"}) else 0.0,
    }


def closure_class(stop_margin: float, winner: str, rollout_stop: bool | None = None) -> str:
    if rollout_stop:
        return "rollout_stop_candidate"
    if winner == "stop" or stop_margin > 0:
        return "static_stop_winner"
    if stop_margin > -2:
        return "near_threshold_continue"
    return "continuation_dominant"


def build_atlas(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    p261_competitions = read_jsonl(P261_ROOT / "phase261_competition_rows.jsonl")
    p262_sources = read_jsonl(P262_ROOT / "phase262_continuation_source_map_rows.jsonl")
    p263_effects = read_jsonl(P263_ROOT / "phase263_channel_causal_effect_rows.jsonl")
    p263_rollouts = read_jsonl(P263_ROOT / "phase263_rollout_probe_rows.jsonl")

    p261_by_key = {path_key(row): row for row in p261_competitions}
    p263_by_key: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in p263_effects:
        p263_by_key[path_key(row)].append(row)
    rollout_by_key: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in p263_rollouts:
        rollout_by_key[path_key(row)].append(row)

    case_bank_rows: list[dict[str, Any]] = []
    path_signature_rows: list[dict[str, Any]] = []
    state_factor_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    cluster_acc: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in p262_sources:
        key = path_key(row)
        model, case_id, variant_id, mode_id, condition, regime_id = key
        source = str(row.get("source_hypothesis", "unknown_source"))
        family_id = family_from_mode(mode_id)
        top_channel = str(row.get("top_continue_channel", "unknown_channel"))
        stop_margin = safe_float(row.get("stop_continue_margin"))
        top_margin = safe_float(row.get("top_continue_vs_stop_margin"))
        winner = str(row.get("competition_winner", "unknown"))
        p261_row = p261_by_key.get((model, case_id, variant_id, mode_id, condition, "plain"), {})
        effects = p263_by_key.get(key, [])
        rollouts = rollout_by_key.get(key, [])
        best_effect = max((safe_float(e.get("stop_margin_delta")) for e in effects), default=0.0)
        best_policy = ""
        if effects:
            best = max(effects, key=lambda e: safe_float(e.get("stop_margin_delta")))
            best_policy = str(best.get("policy_id", ""))
        rollout_stop = any(bool(r.get("model_stop_executed")) for r in rollouts)
        rollout_mean_tokens = round(mean(safe_float(r.get("generated_token_count")) for r in rollouts), 6) if rollouts else 0.0
        state = state_factor_from_condition(condition, regime_id, source)
        path_id = f"phase264:path:{model}:{case_id}:{variant_id}:{condition}:{regime_id}"
        path_class = closure_class(stop_margin, winner, rollout_stop)
        cluster_key = (family_id, source, top_channel, path_class)
        cluster_acc[cluster_key].append(
            {
                "path_id": path_id,
                "model": model,
                "stop_continue_margin": stop_margin,
                "top_continue_vs_stop_margin": top_margin,
                "best_stop_margin_delta": best_effect,
                "rollout_stop": rollout_stop,
            }
        )

        case_bank_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "case_bank_id": f"phase264:case:{model}:{case_id}:{variant_id}:{condition}:{regime_id}",
                "model": model,
                "case_id": case_id,
                "variant_id": variant_id,
                "mode_id": mode_id,
                "family_id": family_id,
                "condition": condition,
                "regime_id": regime_id,
                "source_hypothesis": source,
                "top_continue_channel": top_channel,
                "quality_status": "usable_path_observation",
                "small_model_bias_warning": True,
            }
        )
        path_signature_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "path_id": path_id,
                "model": model,
                "case_id": case_id,
                "variant_id": variant_id,
                "mode_id": mode_id,
                "family_id": family_id,
                "condition": condition,
                "regime_id": regime_id,
                "state_path": state,
                "component_path": ["final_hidden", "lm_head_readout"],
                "readout_winner": winner,
                "readout_winner_channel": top_channel,
                "source_hypothesis": source,
                "rollout_class": "rollout_stop_seen" if rollout_stop else ("rollout_not_tested" if not rollouts else "rollout_no_stop"),
                "closure_class": path_class,
                "phase261_plain_margin": p261_row.get("stop_continue_margin"),
                "phase262_margin": stop_margin,
                "phase263_best_stop_margin_delta": round(best_effect, 6),
                "phase263_best_policy": best_policy,
                "evidence_level": evidence_level(row, effects, rollouts),
            }
        )
        state_factor_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "state_factor_id": f"phase264:state:{model}:{case_id}:{variant_id}:{condition}:{regime_id}",
                "path_id": path_id,
                "model": model,
                "family_id": family_id,
                "mode_id": mode_id,
                "condition": condition,
                "regime_id": regime_id,
                **state,
                "state_factor_note": "coordinate labels from observed condition/regime, not orthogonal true factors",
            }
        )
        readout_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "readout_path_id": f"phase264:readout:{model}:{case_id}:{variant_id}:{condition}:{regime_id}",
                "path_id": path_id,
                "model": model,
                "family_id": family_id,
                "mode_id": mode_id,
                "condition": condition,
                "regime_id": regime_id,
                "r_stop": row.get("r_stop"),
                "r_continue": row.get("r_continue"),
                "stop_continue_margin": stop_margin,
                "top_continue_channel": top_channel,
                "top_continue_vs_stop_margin": top_margin,
                "competition_winner": winner,
                "source_hypothesis": source,
            }
        )
        if rollouts:
            for rollout in rollouts:
                rollout_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase264",
                        "created_at": utc_now(),
                        "rollout_path_id": f"phase264:{rollout.get('rollout_id')}",
                        "path_id": path_id,
                        "model": model,
                        "family_id": family_id,
                        "policy_id": rollout.get("policy_id"),
                        "generated_token_count": rollout.get("generated_token_count"),
                        "model_stop_executed": rollout.get("model_stop_executed"),
                        "alias_hit": rollout.get("alias_hit"),
                        "has_because": rollout.get("has_because"),
                        "has_list_marker": rollout.get("has_list_marker"),
                        "rollout_mean_tokens_for_path": rollout_mean_tokens,
                    }
                )
        if best_effect > 2.0 or rollout_stop or path_class == "near_threshold_continue":
            candidate_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase264",
                    "created_at": utc_now(),
                    "candidate_id": f"phase264:candidate:{model}:{case_id}:{variant_id}:{condition}:{regime_id}",
                    "path_id": path_id,
                    "model": model,
                    "family_id": family_id,
                    "mode_id": mode_id,
                    "condition": condition,
                    "regime_id": regime_id,
                    "source_hypothesis": source,
                    "top_continue_channel": top_channel,
                    "candidate_type": "stable_path_for_causal_audit" if best_effect > 2.0 else "near_threshold_path",
                    "best_stop_margin_delta": round(best_effect, 6),
                    "rollout_stop_seen": rollout_stop,
                    "priority_score": round(best_effect + (4.0 if rollout_stop else 0.0) - max(top_margin, 0) * 0.05, 6),
                    "status": "candidate_not_closure",
                }
            )
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "observation_id": path_id,
                "case_id": case_id,
                "model": model,
                "family_id": family_id,
                "mode_id": mode_id,
                "variant_id": variant_id,
                "level": "physical_path_atlas",
                "component": source,
                "metric_name": "stop_continue_margin",
                "metric_value": stop_margin,
                "metric_unit": "logit",
                "winner": winner,
                "path_class": path_class,
            }
        )

    for (family_id, source, channel, path_class), items in cluster_acc.items():
        models = Counter(str(i["model"]) for i in items)
        cluster_id = f"phase264:cluster:{family_id}:{source}:{channel}:{path_class}"
        cluster_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "cluster_id": cluster_id,
                "family_id": family_id,
                "source_hypothesis": source,
                "top_continue_channel": channel,
                "closure_class": path_class,
                "path_count": len(items),
                "model_counts": dict(models),
                "cross_model_coverage": len(models),
                "mean_stop_continue_margin": round(mean(safe_float(i["stop_continue_margin"]) for i in items), 6),
                "mean_top_continue_vs_stop_margin": round(mean(safe_float(i["top_continue_vs_stop_margin"]) for i in items), 6),
                "mean_best_stop_margin_delta": round(mean(safe_float(i["best_stop_margin_delta"]) for i in items), 6),
                "rollout_stop_count": sum(1 for i in items if i["rollout_stop"]),
                "stability_status": "cross_model_stable" if len(models) >= 2 and len(items) >= 10 else "local_cluster",
            }
        )
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "edge_id": f"phase264:edge:{family_id}:{source}:{channel}:{path_class}",
                "source": f"node:{family_id}",
                "target": f"node:{channel}",
                "edge_type": "physical_path_cluster_competition",
                "evidence_type": "aggregated_phase261_262_263_path_atlas",
                "effect_size": round(mean(safe_float(i["top_continue_vs_stop_margin"]) for i in items), 6),
                "supporting_phases": ["Phase261", "Phase262", "Phase263", "Phase264"],
                "status": "path_cluster_not_final_mechanism",
            }
        )

    metrics.extend(make_metrics(path_signature_rows, cluster_rows, candidate_rows))
    report = make_report(path_signature_rows, cluster_rows, candidate_rows, metrics)

    write_jsonl(out_dir / "phase264_mode_family_case_bank_v3.jsonl", case_bank_rows)
    write_jsonl(out_dir / "phase264_internal_path_trace_rows.jsonl", path_signature_rows)
    write_jsonl(out_dir / "phase264_state_factor_projection_rows.jsonl", state_factor_rows)
    write_jsonl(out_dir / "phase264_readout_competition_rows.jsonl", readout_rows)
    write_jsonl(out_dir / "phase264_rollout_trace_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase264_path_cluster_rows.jsonl", cluster_rows)
    write_jsonl(out_dir / "phase264_mechanism_candidate_rows.jsonl", candidate_rows)
    write_jsonl(out_dir / "phase264_observations.jsonl", observations)
    write_jsonl(out_dir / "phase264_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase264_graph_edges.jsonl", edges)
    (out_dir / "phase264_language_pattern_physical_path_atlas_report.md").write_text(report, encoding="utf-8")

    progress = {
        "pattern_family_atlas": 0.86,
        "physical_path_atlas": 0.24,
        "state_factor_atlas": 0.34,
        "path_cluster_mining": 0.12,
        "trace_signature_validation": 0.47,
        "readout_competition_trace": 0.78,
        "stepwise_rollout_trace": 0.43,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.66,
    }
    summary = {
        "phase": PHASE,
        "source_phases": SOURCE_PHASES,
        "title": "Language pattern physical path atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "case_bank_rows": len(case_bank_rows),
        "path_signature_rows": len(path_signature_rows),
        "state_factor_rows": len(state_factor_rows),
        "readout_rows": len(readout_rows),
        "rollout_rows": len(rollout_rows),
        "path_cluster_rows": len(cluster_rows),
        "mechanism_candidate_rows": len(candidate_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "top_clusters": top_clusters(cluster_rows),
        "top_candidates": top_candidates(candidate_rows),
        "progress": progress,
    }
    write_json(out_dir / "phase264_cross_model_summary.json", summary)

    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase264", "updated_at": utc_now()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evidence_level(row: dict[str, Any], effects: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> int:
    level = 1
    if safe_float(row.get("top_continue_vs_stop_margin")) > 0:
        level = 2
    if effects:
        level = 4
    if any(bool(r.get("model_stop_executed")) for r in rollouts):
        level = 5
    return level


def make_metrics(paths: list[dict[str, Any]], clusters: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, value in mean_by(paths, "family_id", "phase262_margin").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase264",
                "created_at": utc_now(),
                "metric_id": f"phase264:family:{family}:mean_stop_continue_margin",
                "scope": "physical_path_atlas",
                "family_id": family,
                "metric_name": "mean_stop_continue_margin",
                "metric_value": value,
                "rows": sum(1 for p in paths if p.get("family_id") == family),
            }
        )
    stable_clusters = sum(1 for c in clusters if c.get("stability_status") == "cross_model_stable")
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase264",
            "created_at": utc_now(),
            "metric_id": "phase264:path_clusters:cross_model_stable_count",
            "scope": "path_cluster_mining",
            "metric_name": "cross_model_stable_cluster_count",
            "metric_value": stable_clusters,
            "rows": len(clusters),
        }
    )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase264",
            "created_at": utc_now(),
            "metric_id": "phase264:mechanism_candidates:count",
            "scope": "mechanism_candidate_mining",
            "metric_name": "mechanism_candidate_count",
            "metric_value": len(candidates),
            "rows": len(candidates),
        }
    )
    return rows


def top_clusters(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (int(r.get("cross_model_coverage", 0)), int(r.get("path_count", 0)), safe_float(r.get("mean_best_stop_margin_delta"))), reverse=True)
    return ranked[:limit]


def top_candidates(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: safe_float(r.get("priority_score")), reverse=True)
    return ranked[:limit]


def make_report(paths: list[dict[str, Any]], clusters: list[dict[str, Any]], candidates: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase264 Language Pattern Physical Path Atlas",
        "",
        f"- path_signature_rows: {len(paths)}",
        f"- path_cluster_rows: {len(clusters)}",
        f"- mechanism_candidate_rows: {len(candidates)}",
        f"- metrics: {json.dumps(metrics, ensure_ascii=False)}",
        f"- top_clusters: {json.dumps(top_clusters(clusters, 5), ensure_ascii=False)}",
        f"- top_candidates: {json.dumps(top_candidates(candidates, 5), ensure_ascii=False)}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    args = parser.parse_args()
    build_atlas(args.round_name)


if __name__ == "__main__":
    import argparse

    main()
