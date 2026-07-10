#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 288
SCHEMA_VERSION = "2.15.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase288_pattern_atlas_feature_mining"
FILL_PHASES = ["phase275", "phase277", "phase279", "phase283"]
CLOSURE_PHASES = ["phase281", "phase285"]


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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model")), str(row.get("case_id"))


def load_component_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in FILL_PHASES:
        rows.extend(read_jsonl(V2 / f"{phase}_component_summary_rows.jsonl"))
    return rows


def load_causal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in FILL_PHASES:
        rows.extend(read_jsonl(V2 / f"{phase}_causal_fill_rows.jsonl"))
    return rows


def load_closure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in CLOSURE_PHASES:
        rows.extend(read_jsonl(V2 / f"{phase}_closure_quality_rows.jsonl"))
    return rows


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k)) for k in keys)].append(row)
    return out


def detail_readout(row: dict[str, Any]) -> dict[str, Any]:
    detail_ref = row.get("detail_ref")
    if not detail_ref:
        return {}
    return read_json(V2 / str(detail_ref)).get("readout") or {}


def layer_band(layer: Any) -> str:
    value = int(safe_float(layer, -1))
    if value < 0:
        return "unknown"
    if value < 12:
        return "early"
    if value < 28:
        return "middle"
    return "late"


def curve_cluster(row: dict[str, Any]) -> str:
    component = str(row.get("dominant_positive_component"))
    winner = str(row.get("final_winner"))
    mlp_band = layer_band(row.get("strongest_mlp_layer"))
    attn_band = layer_band(row.get("strongest_attn_layer"))
    margin = safe_float(row.get("final_continue_stop_margin"))
    if component == "attention":
        return f"{attn_band}_attention_routed_{winner}"
    if component == "mlp" and margin > 5:
        return f"{mlp_band}_mlp_strong_continue"
    if component == "mlp":
        return f"{mlp_band}_mlp_{winner}"
    return f"{component}_{winner}"


def family_feature_matrix(signatures: list[dict[str, Any]], components: list[dict[str, Any]], causal: list[dict[str, Any]], closure: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comp_g = group(components, "family_id")
    causal_g = group(causal, "family_id")
    closure_g = group(closure, "family_id")
    gap_g = group(gaps, "family_id")
    rows = []
    for (family,), sigs in sorted(group(signatures, "family_id").items()):
        comps = comp_g.get((family,), [])
        crows = causal_g.get((family,), [])
        qrows = closure_g.get((family,), [])
        grows = gap_g.get((family,), [])
        flags = Counter()
        for row in grows:
            for name, value in (row.get("remaining_gap_flags") or {}).items():
                if value:
                    flags[name] += 1
        continue_wins = sum(1 for r in sigs if (r.get("path_signature") or {}).get("readout_winner") == "continue")
        stop_wins = sum(1 for r in sigs if (r.get("path_signature") or {}).get("readout_winner") == "stop")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "feature_row_id": f"phase288:family_feature:{family}",
                "family_id": family,
                "signature_rows": len(sigs),
                "mean_behavior_score": mean_safe([safe_float((r.get("scores") or {}).get("behavior")) for r in sigs]),
                "answer_correct_proxy_rate": rate(sum(1 for r in sigs if safe_float((r.get("scores") or {}).get("behavior")) >= 0.5), len(sigs)),
                "mean_readout_score": mean_safe([safe_float((r.get("scores") or {}).get("readout")) for r in sigs]),
                "mean_rollout_score": mean_safe([safe_float((r.get("scores") or {}).get("rollout")) for r in sigs]),
                "continue_win_rate": rate(continue_wins, len(sigs)),
                "stop_win_rate": rate(stop_wins, len(sigs)),
                "component_summary_rows": len(comps),
                "mlp_dominance_rate": rate(sum(1 for r in comps if r.get("dominant_positive_component") == "mlp"), len(comps)),
                "attention_dominance_rate": rate(sum(1 for r in comps if r.get("dominant_positive_component") == "attention"), len(comps)),
                "mean_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in comps]),
                "mean_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in comps]),
                "causal_rows": len(crows),
                "causal_effect_supported_rate": rate(sum(1 for r in crows if r.get("causal_effect_supported")), len(crows)),
                "side_effect_risk_rate": rate(sum(1 for r in crows if r.get("side_effect_risk")), len(crows)),
                "closure_quality_rows": len(qrows),
                "closure_rejected_rate": rate(sum(1 for r in qrows if not r.get("four_condition_closed")), len(qrows)),
                "stop_not_winner_rate": rate(sum(1 for r in qrows if "stop_not_winner" in (r.get("closure_blockers") or [])), len(qrows)),
                "remaining_need_component_path": flags.get("need_component_path", 0),
                "remaining_need_causal_audit": flags.get("need_causal_audit", 0),
                "remaining_need_closure_quality": flags.get("need_closure_quality", 0),
                "remaining_need_readout_competition": flags.get("need_readout_competition", 0),
            }
        )
    return rows


def model_feature_matrix(signatures: list[dict[str, Any]], components: list[dict[str, Any]], causal: list[dict[str, Any]], closure: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comp_g = group(components, "model")
    causal_g = group(causal, "model")
    closure_g = group(closure, "model")
    gap_g = group(gaps, "model")
    rows = []
    for (model,), sigs in sorted(group(signatures, "model").items()):
        comps = comp_g.get((model,), [])
        crows = causal_g.get((model,), [])
        qrows = closure_g.get((model,), [])
        grows = gap_g.get((model,), [])
        flags = Counter()
        for row in grows:
            for name, value in (row.get("remaining_gap_flags") or {}).items():
                if value:
                    flags[name] += 1
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "feature_row_id": f"phase288:model_feature:{model}",
                "model": model,
                "signature_rows": len(sigs),
                "mean_behavior_score": mean_safe([safe_float((r.get("scores") or {}).get("behavior")) for r in sigs]),
                "continue_win_rate": rate(sum(1 for r in sigs if (r.get("path_signature") or {}).get("readout_winner") == "continue"), len(sigs)),
                "component_summary_rows": len(comps),
                "mlp_dominance_rate": rate(sum(1 for r in comps if r.get("dominant_positive_component") == "mlp"), len(comps)),
                "attention_dominance_rate": rate(sum(1 for r in comps if r.get("dominant_positive_component") == "attention"), len(comps)),
                "causal_rows": len(crows),
                "low_side_effect_supported_rate": rate(sum(1 for r in crows if r.get("side_effect_level") == "lower" and r.get("causal_effect_supported")), sum(1 for r in crows if r.get("side_effect_level") == "lower")),
                "side_effect_risk_rate": rate(sum(1 for r in crows if r.get("side_effect_risk")), len(crows)),
                "closure_quality_rows": len(qrows),
                "closure_rejection_rate": rate(sum(1 for r in qrows if not r.get("four_condition_closed")), len(qrows)),
                "remaining_need_component_path": flags.get("need_component_path", 0),
                "remaining_need_causal_audit": flags.get("need_causal_audit", 0),
                "remaining_need_closure_quality": flags.get("need_closure_quality", 0),
                "remaining_need_readout_competition": flags.get("need_readout_competition", 0),
                "model_specific_risk": "glm4_high_side_effect" if model == "glm4" else "baseline_small_model_bias",
            }
        )
    return rows


def component_distribution_rows(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (family, model, component), bucket in sorted(group(components, "family_id", "model", "dominant_positive_component").items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "component_distribution_id": f"phase288:component:{family}:{model}:{component}",
                "family_id": family,
                "model": model,
                "dominant_positive_component": component,
                "rows": len(bucket),
                "mean_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in bucket]),
                "mean_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in bucket]),
                "mean_final_continue_stop_margin": mean_safe([safe_float(r.get("final_continue_stop_margin")) for r in bucket]),
                "strongest_mlp_layer_counts": dict(Counter(str(r.get("strongest_mlp_layer")) for r in bucket)),
            }
        )
    return rows


def continue_channel_distribution_rows(signatures: list[dict[str, Any]], closure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in signatures:
        readout = detail_readout(row)
        channel = readout.get("top_continue_channel") or (row.get("path_signature") or {}).get("top_competitor") or "unknown"
        observations.append({**row, "top_continue_channel": channel, "top_continue_vs_stop_margin": readout.get("top_continue_vs_stop_margin"), "competition_winner": readout.get("competition_winner") or (row.get("path_signature") or {}).get("readout_winner")})
    for row in closure:
        observations.append(row)
    rows = []
    for (family, model, channel), bucket in sorted(group(observations, "family_id", "model", "top_continue_channel").items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "continue_channel_distribution_id": f"phase288:continue_channel:{family}:{model}:{channel}",
                "family_id": family,
                "model": model,
                "top_continue_channel": channel,
                "rows": len(bucket),
                "continue_winner_rate": rate(sum(1 for r in bucket if r.get("competition_winner") == "continue" or (r.get("path_signature") or {}).get("readout_winner") == "continue"), len(bucket)),
                "mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in bucket if r.get("top_continue_vs_stop_margin") is not None]),
            }
        )
    return rows


def layer_curve_cluster_rows(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in components:
        cluster = curve_cluster(row)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "layer_curve_cluster_id": f"phase288:layer_cluster:{row.get('model')}:{row.get('case_id')}",
                "model": row.get("model"),
                "case_id": row.get("case_id"),
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row.get("variant_id"),
                "cluster_id": cluster,
                "dominant_positive_component": row.get("dominant_positive_component"),
                "strongest_mlp_layer": row.get("strongest_mlp_layer"),
                "strongest_mlp_layer_band": layer_band(row.get("strongest_mlp_layer")),
                "strongest_attn_layer": row.get("strongest_attn_layer"),
                "strongest_attn_layer_band": layer_band(row.get("strongest_attn_layer")),
                "final_winner": row.get("final_winner"),
                "final_continue_stop_margin": row.get("final_continue_stop_margin"),
                "sum_positive_mlp_delta": row.get("sum_positive_mlp_delta"),
                "sum_positive_attn_delta": row.get("sum_positive_attn_delta"),
            }
        )
    return rows


def side_effect_distribution_rows(causal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (model, family, patch, level), bucket in sorted(group(causal, "model", "family_id", "patch_type", "side_effect_level").items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "side_effect_distribution_id": f"phase288:side_effect:{model}:{family}:{patch}:{level}",
                "model": model,
                "family_id": family,
                "patch_type": patch,
                "side_effect_level": level,
                "rows": len(bucket),
                "causal_effect_supported_rate": rate(sum(1 for r in bucket if r.get("causal_effect_supported")), len(bucket)),
                "side_effect_risk_rate": rate(sum(1 for r in bucket if r.get("side_effect_risk")), len(bucket)),
                "winner_changed_rate": rate(sum(1 for r in bucket if r.get("winner_changed")), len(bucket)),
                "mean_delta_continue_stop_margin": mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in bucket]),
                "mean_delta_target_logit": mean_safe([safe_float(r.get("delta_target_logit")) for r in bucket]),
            }
        )
    return rows


def closure_bottleneck_rows(closure: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    expanded: list[dict[str, Any]] = []
    for row in closure:
        blockers = row.get("closure_blockers") or []
        if not blockers:
            blockers = ["none"]
        for blocker in blockers:
            expanded.append({**row, "closure_blocker": blocker})
    for (model, family, blocker), bucket in sorted(group(expanded, "model", "family_id", "closure_blocker").items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "closure_bottleneck_id": f"phase288:closure_bottleneck:{model}:{family}:{blocker}",
                "model": model,
                "family_id": family,
                "closure_blocker": blocker,
                "rows": len(bucket),
                "semantic_done_rate": rate(sum(1 for r in bucket if r.get("semantic_done")), len(bucket)),
                "stop_wins_rate": rate(sum(1 for r in bucket if r.get("stop_wins")), len(bucket)),
                "continue_suppressed_rate": rate(sum(1 for r in bucket if r.get("continue_suppressed")), len(bucket)),
                "rollout_stable_rate": rate(sum(1 for r in bucket if r.get("rollout_stable")), len(bucket)),
            }
        )
    return rows


def gap_heatmap_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (family, model), bucket in sorted(group(gaps, "family_id", "model").items()):
        flags = Counter()
        for row in bucket:
            for name, value in (row.get("remaining_gap_flags") or {}).items():
                if value:
                    flags[name] += 1
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase288",
                "created_at": now(),
                "gap_heatmap_id": f"phase288:gap_heatmap:{family}:{model}",
                "family_id": family,
                "model": model,
                "rows": len(bucket),
                "need_component_path": flags.get("need_component_path", 0),
                "need_causal_audit": flags.get("need_causal_audit", 0),
                "need_layer_path": flags.get("need_layer_path", 0),
                "need_closure_quality": flags.get("need_closure_quality", 0),
                "need_readout_competition": flags.get("need_readout_competition", 0),
                "candidate_not_closed": flags.get("candidate_not_closed", 0),
                "good_behavior_low_path": flags.get("good_behavior_low_path", 0),
                "good_readout_low_causal": flags.get("good_readout_low_causal", 0),
                "open_gap_total": sum(flags.values()),
            }
        )
    return rows


def write_report(summary: dict[str, Any], family_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], cluster_rows: list[dict[str, Any]]) -> None:
    top_families = sorted(family_rows, key=lambda r: r["remaining_need_component_path"] + r["remaining_need_causal_audit"] + r["remaining_need_closure_quality"], reverse=True)[:6]
    cluster_counts = Counter(str(r.get("cluster_id")) for r in cluster_rows)
    lines = [
        "# Phase288 Pattern Atlas Feature Mining",
        "",
        f"- signature_rows: {summary['signature_rows']}",
        f"- component_summary_rows: {summary['component_summary_rows']}",
        f"- causal_rows: {summary['causal_rows']}",
        f"- closure_quality_rows: {summary['closure_quality_rows']}",
        f"- gap_rows: {summary['gap_rows']}",
        f"- global_mlp_dominance_rate: {summary['global_mlp_dominance_rate']}",
        f"- global_continue_win_rate: {summary['global_continue_win_rate']}",
        f"- global_closure_closed_count: {summary['global_closure_closed_count']}",
        "",
        "## Model Matrix",
        "",
    ]
    for row in model_rows:
        lines.append(
            f"- {row['model']}: mlp={row['mlp_dominance_rate']} side_effect={row['side_effect_risk_rate']} "
            f"closure_reject={row['closure_rejection_rate']} need_closure={row['remaining_need_closure_quality']}"
        )
    lines.extend(["", "## Top Gap Families", ""])
    for row in top_families:
        lines.append(
            f"- {row['family_id']}: need_component={row['remaining_need_component_path']} "
            f"need_causal={row['remaining_need_causal_audit']} need_closure={row['remaining_need_closure_quality']}"
        )
    lines.extend(["", "## Layer Clusters", ""])
    for name, count in cluster_counts.most_common(10):
        lines.append(f"- {name}: {count}")
    report = "\n".join(lines) + "\n"
    (OUT / "phase288_feature_mining_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase288_feature_mining_report.md").write_text(report, encoding="utf-8")


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase288_family_feature_matrix",
        "phase288_model_feature_matrix",
        "phase288_component_distribution_rows",
        "phase288_continue_channel_distribution_rows",
        "phase288_layer_curve_cluster_rows",
        "phase288_side_effect_distribution_rows",
        "phase288_closure_bottleneck_rows",
        "phase288_gap_heatmap_rows",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase288_feature_mining_summary"] = "phase288_feature_mining_summary.json"
    files["phase288_feature_mining_report"] = "phase288_feature_mining_report.md"
    manifest["latest_feature_mining_phase"] = "Phase288"
    manifest["phase288_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in [
        "phase288_feature_mining_summary.json",
        "phase288_family_feature_matrix.jsonl",
        "phase288_model_feature_matrix.jsonl",
        "phase288_gap_heatmap_rows.jsonl",
        "phase288_closure_bottleneck_rows.jsonl",
    ]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase288_summary_ref"] = "phase288_feature_mining_summary.json"
    client["phase288_family_feature_matrix_ref"] = "phase288_family_feature_matrix.jsonl"
    client["phase288_model_feature_matrix_ref"] = "phase288_model_feature_matrix.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase288_family_feature_matrix"] = "family-level feature matrix from all current atlas data"
    tables["phase288_model_feature_matrix"] = "model-level feature matrix from all current atlas data"
    tables["phase288_component_distribution_rows"] = "family x model x component distribution"
    tables["phase288_continue_channel_distribution_rows"] = "continue-channel distribution from readout details and closure scans"
    tables["phase288_layer_curve_cluster_rows"] = "coarse layer trajectory clusters from component summaries"
    tables["phase288_side_effect_distribution_rows"] = "side-effect distribution by model, family, patch, level"
    tables["phase288_closure_bottleneck_rows"] = "closure failure bottlenecks by model and family"
    tables["phase288_gap_heatmap_rows"] = "family x model remaining gap heatmap"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    signatures = read_jsonl(V2 / "path_signature_rows.jsonl")
    components = load_component_rows()
    causal = load_causal_rows()
    closure = load_closure_rows()
    gaps = read_jsonl(V2 / "phase286_recalibrated_gap_rows.jsonl")
    family_rows = family_feature_matrix(signatures, components, causal, closure, gaps)
    model_rows = model_feature_matrix(signatures, components, causal, closure, gaps)
    component_rows = component_distribution_rows(components)
    channel_rows = continue_channel_distribution_rows(signatures, closure)
    cluster_rows = layer_curve_cluster_rows(components)
    side_rows = side_effect_distribution_rows(causal)
    bottleneck_rows = closure_bottleneck_rows(closure)
    heatmap_rows = gap_heatmap_rows(gaps)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase288",
        "created_at": now(),
        "signature_rows": len(signatures),
        "component_summary_rows": len(components),
        "causal_rows": len(causal),
        "closure_quality_rows": len(closure),
        "gap_rows": len(gaps),
        "family_feature_rows": len(family_rows),
        "model_feature_rows": len(model_rows),
        "component_distribution_rows": len(component_rows),
        "continue_channel_distribution_rows": len(channel_rows),
        "layer_curve_cluster_rows": len(cluster_rows),
        "side_effect_distribution_rows": len(side_rows),
        "closure_bottleneck_rows": len(bottleneck_rows),
        "gap_heatmap_rows": len(heatmap_rows),
        "global_mlp_dominance_rate": rate(sum(1 for r in components if r.get("dominant_positive_component") == "mlp"), len(components)),
        "global_attention_dominance_rate": rate(sum(1 for r in components if r.get("dominant_positive_component") == "attention"), len(components)),
        "global_continue_win_rate": rate(sum(1 for r in signatures if (r.get("path_signature") or {}).get("readout_winner") == "continue"), len(signatures)),
        "global_side_effect_risk_rate": rate(sum(1 for r in causal if r.get("side_effect_risk")), len(causal)),
        "global_closure_closed_count": sum(1 for r in closure if r.get("four_condition_closed")),
        "global_closure_rejected_count": sum(1 for r in closure if not r.get("four_condition_closed")),
        "top_layer_cluster_counts": dict(Counter(str(r.get("cluster_id")) for r in cluster_rows).most_common(12)),
        "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in channel_rows for _ in range(int(r.get("rows", 0)))).most_common(12)),
        "progress_estimate": {
            "pattern_family_atlas": 0.64,
            "physical_distribution_puzzle": 0.61,
            "feature_mining": 0.30,
            "mechanism_audit": 0.40,
            "closure": 0.20,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "phase288_family_feature_matrix.jsonl": family_rows,
        "phase288_model_feature_matrix.jsonl": model_rows,
        "phase288_component_distribution_rows.jsonl": component_rows,
        "phase288_continue_channel_distribution_rows.jsonl": channel_rows,
        "phase288_layer_curve_cluster_rows.jsonl": cluster_rows,
        "phase288_side_effect_distribution_rows.jsonl": side_rows,
        "phase288_closure_bottleneck_rows.jsonl": bottleneck_rows,
        "phase288_gap_heatmap_rows.jsonl": heatmap_rows,
    }
    for name, rows in tables.items():
        write_jsonl(OUT / name, rows)
        write_jsonl(V2 / name, rows)
    write_json(OUT / "phase288_feature_mining_summary.json", summary)
    write_json(V2 / "phase288_feature_mining_summary.json", summary)
    write_report(summary, family_rows, model_rows, cluster_rows)
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
