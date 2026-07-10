#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 289
SCHEMA_VERSION = "2.16.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase289_feature_reuse_delta_analysis"


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


FEATURES = [
    "answer_correct_proxy_rate",
    "mean_behavior_score",
    "mean_readout_score",
    "mean_rollout_score",
    "continue_win_rate",
    "mlp_dominance_rate",
    "attention_dominance_rate",
    "mean_positive_mlp_delta",
    "mean_positive_attn_delta",
    "causal_effect_supported_rate",
    "side_effect_risk_rate",
    "closure_rejected_rate",
    "stop_not_winner_rate",
]


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def shared_backbone(family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase289",
        "created_at": now(),
        "shared_backbone_id": "phase289:shared_backbone:global",
        **{f"shared_{name}": mean([safe_float(r.get(name)) for r in family_rows]) for name in FEATURES},
    }


def family_delta_rows(family_rows: list[dict[str, Any]], shared: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in family_rows:
        deltas = {f"delta_{name}": round(safe_float(row.get(name)) - safe_float(shared.get(f"shared_{name}")), 6) for name in FEATURES}
        top = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        if deltas["delta_side_effect_risk_rate"] > 0.1:
            label = "high_side_effect_family"
        elif deltas["delta_mlp_dominance_rate"] > 0.04:
            label = "mlp_reuse_strong_family"
        elif deltas["delta_attention_dominance_rate"] > 0.04:
            label = "attention_delta_family"
        elif deltas["delta_mean_behavior_score"] > 0.08:
            label = "high_behavior_family"
        else:
            label = "shared_backbone_family"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase289",
                "created_at": now(),
                "family_delta_id": f"phase289:family_delta:{row['family_id']}",
                "family_id": row["family_id"],
                "delta_label": label,
                "top_abs_delta_features": [{"feature": k.replace("delta_", ""), "delta": v} for k, v in top],
                **deltas,
            }
        )
    return rows


def model_delta_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared = {name: mean([safe_float(r.get(name)) for r in model_rows]) for name in ["mlp_dominance_rate", "attention_dominance_rate", "side_effect_risk_rate", "low_side_effect_supported_rate", "continue_win_rate", "closure_rejection_rate"]}
    rows = []
    for row in model_rows:
        deltas = {f"delta_{name}": round(safe_float(row.get(name)) - shared[name], 6) for name in shared}
        if row.get("model") == "glm4" and deltas["delta_side_effect_risk_rate"] > 0:
            label = "glm4_high_risk_delta"
        elif deltas["delta_low_side_effect_supported_rate"] > 0.1:
            label = "low_side_effect_strong_model"
        else:
            label = "baseline_model_delta"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase289",
                "created_at": now(),
                "model_delta_id": f"phase289:model_delta:{row['model']}",
                "model": row["model"],
                "delta_label": label,
                **deltas,
            }
        )
    return rows


def audit_candidate_rows(component_rows: list[dict[str, Any]], side_rows: list[dict[str, Any]], heat_rows: list[dict[str, Any]], closure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in side_rows:
        risk = safe_float(row.get("side_effect_risk_rate"))
        support = safe_float(row.get("causal_effect_supported_rate"))
        if risk >= 0.5 or (support >= 0.7 and risk <= 0.25):
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase289",
                    "created_at": now(),
                    "candidate_type": "side_effect_distribution",
                    "candidate_id": row.get("side_effect_distribution_id"),
                    "model": row.get("model"),
                    "family_id": row.get("family_id"),
                    "priority_reason": "high_risk" if risk >= 0.5 else "clean_causal_edge",
                    "priority_score": round(abs(risk - 0.25) + support, 6),
                    "recommended_next": "source_restricted_or_subspace_audit" if risk >= 0.5 else "closure_quality_probe",
                    "source_row": row,
                }
            )
    for row in heat_rows:
        total = int(row.get("open_gap_total") or 0)
        if total >= 155:
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase289",
                    "created_at": now(),
                    "candidate_type": "gap_heatmap_hotspot",
                    "candidate_id": row.get("gap_heatmap_id"),
                    "model": row.get("model"),
                    "family_id": row.get("family_id"),
                    "priority_reason": "large_open_gap_region",
                    "priority_score": total,
                    "recommended_next": "queue_driven_physical_path_fill",
                    "source_row": row,
                }
            )
    for row in closure_rows:
        if row.get("closure_blocker") in {"stop_not_winner", "continue_not_suppressed"} and int(row.get("rows") or 0) >= 2:
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase289",
                    "created_at": now(),
                    "candidate_type": "closure_bottleneck",
                    "candidate_id": row.get("closure_bottleneck_id"),
                    "model": row.get("model"),
                    "family_id": row.get("family_id"),
                    "priority_reason": row.get("closure_blocker"),
                    "priority_score": int(row.get("rows") or 0),
                    "recommended_next": "readout_competition_channel_decomposition",
                    "source_row": row,
                }
            )
    candidates.sort(key=lambda r: (-safe_float(r.get("priority_score")), str(r.get("candidate_type")), str(r.get("model")), str(r.get("family_id"))))
    for rank, row in enumerate(candidates, start=1):
        row["phase289_candidate_rank"] = rank
    return candidates


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase289_shared_backbone",
        "phase289_family_delta_rows",
        "phase289_model_delta_rows",
        "phase289_feature_driven_audit_candidates",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase289_summary"] = "phase289_summary.json"
    files["phase289_report"] = "phase289_report.md"
    manifest["latest_feature_delta_phase"] = "Phase289"
    manifest["phase289_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase289_summary.json", "phase289_family_delta_rows.jsonl", "phase289_model_delta_rows.jsonl", "phase289_feature_driven_audit_candidates.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase289_summary_ref"] = "phase289_summary.json"
    client["phase289_audit_candidates_ref"] = "phase289_feature_driven_audit_candidates.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase289_shared_backbone"] = "global shared backbone estimated from family feature matrix"
    tables["phase289_family_delta_rows"] = "family deltas from shared backbone"
    tables["phase289_model_delta_rows"] = "model deltas from model average"
    tables["phase289_feature_driven_audit_candidates"] = "next mechanism audit candidates selected from atlas features"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    family = read_jsonl(V2 / "phase288_family_feature_matrix.jsonl")
    model = read_jsonl(V2 / "phase288_model_feature_matrix.jsonl")
    component = read_jsonl(V2 / "phase288_component_distribution_rows.jsonl")
    side = read_jsonl(V2 / "phase288_side_effect_distribution_rows.jsonl")
    heat = read_jsonl(V2 / "phase288_gap_heatmap_rows.jsonl")
    closure = read_jsonl(V2 / "phase288_closure_bottleneck_rows.jsonl")
    shared = shared_backbone(family)
    family_delta = family_delta_rows(family, shared)
    model_delta = model_delta_rows(model)
    candidates = audit_candidate_rows(component, side, heat, closure)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase289",
        "created_at": now(),
        "family_delta_rows": len(family_delta),
        "model_delta_rows": len(model_delta),
        "feature_driven_audit_candidates": len(candidates),
        "candidate_type_counts": dict(Counter(str(r.get("candidate_type")) for r in candidates)),
        "recommended_next_counts": dict(Counter(str(r.get("recommended_next")) for r in candidates)),
        "family_delta_label_counts": dict(Counter(str(r.get("delta_label")) for r in family_delta)),
        "model_delta_label_counts": dict(Counter(str(r.get("delta_label")) for r in model_delta)),
        "shared_backbone": shared,
        "progress_estimate": {
            "pattern_family_atlas": 0.65,
            "feature_mining": 0.38,
            "reuse_delta_analysis": 0.25,
            "mechanism_audit": 0.40,
            "closure": 0.20,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase289_shared_backbone.jsonl", [shared])
    write_jsonl(OUT / "phase289_family_delta_rows.jsonl", family_delta)
    write_jsonl(OUT / "phase289_model_delta_rows.jsonl", model_delta)
    write_jsonl(OUT / "phase289_feature_driven_audit_candidates.jsonl", candidates)
    write_json(OUT / "phase289_summary.json", summary)
    write_jsonl(V2 / "phase289_shared_backbone.jsonl", [shared])
    write_jsonl(V2 / "phase289_family_delta_rows.jsonl", family_delta)
    write_jsonl(V2 / "phase289_model_delta_rows.jsonl", model_delta)
    write_jsonl(V2 / "phase289_feature_driven_audit_candidates.jsonl", candidates)
    write_json(V2 / "phase289_summary.json", summary)
    report = "\n".join(
        [
            "# Phase289 Feature Reuse-Delta Analysis",
            "",
            f"- family_delta_rows: {summary['family_delta_rows']}",
            f"- model_delta_rows: {summary['model_delta_rows']}",
            f"- feature_driven_audit_candidates: {summary['feature_driven_audit_candidates']}",
            f"- candidate_type_counts: {json.dumps(summary['candidate_type_counts'], ensure_ascii=False)}",
            f"- recommended_next_counts: {json.dumps(summary['recommended_next_counts'], ensure_ascii=False)}",
            f"- family_delta_label_counts: {json.dumps(summary['family_delta_label_counts'], ensure_ascii=False)}",
            f"- model_delta_label_counts: {json.dumps(summary['model_delta_label_counts'], ensure_ascii=False)}",
            "",
            "This phase turns atlas features into reuse/delta structure and next audit candidates.",
        ]
    ) + "\n"
    (OUT / "phase289_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase289_report.md").write_text(report, encoding="utf-8")
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
