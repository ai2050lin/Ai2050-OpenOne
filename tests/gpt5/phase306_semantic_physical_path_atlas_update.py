#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE = "Phase306"
SCHEMA_VERSION = "2.33.0"


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


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k)) for k in keys)].append(row)
    return out


def main() -> None:
    summaries = read_jsonl(V2 / "phase305_semantic_component_summary_rows.jsonl")
    component_rows = read_jsonl(V2 / "phase305_semantic_component_rows.jsonl")
    behavior = read_jsonl(V2 / "phase302_semantic_behavior_rows.jsonl")
    by_model_attr = group(summaries, "model", "attribute_type")
    behavior_by_model_attr = group(behavior, "model", "attribute_type")
    path_rows: list[dict[str, Any]] = []
    for key, vals in sorted(by_model_attr.items()):
        model, attr = key
        bvals = behavior_by_model_attr.get(key, [])
        target_rate = mean_safe([1.0 if v.get("final_semantic_winner") == "target" else 0.0 for v in vals])
        attn = mean_safe([safe_float(v.get("sum_positive_attn_semantic_delta")) for v in vals])
        mlp = mean_safe([safe_float(v.get("sum_positive_mlp_semantic_delta")) for v in vals])
        residual = mean_safe([safe_float(v.get("sum_positive_residual_semantic_delta")) for v in vals])
        dominant = Counter(str(v.get("dominant_positive_semantic_component")) for v in vals).most_common(1)[0][0]
        behavior_success = mean_safe([1.0 if v.get("answer_correct_proxy") else 0.0 for v in bvals])
        evidence_score = clamp(0.30 * target_rate + 0.25 * min(attn / 25.0, 1.0) + 0.25 * min(mlp / 25.0, 1.0) + 0.20 * behavior_success)
        path_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "semantic_path_cell_id": f"phase306:path_cell:{model}:{attr}",
                "model": model,
                "attribute_type": attr,
                "summary_rows": len(vals),
                "behavior_rows": len(bvals),
                "behavior_answer_correct_proxy_rate": behavior_success,
                "final_target_winner_rate": target_rate,
                "mean_final_semantic_margin": mean_safe([safe_float(v.get("final_semantic_margin")) for v in vals]),
                "mean_positive_attention_semantic_delta": attn,
                "mean_positive_mlp_semantic_delta": mlp,
                "mean_positive_residual_semantic_delta": residual,
                "dominant_semantic_component": dominant,
                "dominant_component_counts": dict(Counter(str(v.get("dominant_positive_semantic_component")) for v in vals)),
                "semantic_physical_path_score": evidence_score,
                "status": "internal_semantic_path_observed_not_causal",
            }
        )
    by_attr = group(path_rows, "attribute_type")
    attr_rows = []
    for (attr,), vals in sorted(by_attr.items()):
        attr_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "semantic_attribute_atlas_id": f"phase306:attribute:{attr}",
                "attribute_type": attr,
                "model_cells": len(vals),
                "mean_semantic_physical_path_score": mean_safe([safe_float(v.get("semantic_physical_path_score")) for v in vals]),
                "mean_target_winner_rate": mean_safe([safe_float(v.get("final_target_winner_rate")) for v in vals]),
                "mean_behavior_success_rate": mean_safe([safe_float(v.get("behavior_answer_correct_proxy_rate")) for v in vals]),
                "dominant_component_counts": dict(Counter(str(v.get("dominant_semantic_component")) for v in vals)),
                "recommended_next": choose_next(attr, vals),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "semantic_component_rows": len(component_rows),
        "semantic_component_summary_rows": len(summaries),
        "semantic_path_cell_rows": len(path_rows),
        "semantic_attribute_atlas_rows": len(attr_rows),
        "mean_semantic_physical_path_score": mean_safe([safe_float(r.get("semantic_physical_path_score")) for r in path_rows]),
        "mean_final_target_winner_rate": mean_safe([safe_float(r.get("final_target_winner_rate")) for r in path_rows]),
        "dominant_component_counts": dict(Counter(str(r.get("dominant_semantic_component")) for r in path_rows)),
        "attribute_recommended_next_counts": dict(Counter(str(r.get("recommended_next")) for r in attr_rows)),
        "progress": {
            "language_pattern_family_atlas": 0.82,
            "semantic_reuse_delta_subatlas": 0.43,
            "semantic_internal_physical_path": 0.22,
            "sample_type_coverage": 0.72,
            "large_data_feature_mining": 0.72,
            "physical_distribution_puzzle": 0.76,
            "mechanism_causal_audit": 0.52,
            "closure": 0.21,
        },
        "hard_limits": [
            "Phase305/306 localizes last-position semantic component paths, not object-token/query-token paths.",
            "No causal patch was applied; evidence is observational component attribution.",
            "Semantic distractor sets remain hand-built and need alias expansion.",
            "Only 36 semantic internal cases are traced.",
        ],
    }
    write_jsonl(V2 / "phase306_semantic_physical_path_cell_rows.jsonl", path_rows)
    write_jsonl(V2 / "phase306_semantic_attribute_atlas_rows.jsonl", attr_rows)
    write_json(V2 / "phase306_semantic_physical_path_atlas_summary.json", payload)
    write_json(V2 / "progress.json", {**read_json(V2 / "progress.json"), **payload["progress"], "last_phase": PHASE, "updated_at": now()})
    update_manifest(payload)
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def choose_next(attr: str, vals: list[dict[str, Any]]) -> str:
    score = mean_safe([safe_float(v.get("semantic_physical_path_score")) for v in vals])
    target = mean_safe([safe_float(v.get("final_target_winner_rate")) for v in vals])
    if score >= 0.55 and target >= 0.55:
        return "expand_object_query_position_trace"
    if target < 0.4:
        return "expand_alias_and_distractor_calibration"
    if attr in {"difference", "use", "shape"}:
        return "contrast_delta_path_followup"
    return "component_path_coverage_expand"


def update_manifest(summary: dict[str, Any]) -> None:
    path = V2 / "manifest.json"
    manifest = read_json(path)
    manifest.setdefault("generated_files", [])
    for name in [
        "phase306_semantic_physical_path_cell_rows.jsonl",
        "phase306_semantic_attribute_atlas_rows.jsonl",
        "phase306_semantic_physical_path_atlas_summary.json",
    ]:
        if name not in manifest["generated_files"]:
            manifest["generated_files"].append(name)
    manifest["last_phase"] = PHASE
    manifest["updated_at"] = now()
    manifest["phase306_summary"] = {
        "semantic_path_cell_rows": summary["semantic_path_cell_rows"],
        "mean_semantic_physical_path_score": summary["mean_semantic_physical_path_score"],
        "mean_final_target_winner_rate": summary["mean_final_target_winner_rate"],
    }
    write_json(path, manifest)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase306 Semantic Physical Path Atlas Update",
        "",
        f"- semantic_component_rows: {summary['semantic_component_rows']}",
        f"- semantic_component_summary_rows: {summary['semantic_component_summary_rows']}",
        f"- semantic_path_cell_rows: {summary['semantic_path_cell_rows']}",
        f"- semantic_attribute_atlas_rows: {summary['semantic_attribute_atlas_rows']}",
        f"- mean_semantic_physical_path_score: {summary['mean_semantic_physical_path_score']}",
        f"- mean_final_target_winner_rate: {summary['mean_final_target_winner_rate']}",
        f"- dominant_component_counts: {json.dumps(summary['dominant_component_counts'], ensure_ascii=False)}",
        f"- attribute_recommended_next_counts: {json.dumps(summary['attribute_recommended_next_counts'], ensure_ascii=False)}",
        "",
        "This is observational semantic component attribution, not closure.",
    ]
    (V2 / "phase306_semantic_physical_path_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
