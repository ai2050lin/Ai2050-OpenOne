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

PHASE = 297
SCHEMA_VERSION = "2.24.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase297_feature_algorithm_v4_with_component_paths"


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


def clamp(v: float) -> float:
    return round(max(0.0, min(1.0, v)), 6)


def main_rows() -> list[dict[str, Any]]:
    v3 = {(str(r["family_id"]), str(r["model"])): r for r in read_jsonl(V2 / "phase295_feature_matrix_v3_rows.jsonl")}
    comps_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(V2 / "phase296_component_summary_rows.jsonl"):
        comps_by_cell[(str(row.get("family_id")), str(row.get("model")))].append(row)
    rows = []
    for cell, base in sorted(v3.items()):
        comps = comps_by_cell.get(cell, [])
        comp_rows = len(comps)
        mlp_rate = sum(1 for r in comps if r.get("dominant_positive_component") == "mlp") / comp_rows if comp_rows else 0.0
        attn_rate = sum(1 for r in comps if r.get("dominant_positive_component") == "attention") / comp_rows if comp_rows else 0.0
        continue_rate = sum(1 for r in comps if r.get("final_winner") == "continue") / comp_rows if comp_rows else 0.0
        component_coverage_expanded = comp_rows / 36.0
        mlp_strength = mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in comps])
        attn_strength = mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in comps])
        component_path_score = clamp(0.35 * component_coverage_expanded + 0.25 * mlp_rate + 0.15 * attn_rate + 0.15 * min(mlp_strength / 35.0, 1.0) + 0.10 * (1.0 - continue_rate))
        v3_score = safe_float(base.get("atlas_completion_v3"))
        v4 = clamp(0.78 * v3_score + 0.22 * component_path_score)
        if comp_rows == 0:
            priority = "need_expanded_component_path"
        elif continue_rate >= 1.0 and mlp_rate >= 0.75:
            priority = "mlp_continue_path_causal_audit"
        elif attn_rate >= 0.3:
            priority = "attention_route_followup"
        else:
            priority = str(base.get("next_priority"))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase297",
                "created_at": now(),
                "feature_v4_id": f"phase297:feature_v4:{cell[0]}:{cell[1]}",
                "family_id": cell[0],
                "model": cell[1],
                "atlas_completion_v3": v3_score,
                "expanded_component_summary_rows": comp_rows,
                "expanded_component_coverage": round(component_coverage_expanded, 6),
                "expanded_mlp_dominance_rate": round(mlp_rate, 6),
                "expanded_attention_dominance_rate": round(attn_rate, 6),
                "expanded_component_continue_winner_rate": round(continue_rate, 6),
                "expanded_mean_sum_positive_mlp_delta": mlp_strength,
                "expanded_mean_sum_positive_attn_delta": attn_strength,
                "component_path_score": component_path_score,
                "atlas_completion_v4": v4,
                "completion_delta_v4_minus_v3": round(v4 - v3_score, 6),
                "next_priority": priority,
            }
        )
    rows.sort(key=lambda r: (safe_float(r["atlas_completion_v4"]), str(r["model"]), str(r["family_id"])))
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(safe_float(row["atlas_completion_v4"]))
        by_family[str(row["family_id"])].append(safe_float(row["atlas_completion_v4"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase297",
        "created_at": now(),
        "feature_v4_rows": len(rows),
        "mean_atlas_completion_v4": mean_safe([safe_float(r["atlas_completion_v4"]) for r in rows]),
        "mean_completion_delta_v4_minus_v3": mean_safe([safe_float(r["completion_delta_v4_minus_v3"]) for r in rows]),
        "mean_expanded_component_coverage": mean_safe([safe_float(r["expanded_component_coverage"]) for r in rows]),
        "mean_expanded_mlp_dominance_rate": mean_safe([safe_float(r["expanded_mlp_dominance_rate"]) for r in rows]),
        "mean_expanded_component_continue_winner_rate": mean_safe([safe_float(r["expanded_component_continue_winner_rate"]) for r in rows]),
        "next_priority_counts": dict(Counter(str(r["next_priority"]) for r in rows)),
        "model_completion_v4": {k: mean_safe(v) for k, v in sorted(by_model.items())},
        "family_completion_v4": {k: mean_safe(v) for k, v in sorted(by_family.items())},
        "progress_estimate": {
            "pattern_family_atlas": 0.76,
            "sample_type_coverage": 0.68,
            "feature_mining": 0.62,
            "physical_distribution_puzzle": 0.70,
            "mechanism_audit": 0.47,
            "closure": 0.21,
        },
    }


def update_v2(payload: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase297_feature_matrix_v4_rows"] = "phase297_feature_matrix_v4_rows.jsonl"
    files["phase297_summary"] = "phase297_summary.json"
    files["phase297_report"] = "phase297_report.md"
    manifest["latest_feature_algorithm_phase"] = "Phase297"
    manifest["phase297_summary"] = payload
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase297_summary.json", "phase297_feature_matrix_v4_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase297_summary_ref"] = "phase297_summary.json"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    schema.setdefault("tables", {})["phase297_feature_matrix_v4_rows"] = "feature algorithm v4 with expanded component path evidence"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    rows = main_rows()
    payload = summary(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase297_feature_matrix_v4_rows.jsonl", rows)
    write_jsonl(V2 / "phase297_feature_matrix_v4_rows.jsonl", rows)
    write_json(OUT / "phase297_summary.json", payload)
    write_json(V2 / "phase297_summary.json", payload)
    report = "\n".join([
        "# Phase297 Feature Algorithm V4 With Component Paths",
        "",
        f"- mean_atlas_completion_v4: {payload['mean_atlas_completion_v4']}",
        f"- mean_completion_delta_v4_minus_v3: {payload['mean_completion_delta_v4_minus_v3']}",
        f"- mean_expanded_component_coverage: {payload['mean_expanded_component_coverage']}",
        f"- mean_expanded_mlp_dominance_rate: {payload['mean_expanded_mlp_dominance_rate']}",
        f"- mean_expanded_component_continue_winner_rate: {payload['mean_expanded_component_continue_winner_rate']}",
        f"- next_priority_counts: {json.dumps(payload['next_priority_counts'], ensure_ascii=False)}",
    ]) + "\n"
    (OUT / "phase297_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase297_report.md").write_text(report, encoding="utf-8")
    update_v2(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
