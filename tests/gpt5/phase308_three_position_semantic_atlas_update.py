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
PHASE = "Phase308"
SCHEMA_VERSION = "2.35.0"


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
    summaries = read_jsonl(V2 / "phase307_three_position_summary_rows.jsonl")
    components = read_jsonl(V2 / "phase307_three_position_component_rows.jsonl")
    position_cells = build_position_cells(summaries)
    attribute_cells = build_attribute_cells(summaries)
    route_rows = build_route_rows(position_cells)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "three_position_component_rows": len(components),
        "three_position_summary_rows": len(summaries),
        "position_cell_rows": len(position_cells),
        "attribute_position_cell_rows": len(attribute_cells),
        "route_rows": len(route_rows),
        "dominant_component_counts": dict(Counter(str(r.get("dominant_component")) for r in position_cells)),
        "mean_position_path_score": mean_safe([safe_float(r.get("position_path_score")) for r in position_cells]),
        "route_type_counts": dict(Counter(str(r.get("route_type")) for r in route_rows)),
        "progress": {
            "language_pattern_family_atlas": 0.83,
            "semantic_reuse_delta_subatlas": 0.46,
            "semantic_internal_physical_path": 0.30,
            "sample_type_coverage": 0.72,
            "large_data_feature_mining": 0.73,
            "physical_distribution_puzzle": 0.77,
            "mechanism_causal_audit": 0.52,
            "closure": 0.21,
        },
        "hard_limits": [
            "Token position matching uses tokenizer subsequence search and may be imperfect.",
            "Phase307/308 remains observational, not causal.",
            "Shared backbone and delta subspaces are not yet PCA/subspace-validated.",
            "Only 36 cases were traced across three positions.",
        ],
    }
    write_jsonl(V2 / "phase308_three_position_semantic_position_cell_rows.jsonl", position_cells)
    write_jsonl(V2 / "phase308_three_position_semantic_attribute_cell_rows.jsonl", attribute_cells)
    write_jsonl(V2 / "phase308_three_position_semantic_route_rows.jsonl", route_rows)
    write_json(V2 / "phase308_three_position_semantic_atlas_summary.json", payload)
    write_json(V2 / "progress.json", {**read_json(V2 / "progress.json"), **payload["progress"], "last_phase": PHASE, "updated_at": now()})
    update_manifest(payload)
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_position_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (model, pos), vals in sorted(group(rows, "model", "position_role").items()):
        attn = mean_safe([safe_float(v.get("sum_positive_attn_semantic_delta")) for v in vals])
        mlp = mean_safe([safe_float(v.get("sum_positive_mlp_semantic_delta")) for v in vals])
        residual = mean_safe([safe_float(v.get("sum_positive_residual_semantic_delta")) for v in vals])
        target_rate = mean_safe([1.0 if v.get("final_layer_out_semantic_winner") == "target" else 0.0 for v in vals])
        score = clamp(0.30 * target_rate + 0.30 * min(attn / 25.0, 1.0) + 0.30 * min(mlp / 25.0, 1.0) + 0.10 * (1.0 - min(residual / 5.0, 1.0)))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "position_cell_id": f"phase308:position:{model}:{pos}",
                "model": model,
                "position_role": pos,
                "rows": len(vals),
                "target_winner_rate": target_rate,
                "mean_positive_attention_semantic_delta": attn,
                "mean_positive_mlp_semantic_delta": mlp,
                "mean_positive_residual_semantic_delta": residual,
                "dominant_component": "attention" if attn >= mlp else "mlp",
                "dominant_component_counts": dict(Counter(str(v.get("dominant_positive_semantic_component")) for v in vals)),
                "position_path_score": score,
            }
        )
    return out


def build_attribute_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (attr, pos), vals in sorted(group(rows, "attribute_type", "position_role").items()):
        attn = mean_safe([safe_float(v.get("sum_positive_attn_semantic_delta")) for v in vals])
        mlp = mean_safe([safe_float(v.get("sum_positive_mlp_semantic_delta")) for v in vals])
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "attribute_position_cell_id": f"phase308:attribute_position:{attr}:{pos}",
                "attribute_type": attr,
                "position_role": pos,
                "rows": len(vals),
                "target_winner_rate": mean_safe([1.0 if v.get("final_layer_out_semantic_winner") == "target" else 0.0 for v in vals]),
                "mean_positive_attention_semantic_delta": attn,
                "mean_positive_mlp_semantic_delta": mlp,
                "dominant_component": "attention" if attn >= mlp else "mlp",
            }
        )
    return out


def build_route_rows(position_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = defaultdict(dict)
    for row in position_cells:
        by_model[str(row["model"])][str(row["position_role"])] = row
    out = []
    for model, cells in sorted(by_model.items()):
        object_dom = cells.get("object", {}).get("dominant_component")
        query_dom = cells.get("query", {}).get("dominant_component")
        last_dom = cells.get("last", {}).get("dominant_component")
        route_type = f"{object_dom}->{query_dom}->{last_dom}"
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "route_id": f"phase308:route:{model}",
                "model": model,
                "object_dominant_component": object_dom,
                "query_dominant_component": query_dom,
                "last_dominant_component": last_dom,
                "route_type": route_type,
                "object_score": cells.get("object", {}).get("position_path_score"),
                "query_score": cells.get("query", {}).get("position_path_score"),
                "last_score": cells.get("last", {}).get("position_path_score"),
                "route_interpretation": interpret_route(route_type),
            }
        )
    return out


def interpret_route(route_type: str) -> str:
    if route_type == "mlp->mlp->attention":
        return "object/query write with answer readout routing"
    if route_type == "mlp->mlp->mlp":
        return "mlp-dominant semantic write path"
    if route_type == "attention->attention->attention":
        return "attention-dominant semantic routing path"
    return "mixed semantic route"


def update_manifest(summary: dict[str, Any]) -> None:
    path = V2 / "manifest.json"
    manifest = read_json(path)
    manifest.setdefault("generated_files", [])
    for name in [
        "phase308_three_position_semantic_position_cell_rows.jsonl",
        "phase308_three_position_semantic_attribute_cell_rows.jsonl",
        "phase308_three_position_semantic_route_rows.jsonl",
        "phase308_three_position_semantic_atlas_summary.json",
    ]:
        if name not in manifest["generated_files"]:
            manifest["generated_files"].append(name)
    manifest["last_phase"] = PHASE
    manifest["updated_at"] = now()
    manifest["phase308_summary"] = {
        "position_cell_rows": summary["position_cell_rows"],
        "mean_position_path_score": summary["mean_position_path_score"],
        "route_type_counts": summary["route_type_counts"],
    }
    write_json(path, manifest)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase308 Three Position Semantic Atlas Update",
        "",
        f"- three_position_component_rows: {summary['three_position_component_rows']}",
        f"- three_position_summary_rows: {summary['three_position_summary_rows']}",
        f"- position_cell_rows: {summary['position_cell_rows']}",
        f"- attribute_position_cell_rows: {summary['attribute_position_cell_rows']}",
        f"- route_rows: {summary['route_rows']}",
        f"- dominant_component_counts: {json.dumps(summary['dominant_component_counts'], ensure_ascii=False)}",
        f"- mean_position_path_score: {summary['mean_position_path_score']}",
        f"- route_type_counts: {json.dumps(summary['route_type_counts'], ensure_ascii=False)}",
        "",
        "This summarizes object/query/last semantic component paths. It is not causal closure.",
    ]
    (V2 / "phase308_three_position_semantic_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
