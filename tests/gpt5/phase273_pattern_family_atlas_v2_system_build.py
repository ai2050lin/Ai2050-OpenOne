#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 273
SCHEMA_VERSION = "2.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
FAMILIES = [
    "content_knowledge",
    "output_protocol",
    "reasoning_constraint",
    "syntax_structure",
    "language_action",
    "cross_lingual",
    "readout_competition",
    "state_drift",
    "closure",
]

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "tests/result/pattern_family_atlas/v1"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
DETAIL_DIR = V2 / "case_details"


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


def clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model")), str(row.get("case_id"))


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out[key(row)] = row
    return out


def group_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return out


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def score_behavior(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    parts = [
        1.0 if row.get("answer_correct_proxy") else 0.0,
        1.0 if row.get("pattern_matched_proxy") else 0.0,
        0.0 if row.get("has_drift_marker") else 1.0,
    ]
    return mean_safe(parts)


def score_readout(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    rank_score = 1.0 if int(row.get("target_rank") or 999999) == 1 else 0.25
    margin_score = clamp01((safe_float(row.get("target_margin_vs_winner")) + 10.0) / 30.0)
    stop_score = clamp01((safe_float(row.get("stop_continue_margin")) + 10.0) / 20.0)
    return mean_safe([rank_score, margin_score, stop_score])


def score_layer(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    stable = 1.0 if int(row.get("stable_continue_from_layer") or 0) == 0 else 0.6
    layers = max(1, int(row.get("num_layers_observed") or 1))
    early = 1.0 - min(1.0, safe_float(row.get("first_continue_win_layer")) / layers)
    margin = clamp01((safe_float(row.get("final_continue_stop_margin")) + 10.0) / 30.0)
    return mean_safe([stable, early, margin])


def score_component(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    mlp = clamp01(safe_float(row.get("sum_positive_mlp_delta")) / 40.0)
    attn = clamp01(safe_float(row.get("sum_positive_attn_delta")) / 25.0)
    residual = clamp01(safe_float(row.get("sum_positive_residual_delta")) / 1.0)
    dominant = 1.0 if row.get("dominant_positive_component") in {"mlp", "attn", "residual"} else 0.0
    return mean_safe([mlp, attn, residual, dominant])


def score_causal(rows270: list[dict[str, Any]], rows271: list[dict[str, Any]], rows272: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    if rows270:
        vals.append(sum(1 for r in rows270 if r.get("writer_set_supported")) / len(rows270))
    if rows271:
        vals.append(sum(1 for r in rows271 if r.get("clean_edge_candidate")) / len(rows271))
    if rows272:
        vals.append(sum(1 for r in rows272 if r.get("strict_protocol_clean")) / len(rows272))
    return mean_safe(vals)


def score_rollout(rows272: list[dict[str, Any]], behavior: dict[str, Any] | None) -> float:
    vals: list[float] = []
    if rows272:
        vals.append(clamp01((mean([safe_float(r.get("protocol_stop_margin_delta")) for r in rows272]) + 2.0) / 4.0))
    if behavior:
        vals.append(0.0 if behavior.get("has_drift_marker") else 1.0)
    return mean_safe(vals)


def score_closure(rows272: list[dict[str, Any]], behavior: dict[str, Any] | None) -> float:
    vals: list[float] = []
    if rows272:
        vals.append(sum(1 for r in rows272 if r.get("strict_protocol_clean")) / len(rows272))
    if behavior:
        vals.append(1.0 if (behavior.get("answer_correct_proxy") and behavior.get("pattern_matched_proxy")) else 0.0)
    return mean_safe(vals)


def status_from_scores(scores: dict[str, float]) -> str:
    if scores["closure"] >= 0.7 and scores["causal"] >= 0.5 and scores["overall"] >= 0.65:
        return "high_quality_candidate_not_closed"
    if scores["component_path"] >= 0.45 and scores["causal"] >= 0.25:
        return "path_candidate_not_closed"
    if scores["behavior"] > 0.0 or scores["readout"] > 0.0:
        return "mapped_partial"
    return "not_enough_data"


def main() -> None:
    if not V1.exists():
        raise SystemExit(f"missing v1 atlas: {V1}")
    if V2.exists():
        shutil.rmtree(V2)
    V2.mkdir(parents=True)
    DETAIL_DIR.mkdir(parents=True)

    families = read_jsonl(V1 / "families.jsonl")
    modes = read_jsonl(V1 / "modes.jsonl")
    cases = read_jsonl(V1 / "mode_family_case_bank_v3.jsonl")
    behavior = by_key(read_jsonl(V1 / "phase266_behavior_rows.jsonl"))
    readout = by_key(read_jsonl(V1 / "phase266_readout_rows.jsonl"))
    layer = by_key(read_jsonl(V1 / "phase267_family_path_signature_rows.jsonl"))
    component = by_key(read_jsonl(V1 / "phase268_component_summary_rows.jsonl"))
    comp270 = group_by_key(read_jsonl(V1 / "phase270_writer_set_rows.jsonl"))
    fiber271 = group_by_key(read_jsonl(V1 / "phase271_closure_fiber_rows.jsonl"))
    fiber272 = group_by_key(read_jsonl(V1 / "phase272_span_protocol_fiber_rows.jsonl"))
    case_map = {str(r["case_id"]): r for r in cases}

    all_keys = set(behavior) | set(readout) | set(layer) | set(component) | set(comp270) | set(fiber271) | set(fiber272)
    path_rows: list[dict[str, Any]] = []
    detail_refs: list[dict[str, Any]] = []
    for model, case_id in sorted(all_keys):
        if model not in MODELS or case_id not in case_map:
            continue
        source = case_map[case_id]
        b = behavior.get((model, case_id))
        r = readout.get((model, case_id))
        l = layer.get((model, case_id))
        c = component.get((model, case_id))
        rows270 = comp270.get((model, case_id), [])
        rows271 = fiber271.get((model, case_id), [])
        rows272 = fiber272.get((model, case_id), [])
        scores = {
            "behavior": score_behavior(b),
            "readout": score_readout(r),
            "layer_path": score_layer(l),
            "component_path": score_component(c),
            "causal": score_causal(rows270, rows271, rows272),
            "rollout": score_rollout(rows272, b),
            "closure": score_closure(rows272, b),
        }
        scores["overall"] = mean_safe(list(scores.values()))
        dominant_layers = sorted(
            {
                int(x)
                for x in [
                    c.get("strongest_mlp_layer") if c else None,
                    c.get("strongest_attn_layer") if c else None,
                    c.get("strongest_residual_layer") if c else None,
                ]
                if x is not None
            }
        )
        detail_path = f"case_details/{model}__{case_id}.json"
        row = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase273",
            "created_at": now(),
            "signature_id": f"phase273:signature:{model}:{case_id}",
            "case_id": case_id,
            "model": model,
            "family_id": source.get("family_id"),
            "mode_id": source.get("mode_id"),
            "variant_id": source.get("variant_id"),
            "path_schema_id": source.get("path_schema_id"),
            "prompt_hash": sha(str(source.get("prompt", ""))),
            "target": source.get("target"),
            "path_signature": {
                "trigger": source.get("continuation_trigger"),
                "state": [k for k, v in source.items() if k.startswith("S_") and v],
                "dominant_layers": dominant_layers,
                "attention_route_score": round(safe_float(c.get("sum_positive_attn_delta")) if c else 0.0, 6),
                "mlp_write_score": round(safe_float(c.get("sum_positive_mlp_delta")) if c else 0.0, 6),
                "compensation_score": mean_safe([abs(safe_float(x.get("window_minus_single_delta"))) for x in rows270]),
                "readout_winner": (r or {}).get("competition_winner") or (l or {}).get("final_competition_winner"),
                "top_competitor": (r or {}).get("second_competitor"),
                "strict_protocol_clean_count": sum(1 for x in rows272 if x.get("strict_protocol_clean")),
            },
            "scores": scores,
            "status": status_from_scores(scores),
            "detail_ref": detail_path,
        }
        path_rows.append(row)
        detail = {
            "schema_version": SCHEMA_VERSION,
            "case": source,
            "model": model,
            "path_signature": row,
            "behavior": b,
            "readout": r,
            "layer_trace_summary": l,
            "component_summary": c,
            "writer_set_rows": rows270,
            "closure_fiber_rows": rows271,
            "span_protocol_fiber_rows": rows272,
        }
        write_json(V2 / detail_path, detail)
        detail_refs.append({"case_id": case_id, "model": model, "detail_ref": detail_path})

    atlas_scores: list[dict[str, Any]] = []
    for family in FAMILIES:
        for model in MODELS:
            rows = [r for r in path_rows if r["family_id"] == family and r["model"] == model]
            if not rows:
                continue
            score_names = ["behavior", "readout", "layer_path", "component_path", "causal", "rollout", "closure", "overall"]
            atlas_scores.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "score_id": f"phase273:score:{family}:{model}",
                    "phase_id": "Phase273",
                    "created_at": now(),
                    "family_id": family,
                    "model": model,
                    "case_count": len(rows),
                    "scores": {name: mean_safe([safe_float(r["scores"][name]) for r in rows]) for name in score_names},
                    "status_counts": dict(Counter(str(r["status"]) for r in rows)),
                }
            )
    for family in FAMILIES:
        rows = [r for r in path_rows if r["family_id"] == family]
        if rows:
            atlas_scores.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "score_id": f"phase273:score:{family}:cross_model",
                    "phase_id": "Phase273",
                    "created_at": now(),
                    "family_id": family,
                    "model": "cross_model",
                    "case_count": len(rows),
                    "scores": {name: mean_safe([safe_float(r["scores"][name]) for r in rows]) for name in ["behavior", "readout", "layer_path", "component_path", "causal", "rollout", "closure", "overall"]},
                    "status_counts": dict(Counter(str(r["status"]) for r in rows)),
                }
            )

    graph_nodes = []
    graph_edges = []
    for family in families:
        graph_nodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "node_id": f"family:{family['family_id']}",
                "node_type": "family",
                "label": family.get("family_name") or family["family_id"],
                "family_id": family["family_id"],
            }
        )
    for row in path_rows:
        node_id = f"case:{row['model']}:{row['case_id']}"
        graph_nodes.append({"schema_version": SCHEMA_VERSION, "node_id": node_id, "node_type": "case", "label": row["case_id"], "model": row["model"], "family_id": row["family_id"], "status": row["status"]})
        graph_edges.append({"schema_version": SCHEMA_VERSION, "edge_id": f"edge:{row['family_id']}:{row['model']}:{row['case_id']}", "source": f"family:{row['family_id']}", "target": node_id, "edge_type": "family_case_path", "weight": row["scores"]["overall"]})

    schema = {
        "schema_version": SCHEMA_VERSION,
        "primary_table": "path_signature_rows.jsonl",
        "tables": {
            "path_signature_rows": "one row per model-case physical path signature",
            "atlas_scores": "family x model score matrix",
            "case_details": "detail json loaded on demand by client",
        },
        "score_fields": ["behavior", "readout", "layer_path", "component_path", "causal", "rollout", "closure", "overall"],
        "status_values": ["not_enough_data", "mapped_partial", "path_candidate_not_closed", "high_quality_candidate_not_closed"],
    }
    manifest = {
        "program_id": "pattern_family_atlas_v2",
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase273",
        "created_at": now(),
        "source": "tests/result/pattern_family_atlas/v1",
        "models": MODELS,
        "families": len(FAMILIES),
        "path_signatures": len(path_rows),
        "atlas_scores": len(atlas_scores),
        "detail_files": len(detail_refs),
        "files": {
            "schema": "schema.json",
            "client_index": "client_index.json",
            "families": "families.jsonl",
            "modes": "modes.jsonl",
            "cases": "cases.jsonl",
            "path_signature_rows": "path_signature_rows.jsonl",
            "atlas_scores": "atlas_scores.jsonl",
            "graph_nodes": "graph_nodes.jsonl",
            "graph_edges": "graph_edges.jsonl",
            "summary": "summary.md",
        },
    }
    family_matrix = defaultdict(dict)
    for score in atlas_scores:
        if score["model"] != "cross_model":
            family_matrix[score["family_id"]][score["model"]] = score["scores"]
    client_index = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint": "manifest.json",
        "load_strategy": "summary_first_detail_on_demand",
        "views": [
            "overview",
            "family_matrix",
            "path_explorer",
            "component_view",
            "causal_audit",
            "case_detail",
        ],
        "initial_files": ["manifest.json", "client_index.json", "atlas_scores.jsonl", "families.jsonl"],
        "detail_prefix": "case_details/",
        "family_matrix": family_matrix,
        "detail_refs": detail_refs,
        "top_candidates": sorted(path_rows, key=lambda r: r["scores"]["overall"], reverse=True)[:30],
    }
    summary_lines = [
        "# Pattern Family Atlas v2 System Build",
        "",
        f"- phase: Phase273",
        f"- schema_version: {SCHEMA_VERSION}",
        f"- path_signatures: {len(path_rows)}",
        f"- atlas_scores: {len(atlas_scores)}",
        f"- case_details: {len(detail_refs)}",
        "",
        "This v2 atlas consolidates v1/Phase266-272 rows into a primary path signature table, family-model score matrix, graph rows, and detail-on-demand JSON files.",
        "",
        "Priority: physical distribution atlas first, closure only after high-quality candidates are isolated.",
    ]

    write_json(V2 / "manifest.json", manifest)
    write_json(V2 / "schema.json", schema)
    write_json(V2 / "client_index.json", client_index)
    write_jsonl(V2 / "families.jsonl", [{**r, "schema_version": SCHEMA_VERSION} for r in families])
    write_jsonl(V2 / "modes.jsonl", [{**r, "schema_version": SCHEMA_VERSION} for r in modes])
    write_jsonl(V2 / "cases.jsonl", [{**r, "schema_version": SCHEMA_VERSION} for r in cases])
    write_jsonl(V2 / "path_signature_rows.jsonl", path_rows)
    write_jsonl(V2 / "atlas_scores.jsonl", atlas_scores)
    write_jsonl(V2 / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(V2 / "graph_edges.jsonl", graph_edges)
    (V2 / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({"phase": PHASE, "status": "complete", "path_signatures": len(path_rows), "atlas_scores": len(atlas_scores), "case_details": len(detail_refs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
