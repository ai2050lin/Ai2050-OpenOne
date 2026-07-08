#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


PHASE = 247
SOURCE_PHASE = 246
SCHEMA_VERSION = "1.0.0"
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
RESULT_ROOT = Path("tests/result/phase247_raw_vector_factor_decomposition")
ROUND_DEFAULT = "raw_vector_factor_decomposition"

VECTOR_NAMES = ["delta_down_out", "delta_product", "delta_residual"]
REGIME_GROUPS = {
    "boundary_regime": {"answer_boundary", "newline_boundary", "period_stop"},
    "continuation_regime": {"the_continuation", "be_continuation", "for_continuation", "comma_repeat"},
    "reason_regime": {"because_reason"},
    "other_regime": {"none"},
}


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
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denom


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def projection(vec: torch.Tensor, direction: torch.Tensor) -> dict[str, float]:
    if vec.numel() != direction.numel():
        return {"dot": 0.0, "cosine": 0.0, "signed_norm_fraction": 0.0}
    u = unit(direction)
    dot = float(torch.dot(vec.float(), u.float()).item())
    vec_norm = norm(vec)
    return {
        "dot": round(dot, 6),
        "cosine": round(dot / max(vec_norm, 1e-8), 6),
        "signed_norm_fraction": round(dot / max(vec_norm, 1e-8), 6),
    }


def gram_schmidt(directions: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    basis: list[tuple[str, torch.Tensor]] = []
    for name, vec in directions:
        v = vec.detach().float().cpu().clone()
        for _prev_name, b in basis:
            v = v - torch.dot(v, b) * b
        n = torch.linalg.vector_norm(v).item()
        if n > 1e-8:
            basis.append((name, v / n))
    return basis


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    validation_rows = read_jsonl(PHASE246_DIR / "phase246_causal_validation_rows.jsonl")
    summary = read_json(PHASE246_DIR / "phase246_cross_model_summary.json")
    if not manifest:
        raise FileNotFoundError(f"missing raw vector manifest under {PHASE246_DIR}")
    no_intervention = {key_for(row): row for row in validation_rows if row.get("intervention") == "no_intervention"}
    return manifest, validation_rows, no_intervention, summary


def load_vectors(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in manifest:
        path = Path(str(item.get("path")))
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        rows.append({**item, "payload": payload})
    return rows


def regime_group(winning_regime: str) -> str:
    for group, regimes in REGIME_GROUPS.items():
        if winning_regime in regimes:
            return group
    return "other_regime"


def build_regime_directions(vector_rows: list[dict[str, Any]], no_intervention: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for item in vector_rows:
        row = no_intervention.get(key_for(item), {})
        group = regime_group(str(row.get("winning_regime") or "none"))
        direction = item["payload"].get("competitor_direction")
        if torch.is_tensor(direction):
            buckets[(str(item.get("model")), group)].append(unit(direction))
    now = utc_now()
    out = []
    for (model, group), vectors in buckets.items():
        if not vectors:
            continue
        direction = unit(torch.stack(vectors).mean(dim=0))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "regime_direction_id": f"phase247:regime:{model}:{group}",
                "model": model,
                "regime_group": group,
                "source_direction": "mean_top_competitor_direction_from_phase246",
                "row_count": len(vectors),
                "direction_norm": round(norm(direction), 6),
                "direction_vector": direction,
            }
        )
    return out


def build_factor_direction_rows(vector_rows: list[dict[str, Any]], regime_dirs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    target_by_model: dict[str, list[torch.Tensor]] = defaultdict(list)
    competitor_by_model: dict[str, list[torch.Tensor]] = defaultdict(list)
    for item in vector_rows:
        target = item["payload"].get("target_direction")
        competitor = item["payload"].get("competitor_direction")
        if torch.is_tensor(target):
            target_by_model[str(item.get("model"))].append(unit(target))
        if torch.is_tensor(competitor):
            competitor_by_model[str(item.get("model"))].append(unit(competitor))
    for model, vectors in target_by_model.items():
        direction = unit(torch.stack(vectors).mean(dim=0))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "factor_direction_id": f"phase247:factor:{model}:target",
                "model": model,
                "factor": "target",
                "source": "mean_target_unembed_direction_from_phase246",
                "row_count": len(vectors),
                "direction_norm": round(norm(direction), 6),
                "status": "raw_direction_available",
            }
        )
    for model, vectors in competitor_by_model.items():
        direction = unit(torch.stack(vectors).mean(dim=0))
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "factor_direction_id": f"phase247:factor:{model}:top_competitor",
                "model": model,
                "factor": "top_competitor",
                "source": "mean_top_competitor_direction_from_phase246",
                "row_count": len(vectors),
                "direction_norm": round(norm(direction), 6),
                "status": "raw_direction_available_but_regime_incomplete",
            }
        )
    for reg in regime_dirs:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "factor_direction_id": reg["regime_direction_id"].replace(":regime:", ":factor:"),
                "model": reg["model"],
                "factor": reg["regime_group"],
                "source": reg["source_direction"],
                "row_count": reg["row_count"],
                "direction_norm": reg["direction_norm"],
                "status": "empirical_regime_direction_limited_by_phase246_samples",
            }
        )
    for model in sorted(set([str(x.get("model")) for x in vector_rows])):
        for factor in ["protocol", "boundary", "closure"]:
            out.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase247",
                    "created_at": now,
                    "factor_direction_id": f"phase247:factor:{model}:{factor}",
                    "model": model,
                    "factor": factor,
                    "source": "not_available_in_phase246_raw_vectors",
                    "row_count": 0,
                    "direction_norm": 0.0,
                    "status": "direction_gap_requires_new_raw_capture_or_regime_token_bank",
                }
            )
    return out


def intervention_effects(validation_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in validation_rows:
        key = key_for(row)
        intervention = str(row.get("intervention"))
        out[key][f"{intervention}_margin_delta"] = safe_float(row.get("target_margin_delta_vs_original"))
    return out


def projection_rows(
    vector_rows: list[dict[str, Any]],
    no_intervention: dict[tuple[str, str, str], dict[str, Any]],
    effects: dict[tuple[str, str, str], dict[str, float]],
    regime_dirs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now = utc_now()
    regime_by_model_group = {(x["model"], x["regime_group"]): x["direction_vector"] for x in regime_dirs}
    rows = []
    for item in vector_rows:
        payload = item["payload"]
        key = key_for(item)
        meta = no_intervention.get(key, {})
        group = regime_group(str(meta.get("winning_regime") or "none"))
        directions: dict[str, torch.Tensor] = {}
        if torch.is_tensor(payload.get("target_direction")):
            directions["target"] = unit(payload["target_direction"])
        if torch.is_tensor(payload.get("competitor_direction")):
            directions["top_competitor"] = unit(payload["competitor_direction"])
        if (str(item.get("model")), group) in regime_by_model_group:
            directions[group] = unit(regime_by_model_group[(str(item.get("model")), group)])
        orth_basis = dict(gram_schmidt(list(directions.items())))
        for vec_name in VECTOR_NAMES:
            vec = payload.get(vec_name)
            if not torch.is_tensor(vec):
                continue
            for direction_name, direction in directions.items():
                stats = projection(vec, direction)
                orth_stats = projection(vec, orth_basis[direction_name]) if direction_name in orth_basis else {"dot": 0.0, "cosine": 0.0}
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase247",
                        "created_at": now,
                        "projection_id": f"phase247:projection:{item['model']}:{item['case_id']}:{item['variant_id']}:{vec_name}:{direction_name}",
                        "model": item["model"],
                        "case_id": item["case_id"],
                        "variant_id": item["variant_id"],
                        "family_id": meta.get("family_id"),
                        "mode_id": meta.get("mode_id"),
                        "signature_class": meta.get("signature_class"),
                        "data_split": meta.get("data_split"),
                        "winning_regime": meta.get("winning_regime"),
                        "regime_group": group,
                        "vector_name": vec_name,
                        "direction_name": direction_name,
                        "vector_norm": round(norm(vec), 6),
                        "projection_dot": stats["dot"],
                        "projection_cosine": stats["cosine"],
                        "orth_projection_dot": orth_stats["dot"],
                        "orth_projection_cosine": orth_stats["cosine"],
                        **effects.get(key, {}),
                    }
                )
    rows.sort(key=lambda x: (x["model"], x["case_id"], x["variant_id"], x["vector_name"], x["direction_name"]))
    return rows


def prediction_rows(projections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    scopes = [
        ("all", lambda r: True),
        ("qwen3", lambda r: r.get("model") == "qwen3"),
        ("glm4", lambda r: r.get("model") == "glm4"),
        ("deepseek7b", lambda r: r.get("model") == "deepseek7b"),
        ("high_component_high_readout", lambda r: r.get("signature_class") == "high_component_high_readout"),
        ("high_component_low_readout", lambda r: r.get("signature_class") == "high_component_low_readout"),
    ]
    for scope, predicate in scopes:
        scoped = [r for r in projections if predicate(r)]
        for vector_name in VECTOR_NAMES:
            for direction_name in ["target", "top_competitor", "boundary_regime", "continuation_regime", "reason_regime", "other_regime"]:
                items = [r for r in scoped if r.get("vector_name") == vector_name and r.get("direction_name") == direction_name]
                if len(items) < 3:
                    continue
                proj = [safe_float(x.get("orth_projection_cosine")) for x in items]
                target_gain = [safe_float(x.get("target_unembed_injection_margin_delta")) for x in items]
                suppression_gain = [safe_float(x.get("top_competitor_suppression_margin_delta")) for x in items]
                ablation_gain = [safe_float(x.get("down_out_delta_ablation_margin_delta")) for x in items]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase247",
                        "created_at": now,
                        "prediction_id": f"phase247:prediction:{scope}:{vector_name}:{direction_name}",
                        "scope": scope,
                        "vector_name": vector_name,
                        "direction_name": direction_name,
                        "row_count": len(items),
                        "mean_projection_cosine": round(mean(proj), 6),
                        "corr_projection_target_injection_gain": round(pearson(proj, target_gain), 6),
                        "corr_projection_competitor_suppression_gain": round(pearson(proj, suppression_gain), 6),
                        "corr_projection_ablation_gain": round(pearson(proj, ablation_gain), 6),
                        "mean_target_injection_gain": round(mean(target_gain), 6),
                        "mean_competitor_suppression_gain": round(mean(suppression_gain), 6),
                        "mean_ablation_gain": round(mean(ablation_gain), 6),
                    }
                )
    rows.sort(key=lambda x: (x["scope"], x["vector_name"], x["direction_name"]))
    return rows


def candidate_rows(projections: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in projections:
        by_key[key_for(row)].append(row)
    rows = []
    for key, items in by_key.items():
        target_proj = max(
            [safe_float(x.get("orth_projection_cosine")) for x in items if x.get("direction_name") == "target"],
            default=0.0,
        )
        competitor_proj = max(
            [safe_float(x.get("orth_projection_cosine")) for x in items if "competitor" in str(x.get("direction_name"))],
            default=0.0,
        )
        regime_proj = max(
            [safe_float(x.get("orth_projection_cosine")) for x in items if str(x.get("direction_name")).endswith("_regime")],
            default=0.0,
        )
        meta = items[0]
        target_gain = safe_float(meta.get("target_unembed_injection_margin_delta"))
        suppression_gain = safe_float(meta.get("top_competitor_suppression_margin_delta"))
        ablation_gain = safe_float(meta.get("down_out_delta_ablation_margin_delta"))
        if target_gain >= 1.0 and target_proj > 0:
            route = "target_pressure_direction_candidate"
        elif suppression_gain >= 1.0 and competitor_proj > 0:
            route = "competitor_regime_candidate"
        elif ablation_gain <= -1.0:
            route = "component_necessity_candidate"
        elif ablation_gain >= 1.0:
            route = "component_opposite_signal_candidate"
        else:
            route = "mixed_or_weak_candidate"
        score = 0.35 * max(0.0, target_proj) + 0.25 * max(0.0, competitor_proj) + 0.2 * max(0.0, regime_proj)
        score += 0.1 * min(1.0, abs(target_gain) / 10.0) + 0.1 * min(1.0, abs(suppression_gain) / 10.0)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "candidate_id": f"phase247:regime_test:{key[0]}:{key[1]}:{key[2]}",
                "model": key[0],
                "case_id": key[1],
                "variant_id": key[2],
                "family_id": meta.get("family_id"),
                "mode_id": meta.get("mode_id"),
                "signature_class": meta.get("signature_class"),
                "data_split": meta.get("data_split"),
                "winning_regime": meta.get("winning_regime"),
                "target_projection_max": round(target_proj, 6),
                "competitor_projection_max": round(competitor_proj, 6),
                "regime_projection_max": round(regime_proj, 6),
                "target_injection_gain": target_gain,
                "competitor_suppression_gain": suppression_gain,
                "ablation_gain": ablation_gain,
                "candidate_route": route,
                "candidate_score": round(score, 6),
                "recommended_next_test": "regime_level_causal_test" if "competitor" in route else "direction_specific_component_test",
            }
        )
    rows.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
    return rows[:20]


def observation_rows(projections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in projections:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "observation_id": f"phase247:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['vector_name']}:{row['direction_name']}",
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "raw_vector_factor_projection",
                "component": row["vector_name"],
                "metric_name": f"projection_cosine:{row['direction_name']}",
                "metric_value": safe_float(row.get("orth_projection_cosine")),
                "metric_unit": "cosine",
                "signature_class": row.get("signature_class"),
                "winning_regime": row.get("winning_regime"),
            }
        )
    return rows


def metric_rows(predictions: list[dict[str, Any]], factor_rows: list[dict[str, Any]], summary_counts: dict[str, Any]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in predictions:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "metric_id": row["prediction_id"],
                "scope": "projection_prediction",
                "metric_name": "corr_projection_target_injection_gain",
                "metric_value": row["corr_projection_target_injection_gain"],
                "vector_name": row["vector_name"],
                "direction_name": row["direction_name"],
                "rows": row["row_count"],
            }
        )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase247",
            "created_at": now,
            "metric_id": "phase247:direction_gap_count",
            "scope": "direction_coverage",
            "metric_name": "direction_gap_count",
            "metric_value": summary_counts.get("direction_gap_count", 0),
            "rows": len(factor_rows),
        }
    )
    return rows


def graph_edges(predictions: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in sorted(predictions, key=lambda x: abs(safe_float(x.get("corr_projection_target_injection_gain"))), reverse=True)[:20]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "edge_id": f"phase247:prediction:{row['prediction_id']}",
                "source": f"projection:{row['vector_name']}:{row['direction_name']}",
                "target": "intervention:target_unembed_injection",
                "edge_type": "projection_prediction",
                "evidence_type": "raw_vector_projection_correlation",
                "effect_direction": "positive" if safe_float(row.get("corr_projection_target_injection_gain")) >= 0 else "negative",
                "effect_size": safe_float(row.get("corr_projection_target_injection_gain")),
                "confidence": 0.42,
                "supporting_phases": ["Phase246", "Phase247"],
                "status": "direction_analysis_not_closure",
            }
        )
    for row in candidates[:12]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase247",
                "created_at": now,
                "edge_id": f"phase247:candidate:{row['candidate_id']}",
                "source": f"raw_delta:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "target": f"next_test:{row['recommended_next_test']}",
                "edge_type": "next_regime_test_candidate",
                "evidence_type": "projection_plus_phase246_intervention",
                "effect_direction": row["candidate_route"],
                "effect_size": row["candidate_score"],
                "confidence": 0.46,
                "supporting_phases": ["Phase246", "Phase247"],
                "status": "candidate_not_closure",
            }
        )
    return rows


def update_pattern_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress_path = ATLAS_ROOT / "progress.json"
    progress = read_json(progress_path)
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase247",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "build regime-level direction banks before more suppression tests",
            "small_model_bias_warning": "Phase247 uses Phase246 raw vectors from qwen3/glm4/deepseek7b; target and top-competitor directions are real saved vectors, but protocol/boundary/closure remain direction gaps.",
        }
    )
    write_json(progress_path, progress)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase247 raw-vector factor direction decomposition",
        "",
        "Phase247 analyzes Phase246 raw delta vectors. It does not run new model forwards and does not claim closure.",
        "",
        "## Counts",
        "",
        f"- raw_vector_rows: {summary['raw_vector_rows']}",
        f"- factor_direction_rows: {summary['factor_direction_rows']}",
        f"- regime_direction_rows: {summary['regime_direction_rows']}",
        f"- projection_rows: {summary['projection_rows']}",
        f"- prediction_rows: {summary['prediction_rows']}",
        f"- next_test_candidate_rows: {summary['next_test_candidate_rows']}",
        "",
        "## Key Signals",
        "",
        f"- best_target_prediction_corr: {summary['best_target_prediction_corr']}",
        f"- best_competitor_suppression_prediction_corr: {summary['best_competitor_suppression_prediction_corr']}",
        f"- direction_gap_count: {summary['direction_gap_count']}",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation, no_intervention, phase246_summary = load_inputs()
    vector_rows = load_vectors(manifest)
    regime_dirs = build_regime_directions(vector_rows, no_intervention)
    factor_rows = build_factor_direction_rows(vector_rows, regime_dirs)
    effects = intervention_effects(validation)
    projections = projection_rows(vector_rows, no_intervention, effects, regime_dirs)
    predictions = prediction_rows(projections)
    candidates = candidate_rows(projections, predictions)
    observations = observation_rows(projections)
    summary_counts = {"direction_gap_count": sum(1 for x in factor_rows if "direction_gap" in str(x.get("status")))}
    metrics = metric_rows(predictions, factor_rows, summary_counts)
    edges = graph_edges(predictions, candidates)
    best_target = max([abs(safe_float(x.get("corr_projection_target_injection_gain"))) for x in predictions], default=0.0)
    best_suppression = max([abs(safe_float(x.get("corr_projection_competitor_suppression_gain"))) for x in predictions], default=0.0)
    progress = {
        "pattern_family_atlas": 0.75,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.39,
        "high_value_trace_selection": 0.61,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.36,
        "focused_causal_validation": 0.22,
        "raw_delta_vector_archive": 0.24,
        "raw_vector_factor_decomposition": 0.20,
        "regime_field_direction_bank": 0.10,
        "gate_up_product_signature": 0.45,
        "residual_state_signature": 0.42,
        "readout_competition_trace": 0.64,
        "stepwise_rollout_trace": 0.23,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.56,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Raw-vector factor direction decomposition",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "phase246_status": phase246_summary.get("status"),
        "raw_vector_rows": len(vector_rows),
        "factor_direction_rows": len(factor_rows),
        "regime_direction_rows": len(regime_dirs),
        "projection_rows": len(projections),
        "prediction_rows": len(predictions),
        "next_test_candidate_rows": len(candidates),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "direction_gap_count": summary_counts["direction_gap_count"],
        "best_target_prediction_corr": round(best_target, 6),
        "best_competitor_suppression_prediction_corr": round(best_suppression, 6),
        "candidate_route_counts": dict(Counter(str(x.get("candidate_route")) for x in candidates).most_common()),
        "regime_direction_counts": dict(Counter(str(x.get("regime_group")) for x in regime_dirs).most_common()),
        "pattern_atlas_progress": progress,
        "judgement": "raw_direction_decomposition_not_closure",
        "limitations": [
            "No new model forward passes were run.",
            "Target and top-competitor directions are real saved vectors from Phase246.",
            "Boundary/protocol/closure directions remain unavailable as raw directions and must not be treated as solved.",
            "Empirical regime directions are limited by the 15 Phase246 candidates.",
        ],
    }
    write_json(out_dir / "phase247_factor_decomposition_summary.json", payload)
    write_jsonl(out_dir / "phase247_factor_direction_rows.jsonl", [{k: v for k, v in r.items() if k != "direction_vector"} for r in factor_rows])
    write_jsonl(out_dir / "phase247_regime_direction_rows.jsonl", [{k: v for k, v in r.items() if k != "direction_vector"} for r in regime_dirs])
    write_jsonl(out_dir / "phase247_raw_delta_projection_rows.jsonl", projections)
    write_jsonl(out_dir / "phase247_projection_prediction_rows.jsonl", predictions)
    write_jsonl(out_dir / "phase247_regime_test_candidate_rows.jsonl", candidates)
    write_jsonl(out_dir / "phase247_observations.jsonl", observations)
    write_jsonl(out_dir / "phase247_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase247_graph_edges.jsonl", edges)
    write_report(out_dir / "phase247_factor_direction_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase247 raw-vector factor direction decomposition")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.round_name)


if __name__ == "__main__":
    main()
