#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = 248
SOURCE_PHASE = 247
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
RESULT_ROOT = Path("tests/result/phase248_regime_level_direction_bank")
ROUND_DEFAULT = "regime_level_direction_bank"
VECTOR_NAMES = ["delta_down_out", "delta_product", "delta_residual"]

REGIME_TEXTS = {
    "continuation_regime": [" the", " The", " is", " are", " be", " for", " For", " and", " which", " object"],
    "answer_boundary_regime": ["Answer", " Answer", " answer", ":", ":\n", "\nAnswer", " final"],
    "newline_boundary_regime": ["\n", "\n\n", " \n", "\n-", "\n1"],
    "period_stop_regime": [".", " .", ".\n", ". ", "。"],
    "because_reason_regime": [" because", " Because", " since", " therefore", " reason", " why"],
    "comma_repeat_regime": [",", " ,", ", ", " and", "，"],
    "protocol_short_regime": [" one", " word", " only", " exactly", " short", " final"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def projection_cosine(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    denom = norm(vec) * norm(direction)
    if denom <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(vec.float(), direction.float(), dim=0).item())


def token_ids_for_texts(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def get_output_weight(model: Any) -> torch.Tensor:
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("output embedding weight not found")
    return weight.detach().float().cpu()


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_raw_vectors_for_model(model_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, float]]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    validation = read_jsonl(PHASE246_DIR / "phase246_causal_validation_rows.jsonl")
    model_manifest = [x for x in manifest if str(x.get("model")) == model_name]
    vectors = []
    for item in model_manifest:
        path = Path(str(item.get("path")))
        if path.exists():
            vectors.append({**item, "payload": torch.load(path, map_location="cpu")})
    effects: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in validation:
        if str(row.get("model")) != model_name:
            continue
        effects[key_for(row)][f"{row.get('intervention')}_margin_delta"] = safe_float(row.get("target_margin_delta_vs_original"))
    return vectors, model_manifest, effects


def build_bank_for_model(model_name: str, model: Any, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    output_weight = get_output_weight(model)
    now = utc_now()
    rows = []
    directions: dict[str, torch.Tensor] = {}
    for regime, texts in REGIME_TEXTS.items():
        token_ids = token_ids_for_texts(tokenizer, texts)
        if regime == "period_stop_regime" and tokenizer.eos_token_id is not None:
            token_ids = sorted(set(token_ids + [int(tokenizer.eos_token_id)]))
        vectors = [unit(output_weight[token_id]) for token_id in token_ids if 0 <= token_id < output_weight.shape[0]]
        if vectors:
            direction = unit(torch.stack(vectors).mean(dim=0))
        else:
            direction = torch.zeros(output_weight.shape[1])
        directions[regime] = direction
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "regime_bank_id": f"phase248:bank:{model_name}:{regime}",
                "model": model_name,
                "regime": regime,
                "source": "token_bank_output_embedding_mean",
                "texts": texts,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "direction_norm": round(norm(direction), 6),
                "coverage_status": "available" if token_ids else "missing_tokens",
            }
        )
    return rows, directions


def projection_rows(model_name: str, vectors: list[dict[str, Any]], directions: dict[str, torch.Tensor], effects: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for item in vectors:
        payload = item["payload"]
        for vector_name in VECTOR_NAMES:
            vec = payload.get(vector_name)
            if not torch.is_tensor(vec):
                continue
            for regime, direction in directions.items():
                key = key_for(item)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase248",
                        "created_at": now,
                        "projection_id": f"phase248:projection:{model_name}:{item['case_id']}:{item['variant_id']}:{vector_name}:{regime}",
                        "model": model_name,
                        "case_id": item["case_id"],
                        "variant_id": item["variant_id"],
                        "vector_name": vector_name,
                        "regime": regime,
                        "vector_norm": round(norm(vec), 6),
                        "projection_cosine": round(projection_cosine(vec, direction), 6),
                        "projection_dot": round(float(torch.dot(vec.float(), unit(direction).float()).item()), 6) if vec.numel() == direction.numel() else 0.0,
                        **effects.get(key, {}),
                    }
                )
    return rows


def prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    scopes = ["all", *sorted(set(str(x.get("model")) for x in rows))]
    for scope in scopes:
        scoped = rows if scope == "all" else [x for x in rows if str(x.get("model")) == scope]
        for vector_name in VECTOR_NAMES:
            for regime in REGIME_TEXTS:
                items = [x for x in scoped if x.get("vector_name") == vector_name and x.get("regime") == regime]
                if len(items) < 3:
                    continue
                proj = [safe_float(x.get("projection_cosine")) for x in items]
                target_gain = [safe_float(x.get("target_unembed_injection_margin_delta")) for x in items]
                suppression_gain = [safe_float(x.get("top_competitor_suppression_margin_delta")) for x in items]
                ablation_gain = [safe_float(x.get("down_out_delta_ablation_margin_delta")) for x in items]
                out.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase248",
                        "created_at": now,
                        "prediction_id": f"phase248:prediction:{scope}:{vector_name}:{regime}",
                        "scope": scope,
                        "vector_name": vector_name,
                        "regime": regime,
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
    out.sort(key=lambda x: (x["scope"], x["vector_name"], x["regime"]))
    return out


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key_for(row)].append(row)
    out = []
    for key, items in buckets.items():
        continuation = max([safe_float(x.get("projection_cosine")) for x in items if x.get("regime") == "continuation_regime"], default=0.0)
        boundary = max([safe_float(x.get("projection_cosine")) for x in items if "boundary" in str(x.get("regime")) or x.get("regime") == "period_stop_regime"], default=0.0)
        reason = max([safe_float(x.get("projection_cosine")) for x in items if x.get("regime") == "because_reason_regime"], default=0.0)
        protocol = max([safe_float(x.get("projection_cosine")) for x in items if x.get("regime") == "protocol_short_regime"], default=0.0)
        meta = items[0]
        suppression_gain = safe_float(meta.get("top_competitor_suppression_margin_delta"))
        target_gain = safe_float(meta.get("target_unembed_injection_margin_delta"))
        route_scores = {
            "continuation_regime_test": continuation + max(0.0, abs(suppression_gain) / 20.0),
            "boundary_regime_test": boundary + max(0.0, abs(suppression_gain) / 25.0),
            "reason_regime_test": reason + max(0.0, abs(target_gain) / 30.0),
            "protocol_regime_test": protocol + max(0.0, abs(target_gain) / 30.0),
        }
        route, score = max(route_scores.items(), key=lambda x: x[1])
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "candidate_id": f"phase248:regime:{key[0]}:{key[1]}:{key[2]}",
                "model": key[0],
                "case_id": key[1],
                "variant_id": key[2],
                "continuation_projection": round(continuation, 6),
                "boundary_projection": round(boundary, 6),
                "reason_projection": round(reason, 6),
                "protocol_projection": round(protocol, 6),
                "target_injection_gain": target_gain,
                "competitor_suppression_gain": suppression_gain,
                "recommended_next_test": route,
                "candidate_score": round(score, 6),
            }
        )
    out.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
    return out


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "observation_id": f"phase248:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['vector_name']}:{row['regime']}",
                "case_id": row["case_id"],
                "model": row["model"],
                "variant_id": row["variant_id"],
                "level": "regime_direction_projection",
                "component": row["vector_name"],
                "metric_name": f"projection_cosine:{row['regime']}",
                "metric_value": safe_float(row.get("projection_cosine")),
                "metric_unit": "cosine",
            }
        )
    return out


def metric_rows(predictions: list[dict[str, Any]], bank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in predictions:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "metric_id": row["prediction_id"],
                "scope": "regime_projection_prediction",
                "metric_name": "corr_projection_competitor_suppression_gain",
                "metric_value": row["corr_projection_competitor_suppression_gain"],
                "vector_name": row["vector_name"],
                "regime": row["regime"],
                "rows": row["row_count"],
            }
        )
    for row in bank_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "metric_id": row["regime_bank_id"],
                "scope": "regime_direction_bank",
                "metric_name": "token_count",
                "metric_value": row["token_count"],
                "model": row["model"],
                "regime": row["regime"],
                "rows": row["token_count"],
            }
        )
    return rows


def graph_edges(predictions: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in sorted(predictions, key=lambda x: abs(safe_float(x.get("corr_projection_competitor_suppression_gain"))), reverse=True)[:20]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "edge_id": f"phase248:prediction:{row['prediction_id']}",
                "source": f"regime:{row['regime']}",
                "target": "intervention:competitor_suppression",
                "edge_type": "regime_projection_prediction",
                "evidence_type": "token_bank_projection_correlation",
                "effect_direction": "positive" if safe_float(row.get("corr_projection_competitor_suppression_gain")) >= 0 else "negative",
                "effect_size": safe_float(row.get("corr_projection_competitor_suppression_gain")),
                "confidence": 0.40,
                "supporting_phases": ["Phase246", "Phase247", "Phase248"],
                "status": "regime_bank_candidate_not_closure",
            }
        )
    for row in candidates[:12]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase248",
                "created_at": now,
                "edge_id": f"phase248:candidate:{row['candidate_id']}",
                "source": f"raw_delta:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "target": f"next_test:{row['recommended_next_test']}",
                "edge_type": "regime_level_test_candidate",
                "evidence_type": "regime_projection_plus_phase246_intervention",
                "effect_direction": row["recommended_next_test"],
                "effect_size": row["candidate_score"],
                "confidence": 0.45,
                "supporting_phases": ["Phase246", "Phase247", "Phase248"],
                "status": "candidate_not_closure",
            }
        )
    return rows


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    try:
        model, tokenizer, _device, _attn = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bank_rows, directions = build_bank_for_model(args.model, model, tokenizer)
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vectors, manifest, effects = load_raw_vectors_for_model(args.model)
    projections = projection_rows(args.model, vectors, directions, effects)
    predictions = prediction_rows(projections)
    candidates = candidate_rows(projections)
    observations = observation_rows(projections)
    metrics = metric_rows(predictions, bank_rows)
    edges = graph_edges(predictions, candidates)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Regime-level direction bank",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "regime_bank_rows": len(bank_rows),
        "raw_vector_rows": len(vectors),
        "projection_rows": len(projections),
        "prediction_rows": len(predictions),
        "regime_test_candidate_rows": len(candidates),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "token_bank_coverage": {row["regime"]: row["token_count"] for row in bank_rows},
        "top_candidate_routes": dict(Counter(str(x.get("recommended_next_test")) for x in candidates[:10]).most_common()),
    }
    write_json(out_dir / f"phase248_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase248_{args.model}_regime_direction_rows.jsonl", bank_rows)
    write_jsonl(out_dir / f"phase248_{args.model}_regime_projection_rows.jsonl", projections)
    write_jsonl(out_dir / f"phase248_{args.model}_projection_prediction_rows.jsonl", predictions)
    write_jsonl(out_dir / f"phase248_{args.model}_regime_test_candidate_rows.jsonl", candidates)
    write_jsonl(out_dir / f"phase248_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase248_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase248_{args.model}_graph_edges.jsonl", edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase248_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    bank_rows: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for model in MODELS:
        bank_rows.extend(read_jsonl(out_dir / f"phase248_{model}_regime_direction_rows.jsonl"))
        projections.extend(read_jsonl(out_dir / f"phase248_{model}_regime_projection_rows.jsonl"))
        predictions.extend(read_jsonl(out_dir / f"phase248_{model}_projection_prediction_rows.jsonl"))
        candidates.extend(read_jsonl(out_dir / f"phase248_{model}_regime_test_candidate_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase248_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase248_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase248_{model}_graph_edges.jsonl"))
    progress = {
        "pattern_family_atlas": 0.76,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.39,
        "high_value_trace_selection": 0.62,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.36,
        "focused_causal_validation": 0.22,
        "raw_delta_vector_archive": 0.25,
        "raw_vector_factor_decomposition": 0.22,
        "regime_field_direction_bank": 0.22,
        "gate_up_product_signature": 0.45,
        "residual_state_signature": 0.42,
        "readout_competition_trace": 0.66,
        "stepwise_rollout_trace": 0.23,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.57,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model regime-level direction bank",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "regime_bank_rows": len(bank_rows),
        "raw_vector_rows": sum(int(x.get("raw_vector_rows") or 0) for x in summaries),
        "projection_rows": len(projections),
        "prediction_rows": len(predictions),
        "regime_test_candidate_rows": len(candidates),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "candidate_route_counts": dict(Counter(str(x.get("recommended_next_test")) for x in candidates).most_common()),
        "pattern_atlas_progress": progress,
        "judgement": "regime_direction_bank_not_closure",
    }
    write_json(out_dir / "phase248_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase248_regime_direction_rows.jsonl", bank_rows)
    write_jsonl(out_dir / "phase248_regime_projection_rows.jsonl", projections)
    write_jsonl(out_dir / "phase248_projection_prediction_rows.jsonl", predictions)
    write_jsonl(out_dir / "phase248_regime_test_candidate_rows.jsonl", candidates)
    write_jsonl(out_dir / "phase248_observations.jsonl", observations)
    write_jsonl(out_dir / "phase248_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase248_graph_edges.jsonl", edges)
    write_report(out_dir / "phase248_regime_direction_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase248 regime-level direction bank",
        "",
        "Phase248 builds token-bank regime directions and reprojects Phase246 raw deltas. It is not closure validation.",
        "",
        "## Counts",
        "",
        f"- regime_bank_rows: {summary['regime_bank_rows']}",
        f"- projection_rows: {summary['projection_rows']}",
        f"- prediction_rows: {summary['prediction_rows']}",
        f"- regime_test_candidate_rows: {summary['regime_test_candidate_rows']}",
        "",
        "## Candidate Routes",
        "",
        "```json",
        json.dumps(summary["candidate_route_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "latest_phase": "Phase248",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "run small regime-level causal tests on bank-selected candidates",
            "small_model_bias_warning": "Phase248 regime banks are token-bank output directions in qwen3/glm4/deepseek7b and are not full natural regime fields.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase248 regime-level direction bank")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    eval_model(args)


if __name__ == "__main__":
    main()
