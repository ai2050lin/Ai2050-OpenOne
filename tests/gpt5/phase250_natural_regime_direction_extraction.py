#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
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
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402


PHASE = 250
SOURCE_PHASE = 249
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase250_natural_regime_direction_extraction")
ROUND_DEFAULT = "natural_regime_direction_extraction"

SPECS = {
    "qwen3": {"observe_layer": 33},
    "glm4": {"observe_layer": 32},
    "deepseek7b": {"observe_layer": 27},
}

CONTRASTS = {
    "natural_protocol_short": ("one_word_strict", "explain_instruction"),
    "natural_continuation_explain": ("explain_instruction", "one_word_strict"),
    "natural_answer_boundary": ("full", "no_answer_anchor"),
    "natural_target_seed": ("target_seeded", "full"),
    "natural_concise_answer": ("short_answer_instruction", "explain_instruction"),
}

VECTOR_NAMES = ["delta_residual", "delta_down_out", "delta_product"]


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


def vector_norm(vec: torch.Tensor | None) -> float:
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
    if vector_norm(a) <= 1e-8 or vector_norm(b) <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


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


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_rows(model_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    return [x for x in rows if str(x.get("model")) == model_name]


def select_case_ids(rows: list[dict[str, Any]], per_family: int) -> dict[str, list[str]]:
    by_case: dict[str, dict[str, Any]] = {}
    variants_by_case: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_id = str(row["case_id"])
        variants_by_case[case_id].add(str(row["variant_id"]))
        by_case.setdefault(case_id, row)
    out: dict[str, list[str]] = defaultdict(list)
    needed = {variant for pair in CONTRASTS.values() for variant in pair}
    for case_id in sorted(variants_by_case):
        if not needed.issubset(variants_by_case[case_id]):
            continue
        family_id = str(by_case[case_id].get("family_id"))
        if len(out[family_id]) < per_family:
            out[family_id].append(case_id)
    return out


def load_raw_vectors(model_name: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, float]]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    validation = read_jsonl(PHASE246_DIR / "phase246_causal_validation_rows.jsonl")
    vectors = []
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if path.exists():
            vectors.append({**item, "payload": torch.load(path, map_location="cpu")})
    effects: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in validation:
        if str(row.get("model")) != model_name:
            continue
        effects[key_for(row)][f"{row.get('intervention')}_margin_delta"] = safe_float(row.get("target_margin_delta_vs_original"))
    return vectors, effects


def capture_hidden(model: Any, tokenizer: Any, device: torch.device, prompt: str, layer_idx: int) -> torch.Tensor:
    _internal, hidden, _logits = p228.capture_internal(model, tokenizer, device, prompt, [], [int(layer_idx)])
    vec = hidden.get(int(layer_idx))
    if not torch.is_tensor(vec):
        raise RuntimeError(f"missing hidden state at layer {layer_idx}")
    return vec.float().cpu()


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    vector_dir = out_dir / "natural_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    rows = load_behavior_rows(args.model)
    by_variant = {(str(x["case_id"]), str(x["variant_id"])): x for x in rows}
    selected_by_family = select_case_ids(rows, int(args.cases_per_family))
    selected_cases = [case_id for family in sorted(selected_by_family) for case_id in selected_by_family[family]]
    layer_idx = int(SPECS[args.model]["observe_layer"])
    sample_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    bucket: dict[str, list[torch.Tensor]] = defaultdict(list)
    family_bucket: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    model = None
    tokenizer = None
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        total = len(selected_cases) * len(CONTRASTS)
        done = 0
        for case_id in selected_cases:
            family_id = str(by_variant[(case_id, "full")].get("family_id"))
            mode_id = str(by_variant[(case_id, "full")].get("mode_id"))
            for contrast_id, (positive_variant, negative_variant) in CONTRASTS.items():
                pos = by_variant.get((case_id, positive_variant))
                neg = by_variant.get((case_id, negative_variant))
                if pos is None or neg is None:
                    missing_rows.append({"model": args.model, "case_id": case_id, "contrast_id": contrast_id, "reason": "missing_contrast_variant"})
                    continue
                try:
                    pos_hidden = capture_hidden(model, tokenizer, device, str(pos["prompt_variant"]), layer_idx)
                    neg_hidden = capture_hidden(model, tokenizer, device, str(neg["prompt_variant"]), layer_idx)
                except Exception as exc:  # noqa: BLE001
                    missing_rows.append({"model": args.model, "case_id": case_id, "contrast_id": contrast_id, "reason": str(exc)})
                    continue
                delta = pos_hidden - neg_hidden
                bucket[contrast_id].append(delta)
                family_bucket[(contrast_id, family_id)].append(delta)
                sample_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase250",
                        "created_at": utc_now(),
                        "sample_id": f"phase250:sample:{args.model}:{case_id}:{contrast_id}",
                        "model": args.model,
                        "case_id": case_id,
                        "family_id": family_id,
                        "mode_id": mode_id,
                        "contrast_id": contrast_id,
                        "positive_variant": positive_variant,
                        "negative_variant": negative_variant,
                        "observe_layer": layer_idx,
                        "delta_norm": round(vector_norm(delta), 6),
                    }
                )
                done += 1
                if done % 20 == 0 or done == total:
                    log(f"{args.model}: natural contrast {done}/{total}")
        for contrast_id, vecs in bucket.items():
            direction = unit(torch.stack(vecs).mean(dim=0)) if vecs else torch.zeros(1)
            path = vector_dir / f"{safe_slug(args.model + '_' + contrast_id + '_global')}.pt"
            torch.save({"model": args.model, "contrast_id": contrast_id, "scope": "global", "direction": direction}, path)
            direction_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase250",
                    "created_at": utc_now(),
                    "direction_id": f"phase250:natural:{args.model}:{contrast_id}:global",
                    "model": args.model,
                    "contrast_id": contrast_id,
                    "scope": "global",
                    "family_id": "all",
                    "observe_layer": layer_idx,
                    "sample_count": len(vecs),
                    "direction_norm": round(vector_norm(direction), 6),
                    "mean_sample_delta_norm": round(mean(vector_norm(x) for x in vecs), 6) if vecs else 0.0,
                    "path": str(path),
                }
            )
        for (contrast_id, family_id), vecs in family_bucket.items():
            direction = unit(torch.stack(vecs).mean(dim=0)) if vecs else torch.zeros(1)
            path = vector_dir / f"{safe_slug(args.model + '_' + contrast_id + '_' + family_id)}.pt"
            torch.save({"model": args.model, "contrast_id": contrast_id, "scope": "family", "family_id": family_id, "direction": direction}, path)
            direction_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase250",
                    "created_at": utc_now(),
                    "direction_id": f"phase250:natural:{args.model}:{contrast_id}:{family_id}",
                    "model": args.model,
                    "contrast_id": contrast_id,
                    "scope": "family",
                    "family_id": family_id,
                    "observe_layer": layer_idx,
                    "sample_count": len(vecs),
                    "direction_norm": round(vector_norm(direction), 6),
                    "mean_sample_delta_norm": round(mean(vector_norm(x) for x in vecs), 6) if vecs else 0.0,
                    "path": str(path),
                }
            )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    global_directions = {}
    for row in direction_rows:
        if row["scope"] == "global":
            payload = torch.load(row["path"], map_location="cpu")
            global_directions[str(row["contrast_id"])] = payload["direction"]
    raw_vectors, effects = load_raw_vectors(args.model)
    for item in raw_vectors:
        key = key_for(item)
        payload = item["payload"]
        for vector_name in VECTOR_NAMES:
            vec = payload.get(vector_name)
            if not torch.is_tensor(vec):
                continue
            for contrast_id, direction in global_directions.items():
                projection_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase250",
                        "created_at": utc_now(),
                        "projection_id": f"phase250:projection:{args.model}:{key[1]}:{key[2]}:{vector_name}:{contrast_id}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "vector_name": vector_name,
                        "contrast_id": contrast_id,
                        "projection_cosine": round(cosine(vec, direction), 6),
                        "vector_norm": round(vector_norm(vec), 6),
                        **effects.get(key, {}),
                    }
                )
    for vector_name in VECTOR_NAMES:
        for contrast_id in CONTRASTS:
            items = [x for x in projection_rows if x["vector_name"] == vector_name and x["contrast_id"] == contrast_id]
            if len(items) < 3:
                continue
            proj = [safe_float(x.get("projection_cosine")) for x in items]
            target_gain = [safe_float(x.get("target_unembed_injection_margin_delta")) for x in items]
            suppression_gain = [safe_float(x.get("top_competitor_suppression_margin_delta")) for x in items]
            prediction_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase250",
                    "created_at": utc_now(),
                    "prediction_id": f"phase250:prediction:{args.model}:{vector_name}:{contrast_id}",
                    "model": args.model,
                    "vector_name": vector_name,
                    "contrast_id": contrast_id,
                    "row_count": len(items),
                    "mean_projection_cosine": round(mean(proj), 6),
                    "corr_projection_target_injection_gain": round(pearson(proj, target_gain), 6),
                    "corr_projection_competitor_suppression_gain": round(pearson(proj, suppression_gain), 6),
                    "mean_target_injection_gain": round(mean(target_gain), 6),
                    "mean_competitor_suppression_gain": round(mean(suppression_gain), 6),
                }
            )
    observations = observation_rows(projection_rows)
    metrics = metric_rows(direction_rows, prediction_rows)
    edges = graph_edges(prediction_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Natural regime direction extraction",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "selected_case_count": len(selected_cases),
        "sample_rows": len(sample_rows),
        "direction_rows": len(direction_rows),
        "projection_rows": len(projection_rows),
        "prediction_rows": len(prediction_rows),
        "missing_rows": len(missing_rows),
        "contrast_counts": dict(Counter(x["contrast_id"] for x in sample_rows).most_common()),
        "family_counts": {k: len(v) for k, v in selected_by_family.items()},
    }
    write_json(out_dir / f"phase250_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase250_{args.model}_natural_direction_sample_rows.jsonl", sample_rows)
    write_jsonl(out_dir / f"phase250_{args.model}_natural_direction_rows.jsonl", direction_rows)
    write_jsonl(out_dir / f"phase250_{args.model}_natural_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / f"phase250_{args.model}_natural_prediction_rows.jsonl", prediction_rows)
    write_jsonl(out_dir / f"phase250_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase250_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase250_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase250_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase250",
                "created_at": now,
                "observation_id": f"phase250:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['vector_name']}:{row['contrast_id']}",
                "case_id": row["case_id"],
                "model": row["model"],
                "variant_id": row["variant_id"],
                "level": "natural_regime_projection",
                "component": row["vector_name"],
                "metric_name": f"projection_cosine:{row['contrast_id']}",
                "metric_value": safe_float(row.get("projection_cosine")),
                "metric_unit": "cosine",
            }
        )
    return out


def metric_rows(direction_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in direction_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase250",
                "created_at": now,
                "metric_id": row["direction_id"],
                "scope": "natural_regime_direction",
                "metric_name": "sample_count",
                "metric_value": row["sample_count"],
                "model": row["model"],
                "contrast_id": row["contrast_id"],
                "family_id": row["family_id"],
                "rows": row["sample_count"],
            }
        )
    for row in prediction_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase250",
                "created_at": now,
                "metric_id": row["prediction_id"],
                "scope": "natural_projection_prediction",
                "metric_name": "corr_projection_competitor_suppression_gain",
                "metric_value": row["corr_projection_competitor_suppression_gain"],
                "model": row["model"],
                "vector_name": row["vector_name"],
                "contrast_id": row["contrast_id"],
                "rows": row["row_count"],
            }
        )
    return rows


def graph_edges(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in sorted(prediction_rows, key=lambda x: abs(safe_float(x.get("corr_projection_competitor_suppression_gain"))), reverse=True)[:20]:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase250",
                "created_at": now,
                "edge_id": f"phase250:prediction:{row['prediction_id']}",
                "source": f"natural_direction:{row['contrast_id']}",
                "target": "intervention:competitor_suppression",
                "edge_type": "natural_regime_projection_prediction",
                "model": row["model"],
                "evidence_type": "natural_contrast_projection_correlation",
                "effect_direction": "positive" if safe_float(row.get("corr_projection_competitor_suppression_gain")) >= 0 else "negative",
                "effect_size": safe_float(row.get("corr_projection_competitor_suppression_gain")),
                "confidence": 0.42,
                "supporting_phases": ["Phase241", "Phase246", "Phase249", "Phase250"],
                "status": "natural_direction_candidate_not_closure",
            }
        )
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase250_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    sample_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        sample_rows.extend(read_jsonl(out_dir / f"phase250_{model}_natural_direction_sample_rows.jsonl"))
        direction_rows.extend(read_jsonl(out_dir / f"phase250_{model}_natural_direction_rows.jsonl"))
        projection_rows.extend(read_jsonl(out_dir / f"phase250_{model}_natural_projection_rows.jsonl"))
        prediction_rows.extend(read_jsonl(out_dir / f"phase250_{model}_natural_prediction_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase250_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase250_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase250_{model}_graph_edges.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase250_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.78,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.40,
        "high_value_trace_selection": 0.63,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.36,
        "focused_causal_validation": 0.23,
        "raw_delta_vector_archive": 0.26,
        "raw_vector_factor_decomposition": 0.23,
        "regime_field_direction_bank": 0.30,
        "natural_regime_direction_bank": 0.20,
        "regime_level_causal_validation": 0.18,
        "gate_up_product_signature": 0.45,
        "residual_state_signature": 0.45,
        "readout_competition_trace": 0.68,
        "stepwise_rollout_trace": 0.24,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.59,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model natural regime direction extraction",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "sample_rows": len(sample_rows),
        "direction_rows": len(direction_rows),
        "projection_rows": len(projection_rows),
        "prediction_rows": len(prediction_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "contrast_counts": dict(Counter(str(x.get("contrast_id")) for x in sample_rows).most_common()),
        "top_prediction_edges": sorted(
            [
                {
                    "model": x.get("model"),
                    "vector_name": x.get("vector_name"),
                    "contrast_id": x.get("contrast_id"),
                    "rows": x.get("row_count"),
                    "corr_projection_competitor_suppression_gain": x.get("corr_projection_competitor_suppression_gain"),
                }
                for x in prediction_rows
            ],
            key=lambda x: abs(safe_float(x.get("corr_projection_competitor_suppression_gain"))),
            reverse=True,
        )[:10],
        "pattern_atlas_progress": progress,
        "judgement": "natural_direction_bank_created_not_causal_closure",
    }
    write_json(out_dir / "phase250_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase250_natural_direction_sample_rows.jsonl", sample_rows)
    write_jsonl(out_dir / "phase250_natural_direction_rows.jsonl", direction_rows)
    write_jsonl(out_dir / "phase250_natural_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / "phase250_natural_prediction_rows.jsonl", prediction_rows)
    write_jsonl(out_dir / "phase250_observations.jsonl", observations)
    write_jsonl(out_dir / "phase250_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase250_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase250_missing_rows.jsonl", missing_rows)
    write_report(out_dir / "phase250_natural_regime_direction_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase250 natural regime direction extraction",
        "",
        "Phase250 replaces static token-bank directions with natural contrast directions from Phase241 variants.",
        "It is direction extraction and projection calibration, not causal closure.",
        "",
        "## Counts",
        "",
        f"- sample_rows: {summary['sample_rows']}",
        f"- direction_rows: {summary['direction_rows']}",
        f"- projection_rows: {summary['projection_rows']}",
        f"- prediction_rows: {summary['prediction_rows']}",
        "",
        "## Top Prediction Edges",
        "",
        "```json",
        json.dumps(summary["top_prediction_edges"], ensure_ascii=False, indent=2),
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
            "latest_phase": "Phase250",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "use natural contrast directions for next causal validation instead of token-bank proxy directions",
            "small_model_bias_warning": "Phase250 natural directions are extracted from qwen3/glm4/deepseek7b small models; they are atlas evidence, not language-mechanism closure.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase250 natural regime direction extraction")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-family", type=int, default=2)
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
    evaluate_model(args)


if __name__ == "__main__":
    main()
