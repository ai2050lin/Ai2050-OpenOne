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
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402


PHASE = 244
SOURCE_PHASE = 243
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE243_DIR = Path("tests/result/phase243_candidate_clustering_and_casebank_v2/candidate_clustering_and_casebank_v2")
RESULT_ROOT = Path("tests/result/phase244_first_internal_trace_batch")
ROUND_DEFAULT = "first_internal_trace_batch"

SPECS = {
    "qwen3": {"source_layers": [29], "observe_layers": [29, 31, 33]},
    "glm4": {"source_layers": [30], "observe_layers": [28, 30, 32]},
    "deepseek7b": {"source_layers": [24], "observe_layers": [24, 26, 27]},
}

COMPONENTS = ["gate", "up", "product", "down_out", "recomputed_product"]
ROLLOUT_TESTS = {"stepwise_rollout_trace", "rollout_closure_trace", "cross_model_structure_comparison"}


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


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b) or a.numel() != b.numel():
        return 0.0
    denom = norm(a) * norm(b)
    if denom <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def tensor_stats(vec: torch.Tensor | None, baseline: torch.Tensor | None) -> dict[str, float]:
    vec_norm = norm(vec)
    if vec is None or baseline is None or not torch.is_tensor(vec) or not torch.is_tensor(baseline) or vec.numel() != baseline.numel():
        return {
            "vector_norm": round(vec_norm, 6),
            "delta_norm_vs_full": 0.0,
            "relative_delta_vs_full": 0.0,
            "cosine_vs_full": 0.0,
        }
    delta_norm = norm(vec.float() - baseline.float())
    return {
        "vector_norm": round(vec_norm, 6),
        "delta_norm_vs_full": round(delta_norm, 6),
        "relative_delta_vs_full": round(delta_norm / max(norm(baseline), 1e-6), 6),
        "cosine_vs_full": round(cosine(vec, baseline), 6),
    }


def load_inputs(max_trace_rows: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    selected = read_jsonl(PHASE243_DIR / "phase243_trace_selection_rows.jsonl")
    if not selected:
        raise FileNotFoundError(f"missing phase243_trace_selection_rows.jsonl under {PHASE243_DIR}")
    selected.sort(key=lambda x: int(x.get("trace_rank") or 10**9))
    if max_trace_rows > 0:
        selected = selected[:max_trace_rows]
    behavior = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    if not behavior:
        raise FileNotFoundError(f"missing phase241_large_scale_behavior_rows.jsonl under {PHASE241_DIR}")
    by_key = {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in behavior}
    return selected, by_key


def base_row(model_name: str, selected: dict[str, Any], behavior: dict[str, Any], run_id: str) -> dict[str, Any]:
    keys = [
        "case_id",
        "family_id",
        "mode_id",
        "variant_id",
        "target",
        "target_aliases",
        "expected_pattern",
        "semantic_match",
        "protocol_match",
        "over_generation",
        "closure_signal",
        "negative_category",
        "failure_type",
        "winning_regime",
        "second_competitor",
        "target_margin_vs_winner",
        "target_rank",
        "top_token",
    ]
    row = {key: behavior.get(key) for key in keys}
    row.update(
        {
            "phase": PHASE,
            "phase_id": "Phase244",
            "source_phase": SOURCE_PHASE,
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "run_id": run_id,
            "model": model_name,
            "prompt_variant": behavior.get("prompt_variant"),
            "prompt": behavior.get("prompt"),
            "selected_trace_id": selected.get("trace_selection_id"),
            "trace_rank": selected.get("trace_rank"),
            "recommended_next_test": selected.get("recommended_next_test"),
            "trace_batch": selected.get("trace_batch"),
            "data_split": selected.get("data_split"),
            "cluster_key": selected.get("cluster_key"),
            "stable_winner_regime": selected.get("stable_winner_regime"),
            "candidate_score": selected.get("candidate_score"),
            "selection_reasons": selected.get("selection_reasons", []),
        }
    )
    return row


def component_trace_rows(
    row_base: dict[str, Any],
    internal: dict[int, dict[str, torch.Tensor]],
    baseline_internal: dict[int, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, part_map in internal.items():
        base_parts = baseline_internal.get(layer_idx, {})
        for component in COMPONENTS:
            vec = part_map.get(component)
            if not torch.is_tensor(vec) or vec.ndim != 1:
                continue
            rows.append(
                {
                    **row_base,
                    "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:l{layer_idx}:{component}",
                    "trace_level": "gate_up_product",
                    "layer_idx": int(layer_idx),
                    "component": component,
                    **tensor_stats(vec, base_parts.get(component)),
                    "product_rel_error": round(safe_float(part_map.get("product_rel_error")), 8),
                }
            )
    return rows


def residual_trace_rows(
    row_base: dict[str, Any],
    hidden: dict[int, torch.Tensor],
    baseline_hidden: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, vec in hidden.items():
        rows.append(
            {
                **row_base,
                "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:l{layer_idx}:residual_state",
                "trace_level": "residual_state",
                "layer_idx": int(layer_idx),
                "component": "residual_state",
                **tensor_stats(vec, baseline_hidden.get(layer_idx)),
            }
        )
    return rows


def readout_trace_row(row_base: dict[str, Any], tokenizer: Any, logits: torch.Tensor, baseline_behavior: dict[str, Any]) -> dict[str, Any]:
    readout = p239.readout_metrics(tokenizer, logits, list(row_base.get("target_aliases") or []))
    winner = str(readout.get("winning_regime"))
    stable = str(row_base.get("stable_winner_regime"))
    baseline_margin = safe_float(baseline_behavior.get("target_margin_vs_winner"))
    margin = safe_float(readout.get("target_margin_vs_winner"))
    return {
        **row_base,
        **readout,
        "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:readout",
        "trace_level": "readout_competition",
        "component": "lm_head",
        "baseline_variant_id": "full",
        "baseline_winning_regime": baseline_behavior.get("winning_regime"),
        "baseline_target_margin_vs_winner": baseline_margin,
        "target_margin_delta_vs_full": round(margin - baseline_margin, 6),
        "winner_changed_vs_full": winner != str(baseline_behavior.get("winning_regime")),
        "stable_winner_match": winner == stable,
    }


def stepwise_rollout_rows(
    row_base: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    if max_steps <= 0:
        return []
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    rows = []
    generated_ids: list[int] = []
    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            outputs = model(**encoded)
            logits = outputs.logits[0, -1, :].detach().float().cpu()
            readout = p239.readout_metrics(tokenizer, logits, list(row_base.get("target_aliases") or []))
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
            generated_ids.append(next_id)
            token_text = tokenizer.decode([next_id])
            rows.append(
                {
                    **row_base,
                    **readout,
                    "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:step{step}",
                    "trace_level": "stepwise_rollout",
                    "component": "argmax_decode",
                    "step_index": step,
                    "generated_token_id": next_id,
                    "generated_token": token_text,
                    "partial_generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                }
            )
            next_tensor = torch.tensor([[next_id]], device=device, dtype=encoded["input_ids"].dtype)
            encoded["input_ids"] = torch.cat([encoded["input_ids"], next_tensor], dim=1)
            if "attention_mask" in encoded:
                encoded["attention_mask"] = torch.cat([encoded["attention_mask"], torch.ones_like(next_tensor)], dim=1)
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and next_id == int(eos_id):
                break
    return rows


def aggregate_metrics(rows: list[dict[str, Any]], readout_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    metrics = []
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                str(row["recommended_next_test"]),
                str(row["trace_level"]),
                str(row.get("layer_idx", "none")),
                str(row["component"]),
            )
        ].append(row)
    for (test, level, layer_idx, component), items in buckets.items():
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "metric_id": f"phase244:{items[0]['model']}:{test}:{level}:l{layer_idx}:{component}",
                "scope": "trace_batch_component",
                "model": items[0]["model"],
                "recommended_next_test": test,
                "trace_level": level,
                "layer_idx": layer_idx,
                "component": component,
                "metric_name": "mean_relative_delta_vs_full",
                "metric_value": round(mean(safe_float(x.get("relative_delta_vs_full")) for x in items), 6),
                "mean_delta_norm_vs_full": round(mean(safe_float(x.get("delta_norm_vs_full")) for x in items), 6),
                "mean_cosine_vs_full": round(mean(safe_float(x.get("cosine_vs_full")) for x in items), 6),
                "rows": len(items),
            }
        )
    readout_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in readout_rows:
        readout_buckets[str(row["recommended_next_test"])].append(row)
    for test, items in readout_buckets.items():
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "metric_id": f"phase244:{items[0]['model']}:{test}:readout_margin",
                "scope": "trace_batch_readout",
                "model": items[0]["model"],
                "recommended_next_test": test,
                "trace_level": "readout_competition",
                "component": "lm_head",
                "metric_name": "mean_target_margin_delta_vs_full",
                "metric_value": round(mean(safe_float(x.get("target_margin_delta_vs_full")) for x in items), 6),
                "stable_winner_match_rate": round(sum(1 for x in items if x.get("stable_winner_match")) / max(1, len(items)), 4),
                "winner_changed_rate": round(sum(1 for x in items if x.get("winner_changed_vs_full")) / max(1, len(items)), 4),
                "winner_regimes": dict(Counter(str(x.get("winning_regime")) for x in items).most_common()),
                "rows": len(items),
            }
        )
    rollout_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        rollout_buckets[str(row["recommended_next_test"])].append(row)
    for test, items in rollout_buckets.items():
        first_steps = [x for x in items if int(x.get("step_index") or 0) == 1]
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "metric_id": f"phase244:{items[0]['model']}:{test}:stepwise_rollout",
                "scope": "trace_batch_rollout",
                "model": items[0]["model"],
                "recommended_next_test": test,
                "trace_level": "stepwise_rollout",
                "component": "argmax_decode",
                "metric_name": "first_step_target_rank_mean",
                "metric_value": round(mean(safe_float(x.get("target_rank")) for x in first_steps), 4) if first_steps else 0.0,
                "first_step_winner_regimes": dict(Counter(str(x.get("winning_regime")) for x in first_steps).most_common()),
                "rows": len(items),
            }
        )
    metrics.sort(key=lambda x: str(x["metric_id"]))
    return metrics


def observation_rows(trace_rows: list[dict[str, Any]], readout_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    observations = []
    for row in trace_rows:
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "observation_id": f"phase244:{row['model']}:{row['case_id']}:{row['variant_id']}:l{row['layer_idx']}:{row['component']}",
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "level": row["trace_level"],
                "layer_idx": row["layer_idx"],
                "component": row["component"],
                "metric_name": "relative_delta_vs_full",
                "metric_value": safe_float(row["relative_delta_vs_full"]),
                "metric_unit": "ratio",
                "recommended_next_test": row["recommended_next_test"],
                "winning_regime": row.get("winning_regime", ""),
                "second_competitor": row.get("second_competitor", ""),
            }
        )
    for row in readout_rows:
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "observation_id": f"phase244:{row['model']}:{row['case_id']}:{row['variant_id']}:readout",
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "level": "readout_competition",
                "component": "lm_head",
                "metric_name": "target_margin_delta_vs_full",
                "metric_value": safe_float(row["target_margin_delta_vs_full"]),
                "metric_unit": "logit",
                "recommended_next_test": row["recommended_next_test"],
                "winning_regime": row.get("winning_regime", ""),
                "second_competitor": row.get("second_competitor", ""),
            }
        )
    for row in rollout_rows:
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase244",
                "created_at": now,
                "observation_id": f"phase244:{row['model']}:{row['case_id']}:{row['variant_id']}:step{row['step_index']}",
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "level": "stepwise_rollout",
                "component": "argmax_decode",
                "metric_name": "target_rank",
                "metric_value": safe_float(row["target_rank"]),
                "metric_unit": "rank",
                "recommended_next_test": row["recommended_next_test"],
                "winning_regime": row.get("winning_regime", ""),
                "generated_token": row.get("generated_token", ""),
            }
        )
    return observations


def graph_edges(model_name: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    edges = []
    for row in metrics:
        if row.get("trace_level") == "gate_up_product" and row.get("component") in {"product", "down_out"}:
            edges.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase244",
                    "created_at": now,
                    "edge_id": f"phase244:{model_name}:{row['recommended_next_test']}:{row['component']}:to_readout",
                    "source": f"component:{model_name}:{row['component']}",
                    "target": f"node:ReadoutCompetition:{model_name}",
                    "edge_type": "candidate_internal_trace",
                    "model": model_name,
                    "recommended_next_test": row["recommended_next_test"],
                    "evidence_type": "mean_relative_delta_vs_full",
                    "effect_direction": "activation_changed" if safe_float(row["metric_value"]) >= 0.03 else "weak_change",
                    "effect_size": safe_float(row["metric_value"]),
                    "confidence": round(0.35 + min(0.2, safe_float(row["metric_value"])), 4),
                    "supporting_phases": ["Phase241", "Phase242", "Phase243", "Phase244"],
                    "status": "trace_evidence_not_causal_closure",
                }
            )
        if row.get("trace_level") == "readout_competition":
            edges.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase244",
                    "created_at": now,
                    "edge_id": f"phase244:{model_name}:{row['recommended_next_test']}:readout_to_rollout",
                    "source": f"node:ReadoutCompetition:{model_name}",
                    "target": f"node:StepwiseRollout:{model_name}",
                    "edge_type": "readout_rollout_candidate",
                    "model": model_name,
                    "recommended_next_test": row["recommended_next_test"],
                    "evidence_type": "target_margin_delta",
                    "effect_direction": "mixed",
                    "effect_size": safe_float(row["metric_value"]),
                    "confidence": 0.42,
                    "supporting_phases": ["Phase241", "Phase244"],
                    "status": "trace_evidence_not_causal_closure",
                }
            )
    return edges


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows, behavior_by_key = load_inputs(int(args.max_trace_rows))
    spec = SPECS[args.model]
    run_id = f"phase244:{args.model}:{args.round_name}"
    model = None
    tokenizer = None
    component_rows_all: list[dict[str, Any]] = []
    residual_rows_all: list[dict[str, Any]] = []
    readout_rows_all: list[dict[str, Any]] = []
    rollout_rows_all: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    baseline_cache: dict[str, tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor]]] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for index, selected in enumerate(selected_rows, start=1):
            key = (args.model, str(selected.get("case_id")), str(selected.get("variant_id")))
            behavior = behavior_by_key.get(key)
            baseline_behavior = behavior_by_key.get((args.model, str(selected.get("case_id")), "full"))
            if behavior is None or baseline_behavior is None:
                missing_rows.append({"model": args.model, "case_id": selected.get("case_id"), "variant_id": selected.get("variant_id")})
                continue
            baseline_key = str(selected.get("case_id"))
            if baseline_key not in baseline_cache:
                baseline_internal, baseline_hidden, _baseline_logits = p228.capture_internal(
                    model,
                    tokenizer,
                    device,
                    str(baseline_behavior["prompt_variant"]),
                    list(spec["source_layers"]),
                    list(spec["observe_layers"]),
                )
                baseline_cache[baseline_key] = (baseline_behavior, baseline_internal, baseline_hidden)
            base_behavior, baseline_internal, baseline_hidden = baseline_cache[baseline_key]
            internal, hidden, logits = p228.capture_internal(
                model,
                tokenizer,
                device,
                str(behavior["prompt_variant"]),
                list(spec["source_layers"]),
                list(spec["observe_layers"]),
            )
            row_base = base_row(args.model, selected, behavior, run_id)
            component_rows_all.extend(component_trace_rows(row_base, internal, baseline_internal))
            residual_rows_all.extend(residual_trace_rows(row_base, hidden, baseline_hidden))
            readout_rows_all.append(readout_trace_row(row_base, tokenizer, logits, base_behavior))
            if str(selected.get("recommended_next_test")) in ROLLOUT_TESTS:
                rollout_rows_all.extend(
                    stepwise_rollout_rows(row_base, model, tokenizer, device, str(behavior["prompt_variant"]), int(args.max_rollout_steps))
                )
            del logits
            if index % max(1, int(args.log_every)) == 0:
                log(
                    f"{args.model}: selected={index}/{len(selected_rows)} component_rows={len(component_rows_all)} "
                    f"residual_rows={len(residual_rows_all)} readout_rows={len(readout_rows_all)}"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_trace = component_rows_all + residual_rows_all
    metrics = aggregate_metrics(all_trace, readout_rows_all, rollout_rows_all)
    observations = observation_rows(all_trace, readout_rows_all, rollout_rows_all)
    edges = graph_edges(args.model, metrics)
    by_test = Counter(str(x.get("recommended_next_test")) for x in readout_rows_all)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "First internal trace batch",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "selected_rows": len(selected_rows),
        "completed_cases": len(readout_rows_all),
        "missing_rows": len(missing_rows),
        "component_trace_rows": len(component_rows_all),
        "residual_trace_rows": len(residual_rows_all),
        "readout_trace_rows": len(readout_rows_all),
        "stepwise_rollout_rows": len(rollout_rows_all),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "trace_selection_by_test": dict(by_test.most_common()),
        "mean_component_relative_delta": round(mean(safe_float(x.get("relative_delta_vs_full")) for x in component_rows_all), 6)
        if component_rows_all
        else 0.0,
        "mean_residual_relative_delta": round(mean(safe_float(x.get("relative_delta_vs_full")) for x in residual_rows_all), 6)
        if residual_rows_all
        else 0.0,
        "mean_readout_margin_delta_vs_full": round(mean(safe_float(x.get("target_margin_delta_vs_full")) for x in readout_rows_all), 6)
        if readout_rows_all
        else 0.0,
        "stable_winner_match_rate": round(sum(1 for x in readout_rows_all if x.get("stable_winner_match")) / max(1, len(readout_rows_all)), 4),
    }
    write_json(out_dir / f"phase244_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase244_{args.model}_component_trace_rows.jsonl", component_rows_all)
    write_jsonl(out_dir / f"phase244_{args.model}_residual_trace_rows.jsonl", residual_rows_all)
    write_jsonl(out_dir / f"phase244_{args.model}_readout_trace_rows.jsonl", readout_rows_all)
    write_jsonl(out_dir / f"phase244_{args.model}_stepwise_rollout_rows.jsonl", rollout_rows_all)
    write_jsonl(out_dir / f"phase244_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase244_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase244_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase244_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "completed_cases": len(readout_rows_all)}, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase244_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    component_rows = []
    residual_rows = []
    readout_rows = []
    rollout_rows = []
    observations = []
    metrics = []
    edges = []
    for model in MODELS:
        component_rows.extend(read_jsonl(out_dir / f"phase244_{model}_component_trace_rows.jsonl"))
        residual_rows.extend(read_jsonl(out_dir / f"phase244_{model}_residual_trace_rows.jsonl"))
        readout_rows.extend(read_jsonl(out_dir / f"phase244_{model}_readout_trace_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase244_{model}_stepwise_rollout_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase244_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase244_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase244_{model}_graph_edges.jsonl"))
    by_model = {x["model"]: x for x in summaries}
    by_test = Counter(str(x.get("recommended_next_test")) for x in readout_rows)
    mean_component = mean(safe_float(x.get("relative_delta_vs_full")) for x in component_rows) if component_rows else 0.0
    mean_residual = mean(safe_float(x.get("relative_delta_vs_full")) for x in residual_rows) if residual_rows else 0.0
    mean_margin_delta = mean(safe_float(x.get("target_margin_delta_vs_full")) for x in readout_rows) if readout_rows else 0.0
    progress = {
        "pattern_family_atlas": 0.72,
        "candidate_clustering": 0.40,
        "case_bank_calibration": 0.36,
        "high_value_trace_selection": 0.55,
        "first_internal_trace_batch": 0.30 if len(summaries) == 3 else round(0.1 * len(summaries), 2),
        "gate_up_product_signature": 0.38 if component_rows else 0.32,
        "residual_state_signature": 0.37 if residual_rows else 0.30,
        "readout_competition_trace": 0.58 if readout_rows else 0.50,
        "stepwise_rollout_trace": 0.18 if rollout_rows else 0.12,
        "causal_closure": 0.10,
        "general_language_mechanism_confidence": 0.53 if len(summaries) == 3 else 0.52,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model first internal trace batch",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": list(by_model),
        "model_summaries": by_model,
        "component_trace_rows": len(component_rows),
        "residual_trace_rows": len(residual_rows),
        "readout_trace_rows": len(readout_rows),
        "stepwise_rollout_rows": len(rollout_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "trace_selection_by_test": dict(by_test.most_common()),
        "mean_component_relative_delta": round(mean_component, 6),
        "mean_residual_relative_delta": round(mean_residual, 6),
        "mean_readout_margin_delta_vs_full": round(mean_margin_delta, 6),
        "stable_winner_match_rate": round(sum(1 for x in readout_rows if x.get("stable_winner_match")) / max(1, len(readout_rows)), 4),
        "pattern_atlas_progress": progress,
        "judgement": "trace_evidence_only_not_causal_closure",
    }
    write_json(out_dir / "phase244_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase244_cross_model_component_trace_rows.jsonl", component_rows)
    write_jsonl(out_dir / "phase244_cross_model_residual_trace_rows.jsonl", residual_rows)
    write_jsonl(out_dir / "phase244_cross_model_readout_trace_rows.jsonl", readout_rows)
    write_jsonl(out_dir / "phase244_cross_model_stepwise_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase244_cross_model_observations.jsonl", observations)
    write_jsonl(out_dir / "phase244_cross_model_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase244_cross_model_graph_edges.jsonl", edges)
    write_report(out_dir / "phase244_first_internal_trace_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps({"phase": PHASE, "status": payload["status"], "summary": payload}, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase244 first internal trace batch",
        "",
        "## Core result",
        "",
        "Phase244 converts the Phase243 high-value candidates into fixed-format internal trace rows.",
        "This is trace evidence, not causal closure.",
        "",
        "## Counts",
        "",
        f"- models: {', '.join(summary.get('models', []))}",
        f"- component_trace_rows: {summary.get('component_trace_rows')}",
        f"- residual_trace_rows: {summary.get('residual_trace_rows')}",
        f"- readout_trace_rows: {summary.get('readout_trace_rows')}",
        f"- stepwise_rollout_rows: {summary.get('stepwise_rollout_rows')}",
        f"- trace_selection_by_test: {json.dumps(summary.get('trace_selection_by_test', {}), ensure_ascii=False)}",
        "",
        "## Aggregate signals",
        "",
        f"- mean_component_relative_delta: {summary.get('mean_component_relative_delta')}",
        f"- mean_residual_relative_delta: {summary.get('mean_residual_relative_delta')}",
        f"- mean_readout_margin_delta_vs_full: {summary.get('mean_readout_margin_delta_vs_full')}",
        f"- stable_winner_match_rate: {summary.get('stable_winner_match_rate')}",
        "",
        "## Pattern Atlas progress",
        "",
        "```json",
        json.dumps(summary.get("pattern_atlas_progress", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Next",
        "",
        "Phase245 should validate whether the strongest trace signatures survive larger validation/frozen splits and then design focused causal probes.",
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
            "latest_phase": "Phase244",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "validate first internal trace signatures before causal closure",
            "small_model_bias_warning": "qwen3/glm4/deepseek7b are small or mid local models; internal mechanisms may be rough and should not be treated as final language mechanism.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase244 first internal trace batch")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-trace-rows", type=int, default=100)
    parser.add_argument("--max-rollout-steps", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
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
