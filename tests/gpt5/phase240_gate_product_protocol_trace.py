#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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


PHASE = 240
SOURCE_PHASE = 239
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase240_gate_product_protocol_trace")
ROUND_DEFAULT = "gate_product_protocol_trace"

SPECS = {
    "qwen3": {"source_layers": [29], "observe_layers": [29, 31, 33]},
    "glm4": {"source_layers": [30], "observe_layers": [28, 30, 32]},
    "deepseek7b": {"source_layers": [24], "observe_layers": [24, 26, 27]},
}

TRACE_VARIANTS = [
    "full",
    "strong_answer_anchor",
    "one_word_strict",
    "short_answer_instruction",
    "explain_instruction",
    "target_seeded",
]

COMPONENTS = ["gate", "up", "product", "down_out", "recomputed_product"]


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


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None or not torch.is_tensor(a) or not torch.is_tensor(b):
        return 0.0
    if a.numel() != b.numel():
        return 0.0
    denom = norm(a) * norm(b)
    if denom <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def tensor_stats(vec: torch.Tensor | None, baseline: torch.Tensor | None = None) -> dict[str, float]:
    vec_norm = norm(vec)
    if vec is None or baseline is None or not torch.is_tensor(vec) or not torch.is_tensor(baseline) or vec.numel() != baseline.numel():
        return {
            "vector_norm": round(vec_norm, 6),
            "delta_norm_vs_full": 0.0,
            "relative_delta_vs_full": 0.0,
            "cosine_vs_full": 0.0,
        }
    delta_norm = norm(vec.float() - baseline.float())
    base_norm = norm(baseline)
    return {
        "vector_norm": round(vec_norm, 6),
        "delta_norm_vs_full": round(delta_norm, 6),
        "relative_delta_vs_full": round(delta_norm / max(base_norm, 1e-6), 6),
        "cosine_vs_full": round(cosine(vec, baseline), 6),
    }


def component_rows(
    row_base: dict[str, Any],
    internal: dict[int, dict[str, torch.Tensor]],
    baseline_internal: dict[int, dict[str, torch.Tensor]] | None,
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, part_map in internal.items():
        base_parts = baseline_internal.get(layer_idx, {}) if baseline_internal else {}
        for component in COMPONENTS:
            vec = part_map.get(component)
            if not torch.is_tensor(vec) or vec.ndim != 1:
                continue
            stats = tensor_stats(vec, base_parts.get(component))
            rows.append(
                {
                    **row_base,
                    "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:layer{layer_idx}:{component}",
                    "trace_level": "gate_up_product",
                    "layer_idx": int(layer_idx),
                    "component": component,
                    **stats,
                    "product_rel_error": round(safe_float(part_map.get("product_rel_error")), 8),
                }
            )
    return rows


def residual_rows(
    row_base: dict[str, Any],
    hidden: dict[int, torch.Tensor],
    baseline_hidden: dict[int, torch.Tensor] | None,
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, vec in hidden.items():
        base_vec = baseline_hidden.get(layer_idx) if baseline_hidden else None
        stats = tensor_stats(vec, base_vec)
        rows.append(
            {
                **row_base,
                "trace_id": f"{row_base['run_id']}:{row_base['case_id']}:{row_base['variant_id']}:layer{layer_idx}:residual_state",
                "trace_level": "residual_state",
                "layer_idx": int(layer_idx),
                "component": "residual_state",
                **stats,
            }
        )
    return rows


def aggregate_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["variant_id"]),
            str(row["trace_level"]),
            int(row["layer_idx"]),
            str(row["component"]),
        )
        buckets[key].append(row)
    out = []
    for (model, variant_id, trace_level, layer_idx, component), items in buckets.items():
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase240",
                "created_at": utc_now(),
                "metric_id": f"phase240:{model}:{variant_id}:{trace_level}:l{layer_idx}:{component}",
                "scope": "variant_component",
                "model": model,
                "family_id": "output_protocol",
                "mode_id": "short_answer",
                "variant_id": variant_id,
                "trace_level": trace_level,
                "layer_idx": layer_idx,
                "component": component,
                "metric_name": "mean_relative_delta_vs_full",
                "metric_value": round(mean(safe_float(x.get("relative_delta_vs_full")) for x in items), 6),
                "mean_delta_norm_vs_full": round(mean(safe_float(x.get("delta_norm_vs_full")) for x in items), 6),
                "mean_cosine_vs_full": round(mean(safe_float(x.get("cosine_vs_full")) for x in items), 6),
                "rows": len(items),
            }
        )
    out.sort(key=lambda x: (x["model"], x["variant_id"], x["trace_level"], x["layer_idx"], x["component"]))
    return out


def behavior_metric_rows(model_name: str, behavior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior_rows:
        buckets[str(row["variant_id"])].append(row)
    out = []
    for variant_id, items in buckets.items():
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase240",
                "created_at": utc_now(),
                "metric_id": f"phase240:{model_name}:behavior:{variant_id}",
                "scope": "variant_behavior",
                "model": model_name,
                "family_id": "output_protocol",
                "mode_id": "short_answer",
                "variant_id": variant_id,
                "metric_name": "protocol_trace_behavior",
                "metric_value": round(mean(safe_float(x["calibrated_behavior_score"]) for x in items), 4),
                "protocol_match_rate": round(sum(1 for x in items if x.get("protocol_match")) / max(1, len(items)), 4),
                "over_generation_rate": round(sum(1 for x in items if x.get("over_generation")) / max(1, len(items)), 4),
                "mean_target_margin_vs_winner": round(mean(safe_float(x["target_margin_vs_winner"]) for x in items), 4),
                "winner_regimes": dict(Counter(str(x["winning_regime"]) for x in items).most_common()),
                "rows": len(items),
            }
        )
    return out


def trace_decision(rows: list[dict[str, Any]], behavior_rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_variants = {"strong_answer_anchor", "one_word_strict", "short_answer_instruction"}
    strict_trace = [
        x for x in rows
        if x.get("variant_id") in strict_variants and x.get("trace_level") == "gate_up_product" and x.get("component") in {"product", "down_out"}
    ]
    strict_behavior = [x for x in behavior_rows if x.get("variant_id") in strict_variants]
    mean_delta = mean([safe_float(x.get("relative_delta_vs_full")) for x in strict_trace]) if strict_trace else 0.0
    mean_margin_delta = mean([safe_float(x.get("target_margin_vs_winner")) - safe_float(x.get("baseline_target_margin_vs_winner")) for x in strict_behavior]) if strict_behavior else 0.0
    protocol_rate = sum(1 for x in strict_behavior if x.get("protocol_match")) / max(1, len(strict_behavior))
    over_rate = sum(1 for x in strict_behavior if x.get("over_generation")) / max(1, len(strict_behavior))
    if mean_delta < 0.03:
        decision = "protocol_state_weak_or_not_written"
    elif protocol_rate < 0.15 and mean_margin_delta <= 0.25:
        decision = "protocol_state_written_but_readout_competition_failed"
    elif protocol_rate < 0.15 and over_rate > 0.5:
        decision = "readout_or_rollout_closure_failed"
    else:
        decision = "mixed_or_partially_repaired"
    return {
        "decision": decision,
        "strict_mean_product_down_relative_delta": round(mean_delta, 6),
        "strict_mean_margin_delta": round(mean_margin_delta, 4),
        "strict_protocol_match_rate": round(protocol_rate, 4),
        "strict_over_generation_rate": round(over_rate, 4),
    }


def observation_rows(trace_rows: list[dict[str, Any]], behavior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    observations = []
    for row in trace_rows:
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase240",
                "created_at": now,
                "observation_id": f"phase240:{row['model']}:{row['case_id']}:{row['variant_id']}:l{row['layer_idx']}:{row['component']}:relative_delta",
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
                "winning_regime": row.get("winning_regime", ""),
                "second_competitor": row.get("second_competitor", ""),
                "drift_type": row.get("drift_type", ""),
            }
        )
    for row in behavior_rows:
        observations.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase240",
                "created_at": now,
                "observation_id": f"phase240:{row['model']}:{row['case_id']}:{row['variant_id']}:readout_margin",
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "level": "readout_competition",
                "metric_name": "target_margin_vs_winner",
                "metric_value": safe_float(row["target_margin_vs_winner"]),
                "metric_unit": "logit",
                "winning_regime": row.get("winning_regime", ""),
                "second_competitor": row.get("second_competitor", ""),
                "drift_type": row.get("drift_type", ""),
            }
        )
    return observations


def graph_edges(model_name: str, decisions: dict[str, Any], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    edges = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase240",
            "created_at": now,
            "edge_id": f"phase240:{model_name}:prompt_protocol_to_gate_up_product",
            "source": "node:PromptProtocol",
            "target": f"node:GateUpProduct:{model_name}",
            "edge_type": "protocol_state_trace",
            "family_id": "output_protocol",
            "mode_id": "short_answer",
            "model": model_name,
            "evidence_type": "internal_activation_delta",
            "effect_direction": "positive" if decisions["strict_mean_product_down_relative_delta"] >= 0.03 else "weak_or_absent",
            "effect_size": decisions["strict_mean_product_down_relative_delta"],
            "confidence": round(0.35 + min(0.25, decisions["strict_mean_product_down_relative_delta"]), 4),
            "supporting_phases": ["Phase239", "Phase240"],
            "status": decisions["decision"],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase240",
            "created_at": now,
            "edge_id": f"phase240:{model_name}:gate_up_product_to_readout_competition",
            "source": f"node:GateUpProduct:{model_name}",
            "target": f"node:ReadoutCompetition:{model_name}",
            "edge_type": "readout_competition_trace",
            "family_id": "output_protocol",
            "mode_id": "short_answer",
            "model": model_name,
            "evidence_type": "activation_delta_plus_target_margin",
            "effect_direction": "blocked" if "readout" in decisions["decision"] else "mixed",
            "effect_size": decisions["strict_mean_margin_delta"],
            "confidence": 0.46,
            "supporting_phases": ["Phase239", "Phase240"],
            "status": decisions["decision"],
        },
    ]
    top_product = [
        x for x in metric_rows
        if x.get("trace_level") == "gate_up_product" and x.get("component") in {"product", "down_out"} and x.get("variant_id") != "full"
    ]
    top_product.sort(key=lambda x: safe_float(x.get("metric_value")), reverse=True)
    for row in top_product[:6]:
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase240",
                "created_at": now,
                "edge_id": f"phase240:{model_name}:{row['variant_id']}:l{row['layer_idx']}:{row['component']}",
                "source": f"prompt_variant:{row['variant_id']}",
                "target": f"component:{model_name}:layer{row['layer_idx']}:{row['component']}",
                "edge_type": "variant_component_delta",
                "family_id": "output_protocol",
                "mode_id": "short_answer",
                "model": model_name,
                "evidence_type": "mean_relative_delta_vs_full",
                "effect_direction": "activation_changed",
                "effect_size": safe_float(row["metric_value"]),
                "confidence": 0.42,
                "supporting_phases": ["Phase240"],
                "status": "candidate_trace_edge",
            }
        )
    return edges


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SPECS[args.model]
    cases = p239.load_selected_cases(int(args.max_cases))
    run_id = f"phase240:{args.model}:{args.round_name}"
    model = None
    tokenizer = None
    trace_rows: list[dict[str, Any]] = []
    residual_trace_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for case_index, case in enumerate(cases, start=1):
            variants = p239.prompt_variants(str(case["prompt"]), str(case["target"]))
            variants = {k: v for k, v in variants.items() if k in TRACE_VARIANTS}
            baseline_internal = None
            baseline_hidden = None
            baseline_behavior = None
            case_cache: dict[str, tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor], torch.Tensor]] = {}
            for variant_id in TRACE_VARIANTS:
                if variant_id not in variants:
                    continue
                prompt = variants[variant_id]
                internal, hidden, logits = p228.capture_internal(
                    model,
                    tokenizer,
                    device,
                    prompt,
                    list(spec["source_layers"]),
                    list(spec["observe_layers"]),
                )
                case_cache[variant_id] = (internal, hidden, logits)
                if variant_id == "full":
                    baseline_internal = internal
                    baseline_hidden = hidden
            if baseline_internal is None or baseline_hidden is None:
                continue
            for variant_id in TRACE_VARIANTS:
                if variant_id not in variants or variant_id not in case_cache:
                    continue
                prompt = variants[variant_id]
                internal, hidden, logits = case_cache[variant_id]
                readout = p239.readout_metrics(tokenizer, logits, list(case["target_aliases"]))
                output = p239.generate_text(model, tokenizer, device, prompt, int(args.max_new_tokens))
                behavior = p239.classify_output(output, list(case["target_aliases"]), variant_id, str(case.get("expected_pattern") or ""))
                row_base = {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "schema_version": SCHEMA_VERSION,
                    "created_at": utc_now(),
                    "run_id": run_id,
                    "model": args.model,
                    "case_index": case_index,
                    **case,
                    "variant_id": variant_id,
                    "prompt_variant": prompt,
                    **readout,
                    **behavior,
                }
                if variant_id == "full":
                    baseline_behavior = row_base
                    row_base["baseline_target_margin_vs_winner"] = row_base["target_margin_vs_winner"]
                    row_base["baseline_winning_regime"] = row_base["winning_regime"]
                    row_base["target_margin_delta_vs_full"] = 0.0
                    row_base["winner_changed_vs_full"] = False
                elif baseline_behavior is not None:
                    row_base["baseline_target_margin_vs_winner"] = baseline_behavior["target_margin_vs_winner"]
                    row_base["baseline_winning_regime"] = baseline_behavior["winning_regime"]
                    row_base["target_margin_delta_vs_full"] = round(
                        safe_float(row_base["target_margin_vs_winner"]) - safe_float(baseline_behavior["target_margin_vs_winner"]), 4
                    )
                    row_base["winner_changed_vs_full"] = row_base["winning_regime"] != baseline_behavior["winning_regime"]
                behavior_rows.append(row_base)
                trace_rows.extend(component_rows(row_base, internal, baseline_internal))
                residual_trace_rows.extend(residual_rows(row_base, hidden, baseline_hidden))
                del logits
            del case_cache
            log(f"{args.model}: case={case_index}/{len(cases)} behavior_rows={len(behavior_rows)} trace_rows={len(trace_rows)}")
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

    all_trace_rows = trace_rows + residual_trace_rows
    trace_metrics = aggregate_trace(all_trace_rows)
    behavior_metrics = behavior_metric_rows(args.model, behavior_rows)
    decisions = trace_decision(all_trace_rows, behavior_rows)
    observations = observation_rows(all_trace_rows, behavior_rows)
    edges = graph_edges(args.model, decisions, trace_metrics)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Gate/up/product protocol trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(cases),
        "behavior_rows": len(behavior_rows),
        "gate_product_trace_rows": len(trace_rows),
        "residual_trace_rows": len(residual_trace_rows),
        "observation_rows": len(observations),
        "metric_rows": len(trace_metrics) + len(behavior_metrics),
        "graph_edges": len(edges),
        "mean_behavior_score": round(mean(safe_float(x["calibrated_behavior_score"]) for x in behavior_rows), 4) if behavior_rows else 0.0,
        "protocol_match_rate": round(sum(1 for x in behavior_rows if x.get("protocol_match")) / max(1, len(behavior_rows)), 4),
        "decision": decisions,
    }
    write_json(out_dir / f"phase240_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase240_{args.model}_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / f"phase240_{args.model}_gate_product_protocol_rows.jsonl", trace_rows)
    write_jsonl(out_dir / f"phase240_{args.model}_residual_protocol_rows.jsonl", residual_trace_rows)
    write_jsonl(out_dir / f"phase240_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase240_{args.model}_metrics.jsonl", trace_metrics + behavior_metrics)
    write_jsonl(out_dir / f"phase240_{args.model}_graph_edges.jsonl", edges)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "decision": decisions}, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    behavior_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    residual_rows_all: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    summaries = []
    for model in MODELS:
        summaries.append(read_json(out_dir / f"phase240_{model}_summary.json"))
        behavior_rows.extend(read_jsonl(out_dir / f"phase240_{model}_behavior_rows.jsonl"))
        gate_rows.extend(read_jsonl(out_dir / f"phase240_{model}_gate_product_protocol_rows.jsonl"))
        residual_rows_all.extend(read_jsonl(out_dir / f"phase240_{model}_residual_protocol_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase240_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase240_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase240_{model}_graph_edges.jsonl"))
    summaries = [x for x in summaries if x]
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model gate/up/product protocol trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "behavior_rows": len(behavior_rows),
        "gate_product_trace_rows": len(gate_rows),
        "residual_trace_rows": len(residual_rows_all),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "mean_behavior_score": round(mean(safe_float(x["calibrated_behavior_score"]) for x in behavior_rows), 4) if behavior_rows else 0.0,
        "protocol_match_rate": round(sum(1 for x in behavior_rows if x.get("protocol_match")) / max(1, len(behavior_rows)), 4),
        "model_decisions": {x.get("model"): x.get("decision") for x in summaries},
        "top_component_deltas": top_component_deltas(metrics),
    }
    write_json(out_dir / "phase240_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase240_cross_model_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / "phase240_cross_model_gate_product_protocol_rows.jsonl", gate_rows)
    write_jsonl(out_dir / "phase240_cross_model_residual_protocol_rows.jsonl", residual_rows_all)
    write_jsonl(out_dir / "phase240_cross_model_observations.jsonl", observations)
    write_jsonl(out_dir / "phase240_cross_model_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase240_cross_model_graph_edges.jsonl", edges)
    write_report(out_dir / "phase240_protocol_trace_report.md", payload, summaries)
    update_atlas(payload, observations, metrics, edges)
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "behavior_rows": len(behavior_rows)}, ensure_ascii=False, indent=2))
    return payload


def top_component_deltas(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        x for x in metrics
        if x.get("scope") == "variant_component"
        and x.get("trace_level") in {"gate_up_product", "residual_state"}
        and x.get("variant_id") != "full"
    ]
    rows.sort(key=lambda x: safe_float(x.get("metric_value")), reverse=True)
    return rows[:30]


def write_report(path: Path, payload: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = ["# Phase240 Gate/Up/Product Protocol Trace", ""]
    lines.append(f"behavior_rows: {payload['behavior_rows']}")
    lines.append(f"gate_product_trace_rows: {payload['gate_product_trace_rows']}")
    lines.append(f"residual_trace_rows: {payload['residual_trace_rows']}")
    lines.append(f"mean_behavior_score: {payload['mean_behavior_score']}")
    lines.append(f"protocol_match_rate: {payload['protocol_match_rate']}")
    lines.extend(["", "## Model Decisions", "", "| model | decision | strict delta | margin delta | protocol match | over generation |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in summaries:
        d = row.get("decision") or {}
        lines.append(
            f"| {row.get('model')} | {d.get('decision')} | {d.get('strict_mean_product_down_relative_delta')} | "
            f"{d.get('strict_mean_margin_delta')} | {d.get('strict_protocol_match_rate')} | {d.get('strict_over_generation_rate')} |"
        )
    lines.extend(["", "## Top Component Deltas", "", "| model | variant | level | layer | component | relative delta | cosine | rows |", "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |"])
    for row in payload["top_component_deltas"][:30]:
        lines.append(
            f"| {row.get('model')} | {row.get('variant_id')} | {row.get('trace_level')} | {row.get('layer_idx')} | "
            f"{row.get('component')} | {row.get('metric_value')} | {row.get('mean_cosine_vs_full')} | {row.get('rows')} |"
        )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "This phase is a trace, not a causal closure. It marks whether protocol prompts change gate/up/product/residual states and whether the change reaches readout competition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_atlas(payload: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase240"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.56
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.48
        progress.setdefault("levels", {})["gate_up_product"] = 0.30
        progress.setdefault("levels", {})["residual_state"] = 0.30
        progress.setdefault("levels", {})["readout_competition"] = max(0.48, safe_float(progress.get("levels", {}).get("readout_competition")))
        progress["next_phase"] = "Phase241_protocol_rollout_closure_trace"
        progress["latest_phase"] = {
            "phase_id": "Phase240",
            "title": "gate/up/product 协议状态追踪",
            "behavior_rows": payload["behavior_rows"],
            "gate_product_trace_rows": payload["gate_product_trace_rows"],
            "residual_trace_rows": payload["residual_trace_rows"],
            "mean_behavior_score": payload["mean_behavior_score"],
            "protocol_match_rate": payload["protocol_match_rate"],
            "model_decisions": payload["model_decisions"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase240 Gate/Product Protocol Trace Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- behavior_rows: {payload['behavior_rows']}\n"
        f"- gate_product_trace_rows: {payload['gate_product_trace_rows']}\n"
        f"- residual_trace_rows: {payload['residual_trace_rows']}\n"
        f"- mean_behavior_score: {payload['mean_behavior_score']}\n"
        f"- protocol_match_rate: {payload['protocol_match_rate']}\n"
        f"- model_decisions: {payload['model_decisions']}\n"
        f"- top_component_deltas: {payload['top_component_deltas'][:5]}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase240 gate/up/product protocol trace")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
