#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase248_regime_level_direction_bank as p248  # noqa: E402


PHASE = 249
SOURCE_PHASE = 248
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE248_DIR = Path("tests/result/phase248_regime_level_direction_bank/regime_level_direction_bank")
RESULT_ROOT = Path("tests/result/phase249_regime_level_causal_validation")
ROUND_DEFAULT = "regime_level_causal_validation"

SPECS = {
    "qwen3": {"final_observe_layer": 33},
    "glm4": {"final_observe_layer": 32},
    "deepseek7b": {"final_observe_layer": 27},
}

ROUTE_TO_REGIME = {
    "continuation_regime_test": "continuation_regime",
    "protocol_regime_test": "protocol_short_regime",
    "reason_regime_test": "because_reason_regime",
    "boundary_regime_test": "answer_boundary_regime",
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


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def vector_norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = torch.linalg.vector_norm(vec).item()
    if norm <= 1e-8:
        return torch.zeros_like(vec)
    return vec / norm * float(target_norm)


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    if not rows:
        raise FileNotFoundError(f"missing Phase241 behavior rows under {PHASE241_DIR}")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_phase248_candidates(model_name: str, max_candidates_per_model: int) -> list[dict[str, Any]]:
    rows = read_jsonl(PHASE248_DIR / "phase248_regime_test_candidate_rows.jsonl")
    rows = [x for x in rows if str(x.get("model")) == model_name]
    rows.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
    if max_candidates_per_model > 0:
        rows = rows[:max_candidates_per_model]
    return rows


def load_raw_vectors(model_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        out[key_for(item)] = {**item, "payload": payload}
    return out


def load_phase246_effects(model_name: str) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = read_jsonl(PHASE246_DIR / "phase246_causal_validation_rows.jsonl")
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if str(row.get("model")) != model_name:
            continue
        out[key_for(row)][f"phase246_{row.get('intervention')}_margin_delta"] = safe_float(row.get("target_margin_delta_vs_original"))
    return out


def get_output_embedding_weight(model: Any) -> torch.Tensor:
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("model output embeddings do not expose weight")
    return weight.detach().float().cpu()


def build_regime_directions(model: Any, tokenizer: Any) -> dict[str, torch.Tensor]:
    output_weight = get_output_embedding_weight(model)
    directions: dict[str, torch.Tensor] = {}
    for regime, texts in p248.REGIME_TEXTS.items():
        token_ids = p248.token_ids_for_texts(tokenizer, texts)
        if regime == "period_stop_regime" and tokenizer.eos_token_id is not None:
            token_ids = sorted(set(token_ids + [int(tokenizer.eos_token_id)]))
        vectors = [p248.unit(output_weight[token_id]) for token_id in token_ids if 0 <= token_id < output_weight.shape[0]]
        directions[regime] = p248.unit(torch.stack(vectors).mean(dim=0)) if vectors else torch.zeros(output_weight.shape[1])
    return directions


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    delta_device = delta.to(device=changed.device, dtype=changed.dtype)
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta_device
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def residual_direction_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_logits(model: Any, tokenizer: Any, device: torch.device, prompt: str, hook_ctx: Any) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with hook_ctx:
        with torch.inference_mode():
            outputs = model(**encoded)
    return outputs.logits[0, -1, :].detach().float().cpu()


def rollout_text(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int, hook_ctx_factory: Any) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(encoded["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with hook_ctx_factory():
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
    return tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()


def intervention_context(model: Any, intervention: str, layer_idx: int, regime_direction: torch.Tensor, target_direction: torch.Tensor, competitor_direction: torch.Tensor) -> Any:
    if intervention == "regime_suppression":
        return residual_direction_hook(model, layer_idx, regime_direction, sign=-1.0)
    if intervention == "regime_injection":
        return residual_direction_hook(model, layer_idx, regime_direction, sign=1.0)
    if intervention == "target_injection_replay":
        return residual_direction_hook(model, layer_idx, target_direction, sign=1.0)
    if intervention == "top_token_suppression_replay":
        return residual_direction_hook(model, layer_idx, competitor_direction, sign=-1.0)
    return no_hook()


def classify_margin(delta: float) -> str:
    if delta >= 1.0:
        return "readout_margin_gain"
    if delta <= -1.0:
        return "readout_margin_harmed"
    return "weak_or_no_readout_change"


def rollout_word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.sub(r"\s+", " ", text).strip().split())


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_by_key = load_behavior_by_key()
    candidates = load_phase248_candidates(args.model, int(args.max_candidates_per_model))
    raw_vectors = load_raw_vectors(args.model)
    phase246_effects = load_phase246_effects(args.model)
    layer_idx = int(SPECS[args.model]["final_observe_layer"])
    run_id = f"phase249:{args.model}:{args.round_name}"
    validation_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        regime_units = build_regime_directions(model, tokenizer)
        for index, candidate in enumerate(candidates, start=1):
            key = key_for(candidate)
            behavior = behavior_by_key.get(key)
            raw_item = raw_vectors.get(key)
            if behavior is None or raw_item is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_behavior_or_raw_vector"})
                continue
            payload = raw_item["payload"]
            delta_residual = payload.get("delta_residual")
            target_direction = payload.get("target_direction")
            competitor_direction = payload.get("competitor_direction")
            if not torch.is_tensor(delta_residual) or not torch.is_tensor(target_direction) or not torch.is_tensor(competitor_direction):
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_direction_tensor"})
                continue
            route = str(candidate.get("recommended_next_test") or "continuation_regime_test")
            regime = ROUTE_TO_REGIME.get(route, "continuation_regime")
            perturb_norm = max(vector_norm(target_direction), vector_norm(competitor_direction), vector_norm(delta_residual) * float(args.perturb_scale), 1e-6)
            regime_direction = normalize(regime_units[regime], perturb_norm)
            prompt = str(behavior["prompt_variant"])
            original_logits = forward_logits(model, tokenizer, device, prompt, no_hook())
            readout_original = p239.readout_metrics(tokenizer, original_logits, list(behavior.get("target_aliases") or []))
            original_margin = safe_float(readout_original.get("target_margin_vs_winner"))
            original_rollout = rollout_text(model, tokenizer, device, prompt, int(args.max_rollout_tokens), no_hook)
            interventions = [
                "no_intervention",
                "regime_suppression",
                "regime_injection",
                "target_injection_replay",
                "top_token_suppression_replay",
            ]
            case_rows: dict[str, dict[str, Any]] = {}
            for intervention in interventions:
                if intervention == "no_intervention":
                    logits_i = original_logits
                    rollout_i = original_rollout
                else:
                    ctx = intervention_context(model, intervention, layer_idx, regime_direction, target_direction, competitor_direction)
                    logits_i = forward_logits(model, tokenizer, device, prompt, ctx)

                    def rollout_ctx_factory(intervention_name: str = intervention) -> Any:
                        return intervention_context(model, intervention_name, layer_idx, regime_direction, target_direction, competitor_direction)

                    rollout_i = rollout_text(model, tokenizer, device, prompt, int(args.max_rollout_tokens), rollout_ctx_factory)
                readout = p239.readout_metrics(tokenizer, logits_i, list(behavior.get("target_aliases") or []))
                margin = safe_float(readout.get("target_margin_vs_winner"))
                margin_delta = margin - original_margin
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase249",
                    "source_phase": "Phase248",
                    "created_at": utc_now(),
                    "run_id": run_id,
                    "validation_id": f"phase249:{args.model}:{key[1]}:{key[2]}:{intervention}",
                    "phase248_candidate_id": candidate.get("candidate_id"),
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                    "data_split": behavior.get("data_split"),
                    "recommended_next_test": route,
                    "tested_regime": regime,
                    "intervention": intervention,
                    "final_observe_layer": layer_idx,
                    "perturb_norm": round(perturb_norm, 6),
                    "target_margin_vs_winner": round(margin, 6),
                    "target_margin_delta_vs_original": round(margin_delta, 6),
                    "target_rank": readout.get("target_rank"),
                    "winning_regime": readout.get("winning_regime"),
                    "second_competitor": readout.get("second_competitor"),
                    "top_token": readout.get("top_token"),
                    "winner_changed_vs_original": readout.get("winning_regime") != readout_original.get("winning_regime"),
                    "rollout_text": rollout_i,
                    "rollout_word_count": rollout_word_count(rollout_i),
                    "rollout_word_delta_vs_original": rollout_word_count(rollout_i) - rollout_word_count(original_rollout),
                    "effect_label": classify_margin(margin_delta),
                    "phase246_top_token_suppression_margin_delta": phase246_effects.get(key, {}).get("phase246_top_competitor_suppression_margin_delta"),
                    "phase246_target_injection_margin_delta": phase246_effects.get(key, {}).get("phase246_target_unembed_injection_margin_delta"),
                    "phase248_candidate_score": candidate.get("candidate_score"),
                }
                validation_rows.append(row)
                case_rows[intervention] = row
                if intervention == "regime_suppression":
                    suppression_rows.append(row)
                if intervention == "regime_injection":
                    injection_rows.append(row)
                rollout_rows.append(
                    {
                        **row,
                        "rollout_id": f"{row['validation_id']}:rollout",
                        "trace_level": "regime_rollout_effect",
                    }
                )
            suppression_delta = safe_float(case_rows.get("regime_suppression", {}).get("target_margin_delta_vs_original"))
            injection_delta = safe_float(case_rows.get("regime_injection", {}).get("target_margin_delta_vs_original"))
            top_replay_delta = safe_float(case_rows.get("top_token_suppression_replay", {}).get("target_margin_delta_vs_original"))
            target_replay_delta = safe_float(case_rows.get("target_injection_replay", {}).get("target_margin_delta_vs_original"))
            phase246_top_delta = safe_float(phase246_effects.get(key, {}).get("phase246_top_competitor_suppression_margin_delta"))
            comparison_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase249",
                    "created_at": utc_now(),
                    "comparison_id": f"phase249:comparison:{args.model}:{key[1]}:{key[2]}",
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                    "recommended_next_test": route,
                    "tested_regime": regime,
                    "regime_suppression_margin_delta": round(suppression_delta, 6),
                    "regime_injection_margin_delta": round(injection_delta, 6),
                    "target_injection_replay_margin_delta": round(target_replay_delta, 6),
                    "top_token_suppression_replay_margin_delta": round(top_replay_delta, 6),
                    "phase246_top_token_suppression_margin_delta": round(phase246_top_delta, 6),
                    "regime_minus_top_token_replay": round(suppression_delta - top_replay_delta, 6),
                    "regime_minus_phase246_top_token": round(suppression_delta - phase246_top_delta, 6),
                    "regime_better_than_top_token_replay": suppression_delta > top_replay_delta,
                    "regime_better_than_phase246_top_token": suppression_delta > phase246_top_delta,
                    "failure_type_hint": "competitor_regime_failure" if suppression_delta > 1.0 else ("target_pressure_failure" if target_replay_delta > 1.0 and suppression_delta <= 1.0 else "mixed_or_unresolved"),
                    "rollout_word_delta_regime_suppression": case_rows.get("regime_suppression", {}).get("rollout_word_delta_vs_original"),
                }
            )
            log(f"{args.model}: candidate={index}/{len(candidates)} validation_rows={len(validation_rows)}")
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

    metrics = metric_rows(args.model, validation_rows, comparison_rows)
    observations = observation_rows(validation_rows)
    edges = graph_edges(validation_rows, comparison_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Regime-level causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_count": len(candidates),
        "missing_rows": len(missing_rows),
        "validation_rows": len(validation_rows),
        "regime_suppression_rows": len(suppression_rows),
        "regime_injection_rows": len(injection_rows),
        "comparison_rows": len(comparison_rows),
        "rollout_effect_rows": len(rollout_rows),
        "mean_regime_suppression_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in suppression_rows), 6) if suppression_rows else 0.0,
        "mean_regime_injection_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in injection_rows), 6) if injection_rows else 0.0,
        "regime_better_than_top_token_replay_count": sum(1 for x in comparison_rows if x.get("regime_better_than_top_token_replay")),
        "regime_better_than_phase246_top_token_count": sum(1 for x in comparison_rows if x.get("regime_better_than_phase246_top_token")),
        "route_counts": dict(Counter(str(x.get("recommended_next_test")) for x in comparison_rows).most_common()),
        "failure_type_hints": dict(Counter(str(x.get("failure_type_hint")) for x in comparison_rows).most_common()),
        "effect_labels": dict(Counter(str(x.get("effect_label")) for x in validation_rows).most_common()),
    }
    write_json(out_dir / f"phase249_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase249_{args.model}_regime_causal_validation_rows.jsonl", validation_rows)
    write_jsonl(out_dir / f"phase249_{args.model}_regime_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / f"phase249_{args.model}_regime_injection_rows.jsonl", injection_rows)
    write_jsonl(out_dir / f"phase249_{args.model}_target_vs_regime_comparison_rows.jsonl", comparison_rows)
    write_jsonl(out_dir / f"phase249_{args.model}_rollout_effect_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase249_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase249_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase249_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase249_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def metric_rows(model_name: str, validation_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    by_intervention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        by_intervention[str(row.get("intervention"))].append(row)
    for intervention, items in by_intervention.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase249",
                "created_at": now,
                "metric_id": f"phase249:{model_name}:{intervention}:margin_delta",
                "scope": "regime_level_causal_validation",
                "model": model_name,
                "intervention": intervention,
                "metric_name": "mean_target_margin_delta_vs_original",
                "metric_value": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in items), 6),
                "winner_changed_rate": round(sum(1 for x in items if x.get("winner_changed_vs_original")) / max(1, len(items)), 4),
                "effect_labels": dict(Counter(str(x.get("effect_label")) for x in items).most_common()),
                "rows": len(items),
            }
        )
    if comparison_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase249",
                "created_at": now,
                "metric_id": f"phase249:{model_name}:regime_vs_top_token",
                "scope": "regime_level_causal_validation",
                "model": model_name,
                "metric_name": "regime_better_than_top_token_replay_rate",
                "metric_value": round(sum(1 for x in comparison_rows if x.get("regime_better_than_top_token_replay")) / len(comparison_rows), 6),
                "rows": len(comparison_rows),
            }
        )
    return rows


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase249",
                "created_at": now,
                "observation_id": f"phase249:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}:margin_delta",
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "regime_level_causal_validation",
                "component": row["intervention"],
                "metric_name": "target_margin_delta_vs_original",
                "metric_value": safe_float(row.get("target_margin_delta_vs_original")),
                "metric_unit": "logit",
                "tested_regime": row.get("tested_regime"),
                "effect_label": row.get("effect_label"),
                "data_split": row.get("data_split"),
            }
        )
    return out


def graph_edges(validation_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in validation_rows:
        if row.get("intervention") == "no_intervention":
            continue
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase249",
                "created_at": now,
                "edge_id": f"phase249:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}",
                "source": f"regime:{row.get('tested_regime')}" if "regime" in str(row.get("intervention")) else f"intervention:{row['intervention']}",
                "target": "node:ReadoutMargin",
                "edge_type": "regime_level_causal_validation",
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "evidence_type": "logit_margin_and_rollout_perturbation",
                "effect_direction": row.get("effect_label"),
                "effect_size": safe_float(row.get("target_margin_delta_vs_original")),
                "confidence": 0.52 if abs(safe_float(row.get("target_margin_delta_vs_original"))) >= 1.0 else 0.36,
                "supporting_phases": ["Phase246", "Phase248", "Phase249"],
                "status": "regime_causal_signal_not_closure",
            }
        )
    for row in comparison_rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase249",
                "created_at": now,
                "edge_id": f"phase249:comparison:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "source": f"regime:{row.get('tested_regime')}",
                "target": "intervention:top_token_suppression",
                "edge_type": "regime_vs_top_token_comparison",
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "evidence_type": "same_case_intervention_delta_comparison",
                "effect_direction": "regime_better" if row.get("regime_better_than_top_token_replay") else "top_token_better_or_equal",
                "effect_size": safe_float(row.get("regime_minus_top_token_replay")),
                "confidence": 0.45,
                "supporting_phases": ["Phase246", "Phase248", "Phase249"],
                "status": "comparison_not_closure",
            }
        )
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase249_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    validation_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        validation_rows.extend(read_jsonl(out_dir / f"phase249_{model}_regime_causal_validation_rows.jsonl"))
        suppression_rows.extend(read_jsonl(out_dir / f"phase249_{model}_regime_suppression_rows.jsonl"))
        injection_rows.extend(read_jsonl(out_dir / f"phase249_{model}_regime_injection_rows.jsonl"))
        comparison_rows.extend(read_jsonl(out_dir / f"phase249_{model}_target_vs_regime_comparison_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase249_{model}_rollout_effect_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase249_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase249_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase249_{model}_graph_edges.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase249_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.77,
        "candidate_clustering": 0.42,
        "case_bank_calibration": 0.39,
        "high_value_trace_selection": 0.62,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.36,
        "focused_causal_validation": 0.23,
        "raw_delta_vector_archive": 0.25,
        "raw_vector_factor_decomposition": 0.22,
        "regime_field_direction_bank": 0.24,
        "regime_level_causal_validation": 0.18,
        "gate_up_product_signature": 0.45,
        "residual_state_signature": 0.43,
        "readout_competition_trace": 0.67,
        "stepwise_rollout_trace": 0.24,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.58,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model regime-level causal validation",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "candidate_count": len(comparison_rows),
        "missing_rows": len(missing_rows),
        "validation_rows": len(validation_rows),
        "regime_suppression_rows": len(suppression_rows),
        "regime_injection_rows": len(injection_rows),
        "target_vs_regime_comparison_rows": len(comparison_rows),
        "rollout_effect_rows": len(rollout_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "mean_regime_suppression_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in suppression_rows), 6) if suppression_rows else 0.0,
        "mean_regime_injection_margin_delta": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in injection_rows), 6) if injection_rows else 0.0,
        "mean_top_token_replay_margin_delta": round(mean(safe_float(x.get("top_token_suppression_replay_margin_delta")) for x in comparison_rows), 6) if comparison_rows else 0.0,
        "regime_better_than_top_token_replay_count": sum(1 for x in comparison_rows if x.get("regime_better_than_top_token_replay")),
        "regime_better_than_phase246_top_token_count": sum(1 for x in comparison_rows if x.get("regime_better_than_phase246_top_token")),
        "route_counts": dict(Counter(str(x.get("recommended_next_test")) for x in comparison_rows).most_common()),
        "tested_regime_counts": dict(Counter(str(x.get("tested_regime")) for x in comparison_rows).most_common()),
        "failure_type_hints": dict(Counter(str(x.get("failure_type_hint")) for x in comparison_rows).most_common()),
        "effect_labels": dict(Counter(str(x.get("effect_label")) for x in validation_rows).most_common()),
        "pattern_atlas_progress": progress,
        "judgement": "first_regime_level_causal_signal_not_closure",
    }
    write_json(out_dir / "phase249_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase249_regime_causal_validation_rows.jsonl", validation_rows)
    write_jsonl(out_dir / "phase249_regime_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / "phase249_regime_injection_rows.jsonl", injection_rows)
    write_jsonl(out_dir / "phase249_target_vs_regime_comparison_rows.jsonl", comparison_rows)
    write_jsonl(out_dir / "phase249_rollout_effect_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase249_observations.jsonl", observations)
    write_jsonl(out_dir / "phase249_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase249_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase249_missing_rows.jsonl", missing_rows)
    write_report(out_dir / "phase249_regime_causal_validation_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase249 regime-level causal validation",
        "",
        "Phase249 tests whether token-bank regime directions have causal readout or early-rollout effects.",
        "It compares regime-level suppression with top-token suppression on the same Phase248 candidates.",
        "",
        "## Counts",
        "",
        f"- candidates: {summary['candidate_count']}",
        f"- validation_rows: {summary['validation_rows']}",
        f"- regime_suppression_rows: {summary['regime_suppression_rows']}",
        f"- target_vs_regime_comparison_rows: {summary['target_vs_regime_comparison_rows']}",
        "",
        "## Mean effects",
        "",
        f"- mean_regime_suppression_margin_delta: {summary['mean_regime_suppression_margin_delta']}",
        f"- mean_regime_injection_margin_delta: {summary['mean_regime_injection_margin_delta']}",
        f"- mean_top_token_replay_margin_delta: {summary['mean_top_token_replay_margin_delta']}",
        f"- regime_better_than_top_token_replay_count: {summary['regime_better_than_top_token_replay_count']}",
        "",
        "## Route counts",
        "",
        "```json",
        json.dumps(summary["route_counts"], ensure_ascii=False, indent=2),
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
            "latest_phase": "Phase249",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "validate regime-level causal signals before rollout and closure tracing",
            "small_model_bias_warning": "Phase249 uses qwen3/glm4/deepseek7b only; token-bank regime directions are proxy evidence, not closure.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase249 regime-level causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates-per-model", type=int, default=10)
    parser.add_argument("--max-rollout-tokens", type=int, default=4)
    parser.add_argument("--perturb-scale", type=float, default=0.35)
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
