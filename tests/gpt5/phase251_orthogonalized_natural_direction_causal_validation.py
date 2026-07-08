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


PHASE = 251
SOURCE_PHASE = 250
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE248_DIR = Path("tests/result/phase248_regime_level_direction_bank/regime_level_direction_bank")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
RESULT_ROOT = Path("tests/result/phase251_orthogonalized_natural_direction_causal_validation")
ROUND_DEFAULT = "orthogonalized_natural_direction_causal_validation"

SPECS = {
    "qwen3": {"final_observe_layer": 33},
    "glm4": {"final_observe_layer": 32},
    "deepseek7b": {"final_observe_layer": 27},
}

ORTH_ORDER = [
    "natural_protocol_short",
    "natural_continuation_explain",
    "natural_answer_boundary",
    "natural_target_seed",
    "natural_concise_answer",
]

ROUTE_TO_DIRECTIONS = {
    "continuation_regime_test": ("continuation_regime", "natural_continuation_explain"),
    "protocol_regime_test": ("protocol_short_regime", "natural_protocol_short"),
    "reason_regime_test": ("because_reason_regime", "natural_continuation_explain"),
    "boundary_regime_test": ("answer_boundary_regime", "natural_answer_boundary"),
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


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    if not rows:
        raise FileNotFoundError(f"missing Phase241 behavior rows under {PHASE241_DIR}")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_candidates(model_name: str, max_candidates_per_model: int) -> list[dict[str, Any]]:
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
        if path.exists():
            out[key_for(item)] = {**item, "payload": torch.load(path, map_location="cpu")}
    return out


def build_tokenbank_directions(model: Any, tokenizer: Any) -> dict[str, torch.Tensor]:
    output_weight = model.get_output_embeddings().weight.detach().float().cpu()
    directions: dict[str, torch.Tensor] = {}
    for regime, texts in p248.REGIME_TEXTS.items():
        token_ids = p248.token_ids_for_texts(tokenizer, texts)
        if regime == "period_stop_regime" and tokenizer.eos_token_id is not None:
            token_ids = sorted(set(token_ids + [int(tokenizer.eos_token_id)]))
        vectors = [unit(output_weight[token_id]) for token_id in token_ids if 0 <= token_id < output_weight.shape[0]]
        directions[regime] = unit(torch.stack(vectors).mean(dim=0)) if vectors else torch.zeros(output_weight.shape[1])
    return directions


def load_natural_directions(model_name: str) -> dict[str, torch.Tensor]:
    rows = read_jsonl(PHASE250_DIR / "phase250_natural_direction_rows.jsonl")
    out: dict[str, torch.Tensor] = {}
    for row in rows:
        if str(row.get("model")) != model_name or str(row.get("scope")) != "global":
            continue
        path = Path(str(row.get("path")))
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            out[str(row["contrast_id"])] = unit(payload["direction"])
    return out


def orthogonalize(directions: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    basis: list[torch.Tensor] = []
    out: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for contrast_id in ORTH_ORDER:
        raw = directions.get(contrast_id)
        if raw is None:
            continue
        vec = raw.clone().float()
        removed_norm = 0.0
        for b in basis:
            proj = torch.dot(vec, b) * b
            removed_norm += vector_norm(proj)
            vec = vec - proj
        ortho = unit(vec)
        if vector_norm(ortho) > 1e-8:
            basis.append(ortho)
        out[contrast_id] = ortho
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": utc_now(),
                "orth_direction_id": f"phase251:orth:{contrast_id}",
                "contrast_id": contrast_id,
                "order_index": ORTH_ORDER.index(contrast_id),
                "raw_norm": round(vector_norm(raw), 6),
                "orth_norm": round(vector_norm(ortho), 6),
                "removed_projection_norm_sum": round(removed_norm, 6),
            }
        )
    return out, rows


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


def rollout_word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.sub(r"\s+", " ", text).strip().split())


def classify(delta: float) -> str:
    if delta >= 1.0:
        return "readout_margin_gain"
    if delta <= -1.0:
        return "readout_margin_harmed"
    return "weak_or_no_readout_change"


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    vector_dir = out_dir / "orthogonalized_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    candidates = load_candidates(args.model, int(args.max_candidates_per_model))
    final_layer = int(SPECS[args.model]["final_observe_layer"])
    validation_rows: list[dict[str, Any]] = []
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
        tokenbank_units = build_tokenbank_directions(model, tokenizer)
        natural_units = load_natural_directions(args.model)
        orth_units, orth_rows_base = orthogonalize(natural_units)
        orth_rows = [{**row, "model": args.model} for row in orth_rows_base]
        for contrast_id, vec in orth_units.items():
            torch.save({"model": args.model, "contrast_id": contrast_id, "direction": vec}, vector_dir / f"{safe_slug(args.model + '_' + contrast_id)}.pt")
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
            tokenbank_id, natural_id = ROUTE_TO_DIRECTIONS.get(route, ("continuation_regime", "natural_continuation_explain"))
            prompt = str(behavior["prompt_variant"])
            perturb_norm = max(vector_norm(target_direction), vector_norm(competitor_direction), vector_norm(delta_residual) * float(args.perturb_scale), 1e-6)
            direction_map = {
                "tokenbank": normalize(tokenbank_units[tokenbank_id], perturb_norm),
                "natural_raw": normalize(natural_units[natural_id], perturb_norm),
                "natural_orth": normalize(orth_units[natural_id], perturb_norm),
            }
            base_logits = forward_logits(model, tokenizer, device, prompt, no_hook())
            base_readout = p239.readout_metrics(tokenizer, base_logits, list(behavior.get("target_aliases") or []))
            base_margin = safe_float(base_readout.get("target_margin_vs_winner"))
            base_rollout = rollout_text(model, tokenizer, device, prompt, int(args.max_rollout_tokens), no_hook)
            case_effects: dict[str, float] = {}
            for direction_source, direction in direction_map.items():
                for action, sign in [("suppression", -1.0), ("injection", 1.0)]:
                    intervention = f"{direction_source}_{action}"

                    def ctx_factory(d: torch.Tensor = direction, s: float = sign) -> Any:
                        return residual_direction_hook(model, final_layer, d, sign=s)

                    logits_i = forward_logits(model, tokenizer, device, prompt, ctx_factory())
                    readout = p239.readout_metrics(tokenizer, logits_i, list(behavior.get("target_aliases") or []))
                    rollout = rollout_text(model, tokenizer, device, prompt, int(args.max_rollout_tokens), ctx_factory)
                    margin = safe_float(readout.get("target_margin_vs_winner"))
                    margin_delta = margin - base_margin
                    case_effects[intervention] = margin_delta
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase251",
                        "source_phase": "Phase250",
                        "created_at": utc_now(),
                        "validation_id": f"phase251:{args.model}:{key[1]}:{key[2]}:{intervention}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "family_id": behavior.get("family_id"),
                        "mode_id": behavior.get("mode_id"),
                        "data_split": behavior.get("data_split"),
                        "recommended_next_test": route,
                        "tokenbank_regime": tokenbank_id,
                        "natural_contrast_id": natural_id,
                        "direction_source": direction_source,
                        "intervention_action": action,
                        "intervention": intervention,
                        "final_observe_layer": final_layer,
                        "perturb_norm": round(perturb_norm, 6),
                        "target_margin_vs_winner": round(margin, 6),
                        "target_margin_delta_vs_original": round(margin_delta, 6),
                        "target_rank": readout.get("target_rank"),
                        "winning_regime": readout.get("winning_regime"),
                        "second_competitor": readout.get("second_competitor"),
                        "winner_changed_vs_original": readout.get("winning_regime") != base_readout.get("winning_regime"),
                        "rollout_text": rollout,
                        "rollout_word_count": rollout_word_count(rollout),
                        "rollout_word_delta_vs_original": rollout_word_count(rollout) - rollout_word_count(base_rollout),
                        "effect_label": classify(margin_delta),
                        "phase248_candidate_score": candidate.get("candidate_score"),
                    }
                    validation_rows.append(row)
                    rollout_rows.append({**row, "rollout_id": f"{row['validation_id']}:rollout", "trace_level": "orthogonalized_natural_rollout_effect"})
            comparison_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase251",
                    "created_at": utc_now(),
                    "comparison_id": f"phase251:compare:{args.model}:{key[1]}:{key[2]}",
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                    "recommended_next_test": route,
                    "tokenbank_regime": tokenbank_id,
                    "natural_contrast_id": natural_id,
                    "tokenbank_suppression_delta": round(case_effects.get("tokenbank_suppression", 0.0), 6),
                    "natural_raw_suppression_delta": round(case_effects.get("natural_raw_suppression", 0.0), 6),
                    "natural_orth_suppression_delta": round(case_effects.get("natural_orth_suppression", 0.0), 6),
                    "natural_raw_minus_tokenbank": round(case_effects.get("natural_raw_suppression", 0.0) - case_effects.get("tokenbank_suppression", 0.0), 6),
                    "natural_orth_minus_tokenbank": round(case_effects.get("natural_orth_suppression", 0.0) - case_effects.get("tokenbank_suppression", 0.0), 6),
                    "natural_orth_better_than_tokenbank": case_effects.get("natural_orth_suppression", 0.0) > case_effects.get("tokenbank_suppression", 0.0),
                    "natural_orth_better_than_raw": case_effects.get("natural_orth_suppression", 0.0) > case_effects.get("natural_raw_suppression", 0.0),
                    "best_suppression_source": max(
                        ["tokenbank", "natural_raw", "natural_orth"],
                        key=lambda source: case_effects.get(f"{source}_suppression", -1e9),
                    ),
                    "best_suppression_delta": round(
                        max(case_effects.get("tokenbank_suppression", 0.0), case_effects.get("natural_raw_suppression", 0.0), case_effects.get("natural_orth_suppression", 0.0)),
                        6,
                    ),
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
    high_conf = high_confidence_rows(comparison_rows, validation_rows)
    observations = observation_rows(validation_rows)
    metrics = metric_rows(args.model, validation_rows, comparison_rows, orth_rows)
    edges = graph_edges(validation_rows, comparison_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Orthogonalized natural direction causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_count": len(candidates),
        "validation_rows": len(validation_rows),
        "comparison_rows": len(comparison_rows),
        "rollout_effect_rows": len(rollout_rows),
        "orthogonalized_direction_rows": len(orth_rows),
        "high_confidence_rollout_candidates": len(high_conf),
        "missing_rows": len(missing_rows),
        "mean_tokenbank_suppression_delta": mean_for(comparison_rows, "tokenbank_suppression_delta"),
        "mean_natural_raw_suppression_delta": mean_for(comparison_rows, "natural_raw_suppression_delta"),
        "mean_natural_orth_suppression_delta": mean_for(comparison_rows, "natural_orth_suppression_delta"),
        "natural_orth_better_than_tokenbank_count": sum(1 for x in comparison_rows if x.get("natural_orth_better_than_tokenbank")),
        "natural_orth_better_than_raw_count": sum(1 for x in comparison_rows if x.get("natural_orth_better_than_raw")),
        "best_suppression_sources": dict(Counter(str(x.get("best_suppression_source")) for x in comparison_rows).most_common()),
        "route_counts": dict(Counter(str(x.get("recommended_next_test")) for x in comparison_rows).most_common()),
    }
    write_json(out_dir / f"phase251_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase251_{args.model}_orthogonalized_natural_direction_rows.jsonl", orth_rows)
    write_jsonl(out_dir / f"phase251_{args.model}_tokenbank_vs_natural_direction_rows.jsonl", comparison_rows)
    write_jsonl(out_dir / f"phase251_{args.model}_natural_direction_causal_rows.jsonl", validation_rows)
    write_jsonl(out_dir / f"phase251_{args.model}_rollout_effect_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase251_{args.model}_high_confidence_rollout_candidates.jsonl", high_conf)
    write_jsonl(out_dir / f"phase251_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase251_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase251_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase251_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def mean_for(rows: list[dict[str, Any]], key: str) -> float:
    return round(mean(safe_float(x.get(key)) for x in rows), 6) if rows else 0.0


def high_confidence_rows(comparison_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(x["model"], x["case_id"], x["variant_id"], x["intervention"]): x for x in validation_rows}
    out = []
    for row in comparison_rows:
        best_delta = safe_float(row.get("best_suppression_delta"))
        if best_delta < 1.0:
            continue
        key = (row["model"], row["case_id"], row["variant_id"], f"{row['best_suppression_source']}_suppression")
        vrow = by_key.get(key, {})
        out.append(
            {
                **row,
                "candidate_id": f"phase251:high_conf:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "rollout_text": vrow.get("rollout_text"),
                "rollout_word_delta_vs_original": vrow.get("rollout_word_delta_vs_original"),
                "selection_reason": "best_suppression_delta_ge_1",
            }
        )
    out.sort(key=lambda x: safe_float(x.get("best_suppression_delta")), reverse=True)
    return out


def observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase251",
            "created_at": now,
            "observation_id": f"phase251:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}:margin_delta",
            "case_id": row["case_id"],
            "model": row["model"],
            "family_id": row.get("family_id"),
            "mode_id": row.get("mode_id"),
            "variant_id": row["variant_id"],
            "level": "orthogonalized_natural_direction_causal_validation",
            "component": row["intervention"],
            "metric_name": "target_margin_delta_vs_original",
            "metric_value": safe_float(row.get("target_margin_delta_vs_original")),
            "metric_unit": "logit",
            "direction_source": row.get("direction_source"),
            "effect_label": row.get("effect_label"),
        }
        for row in rows
    ]


def metric_rows(model_name: str, validation_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], orth_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        grouped[(str(row.get("direction_source")), str(row.get("intervention_action")))].append(row)
    for (direction_source, action), items in grouped.items():
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": now,
                "metric_id": f"phase251:{model_name}:{direction_source}:{action}:margin_delta",
                "scope": "orthogonalized_natural_direction_causal_validation",
                "model": model_name,
                "direction_source": direction_source,
                "intervention_action": action,
                "metric_name": "mean_target_margin_delta_vs_original",
                "metric_value": round(mean(safe_float(x.get("target_margin_delta_vs_original")) for x in items), 6),
                "rows": len(items),
            }
        )
    for row in orth_rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": now,
                "metric_id": f"phase251:{model_name}:orth:{row['contrast_id']}",
                "scope": "orthogonalized_natural_direction",
                "model": model_name,
                "contrast_id": row["contrast_id"],
                "metric_name": "removed_projection_norm_sum",
                "metric_value": row["removed_projection_norm_sum"],
                "rows": 1,
            }
        )
    if comparison_rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": now,
                "metric_id": f"phase251:{model_name}:natural_orth_better_than_tokenbank",
                "scope": "orthogonalized_natural_direction_causal_validation",
                "model": model_name,
                "metric_name": "natural_orth_better_than_tokenbank_rate",
                "metric_value": round(sum(1 for x in comparison_rows if x.get("natural_orth_better_than_tokenbank")) / len(comparison_rows), 6),
                "rows": len(comparison_rows),
            }
        )
    return out


def graph_edges(validation_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in validation_rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": now,
                "edge_id": f"phase251:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['intervention']}",
                "source": f"direction:{row['direction_source']}:{row.get('natural_contrast_id') or row.get('tokenbank_regime')}",
                "target": "node:ReadoutMargin",
                "edge_type": "orthogonalized_natural_direction_causal_validation",
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "evidence_type": "logit_margin_and_rollout_perturbation",
                "effect_direction": row.get("effect_label"),
                "effect_size": safe_float(row.get("target_margin_delta_vs_original")),
                "confidence": 0.52 if abs(safe_float(row.get("target_margin_delta_vs_original"))) >= 1.0 else 0.36,
                "supporting_phases": ["Phase248", "Phase249", "Phase250", "Phase251"],
                "status": "direction_comparison_not_closure",
            }
        )
    for row in comparison_rows:
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase251",
                "created_at": now,
                "edge_id": f"phase251:compare:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "source": "direction:natural_orth",
                "target": "direction:tokenbank",
                "edge_type": "direction_source_comparison",
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "evidence_type": "same_case_direction_suppression_delta",
                "effect_direction": "natural_orth_better" if row.get("natural_orth_better_than_tokenbank") else "tokenbank_better_or_equal",
                "effect_size": safe_float(row.get("natural_orth_minus_tokenbank")),
                "confidence": 0.44,
                "supporting_phases": ["Phase249", "Phase250", "Phase251"],
                "status": "direction_source_comparison_not_closure",
            }
        )
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase251_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    orth_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    high_conf: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        orth_rows.extend(read_jsonl(out_dir / f"phase251_{model}_orthogonalized_natural_direction_rows.jsonl"))
        comparison_rows.extend(read_jsonl(out_dir / f"phase251_{model}_tokenbank_vs_natural_direction_rows.jsonl"))
        validation_rows.extend(read_jsonl(out_dir / f"phase251_{model}_natural_direction_causal_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase251_{model}_rollout_effect_rows.jsonl"))
        high_conf.extend(read_jsonl(out_dir / f"phase251_{model}_high_confidence_rollout_candidates.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase251_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase251_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase251_{model}_graph_edges.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase251_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.79,
        "candidate_clustering": 0.43,
        "case_bank_calibration": 0.40,
        "high_value_trace_selection": 0.64,
        "first_internal_trace_batch": 0.38,
        "trace_signature_validation": 0.37,
        "focused_causal_validation": 0.24,
        "raw_delta_vector_archive": 0.26,
        "raw_vector_factor_decomposition": 0.24,
        "regime_field_direction_bank": 0.33,
        "natural_regime_direction_bank": 0.28,
        "regime_level_causal_validation": 0.23,
        "orthogonalized_direction_validation": 0.16,
        "residual_state_signature": 0.46,
        "readout_competition_trace": 0.69,
        "stepwise_rollout_trace": 0.26,
        "causal_closure": 0.12,
        "general_language_mechanism_confidence": 0.60,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model orthogonalized natural direction causal validation",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "candidate_count": len(comparison_rows),
        "validation_rows": len(validation_rows),
        "comparison_rows": len(comparison_rows),
        "rollout_effect_rows": len(rollout_rows),
        "orthogonalized_direction_rows": len(orth_rows),
        "high_confidence_rollout_candidates": len(high_conf),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "mean_tokenbank_suppression_delta": mean_for(comparison_rows, "tokenbank_suppression_delta"),
        "mean_natural_raw_suppression_delta": mean_for(comparison_rows, "natural_raw_suppression_delta"),
        "mean_natural_orth_suppression_delta": mean_for(comparison_rows, "natural_orth_suppression_delta"),
        "natural_orth_better_than_tokenbank_count": sum(1 for x in comparison_rows if x.get("natural_orth_better_than_tokenbank")),
        "natural_orth_better_than_raw_count": sum(1 for x in comparison_rows if x.get("natural_orth_better_than_raw")),
        "best_suppression_sources": dict(Counter(str(x.get("best_suppression_source")) for x in comparison_rows).most_common()),
        "route_counts": dict(Counter(str(x.get("recommended_next_test")) for x in comparison_rows).most_common()),
        "top_high_confidence_candidates": high_conf[:10],
        "pattern_atlas_progress": progress,
        "judgement": "orthogonalized_natural_direction_test_not_closure",
    }
    write_json(out_dir / "phase251_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase251_orthogonalized_natural_direction_rows.jsonl", orth_rows)
    write_jsonl(out_dir / "phase251_tokenbank_vs_natural_direction_rows.jsonl", comparison_rows)
    write_jsonl(out_dir / "phase251_natural_direction_causal_rows.jsonl", validation_rows)
    write_jsonl(out_dir / "phase251_rollout_effect_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase251_high_confidence_rollout_candidates.jsonl", high_conf)
    write_jsonl(out_dir / "phase251_observations.jsonl", observations)
    write_jsonl(out_dir / "phase251_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase251_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase251_missing_rows.jsonl", missing_rows)
    write_report(out_dir / "phase251_natural_direction_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase251 orthogonalized natural direction causal validation",
        "",
        "Phase251 compares token-bank directions, natural contrast directions, and orthogonalized natural directions.",
        "It is a direction-source validation stage, not closure.",
        "",
        "## Counts",
        "",
        f"- candidates: {summary['candidate_count']}",
        f"- validation_rows: {summary['validation_rows']}",
        f"- high_confidence_rollout_candidates: {summary['high_confidence_rollout_candidates']}",
        "",
        "## Mean Suppression Effects",
        "",
        f"- tokenbank: {summary['mean_tokenbank_suppression_delta']}",
        f"- natural_raw: {summary['mean_natural_raw_suppression_delta']}",
        f"- natural_orth: {summary['mean_natural_orth_suppression_delta']}",
        f"- natural_orth_better_than_tokenbank_count: {summary['natural_orth_better_than_tokenbank_count']}",
        "",
        "## Best Suppression Sources",
        "",
        "```json",
        json.dumps(summary["best_suppression_sources"], ensure_ascii=False, indent=2),
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
            "latest_phase": "Phase251",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "use high-confidence direction-source candidates for rollout and closure tracing",
            "small_model_bias_warning": "Phase251 uses qwen3/glm4/deepseek7b only; orthogonalized natural directions are causal signals, not closure.",
        }
    )
    write_json(progress_path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase251 orthogonalized natural direction causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates-per-model", type=int, default=10)
    parser.add_argument("--max-rollout-tokens", type=int, default=8)
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
