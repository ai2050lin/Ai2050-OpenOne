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
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402


PHASE = 253
SOURCE_PHASE = 252
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE251_DIR = Path("tests/result/phase251_orthogonalized_natural_direction_causal_validation/orthogonalized_natural_direction_causal_validation")
RESULT_ROOT = Path("tests/result/phase253_control_readout_coupling_validation")
ROUND_DEFAULT = "control_readout_coupling_validation"

SPECS = {
    "qwen3": {"observe_layers": [20, 26, 29, 31, 33]},
    "glm4": {"observe_layers": [20, 26, 28, 30, 32]},
    "deepseek7b": {"observe_layers": [16, 22, 24, 26, 27]},
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


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel() or norm(a) <= 1e-8 or norm(b) <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def dot(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    return float(torch.dot(vec.float(), unit(direction).float()).item())


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return 0.0 if denom <= 1e-12 else sum(x * y for x, y in zip(dx, dy)) / denom


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_raw_vectors(model_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    out = {}
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if path.exists():
            out[key_for(item)] = {**item, "payload": torch.load(path, map_location="cpu")}
    return out


def load_natural_directions(model_name: str) -> dict[str, torch.Tensor]:
    rows = read_jsonl(PHASE250_DIR / "phase250_natural_direction_rows.jsonl")
    out = {}
    for row in rows:
        if str(row.get("model")) != model_name or str(row.get("scope")) != "global":
            continue
        path = Path(str(row.get("path")))
        if path.exists():
            out[str(row["contrast_id"])] = unit(torch.load(path, map_location="cpu")["direction"])
    return out


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta.to(device=changed.device, dtype=changed.dtype)
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def residual_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_hidden_logits(model: Any, tokenizer: Any, device: torch.device, prompt: str, observe_layers: list[int], hook_ctx: Any) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with hook_ctx:
        with torch.inference_mode():
            result = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = {
        int(layer): result.hidden_states[int(layer) + 1][0, last_pos].detach().float().cpu()
        for layer in observe_layers
        if int(layer) + 1 < len(result.hidden_states)
    }
    logits = result.logits[0, last_pos].detach().float().cpu()
    return hidden, logits


def stepwise_rollout(model: Any, tokenizer: Any, device: torch.device, prompt: str, steps: int, hook_factory: Any) -> tuple[str, list[dict[str, Any]]]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(encoded["input_ids"].shape[-1])
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    rows = []
    continuation_ids = set(p252.token_ids(tokenizer, [" the", " The", " is", " are", " and", " which", " because", " because"]))
    for step in range(1, int(steps) + 1):
        with hook_factory():
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0, -1, :].detach().float().cpu()
        next_id = int(torch.argmax(logits).item())
        scores = p252.closure_scores(tokenizer, logits)
        rows.append(
            {
                "step": step,
                "token_id": next_id,
                "token_text": tokenizer.decode([next_id]),
                "is_continuation_token": next_id in continuation_ids,
                **{k: round(v, 6) for k, v in scores.items()},
            }
        )
        nxt = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, nxt], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(nxt)], dim=1)
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
    text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True).strip()
    return text, rows


def update_stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"mean_closure_proxy_margin": 0.0, "continuation_token_rate": 0.0, "final_closure_proxy_margin": 0.0}
    return {
        "mean_closure_proxy_margin": round(mean(safe_float(x.get("closure_proxy_margin")) for x in rows), 6),
        "continuation_token_rate": round(sum(1 for x in rows if x.get("is_continuation_token")) / len(rows), 6),
        "final_closure_proxy_margin": safe_float(rows[-1].get("closure_proxy_margin")),
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    high_conf = read_jsonl(PHASE251_DIR / f"phase251_{args.model}_high_confidence_rollout_candidates.jsonl")
    observe_layers = list(SPECS[args.model]["observe_layers"])
    natural_dirs = load_natural_directions(args.model)
    model_obj = None
    tokenizer = None
    projection_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank_dirs = p252.build_tokenbank_directions(model_obj, tokenizer)
        for idx, item in enumerate(high_conf[: int(args.max_candidates_per_model)], start=1):
            key = key_for(item)
            behavior = behavior_by_key.get(key)
            raw_item = raw_vectors.get(key)
            if behavior is None or raw_item is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_behavior_or_raw"})
                continue
            raw = raw_item["payload"]
            route = str(item.get("recommended_next_test"))
            tokenbank_id, natural_id = p251.ROUTE_TO_DIRECTIONS.get(route, ("continuation_regime", "natural_continuation_explain"))
            perturb_norm = max(p252.norm(raw.get("target_direction")), p252.norm(raw.get("competitor_direction")), p252.norm(raw.get("delta_residual")) * float(args.perturb_scale), 1e-6)
            control_dir = normalize(natural_dirs[natural_id], perturb_norm)
            readout_dir = normalize(tokenbank_dirs[tokenbank_id], perturb_norm)
            combined_dir = normalize(unit(control_dir) + unit(readout_dir), perturb_norm)
            prompt = str(behavior["prompt_variant"])
            conditions: list[tuple[str, Any]] = [("no_intervention", no_hook)]
            for source_layer in observe_layers:
                conditions.append((f"natural_raw_suppression_at_L{source_layer}", lambda layer=source_layer, d=control_dir: residual_hook(model_obj, layer, d, sign=-1.0)))
            base_hidden, base_logits = capture_hidden_logits(model_obj, tokenizer, device, prompt, observe_layers, no_hook())
            base_readout = p239.readout_metrics(tokenizer, base_logits, list(behavior.get("target_aliases") or []))
            base_closure = p252.closure_scores(tokenizer, base_logits)
            baseline_by_layer = {
                layer: {
                    "control_projection": dot(vec, control_dir),
                    "readout_projection": dot(vec, readout_dir),
                }
                for layer, vec in base_hidden.items()
            }
            for condition, factory in conditions:
                hidden, logits = capture_hidden_logits(model_obj, tokenizer, device, prompt, observe_layers, factory())
                readout = p239.readout_metrics(tokenizer, logits, list(behavior.get("target_aliases") or []))
                closure = p252.closure_scores(tokenizer, logits)
                for layer, vec in hidden.items():
                    cproj = dot(vec, control_dir)
                    rproj = dot(vec, readout_dir)
                    projection_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase253",
                            "created_at": utc_now(),
                            "projection_id": f"phase253:projection:{args.model}:{key[1]}:{key[2]}:{condition}:L{layer}",
                            "model": args.model,
                            "case_id": key[1],
                            "variant_id": key[2],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                            "condition": condition,
                            "layer": layer,
                            "tokenbank_regime": tokenbank_id,
                            "natural_contrast_id": natural_id,
                            "control_projection": round(cproj, 6),
                            "readout_projection": round(rproj, 6),
                            "control_projection_delta": round(cproj - baseline_by_layer.get(layer, {}).get("control_projection", cproj), 6),
                            "readout_projection_delta": round(rproj - baseline_by_layer.get(layer, {}).get("readout_projection", rproj), 6),
                        }
                    )
                effect_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase253",
                        "created_at": utc_now(),
                        "effect_id": f"phase253:effect:{args.model}:{key[1]}:{key[2]}:{condition}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "family_id": behavior.get("family_id"),
                        "mode_id": behavior.get("mode_id"),
                        "condition": condition,
                        "target_margin_delta": round(safe_float(readout.get("target_margin_vs_winner")) - safe_float(base_readout.get("target_margin_vs_winner")), 6),
                        "closure_proxy_delta": round(safe_float(closure.get("closure_proxy_margin")) - safe_float(base_closure.get("closure_proxy_margin")), 6),
                        "winning_regime": readout.get("winning_regime"),
                        "base_winning_regime": base_readout.get("winning_regime"),
                    }
                )
            by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in projection_rows:
                if row["model"] == args.model and row["case_id"] == key[1] and row["variant_id"] == key[2]:
                    by_condition[str(row["condition"])].append(row)
            for condition, rows in by_condition.items():
                rows = sorted(rows, key=lambda x: int(x["layer"]))
                coupling_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase253",
                        "created_at": utc_now(),
                        "coupling_id": f"phase253:coupling:{args.model}:{key[1]}:{key[2]}:{condition}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "condition": condition,
                        "control_readout_layer_corr": round(pearson([safe_float(x["control_projection"]) for x in rows], [safe_float(x["readout_projection"]) for x in rows]), 6),
                        "control_first_to_readout_last_delta": round(safe_float(rows[-1]["readout_projection"]) - safe_float(rows[0]["control_projection"]), 6),
                        "readout_late_gain": round(safe_float(rows[-1]["readout_projection"]) - safe_float(rows[0]["readout_projection"]), 6),
                        "control_late_gain": round(safe_float(rows[-1]["control_projection"]) - safe_float(rows[0]["control_projection"]), 6),
                    }
                )
            rollout_conditions = [
                ("no_intervention", no_hook),
                ("tokenbank_suppression", lambda d=readout_dir: residual_hook(model_obj, observe_layers[-1], d, sign=-1.0)),
                ("natural_raw_suppression", lambda d=control_dir: residual_hook(model_obj, observe_layers[-1], d, sign=-1.0)),
                ("combined_suppression", lambda d=combined_dir: residual_hook(model_obj, observe_layers[-1], d, sign=-1.0)),
            ]
            for condition, factory in rollout_conditions:
                text, trace = stepwise_rollout(model_obj, tokenizer, device, prompt, int(args.rollout_steps), factory)
                stats = update_stats_for_rows(trace)
                for step_row in trace:
                    rollout_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase253",
                            "created_at": utc_now(),
                            "rollout_id": f"phase253:rollout:{args.model}:{key[1]}:{key[2]}:{condition}:step{step_row['step']}",
                            "model": args.model,
                            "case_id": key[1],
                            "variant_id": key[2],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                            "condition": condition,
                            "generated_text": text,
                            **stats,
                            **step_row,
                        }
                    )
            log(f"{args.model}: candidate={idx}/{min(len(high_conf), int(args.max_candidates_per_model))} projection_rows={len(projection_rows)} rollout_rows={len(rollout_rows)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    observations = observation_rows(projection_rows, effect_rows, rollout_rows)
    metrics = metric_rows(args.model, projection_rows, coupling_rows, effect_rows, rollout_rows)
    edges = graph_edges(projection_rows, coupling_rows, effect_rows)
    closure_candidates = closure_candidate_rows(rollout_rows, effect_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Control-to-readout coupling map validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_count": len({(x["case_id"], x["variant_id"]) for x in projection_rows}),
        "control_readout_projection_rows": len(projection_rows),
        "layerwise_coupling_rows": len(coupling_rows),
        "suppression_projection_effect_rows": len(effect_rows),
        "rollout_32token_rows": len(rollout_rows),
        "closure_validation_candidate_rows": len(closure_candidates),
        "missing_rows": len(missing_rows),
        "mean_closure_proxy_by_condition": mean_by(rollout_rows, "condition", "closure_proxy_margin"),
        "mean_continuation_rate_by_condition": mean_by(rollout_rows, "condition", "is_continuation_token"),
        "mean_target_margin_delta_by_condition": mean_by(effect_rows, "condition", "target_margin_delta"),
        "mean_closure_proxy_delta_by_condition": mean_by(effect_rows, "condition", "closure_proxy_delta"),
    }
    write_json(out_dir / f"phase253_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase253_{args.model}_control_readout_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / f"phase253_{args.model}_layerwise_coupling_rows.jsonl", coupling_rows)
    write_jsonl(out_dir / f"phase253_{args.model}_suppression_projection_effect_rows.jsonl", effect_rows)
    write_jsonl(out_dir / f"phase253_{args.model}_32token_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase253_{args.model}_closure_validation_candidates.jsonl", closure_candidates)
    write_jsonl(out_dir / f"phase253_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase253_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase253_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase253_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, bool):
            grouped[str(row.get(group_key))].append(1.0 if value else 0.0)
        else:
            grouped[str(row.get(group_key))].append(safe_float(value))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def closure_candidate_rows(rollout_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effect_by_key = {(x["model"], x["case_id"], x["variant_id"], x["condition"]): x for x in effect_rows}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[(row["model"], row["case_id"], row["variant_id"], row["condition"])].append(row)
    out = []
    for key, rows in grouped.items():
        if key[3] == "no_intervention":
            continue
        stats = update_stats_for_rows(rows)
        effect = effect_by_key.get(key, {})
        if stats["final_closure_proxy_margin"] > 0 or safe_float(effect.get("closure_proxy_delta")) > 1.0:
            out.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase253",
                    "created_at": utc_now(),
                    "candidate_id": f"phase253:closure_candidate:{':'.join(key)}",
                    "model": key[0],
                    "case_id": key[1],
                    "variant_id": key[2],
                    "condition": key[3],
                    **stats,
                    "closure_proxy_delta": effect.get("closure_proxy_delta"),
                    "target_margin_delta": effect.get("target_margin_delta"),
                    "selection_reason": "positive_final_closure_or_delta_gt_1",
                }
            )
    out.sort(key=lambda x: (safe_float(x.get("final_closure_proxy_margin")), safe_float(x.get("closure_proxy_delta"))), reverse=True)
    return out


def observation_rows(projections: list[dict[str, Any]], effects: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in projections:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "observation_id": row["projection_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "control_readout_projection",
                "component": row["condition"],
                "metric_name": "readout_projection",
                "metric_value": row["readout_projection"],
                "metric_unit": "projection",
            }
        )
    for row in effects:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "observation_id": row["effect_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "suppression_projection_effect",
                "component": row["condition"],
                "metric_name": "closure_proxy_delta",
                "metric_value": row["closure_proxy_delta"],
                "metric_unit": "logit",
            }
        )
    for row in rollouts:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "observation_id": row["rollout_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "32token_rollout_trace",
                "component": row["condition"],
                "metric_name": "closure_proxy_margin",
                "metric_value": row["closure_proxy_margin"],
                "metric_unit": "logit",
            }
        )
    return rows


def metric_rows(model_name: str, projections: list[dict[str, Any]], coupling: list[dict[str, Any]], effects: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for condition, value in mean_by(effects, "condition", "closure_proxy_delta").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "metric_id": f"phase253:{model_name}:{condition}:mean_closure_proxy_delta",
                "scope": "suppression_projection_effect",
                "model": model_name,
                "condition": condition,
                "metric_name": "mean_closure_proxy_delta",
                "metric_value": value,
                "rows": sum(1 for x in effects if x.get("condition") == condition),
            }
        )
    for condition, value in mean_by(rollouts, "condition", "closure_proxy_margin").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "metric_id": f"phase253:{model_name}:{condition}:mean_rollout_closure_proxy",
                "scope": "32token_rollout_trace",
                "model": model_name,
                "condition": condition,
                "metric_name": "mean_closure_proxy_margin",
                "metric_value": value,
                "rows": sum(1 for x in rollouts if x.get("condition") == condition),
            }
        )
    if coupling:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "metric_id": f"phase253:{model_name}:mean_control_readout_layer_corr",
                "scope": "layerwise_coupling",
                "model": model_name,
                "metric_name": "mean_control_readout_layer_corr",
                "metric_value": round(mean(safe_float(x.get("control_readout_layer_corr")) for x in coupling), 6),
                "rows": len(coupling),
            }
        )
    return rows


def graph_edges(projections: list[dict[str, Any]], coupling: list[dict[str, Any]], effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in coupling:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "edge_id": row["coupling_id"],
                "source": "axis:control",
                "target": "axis:readout",
                "edge_type": "layerwise_control_readout_coupling",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "layer_projection_correlation",
                "effect_direction": "aligned" if safe_float(row.get("control_readout_layer_corr")) >= 0 else "opposed",
                "effect_size": row["control_readout_layer_corr"],
                "confidence": 0.44,
                "supporting_phases": ["Phase251", "Phase252", "Phase253"],
                "status": "coupling_candidate_not_closure",
            }
        )
    for row in effects:
        if row.get("condition") == "no_intervention":
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase253",
                "created_at": now,
                "edge_id": f"phase253:effect_edge:{row['effect_id']}",
                "source": f"intervention:{row['condition']}",
                "target": "node:ClosureProxy",
                "edge_type": "suppression_to_closure_proxy",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "closure_proxy_delta",
                "effect_direction": "improved" if safe_float(row.get("closure_proxy_delta")) > 0 else "harmed_or_weak",
                "effect_size": row["closure_proxy_delta"],
                "confidence": 0.46,
                "supporting_phases": ["Phase252", "Phase253"],
                "status": "closure_proxy_not_model_close",
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase253_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    projection_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    closure_candidates: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        projection_rows.extend(read_jsonl(out_dir / f"phase253_{model}_control_readout_projection_rows.jsonl"))
        coupling_rows.extend(read_jsonl(out_dir / f"phase253_{model}_layerwise_coupling_rows.jsonl"))
        effect_rows.extend(read_jsonl(out_dir / f"phase253_{model}_suppression_projection_effect_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase253_{model}_32token_rollout_rows.jsonl"))
        closure_candidates.extend(read_jsonl(out_dir / f"phase253_{model}_closure_validation_candidates.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase253_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase253_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase253_{model}_graph_edges.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase253_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.81,
        "candidate_clustering": 0.43,
        "case_bank_calibration": 0.40,
        "high_value_trace_selection": 0.66,
        "trace_signature_validation": 0.38,
        "focused_causal_validation": 0.25,
        "raw_vector_factor_decomposition": 0.25,
        "regime_field_direction_bank": 0.35,
        "natural_regime_direction_bank": 0.30,
        "regime_level_causal_validation": 0.25,
        "shared_subspace_analysis": 0.20,
        "coupled_regime_field_analysis": 0.22,
        "control_readout_coupling": 0.18,
        "residual_state_signature": 0.49,
        "readout_competition_trace": 0.71,
        "stepwise_rollout_trace": 0.34,
        "causal_closure": 0.14,
        "general_language_mechanism_confidence": 0.62,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model control-to-readout coupling map validation",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "candidate_count": len({(x["model"], x["case_id"], x["variant_id"]) for x in projection_rows}),
        "control_readout_projection_rows": len(projection_rows),
        "layerwise_coupling_rows": len(coupling_rows),
        "suppression_projection_effect_rows": len(effect_rows),
        "rollout_32token_rows": len(rollout_rows),
        "closure_validation_candidate_rows": len(closure_candidates),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "mean_closure_proxy_by_condition": mean_by(rollout_rows, "condition", "closure_proxy_margin"),
        "mean_continuation_rate_by_condition": mean_by(rollout_rows, "condition", "is_continuation_token"),
        "mean_target_margin_delta_by_condition": mean_by(effect_rows, "condition", "target_margin_delta"),
        "mean_closure_proxy_delta_by_condition": mean_by(effect_rows, "condition", "closure_proxy_delta"),
        "top_closure_candidates": closure_candidates[:10],
        "pattern_atlas_progress": progress,
        "judgement": "control_readout_coupling_candidate_not_model_close",
    }
    write_json(out_dir / "phase253_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase253_control_readout_projection_rows.jsonl", projection_rows)
    write_jsonl(out_dir / "phase253_layerwise_coupling_rows.jsonl", coupling_rows)
    write_jsonl(out_dir / "phase253_suppression_projection_effect_rows.jsonl", effect_rows)
    write_jsonl(out_dir / "phase253_32token_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / "phase253_closure_validation_candidates.jsonl", closure_candidates)
    write_jsonl(out_dir / "phase253_observations.jsonl", observations)
    write_jsonl(out_dir / "phase253_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase253_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase253_missing_rows.jsonl", missing_rows)
    write_report(out_dir / "phase253_control_readout_coupling_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase253 control-to-readout coupling map validation",
        "",
        "Phase253 tracks layerwise control/readout projections and 32-token rollout closure proxies.",
        "It is not ModelClose validation.",
        "",
        "## Counts",
        "",
        f"- candidates: {summary['candidate_count']}",
        f"- control_readout_projection_rows: {summary['control_readout_projection_rows']}",
        f"- layerwise_coupling_rows: {summary['layerwise_coupling_rows']}",
        f"- rollout_32token_rows: {summary['rollout_32token_rows']}",
        f"- closure_validation_candidate_rows: {summary['closure_validation_candidate_rows']}",
        "",
        "## Mean Closure Proxy",
        "",
        "```json",
        json.dumps(summary["mean_closure_proxy_by_condition"], ensure_ascii=False, indent=2),
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
    progress = read_json(ATLAS_ROOT / "progress.json")
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase253",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "select true closure-validation cases from control-readout coupling traces",
            "small_model_bias_warning": "Phase253 uses qwen3/glm4/deepseek7b only; 32-token rollout remains proxy evidence, not ModelClose.",
        }
    )
    write_json(ATLAS_ROOT / "progress.json", progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase253 control-to-readout coupling validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates-per-model", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=32)
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
