#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_mlp  # noqa: E402


PHASE = 271
SOURCE_PHASE = 270
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE268_ROOT = Path("tests/result/phase268_attention_mlp_continuation_path_attribution/attention_mlp_continuation_path_attribution")
RESULT_ROOT = Path("tests/result/phase271_mlp_writer_direction_closure_fiber_audit")
ROUND_DEFAULT = "mlp_writer_direction_closure_fiber_audit"
WINDOW_RADIUS = 2
PATCH_TYPES = [
    "window_mlp_zero",
    "window_mlp_half",
    "window_mlp_mean_replace",
    "window_mlp_random_same_norm",
]


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
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def random_same_norm(target: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=target.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(target.shape, generator=gen, device=target.device, dtype=target.dtype)
    target_norm = torch.linalg.vector_norm(target.float(), dim=-1, keepdim=True).clamp_min(1e-6).to(target.dtype)
    noise_norm = torch.linalg.vector_norm(noise.float(), dim=-1, keepdim=True).clamp_min(1e-6).to(target.dtype)
    return noise * (target_norm / noise_norm)


def select_cases(model: str, cases_per_model: int) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(PHASE268_ROOT / "phase268_component_summary_rows.jsonl") if r.get("model") == model]
    rows.sort(key=lambda r: (-safe_float(r.get("sum_positive_mlp_delta")), -safe_float(r.get("final_continue_stop_margin"))))
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for row in rows:
        family = str(row.get("family_id"))
        if family in used_families:
            continue
        selected.append(row)
        used_families.add(family)
        if len(selected) >= cases_per_model:
            return selected
    for row in rows:
        if len(selected) >= cases_per_model:
            break
        if row not in selected:
            selected.append(row)
    return selected[:cases_per_model]


def make_window(center: int, n_layers: int, radius: int = WINDOW_RADIUS) -> list[int]:
    return [i for i in range(center - radius, center + radius + 1) if 0 <= i < n_layers]


def alias_token_ids(tokenizer: Any, aliases: list[str]) -> list[int]:
    ids: list[int] = []
    for text in aliases:
        for variant in [text, " " + text]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
    return sorted(set(ids))


def score_logits(tokenizer: Any, logits: torch.Tensor, aliases: list[str]) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
    cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
    scores = {**p262.score_channels(logits, stop_ids, cont_ids), **p239.readout_metrics(tokenizer, logits, aliases)}
    answer_ids = alias_token_ids(tokenizer, aliases)
    if answer_ids:
        answer_logit = float(logits[answer_ids].max().item())
        masked = logits.clone()
        masked[answer_ids] = -float("inf")
        top_non_answer = float(masked.max().item())
        answer_rank = int((logits > answer_logit).sum().item()) + 1
    else:
        answer_logit = safe_float(scores.get("target_logit"), -1e30)
        top_non_answer = float(logits.max().item())
        answer_rank = int((logits > answer_logit).sum().item()) + 1
    stop = safe_float(scores.get("r_stop"), -1e30)
    cont = safe_float(scores.get("r_continue"), -1e30)
    target = safe_float(scores.get("target_logit"), answer_logit)
    winner_value = max(stop, cont, target)
    if winner_value == target:
        winner = "target"
    elif winner_value == cont:
        winner = "continue"
    else:
        winner = "stop"
    blocker_max = max(cont, stop, top_non_answer)
    scores.update(
        {
            "tri_winner": winner,
            "continue_stop_margin": cont - stop,
            "answer_class_logit": answer_logit,
            "answer_class_rank": answer_rank,
            "top_non_answer_logit": top_non_answer,
            "blocker_max_logit": blocker_max,
            "answer_boundary_margin": answer_logit - blocker_max,
            "target_boundary_margin": target - blocker_max,
        }
    )
    return scores


def patch_tensor(output: Any, spec: dict[str, Any]) -> Any:
    y = extract_tensor(output)
    if y is None:
        return output
    patched = y.clone()
    if patched.ndim == 3:
        target = patched[:, -1, :]
        patched[:, -1, :] = transform_target(target, spec)
    elif patched.ndim == 2:
        target = patched[-1:, :]
        patched[-1:, :] = transform_target(target, spec)
    elif patched.ndim == 1:
        target = patched.unsqueeze(0)
        patched = transform_target(target, spec).squeeze(0)
    return replace_tensor(output, patched)


def transform_target(target: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    mode = str(spec["mode"])
    if mode == "scale":
        return target * float(spec["scale"])
    if mode == "mean_replace":
        mean_vector = spec["mean_vector"].to(device=target.device, dtype=target.dtype)
        return mean_vector.reshape(1, -1).expand_as(target)
    if mode == "random_same_norm":
        return random_same_norm(target, int(spec["random_seed"]))
    raise ValueError(f"unknown patch mode {mode}")


def install_mlp_hooks(model_obj: Any, specs: list[dict[str, Any]]) -> list[Any]:
    layers = get_layers(model_obj)
    handles: list[Any] = []
    for spec in specs:
        layer_idx = int(spec["layer"])
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        module = get_mlp(layers[layer_idx])
        if module is None:
            continue

        def hook(_module, _inputs, output, spec=spec):
            return patch_tensor(output, spec)

        handles.append(module.register_forward_hook(hook))
    return handles


def with_hooks_logits(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, specs: list[dict[str, Any]]) -> torch.Tensor:
    handles = install_mlp_hooks(model_obj, specs)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            out = model_obj(**encoded, use_cache=False, return_dict=True)
        return out.logits[0, last_pos].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def with_hooks_generate(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, specs: list[dict[str, Any]], max_new_tokens: int) -> tuple[str, int]:
    handles = install_mlp_hooks(model_obj, specs)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    input_len = int(encoded["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            out = model_obj.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, input_len:].detach().cpu().tolist()
        return tokenizer.decode(new_ids, skip_special_tokens=False), len(new_ids)
    finally:
        for handle in handles:
            handle.remove()


def collect_mlp_means(model_obj: Any, tokenizer: Any, device: torch.device, prompts: list[str], layers_needed: list[int]) -> dict[int, torch.Tensor]:
    layers = get_layers(model_obj)
    sums: dict[int, torch.Tensor] = {}
    counts: Counter[int] = Counter()
    handles: list[Any] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            if y is None:
                return output
            if y.ndim == 3:
                vec = y[:, -1, :].detach().float().cpu().sum(dim=0)
                n = int(y.shape[0])
            elif y.ndim == 2:
                vec = y[-1, :].detach().float().cpu()
                n = 1
            else:
                vec = y.detach().float().cpu()
                n = 1
            sums[layer_idx] = sums.get(layer_idx, torch.zeros_like(vec)) + vec
            counts[layer_idx] += n
            return output

        return hook

    for layer_idx in sorted(set(layers_needed)):
        module = get_mlp(layers[layer_idx])
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(layer_idx)))
    try:
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
            with torch.inference_mode():
                model_obj(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: sums[layer] / max(1, counts[layer]) for layer in sums}


def patch_specs(patch_type: str, center: int, n_layers: int, model: str, case_id: str, means: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
    window = make_window(center, n_layers)
    if patch_type == "window_mlp_zero":
        return [{"layer": layer, "mode": "scale", "scale": 0.0} for layer in window]
    if patch_type == "window_mlp_half":
        return [{"layer": layer, "mode": "scale", "scale": 0.5} for layer in window]
    if patch_type == "window_mlp_mean_replace":
        return [{"layer": layer, "mode": "mean_replace", "mean_vector": means[layer]} for layer in window if layer in means]
    if patch_type == "window_mlp_random_same_norm":
        return [
            {"layer": layer, "mode": "random_same_norm", "random_seed": stable_seed(f"phase271:{model}:{case_id}:L{layer}:window_random")}
            for layer in window
        ]
    raise ValueError(f"unknown patch type {patch_type}")


def closure_fiber(base_scores: dict[str, Any], patched_scores: dict[str, Any], patch_type: str) -> dict[str, Any]:
    target_lift = safe_float(patched_scores.get("answer_class_logit")) - safe_float(base_scores.get("answer_class_logit"))
    rank_margin_delta = safe_float(patched_scores.get("answer_boundary_margin")) - safe_float(base_scores.get("answer_boundary_margin"))
    blocker_suppression = safe_float(base_scores.get("blocker_max_logit")) - safe_float(patched_scores.get("blocker_max_logit"))
    continue_margin_delta = safe_float(patched_scores.get("continue_stop_margin")) - safe_float(base_scores.get("continue_stop_margin"))
    side_effect_score = abs(target_lift) + abs(rank_margin_delta)
    fiber_score = target_lift + rank_margin_delta + blocker_suppression - side_effect_score
    field_admissible = str(base_scores.get("tri_winner")) == "continue"
    blocker_suppressed = blocker_suppression > 0.0
    positive_boundary_move = rank_margin_delta > 0.0
    low_side_effect = side_effect_score < 2.0
    clean_edge_candidate = bool(
        field_admissible
        and blocker_suppressed
        and low_side_effect
        and continue_margin_delta < 0.0
        and patch_type != "window_mlp_random_same_norm"
    )
    return {
        "target_lift": round(target_lift, 6),
        "rank_margin_delta": round(rank_margin_delta, 6),
        "blocker_suppression": round(blocker_suppression, 6),
        "continue_margin_delta": round(continue_margin_delta, 6),
        "side_effect_score": round(side_effect_score, 6),
        "closure_fiber_score": round(fiber_score, 6),
        "field_admissible": field_admissible,
        "blocker_suppressed": blocker_suppressed,
        "positive_boundary_move": positive_boundary_move,
        "low_side_effect": low_side_effect,
        "clean_edge_candidate": clean_edge_candidate,
    }


def row_base(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase271",
        "created_at": utc_now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": source["mode_id"],
        "variant_id": source["variant_id"],
        "path_schema_id": source["path_schema_id"],
        "target": source["target"],
        "top_continue_channel_phase268": case.get("top_continue_channel_phase267"),
        "strongest_mlp_layer_phase268": case.get("strongest_mlp_layer"),
        "strongest_mlp_delta_phase268": case.get("strongest_mlp_delta"),
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_bank = {r["case_id"]: r for r in read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    selected = select_cases(args.model, int(args.cases_per_model))
    model_obj = None
    tokenizer = None
    direction_rows: list[dict[str, Any]] = []
    fiber_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model_obj))
        prompts: list[str] = []
        layers_needed: list[int] = []
        valid_cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for case in selected:
            source = case_bank.get(str(case["case_id"]))
            if not source:
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": "Phase271", "missing_id": f"phase271:missing:{args.model}:{case['case_id']}", "model": args.model, "case_id": case["case_id"], "reason": "case not found"})
                continue
            valid_cases.append((case, source))
            prompts.append(str(source["prompt"]))
            layers_needed.extend(make_window(int(case["strongest_mlp_layer"]), n_layers))
        means = collect_mlp_means(model_obj, tokenizer, device, prompts, layers_needed)
        log(f"{args.model}: collected MLP means for {len(means)} layers")
        for idx, (case, source) in enumerate(valid_cases, start=1):
            center = int(case["strongest_mlp_layer"])
            window = make_window(center, n_layers)
            base = row_base(case, source)
            prompt = str(source["prompt"])
            aliases = [str(x) for x in source.get("target_aliases") or [source.get("target", "")]]
            try:
                base_logits = with_hooks_logits(model_obj, tokenizer, device, prompt, [])
                base_scores = score_logits(tokenizer, base_logits, aliases)
                base_text, base_new_tokens = with_hooks_generate(model_obj, tokenizer, device, prompt, [], int(args.rollout_tokens))
                for patch_type in PATCH_TYPES:
                    specs = patch_specs(patch_type, center, n_layers, args.model, str(case["case_id"]), means)
                    patched_logits = with_hooks_logits(model_obj, tokenizer, device, prompt, specs)
                    patched_scores = score_logits(tokenizer, patched_logits, aliases)
                    fiber = closure_fiber(base_scores, patched_scores, patch_type)
                    patched_text, patched_new_tokens = with_hooks_generate(model_obj, tokenizer, device, prompt, specs, int(args.rollout_tokens))
                    delta_continue_stop = fiber["continue_margin_delta"]
                    row = {
                        **base,
                        "direction_control_id": f"phase271:direction:{args.model}:{case['case_id']}:L{center}:{patch_type}",
                        "patch_type": patch_type,
                        "center_layer": center,
                        "window_layers": window,
                        "patched_component_count": len(specs),
                        "base_continue_stop_margin": round(safe_float(base_scores.get("continue_stop_margin")), 6),
                        "patched_continue_stop_margin": round(safe_float(patched_scores.get("continue_stop_margin")), 6),
                        "delta_continue_stop_margin": delta_continue_stop,
                        "base_winner": base_scores.get("tri_winner"),
                        "patched_winner": patched_scores.get("tri_winner"),
                        "winner_changed": base_scores.get("tri_winner") != patched_scores.get("tri_winner"),
                        "base_answer_boundary_margin": round(safe_float(base_scores.get("answer_boundary_margin")), 6),
                        "patched_answer_boundary_margin": round(safe_float(patched_scores.get("answer_boundary_margin")), 6),
                        "base_answer_rank": int(base_scores.get("answer_class_rank", 0)),
                        "patched_answer_rank": int(patched_scores.get("answer_class_rank", 0)),
                        "direction_effect_supported": bool(delta_continue_stop < -1.0 or base_scores.get("tri_winner") != patched_scores.get("tri_winner")),
                        "state_integrity_risk": bool(patch_type == "window_mlp_random_same_norm" and abs(delta_continue_stop) > 1.0),
                        **fiber,
                    }
                    direction_rows.append(row)
                    fiber_rows.append(
                        {
                            **base,
                            "closure_fiber_id": row["direction_control_id"].replace(":direction:", ":fiber:"),
                            "patch_type": patch_type,
                            "center_layer": center,
                            "window_layers": window,
                            **fiber,
                        }
                    )
                    if patch_type in {"window_mlp_random_same_norm", "window_mlp_mean_replace"}:
                        control_rows.append({**row, "control_id": row["direction_control_id"].replace(":direction:", ":control:"), "control_type": patch_type})
                    rollout_rows.append(
                        {
                            **base,
                            "rollout_effect_id": row["direction_control_id"].replace(":direction:", ":rollout:"),
                            "patch_type": patch_type,
                            "center_layer": center,
                            "window_layers": window,
                            "base_text": base_text[:300],
                            "patched_text": patched_text[:300],
                            "base_new_tokens": base_new_tokens,
                            "patched_new_tokens": patched_new_tokens,
                            "rollout_changed": base_text != patched_text,
                        }
                    )
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase271",
                            "created_at": utc_now(),
                            "observation_id": row["direction_control_id"].replace(":direction:", ":obs:"),
                            "case_id": case["case_id"],
                            "model": args.model,
                            "family_id": case["family_id"],
                            "level": "mlp_writer_direction_closure_fiber_audit",
                            "component": f"{patch_type}:L{center}:window{window}",
                            "metric_name": "closure_fiber_score",
                            "metric_value": row["closure_fiber_score"],
                            "metric_unit": "fiber_score",
                            "winner": row["patched_winner"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase271",
                        "created_at": utc_now(),
                        "missing_id": f"phase271:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case.get("family_id"),
                        "reason": repr(exc),
                    }
                )
            log(f"{args.model}: direction/fiber audited {idx}/{len(valid_cases)} cases")
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
    metrics = make_metrics(args.model, direction_rows, fiber_rows, control_rows)
    edges = make_edges(args.model, direction_rows, fiber_rows)
    payload = summarize_model(args.model, selected, direction_rows, fiber_rows, control_rows, rollout_rows, observations, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, direction_rows, fiber_rows, control_rows, rollout_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, direction: list[dict[str, Any]], fiber: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_patch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direction:
        by_patch[str(row["patch_type"])].append(row)
    for patch_type, vals in sorted(by_patch.items()):
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase271",
                "created_at": utc_now(),
                "metric_id": f"phase271:{model}:{patch_type}:direction_effect",
                "scope": "mlp_writer_direction_closure_fiber_audit",
                "model": model,
                "patch_type": patch_type,
                "metric_name": "mean_delta_continue_stop_margin",
                "metric_value": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in vals]),
                "mean_closure_fiber_score": mean_safe([safe_float(r["closure_fiber_score"]) for r in vals]),
                "direction_effect_supported_rate": round(sum(1 for r in vals if r.get("direction_effect_supported")) / len(vals), 6) if vals else 0.0,
                "state_integrity_risk_rate": round(sum(1 for r in vals if r.get("state_integrity_risk")) / len(vals), 6) if vals else 0.0,
                "clean_edge_candidate_rate": round(sum(1 for r in vals if r.get("clean_edge_candidate")) / len(vals), 6) if vals else 0.0,
                "rows": len(vals),
            }
        )
    clean_candidates = [r for r in fiber if r.get("clean_edge_candidate")]
    out.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase271",
            "created_at": utc_now(),
            "metric_id": f"phase271:{model}:clean_edge_candidate_rate",
            "scope": "closure_fiber_quality_control",
            "model": model,
            "metric_name": "clean_edge_candidate_rate",
            "metric_value": round(len(clean_candidates) / len(fiber), 6) if fiber else 0.0,
            "rows": len(fiber),
        }
    )
    return out


def make_edges(model: str, direction: list[dict[str, Any]], fiber: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], r["patch_type"], bool(r["direction_effect_supported"]), bool(r["state_integrity_risk"])) for r in direction)
    for (family, patch_type, supported, risk), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase271",
                "created_at": utc_now(),
                "edge_id": f"phase271:{model}:{family}:{patch_type}:{supported}:{risk}",
                "source": f"node:{family}",
                "target": f"node:{patch_type}",
                "edge_type": "mlp_direction_control_effect",
                "model": model,
                "direction_effect_supported": supported,
                "state_integrity_risk": risk,
                "effect_size": count,
                "status": "direction_control_not_closure",
            }
        )
    clean_grouped = Counter((r["family_id"], r["patch_type"], bool(r["clean_edge_candidate"])) for r in fiber)
    for (family, patch_type, clean, count) in [(a, b, c, n) for (a, b, c), n in clean_grouped.items()]:
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase271",
                "created_at": utc_now(),
                "edge_id": f"phase271:{model}:{family}:{patch_type}:clean:{clean}",
                "source": f"node:{family}",
                "target": f"node:closure_fiber:{patch_type}",
                "edge_type": "closure_fiber_quality_control",
                "model": model,
                "clean_edge_candidate": clean,
                "effect_size": count,
                "status": "quality_control_not_closure",
            }
        )
    return edges


def summarize_model(model: str, selected: list[dict[str, Any]], direction: list[dict[str, Any]], fiber: list[dict[str, Any]], controls: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP writer direction and closure-fiber control audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(selected),
        "direction_control_rows": len(direction),
        "closure_fiber_rows": len(fiber),
        "control_rows": len(controls),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in direction)),
        "direction_effect_supported_counts": dict(Counter(str(r["direction_effect_supported"]) for r in direction)),
        "state_integrity_risk_counts": dict(Counter(str(r["state_integrity_risk"]) for r in direction)),
        "clean_edge_candidate_counts": dict(Counter(str(r["clean_edge_candidate"]) for r in fiber)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in direction)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in direction]),
        "mean_closure_fiber_score": mean_safe([safe_float(r["closure_fiber_score"]) for r in fiber]),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], direction: list[dict[str, Any]], fiber: list[dict[str, Any]], controls: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase271_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase271_{model}_direction_control_rows.jsonl", direction)
    write_jsonl(out_dir / f"phase271_{model}_closure_fiber_rows.jsonl", fiber)
    write_jsonl(out_dir / f"phase271_{model}_control_rows.jsonl", controls)
    write_jsonl(out_dir / f"phase271_{model}_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / f"phase271_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase271_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase271_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase271_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase271_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    direction: list[dict[str, Any]] = []
    fiber: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        direction.extend(read_jsonl(out_dir / f"phase271_{model}_direction_control_rows.jsonl"))
        fiber.extend(read_jsonl(out_dir / f"phase271_{model}_closure_fiber_rows.jsonl"))
        controls.extend(read_jsonl(out_dir / f"phase271_{model}_control_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase271_{model}_rollout_effect_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase271_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase271_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase271_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase271_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.91,
        "physical_path_atlas": 0.43,
        "multi_family_case_bank": 0.46,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.40,
        "path_cluster_mining": 0.19,
        "trace_signature_validation": 0.56,
        "readout_competition_trace": 0.82,
        "component_path_atlas": 0.27,
        "closure_fiber_quality_control": 0.18,
        "stepwise_rollout_trace": 0.46,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.70,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP writer direction and closure-fiber control audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "direction_control_rows": len(direction),
        "closure_fiber_rows": len(fiber),
        "control_rows": len(controls),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in direction)),
        "direction_effect_supported_counts": dict(Counter(str(r["direction_effect_supported"]) for r in direction)),
        "state_integrity_risk_counts": dict(Counter(str(r["state_integrity_risk"]) for r in direction)),
        "clean_edge_candidate_counts": dict(Counter(str(r["clean_edge_candidate"]) for r in fiber)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in direction)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in direction]),
        "mean_closure_fiber_score": mean_safe([safe_float(r["closure_fiber_score"]) for r in fiber]),
        "progress": progress,
    }
    write_json(out_dir / "phase271_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase271_direction_control_rows.jsonl", direction)
    write_jsonl(out_dir / "phase271_closure_fiber_rows.jsonl", fiber)
    write_jsonl(out_dir / "phase271_control_rows.jsonl", controls)
    write_jsonl(out_dir / "phase271_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase271_observations.jsonl", observations)
    write_jsonl(out_dir / "phase271_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase271_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase271_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase271_direction_control_rows.jsonl", direction)
    write_jsonl(ATLAS_ROOT / "phase271_closure_fiber_rows.jsonl", fiber)
    write_jsonl(ATLAS_ROOT / "phase271_control_rows.jsonl", controls)
    write_jsonl(ATLAS_ROOT / "phase271_rollout_effect_rows.jsonl", rollout)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase271", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase271 MLP Writer Direction And Closure-Fiber Control Audit",
        "",
        f"- status: {payload['status']}",
        f"- direction_control_rows: {payload['direction_control_rows']}",
        f"- closure_fiber_rows: {payload['closure_fiber_rows']}",
        f"- control_rows: {payload['control_rows']}",
        f"- rollout_effect_rows: {payload['rollout_effect_rows']}",
        f"- patch_counts: {json.dumps(payload['patch_counts'], ensure_ascii=False)}",
        f"- direction_effect_supported_counts: {json.dumps(payload['direction_effect_supported_counts'], ensure_ascii=False)}",
        f"- state_integrity_risk_counts: {json.dumps(payload['state_integrity_risk_counts'], ensure_ascii=False)}",
        f"- clean_edge_candidate_counts: {json.dumps(payload['clean_edge_candidate_counts'], ensure_ascii=False)}",
        f"- mean_delta_continue_stop_margin: {payload['mean_delta_continue_stop_margin']}",
        f"- mean_closure_fiber_score: {payload['mean_closure_fiber_score']}",
        "",
        "Note: This imports GLM5-style blocker, boundary, and side-effect controls as atlas fields. It is not closure.",
    ]
    (out_dir / "phase271_direction_closure_fiber_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=6)
    parser.add_argument("--rollout-tokens", type=int, default=6)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args)
        return
    for model in MODELS:
        args.model = model
        evaluate_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
