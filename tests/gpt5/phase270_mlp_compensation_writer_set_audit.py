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
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_mlp  # noqa: E402


PHASE = 270
SOURCE_PHASE = 269
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE268_ROOT = Path("tests/result/phase268_attention_mlp_continuation_path_attribution/attention_mlp_continuation_path_attribution")
RESULT_ROOT = Path("tests/result/phase270_mlp_compensation_writer_set_audit")
ROUND_DEFAULT = "mlp_compensation_writer_set_audit"
WINDOW_RADIUS = 2


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


def replace_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple) and output:
        return (tensor, *output[1:])
    return output


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def select_cases(model: str, cases_per_model: int) -> list[dict[str, Any]]:
    phase269 = read_jsonl(
        RESULT_ROOT.parent
        / "phase269_mlp_continuation_writer_necessity_audit"
        / "mlp_continuation_writer_necessity_audit"
        / f"phase269_{model}_mlp_necessity_rows.jsonl"
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in phase269:
        case_id = str(row.get("case_id"))
        if case_id in seen:
            continue
        seen.add(case_id)
        selected.append(
            {
                "model": model,
                "case_id": case_id,
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row.get("variant_id"),
                "path_schema_id": row.get("path_schema_id"),
                "strongest_mlp_layer": row.get("strongest_mlp_layer_phase268"),
                "strongest_mlp_delta": row.get("strongest_mlp_delta_phase268"),
                "top_continue_channel_phase267": row.get("top_continue_channel_phase268"),
            }
        )
        if len(selected) >= cases_per_model:
            return selected

    rows = [r for r in read_jsonl(PHASE268_ROOT / "phase268_component_summary_rows.jsonl") if r.get("model") == model]
    rows.sort(key=lambda r: (-safe_float(r.get("sum_positive_mlp_delta")), -safe_float(r.get("final_continue_stop_margin"))))
    return rows[:cases_per_model]


def score_logits(tokenizer: Any, logits: torch.Tensor, aliases: list[str]) -> dict[str, Any]:
    stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
    cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
    scores = {**p262.score_channels(logits.detach().float().cpu(), stop_ids, cont_ids), **p239.readout_metrics(tokenizer, logits.detach().float().cpu(), aliases)}
    target = safe_float(scores.get("target_logit"), -1e30)
    stop = safe_float(scores.get("r_stop"), -1e30)
    cont = safe_float(scores.get("r_continue"), -1e30)
    winner_value = max(stop, cont, target)
    if winner_value == target:
        winner = "target"
    elif winner_value == cont:
        winner = "continue"
    else:
        winner = "stop"
    scores["tri_winner"] = winner
    scores["continue_stop_margin"] = cont - stop
    return scores


def patch_tensor(output: Any, *, scale: float | None = None, random_seed: int | None = None) -> Any:
    y = extract_tensor(output)
    if y is None:
        return output
    patched = y.clone()
    if patched.ndim == 3:
        target = patched[:, -1, :]
        if random_seed is None:
            patched[:, -1, :] = target * float(scale)
        else:
            patched[:, -1, :] = random_same_norm(target, random_seed)
    elif patched.ndim == 2:
        target = patched[-1:, :]
        if random_seed is None:
            patched[-1:, :] = target * float(scale)
        else:
            patched[-1:, :] = random_same_norm(target, random_seed)
    elif patched.ndim == 1:
        target = patched.unsqueeze(0)
        if random_seed is None:
            patched = patched * float(scale)
        else:
            patched = random_same_norm(target, random_seed).squeeze(0)
    return replace_tensor(output, patched)


def random_same_norm(target: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=target.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(target.shape, generator=gen, device=target.device, dtype=target.dtype)
    target_norm = torch.linalg.vector_norm(target.float(), dim=-1, keepdim=True).clamp_min(1e-6).to(target.dtype)
    noise_norm = torch.linalg.vector_norm(noise.float(), dim=-1, keepdim=True).clamp_min(1e-6).to(target.dtype)
    return noise * (target_norm / noise_norm)


def install_component_hooks(model_obj: Any, specs: list[dict[str, Any]]) -> list[Any]:
    layers = get_layers(model_obj)
    handles: list[Any] = []
    for spec in specs:
        layer_idx = int(spec["layer"])
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        module = get_mlp(layers[layer_idx]) if spec["component"] == "mlp" else get_attn(layers[layer_idx])
        if module is None:
            continue

        def hook(_module, _inputs, output, spec=spec):
            return patch_tensor(
                output,
                scale=spec.get("scale"),
                random_seed=spec.get("random_seed"),
            )

        handles.append(module.register_forward_hook(hook))
    return handles


def with_hooks_logits(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, specs: list[dict[str, Any]]) -> torch.Tensor:
    handles = install_component_hooks(model_obj, specs)
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
    handles = install_component_hooks(model_obj, specs)
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


def make_window(center: int, n_layers: int, radius: int = WINDOW_RADIUS) -> list[int]:
    return [i for i in range(center - radius, center + radius + 1) if 0 <= i < n_layers]


def patch_specs(patch_type: str, center: int, n_layers: int, model: str, case_id: str) -> list[dict[str, Any]]:
    window = make_window(center, n_layers)
    if patch_type == "single_mlp_zero":
        return [{"layer": center, "component": "mlp", "scale": 0.0}]
    if patch_type == "window_mlp_zero":
        return [{"layer": layer, "component": "mlp", "scale": 0.0} for layer in window]
    if patch_type == "window_mlp_half":
        return [{"layer": layer, "component": "mlp", "scale": 0.5} for layer in window]
    if patch_type == "attn_mlp_window_zero":
        return [
            {"layer": layer, "component": component, "scale": 0.0}
            for layer in window
            for component in ["attn", "mlp"]
        ]
    if patch_type == "random_same_norm_control":
        return [
            {
                "layer": center,
                "component": "mlp",
                "random_seed": stable_seed(f"phase270:{model}:{case_id}:L{center}:random_same_norm_control"),
            }
        ]
    raise ValueError(f"unknown patch_type {patch_type}")


PATCH_TYPES = [
    "single_mlp_zero",
    "window_mlp_zero",
    "window_mlp_half",
    "attn_mlp_window_zero",
    "random_same_norm_control",
]


def row_base(case: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase270",
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
    compensation_rows: list[dict[str, Any]] = []
    writer_rows: list[dict[str, Any]] = []
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
        for idx, case in enumerate(selected, start=1):
            source = case_bank.get(str(case["case_id"]))
            if not source:
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": "Phase270", "missing_id": f"phase270:missing:{args.model}:{case['case_id']}", "model": args.model, "case_id": case["case_id"], "reason": "case not found"})
                continue
            center = int(case["strongest_mlp_layer"])
            window = make_window(center, n_layers)
            base = row_base(case, source)
            prompt = str(source["prompt"])
            aliases = [str(x) for x in source.get("target_aliases") or [source.get("target", "")]]
            try:
                base_logits = with_hooks_logits(model_obj, tokenizer, device, prompt, [])
                base_scores = score_logits(tokenizer, base_logits, aliases)
                base_text, base_new_tokens = with_hooks_generate(model_obj, tokenizer, device, prompt, [], int(args.rollout_tokens))
                per_case: dict[str, dict[str, Any]] = {}
                for patch_type in PATCH_TYPES:
                    specs = patch_specs(patch_type, center, n_layers, args.model, str(case["case_id"]))
                    patched_logits = with_hooks_logits(model_obj, tokenizer, device, prompt, specs)
                    patched_scores = score_logits(tokenizer, patched_logits, aliases)
                    patched_text, patched_new_tokens = with_hooks_generate(model_obj, tokenizer, device, prompt, specs, int(args.rollout_tokens))
                    delta_continue_stop = safe_float(patched_scores.get("continue_stop_margin")) - safe_float(base_scores.get("continue_stop_margin"))
                    delta_target = safe_float(patched_scores.get("target_logit")) - safe_float(base_scores.get("target_logit"))
                    row = {
                        **base,
                        "compensation_id": f"phase270:compensation:{args.model}:{case['case_id']}:L{center}:{patch_type}",
                        "patch_type": patch_type,
                        "center_layer": center,
                        "window_layers": window,
                        "patched_components": sorted({str(s["component"]) for s in specs}),
                        "patched_component_count": len(specs),
                        "base_continue_stop_margin": round(safe_float(base_scores.get("continue_stop_margin")), 6),
                        "patched_continue_stop_margin": round(safe_float(patched_scores.get("continue_stop_margin")), 6),
                        "delta_continue_stop_margin": round(delta_continue_stop, 6),
                        "base_winner": base_scores.get("tri_winner"),
                        "patched_winner": patched_scores.get("tri_winner"),
                        "winner_changed": base_scores.get("tri_winner") != patched_scores.get("tri_winner"),
                        "base_target_logit": base_scores.get("target_logit"),
                        "patched_target_logit": patched_scores.get("target_logit"),
                        "delta_target_logit": round(delta_target, 6),
                        "effect_supported": bool(delta_continue_stop < -1.0 or base_scores.get("tri_winner") != patched_scores.get("tri_winner")),
                        "reverse_effect": bool(delta_continue_stop > 1.0),
                    }
                    compensation_rows.append(row)
                    per_case[patch_type] = row
                    if patch_type == "random_same_norm_control":
                        control_rows.append({**row, "control_id": row["compensation_id"].replace(":compensation:", ":control:"), "control_type": "random_same_norm_last_token_mlp"})
                    rollout_rows.append(
                        {
                            **base,
                            "rollout_effect_id": row["compensation_id"].replace(":compensation:", ":rollout:"),
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
                            "phase_id": "Phase270",
                            "created_at": utc_now(),
                            "observation_id": row["compensation_id"].replace(":compensation:", ":obs:"),
                            "case_id": case["case_id"],
                            "model": args.model,
                            "family_id": case["family_id"],
                            "level": "mlp_compensation_writer_set_audit",
                            "component": f"{patch_type}:L{center}:window{window}",
                            "metric_name": "delta_continue_stop_margin",
                            "metric_value": row["delta_continue_stop_margin"],
                            "metric_unit": "logit_margin",
                            "winner": row["patched_winner"],
                        }
                    )
                single = per_case.get("single_mlp_zero", {})
                window_zero = per_case.get("window_mlp_zero", {})
                combined = per_case.get("attn_mlp_window_zero", {})
                control = per_case.get("random_same_norm_control", {})
                single_delta = safe_float(single.get("delta_continue_stop_margin"))
                window_delta = safe_float(window_zero.get("delta_continue_stop_margin"))
                combined_delta = safe_float(combined.get("delta_continue_stop_margin"))
                control_delta = safe_float(control.get("delta_continue_stop_margin"))
                writer_set_supported = bool(
                    window_delta < single_delta - 1.0
                    or (not single.get("winner_changed") and bool(window_zero.get("winner_changed")))
                )
                compensation_suspected = bool(single_delta > 1.0 or (single_delta > -1.0 and window_delta < single_delta - 1.0))
                writer = {
                    **base,
                    "writer_set_id": f"phase270:writer_set:{args.model}:{case['case_id']}:L{center}",
                    "center_layer": center,
                    "window_layers": window,
                    "single_delta_continue_stop_margin": round(single_delta, 6),
                    "window_delta_continue_stop_margin": round(window_delta, 6),
                    "combined_attn_mlp_delta_continue_stop_margin": round(combined_delta, 6),
                    "random_control_delta_continue_stop_margin": round(control_delta, 6),
                    "window_minus_single_delta": round(window_delta - single_delta, 6),
                    "combined_minus_window_delta": round(combined_delta - window_delta, 6),
                    "writer_set_supported": writer_set_supported,
                    "compensation_suspected": compensation_suspected,
                    "single_winner_changed": bool(single.get("winner_changed")),
                    "window_winner_changed": bool(window_zero.get("winner_changed")),
                    "combined_winner_changed": bool(combined.get("winner_changed")),
                    "control_large_effect": bool(abs(control_delta) > 1.0),
                    "status": "small_scale_writer_set_audit_not_closure",
                }
                writer_rows.append(writer)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase270",
                        "created_at": utc_now(),
                        "missing_id": f"phase270:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case.get("family_id"),
                        "reason": repr(exc),
                    }
                )
            log(f"{args.model}: compensation/window audited {idx}/{len(selected)} cases")
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
    metrics = make_metrics(args.model, compensation_rows, writer_rows, control_rows)
    edges = make_edges(args.model, writer_rows, compensation_rows)
    payload = summarize_model(args.model, selected, compensation_rows, writer_rows, control_rows, rollout_rows, observations, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, compensation_rows, writer_rows, control_rows, rollout_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, compensation: list[dict[str, Any]], writer: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_patch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in compensation:
        by_patch[str(row["patch_type"])].append(row)
    for patch_type, vals in sorted(by_patch.items()):
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase270",
                "created_at": utc_now(),
                "metric_id": f"phase270:{model}:{patch_type}:mean_effect",
                "scope": "mlp_compensation_writer_set_audit",
                "model": model,
                "patch_type": patch_type,
                "metric_name": "mean_delta_continue_stop_margin",
                "metric_value": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in vals]),
                "effect_supported_rate": round(sum(1 for r in vals if r.get("effect_supported")) / len(vals), 6) if vals else 0.0,
                "reverse_effect_rate": round(sum(1 for r in vals if r.get("reverse_effect")) / len(vals), 6) if vals else 0.0,
                "winner_change_rate": round(sum(1 for r in vals if r.get("winner_changed")) / len(vals), 6) if vals else 0.0,
                "rows": len(vals),
            }
        )
    out.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase270",
            "created_at": utc_now(),
            "metric_id": f"phase270:{model}:writer_set_support_rate",
            "scope": "mlp_compensation_writer_set_audit",
            "model": model,
            "metric_name": "writer_set_supported_rate",
            "metric_value": round(sum(1 for r in writer if r.get("writer_set_supported")) / len(writer), 6) if writer else 0.0,
            "rows": len(writer),
        }
    )
    out.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase270",
            "created_at": utc_now(),
            "metric_id": f"phase270:{model}:control_large_effect_rate",
            "scope": "mlp_compensation_writer_set_audit",
            "model": model,
            "metric_name": "random_same_norm_large_effect_rate",
            "metric_value": round(sum(1 for r in controls if abs(safe_float(r.get("delta_continue_stop_margin"))) > 1.0) / len(controls), 6) if controls else 0.0,
            "rows": len(controls),
        }
    )
    return out


def make_edges(model: str, writer: list[dict[str, Any]], compensation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row in writer:
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase270",
                "created_at": utc_now(),
                "edge_id": f"phase270:{model}:{row['case_id']}:writer_set:L{row['center_layer']}",
                "source": f"node:{row['family_id']}",
                "target": f"node:MLP_window_{row['window_layers'][0]}_{row['window_layers'][-1]}",
                "edge_type": "cross_layer_mlp_writer_set_candidate",
                "model": model,
                "writer_set_supported": bool(row.get("writer_set_supported")),
                "compensation_suspected": bool(row.get("compensation_suspected")),
                "effect_size": row.get("window_minus_single_delta"),
                "status": "small_scale_path_audit_not_closure",
            }
        )
    grouped = Counter((r["family_id"], r["patch_type"], bool(r["effect_supported"]), bool(r["reverse_effect"])) for r in compensation)
    for (family, patch_type, supported, reverse), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase270",
                "created_at": utc_now(),
                "edge_id": f"phase270:{model}:{family}:{patch_type}:{supported}:{reverse}",
                "source": f"node:{family}",
                "target": f"node:{patch_type}",
                "edge_type": "component_window_causal_effect",
                "model": model,
                "effect_supported": supported,
                "reverse_effect": reverse,
                "effect_size": count,
                "status": "small_scale_path_audit_not_closure",
            }
        )
    return edges


def summarize_model(model: str, selected: list[dict[str, Any]], compensation: list[dict[str, Any]], writer: list[dict[str, Any]], controls: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP compensation and cross-layer writer set audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(selected),
        "compensation_rows": len(compensation),
        "writer_set_rows": len(writer),
        "control_rows": len(controls),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in compensation)),
        "effect_supported_counts": dict(Counter(str(r["effect_supported"]) for r in compensation)),
        "reverse_effect_counts": dict(Counter(str(r["reverse_effect"]) for r in compensation)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in compensation)),
        "writer_set_supported_counts": dict(Counter(str(r["writer_set_supported"]) for r in writer)),
        "compensation_suspected_counts": dict(Counter(str(r["compensation_suspected"]) for r in writer)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in compensation]),
        "mean_control_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in controls]),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], compensation: list[dict[str, Any]], writer: list[dict[str, Any]], controls: list[dict[str, Any]], rollout: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase270_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase270_{model}_compensation_rows.jsonl", compensation)
    write_jsonl(out_dir / f"phase270_{model}_writer_set_rows.jsonl", writer)
    write_jsonl(out_dir / f"phase270_{model}_control_rows.jsonl", controls)
    write_jsonl(out_dir / f"phase270_{model}_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / f"phase270_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase270_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase270_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase270_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase270_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    compensation: list[dict[str, Any]] = []
    writer: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        compensation.extend(read_jsonl(out_dir / f"phase270_{model}_compensation_rows.jsonl"))
        writer.extend(read_jsonl(out_dir / f"phase270_{model}_writer_set_rows.jsonl"))
        controls.extend(read_jsonl(out_dir / f"phase270_{model}_control_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase270_{model}_rollout_effect_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase270_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase270_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase270_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase270_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.90,
        "physical_path_atlas": 0.41,
        "multi_family_case_bank": 0.45,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.39,
        "path_cluster_mining": 0.18,
        "trace_signature_validation": 0.55,
        "readout_competition_trace": 0.80,
        "component_path_atlas": 0.24,
        "stepwise_rollout_trace": 0.45,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.70,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "MLP compensation and cross-layer writer set audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "compensation_rows": len(compensation),
        "writer_set_rows": len(writer),
        "control_rows": len(controls),
        "rollout_effect_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in compensation)),
        "effect_supported_counts": dict(Counter(str(r["effect_supported"]) for r in compensation)),
        "reverse_effect_counts": dict(Counter(str(r["reverse_effect"]) for r in compensation)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in compensation)),
        "writer_set_supported_counts": dict(Counter(str(r["writer_set_supported"]) for r in writer)),
        "compensation_suspected_counts": dict(Counter(str(r["compensation_suspected"]) for r in writer)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollout)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in compensation]),
        "mean_control_delta_continue_stop_margin": mean_safe([safe_float(r["delta_continue_stop_margin"]) for r in controls]),
        "progress": progress,
    }
    write_json(out_dir / "phase270_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase270_compensation_rows.jsonl", compensation)
    write_jsonl(out_dir / "phase270_writer_set_rows.jsonl", writer)
    write_jsonl(out_dir / "phase270_control_rows.jsonl", controls)
    write_jsonl(out_dir / "phase270_rollout_effect_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase270_observations.jsonl", observations)
    write_jsonl(out_dir / "phase270_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase270_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase270_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase270_compensation_rows.jsonl", compensation)
    write_jsonl(ATLAS_ROOT / "phase270_writer_set_rows.jsonl", writer)
    write_jsonl(ATLAS_ROOT / "phase270_control_rows.jsonl", controls)
    write_jsonl(ATLAS_ROOT / "phase270_rollout_effect_rows.jsonl", rollout)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase270", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase270 MLP Compensation Writer Set Audit",
        "",
        f"- status: {payload['status']}",
        f"- compensation_rows: {payload['compensation_rows']}",
        f"- writer_set_rows: {payload['writer_set_rows']}",
        f"- control_rows: {payload['control_rows']}",
        f"- rollout_effect_rows: {payload['rollout_effect_rows']}",
        f"- patch_counts: {json.dumps(payload['patch_counts'], ensure_ascii=False)}",
        f"- effect_supported_counts: {json.dumps(payload['effect_supported_counts'], ensure_ascii=False)}",
        f"- reverse_effect_counts: {json.dumps(payload['reverse_effect_counts'], ensure_ascii=False)}",
        f"- writer_set_supported_counts: {json.dumps(payload['writer_set_supported_counts'], ensure_ascii=False)}",
        f"- compensation_suspected_counts: {json.dumps(payload['compensation_suspected_counts'], ensure_ascii=False)}",
        f"- mean_delta_continue_stop_margin: {payload['mean_delta_continue_stop_margin']}",
        f"- mean_control_delta_continue_stop_margin: {payload['mean_control_delta_continue_stop_margin']}",
        "",
        "Note: This tests compensation and cross-layer writer-set candidates. It is not closure.",
    ]
    (out_dir / "phase270_compensation_writer_set_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=2)
    parser.add_argument("--rollout-tokens", type=int, default=8)
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
