from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import get_layers, load_probe_model, release_loaded
from phase289_contract_scan import (
    build_pairs,
    choose_layers,
    interp,
    mean,
    module_device_dtype,
    parse_alphas,
    parse_csv,
    select_pairs,
    tokenize,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[phase290] {msg}", flush=True)


def finite_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach().float()
    return {
        "norm": finite_float(float(t.norm())),
        "mean": finite_float(float(t.mean())),
        "std": finite_float(float(t.std())),
        "finite": 1.0 if torch.isfinite(t).all().item() else 0.0,
    }


def norm_span_violation(value: float, a_ref: float, b_ref: float, low: float, high: float) -> float:
    if not (math.isfinite(value) and math.isfinite(a_ref) and math.isfinite(b_ref)):
        return 1.0
    lower = low * max(min(a_ref, b_ref), 1e-8)
    upper = high * max(a_ref, b_ref, 1e-8)
    return 1.0 if value < lower or value > upper else 0.0


def capture_outputs(
    loaded: Any,
    text: str,
    target_layers: list[int],
    seq_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_output_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            val = output[0] if isinstance(output, tuple) else output
            captured.setdefault(layer_idx, {})[name] = val.detach().cpu().clone()
        return hook

    def make_next_hook(prev_layer_idx: int):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                captured.setdefault(prev_layer_idx, {})["next_resid_in"] = inputs[0].detach().cpu().clone()
            val = output[0] if isinstance(output, tuple) else output
            captured.setdefault(prev_layer_idx, {})["next_layer_out"] = val.detach().cpu().clone()
        return hook

    for layer_idx in target_layers:
        hooks.append(layers[layer_idx].self_attn.register_forward_hook(make_output_hook(layer_idx, "attn")))
        hooks.append(layers[layer_idx].mlp.register_forward_hook(make_output_hook(layer_idx, "mlp")))
        hooks.append(layers[layer_idx].register_forward_hook(make_output_hook(layer_idx, "resid")))
        if layer_idx + 1 < len(layers):
            hooks.append(layers[layer_idx + 1].register_forward_hook(make_next_hook(layer_idx)))

    with torch.no_grad():
        loaded.model(**tokenize(loaded, text, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def baseline_logits(loaded: Any, text: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, text, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def add_patch_stats(
    natural: dict[str, float],
    key: str,
    value: torch.Tensor,
    a_ref: torch.Tensor,
    b_ref: torch.Tensor,
    norm_low: float,
    norm_high: float,
) -> None:
    stats = tensor_stats(value)
    a_stats = tensor_stats(a_ref)
    b_stats = tensor_stats(b_ref)
    natural[f"patch_{key}_norm"] = stats["norm"]
    natural[f"patch_{key}_finite"] = stats["finite"]
    natural[f"patch_{key}_norm_ratio_to_a"] = stats["norm"] / max(a_stats["norm"], 1e-8)
    natural[f"patch_{key}_norm_ratio_to_b"] = stats["norm"] / max(b_stats["norm"], 1e-8)
    natural[f"patch_{key}_norm_illegal"] = norm_span_violation(
        stats["norm"], a_stats["norm"], b_stats["norm"], norm_low, norm_high
    )


def forward_patch(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    patch_type: str,
    alpha: float,
    a_out: dict[str, torch.Tensor],
    b_out: dict[str, torch.Tensor],
    norm_low: float,
    norm_high: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []
    natural: dict[str, float] = {}

    def patch_hook(value: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            ref = output[0] if isinstance(output, tuple) else output
            patched = ref.clone()
            seq = min(value.shape[1], patched.shape[1])
            patched[:, :seq, :] = value[:, :seq, :]
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook

    if patch_type in {"attn", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
        device, dtype = module_device_dtype(layer.self_attn.o_proj)
        if patch_type == "cross_aattn_bmlp":
            attn_alpha = 0.0
        else:
            attn_alpha = 1.0 if patch_type == "cross_battn_amlp" else alpha
        attn_value = interp(a_out["attn"], b_out["attn"], attn_alpha, device, dtype)
        hooks.append(layer.self_attn.register_forward_hook(patch_hook(attn_value)))
        add_patch_stats(natural, "attn", attn_value, a_out["attn"], b_out["attn"], norm_low, norm_high)

    if patch_type in {"mlp", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
        device, dtype = module_device_dtype(layer.mlp)
        if patch_type == "cross_battn_amlp":
            mlp_alpha = 0.0
        else:
            mlp_alpha = 1.0 if patch_type == "cross_aattn_bmlp" else alpha
        mlp_value = interp(a_out["mlp"], b_out["mlp"], mlp_alpha, device, dtype)
        hooks.append(layer.mlp.register_forward_hook(patch_hook(mlp_value)))
        add_patch_stats(natural, "mlp", mlp_value, a_out["mlp"], b_out["mlp"], norm_low, norm_high)

    if patch_type == "resid":
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        resid_value = interp(a_out["resid"], b_out["resid"], alpha, device, dtype)
        hooks.append(layer.register_forward_hook(patch_hook(resid_value)))
        add_patch_stats(natural, "resid", resid_value, a_out["resid"], b_out["resid"], norm_low, norm_high)

    if layer_idx + 1 < len(layers):
        def next_layer_hook(_module: Any, inputs: Any, output: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                next_in = inputs[0].detach().float()
                stats = tensor_stats(next_in)
                natural["next_resid_in_norm"] = stats["norm"]
                natural["next_resid_in_finite"] = stats["finite"]
                if "next_resid_in" in a_out and "next_resid_in" in b_out:
                    a_norm = tensor_stats(a_out["next_resid_in"])["norm"]
                    b_norm = tensor_stats(b_out["next_resid_in"])["norm"]
                    natural["next_resid_in_norm_ratio_to_a"] = stats["norm"] / max(a_norm, 1e-8)
                    natural["next_resid_in_norm_ratio_to_b"] = stats["norm"] / max(b_norm, 1e-8)
                    natural["next_resid_in_norm_illegal"] = norm_span_violation(
                        stats["norm"], a_norm, b_norm, norm_low, norm_high
                    )
            val = output[0] if isinstance(output, tuple) else output
            next_out = val.detach().float()
            stats = tensor_stats(next_out)
            natural["next_layer_out_norm"] = stats["norm"]
            natural["next_layer_out_finite"] = stats["finite"]
            if "next_layer_out" in a_out and "next_layer_out" in b_out:
                a_norm = tensor_stats(a_out["next_layer_out"])["norm"]
                b_norm = tensor_stats(b_out["next_layer_out"])["norm"]
                natural["next_layer_out_norm_ratio_to_a"] = stats["norm"] / max(a_norm, 1e-8)
                natural["next_layer_out_norm_ratio_to_b"] = stats["norm"] / max(b_norm, 1e-8)
                natural["next_layer_out_norm_illegal"] = norm_span_violation(
                    stats["norm"], a_norm, b_norm, norm_low, norm_high
                )

        hooks.append(layers[layer_idx + 1].register_forward_hook(next_layer_hook))

    result = None
    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        result = out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()
    return result, natural


def compute_metrics(
    patched: torch.Tensor | None,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    kl_ab: float,
) -> dict[str, float] | None:
    if patched is None:
        return None
    if not torch.isfinite(patched).all():
        return {
            "kl_ratio": float("nan"),
            "progress": float("nan"),
            "cos_dir": float("nan"),
            "logit_delta_ratio": float("nan"),
            "logits_finite": 0.0,
            "finite": 0.0,
        }
    kl_p = float(F.kl_div(F.log_softmax(patched, dim=-1), F.softmax(logits_b, dim=-1), reduction="sum"))
    delta_b = logits_b - logits_a
    delta_p = patched - logits_a
    norm_b = float(delta_b.norm())
    norm_p = float(delta_p.norm())
    if norm_b <= 1e-8 or norm_p <= 1e-8:
        cos_dir = 0.0
        progress = 0.0
        logit_delta_ratio = 0.0
    else:
        cos_dir = float(torch.dot(delta_p, delta_b) / (delta_p.norm() * delta_b.norm()))
        logit_delta_ratio = norm_p / norm_b
        progress = cos_dir * min(logit_delta_ratio, 2.0)
    return {
        "kl_ratio": kl_p / max(kl_ab, 1e-6),
        "progress": progress,
        "cos_dir": cos_dir,
        "logit_delta_ratio": logit_delta_ratio,
        "logits_finite": 1.0,
        "finite": 1.0,
    }


def illegal_flag(row: dict[str, Any]) -> bool:
    for key, val in row.items():
        if key.endswith("_norm_illegal") and float(val) >= 1.0:
            return True
    return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    subtype_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    alpha_groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            groups[(int(row["layer"]), str(row["patch_type"]))].append(row)
            subtype_groups[(str(row["subtype"]), str(row["patch_type"]))].append(row)
        alpha_groups[(int(row["layer"]), float(row["alpha"]), str(row["patch_type"]))].append(row)

    layer_curve: dict[str, dict[str, float]] = {}
    for (layer, patch_type), items in sorted(groups.items()):
        slot = layer_curve.setdefault(str(layer), {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_logit_delta_ratio"] = mean([float(x["logit_delta_ratio"]) for x in items])
        slot[f"{patch_type}_nonfinite_rate"] = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
        slot[f"{patch_type}_norm_illegal_rate"] = mean([1.0 if illegal_flag(x) else 0.0 for x in items])

    alpha_curve: dict[str, dict[str, dict[str, float]]] = {}
    for (layer, alpha, patch_type), items in sorted(alpha_groups.items()):
        slot = alpha_curve.setdefault(str(layer), {}).setdefault(str(alpha), {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_nonfinite_rate"] = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
        slot[f"{patch_type}_norm_illegal_rate"] = mean([1.0 if illegal_flag(x) else 0.0 for x in items])

    subtype_summary: dict[str, dict[str, float]] = {}
    for (subtype, patch_type), items in sorted(subtype_groups.items()):
        slot = subtype_summary.setdefault(subtype, {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_nonfinite_rate"] = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
        slot[f"{patch_type}_norm_illegal_rate"] = mean([1.0 if illegal_flag(x) else 0.0 for x in items])

    nonfinite_rows = [row for row in rows if not bool(row.get("finite", 1.0))]
    norm_illegal_rows = [row for row in rows if illegal_flag(row)]

    nonfinite_by_layer = Counter(str(row["layer"]) for row in nonfinite_rows)
    nonfinite_by_patch = Counter(str(row["patch_type"]) for row in nonfinite_rows)
    nonfinite_by_subtype = Counter(str(row["subtype"]) for row in nonfinite_rows)
    nonfinite_by_alpha = Counter(str(row["alpha"]) for row in nonfinite_rows)

    contract_events: list[dict[str, Any]] = []
    for (layer, patch_type), items in sorted(groups.items()):
        total = len(items)
        if not total:
            continue
        nonfinite_rate = sum(1 for item in items if not bool(item.get("finite", 1.0))) / total
        norm_illegal_rate = sum(1 for item in items if illegal_flag(item)) / total
        if nonfinite_rate > 0.0:
            contract_events.append({
                "level": "numeric_illegal",
                "layer": layer,
                "patch_type": patch_type,
                "nonfinite_rate": nonfinite_rate,
                "norm_illegal_rate": norm_illegal_rate,
                "score": 10.0 * nonfinite_rate + 2.0 * norm_illegal_rate,
            })
        elif norm_illegal_rate >= 0.2:
            contract_events.append({
                "level": "norm_illegal",
                "layer": layer,
                "patch_type": patch_type,
                "nonfinite_rate": nonfinite_rate,
                "norm_illegal_rate": norm_illegal_rate,
                "score": 2.0 * norm_illegal_rate,
            })

    for layer, vals in layer_curve.items():
        both_kl = vals.get("both_kl_ratio", float("nan"))
        both_progress = vals.get("both_progress", float("nan"))
        for cross_name in ("cross_battn_amlp", "cross_aattn_bmlp"):
            cross_kl = vals.get(f"{cross_name}_kl_ratio", float("nan"))
            cross_progress = vals.get(f"{cross_name}_progress", float("nan"))
            cross_delta = vals.get(f"{cross_name}_logit_delta_ratio", float("nan"))
            if not (math.isfinite(both_kl) and math.isfinite(both_progress) and math.isfinite(cross_kl) and math.isfinite(cross_progress)):
                continue
            ratio = cross_kl / max(both_kl, 1e-6)
            progress_drop = both_progress - cross_progress
            if ratio >= 2.0 and cross_kl >= 0.5 and progress_drop >= 0.25 and cross_delta >= 0.15:
                contract_events.append({
                    "level": "functional_incompatible",
                    "layer": int(layer),
                    "patch_type": cross_name,
                    "cross_kl_ratio": cross_kl,
                    "both_kl_ratio": both_kl,
                    "kl_ratio_vs_both": ratio,
                    "cross_progress": cross_progress,
                    "both_progress": both_progress,
                    "progress_drop": progress_drop,
                    "cross_logit_delta_ratio": cross_delta,
                    "score": progress_drop + min(ratio, 10.0) / 10.0,
                })

    best_layer = None
    best_value = -math.inf
    for layer, vals in layer_curve.items():
        val = vals.get("both_progress", float("nan"))
        if math.isfinite(val) and val > best_value:
            best_value = val
            best_layer = int(layer)

    return {
        "layer_curve": layer_curve,
        "alpha_curve": alpha_curve,
        "subtype_summary": subtype_summary,
        "contract_events": sorted(contract_events, key=lambda x: float(x.get("score", 0.0)), reverse=True),
        "contract_broken_layers": sorted({int(event["layer"]) for event in contract_events}),
        "best_layer_by_both_progress": best_layer,
        "nonfinite_rows": len(nonfinite_rows),
        "norm_illegal_rows": len(norm_illegal_rows),
        "nonfinite_by_layer": dict(nonfinite_by_layer),
        "nonfinite_by_patch": dict(nonfinite_by_patch),
        "nonfinite_by_subtype": dict(nonfinite_by_subtype),
        "nonfinite_by_alpha": dict(nonfinite_by_alpha),
    }


def checkpoint_path(output_dir: Path, model: str, category: str, label: str) -> Path:
    return output_dir / "checkpoints" / model / f"{category}_{label}.json"


def should_run_patch(patch_type: str, alpha: float, dedupe_cross: bool) -> bool:
    if not dedupe_cross:
        return True
    if patch_type.startswith("cross_"):
        return abs(alpha - 1.0) < 1e-9
    return True


def expected_rows_per_pair(target_layers: list[int], alphas: list[float], patch_types: list[str], dedupe_cross: bool) -> int:
    per_layer = 0
    for alpha in alphas:
        for patch_type in patch_types:
            if should_run_patch(patch_type, alpha, dedupe_cross):
                per_layer += 1
    return len(target_layers) * per_layer


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = set(parse_csv(args.categories)) if args.categories else None
    subtypes = set(parse_csv(args.subtypes)) if args.subtypes else None
    pairs = select_pairs(build_pairs(), categories, subtypes, args.max_pairs_per_subtype)
    if not pairs:
        raise SystemExit("No pairs selected")

    category_label = "-".join(sorted({pair.category for pair in pairs}))
    ckpt = checkpoint_path(output_dir, args.model, category_label, args.label)
    resume_rows: list[dict[str, Any]] = []
    if args.resume and ckpt.exists():
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if data.get("complete"):
            log(f"checkpoint already complete: {ckpt}")
            return data
        resume_rows = list(data.get("results", []))

    loaded = None
    try:
        loaded = load_probe_model(args.model)
        layers = get_layers(loaded.model)
        target_layers = choose_layers(len(layers), args.layer_stride, args.layers)
        alphas = parse_alphas(args.alphas)
        patch_types = parse_csv(args.patch_types)
        expected_rows = expected_rows_per_pair(target_layers, alphas, patch_types, args.dedupe_cross)
        existing_counts = Counter(str(row.get("pair")) for row in resume_rows)
        completed_pairs = {
            name for name, count in existing_counts.items()
            if name and count >= expected_rows
        }

        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(f"pairs={len(pairs)} categories={sorted({p.category for p in pairs})} subtypes={sorted({p.subtype for p in pairs})}")
        log(f"target_layers={target_layers}")
        log(f"alphas={alphas} patch_types={patch_types} dedupe_cross={args.dedupe_cross}")
        if resume_rows:
            log(f"resume rows={len(resume_rows)} expected_rows_per_pair={expected_rows} completed_pairs={len(completed_pairs)}")

        rows: list[dict[str, Any]] = resume_rows
        start = time.time()
        for pair_index, pair in enumerate(pairs):
            if pair.name in completed_pairs:
                continue
            toks_a = len(loaded.tokenizer.encode(pair.a, add_special_tokens=True))
            toks_b = len(loaded.tokenizer.encode(pair.b, add_special_tokens=True))
            seq_len = min(max(toks_a, toks_b), args.max_seq_len)

            out_a = capture_outputs(loaded, pair.a, target_layers, seq_len)
            out_b = capture_outputs(loaded, pair.b, target_layers, seq_len)
            logits_a = baseline_logits(loaded, pair.a, seq_len)
            logits_b = baseline_logits(loaded, pair.b, seq_len)
            kl_ab = float(F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1), reduction="sum"))
            if kl_ab < 1e-8:
                continue

            for layer_idx in target_layers:
                if layer_idx not in out_a or layer_idx not in out_b:
                    continue
                if not {"attn", "mlp", "resid"}.issubset(out_a[layer_idx]):
                    continue
                if not {"attn", "mlp", "resid"}.issubset(out_b[layer_idx]):
                    continue
                for alpha in alphas:
                    for patch_type in patch_types:
                        if not should_run_patch(patch_type, alpha, args.dedupe_cross):
                            continue
                        patched, natural = forward_patch(
                            loaded,
                            pair.a,
                            seq_len,
                            layer_idx,
                            patch_type,
                            alpha,
                            out_a[layer_idx],
                            out_b[layer_idx],
                            args.norm_low,
                            args.norm_high,
                        )
                        metrics = compute_metrics(patched, logits_a, logits_b, kl_ab)
                        if metrics is None:
                            metrics = {
                                "kl_ratio": float("nan"),
                                "progress": float("nan"),
                                "cos_dir": float("nan"),
                                "logit_delta_ratio": float("nan"),
                                "logits_finite": 0.0,
                                "finite": 0.0,
                            }
                        row = {
                            "pair": pair.name,
                            "category": pair.category,
                            "subtype": pair.subtype,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "patch_type": patch_type,
                            "kl_ab": kl_ab,
                            **metrics,
                            **natural,
                        }
                        row["norm_illegal"] = 1.0 if illegal_flag(row) else 0.0
                        rows.append(row)

            if (pair_index + 1) % args.progress_every == 0:
                elapsed = time.time() - start
                log(f"progress pairs={pair_index + 1}/{len(pairs)} rows={len(rows)} elapsed={elapsed:.1f}s")
                partial = {
                    "model": args.model,
                    "complete": False,
                    "num_pairs": len(pairs),
                    "num_results": len(rows),
                    "target_layers": target_layers,
                    "alphas": alphas,
                    "patch_types": patch_types,
                    "dedupe_cross": args.dedupe_cross,
                    "results": rows,
                    "summary": summarize(rows),
                }
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.write_text(json.dumps(partial, indent=2), encoding="utf-8")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "num_pairs": len(pairs),
            "num_results": len(rows),
            "categories": sorted({p.category for p in pairs}),
            "subtypes": sorted({p.subtype for p in pairs}),
            "target_layers": target_layers,
            "alphas": alphas,
            "patch_types": patch_types,
            "dedupe_cross": args.dedupe_cross,
            "norm_low": args.norm_low,
            "norm_high": args.norm_high,
            "results": rows,
            "summary": summarize(rows),
        }
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(data, indent=2), encoding="utf-8")
        (output_dir / f"{args.model}_phase290_contract_break_scan.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log(f"saved checkpoint={ckpt}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase290_contract_break_scan"))
    parser.add_argument("--categories", default="negation,logical,passive,recursive")
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--max-pairs-per-subtype", type=int, default=4)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--layers", default="")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--patch-types", default="attn,mlp,both,resid,cross_battn_amlp,cross_aattn_bmlp")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--label", default="phase290")
    parser.add_argument("--norm-low", type=float, default=0.5)
    parser.add_argument("--norm-high", type=float, default=2.0)
    parser.add_argument("--dedupe-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    data = run(args)
    summary = data.get("summary", {})
    log(
        "done "
        f"model={data.get('model')} pairs={data.get('num_pairs')} rows={data.get('num_results')} "
        f"best_layer={summary.get('best_layer_by_both_progress')} "
        f"broken_layers={summary.get('contract_broken_layers')} "
        f"nonfinite={summary.get('nonfinite_rows')} norm_illegal={summary.get('norm_illegal_rows')}"
    )
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
