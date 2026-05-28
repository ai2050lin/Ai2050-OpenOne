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
from phase289_contract_scan import build_pairs, mean, parse_csv, select_pairs, tokenize
from phase290_contract_break_scan import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_TYPES = ["resid_in", "resid_out", "attn_out", "mlp_out"]


def log(message: str) -> None:
    print(f"[phase294] {message}", flush=True)


def parse_layers(value: str, n_layers: int) -> list[int]:
    out: set[int] = set()
    for item in parse_csv(value):
        layer = int(item)
        out.add(max(0, min(layer, n_layers - 1)))
    return sorted(out)


def finite_float(value: float) -> float:
    return value if math.isfinite(float(value)) else float("nan")


def tensor_norm(value: torch.Tensor) -> float:
    return finite_float(float(value.detach().float().norm()))


def capture_layer_states(
    loaded: Any,
    text: str,
    target_layers: list[int],
    seq_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_pre_hook(layer_idx: int):
        def hook(_module: Any, inputs: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                captured.setdefault(layer_idx, {})["resid_in"] = inputs[0].detach().cpu().clone()
        return hook

    def make_output_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            val = output[0] if isinstance(output, tuple) else output
            captured.setdefault(layer_idx, {})[name] = val.detach().cpu().clone()
        return hook

    for layer_idx in target_layers:
        hooks.append(layers[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx)))
        hooks.append(layers[layer_idx].register_forward_hook(make_output_hook(layer_idx, "resid_out")))
        hooks.append(layers[layer_idx].self_attn.register_forward_hook(make_output_hook(layer_idx, "attn_out")))
        hooks.append(layers[layer_idx].mlp.register_forward_hook(make_output_hook(layer_idx, "mlp_out")))

    with torch.no_grad():
        loaded.model(**tokenize(loaded, text, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def baseline_logits(loaded: Any, text: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, text, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def blend_like_ref(ref: torch.Tensor, a_value: torch.Tensor, b_value: torch.Tensor, alpha: float) -> torch.Tensor:
    a = a_value.to(device=ref.device, dtype=ref.dtype)
    b = b_value.to(device=ref.device, dtype=ref.dtype)
    seq = min(a.shape[1], b.shape[1], ref.shape[1])
    value = ref.clone()
    blended = (1.0 - alpha) * a[:, :seq, :] + alpha * b[:, :seq, :]
    value[:, :seq, :] = blended
    return value


def patch_forward(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    patch_type: str,
    alpha: float,
    a_state: dict[str, torch.Tensor],
    b_state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []
    stats: dict[str, float] = {}

    def pre_hook(_module: Any, inputs: Any) -> Any:
        if not (isinstance(inputs, tuple) and inputs):
            return inputs
        ref = inputs[0]
        patched = blend_like_ref(ref, a_state["resid_in"], b_state["resid_in"], alpha)
        stats["patch_norm"] = tensor_norm(patched)
        stats["a_ref_norm"] = tensor_norm(a_state["resid_in"])
        stats["b_ref_norm"] = tensor_norm(b_state["resid_in"])
        return (patched,) + tuple(inputs[1:])

    def output_hook(key: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            ref = output[0] if isinstance(output, tuple) else output
            patched = blend_like_ref(ref, a_state[key], b_state[key], alpha)
            stats["patch_norm"] = tensor_norm(patched)
            stats["a_ref_norm"] = tensor_norm(a_state[key])
            stats["b_ref_norm"] = tensor_norm(b_state[key])
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook

    if patch_type == "resid_in":
        hooks.append(layer.register_forward_pre_hook(pre_hook))
    elif patch_type == "resid_out":
        hooks.append(layer.register_forward_hook(output_hook("resid_out")))
    elif patch_type == "attn_out":
        hooks.append(layer.self_attn.register_forward_hook(output_hook("attn_out")))
    elif patch_type == "mlp_out":
        hooks.append(layer.mlp.register_forward_hook(output_hook("mlp_out")))
    else:
        raise ValueError(f"unknown patch_type={patch_type}")

    result = None
    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        result = out.logits[0, -1, :].detach().cpu().float().clone()
        stats["finite"] = 1.0 if torch.isfinite(result).all().item() else 0.0
    finally:
        for hook in hooks:
            hook.remove()
    return result, stats


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_patch: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_subtype_patch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer_patch[(int(row["layer"]), str(row["patch_type"]))].append(row)
        by_subtype_patch[(str(row["subtype"]), str(row["patch_type"]))].append(row)

    layer_curve: dict[str, dict[str, float]] = {}
    for (layer, patch_type), items in sorted(by_layer_patch.items()):
        slot = layer_curve.setdefault(str(layer), {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_logit_delta_ratio"] = mean([float(x["logit_delta_ratio"]) for x in items])
        slot[f"{patch_type}_finite_rate"] = mean([float(x.get("finite", 1.0)) for x in items])

    subtype_curve: dict[str, dict[str, float]] = {}
    for (subtype, patch_type), items in sorted(by_subtype_patch.items()):
        slot = subtype_curve.setdefault(subtype, {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])

    best: dict[str, Any] = {}
    for patch_type in PATCH_TYPES:
        candidates = [
            (layer, vals.get(f"{patch_type}_progress", float("nan")))
            for layer, vals in layer_curve.items()
        ]
        candidates = [(layer, value) for layer, value in candidates if math.isfinite(value)]
        if candidates:
            layer, value = max(candidates, key=lambda item: item[1])
            best[patch_type] = {"layer": int(layer), "progress": value}

    return {
        "layer_curve": layer_curve,
        "subtype_curve": subtype_curve,
        "best_by_patch_type": best,
        "nonfinite_rows": sum(1 for row in rows if float(row.get("finite", 1.0)) < 0.5),
    }


def checkpoint_path(output_dir: Path, model: str, category: str, label: str) -> Path:
    return output_dir / "checkpoints" / model / f"{category}_{label}.json"


def expected_rows_per_pair(target_layers: list[int], alphas: list[float], patch_types: list[str]) -> int:
    return len(target_layers) * len(alphas) * len(patch_types)


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
        target_layers = parse_layers(args.layers, len(layers))
        patch_types = parse_csv(args.patch_types)
        alphas = [float(x) for x in parse_csv(args.alphas)]
        expected_rows = expected_rows_per_pair(target_layers, alphas, patch_types)
        completed_pairs = {
            name for name, count in Counter(str(row.get("pair")) for row in resume_rows).items()
            if name and count >= expected_rows
        }

        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(f"pairs={len(pairs)} target_layers={target_layers} patch_types={patch_types} alphas={alphas}")
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

            states_a = capture_layer_states(loaded, pair.a, target_layers, seq_len)
            states_b = capture_layer_states(loaded, pair.b, target_layers, seq_len)
            logits_a = baseline_logits(loaded, pair.a, seq_len)
            logits_b = baseline_logits(loaded, pair.b, seq_len)
            kl_ab = float(F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1), reduction="sum"))
            if kl_ab < 1e-8:
                continue

            for layer_idx in target_layers:
                needed = {"resid_in", "resid_out", "attn_out", "mlp_out"}
                if not needed.issubset(states_a.get(layer_idx, {})):
                    continue
                if not needed.issubset(states_b.get(layer_idx, {})):
                    continue
                for alpha in alphas:
                    for patch_type in patch_types:
                        patched, stats = patch_forward(
                            loaded,
                            pair.a,
                            seq_len,
                            layer_idx,
                            patch_type,
                            alpha,
                            states_a[layer_idx],
                            states_b[layer_idx],
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
                        rows.append({
                            "pair": pair.name,
                            "category": pair.category,
                            "subtype": pair.subtype,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "patch_type": patch_type,
                            "kl_ab": kl_ab,
                            **metrics,
                            **stats,
                        })

            if (pair_index + 1) % args.progress_every == 0:
                log(f"progress pairs={pair_index + 1}/{len(pairs)} rows={len(rows)} elapsed={time.time() - start:.1f}s")
                partial = {
                    "model": args.model,
                    "complete": False,
                    "num_pairs": len(pairs),
                    "num_results": len(rows),
                    "categories": sorted({item.category for item in pairs}),
                    "subtypes": sorted({item.subtype for item in pairs}),
                    "target_layers": target_layers,
                    "alphas": alphas,
                    "patch_types": patch_types,
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
            "categories": sorted({pair.category for pair in pairs}),
            "subtypes": sorted({pair.subtype for pair in pairs}),
            "target_layers": target_layers,
            "alphas": alphas,
            "patch_types": patch_types,
            "results": rows,
            "summary": summarize(rows),
        }
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(data, indent=2), encoding="utf-8")
        out_file = output_dir / f"{args.model}_phase294_dynamic_recompute_pilot.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved checkpoint={ckpt}")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase294_dynamic_recompute_pilot"))
    parser.add_argument("--categories", default="negation,logical,passive,recursive")
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--max-pairs-per-subtype", type=int, default=1)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--patch-types", default="resid_in,resid_out,attn_out,mlp_out")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--label", default="phase294")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    try:
        data = run(args)
        log(f"done rows={data['num_results']} nonfinite={data['summary']['nonfinite_rows']}")
    finally:
        if args.hard_exit_after_model:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
