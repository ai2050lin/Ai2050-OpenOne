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
    interp,
    mean,
    module_device_dtype,
    parse_alphas,
    parse_csv,
    select_pairs,
    tokenize,
)
from phase290_contract_break_scan import compute_metrics, illegal_flag, norm_span_violation, tensor_stats


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[phase291] {msg}", flush=True)


def parse_blocks(value: str, n_layers: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for item in parse_csv(value):
        if "-" in item:
            a_s, b_s = item.split("-", 1)
            start, end = int(a_s), int(b_s)
        else:
            start = end = int(item)
        start = max(0, min(start, n_layers - 1))
        end = max(0, min(end, n_layers - 1))
        if end < start:
            start, end = end, start
        blocks.append((start, end))
    return sorted(set(blocks))


def block_label(block: tuple[int, int]) -> str:
    start, end = block
    return f"L{start}" if start == end else f"L{start}-L{end}"


def block_layers(blocks: list[tuple[int, int]]) -> list[int]:
    out: set[int] = set()
    for start, end in blocks:
        out.update(range(start, end + 1))
    return sorted(out)


def capture_outputs(
    loaded: Any,
    text: str,
    target_layers: list[int],
    seq_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            val = output[0] if isinstance(output, tuple) else output
            captured.setdefault(layer_idx, {})[name] = val.detach().cpu().clone()
        return hook

    for layer_idx in target_layers:
        hooks.append(layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx, "attn")))
        hooks.append(layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx, "mlp")))
        hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx, "resid")))

    with torch.no_grad():
        loaded.model(**tokenize(loaded, text, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def baseline_logits(loaded: Any, text: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, text, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def add_block_patch_stats(
    natural: dict[str, float],
    layer_idx: int,
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
    prefix = f"L{layer_idx}_{key}"
    natural[f"{prefix}_norm"] = stats["norm"]
    natural[f"{prefix}_finite"] = stats["finite"]
    natural[f"{prefix}_norm_ratio_to_a"] = stats["norm"] / max(a_stats["norm"], 1e-8)
    natural[f"{prefix}_norm_ratio_to_b"] = stats["norm"] / max(b_stats["norm"], 1e-8)
    natural[f"{prefix}_norm_illegal"] = norm_span_violation(
        stats["norm"], a_stats["norm"], b_stats["norm"], norm_low, norm_high
    )


def forward_block_patch(
    loaded: Any,
    text: str,
    seq_len: int,
    block: tuple[int, int],
    patch_type: str,
    alpha: float,
    out_a: dict[int, dict[str, torch.Tensor]],
    out_b: dict[int, dict[str, torch.Tensor]],
    norm_low: float,
    norm_high: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    layers = get_layers(loaded.model)
    hooks = []
    natural: dict[str, float] = {}
    start, end = block

    def patch_hook(value: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            ref = output[0] if isinstance(output, tuple) else output
            patched = ref.clone()
            seq = min(value.shape[1], patched.shape[1])
            patched[:, :seq, :] = value[:, :seq, :]
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook

    for layer_idx in range(start, end + 1):
        layer = layers[layer_idx]
        a_layer = out_a[layer_idx]
        b_layer = out_b[layer_idx]
        if patch_type in {"attn", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
            device, dtype = module_device_dtype(layer.self_attn.o_proj)
            if patch_type == "cross_aattn_bmlp":
                attn_alpha = 0.0
            else:
                attn_alpha = 1.0 if patch_type == "cross_battn_amlp" else alpha
            attn_value = interp(a_layer["attn"], b_layer["attn"], attn_alpha, device, dtype)
            hooks.append(layer.self_attn.register_forward_hook(patch_hook(attn_value)))
            add_block_patch_stats(
                natural, layer_idx, "attn", attn_value, a_layer["attn"], b_layer["attn"], norm_low, norm_high
            )

        if patch_type in {"mlp", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
            device, dtype = module_device_dtype(layer.mlp)
            if patch_type == "cross_battn_amlp":
                mlp_alpha = 0.0
            else:
                mlp_alpha = 1.0 if patch_type == "cross_aattn_bmlp" else alpha
            mlp_value = interp(a_layer["mlp"], b_layer["mlp"], mlp_alpha, device, dtype)
            hooks.append(layer.mlp.register_forward_hook(patch_hook(mlp_value)))
            add_block_patch_stats(
                natural, layer_idx, "mlp", mlp_value, a_layer["mlp"], b_layer["mlp"], norm_low, norm_high
            )

        if patch_type == "resid":
            device = next(layer.parameters()).device
            dtype = next(layer.parameters()).dtype
            resid_value = interp(a_layer["resid"], b_layer["resid"], alpha, device, dtype)
            hooks.append(layer.register_forward_hook(patch_hook(resid_value)))
            add_block_patch_stats(
                natural, layer_idx, "resid", resid_value, a_layer["resid"], b_layer["resid"], norm_low, norm_high
            )

    result = None
    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        result = out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()
    return result, natural


def should_run_patch(patch_type: str, alpha: float, dedupe_cross: bool) -> bool:
    if not dedupe_cross:
        return True
    if patch_type.startswith("cross_"):
        return abs(alpha - 1.0) < 1e-9
    return True


def expected_rows_per_pair(blocks: list[tuple[int, int]], alphas: list[float], patch_types: list[str], dedupe_cross: bool) -> int:
    per_block = 0
    for alpha in alphas:
        for patch_type in patch_types:
            if should_run_patch(patch_type, alpha, dedupe_cross):
                per_block += 1
    return len(blocks) * per_block


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_block_patch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_block_alpha_patch: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    by_subtype_patch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block_alpha_patch[(str(row["block"]), float(row["alpha"]), str(row["patch_type"]))].append(row)
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_block_patch[(str(row["block"]), str(row["patch_type"]))].append(row)
            by_subtype_patch[(str(row["subtype"]), str(row["patch_type"]))].append(row)

    block_curve: dict[str, dict[str, float]] = {}
    for (block, patch_type), items in sorted(by_block_patch.items()):
        slot = block_curve.setdefault(block, {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_logit_delta_ratio"] = mean([float(x["logit_delta_ratio"]) for x in items])
        slot[f"{patch_type}_nonfinite_rate"] = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
        slot[f"{patch_type}_norm_illegal_rate"] = mean([1.0 if illegal_flag(x) else 0.0 for x in items])

    alpha_curve: dict[str, dict[str, dict[str, float]]] = {}
    for (block, alpha, patch_type), items in sorted(by_block_alpha_patch.items()):
        slot = alpha_curve.setdefault(block, {}).setdefault(str(alpha), {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])
        slot[f"{patch_type}_nonfinite_rate"] = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
        slot[f"{patch_type}_norm_illegal_rate"] = mean([1.0 if illegal_flag(x) else 0.0 for x in items])

    subtype_signature: dict[str, dict[str, float]] = {}
    for (subtype, patch_type), items in sorted(by_subtype_patch.items()):
        slot = subtype_signature.setdefault(subtype, {})
        slot[f"{patch_type}_progress"] = mean([float(x["progress"]) for x in items])
        slot[f"{patch_type}_kl_ratio"] = mean([float(x["kl_ratio"]) for x in items])

    events: list[dict[str, Any]] = []
    for (block, patch_type), items in sorted(by_block_patch.items()):
        total = len(items)
        if not total:
            continue
        nonfinite_rate = sum(1 for x in items if not bool(x.get("finite", 1.0))) / total
        norm_illegal_rate = sum(1 for x in items if illegal_flag(x)) / total
        if nonfinite_rate > 0.0:
            events.append({
                "level": "numeric_illegal",
                "block": block,
                "patch_type": patch_type,
                "nonfinite_rate": nonfinite_rate,
                "norm_illegal_rate": norm_illegal_rate,
                "score": 10.0 * nonfinite_rate + 2.0 * norm_illegal_rate,
            })
        elif norm_illegal_rate >= 0.2:
            events.append({
                "level": "norm_illegal",
                "block": block,
                "patch_type": patch_type,
                "nonfinite_rate": nonfinite_rate,
                "norm_illegal_rate": norm_illegal_rate,
                "score": 2.0 * norm_illegal_rate,
            })

    for block, vals in block_curve.items():
        both_kl = vals.get("both_kl_ratio", float("nan"))
        both_progress = vals.get("both_progress", float("nan"))
        for cross_name in ("cross_battn_amlp", "cross_aattn_bmlp"):
            cross_kl = vals.get(f"{cross_name}_kl_ratio", float("nan"))
            cross_progress = vals.get(f"{cross_name}_progress", float("nan"))
            cross_delta = vals.get(f"{cross_name}_logit_delta_ratio", float("nan"))
            if not (math.isfinite(both_progress) and math.isfinite(cross_progress)):
                continue
            progress_drop = both_progress - cross_progress
            if math.isfinite(both_kl) and math.isfinite(cross_kl):
                ratio = cross_kl / max(both_kl, 1e-6)
            else:
                ratio = float("nan")
            if (
                math.isfinite(ratio)
                and ratio >= 2.0
                and cross_kl >= 0.5
                and progress_drop >= 0.25
                and cross_delta >= 0.15
            ):
                events.append({
                    "level": "functional_kl_incompatible",
                    "block": block,
                    "patch_type": cross_name,
                    "cross_kl_ratio": cross_kl,
                    "both_kl_ratio": both_kl,
                    "kl_ratio_vs_both": ratio,
                    "cross_progress": cross_progress,
                    "both_progress": both_progress,
                    "progress_drop": progress_drop,
                    "score": progress_drop + min(ratio, 10.0) / 10.0,
                })
            elif both_progress >= 0.4 and progress_drop >= 0.5 and cross_progress <= 0.25:
                events.append({
                    "level": "functional_drop_only",
                    "block": block,
                    "patch_type": cross_name,
                    "cross_kl_ratio": cross_kl,
                    "both_kl_ratio": both_kl,
                    "kl_ratio_vs_both": ratio,
                    "cross_progress": cross_progress,
                    "both_progress": both_progress,
                    "progress_drop": progress_drop,
                    "score": progress_drop,
                })

    best_block = None
    best_val = -math.inf
    for block, vals in block_curve.items():
        val = vals.get("both_progress", float("nan"))
        if math.isfinite(val) and val > best_val:
            best_block = block
            best_val = val

    nonfinite_rows = [row for row in rows if not bool(row.get("finite", 1.0))]
    norm_illegal_rows = [row for row in rows if illegal_flag(row)]
    return {
        "block_curve": block_curve,
        "alpha_curve": alpha_curve,
        "subtype_signature": subtype_signature,
        "contract_events": sorted(events, key=lambda x: float(x.get("score", 0.0)), reverse=True),
        "contract_broken_blocks": sorted({str(event["block"]) for event in events}),
        "best_block_by_both_progress": best_block,
        "nonfinite_rows": len(nonfinite_rows),
        "norm_illegal_rows": len(norm_illegal_rows),
        "nonfinite_by_block": dict(Counter(str(row["block"]) for row in nonfinite_rows)),
        "nonfinite_by_patch": dict(Counter(str(row["patch_type"]) for row in nonfinite_rows)),
        "norm_illegal_by_block": dict(Counter(str(row["block"]) for row in norm_illegal_rows)),
    }


def checkpoint_path(output_dir: Path, model: str, category: str, label: str) -> Path:
    return output_dir / "checkpoints" / model / f"{category}_{label}.json"


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
        blocks = parse_blocks(args.blocks, len(layers))
        target_layers = block_layers(blocks)
        alphas = parse_alphas(args.alphas)
        patch_types = parse_csv(args.patch_types)
        expected_rows = expected_rows_per_pair(blocks, alphas, patch_types, args.dedupe_cross)
        completed_pairs = {
            name for name, count in Counter(str(row.get("pair")) for row in resume_rows).items()
            if name and count >= expected_rows
        }

        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(f"pairs={len(pairs)} categories={sorted({p.category for p in pairs})}")
        log(f"blocks={[block_label(b) for b in blocks]} target_layers={target_layers}")
        log(f"alphas={alphas} patch_types={patch_types} dedupe_cross={args.dedupe_cross}")
        if resume_rows:
            log(f"resume rows={len(resume_rows)} expected_rows_per_pair={expected_rows} completed_pairs={len(completed_pairs)}")

        rows = resume_rows
        start_time = time.time()
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

            for block in blocks:
                needed = set(range(block[0], block[1] + 1))
                if any(layer_idx not in out_a or layer_idx not in out_b for layer_idx in needed):
                    continue
                if any(not {"attn", "mlp", "resid"}.issubset(out_a[layer_idx]) for layer_idx in needed):
                    continue
                if any(not {"attn", "mlp", "resid"}.issubset(out_b[layer_idx]) for layer_idx in needed):
                    continue
                for alpha in alphas:
                    for patch_type in patch_types:
                        if not should_run_patch(patch_type, alpha, args.dedupe_cross):
                            continue
                        patched, natural = forward_block_patch(
                            loaded,
                            pair.a,
                            seq_len,
                            block,
                            patch_type,
                            alpha,
                            out_a,
                            out_b,
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
                            "block": block_label(block),
                            "block_start": block[0],
                            "block_end": block[1],
                            "block_width": block[1] - block[0] + 1,
                            "alpha": alpha,
                            "patch_type": patch_type,
                            "kl_ab": kl_ab,
                            **metrics,
                            **natural,
                        }
                        row["norm_illegal"] = 1.0 if illegal_flag(row) else 0.0
                        rows.append(row)

            if (pair_index + 1) % args.progress_every == 0:
                elapsed = time.time() - start_time
                log(f"progress pairs={pair_index + 1}/{len(pairs)} rows={len(rows)} elapsed={elapsed:.1f}s")
                partial = {
                    "model": args.model,
                    "complete": False,
                    "num_pairs": len(pairs),
                    "num_results": len(rows),
                    "blocks": [block_label(b) for b in blocks],
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
            "categories": sorted({p.category for p in pairs}),
            "subtypes": sorted({p.subtype for p in pairs}),
            "blocks": [block_label(b) for b in blocks],
            "alphas": alphas,
            "patch_types": patch_types,
            "dedupe_cross": args.dedupe_cross,
            "results": rows,
            "summary": summarize(rows),
        }
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(data, indent=2), encoding="utf-8")
        (output_dir / f"{args.model}_phase291_block_contract_scan.json").write_text(
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
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase291_block_contract_scan"))
    parser.add_argument("--categories", default="negation,logical,passive,recursive")
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--max-pairs-per-subtype", type=int, default=999)
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--patch-types", default="attn,mlp,both,resid,cross_battn_amlp,cross_aattn_bmlp")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--label", default="phase291")
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
        f"best_block={summary.get('best_block_by_both_progress')} "
        f"broken_blocks={summary.get('contract_broken_blocks')} "
        f"nonfinite={summary.get('nonfinite_rows')} norm_illegal={summary.get('norm_illegal_rows')}"
    )
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
