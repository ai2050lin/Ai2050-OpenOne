#!/usr/bin/env python3
"""
Phase 156: set-writer residual surface gate closure.

Use Phase155 head-ranking evidence to test whether cumulative top-k attention
heads, MLP projection ablation, or their joint intervention changes true
multi-step generation. This phase focuses on generation hit rather than rank.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import classify_text, surface_strings  # noqa: E402
from phase153_format_syntax_subspace_joint_steering_cuda import (  # noqa: E402
    build_items_ext,
    capture_records_with_format,
    format_contrast_basis,
    format_token_sets,
    route_format,
)
from phase154_format_writer_surface_gate_cuda import joint_basis  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase156_set_writer_surface_gate_closure")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
PHASE155_ROOT = Path("results/gpt5_phase155_head_surface_gate_generation")
GOOD_CLASSES = {"canonical", "synonym", "object_near", "option_like"}
DIFFICULT_FORMATS = {"label_colon", "answer_one_word", "quoted_answer", "list_answer"}
DEFAULT_ATTN_OFFSET = {"qwen3": 0, "glm4": -1, "deepseek7b": 0}
DEFAULT_MLP_OFFSET = {"qwen3": 0, "glm4": 0, "deepseek7b": 0}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_token(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)


def generation_class(text: str, surfaces: dict[str, list[str]]) -> str:
    cls = classify_text(text, surfaces)
    if cls in GOOD_CLASSES:
        return cls
    if re.fullmatch(r"[\s\W_]+", text) or not text.strip():
        return "format_only"
    return cls


def orthonormal_rows(mat: np.ndarray) -> np.ndarray:
    mat = normalize_basis(mat)
    if mat.size == 0:
        return mat.astype(np.float32)
    q, _ = np.linalg.qr(mat.astype(np.float64).T)
    return q.T.astype(np.float32)


def make_head_set_ablation_pre_hook(num_heads: int, head_ids: list[int], positions: list[int]):
    selected = sorted(set(int(h) for h in head_ids))

    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        bidx = torch.arange(y.shape[0], device=y.device)
        pos = torch.tensor(positions, device=y.device, dtype=torch.long)
        for head_id in selected:
            y_view[bidx, pos, head_id, :] = 0
        return (y,) + inputs[1:]

    return hook


def make_mlp_projection_ablation_hook(answer_positions: list[int], basis: np.ndarray, scale: float):
    b = torch.tensor(orthonormal_rows(basis), dtype=torch.float32)

    def hook(_module: Any, _inputs: Any, output: Any):
        x = tensor_from_output(output)
        out = x.clone()
        if b.numel() > 0:
            bb = b.to(out.device)
            bidx = torch.arange(out.shape[0], device=out.device)
            pos = torch.tensor(answer_positions, device=out.device, dtype=torch.long)
            vec = out[bidx, pos, :].float()
            proj = (vec @ bb.T) @ bb
            out[bidx, pos, :] = out[bidx, pos, :] - scale * proj.to(out.dtype)
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    return hook


def forward_next_logits(
    model: Any,
    tokenizer: Any,
    layers: list[Any],
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    attn_layer: int,
    num_heads: int,
    head_ids: list[int] | None,
    mlp_layer: int | None,
    mlp_basis: np.ndarray | None,
    mlp_scale: float,
) -> torch.Tensor:
    rows = []
    attn_module = get_attention_module(layers[attn_layer - 1])
    o_proj = get_o_proj(attn_module)
    mlp_module = get_mlp_module(layers[mlp_layer - 1]) if mlp_layer is not None and mlp_basis is not None else None
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(model.device if hasattr(model, "device") else next(model.parameters()).device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        handles = []
        if head_ids:
            handles.append(o_proj.register_forward_pre_hook(make_head_set_ablation_pre_hook(num_heads, head_ids, ctx["last_pos"])))
        if mlp_module is not None:
            handles.append(mlp_module.register_forward_hook(make_mlp_projection_ablation_hook(ctx["last_pos"], mlp_basis, mlp_scale)))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        for handle in handles:
            handle.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        rows.append(logits.detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def iterative_generate(
    model: Any,
    tokenizer: Any,
    layers: list[Any],
    base_items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    attn_layer: int,
    num_heads: int,
    head_ids: list[int] | None,
    mlp_layer: int | None,
    mlp_basis: np.ndarray | None,
    mlp_scale: float,
    steps: int,
    surfaces: dict[str, list[str]],
) -> dict[str, Any]:
    items = [dict(x) for x in base_items]
    generated = ["" for _ in items]
    first_classes: list[str] = []
    step_classes: list[list[str]] = []
    for step in range(steps):
        logits = forward_next_logits(
            model, tokenizer, layers, items, batch_size, max_length,
            attn_layer, num_heads, head_ids, mlp_layer, mlp_basis, mlp_scale,
        )
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for i, tid in enumerate(ids):
            tok = decode_token(tokenizer, int(tid))
            generated[i] += tok
            items[i]["prompt"] += tok
        classes = [generation_class(text, surfaces) for text in generated]
        if step == 0:
            first_classes = classes
        step_classes.append(classes)
    final_classes = [generation_class(text, surfaces) for text in generated]
    hits = [c in GOOD_CLASSES for c in final_classes]
    fmt_first_later = []
    for i in range(len(items)):
        later_good = any(step_classes[s][i] in GOOD_CLASSES for s in range(1, len(step_classes)))
        fmt_first_later.append(first_classes[i] == "format_only" and later_good)
    return {
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "format_first_answer_later_rate": float(np.mean(fmt_first_later)) if fmt_first_later else 0.0,
        "final_class_rates": {k: float(v / max(1, len(final_classes))) for k, v in Counter(final_classes).items()},
        "examples": generated[:6],
    }


def load_phase155(model_name: str) -> dict[str, Any]:
    path = PHASE155_ROOT / f"phase155_{model_name}_head_surface_gate_generation.json"
    if not path.exists():
        raise SystemExit(f"Missing Phase155 result needed for head ranking: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def global_head_rows(phase155: dict[str, Any]) -> list[dict[str, Any]]:
    by_head: dict[int, list[dict[str, Any]]] = {}
    for case in phase155["results"].values():
        for row in case.get("heads", []):
            by_head.setdefault(int(row["head_id"]), []).append(row)
    rows = []
    for head_id, vals in by_head.items():
        rows.append({
            "head_id": head_id,
            "answer_rank_delta": float(np.mean([v["answer_rank_delta"] for v in vals])),
            "format_rank_delta": float(np.mean([v["format_rank_delta"] for v in vals])),
        })
    return rows


def select_heads(rows: list[dict[str, Any]], kind: str, k: int) -> list[int]:
    if kind == "answer":
        key = lambda r: r["answer_rank_delta"]
    elif kind == "format":
        key = lambda r: r["format_rank_delta"]
    elif kind == "joint":
        key = lambda r: r["answer_rank_delta"] + r["format_rank_delta"]
    else:
        raise ValueError(kind)
    return [int(r["head_id"]) for r in sorted(rows, key=key, reverse=True)[:k]]


def random_heads(num_heads: int, k: int, seed: int, avoid: set[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    choices = [h for h in range(num_heads) if h not in avoid]
    if len(choices) < k:
        choices = list(range(num_heads))
    picks = rng.permutation(choices)[:k]
    return [int(x) for x in picks]


def head_rows_for_case(phase155: dict[str, Any], key: str, fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case = phase155.get("results", {}).get(key)
    if case and case.get("heads"):
        return case["heads"]
    return fallback_rows


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    phase147_path = PHASE147_ROOT / f"phase147_{args.model}_train_router_format_token.json"
    if not phase147_path.exists():
        raise SystemExit(f"Missing Phase147 result: {phase147_path}")
    phase147 = json.loads(phase147_path.read_text(encoding="utf-8"))
    phase155 = load_phase155(args.model)
    fallback_head_rows = global_head_rows(phase155)
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        attn_layer = max(1, min(last_layer, last_layer + int(args.attn_layer_offset)))
        mlp_layer = max(1, min(last_layer, last_layer + int(args.mlp_layer_offset)))
        attn = get_attention_module(layers[attn_layer - 1])
        num_heads = get_num_heads(model, attn)
        all_categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, all_categories)
        _group_ids = format_token_sets(tokenizer)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase156 attn=L{attn_layer}, mlp=L{mlp_layer}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 156,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attention_layer": attn_layer,
            "mlp_layer": mlp_layer,
            "num_heads": num_heads,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "formats": formats,
            "steps": args.steps,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        semantic_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        format_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for split in splits:
            train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                fmt_key = (split, family)
                if fmt_key not in format_cache:
                    fmt_items = []
                    for fmt in formats:
                        for cat in test_categories:
                            fmt_items.extend(build_items_ext(cat, family, train_tpl, train_idx, fmt, options))
                    fmt_records = capture_records_with_format(
                        model, tokenizer, device, layers, fmt_items, cat_local_ids,
                        all_categories, args.batch_size, args.max_length, mlp_layer,
                    )
                    format_cache[fmt_key] = {"records": fmt_records}
                for fmt in formats:
                    rfmt = route_format(fmt)
                    sem_key = (split, family, rfmt)
                    if sem_key not in semantic_cache:
                        train_all = []
                        for cat in all_categories:
                            train_all.extend(build_items_ext(cat, family, train_tpl, train_idx, rfmt, options))
                        sem_records = capture_records(
                            model, tokenizer, device, layers, train_all, cat_local_ids,
                            all_categories, args.batch_size, args.max_length, mlp_layer,
                        )
                        semantic_cache[sem_key] = {
                            "records": sem_records,
                            "ans_centers": centers_from_records(sem_records, all_categories, "answer_vec", len(train_tpl)),
                        }
                    fmt_basis, _fmt_dir = format_contrast_basis(format_cache[fmt_key]["records"], formats, fmt, args.format_rank)
                    for cat in test_categories:
                        held_items = build_items_ext(cat, family, heldout_tpl, test_idx, fmt, options)
                        surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")
                        sem_basis, _ = svd_basis(
                            build_category_contrast_matrix(semantic_cache[sem_key]["ans_centers"], all_categories, cat),
                            args.rank,
                        )
                        mlp_basis = joint_basis(sem_basis, fmt_basis, args.rank + args.format_rank)
                        case_key = f"{split}:{family}:{fmt}:{cat}"
                        rows = head_rows_for_case(phase155, case_key, fallback_head_rows)
                        answer4 = select_heads(rows, "answer", min(4, num_heads))
                        format4 = select_heads(rows, "format", min(4, num_heads))
                        joint1 = select_heads(rows, "joint", 1)
                        joint4 = select_heads(rows, "joint", min(4, num_heads))
                        joint8 = select_heads(rows, "joint", min(8, num_heads))
                        avoid = set(joint8 + answer4 + format4)
                        rand4 = random_heads(num_heads, min(4, num_heads), 15600 + len(result["results"]) * 3, avoid)
                        rand8 = random_heads(num_heads, min(8, num_heads), 15601 + len(result["results"]) * 3, avoid)
                        conditions = {
                            "clean": {"heads": None, "mlp": False},
                            "joint_k1": {"heads": joint1, "mlp": False},
                            "joint_k4": {"heads": joint4, "mlp": False},
                            "joint_k8": {"heads": joint8, "mlp": False},
                            "answer_k4": {"heads": answer4, "mlp": False},
                            "format_k4": {"heads": format4, "mlp": False},
                            "random_k4": {"heads": rand4, "mlp": False},
                            "random_k8": {"heads": rand8, "mlp": False},
                            "mlp_joint": {"heads": None, "mlp": True},
                            "joint_k4_mlp_joint": {"heads": joint4, "mlp": True},
                            "joint_k8_mlp_joint": {"heads": joint8, "mlp": True},
                        }
                        generations = {}
                        for name, cfg in conditions.items():
                            generations[name] = iterative_generate(
                                model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                                attn_layer, num_heads, cfg["heads"],
                                mlp_layer if cfg["mlp"] else None,
                                mlp_basis if cfg["mlp"] else None,
                                args.mlp_ablate_scale, args.steps, surfaces,
                            )
                        result["results"][case_key] = {
                            "n_prompts": len(held_items),
                            "category": cat,
                            "format": fmt,
                            "family": family,
                            "split": split,
                            "head_sets": {
                                "answer_k4": answer4,
                                "format_k4": format4,
                                "joint_k1": joint1,
                                "joint_k4": joint4,
                                "joint_k8": joint8,
                                "random_k4": rand4,
                                "random_k8": rand8,
                            },
                            "used_case_specific_heads": case_key in phase155.get("results", {}),
                            "generations": generations,
                        }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 156 Set-Writer Surface Gate Closure: {result['model']}", ""]
    lines.append(
        f"Generated: {result['timestamp']}; attn=L{result['attention_layer']}; "
        f"mlp=L{result['mlp_layer']}; heads={result['num_heads']}; steps={result['steps']}"
    )
    lines.append("")
    lines.append("| case | clean | joint_k1 | joint_k4 | joint_k8 | answer_k4 | format_k4 | random_k4 | random_k8 | mlp_joint | k4+mlp | k8+mlp |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        gen = item["generations"]
        lines.append(
            f"| {key} | {gen['clean']['hit_rate']:.2f} | "
            f"{gen['joint_k1']['hit_rate']:.2f} | {gen['joint_k4']['hit_rate']:.2f} | "
            f"{gen['joint_k8']['hit_rate']:.2f} | {gen['answer_k4']['hit_rate']:.2f} | "
            f"{gen['format_k4']['hit_rate']:.2f} | {gen['random_k4']['hit_rate']:.2f} | "
            f"{gen['random_k8']['hit_rate']:.2f} | {gen['mlp_joint']['hit_rate']:.2f} | "
            f"{gen['joint_k4_mlp_joint']['hit_rate']:.2f} | {gen['joint_k8_mlp_joint']['hit_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number,clothing,furniture")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--format-rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--attn-layer-offset", type=int, default=None)
    parser.add_argument("--mlp-layer-offset", type=int, default=None)
    parser.add_argument("--mlp-ablate-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.attn_layer_offset is None:
        args.attn_layer_offset = DEFAULT_ATTN_OFFSET[args.model]
    if args.mlp_layer_offset is None:
        args.mlp_layer_offset = DEFAULT_MLP_OFFSET[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase156_{args.model}_set_writer_surface_gate_closure.json"
    md_path = out_dir / f"phase156_{args.model}_set_writer_surface_gate_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
