#!/usr/bin/env python3
"""
Phase 155: head-level surface gate and multi-step causal closure.

For the key final attention layer, ablate each head at the answer site, rank
heads by answer/format damage, then validate top heads with true 3-step greedy
generation.
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj, make_head_ablation_pre_hook  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import classify_text, first_token_set, rank_for_ids, surface_strings  # noqa: E402
from phase153_format_syntax_subspace_joint_steering_cuda import build_items_ext, format_token_sets  # noqa: E402
from phase154_format_writer_surface_gate_cuda import format_target_ids  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase155_head_surface_gate_generation")
GOOD_CLASSES = {"canonical", "synonym", "object_near", "option_like"}
DEFAULT_LAYER_OFFSET = {
    "qwen3": 0,
    "glm4": -1,
    "deepseek7b": 0,
}


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


def clean_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = batch_context(tokenizer, batch, batch_items)
            out = model(**batch, use_cache=False)
            pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
            rows.append(logits.detach().float().cpu())
            del out, batch
            torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def head_ablation_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layer_id: int,
    head_id: int,
    num_heads: int,
) -> torch.Tensor:
    rows = []
    attn = get_attention_module(layers[layer_id - 1])
    o_proj = get_o_proj(attn)
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        positions = torch.tensor(ctx["last_pos"], dtype=torch.long)
        handle = o_proj.register_forward_pre_hook(make_head_ablation_pre_hook(num_heads, head_id, positions))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        handle.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
        rows.append(logits.detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def metric_row(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    tokenizer: Any,
    surfaces: dict[str, list[str]],
    fmt_ids: list[int],
) -> dict[str, float]:
    answer_ids = first_token_set(tokenizer, surfaces["expanded"])
    base_answer = rank_for_ids(base_logits, answer_ids)
    row_answer = rank_for_ids(logits, answer_ids)
    base_format = rank_for_ids(base_logits, fmt_ids)
    row_format = rank_for_ids(logits, fmt_ids)
    return {
        "answer_rank_delta": row_answer["rank"] - base_answer["rank"],
        "answer_argmax_delta": row_answer["argmax"] - base_answer["argmax"],
        "format_rank_delta": row_format["rank"] - base_format["rank"],
        "format_argmax_delta": row_format["argmax"] - base_format["argmax"],
        "answer_rank": row_answer["rank"],
        "format_rank": row_format["rank"],
    }


def clean_metrics(
    logits: torch.Tensor,
    tokenizer: Any,
    surfaces: dict[str, list[str]],
    fmt_ids: list[int],
) -> dict[str, float]:
    answer_ids = first_token_set(tokenizer, surfaces["expanded"])
    ans = rank_for_ids(logits, answer_ids)
    fmt = rank_for_ids(logits, fmt_ids)
    return {
        "answer_rank": ans["rank"],
        "answer_argmax": ans["argmax"],
        "format_rank": fmt["rank"],
        "format_argmax": fmt["argmax"],
    }


def logits_for_generation_step(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layer_id: int,
    head_id: int | None,
    num_heads: int,
) -> torch.Tensor:
    if head_id is None:
        return clean_logits(model, tokenizer, device, items, batch_size, max_length)
    return head_ablation_logits(model, tokenizer, device, layers, items, batch_size, max_length, layer_id, head_id, num_heads)


def iterative_generate(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    base_items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    layer_id: int,
    head_id: int | None,
    num_heads: int,
    steps: int,
    surfaces: dict[str, list[str]],
) -> dict[str, Any]:
    items = [dict(x) for x in base_items]
    generated = ["" for _ in items]
    first_classes: list[str] = []
    step_classes: list[list[str]] = []
    for step in range(steps):
        logits = logits_for_generation_step(
            model, tokenizer, device, layers, items, batch_size, max_length,
            layer_id, head_id, num_heads,
        )
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for i, tid in enumerate(ids):
            tok = decode_token(tokenizer, int(tid))
            generated[i] += tok
            items[i]["prompt"] += tok
        classes = [generation_class(x, surfaces) for x in generated]
        if step == 0:
            first_classes = classes
        step_classes.append(classes)
    final_classes = [generation_class(x, surfaces) for x in generated]
    hits = [c in GOOD_CLASSES for c in final_classes]
    fmt_first_later = []
    for i in range(len(items)):
        later_good = any(step_classes[s][i] in GOOD_CLASSES for s in range(1, len(step_classes)))
        fmt_first_later.append(first_classes[i] == "format_only" and later_good)
    return {
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "format_first_answer_later_rate": float(np.mean(fmt_first_later)) if fmt_first_later else 0.0,
        "final_class_rates": {k: float(v / max(1, len(final_classes))) for k, v in Counter(final_classes).items()},
        "examples": generated[:8],
    }


def deterministic_random_head(num_heads: int, seed: int, avoid: set[int]) -> int:
    rng = np.random.default_rng(seed)
    choices = [h for h in range(num_heads) if h not in avoid]
    if not choices:
        return 0
    return int(choices[int(rng.integers(0, len(choices)))])


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        layer_id = max(1, min(last_layer, last_layer + int(args.layer_offset)))
        attn = get_attention_module(layers[layer_id - 1])
        num_heads = get_num_heads(model, attn)
        categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        group_ids = format_token_sets(tokenizer)
        _cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, list(CATEGORY_OBJECTS.keys()))
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase155 head scan L{layer_id}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 155,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layer_id": layer_id,
            "num_heads": num_heads,
            "categories": categories,
            "families": families,
            "splits": splits,
            "formats": formats,
            "readout_token_labels": token_labels,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = categories
        for split in splits:
            _train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                for fmt in formats:
                    fmt_ids = sorted(set(format_target_ids(fmt, group_ids)))
                    for cat in categories:
                        held_items = build_items_ext(cat, family, heldout_tpl, test_idx, fmt, options)
                        surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")
                        base_logits = clean_logits(model, tokenizer, device, held_items, args.batch_size, args.max_length)
                        head_rows = []
                        for head_id in range(num_heads):
                            patched = head_ablation_logits(
                                model, tokenizer, device, layers, held_items,
                                args.batch_size, args.max_length, layer_id, head_id, num_heads,
                            )
                            row = metric_row(patched, base_logits, tokenizer, surfaces, fmt_ids)
                            row["head_id"] = int(head_id)
                            head_rows.append(row)
                        top_answer = max(head_rows, key=lambda r: r["answer_rank_delta"])
                        top_format = max(head_rows, key=lambda r: r["format_rank_delta"])
                        top_joint = max(head_rows, key=lambda r: r["answer_rank_delta"] + r["format_rank_delta"])
                        random_head = deterministic_random_head(
                            num_heads, 15500 + len(result["results"]), {top_answer["head_id"], top_format["head_id"], top_joint["head_id"]}
                        )
                        generation_heads = {
                            "clean": None,
                            "top_answer": top_answer["head_id"],
                            "top_format": top_format["head_id"],
                            "top_joint": top_joint["head_id"],
                            "random": random_head,
                        }
                        generations = {
                            name: iterative_generate(
                                model, tokenizer, device, layers, held_items, args.batch_size, args.max_length,
                                layer_id, head, num_heads, args.steps, surfaces,
                            )
                            for name, head in generation_heads.items()
                        }
                        key = f"{split}:{family}:{fmt}:{cat}"
                        result["results"][key] = {
                            "n_prompts": len(held_items),
                            "clean": clean_metrics(base_logits, tokenizer, surfaces, fmt_ids),
                            "heads": head_rows,
                            "top_answer": top_answer,
                            "top_format": top_format,
                            "top_joint": top_joint,
                            "random_head": random_head,
                            "generations": generations,
                        }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 155 Head Surface Gate Generation: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}; layer=L{result['layer_id']}; heads={result['num_heads']}")
    lines.append("")
    lines.append("| case | clean hit | top_answer | hit | top_format | hit | top_joint | hit | random | hit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        gen = item["generations"]
        lines.append(
            f"| {key} | {gen['clean']['hit_rate']:.2f} | "
            f"H{item['top_answer']['head_id']} dA{item['top_answer']['answer_rank_delta']:+.1f} | {gen['top_answer']['hit_rate']:.2f} | "
            f"H{item['top_format']['head_id']} dF{item['top_format']['format_rank_delta']:+.1f} | {gen['top_format']['hit_rate']:.2f} | "
            f"H{item['top_joint']['head_id']} dA{item['top_joint']['answer_rank_delta']:+.1f}/dF{item['top_joint']['format_rank_delta']:+.1f} | {gen['top_joint']['hit_rate']:.2f} | "
            f"H{item['random_head']} | {gen['random']['hit_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--layer-offset", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.layer_offset is None:
        args.layer_offset = DEFAULT_LAYER_OFFSET[args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase155_{args.model}_head_surface_gate_generation.json"
    md_path = out_dir / f"phase155_{args.model}_head_surface_gate_generation.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
