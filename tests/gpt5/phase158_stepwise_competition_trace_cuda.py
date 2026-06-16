#!/usr/bin/env python3
"""
Phase 158: step-wise competition trace and top-token ecology.

Extend Phase157 first-step LM-head competition into a true 3-step greedy
trajectory trace. For each intervention, record generated tokens, top-k token
ecology, group margins, and coarse trajectory failure modes.
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
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
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
from phase156_set_writer_surface_gate_closure_cuda import (  # noqa: E402
    DEFAULT_ATTN_OFFSET,
    DEFAULT_MLP_OFFSET,
    global_head_rows,
    head_rows_for_case,
    load_phase155,
    make_head_set_ablation_pre_hook,
    make_mlp_projection_ablation_hook,
    random_heads,
    select_heads,
)
from phase157_final_residual_lmhead_competition_cuda import (  # noqa: E402
    competition_metrics,
    token_groups_for_case,
)


OUT_ROOT = Path("results/gpt5_phase158_stepwise_competition_trace")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
GOOD_CLASSES = {"canonical", "synonym", "object_near", "option_like"}


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


def token_label(token_id: int, text: str, token_groups: dict[str, list[int]]) -> str:
    tid = int(token_id)
    ordered = [
        "correct_expanded",
        "wrong_category",
        "object_copy",
        "format_target",
        "option_label",
        "generic_continue",
        "punctuation",
        "whitespace",
    ]
    for name in ordered:
        if tid in set(token_groups.get(name, [])):
            return name
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punctuation"
    if len(stripped) <= 2 and re.search(r"[A-Za-z]", stripped):
        return "fragment"
    if text and not text.startswith((" ", "\n", "\t")) and re.search(r"[A-Za-z]", text):
        return "fragment"
    return "other"


def trajectory_label(generated: str, step_tokens: list[str], step_labels: list[str], surfaces: dict[str, list[str]], fmt: str) -> str:
    final_cls = generation_class(generated, surfaces)
    if final_cls in GOOD_CLASSES:
        if fmt == "quoted_answer" and generated.strip().startswith(("\"", "'")):
            return "quote_path_success"
        if fmt == "list_answer" and generated.lstrip().startswith(("-", "1")):
            return "list_path_success"
        if step_labels and step_labels[0] in {"format_target", "punctuation", "whitespace"}:
            return "format_then_answer"
        return "correct_surface"
    if any(lbl == "wrong_category" for lbl in step_labels):
        return "wrong_semantic"
    if any(lbl == "object_copy" for lbl in step_labels):
        return "object_copy_trap"
    if any(lbl == "generic_continue" for lbl in step_labels):
        return "generic_continuation_trap"
    if all(lbl in {"format_target", "punctuation", "whitespace", "option_label"} for lbl in step_labels):
        if fmt == "quoted_answer":
            return "quote_path_failure"
        if fmt == "list_answer":
            return "list_path_failure"
        if any(lbl == "option_label" for lbl in step_labels):
            return "option_copy_path"
        return "punctuation_trap"
    if any(lbl == "fragment" for lbl in step_labels):
        return "fragment_trap"
    first_cls = generation_class(step_tokens[0] if step_tokens else "", surfaces)
    if first_cls in GOOD_CLASSES:
        return "first_step_good_later_fail"
    return final_cls


def capture_step_logits(
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
    attn = get_attention_module(layers[attn_layer - 1])
    o_proj = get_o_proj(attn)
    mlp = get_mlp_module(layers[mlp_layer - 1]) if mlp_layer is not None and mlp_basis is not None else None
    device = next(model.parameters()).device
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        handles = []
        if head_ids:
            handles.append(o_proj.register_forward_pre_hook(make_head_set_ablation_pre_hook(num_heads, head_ids, ctx["last_pos"])))
        if mlp is not None and mlp_basis is not None:
            handles.append(mlp.register_forward_hook(make_mlp_projection_ablation_hook(ctx["last_pos"], mlp_basis, mlp_scale)))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        for handle in handles:
            handle.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        bidx = torch.arange(out.logits.shape[0], device=out.logits.device)
        rows.append(out.logits[bidx, pos].detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def summarize_topk(logits: torch.Tensor, tokenizer: Any, token_groups: dict[str, list[int]], top_k: int, example_n: int) -> dict[str, Any]:
    vals, ids = torch.topk(logits.float(), k=top_k, dim=-1)
    label_counts: Counter[str] = Counter()
    top1_counts: Counter[str] = Counter()
    examples = []
    for bi in range(ids.shape[0]):
        row = []
        for rank in range(ids.shape[1]):
            tid = int(ids[bi, rank].item())
            text = decode_token(tokenizer, tid)
            label = token_label(tid, text, token_groups)
            label_counts[label] += 1
            if rank == 0:
                top1_counts[label] += 1
            if bi < example_n:
                row.append({"rank": rank + 1, "id": tid, "token": text, "label": label, "logit": float(vals[bi, rank].item())})
        if bi < example_n:
            examples.append(row)
    denom = max(1, ids.shape[0] * ids.shape[1])
    top1_denom = max(1, ids.shape[0])
    return {
        "top20_label_rates": {k: float(v / denom) for k, v in sorted(label_counts.items())},
        "top1_label_rates": {k: float(v / top1_denom) for k, v in sorted(top1_counts.items())},
        "examples": examples,
    }


def trace_condition(
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
    top_k: int,
    surfaces: dict[str, list[str]],
    token_groups: dict[str, list[int]],
    example_n: int,
    fmt: str,
) -> dict[str, Any]:
    items = [dict(x) for x in base_items]
    generated = ["" for _ in items]
    step_tokens: list[list[str]] = [[] for _ in items]
    step_labels: list[list[str]] = [[] for _ in items]
    step_records = []
    for step in range(steps):
        logits = capture_step_logits(
            model, tokenizer, layers, items, batch_size, max_length,
            attn_layer, num_heads, head_ids, mlp_layer, mlp_basis, mlp_scale,
        )
        comp = competition_metrics(logits, token_groups)
        top = summarize_topk(logits, tokenizer, token_groups, top_k, example_n)
        next_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        labels = []
        toks = []
        for i, tid in enumerate(next_ids):
            tok = decode_token(tokenizer, int(tid))
            label = token_label(int(tid), tok, token_groups)
            toks.append(tok)
            labels.append(label)
            generated[i] += tok
            items[i]["prompt"] += tok
            step_tokens[i].append(tok)
            step_labels[i].append(label)
        step_records.append({
            "step": step + 1,
            "generated_label_rates": {k: float(v / max(1, len(labels))) for k, v in Counter(labels).items()},
            "competition": comp,
            "topk": top,
        })
    final_classes = [generation_class(text, surfaces) for text in generated]
    traj = [trajectory_label(generated[i], step_tokens[i], step_labels[i], surfaces, fmt) for i in range(len(generated))]
    hits = [c in GOOD_CLASSES for c in final_classes]
    examples = []
    for i in range(min(example_n, len(generated))):
        examples.append({
            "generated": generated[i],
            "tokens": step_tokens[i],
            "labels": step_labels[i],
            "final_class": final_classes[i],
            "trajectory": traj[i],
        })
    return {
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "final_class_rates": {k: float(v / max(1, len(final_classes))) for k, v in Counter(final_classes).items()},
        "trajectory_rates": {k: float(v / max(1, len(traj))) for k, v in Counter(traj).items()},
        "steps": step_records,
        "examples": examples,
    }


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
        num_heads = get_num_heads(model, get_attention_module(layers[attn_layer - 1]))
        all_categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        group_ids = format_token_sets(tokenizer)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, all_categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase158 attn=L{attn_layer}, mlp=L{mlp_layer}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 158,
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
            "top_k": args.top_k,
            "conditions": ["clean", "mlp_joint", "joint_k8_mlp_joint", "random_k8"],
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
                        joint8 = select_heads(rows, "joint", min(8, num_heads))
                        rand8 = random_heads(num_heads, min(8, num_heads), 15800 + len(result["results"]), set(joint8))
                        token_groups = token_groups_for_case(tokenizer, cat, fmt, test_categories, held_items, group_ids)
                        configs = {
                            "clean": {"heads": None, "mlp": False},
                            "mlp_joint": {"heads": None, "mlp": True},
                            "joint_k8_mlp_joint": {"heads": joint8, "mlp": True},
                            "random_k8": {"heads": rand8, "mlp": False},
                        }
                        traces = {}
                        for name, cfg in configs.items():
                            traces[name] = trace_condition(
                                model, tokenizer, layers, held_items,
                                args.batch_size, args.max_length, attn_layer, num_heads,
                                cfg["heads"], mlp_layer if cfg["mlp"] else None,
                                mlp_basis if cfg["mlp"] else None,
                                args.mlp_ablate_scale, args.steps, args.top_k,
                                surfaces, token_groups, args.example_prompts, fmt,
                            )
                        result["results"][case_key] = {
                            "n_prompts": len(held_items),
                            "category": cat,
                            "format": fmt,
                            "family": family,
                            "split": split,
                            "head_sets": {"joint_k8": joint8, "random_k8": rand8},
                            "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
                            "conditions": traces,
                        }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 158 Step-wise Competition Trace: {result['model']}", ""]
    lines.append(
        f"Generated: {result['timestamp']}; attn=L{result['attention_layer']}; "
        f"mlp=L{result['mlp_layer']}; heads={result['num_heads']}; steps={result['steps']}"
    )
    lines.append("")
    lines.append("| case | clean hit | mlp hit | k8+mlp hit | random hit | clean traj | mlp traj | k8+mlp traj |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        cond = item["conditions"]

        def top_traj(name: str) -> str:
            rates = cond[name]["trajectory_rates"]
            if not rates:
                return ""
            k, v = max(rates.items(), key=lambda kv: kv[1])
            return f"{k}:{v:.2f}"

        lines.append(
            f"| {key} | {cond['clean']['hit_rate']:.2f} | {cond['mlp_joint']['hit_rate']:.2f} | "
            f"{cond['joint_k8_mlp_joint']['hit_rate']:.2f} | {cond['random_k8']['hit_rate']:.2f} | "
            f"{top_traj('clean')} | {top_traj('mlp_joint')} | {top_traj('joint_k8_mlp_joint')} |"
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--example-prompts", type=int, default=2)
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
    json_path = out_dir / f"phase158_{args.model}_stepwise_competition_trace.json"
    md_path = out_dir / f"phase158_{args.model}_stepwise_competition_trace.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
