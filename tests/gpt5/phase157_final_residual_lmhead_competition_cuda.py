#!/usr/bin/env python3
"""
Phase 157: final residual and LM-head competition decomposition.

For the Phase156 conditions that changed generation hit, capture the answer-site
final hidden state and first-step logits. Decompose output competition into
correct surface answers, wrong categories, format tokens, option labels, generic
continuations, and object-copy tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, CATEGORY_READOUT_WORDS, collect_readout_rows  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import first_token_set, rank_for_ids, surface_strings  # noqa: E402
from phase153_format_syntax_subspace_joint_steering_cuda import (  # noqa: E402
    build_items_ext,
    capture_records_with_format,
    format_contrast_basis,
    format_token_sets,
    route_format,
)
from phase154_format_writer_surface_gate_cuda import format_target_ids, joint_basis  # noqa: E402
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


OUT_ROOT = Path("results/gpt5_phase157_final_residual_lmhead_competition")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")
GENERIC_CONTINUATIONS = [
    " is", " are", " the", " a", " an", " it", " this", " that",
    " answer", " category", " type", " kind", " belongs", " classified",
    " in", " to", " of", " and", " because", " can",
]
PUNCTUATION_STRINGS = [" ", "\n", "\t", ".", ",", ":", ";", "-", "\"", "'", "(", ")", "[", "]"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encoded_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]


def first_ids(tokenizer: Any, strings: list[str]) -> list[int]:
    return first_token_set(tokenizer, strings)


def wrong_category_strings(cat: str, categories: list[str]) -> list[str]:
    out: list[str] = []
    for other in categories:
        if other == cat:
            continue
        out.append(other)
        out.extend(CATEGORY_READOUT_WORDS.get(other, []))
    return sorted(set(out + [" " + x for x in out]))


def object_first_ids(tokenizer: Any, items: list[dict[str, Any]]) -> list[int]:
    ids = []
    for item in items:
        for text in [item["obj"], " " + item["obj"]]:
            toks = encoded_ids(tokenizer, text)
            if toks:
                ids.append(toks[0])
    return sorted(set(ids))


def token_groups_for_case(
    tokenizer: Any,
    cat: str,
    fmt: str,
    categories: list[str],
    items: list[dict[str, Any]],
    group_ids: dict[str, list[int]],
) -> dict[str, list[int]]:
    surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")
    return {
        "correct_expanded": first_ids(tokenizer, surfaces["expanded"]),
        "canonical": first_ids(tokenizer, surfaces["canonical"]),
        "synonym": first_ids(tokenizer, surfaces["synonyms"]),
        "object_near": first_ids(tokenizer, surfaces["object_near"]),
        "wrong_category": first_ids(tokenizer, wrong_category_strings(cat, categories)),
        "format_target": sorted(set(format_target_ids(fmt, group_ids))),
        "punctuation": first_ids(tokenizer, PUNCTUATION_STRINGS),
        "whitespace": group_ids.get("whitespace", []),
        "option_label": group_ids.get("option_label", []),
        "generic_continue": first_ids(tokenizer, GENERIC_CONTINUATIONS),
        "object_copy": object_first_ids(tokenizer, items),
    }


def max_logits_for_ids(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.full((logits.shape[0],), float("-inf"), dtype=torch.float32)
    tid = torch.tensor(sorted(set(ids)), dtype=torch.long)
    return logits[:, tid].float().max(dim=1).values


def group_metric(logits: torch.Tensor, ids: list[int]) -> dict[str, float]:
    if not ids:
        return {"max_logit": float("-inf"), "rank": 0.0, "argmax_rate": 0.0}
    vals = max_logits_for_ids(logits, ids)
    rank = rank_for_ids(logits, ids)
    tid = set(int(x) for x in ids)
    argmax = logits.argmax(dim=-1).detach().cpu().tolist()
    return {
        "max_logit": float(vals.mean().item()),
        "rank": float(rank["rank"]),
        "argmax_rate": float(np.mean([int(x) in tid for x in argmax])) if argmax else 0.0,
    }


def competition_metrics(logits: torch.Tensor, token_groups: dict[str, list[int]]) -> dict[str, Any]:
    groups = {name: group_metric(logits, ids) for name, ids in token_groups.items()}
    correct = max_logits_for_ids(logits, token_groups["correct_expanded"])
    competitor_names = ["wrong_category", "format_target", "punctuation", "option_label", "generic_continue", "object_copy"]
    comp_vals = [max_logits_for_ids(logits, token_groups[name]) for name in competitor_names]
    competitor = torch.stack(comp_vals, dim=1).max(dim=1).values if comp_vals else torch.zeros_like(correct)
    wrong = max_logits_for_ids(logits, token_groups["wrong_category"])
    fmt = max_logits_for_ids(logits, token_groups["format_target"])
    generic = max_logits_for_ids(logits, token_groups["generic_continue"])
    obj = max_logits_for_ids(logits, token_groups["object_copy"])
    return {
        "groups": groups,
        "margins": {
            "correct_vs_competitor": float((correct - competitor).mean().item()),
            "correct_vs_wrong": float((correct - wrong).mean().item()),
            "correct_vs_format": float((correct - fmt).mean().item()),
            "correct_vs_generic": float((correct - generic).mean().item()),
            "correct_vs_object": float((correct - obj).mean().item()),
        },
    }


def projection_norm(hidden: torch.Tensor, basis: np.ndarray) -> float:
    b = torch.tensor(basis, dtype=torch.float32)
    if b.numel() == 0:
        return 0.0
    b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    vals = hidden.float() @ b.T
    return float(vals.norm(dim=1).mean().item())


def capture_condition(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_rows = []
    hidden_rows = []
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
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for handle in handles:
            handle.remove()
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        bidx = torch.arange(out.logits.shape[0], device=out.logits.device)
        logits_rows.append(out.logits[bidx, pos].detach().float().cpu())
        hidden_rows.append(out.hidden_states[-1][bidx, pos].detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(logits_rows, dim=0), torch.cat(hidden_rows, dim=0)


def hidden_delta_metrics(hidden: torch.Tensor, clean_hidden: torch.Tensor, sem_basis: np.ndarray, fmt_basis: np.ndarray) -> dict[str, float]:
    diff = hidden.float() - clean_hidden.float()
    denom = hidden.float().norm(dim=1) * clean_hidden.float().norm(dim=1) + 1e-8
    cos = (hidden.float() * clean_hidden.float()).sum(dim=1) / denom
    return {
        "delta_norm": float(diff.norm(dim=1).mean().item()),
        "cos_to_clean": float(cos.mean().item()),
        "semantic_projection_norm": projection_norm(hidden, sem_basis),
        "format_projection_norm": projection_norm(hidden, fmt_basis),
        "delta_semantic_projection_norm": projection_norm(diff, sem_basis),
        "delta_format_projection_norm": projection_norm(diff, fmt_basis),
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
        log(f"{args.model}: phase157 attn=L{attn_layer}, mlp=L{mlp_layer}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 157,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attention_layer": attn_layer,
            "mlp_layer": mlp_layer,
            "num_heads": num_heads,
            "categories": test_categories,
            "families": families,
            "splits": splits,
            "formats": formats,
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
                        sem_basis, _ = svd_basis(
                            build_category_contrast_matrix(semantic_cache[sem_key]["ans_centers"], all_categories, cat),
                            args.rank,
                        )
                        mlp_basis = joint_basis(sem_basis, fmt_basis, args.rank + args.format_rank)
                        case_key = f"{split}:{family}:{fmt}:{cat}"
                        rows = head_rows_for_case(phase155, case_key, fallback_head_rows)
                        joint8 = select_heads(rows, "joint", min(8, num_heads))
                        rand8 = random_heads(num_heads, min(8, num_heads), 15700 + len(result["results"]), set(joint8))
                        token_groups = token_groups_for_case(tokenizer, cat, fmt, test_categories, held_items, group_ids)
                        configs = {
                            "clean": {"heads": None, "mlp": False},
                            "mlp_joint": {"heads": None, "mlp": True},
                            "joint_k8_mlp_joint": {"heads": joint8, "mlp": True},
                            "random_k8": {"heads": rand8, "mlp": False},
                        }
                        captures: dict[str, Any] = {}
                        clean_logits = None
                        clean_hidden = None
                        for name, cfg in configs.items():
                            logits, hidden = capture_condition(
                                model, tokenizer, layers, held_items,
                                args.batch_size, args.max_length, attn_layer, num_heads,
                                cfg["heads"], mlp_layer if cfg["mlp"] else None,
                                mlp_basis if cfg["mlp"] else None,
                                args.mlp_ablate_scale,
                            )
                            if name == "clean":
                                clean_logits = logits
                                clean_hidden = hidden
                            assert clean_logits is not None and clean_hidden is not None
                            comp = competition_metrics(logits, token_groups)
                            hmet = hidden_delta_metrics(hidden, clean_hidden, sem_basis, fmt_basis)
                            captures[name] = {
                                "competition": comp,
                                "hidden": hmet,
                            }
                        result["results"][case_key] = {
                            "n_prompts": len(held_items),
                            "category": cat,
                            "format": fmt,
                            "family": family,
                            "split": split,
                            "head_sets": {"joint_k8": joint8, "random_k8": rand8},
                            "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
                            "conditions": captures,
                        }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 157 Final Residual LM-head Competition: {result['model']}", ""]
    lines.append(
        f"Generated: {result['timestamp']}; attn=L{result['attention_layer']}; "
        f"mlp=L{result['mlp_layer']}; heads={result['num_heads']}"
    )
    lines.append("")
    lines.append("| case | clean margin | mlp margin Δ | k8+mlp margin Δ | random margin Δ | mlp correct logit Δ | mlp wrong logit Δ | mlp generic logit Δ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        clean = item["conditions"]["clean"]["competition"]
        mlp = item["conditions"]["mlp_joint"]["competition"]
        joint = item["conditions"]["joint_k8_mlp_joint"]["competition"]
        rnd = item["conditions"]["random_k8"]["competition"]
        cm = clean["margins"]["correct_vs_competitor"]

        def dmargin(cond: dict[str, Any]) -> float:
            return cond["margins"]["correct_vs_competitor"] - cm

        def dgroup(cond: dict[str, Any], group: str) -> float:
            return cond["groups"][group]["max_logit"] - clean["groups"][group]["max_logit"]

        lines.append(
            f"| {key} | {cm:.2f} | {dmargin(mlp):+.2f} | {dmargin(joint):+.2f} | {dmargin(rnd):+.2f} | "
            f"{dgroup(mlp, 'correct_expanded'):+.2f} | {dgroup(mlp, 'wrong_category'):+.2f} | "
            f"{dgroup(mlp, 'generic_continue'):+.2f} |"
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
    json_path = out_dir / f"phase157_{args.model}_final_residual_lmhead_competition.json"
    md_path = out_dir / f"phase157_{args.model}_final_residual_lmhead_competition.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
