#!/usr/bin/env python3
"""
Phase 134: causal head source composition and expansion audit.

For true-last causal heads, remove source-specific value contributions from
finer pre-answer source groups and expand categories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES, find_subsequence  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase130_true_last_attention_read_gateway_cuda import (  # noqa: E402
    REFERENCE_COMPONENT,
    capture_answer_component_centers,
    capture_last_input_pre_answer_centers,
    position_audit,
    run_condition as run_phase130_condition,
    summarize_condition,
)
from phase132_source_value_contribution_cuda import (  # noqa: E402
    compute_source_contribution,
    get_num_kv_heads,
    get_v_proj,
    make_contribution_hook,
)
from phase129_position_corrected_gateway_audit_cuda import (  # noqa: E402
    first_nonpad_positions,
    last_nonpad_positions,
    object_span_in_batch,
)


OUT_ROOT = Path("results/gpt5_phase134_causal_head_source_composition")
TEST_CATEGORIES = ["number", "container", "plant", "time", "clothing", "furniture"]
SOURCE_GROUPS = [
    "special_prefix",
    "pre_object",
    "object_span",
    "object_to_template_bridge",
    "post_object_structural",
    "answer_prompt_tail",
    "all_pre_answer",
]
DEFAULT_HEADS = {
    "qwen3": [11, 10, 28, 3, 31, 2, 5, 20],
    "glm4": [1, 28, 0, 18, 11, 27, 23, 4],
    "deepseek7b": [13, 12, 11, 8, 25, 10, 26, 24],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_span_in_batch(tokenizer: Any, item: dict[str, Any], token_ids: list[int], first_nonpad: int, last_nonpad: int) -> tuple[int, int]:
    active = token_ids[first_nonpad:last_nonpad + 1]
    prompt_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
    start = find_subsequence(active, prompt_ids)
    if start is None:
        return first_nonpad, last_nonpad + 1
    begin = first_nonpad + start
    return begin, min(begin + len(prompt_ids), last_nonpad + 1)


def source_groups_for_item(tokenizer: Any, item: dict[str, Any], token_ids: list[int], first_nonpad: int, last_nonpad: int) -> dict[str, list[int]]:
    prompt_start, _prompt_end = prompt_span_in_batch(tokenizer, item, token_ids, first_nonpad, last_nonpad)
    obj_span = object_span_in_batch(tokenizer, item, token_ids, first_nonpad, last_nonpad)
    obj_start = min(obj_span)
    obj_end = max(obj_span) + 1
    all_pre = list(range(first_nonpad, last_nonpad))
    special_prefix = list(range(first_nonpad, prompt_start))
    pre_object = list(range(prompt_start, obj_start))
    post_object = list(range(obj_end, last_nonpad))
    bridge = post_object[: min(2, len(post_object))]
    tail_len = min(2, len(post_object))
    tail = post_object[-tail_len:] if tail_len else []
    tail_set = set(tail)
    bridge_set = set(bridge)
    structural = [p for p in post_object if p not in tail_set and p not in bridge_set]
    return {
        "special_prefix": special_prefix,
        "pre_object": pre_object,
        "object_span": obj_span,
        "object_to_template_bridge": bridge,
        "post_object_structural": structural,
        "answer_prompt_tail": tail,
        "all_pre_answer": all_pre,
    }


def batch_context(tokenizer: Any, batch: dict[str, torch.Tensor], items: list[dict[str, Any]]) -> dict[str, Any]:
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    token_rows = batch["input_ids"].detach().cpu().tolist()
    source_groups = {name: [] for name in SOURCE_GROUPS}
    for bi, item in enumerate(items):
        groups = source_groups_for_item(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi])
        for name in SOURCE_GROUPS:
            source_groups[name].append(groups[name])
    return {"first_pos": first_pos, "last_pos": last_pos, "source_groups": source_groups}


def run_source_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    last_layer: int,
    num_heads: int,
    num_kv_heads: int,
    source_group: str,
    head_ids: list[int],
    contribution_scale: float,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    attn_module = get_attention_module(layers[last_layer - 1])
    v_proj = get_v_proj(attn_module)
    o_proj = get_o_proj(attn_module)
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        value_store: dict[str, torch.Tensor] = {}

        def v_hook(_module: Any, _inputs: Any, output: Any):
            value_store["value"] = output.detach()

        h_v = v_proj.register_forward_hook(v_hook)
        with torch.no_grad():
            first_out = model(**batch, output_attentions=True, use_cache=False)
        h_v.remove()
        if first_out.attentions is None:
            raise RuntimeError("Model did not return attentions")
        attn = first_out.attentions[last_layer - 1].detach().float().cpu().numpy()
        contribution = compute_source_contribution(
            attn,
            value_store["value"],
            ctx["last_pos"],
            ctx["source_groups"][source_group],
            num_heads,
            num_kv_heads,
        )
        h_o = o_proj.register_forward_pre_hook(
            make_contribution_hook(
                contribution,
                torch.tensor(ctx["last_pos"], dtype=torch.long),
                head_ids,
                num_heads,
                contribution_scale,
            )
        )
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        h_o.remove()
        pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del first_out, out, batch, contribution
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def source_audit(tokenizer: Any, prompts: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    batch = tokenizer([x["prompt"] for x in prompts], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    token_rows = batch["input_ids"].tolist()
    lengths = {name: [] for name in SOURCE_GROUPS}
    for bi, item in enumerate(prompts):
        groups = source_groups_for_item(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi])
        for name in SOURCE_GROUPS:
            lengths[name].append(len(groups[name]))
    return {
        name: {
            "mean_len": float(np.mean(vals)) if vals else 0.0,
            "empty_count": int(sum(1 for x in vals if x == 0)),
        }
        for name, vals in lengths.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        head_ids = [int(x) for x in args.heads.split(",") if x.strip()] if args.heads else DEFAULT_HEADS[args.model]
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        attn = get_attention_module(layers[last_layer - 1])
        num_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, num_heads)
        head_ids = [h for h in head_ids if 0 <= h < num_heads]
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, causal_heads={head_ids}, train/test={args.train_objects}/{args.test_objects}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        log("Capturing monitor and reference centers")
        answer_centers = capture_answer_component_centers(
            model, tokenizer, device, layers, categories, last_layer, "last_block_output_answer",
            args.train_objects, args.batch_size, args.max_length,
        )
        reference_centers = capture_last_input_pre_answer_centers(
            model, tokenizer, device, layers, categories, last_layer,
            args.train_objects, args.batch_size, args.max_length,
        )

        result: dict[str, Any] = {
            "phase": 134,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "causal_heads": head_ids,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "rank": args.rank,
            "reference_scale": args.reference_scale,
            "contribution_scale": args.contribution_scale,
            "source_groups": SOURCE_GROUPS,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            monitor_basis, monitor_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            ref_basis, ref_sv = svd_basis(build_category_contrast_matrix(reference_centers, categories, cat), args.rank)
            baseline = run_phase130_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
            )
            ref_patched = run_phase130_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                patch_component=REFERENCE_COMPONENT, patch_basis=ref_basis, scale=args.reference_scale,
            )
            conditions = []
            for source_group in SOURCE_GROUPS:
                patched = run_source_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    num_heads, num_kv_heads, source_group, head_ids, args.contribution_scale,
                )
                conditions.append({
                    "source_group": source_group,
                    "head_ids": head_ids,
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit(tokenizer, prompts, args.max_length),
                "source_audit": source_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "monitor_singular_values": [float(x) for x in monitor_sv],
                "reference_singular_values": [float(x) for x in ref_sv],
                "reference_condition": {
                    "component": REFERENCE_COMPONENT,
                    **summarize_condition(ref_patched, baseline, target_idx, categories),
                },
                "conditions": conditions,
            }
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['source_group']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 134 Causal Head Source Composition: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; causal heads: {result['causal_heads']}")
    lines.append("")
    lines.append("| category | audit | reference | best source | pre_object | object | bridge | structural | tail | all_pre |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        ref = item["reference_condition"]
        ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
        rows = {x["source_group"]: x for x in item["conditions"]}
        best = min(item["conditions"], key=lambda x: x["target_delta"]) if item["conditions"] else None
        lines.append(
            f"| {cat} | {audit_text} | {ref_text} | {_fmt(best)} | "
            f"{_fmt(rows.get('pre_object'))} | {_fmt(rows.get('object_span'))} | "
            f"{_fmt(rows.get('object_to_template_bridge'))} | {_fmt(rows.get('post_object_structural'))} | "
            f"{_fmt(rows.get('answer_prompt_tail'))} | {_fmt(rows.get('all_pre_answer'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--reference-scale", type=float, default=1.5)
    parser.add_argument("--contribution-scale", type=float, default=1.0)
    parser.add_argument("--heads", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase134_{args.model}_causal_head_source_composition.json"
    md_path = out_dir / f"phase134_{args.model}_causal_head_source_composition.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
