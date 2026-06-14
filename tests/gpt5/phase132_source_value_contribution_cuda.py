#!/usr/bin/env python3
"""
Phase 132: true-last source-specific value contribution.

Remove only the pre-o_proj value contribution that the answer token receives
from selected source groups in the true last attention layer.
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
from phase106_multitemplate_residual_cuda import TEMPLATES  # noqa: E402
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
    scan_last_attention,
    summarize_condition,
)
from phase129_position_corrected_gateway_audit_cuda import (  # noqa: E402
    corrected_positions_for_site,
    first_nonpad_positions,
    last_nonpad_positions,
    object_span_in_batch,
)


OUT_ROOT = Path("results/gpt5_phase132_source_value_contribution")
TEST_CATEGORIES = ["number", "container", "plant"]
SOURCE_GROUPS = ["object_span", "post_object_pre_answer", "all_pre_answer", "self"]
HEAD_MODES = ["all_heads", "top_heads"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_v_proj(attn: Any) -> Any:
    for name in ["v_proj", "value", "value_proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise TypeError(f"Cannot find v projection for {type(attn).__name__}")


def get_num_kv_heads(model: Any, attn: Any, num_heads: int) -> int:
    for obj in [attn, getattr(model, "config", None)]:
        if obj is None:
            continue
        for name in ["num_key_value_heads", "multi_query_group_num", "num_kv_heads"]:
            value = getattr(obj, name, None)
            if value:
                return int(value)
    if hasattr(attn, "num_key_value_groups") and getattr(attn, "num_key_value_groups"):
        return max(1, num_heads // int(getattr(attn, "num_key_value_groups")))
    return num_heads


def batch_context(tokenizer: Any, batch: dict[str, torch.Tensor], items: list[dict[str, Any]]) -> dict[str, Any]:
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    token_rows = batch["input_ids"].detach().cpu().tolist()
    source_groups = {name: [] for name in SOURCE_GROUPS}
    for bi, item in enumerate(items):
        obj_span = object_span_in_batch(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi])
        post_object = corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
        all_pre = list(range(first_pos[bi], last_pos[bi]))
        source_groups["object_span"].append(obj_span)
        source_groups["post_object_pre_answer"].append(post_object)
        source_groups["all_pre_answer"].append(all_pre)
        source_groups["self"].append([last_pos[bi]])
    return {
        "first_pos": first_pos,
        "last_pos": last_pos,
        "source_groups": source_groups,
    }


def make_contribution_hook(
    contribution: torch.Tensor,
    answer_positions: torch.Tensor,
    head_ids: list[int],
    num_heads: int,
    scale: float,
):
    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        bidx = torch.arange(y.shape[0], device=y.device)
        pos = answer_positions.to(y.device)
        contrib = contribution.to(y.device, dtype=y_view.dtype)
        for head_id in head_ids:
            y_view[bidx, pos, head_id, :] = y_view[bidx, pos, head_id, :] - scale * contrib[:, head_id, :]
        return (y,) + inputs[1:]
    return hook


def compute_source_contribution(
    attn_weights: np.ndarray,
    value_output: torch.Tensor,
    answer_positions: list[int],
    source_positions: list[list[int]],
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    values = value_output.detach().float().cpu()
    batch, seq_len, value_width = values.shape
    head_dim = value_width // num_kv_heads
    values = values.view(batch, seq_len, num_kv_heads, head_dim)
    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        values = values.repeat_interleave(repeat, dim=2)
    contribution = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
    for bi, ans in enumerate(answer_positions):
        src = [p for p in source_positions[bi] if 0 <= p < seq_len]
        if not src:
            continue
        weights_np = np.asarray(attn_weights[bi, :, ans, src], dtype=np.float32)
        if weights_np.shape[0] != num_heads:
            weights_np = weights_np.T
        weights = torch.tensor(weights_np, dtype=torch.float32)
        vals = values[bi, src, :, :].permute(1, 0, 2)
        contribution[bi] = (weights.unsqueeze(-1) * vals).sum(dim=1)
    return contribution


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
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        attn = get_attention_module(layers[last_layer - 1])
        num_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, num_heads)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB"
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
            "phase": 132,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "reference_scale": args.reference_scale,
            "contribution_scale": args.contribution_scale,
            "top_k_heads": args.top_k_heads,
            "source_groups": SOURCE_GROUPS,
            "head_modes": HEAD_MODES,
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
            attn_scan = scan_last_attention(
                model, tokenizer, device, layers, prompts, last_layer, num_heads,
                args.batch_size, args.max_length,
            )
            top_heads = [int(h) for h in np.argsort(-attn_scan["pre_answer_mass"])[:args.top_k_heads]]
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "monitor_singular_values": [float(x) for x in monitor_sv],
                "reference_singular_values": [float(x) for x in ref_sv],
                "reference_condition": {
                    "component": REFERENCE_COMPONENT,
                    **summarize_condition(ref_patched, baseline, target_idx, categories),
                },
                "attention_scan": {
                    "pre_answer_mass": attn_scan["pre_answer_mass"].tolist(),
                    "self_mass": attn_scan["self_mass"].tolist(),
                },
                "top_heads": top_heads,
                "conditions": [],
            }
            for source_group in SOURCE_GROUPS:
                for mode in HEAD_MODES:
                    head_ids = list(range(num_heads)) if mode == "all_heads" else top_heads
                    patched = run_source_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                        num_heads, num_kv_heads, source_group, head_ids, args.contribution_scale,
                    )
                    cat_out["conditions"].append({
                        "source_group": source_group,
                        "head_mode": mode,
                        "head_ids": head_ids,
                        **summarize_condition(patched, baseline, target_idx, categories),
                    })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['source_group']}:{row['head_mode']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 132 Source-specific Value Contribution: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; heads: {result['num_heads']}; kv_heads: {result['num_kv_heads']}")
    lines.append("")
    lines.append("| category | audit | reference | best source contribution | all_pre/all_heads | post_object/all_heads | object_span/all_heads | self/all_heads |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        rows = item["conditions"]
        best = min(rows, key=lambda x: x["target_delta"]) if rows else None

        def row(group: str, mode: str = "all_heads") -> dict[str, Any] | None:
            return next((x for x in rows if x["source_group"] == group and x["head_mode"] == mode), None)

        ref = item["reference_condition"]
        ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
        lines.append(
            f"| {cat} | {audit_text} | {ref_text} | {_fmt(best)} | "
            f"{_fmt(row('all_pre_answer'))} | {_fmt(row('post_object_pre_answer'))} | "
            f"{_fmt(row('object_span'))} | {_fmt(row('self'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--reference-scale", type=float, default=1.5)
    parser.add_argument("--contribution-scale", type=float, default=1.0)
    parser.add_argument("--top-k-heads", type=int, default=8)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase132_{args.model}_source_value_contribution.json"
    md_path = out_dir / f"phase132_{args.model}_source_value_contribution.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
