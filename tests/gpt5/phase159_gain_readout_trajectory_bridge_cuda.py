#!/usr/bin/env python3
"""
Phase 159: gain-readout to stepwise trajectory bridge.

Bridge the GLM5 gain-readout chain (v_c -> g*w_D -> DCF) with the GPT5
stepwise generation trace. This script intentionally avoids intervention:
it measures whether clean readout strength predicts true 3-token trajectory
success across categories, template families, object splits, and answer formats.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, CATEGORY_READOUT_WORDS, find_token_id  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase151_surface_answer_generation_closure_cuda import surface_strings  # noqa: E402
from phase153_format_syntax_subspace_joint_steering_cuda import build_items_ext, extended_format_prompt, format_token_sets  # noqa: E402
from phase157_final_residual_lmhead_competition_cuda import token_groups_for_case  # noqa: E402
from phase158_stepwise_competition_trace_cuda import trace_condition  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase159_gain_readout_trajectory_bridge")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_final_norm_weight(model: Any) -> torch.Tensor:
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("model", "ln_f"),
        ("transformer", "ln_f"),
        ("decoder", "final_layer_norm"),
    ]
    for root_name, attr in candidates:
        root = getattr(model, root_name, None)
        mod = getattr(root, attr, None) if root is not None else None
        weight = getattr(mod, "weight", None)
        if weight is not None:
            return weight.detach().float().cpu()
    for mod in model.modules():
        name = mod.__class__.__name__.lower()
        weight = getattr(mod, "weight", None)
        if weight is not None and ("rmsnorm" in name or "layernorm" in name):
            last = weight
    if "last" in locals():
        return last.detach().float().cpu()
    raise TypeError("Cannot locate final normalization weight")


def readout_ids(tokenizer: Any, cat: str) -> list[int]:
    ids = []
    for word in [cat] + CATEGORY_READOUT_WORDS.get(cat, []):
        tid = find_token_id(tokenizer, word)
        if tid is not None:
            ids.append(int(tid))
    return sorted(set(ids))


def competitor_ids(tokenizer: Any, cat: str, categories: list[str]) -> list[int]:
    ids = []
    for other in categories:
        if other == cat:
            continue
        ids.extend(readout_ids(tokenizer, other))
    return sorted(set(ids))


def neutral_base(cat: str, obj: str) -> str:
    if cat in {"abstract", "emotion", "action", "event", "time", "number", "relation", "property"}:
        return f"The term {obj} is a concept."
    return f"The item {obj} is a thing."


def build_neutral_items(items: list[dict[str, Any]], fmt: str, options: list[str]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        base = neutral_base(item["cat"], item["obj"])
        neu = dict(item)
        neu["prompt"] = extended_format_prompt(base, fmt, options)
        out.append(neu)
    return out


def capture_hidden_logits(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    hidden_rows = []
    logits_rows = []
    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, batch_items)
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        bidx = torch.arange(out.logits.shape[0], device=out.logits.device)
        hidden_rows.append(out.hidden_states[-1][bidx, pos].detach().float().cpu())
        logits_rows.append(out.logits[bidx, pos].detach().float().cpu())
        del out, batch
        torch.cuda.empty_cache()
    return torch.cat(hidden_rows, dim=0), torch.cat(logits_rows, dim=0)


def mean_for_ids(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros((logits.shape[0],), dtype=torch.float32)
    return logits[:, sorted(set(ids))].float().mean(dim=1)


def max_for_ids(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.full((logits.shape[0],), float("-inf"), dtype=torch.float32)
    return logits[:, sorted(set(ids))].float().max(dim=1).values


def readout_bridge_metrics(
    rich_hidden: torch.Tensor,
    neutral_hidden: torch.Tensor,
    rich_logits: torch.Tensor,
    neutral_logits: torch.Tensor,
    g_vec: np.ndarray,
    w_u: np.ndarray,
    target_ids: list[int],
    comp_ids: list[int],
) -> dict[str, float]:
    v = (rich_hidden - neutral_hidden).float().numpy()
    h = rich_hidden.float().numpy()
    w_d = w_u[target_ids].mean(axis=0) if target_ids else np.zeros(w_u.shape[1], dtype=np.float32)
    q = g_vec * w_d
    proj = v @ q
    h_rms = np.sqrt(np.mean(h * h, axis=1)) + 1e-8
    v_norm = np.linalg.norm(v, axis=1) + 1e-8
    q_norm = float(np.linalg.norm(q) + 1e-8)
    w_norm = float(np.linalg.norm(w_d) + 1e-8)
    cos_q = proj / (v_norm * q_norm)
    cos_w = (v @ w_d) / (v_norm * w_norm)

    rich_t_mean = mean_for_ids(rich_logits, target_ids)
    neutral_t_mean = mean_for_ids(neutral_logits, target_ids)
    rich_c_mean = mean_for_ids(rich_logits, comp_ids)
    neutral_c_mean = mean_for_ids(neutral_logits, comp_ids)
    rich_t_max = max_for_ids(rich_logits, target_ids)
    rich_c_max = max_for_ids(rich_logits, comp_ids)
    neutral_t_max = max_for_ids(neutral_logits, target_ids)
    neutral_c_max = max_for_ids(neutral_logits, comp_ids)
    t_delta = rich_t_mean - neutral_t_mean
    c_delta = rich_c_mean - neutral_c_mean
    max_t_delta = rich_t_max - neutral_t_max
    max_c_delta = rich_c_max - neutral_c_max

    target_abs = float(t_delta.abs().mean().item())
    comp_abs = float(c_delta.abs().mean().item())
    mode = "target_dominant" if target_abs >= comp_abs else "competitor_dominant"
    if float(c_delta.mean().item()) > 0 and mode == "competitor_dominant":
        mode = "competitor_release"

    return {
        "proj_q_over_rms": float(np.mean(proj / h_rms)),
        "proj_q": float(np.mean(proj)),
        "cos_v_q": float(np.mean(cos_q)),
        "cos_v_wd": float(np.mean(cos_w)),
        "gain_cos_boost": float(np.mean(cos_q - cos_w)),
        "v_norm": float(np.mean(v_norm)),
        "q_norm": q_norm,
        "target_mean_logit": float(rich_t_mean.mean().item()),
        "competitor_mean_logit": float(rich_c_mean.mean().item()),
        "target_max_logit": float(rich_t_max.mean().item()),
        "competitor_max_logit": float(rich_c_max.mean().item()),
        "dcf_mean": float((rich_t_mean - rich_c_mean).mean().item()),
        "dcf_max": float((rich_t_max - rich_c_max).mean().item()),
        "target_delta": float(t_delta.mean().item()),
        "competitor_delta": float(c_delta.mean().item()),
        "dcf_delta": float((t_delta - c_delta).mean().item()),
        "target_max_delta": float(max_t_delta.mean().item()),
        "competitor_max_delta": float(max_c_delta.mean().item()),
        "dcf_max_delta": float((max_t_delta - max_c_delta).mean().item()),
        "tc_mode": mode,
    }


def step_margins(trace: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for step in trace["steps"]:
        sid = int(step["step"])
        margins = step["competition"]["margins"]
        out[f"step{sid}_correct_vs_competitor"] = float(margins["correct_vs_competitor"])
        out[f"step{sid}_correct_vs_wrong"] = float(margins["correct_vs_wrong"])
        out[f"step{sid}_correct_vs_format"] = float(margins["correct_vs_format"])
        out[f"step{sid}_correct_vs_generic"] = float(margins["correct_vs_generic"])
        out[f"step{sid}_correct_vs_object"] = float(margins["correct_vs_object"])
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        layers = get_layers(model)
        last_layer = len(layers)
        attn_layer = last_layer
        num_heads = get_num_heads(model, get_attention_module(layers[attn_layer - 1]))
        categories = parse_str_list(args.categories)
        families = parse_str_list(args.template_families)
        splits = parse_str_list(args.splits)
        formats = parse_str_list(args.formats)
        options = categories
        group_ids = format_token_sets(tokenizer)
        g_vec = get_final_norm_weight(model).numpy().astype(np.float32)
        emb = model.get_output_embeddings()
        if emb is None:
            raise TypeError("Cannot locate output embeddings")
        w_u = emb.weight.detach().float().cpu().numpy()
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase159 layers={last_layer}, attn=L{attn_layer}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 159,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": categories,
            "template_families": families,
            "splits": splits,
            "formats": formats,
            "test_objects": args.test_objects,
            "steps": args.steps,
            "readout_definition": "v_c = h_rich(answer_last) - h_neutral(answer_last); q_c = final_norm_weight * mean(W_U[target_readout_ids])",
            "results": {},
        }
        heldout_tpl = [2]
        for split in splits:
            _train_idx, test_idx = split_indices(split, args.train_objects, args.test_objects)
            for family in families:
                for fmt in formats:
                    for cat in categories:
                        held_items = build_items_ext(cat, family, heldout_tpl, test_idx, fmt, options)
                        neutral_items = build_neutral_items(held_items, fmt, options)
                        rich_h, rich_l = capture_hidden_logits(model, tokenizer, held_items, args.batch_size, args.max_length)
                        neu_h, neu_l = capture_hidden_logits(model, tokenizer, neutral_items, args.batch_size, args.max_length)
                        target = readout_ids(tokenizer, cat)
                        comp = competitor_ids(tokenizer, cat, categories)
                        readout = readout_bridge_metrics(rich_h, neu_h, rich_l, neu_l, g_vec, w_u, target, comp)
                        surfaces = surface_strings(cat, "multiple_choice" if fmt == "multiple_choice" else "label_colon")
                        token_groups = token_groups_for_case(tokenizer, cat, fmt, categories, held_items, group_ids)
                        trace = trace_condition(
                            model, tokenizer, layers, held_items, args.batch_size, args.max_length,
                            attn_layer, num_heads, None, None, None, 0.0, args.steps, args.top_k,
                            surfaces, token_groups, args.example_prompts, fmt,
                        )
                        case_key = f"{split}:{family}:{fmt}:{cat}"
                        result["results"][case_key] = {
                            "category": cat,
                            "family": family,
                            "format": fmt,
                            "split": split,
                            "n_prompts": len(held_items),
                            "target_ids": target,
                            "competitor_ids_n": len(comp),
                            "readout": readout,
                            "trajectory": {
                                "hit_rate": trace["hit_rate"],
                                "final_class_rates": trace["final_class_rates"],
                                "trajectory_rates": trace["trajectory_rates"],
                                "step_margins": step_margins(trace),
                                "examples": trace["examples"],
                            },
                        }
                        log(f"{args.model} {case_key}: hit={trace['hit_rate']:.2f} dcf={readout['dcf_mean']:.2f} proj={readout['proj_q_over_rms']:.2f}")
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 159 Gain-Readout Trajectory Bridge: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}; cases={len(result['results'])}; steps={result['steps']}")
    lines.append("")
    lines.append("| case | hit | tc mode | dcf | dcf delta | proj q/rms | step1 margin | step2 margin | step3 margin | top trajectory |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for key, item in sorted(result["results"].items()):
        readout = item["readout"]
        traj = item["trajectory"]
        rates = traj["trajectory_rates"]
        top_traj = ""
        if rates:
            k, v = max(rates.items(), key=lambda kv: kv[1])
            top_traj = f"{k}:{v:.2f}"
        margins = traj["step_margins"]
        lines.append(
            f"| {key} | {traj['hit_rate']:.2f} | {readout['tc_mode']} | "
            f"{readout['dcf_mean']:.2f} | {readout['dcf_delta']:.2f} | {readout['proj_q_over_rms']:.2f} | "
            f"{margins.get('step1_correct_vs_competitor', 0.0):.2f} | "
            f"{margins.get('step2_correct_vs_competitor', 0.0):.2f} | "
            f"{margins.get('step3_correct_vs_competitor', 0.0):.2f} | {top_traj} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="fruit,animal,clothing,emotion,action,plant,time,container,number,furniture")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,answer_one_word,quoted_answer,list_answer,multiple_choice")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--example-prompts", type=int, default=1)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase159_{args.model}_gain_readout_trajectory_bridge.json"
    md_path = out_dir / f"phase159_{args.model}_gain_readout_trajectory_bridge.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
