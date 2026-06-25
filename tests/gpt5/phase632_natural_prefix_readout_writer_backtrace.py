#!/usr/bin/env python3
"""
Phase 632: Natural Prefix Readout Writer Backtrace
自然前缀读出写入器回溯

Phase 631 proved that a final_norm readout direction can open the token0 prefix
gate. This phase asks which natural layer/component deltas write that direction.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components as collect_answer_components  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval, make_cumulative_patches, token_strings  # noqa: E402
from phase629_format_prefix_gate_localization import (  # noqa: E402
    install_answer_pos_layer_patch_hooks,
    install_prompt_last_patch_hooks,
)


OUT_ROOT = Path("results/glm5_phase632_natural_prefix_readout_writer_backtrace")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_unembed(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    return model.get_output_embeddings().weight


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_scan_layers(model_name: str, n_layers: int) -> List[int]:
    if model_name == "qwen3":
        return list(range(max(0, n_layers - 14), n_layers))
    if model_name == "glm4":
        return list(range(max(0, n_layers - 14), n_layers))
    if model_name == "deepseek7b":
        return list(range(max(0, n_layers - 14), n_layers))
    return list(range(n_layers))


def collect_prompt_last_components(
    model,
    tokenizer,
    device,
    prompt: str,
    layers_to_scan: List[int],
    components: List[str],
) -> Dict[int, Dict[str, torch.Tensor]]:
    layers = get_layers(model)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = len(ids) - 1
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    for li in layers_to_scan:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)

        def save(layer_idx, component, tensor):
            if 0 <= pos < tensor.shape[1]:
                captured[layer_idx][component] = tensor[0, pos].detach().float().cpu()

        if "layer_input" in components:
            def make_pre(layer_idx):
                def hook(_module, inputs):
                    save(layer_idx, "layer_input", inputs[0])
                return hook
            handles.append(layer.register_forward_pre_hook(make_pre(li)))

        if "layer_out" in components:
            def make_layer_out(layer_idx):
                def hook(_module, _inputs, output):
                    save(layer_idx, "layer_out", extract_tensor(output))
                return hook
            handles.append(layer.register_forward_hook(make_layer_out(li)))

        if attn is not None and "attn_out" in components:
            def make_attn(layer_idx):
                def hook(_module, _inputs, output):
                    save(layer_idx, "attn_out", extract_tensor(output))
                return hook
            handles.append(attn.register_forward_hook(make_attn(li)))

        if mlp is not None and "mlp_out" in components:
            def make_mlp(layer_idx):
                def hook(_module, _inputs, output):
                    save(layer_idx, "mlp_out", extract_tensor(output))
                return hook
            handles.append(mlp.register_forward_hook(make_mlp(li)))

    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def token0_logits(
    model,
    tokenizer,
    device,
    prompt: str,
    prompt_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    final_delta: torch.Tensor | None = None,
) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    handles = []
    final_handle = None
    captured = {}
    if prompt_patches:
        handles.extend(install_prompt_last_patch_hooks(model, tokenizer, prompt, prompt_patches))
    if final_delta is not None:
        final_norm = get_final_norm(model)
        if final_norm is not None:
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                captured["pre_final_norm_out"] = y[0, -1].detach().float().cpu()
                y_new = y.clone()
                y_new[0, -1, :] = y_new[0, -1, :] + final_delta.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            final_handle = final_norm.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return {"logits": logits.detach().cpu(), "pre_final_norm_out": captured.get("pre_final_norm_out")}
    finally:
        for h in handles:
            h.remove()
        if final_handle is not None:
            final_handle.remove()


def greedy_generate(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    prompt_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    answer_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    final_delta: torch.Tensor | None = None,
) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    gen = []
    top5 = []
    handles = []
    if prompt_patches:
        handles.extend(install_prompt_last_patch_hooks(model, tokenizer, prompt, prompt_patches))
    if answer_patches:
        handles.extend(install_answer_pos_layer_patch_hooks(model, tokenizer, prompt, answer_patches))
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                final_handle = None
                if final_delta is not None and step == 0:
                    final_norm = get_final_norm(model)
                    if final_norm is not None:
                        def hook(_module, _inputs, output):
                            y = extract_tensor(output)
                            y_new = y.clone()
                            y_new[0, -1, :] = y_new[0, -1, :] + final_delta.to(device=y_new.device, dtype=y_new.dtype)
                            if isinstance(output, tuple):
                                return (y_new,) + output[1:]
                            return y_new
                        final_handle = final_norm.register_forward_hook(hook)
                try:
                    logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                finally:
                    if final_handle is not None:
                        final_handle.remove()
                topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
                top5.append([
                    {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(topv.cpu(), topi.cpu())
                ])
                tid = int(torch.argmax(logits).item())
                gen.append(tid)
                ids.append(tid)
    finally:
        for h in handles:
            h.remove()
    return {"ids": gen, "tokens": token_strings(tokenizer, gen), "text": tokenizer.decode(gen), "top5": top5}


def logit_metrics(logits: torch.Tensor, prefix_id: int, competitor_id: int) -> Dict:
    topv, topi = torch.topk(torch.log_softmax(logits.float(), dim=-1), k=5)
    return {
        "tok0_id": int(torch.argmax(logits).item()),
        "prefix_logit": float(logits[prefix_id].item()),
        "competitor_logit": float(logits[competitor_id].item()),
        "prefix_margin": float((logits[prefix_id] - logits[competitor_id]).item()),
        "top5": [{"id": int(i), "logprob": float(v)} for v, i in zip(topv.cpu(), topi.cpu())],
    }


def node_key(li: int, component: str) -> str:
    return f"L{li}_{component}"


def parse_node(key: str) -> Tuple[int, str]:
    left, component = key.split("_", 1)
    return int(left[1:]), component


def add_stat(stats: Dict[str, Dict], key: str, delta: float, cos: float, norm: float) -> None:
    item = stats.setdefault(key, {
        "node": key,
        "n": 0,
        "sum_margin_delta": 0.0,
        "sum_abs_margin_delta": 0.0,
        "positive": 0,
        "sum_cos": 0.0,
        "sum_delta_norm": 0.0,
    })
    item["n"] += 1
    item["sum_margin_delta"] += delta
    item["sum_abs_margin_delta"] += abs(delta)
    item["positive"] += int(delta > 0)
    item["sum_cos"] += cos
    item["sum_delta_norm"] += norm


def finalize_scan_stats(stats: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for item in stats.values():
        n = max(1, item["n"])
        row = dict(item)
        row["mean_margin_delta"] = item["sum_margin_delta"] / n
        row["mean_abs_margin_delta"] = item["sum_abs_margin_delta"] / n
        row["positive_rate"] = item["positive"] / n
        row["mean_cos"] = item["sum_cos"] / n
        row["mean_delta_norm"] = item["sum_delta_norm"] / n
        row["score"] = row["mean_margin_delta"] * row["positive_rate"]
        rows.append(row)
    rows.sort(key=lambda x: (x["score"], x["mean_margin_delta"], x["positive_rate"]), reverse=True)
    return rows


def summarize_causal(causal_rows: List[Dict]) -> Dict:
    stats = {}
    for row in causal_rows:
        key = f"{row['node']}::{row['mode']}"
        item = stats.setdefault(key, {
            "node": row["node"],
            "mode": row["mode"],
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "sum_margin": 0.0,
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["tok0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["sum_margin"] += row["prefix_margin"]
    out = []
    for item in stats.values():
        n = max(1, item["n"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["mean_prefix_margin"] = item["sum_margin"] / n
        out.append(row)
    out.sort(key=lambda x: (x["exact"], x["tok0_hit"], x["mean_prefix_margin"]), reverse=True)
    return {"by_node_mode": out}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        scan_layers = parse_layers(args.scan_layers) if args.scan_layers else default_scan_layers(args.model, info.n_layers)
        scan_layers = [li for li in scan_layers if 0 <= li < info.n_layers]
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        components = [c.strip() for c in args.components.split(",") if c.strip()]
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        W = get_unembed(model).detach().float().cpu()
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        scan_stats: Dict[str, Dict] = {}
        rows = []
        row_caches = []
        target_seen = 0
        log(
            f"{args.model}: scan_layers={scan_layers}, downstream={downstream_layers}, components={components}, "
            f"raw_cases={len(raw_cases)}, tokenization={tokenization}"
        )

        for si, case in enumerate(raw_cases):
            if answer_prefix_pos(tokenizer, case["base_prompt"]) != answer_prefix_pos(tokenizer, case["repair_prompt"]):
                filtered["token_len_mismatch"] += 1
                continue
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                filtered["not_target"] += 1
                continue
            target_seen += int(target_case)

            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            prefix_id = correct_ids[0]
            base_logits = token0_logits(model, tokenizer, device, case["base_prompt"])["logits"]
            repair_logits = token0_logits(model, tokenizer, device, case["repair_prompt"])["logits"]
            top_id = int(torch.argmax(base_logits).item())
            competitor_id = top_id if top_id != prefix_id else int(torch.topk(base_logits, k=2).indices[1].item())
            wdiff = W[prefix_id] - W[competitor_id]
            wunit = wdiff / max(float(wdiff.norm().item()), 1e-8)
            base_prompt_cache = collect_prompt_last_components(
                model, tokenizer, device, case["base_prompt"], scan_layers, components
            )
            repair_prompt_cache = collect_prompt_last_components(
                model, tokenizer, device, case["repair_prompt"], scan_layers, components
            )
            answer_cache = {
                "base": collect_answer_components(model, tokenizer, device, case["base_prompt"], case["correct"], downstream_layers),
                "repair": collect_answer_components(model, tokenizer, device, case["repair_prompt"], case["correct"], downstream_layers),
            }
            semantic_cumulative = make_cumulative_patches(answer_cache, downstream_layers, "layer_out", False, si * 1009 + 23)
            per_node = {}
            for li in scan_layers:
                for component in components:
                    base_vec = base_prompt_cache.get(li, {}).get(component)
                    repair_vec = repair_prompt_cache.get(li, {}).get(component)
                    if base_vec is None or repair_vec is None:
                        continue
                    delta = repair_vec - base_vec
                    margin_delta = float(torch.dot(wdiff, delta).item())
                    cos = float(F.cosine_similarity(delta, wdiff, dim=0).item()) if float(delta.norm().item()) > 1e-8 else 0.0
                    norm = float(delta.norm().item())
                    key = node_key(li, component)
                    add_stat(scan_stats, key, margin_delta, cos, norm)
                    per_node[key] = {
                        "margin_delta": margin_delta,
                        "cos": cos,
                        "delta_norm": norm,
                    }

            base_metrics = logit_metrics(base_logits, prefix_id, competitor_id)
            repair_metrics = logit_metrics(repair_logits, prefix_id, competitor_id)
            row_summary = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_winner": base,
                "repair_winner": repair,
                "old_top_wrong": base["top_wrong"],
                "correct_ids": correct_ids,
                "correct_tokens": token_strings(tokenizer, correct_ids),
                "old_wrong_ids": old_wrong_ids,
                "old_wrong_tokens": token_strings(tokenizer, old_wrong_ids),
                "prefix_id": prefix_id,
                "prefix_text": tokenizer.decode([prefix_id]),
                "competitor_id": competitor_id,
                "competitor_text": tokenizer.decode([competitor_id]),
                "base_metrics": base_metrics,
                "repair_metrics": repair_metrics,
                "repair_minus_base_margin": repair_metrics["prefix_margin"] - base_metrics["prefix_margin"],
                "top_scan_nodes": sorted(
                    [{"node": k, **v} for k, v in per_node.items()],
                    key=lambda x: x["margin_delta"],
                    reverse=True,
                )[:8],
            }
            rows.append(row_summary)
            row_caches.append({
                "summary": row_summary,
                "case": case,
                "base_prompt_cache": base_prompt_cache,
                "repair_prompt_cache": repair_prompt_cache,
                "semantic_cumulative": semantic_cumulative,
                "wdiff": wdiff,
                "wunit": wunit,
                "old_wrong_ids": old_wrong_ids,
            })

        scan_rank = finalize_scan_stats(scan_stats)
        top_nodes = [item["node"] for item in scan_rank[:args.top_k]]
        log("Top natural prefix readout writer candidates:")
        for item in scan_rank[: min(16, len(scan_rank))]:
            log(
                f"  {item['node']}: mean_delta={item['mean_margin_delta']:.3f} "
                f"pos={item['positive_rate']:.3f} cos={item['mean_cos']:.3f} score={item['score']:.3f}"
            )

        causal_rows = []
        for cache in row_caches:
            row = cache["summary"]
            case = cache["case"]
            prefix_id = row["prefix_id"]
            competitor_id = row["competitor_id"]
            correct_ids = row["correct_ids"]
            old_wrong_ids = cache["old_wrong_ids"]
            baseline_modes = {
                "base": {"prompt": case["base_prompt"], "patches": [], "answer": []},
                "repair_prompt": {"prompt": case["repair_prompt"], "patches": [], "answer": []},
                "semantic_cumulative": {
                    "prompt": case["base_prompt"],
                    "patches": [],
                    "answer": cache["semantic_cumulative"],
                },
            }
            for mode, spec in baseline_modes.items():
                logits = token0_logits(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    prompt_patches=spec["patches"],
                )["logits"]
                metrics = logit_metrics(logits, prefix_id, competitor_id)
                gen = greedy_generate(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    prompt_patches=spec["patches"],
                    answer_patches=spec["answer"],
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                causal_rows.append({
                    "sample_idx": row["sample_idx"],
                    "node": "__baseline__",
                    "mode": mode,
                    "tok0_id": metrics["tok0_id"],
                    "tok0_text": tokenizer.decode([metrics["tok0_id"]]),
                    "prefix_id": prefix_id,
                    "competitor_id": competitor_id,
                    "prefix_margin": metrics["prefix_margin"],
                    "eval": ev,
                    "generation_text": gen["text"] if len(causal_rows) < 120 else "",
                })
            for key in top_nodes:
                li, component = parse_node(key)
                base_vec = cache["base_prompt_cache"].get(li, {}).get(component)
                repair_vec = cache["repair_prompt_cache"].get(li, {}).get(component)
                if base_vec is None or repair_vec is None:
                    continue
                delta = repair_vec - base_vec
                random_target = base_vec + random_same_norm(delta, seed=row["sample_idx"] * 917 + li * 31 + len(component))
                reverse_target = base_vec - delta
                modes = {
                    "restore": {
                        "prompt": case["base_prompt"],
                        "patches": [(li, component, repair_vec)],
                        "answer": [],
                    },
                    "restore_semantic": {
                        "prompt": case["base_prompt"],
                        "patches": [(li, component, repair_vec)],
                        "answer": cache["semantic_cumulative"],
                    },
                    "random_semantic": {
                        "prompt": case["base_prompt"],
                        "patches": [(li, component, random_target)],
                        "answer": cache["semantic_cumulative"],
                    },
                    "reverse_semantic": {
                        "prompt": case["base_prompt"],
                        "patches": [(li, component, reverse_target)],
                        "answer": cache["semantic_cumulative"],
                    },
                    "remove_from_repair": {
                        "prompt": case["repair_prompt"],
                        "patches": [(li, component, base_vec)],
                        "answer": [],
                    },
                }
                if args.include_oracle:
                    base_norm = float(token0_logits(model, tokenizer, device, case["base_prompt"])["logits"].norm().item())
                    modes["oracle_readout_semantic"] = {
                        "prompt": case["base_prompt"],
                        "patches": [],
                        "answer": cache["semantic_cumulative"],
                        "final_delta": cache["wunit"] * (base_norm * args.oracle_scale),
                    }
                for mode, spec in modes.items():
                    logits = token0_logits(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        prompt_patches=spec.get("patches"),
                        final_delta=spec.get("final_delta"),
                    )["logits"]
                    metrics = logit_metrics(logits, prefix_id, competitor_id)
                    gen = greedy_generate(
                        model,
                        tokenizer,
                        device,
                        spec["prompt"],
                        max_new_tokens,
                        prompt_patches=spec.get("patches"),
                        answer_patches=spec.get("answer"),
                        final_delta=spec.get("final_delta"),
                    )
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    causal_rows.append({
                        "sample_idx": row["sample_idx"],
                        "node": key,
                        "mode": mode,
                        "tok0_id": metrics["tok0_id"],
                        "tok0_text": tokenizer.decode([metrics["tok0_id"]]),
                        "prefix_id": prefix_id,
                        "competitor_id": competitor_id,
                        "prefix_margin": metrics["prefix_margin"],
                        "eval": ev,
                        "generation_text": gen["text"] if len(causal_rows) < 120 else "",
                    })

        causal_summary = summarize_causal(causal_rows)
        log("Best causal node/mode:")
        for item in causal_summary["by_node_mode"][:16]:
            log(
                f"  {item['node']}::{item['mode']}: tok0={item['tok0_hit']}/{item['n']} "
                f"exact={item['exact']}/{item['n']} margin={item['mean_prefix_margin']:.3f}"
            )

        return {
            "phase": 632,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "scan_layers": scan_layers,
            "downstream_layers": downstream_layers,
            "components": components,
            "top_k": args.top_k,
            "tokenization": tokenization,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "scan_rank": scan_rank,
            "top_nodes": top_nodes,
            "causal_summary": causal_summary,
            "rows": rows,
            "causal_rows": causal_rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--downstream-layers", default="")
    parser.add_argument("--components", default="layer_input,attn_out,mlp_out,layer_out")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--oracle-scale", type=float, default=0.25)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_k = 2
        if not args.scan_layers:
            if args.model == "qwen3":
                args.scan_layers = "26,27,28,29"
                args.downstream_layers = args.downstream_layers or "29,30"
            elif args.model == "glm4":
                args.scan_layers = "31,32,33,34"
                args.downstream_layers = args.downstream_layers or "34,35"
            else:
                args.scan_layers = "20,21,22,23"
                args.downstream_layers = args.downstream_layers or "22,23"
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        args.top_k = max(args.top_k, 6)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase632_{args.model}_natural_prefix_readout_writer_backtrace_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
