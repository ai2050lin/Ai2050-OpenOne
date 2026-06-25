#!/usr/bin/env python3
"""
Phase 629: Format/Prefix Gate Localization
格式/前缀门定位

Phase 628 showed that forced token0 plus semantic cumulative layer_out almost
closes natural generation. This phase stops forcing token0 and instead patches
prompt_last components to locate where the format/prefix gate is written.
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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components  # noqa: E402
from phase623_selection_result_state_separation import default_specs, make_mode_specs  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase628_prefix_format_semantic_integration import (  # noqa: E402
    generation_eval,
    make_cumulative_patches,
    token_strings,
)


OUT_ROOT = Path("results/glm5_phase629_format_prefix_gate_localization")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_format_layers(model_name: str, n_layers: int) -> List[int]:
    if model_name == "qwen3":
        return [li for li in range(27, 33) if li < n_layers]
    if model_name == "glm4":
        return [li for li in range(32, 38) if li < n_layers]
    if model_name == "deepseek7b":
        return [li for li in range(20, 26) if li < n_layers]
    return list(range(max(0, n_layers - 8), n_layers - 1))


def collect_prompt_last_components(model, tokenizer, device, prompt: str, layers_to_scan: List[int]) -> Dict[int, Dict]:
    layers = get_layers(model)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = len(ids) - 1
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    for li in layers_to_scan:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)

        def make_layer_pre(layer_idx):
            def hook(_module, inputs):
                x = inputs[0]
                if 0 <= pos < x.shape[1]:
                    captured[layer_idx]["layer_input"] = x[0, pos].detach().float().cpu()
            return hook

        def make_layer_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if 0 <= pos < y.shape[1]:
                    captured[layer_idx]["layer_out"] = y[0, pos].detach().float().cpu()
            return hook

        def make_attn_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if 0 <= pos < y.shape[1]:
                    captured[layer_idx]["attn_out"] = y[0, pos].detach().float().cpu()
            return hook

        def make_mlp_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if 0 <= pos < y.shape[1]:
                    captured[layer_idx]["mlp_out"] = y[0, pos].detach().float().cpu()
            return hook

        handles.append(layer.register_forward_pre_hook(make_layer_pre(li)))
        handles.append(layer.register_forward_hook(make_layer_out(li)))
        if attn is not None:
            handles.append(attn.register_forward_hook(make_attn_out(li)))
        if mlp is not None:
            handles.append(mlp.register_forward_hook(make_mlp_out(li)))

    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def install_prompt_last_patch_hooks(
    model,
    tokenizer,
    prompt: str,
    patches: List[Tuple[int, str, torch.Tensor]],
):
    layers = get_layers(model)
    pos = len(tokenizer.encode(prompt, add_special_tokens=False)) - 1
    handles = []
    for li, component, target in patches:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        target = target.float().cpu()

        if component == "layer_input":
            def hook(_module, inputs, pos=pos, target=target):
                x = inputs[0]
                x_new = x.clone()
                if 0 <= pos < x_new.shape[1]:
                    x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
                return replace_input(inputs, x_new)
            handles.append(layer.register_forward_pre_hook(hook))
        elif component == "layer_out":
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(layer.register_forward_hook(hook))
        elif component == "attn_out" and attn is not None:
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(attn.register_forward_hook(hook))
        elif component == "mlp_out" and mlp is not None:
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(mlp.register_forward_hook(hook))
    return handles


def install_answer_pos_layer_patch_hooks(model, tokenizer, prompt: str, patches: List[Tuple[int, str, torch.Tensor]]):
    layers = get_layers(model)
    pos = len(tokenizer.encode(prompt, add_special_tokens=False))
    handles = []
    for li, component, target in patches:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        target = target.float().cpu()

        if component == "layer_input":
            def hook(_module, inputs, pos=pos, target=target):
                x = inputs[0]
                x_new = x.clone()
                if 0 <= pos < x_new.shape[1]:
                    x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
                return replace_input(inputs, x_new)
            handles.append(layer.register_forward_pre_hook(hook))
        elif component == "layer_out":
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(layer.register_forward_hook(hook))
        elif component == "attn_out" and attn is not None:
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(attn.register_forward_hook(hook))
        elif component == "mlp_out" and mlp is not None:
            def hook(_module, _inputs, output, pos=pos, target=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(mlp.register_forward_hook(hook))
    return handles


def greedy_generate_ids(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    prompt_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    answer_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
) -> Dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    gen = []
    top5 = []
    handles = []
    if prompt_patches:
        handles.extend(install_prompt_last_patch_hooks(model, tokenizer, prompt, prompt_patches))
    if answer_patches:
        handles.extend(install_answer_pos_layer_patch_hooks(model, tokenizer, prompt, answer_patches))
    try:
        with torch.inference_mode():
            for _step in range(max_new_tokens):
                out = model(input_ids=torch.tensor([ids], device=device), return_dict=True)
                logits = out.logits[0, -1].float()
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
    return {
        "ids": gen,
        "tokens": token_strings(tokenizer, gen),
        "text": tokenizer.decode(gen),
        "top5": top5,
    }


def prompt_patch_from_cache(cache: Dict[str, Dict], li: int, component: str, random_mode: bool, seed: int):
    base = cache["base"].get(li, {}).get(component)
    repair = cache["repair"].get(li, {}).get(component)
    if base is None or repair is None:
        return []
    delta = repair.float().cpu() - base.float().cpu()
    if random_mode:
        delta = random_same_norm(delta, seed=seed + li * 997)
    return [(li, component, base.float().cpu() + delta)]


def summarize(rows: List[Dict]) -> Dict:
    modes = sorted({m for r in rows for m in r["generations"]})
    by_mode = {}
    for mode in modes:
        items = [r["generations"][mode] for r in rows if mode in r["generations"]]
        entry = {
            "mode": mode,
            "n": len(items),
            "exact_correct": 0,
            "exact_wrong": 0,
            "mean_prefix_correct_len": 0.0,
            "pos_correct": {},
        }
        for item in items:
            ev = item["eval"]
            entry["exact_correct"] += int(ev["exact_correct"])
            entry["exact_wrong"] += int(ev["exact_wrong"])
            entry["mean_prefix_correct_len"] += ev["prefix_correct_len"]
            for pos_item in ev["per_pos"]:
                p = str(pos_item["pos"])
                entry["pos_correct"].setdefault(p, 0)
                entry["pos_correct"][p] += int(pos_item["is_correct"])
        n = max(1, len(items))
        entry["mean_prefix_correct_len"] /= n
        entry["exact_correct_rate"] = entry["exact_correct"] / n
        entry["exact_wrong_rate"] = entry["exact_wrong"] / n
        entry["pos_correct_rate"] = {p: c / n for p, c in sorted(entry["pos_correct"].items(), key=lambda kv: int(kv[0]))}
        by_mode[mode] = entry

    best_exact = sorted(by_mode.values(), key=lambda x: (x["exact_correct"], x["mean_prefix_correct_len"]), reverse=True)
    best_tok0 = sorted(by_mode.values(), key=lambda x: (x.get("pos_correct_rate", {}).get("0", 0.0), x["exact_correct"]), reverse=True)
    log("Best exact modes:")
    for item in best_exact[:12]:
        log(f"  {item['mode']}: exact={item['exact_correct']}/{item['n']} prefix={item['mean_prefix_correct_len']:.2f} pos={item['pos_correct_rate']}")
    log("Best tok0 modes:")
    for item in best_tok0[:12]:
        log(f"  {item['mode']}: exact={item['exact_correct']}/{item['n']} prefix={item['mean_prefix_correct_len']:.2f} pos={item['pos_correct_rate']}")
    return {"by_mode": by_mode, "best_exact": best_exact, "best_tok0": best_tok0}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        format_layers = parse_layers(args.format_layers) if args.format_layers else default_format_layers(args.model, info.n_layers)
        format_layers = [li for li in format_layers if 0 <= li < info.n_layers]
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        scan_layers = sorted(set(format_layers + downstream_layers))
        mode_specs = make_mode_specs(default_specs(args.model))
        result_layers = sorted({li for li, _comp, _part in mode_specs["result_only"] if 0 <= li < info.n_layers})
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        target_seen = 0
        components = [c.strip() for c in args.components.split(",") if c.strip()]
        log(
            f"{args.model}: format_layers={format_layers}, downstream={downstream_layers}, "
            f"components={components}, raw_cases={len(raw_cases)}, tokenization={tokenization}"
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

            prompt_cache = {
                "base": collect_prompt_last_components(model, tokenizer, device, case["base_prompt"], format_layers),
                "repair": collect_prompt_last_components(model, tokenizer, device, case["repair_prompt"], format_layers),
            }
            comp_cache = {
                "base": collect_components(model, tokenizer, device, case["base_prompt"], case["correct"], scan_layers),
                "repair": collect_components(model, tokenizer, device, case["repair_prompt"], case["correct"], scan_layers),
            }
            semantic_cumulative = make_cumulative_patches(comp_cache, downstream_layers, "layer_out", False, si * 1009 + 23)
            semantic_cumulative_random = make_cumulative_patches(comp_cache, downstream_layers, "layer_out", True, si * 1009 + 29)
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_winner": base,
                "repair_winner": repair,
                "old_top_wrong": base["top_wrong"],
                "correct_ids": correct_ids,
                "correct_tokens": token_strings(tokenizer, correct_ids),
                "old_wrong_ids": old_wrong_ids,
                "old_wrong_tokens": token_strings(tokenizer, old_wrong_ids),
                "generations": {},
            }
            modes: Dict[str, Dict] = {
                "base": {"prompt": case["base_prompt"], "prompt_patches": [], "answer_patches": []},
                "repair_prompt": {"prompt": case["repair_prompt"], "prompt_patches": [], "answer_patches": []},
                "semantic_cumulative_only": {
                    "prompt": case["base_prompt"], "prompt_patches": [], "answer_patches": semantic_cumulative,
                },
                "semantic_cumulative_random": {
                    "prompt": case["base_prompt"], "prompt_patches": [], "answer_patches": semantic_cumulative_random,
                },
            }

            for li in format_layers:
                for comp in components:
                    patch = prompt_patch_from_cache(prompt_cache, li, comp, False, si * 1009 + 31)
                    if not patch:
                        continue
                    key = f"format_L{li}_{comp}"
                    modes[key] = {"prompt": case["base_prompt"], "prompt_patches": patch, "answer_patches": []}
                    modes[f"{key}_semantic"] = {
                        "prompt": case["base_prompt"], "prompt_patches": patch, "answer_patches": semantic_cumulative,
                    }
                    rand_patch = prompt_patch_from_cache(prompt_cache, li, comp, True, si * 1009 + 37)
                    modes[f"{key}_random_semantic"] = {
                        "prompt": case["base_prompt"], "prompt_patches": rand_patch, "answer_patches": semantic_cumulative,
                    }

            for mode, spec in modes.items():
                gen = greedy_generate_ids(
                    model, tokenizer, device, spec["prompt"], max_new_tokens,
                    prompt_patches=spec["prompt_patches"],
                    answer_patches=spec["answer_patches"],
                )
                row["generations"][mode] = {
                    "mode": mode,
                    "generation": gen,
                    "eval": generation_eval(gen, correct_ids, old_wrong_ids),
                }
            rows.append(row)

        return {
            "phase": 629,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "format_layers": format_layers,
            "downstream_layers": downstream_layers,
            "result_layers": result_layers,
            "components": components,
            "tokenization": tokenization,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summarize(rows),
            "rows": rows,
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
    parser.add_argument("--format-layers", default="")
    parser.add_argument("--downstream-layers", default="")
    parser.add_argument("--components", default="layer_input,attn_out,mlp_out,layer_out")
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.components = "layer_input,layer_out"
        if not args.format_layers:
            if args.model == "qwen3":
                args.format_layers = "29,30"
            elif args.model == "glm4":
                args.format_layers = "34,35"
            else:
                args.format_layers = "22,23"
        if not args.downstream_layers:
            args.downstream_layers = args.format_layers
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase629_{args.model}_format_prefix_gate_localization_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
