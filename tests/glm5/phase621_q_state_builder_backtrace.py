#!/usr/bin/env python3
"""
Phase 621: Q State Builder Backtrace
Q 状态生成器回溯

Phase 620 showed answer-position Q is sufficient to select the correct value
token. This phase backtraces which upstream residual component can regenerate
that Q state, the correct-value-token attention mass, and the final switch.
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
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import COMPONENTS, collect_components, default_layers  # noqa: E402
from phase618_attention_source_pattern_content import full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402
from phase620_value_token_selection_cause_audit import alpha_group_mass, top_heads  # noqa: E402


OUT_ROOT = Path("results/glm5_phase621_q_state_builder_backtrace")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def selection_layers(model_name: str, n_layers: int) -> List[int]:
    if model_name == "qwen3":
        return [li for li in [27, 28, 29] if li < n_layers]
    if model_name == "glm4":
        return [li for li in [32, 33, 34] if li < n_layers]
    if model_name == "deepseek7b":
        return [li for li in [20, 21, 22] if li < n_layers]
    return list(range(max(0, n_layers - 4), n_layers))


def make_patch_hook(model, tokenizer, prompt: str, answer: str, layer_idx: int, component: str,
                    target: torch.Tensor):
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = get_attn(layer)
    mlp = get_mlp(layer)
    pos = answer_prefix_pos(tokenizer, prompt)
    target = target.float().cpu()

    if component == "layer_input":
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
            return replace_input(inputs, x_new)
        return layer.register_forward_pre_hook(hook)

    if component == "layer_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            if pos < y_new.shape[1]:
                y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        return layer.register_forward_hook(hook)

    if component == "attn_out":
        if attn is None:
            return None
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            if pos < y_new.shape[1]:
                y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        return attn.register_forward_hook(hook)

    if component == "mlp_out":
        if mlp is None:
            return None
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            if pos < y_new.shape[1]:
                y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        return mlp.register_forward_hook(hook)
    raise ValueError(component)


def collect_q_alpha(model, tokenizer, device, prompt: str, answer: str, sel_layers: List[int]) -> Dict:
    layers = get_layers(model)
    pos = answer_prefix_pos(tokenizer, prompt)
    q: Dict[int, torch.Tensor] = {}
    handles = []
    for li in sel_layers:
        attn = layers[li].self_attn

        def make_q_hook(layer_idx):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if pos < tensor.shape[1]:
                    q[layer_idx] = tensor[0, pos].detach().float().cpu()
            return hook

        handles.append(attn.q_proj.register_forward_hook(make_q_hook(li)))
    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                output_attentions=True,
                return_dict=True,
            )
        if out.attentions is None:
            raise RuntimeError("output_attentions did not return attention weights")
        alpha = {li: out.attentions[li][0, :, pos, :].detach().float().cpu() for li in sel_layers}
    finally:
        for h in handles:
            h.remove()
    return {"q": q, "alpha": alpha}


def patched_q_alpha(model, tokenizer, device, prompt: str, answer: str, patch_layer: int, component: str,
                    target: torch.Tensor, sel_layers: List[int]) -> Dict:
    handle = make_patch_hook(model, tokenizer, prompt, answer, patch_layer, component, target)
    try:
        return collect_q_alpha(model, tokenizer, device, prompt, answer, sel_layers)
    finally:
        if handle is not None:
            handle.remove()


def q_metrics(base_diag: Dict, repair_diag: Dict, patched_diag: Dict,
              heads_selected: Dict[int, List[int]], heads_by_layer: Dict[int, int]) -> Dict[str, float]:
    cos_vals = []
    proj_vals = []
    norm_ratios = []
    for li, heads in heads_selected.items():
        if li not in base_diag["q"] or li not in repair_diag["q"] or li not in patched_diag["q"]:
            continue
        n_heads = heads_by_layer[li]
        base_q = base_diag["q"][li].float()
        repair_q = repair_diag["q"][li].float()
        patched_q = patched_diag["q"][li].float()
        head_dim = base_q.numel() // max(1, n_heads)
        for hi in heads:
            start = hi * head_dim
            end = start + head_dim
            target = repair_q[start:end] - base_q[start:end]
            got = patched_q[start:end] - base_q[start:end]
            target_norm = float(target.norm().item())
            if target_norm <= 1e-8:
                continue
            cos_vals.append(float(F.cosine_similarity(got, target, dim=0).item()))
            proj_vals.append(float(torch.dot(got, target).item() / (target_norm * target_norm)))
            norm_ratios.append(float(got.norm().item() / target_norm))
    def mean(xs):
        return sum(xs) / max(1, len(xs))
    return {
        "q_delta_cos": mean(cos_vals),
        "q_delta_projection": mean(proj_vals),
        "q_delta_norm_ratio": mean(norm_ratios),
    }


def patch_score(model, tokenizer, device, prompt: str, answer: str, patch_layer: int, component: str,
                target: torch.Tensor) -> float:
    handle = make_patch_hook(model, tokenizer, prompt, answer, patch_layer, component, target)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                p = start + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        if handle is not None:
            handle.remove()


def patched_scores(model, tokenizer, device, prompt: str, values: List[str], patch_layer: int,
                   component: str, comp_cache: Dict[str, Dict], random_mode: bool, seed: int) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        base = comp_cache[ans]["base"].get(patch_layer, {}).get(component)
        repair = comp_cache[ans]["repair"].get(patch_layer, {}).get(component)
        if base is None or repair is None:
            scores[ans] = -100.0
            continue
        delta = repair.float().cpu() - base.float().cpu()
        if random_mode:
            delta = random_same_norm(delta, seed=seed + ai * 997 + patch_layer)
        target = base.float().cpu() + delta
        scores[ans] = patch_score(model, tokenizer, device, prompt, ans, patch_layer, component, target)
    return scores


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "layer": items[0]["layer"],
            "component": items[0]["component"],
            "random": items[0]["random"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "mean_q_delta_cos": 0.0,
            "mean_q_delta_projection": 0.0,
            "mean_q_delta_norm_ratio": 0.0,
            "mean_correct_value_alpha_delta": 0.0,
            "mean_correct_rule_alpha_delta": 0.0,
            "mean_wrong_relation_alpha_delta": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            metric = item["metric"]
            entry["mean_margin_gain"] += metric["margin_gain"]
            entry["mean_correct_delta"] += metric["correct_delta"]
            entry["mean_wrong_delta"] += metric["old_top_wrong_delta"]
            entry["mean_q_delta_cos"] += item["q_metrics"]["q_delta_cos"]
            entry["mean_q_delta_projection"] += item["q_metrics"]["q_delta_projection"]
            entry["mean_q_delta_norm_ratio"] += item["q_metrics"]["q_delta_norm_ratio"]
            entry["mean_correct_value_alpha_delta"] += item["alpha_delta"].get("correct_value_token", 0.0)
            entry["mean_correct_rule_alpha_delta"] += item["alpha_delta"].get("correct_rule_line", 0.0)
            entry["mean_wrong_relation_alpha_delta"] += item["alpha_delta"].get("wrong_same_relation_lines", 0.0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        by_patch[key] = entry
    best = sorted(
        by_patch.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_q_delta_projection"]),
        reverse=True,
    )
    log("Best Q-state builder backtrace patches:")
    for item in best[:18]:
        flag = "random" if item["random"] else "real"
        log(
            f"  L{item['layer']} {item['component']} {flag}: switch={item['switch']}/{item['n']} "
            f"margin={item['mean_margin_gain']:.3f} qproj={item['mean_q_delta_projection']:.3f} "
            f"alpha_cv={item['mean_correct_value_alpha_delta']:.5f}"
        )
    return {"by_patch": by_patch, "best": best[:200]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        patch_layers = parse_layers(args.patch_layers) if args.patch_layers else default_layers(args.model, info.n_layers)
        patch_layers = [li for li in patch_layers if 0 <= li < info.n_layers]
        sel_layers = parse_layers(args.selection_layers) if args.selection_layers else selection_layers(args.model, info.n_layers)
        sel_layers = [li for li in sel_layers if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in sel_layers}
        heads_selected = {li: top_heads(args.model, heads_by_layer[li], args.top_k) for li in sel_layers}
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        log(
            f"{args.model}: patch_layers={patch_layers}, sel_layers={sel_layers}, "
            f"heads={heads_selected}, raw_cases={len(raw_cases)}"
        )

        for si, case in enumerate(raw_cases):
            base_len = answer_prefix_pos(tokenizer, case["base_prompt"])
            repair_len = answer_prefix_pos(tokenizer, case["repair_prompt"])
            if base_len != repair_len:
                filtered["token_len_mismatch"] += 1
                continue
            group_tokens = rule_micro_groups(tokenizer, case["base_prompt"], case, base_len)
            if not group_tokens.get("correct_value_token"):
                filtered["empty_correct_value"] += 1
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
            old_top_wrong = base["top_wrong"]

            comp_cache = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                comp_cache[ans] = {
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, patch_layers),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, patch_layers),
                }

            base_diag = collect_q_alpha(model, tokenizer, device, case["base_prompt"], case["correct"], sel_layers)
            repair_diag = collect_q_alpha(model, tokenizer, device, case["repair_prompt"], case["correct"], sel_layers)
            base_mass = alpha_group_mass(base_diag["alpha"], group_tokens, heads_selected)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_prompt_len": base_len,
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, case["correct"], old_top_wrong),
                "group_sizes": {k: len(v) for k, v in group_tokens.items()},
                "patches": {},
            }

            for li in patch_layers:
                for component in COMPONENTS:
                    for random_mode in [False, True]:
                        suffix = "random" if random_mode else "real"
                        key = f"L{li}|{component}|{suffix}"
                        scores = patched_scores(
                            model, tokenizer, device, case["base_prompt"], values, li, component,
                            comp_cache, random_mode=random_mode, seed=si * 1009 + li * 101 + len(component),
                        )
                        patched = winner_stats(scores, case["correct"])

                        base_vec = comp_cache[case["correct"]]["base"].get(li, {}).get(component)
                        repair_vec = comp_cache[case["correct"]]["repair"].get(li, {}).get(component)
                        if base_vec is None or repair_vec is None:
                            continue
                        delta = repair_vec.float().cpu() - base_vec.float().cpu()
                        if random_mode:
                            delta = random_same_norm(delta, seed=si * 1009 + li * 101 + len(component))
                        target = base_vec.float().cpu() + delta
                        diag = patched_q_alpha(
                            model, tokenizer, device, case["base_prompt"], case["correct"],
                            li, component, target, sel_layers,
                        )
                        q_m = q_metrics(base_diag, repair_diag, diag, heads_selected, heads_by_layer)
                        mass = alpha_group_mass(diag["alpha"], group_tokens, heads_selected)
                        alpha_delta = {g: mass.get(g, 0.0) - base_mass.get(g, 0.0) for g in mass}
                        row["patches"][key] = {
                            "layer": li,
                            "component": component,
                            "random": random_mode,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                            "q_metrics": q_m,
                            "alpha_delta": alpha_delta,
                        }
            rows.append(row)

        return {
            "phase": 621,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "patch_layers": patch_layers,
            "selection_layers": sel_layers,
            "heads_by_layer": heads_by_layer,
            "selected_heads": heads_selected,
            "top_k": args.top_k,
            "components": COMPONENTS,
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
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--patch-layers", default="")
    parser.add_argument("--selection-layers", default="")
    parser.add_argument("--top-k", type=int, default=6)
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
        args.top_k = min(args.top_k, 2)
        if not args.patch_layers:
            layers = default_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)
            args.patch_layers = str(layers[-1])
        if not args.selection_layers:
            layers = selection_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)
            args.selection_layers = str(layers[-1])
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 16)
        args.max_samples = max(args.max_samples, 128)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase621_{args.model}_q_state_builder_backtrace_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
