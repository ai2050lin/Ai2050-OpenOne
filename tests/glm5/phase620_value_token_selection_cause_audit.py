#!/usr/bin/env python3
"""
Phase 620: Value Token Selection Cause Audit
正确值词元选择原因审计

Phase 619 showed that DS7B's value-rule-line effect is concentrated at the
correct value token. This phase asks why that token is selected:
is the repair mainly from answer-position Q, source-token K, or QK coupling?
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
from typing import Dict, List, Set

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase610_head_cumulative_mixture import TOP_HEADS  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase613_qk_routing_factor_split import clone_out, n_kv_heads_for, restore_out  # noqa: E402
from phase618_attention_source_pattern_content import default_layers, full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402


OUT_ROOT = Path("results/glm5_phase620_value_token_selection_cause_audit")
PATCH_MODES = [
    "q_only",
    "q_random_same_norm",
    "k_correct_value",
    "qk_correct_value",
    "k_all_value_rule_lines",
    "qk_all_value_rule_lines",
]
ALPHA_GROUPS = [
    "correct_value_token",
    "correct_rule_line",
    "all_value_rule_lines",
    "wrong_same_relation_lines",
    "wrong_same_category_lines",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def top_heads(model: str, n_heads: int, k: int) -> List[int]:
    heads = [h for h in TOP_HEADS.get(model, []) if 0 <= h < n_heads]
    if not heads:
        heads = list(range(min(k, n_heads)))
    return heads[:k]


def kv_heads_for(q_heads: List[int], n_heads: int, n_kv: int) -> Set[int]:
    group = max(1, n_heads // max(1, n_kv))
    return {min(n_kv - 1, max(0, h // group)) for h in q_heads}


def collect_projection_parts(model, tokenizer, device, prompt: str, answer: str, layers_to_scan: List[int]) -> Dict[int, Dict]:
    layers = get_layers(model)
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    for li in layers_to_scan:
        attn = layers[li].self_attn

        def make_hook(layer_idx: int, name: str):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[layer_idx][name] = tensor[0].detach().float().cpu()
            return hook

        handles.append(attn.q_proj.register_forward_hook(make_hook(li, "q_proj")))
        handles.append(attn.k_proj.register_forward_hook(make_hook(li, "k_proj")))

    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    for li in layers_to_scan:
        captured[li]["answer_pos"] = torch.tensor(pos)
    return captured


def patch_projection_score_multi(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layers_to_scan: List[int],
    base_parts: Dict[int, Dict],
    repair_parts: Dict[int, Dict],
    heads_by_layer: Dict[int, int],
    kv_by_layer: Dict[int, int],
    heads_by_layer_selected: Dict[int, List[int]],
    group_tokens: Dict[str, List[int]],
    mode: str,
    seed: int,
) -> float:
    layers = get_layers(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    handles = []

    for op_idx, li in enumerate(layers_to_scan):
        attn = layers[li].self_attn
        n_heads = heads_by_layer[li]
        n_kv = kv_by_layer[li]
        q_dim = base_parts[li]["q_proj"].shape[-1]
        kv_dim = base_parts[li]["k_proj"].shape[-1]
        head_dim = q_dim // max(1, n_heads)
        kv_head_dim = kv_dim // max(1, n_kv)
        q_heads = heads_by_layer_selected[li]
        selected_kv = kv_heads_for(q_heads, n_heads, n_kv)
        k_group = []
        if "correct_value" in mode:
            k_group = group_tokens.get("correct_value_token", [])
        elif "all_value_rule_lines" in mode:
            k_group = group_tokens.get("all_value_rule_lines", [])

        def q_hook(_module, _inputs, output, li=li, q_heads=q_heads, head_dim=head_dim, op_idx=op_idx):
            x, rest = clone_out(output)
            if ("q_only" in mode or "qk_" in mode) and pos < x.shape[1]:
                for hi in q_heads:
                    start = hi * head_dim
                    end = start + head_dim
                    if mode == "q_random_same_norm":
                        delta = repair_parts[li]["q_proj"][pos, start:end] - base_parts[li]["q_proj"][pos, start:end]
                        patched = base_parts[li]["q_proj"][pos, start:end] + random_same_norm(
                            delta, seed=seed + op_idx * 1009 + li * 101 + hi
                        )
                    else:
                        patched = repair_parts[li]["q_proj"][pos, start:end]
                    x[0, pos, start:end] = patched.to(device=x.device, dtype=x.dtype)
            return restore_out(x, rest)

        def k_hook(_module, _inputs, output, li=li, selected_kv=selected_kv, kv_head_dim=kv_head_dim, k_group=k_group):
            x, rest = clone_out(output)
            if ("k_" in mode or "qk_" in mode) and k_group:
                for hi in selected_kv:
                    start = hi * kv_head_dim
                    end = start + kv_head_dim
                    for tok in k_group:
                        if 0 <= tok < x.shape[1]:
                            x[0, tok, start:end] = repair_parts[li]["k_proj"][tok, start:end].to(
                                device=x.device, dtype=x.dtype
                            )
            return restore_out(x, rest)

        handles.append(attn.q_proj.register_forward_hook(q_hook))
        handles.append(attn.k_proj.register_forward_hook(k_hook))

    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start_pos = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                p = start_pos + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def collect_alpha(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layers_to_scan: List[int],
    base_parts: Dict[int, Dict] | None,
    repair_parts: Dict[int, Dict] | None,
    heads_by_layer: Dict[int, int],
    heads_by_layer_selected: Dict[int, List[int]],
    mode: str,
    seed: int,
) -> Dict[int, torch.Tensor]:
    layers = get_layers(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = full_ids(tokenizer, prompt, answer)
    pos = len(prompt_ids)
    handles = []

    if mode in ("q_only", "q_random_same_norm"):
        for op_idx, li in enumerate(layers_to_scan):
            attn = layers[li].self_attn
            n_heads = heads_by_layer[li]
            q_heads = heads_by_layer_selected[li]
            q_dim = base_parts[li]["q_proj"].shape[-1]  # type: ignore[index]
            head_dim = q_dim // max(1, n_heads)

            def q_hook(_module, _inputs, output, li=li, q_heads=q_heads, head_dim=head_dim, op_idx=op_idx):
                x, rest = clone_out(output)
                if pos < x.shape[1]:
                    for hi in q_heads:
                        start = hi * head_dim
                        end = start + head_dim
                        if mode == "q_random_same_norm":
                            delta = repair_parts[li]["q_proj"][pos, start:end] - base_parts[li]["q_proj"][pos, start:end]  # type: ignore[index]
                            patched = base_parts[li]["q_proj"][pos, start:end] + random_same_norm(  # type: ignore[index]
                                delta, seed=seed + op_idx * 1009 + li * 101 + hi
                            )
                        else:
                            patched = repair_parts[li]["q_proj"][pos, start:end]  # type: ignore[index]
                        x[0, pos, start:end] = patched.to(device=x.device, dtype=x.dtype)
                return restore_out(x, rest)

            handles.append(attn.q_proj.register_forward_hook(q_hook))

    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                output_attentions=True,
                return_dict=True,
            )
        if out.attentions is None:
            raise RuntimeError("output_attentions did not return attention weights")
        return {li: out.attentions[li][0, :, pos, :].detach().float().cpu() for li in layers_to_scan}
    finally:
        for h in handles:
            h.remove()


def alpha_group_mass(alpha_by_layer: Dict[int, torch.Tensor], group_tokens: Dict[str, List[int]],
                     heads_by_layer_selected: Dict[int, List[int]]) -> Dict[str, float]:
    out = {}
    for group in ALPHA_GROUPS:
        vals = []
        toks = group_tokens.get(group, [])
        for li, alpha in alpha_by_layer.items():
            heads = heads_by_layer_selected[li]
            max_seq = alpha.shape[-1]
            valid = [t for t in toks if 0 <= t < max_seq]
            if not valid:
                continue
            for hi in heads:
                vals.append(float(alpha[hi, valid].sum().cpu()))
        out[group] = sum(vals) / max(1, len(vals))
    return out


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "mode": items[0]["mode"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            metric = item["metric"]
            entry["mean_margin_gain"] += metric["margin_gain"]
            entry["mean_correct_delta"] += metric["correct_delta"]
            entry["mean_wrong_delta"] += metric["old_top_wrong_delta"]
            entry["positive_margin"] += int(metric["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry

    alpha = {}
    for group in ALPHA_GROUPS:
        alpha[group] = {}
        for mode in ["base", "repair", "q_only", "q_random_same_norm"]:
            vals = [r["alpha_mass"][mode].get(group, 0.0) for r in rows if mode in r["alpha_mass"]]
            alpha[group][mode] = sum(vals) / max(1, len(vals))
        alpha[group]["repair_minus_base"] = alpha[group]["repair"] - alpha[group]["base"]
        alpha[group]["q_only_minus_base"] = alpha[group]["q_only"] - alpha[group]["base"]
        alpha[group]["q_random_minus_base"] = alpha[group]["q_random_same_norm"] - alpha[group]["base"]

    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
    log("Best value-token selection-cause patches:")
    for item in best[:12]:
        log(f"  {item['mode']}: switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}")
    log("Alpha mass deltas:")
    for group, item in alpha.items():
        log(
            f"  {group}: repair-base={item['repair_minus_base']:.5f} "
            f"q-base={item['q_only_minus_base']:.5f} random-base={item['q_random_minus_base']:.5f}"
        )
    return {"by_patch": by_patch, "best": best[:100], "alpha": alpha}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        kv_by_layer = {}
        heads_selected = {}
        for li in layers_to_scan:
            attn = get_layers(model)[li].self_attn
            n_heads = heads_by_layer[li]
            q_dim = int(attn.q_proj.out_features)
            head_dim = q_dim // max(1, n_heads)
            kv_by_layer[li] = n_kv_heads_for(attn, n_heads, head_dim)
            heads_selected[li] = top_heads(args.model, n_heads, args.top_k)

        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        log(
            f"{args.model}: layers={info.n_layers}, scan_layers={layers_to_scan}, "
            f"heads={heads_by_layer}, kv={kv_by_layer}, selected={heads_selected}, raw_cases={len(raw_cases)}"
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

            cache = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                cache[ans] = {
                    "base": collect_projection_parts(model, tokenizer, device, case["base_prompt"], ans, layers_to_scan),
                    "repair": collect_projection_parts(model, tokenizer, device, case["repair_prompt"], ans, layers_to_scan),
                }

            alpha_base = collect_alpha(
                model, tokenizer, device, case["base_prompt"], case["correct"],
                layers_to_scan, None, None, heads_by_layer, heads_selected,
                mode="base", seed=si * 1009,
            )
            alpha_repair = collect_alpha(
                model, tokenizer, device, case["repair_prompt"], case["correct"],
                layers_to_scan, None, None, heads_by_layer, heads_selected,
                mode="repair", seed=si * 1009 + 1,
            )
            alpha_q = collect_alpha(
                model, tokenizer, device, case["base_prompt"], case["correct"],
                layers_to_scan, cache[case["correct"]]["base"], cache[case["correct"]]["repair"],
                heads_by_layer, heads_selected, mode="q_only", seed=si * 1009 + 2,
            )
            alpha_q_random = collect_alpha(
                model, tokenizer, device, case["base_prompt"], case["correct"],
                layers_to_scan, cache[case["correct"]]["base"], cache[case["correct"]]["repair"],
                heads_by_layer, heads_selected, mode="q_random_same_norm", seed=si * 1009 + 3,
            )

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_prompt_len": base_len,
                "target_case": target_case,
                "group_sizes": {k: len(v) for k, v in group_tokens.items()},
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, case["correct"], old_top_wrong),
                "alpha_mass": {
                    "base": alpha_group_mass(alpha_base, group_tokens, heads_selected),
                    "repair": alpha_group_mass(alpha_repair, group_tokens, heads_selected),
                    "q_only": alpha_group_mass(alpha_q, group_tokens, heads_selected),
                    "q_random_same_norm": alpha_group_mass(alpha_q_random, group_tokens, heads_selected),
                },
                "patches": {},
            }

            for mode in PATCH_MODES:
                scores = {}
                for ai, ans in enumerate(values):
                    scores[ans] = patch_projection_score_multi(
                        model, tokenizer, device, case["base_prompt"], ans, layers_to_scan,
                        cache[ans]["base"], cache[ans]["repair"],
                        heads_by_layer, kv_by_layer, heads_selected,
                        group_tokens, mode, seed=si * 1009 + ai * 997 + len(mode),
                    )
                patched = winner_stats(scores, case["correct"])
                row["patches"][mode] = {
                    "mode": mode,
                    "winner": patched,
                    "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                }
            rows.append(row)

        return {
            "phase": 620,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "kv_heads_by_layer": kv_by_layer,
            "selected_heads": heads_selected,
            "top_k": args.top_k,
            "patch_modes": PATCH_MODES,
            "alpha_groups": ALPHA_GROUPS,
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
    parser.add_argument("--layers", default="")
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
        if not args.layers:
            layers = default_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)
            args.layers = str(layers[-1])
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
    out_path = out_dir / f"phase620_{args.model}_value_token_selection_cause_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
