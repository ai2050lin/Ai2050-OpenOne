#!/usr/bin/env python3
"""
Phase 626: Multi-Layer Final Bridge and Token-Position Readout Audit
多层最终桥接与词元位置读出审计

Phase 625 found a partial final-norm bridge but showed a single MLP split cannot
explain result_only. This phase separates answer token positions and tests final
norm direct patches plus multi-layer cumulative patches.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components  # noqa: E402
from phase618_attention_source_pattern_content import full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402
from phase620_value_token_selection_cause_audit import top_heads  # noqa: E402
from phase621_q_state_builder_backtrace import collect_q_alpha, make_patch_hook, selection_layers  # noqa: E402
from phase622_residual_state_direction_decomposition import q_backproj_direction  # noqa: E402
from phase623_selection_result_state_separation import build_patch_targets, default_specs, make_mode_specs, patch_score_multi  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers, install_patch_hooks  # noqa: E402
from phase625_final_readout_bridge_mlp_causal_split import collect_final_norm_sequence  # noqa: E402


OUT_ROOT = Path("results/glm5_phase626_multilayer_final_bridge_token_position_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def token_strings(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([tid]) for tid in ids]


def token_logprobs(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    patches: List[Tuple[int, str, torch.Tensor]],
) -> List[float]:
    handles = install_patch_hooks(model, tokenizer, prompt, answer, patches)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    vals = []
    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                vals.append(float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu()))
        return vals
    finally:
        for h in handles:
            h.remove()


def patch_score_with_final_norm(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    patch_kind: str,
    seq_target: Dict,
    token_mode: str,
    random_seed: int = 0,
    seq_base: Dict | None = None,
) -> float:
    final_norm = get_final_norm(model)
    if final_norm is None:
        return -100.0
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    start = len(prompt_ids) - 1
    positions = [start + i for i in range(len(ans_ids))]
    if token_mode == "token0":
        use_indices = [0]
    elif token_mode == "last":
        use_indices = [len(ans_ids) - 1]
    else:
        use_indices = list(range(len(ans_ids)))

    targets = []
    for idx in use_indices:
        target = seq_target[patch_kind][idx].float().cpu()
        if seq_base is not None and "random" in token_mode:
            base = seq_base[patch_kind][idx].float().cpu()
            target = base + random_same_norm(target - base, seed=random_seed + idx * 101)
        targets.append((positions[idx], target))
    handle = None
    if patch_kind == "input":
        def pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            for pos, target in targets:
                if 0 <= pos < x_new.shape[1]:
                    x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])
        handle = final_norm.register_forward_pre_hook(pre_hook)
    elif patch_kind == "output":
        def out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            for pos, target in targets:
                if 0 <= pos < y_new.shape[1]:
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = final_norm.register_forward_hook(out_hook)
    else:
        raise ValueError(patch_kind)
    try:
        return sum(token_logprobs(model, tokenizer, device, prompt, answer, []))
    finally:
        if handle is not None:
            handle.remove()


def final_norm_scores(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    seq_cache: Dict[str, Dict],
    mode: str,
    seed: int,
) -> Dict[str, float]:
    kind, token_mode = mode.split(":", 1)
    scores = {}
    for ai, ans in enumerate(values):
        scores[ans] = patch_score_with_final_norm(
            model,
            tokenizer,
            device,
            case["base_prompt"],
            ans,
            kind,
            seq_cache[ans]["repair"],
            token_mode,
            random_seed=seed + ai * 997,
            seq_base=seq_cache[ans]["base"],
        )
    return scores


def cumulative_patches(
    comp_cache: Dict,
    layers: List[int],
    component: str,
    random_mode: bool,
    seed: int,
) -> List[Tuple[int, str, torch.Tensor]]:
    patches = []
    for li in layers:
        base = comp_cache["base"].get(li, {}).get(component)
        repair = comp_cache["repair"].get(li, {}).get(component)
        if base is None or repair is None:
            continue
        delta = repair.float().cpu() - base.float().cpu()
        if random_mode:
            delta = random_same_norm(delta, seed=seed + li * 101)
        patches.append((li, component, base.float().cpu() + delta))
    return patches


def cumulative_scores(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    comp_cache: Dict[str, Dict],
    layers: List[int],
    component: str,
    random_mode: bool,
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        patches = cumulative_patches(comp_cache[ans], layers, component, random_mode, seed + ai * 997)
        scores[ans] = patch_score_multi(model, tokenizer, device, case["base_prompt"], ans, patches)
    return scores


def summarize(rows: List[Dict]) -> Dict:
    score_keys = sorted({k for r in rows for k in r["score_modes"]})
    score_modes = {}
    for key in score_keys:
        items = [r["score_modes"][key] for r in rows if key in r["score_modes"]]
        entry = {
            "mode": key,
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            metric = item["metric"]
            entry["mean_margin_gain"] += metric["margin_gain"]
            entry["mean_correct_delta"] += metric["correct_delta"]
            entry["mean_wrong_delta"] += metric["old_top_wrong_delta"]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        score_modes[key] = entry

    token_keys = sorted({k for r in rows for k in r["token_position"]})
    token_position = {}
    for key in token_keys:
        items = [r["token_position"][key] for r in rows if key in r["token_position"]]
        entry = {"key": key, "n": len(items), "mean_correct_delta": 0.0, "mean_wrong_delta": 0.0, "mean_margin_delta": 0.0}
        for item in items:
            entry["mean_correct_delta"] += item["correct_delta"]
            entry["mean_wrong_delta"] += item["wrong_delta"]
            entry["mean_margin_delta"] += item["margin_delta"]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        token_position[key] = entry
    best = sorted(score_modes.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
    log("Best score modes:")
    for item in best[:20]:
        log(f"  {item['mode']}: switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}")
    return {"score_modes": score_modes, "token_position": token_position, "best_scores": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        specs = default_specs(args.model)
        mode_specs = make_mode_specs(specs)
        patch_layers = sorted({li for li, _comp, _part in mode_specs["result_only"]})
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        scan_layers = sorted(set(patch_layers + downstream_layers))
        sel_layers = parse_layers(args.selection_layers) if args.selection_layers else selection_layers(args.model, info.n_layers)
        sel_layers = [li for li in sel_layers if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in sel_layers}
        heads_selected = {li: top_heads(args.model, heads_by_layer[li], args.top_k) for li in sel_layers}
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        log(
            f"{args.model}: result_layers={patch_layers}, downstream={downstream_layers}, "
            f"sel_layers={sel_layers}, raw_cases={len(raw_cases)}, tokenization={tokenization}"
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

            comp_cache: Dict[str, Dict] = {}
            direction_cache: Dict[str, torch.Tensor] = {}
            seq_cache: Dict[str, Dict] = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                comp_cache[ans] = {
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, scan_layers),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, scan_layers),
                }
                base_diag = collect_q_alpha(model, tokenizer, device, case["base_prompt"], ans, sel_layers)
                repair_diag = collect_q_alpha(model, tokenizer, device, case["repair_prompt"], ans, sel_layers)
                direction_cache[ans] = q_backproj_direction(
                    model, base_diag, repair_diag, sel_layers, heads_selected, heads_by_layer
                )
                seq_cache[ans] = {
                    "base": collect_final_norm_sequence(model, tokenizer, device, case["base_prompt"], ans, []),
                    "repair": collect_final_norm_sequence(model, tokenizer, device, case["repair_prompt"], ans, []),
                }

            correct = case["correct"]
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base": base,
                "repair_prompt": repair,
                "target_case": target_case,
                "score_modes": {},
                "token_position": {},
            }

            result_scores = {}
            result_token_lps = {}
            base_token_lps = {}
            for ans in values:
                patches, _info = build_patch_targets(
                    mode_specs["result_only"], comp_cache[ans], direction_cache[ans],
                    seed=si * 1009 + 17,
                )
                result_scores[ans] = patch_score_multi(model, tokenizer, device, case["base_prompt"], ans, patches)
                result_token_lps[ans] = token_logprobs(model, tokenizer, device, case["base_prompt"], ans, patches)
                base_token_lps[ans] = token_logprobs(model, tokenizer, device, case["base_prompt"], ans, [])
            row["score_modes"]["result_only"] = {
                "winner": winner_stats(result_scores, correct),
                "metric": candidate_delta_metric(base_scores, result_scores, correct, old_top_wrong),
            }

            for i in range(min(len(base_token_lps[correct]), len(base_token_lps[old_top_wrong]))):
                c_delta = result_token_lps[correct][i] - base_token_lps[correct][i]
                w_delta = result_token_lps[old_top_wrong][i] - base_token_lps[old_top_wrong][i]
                row["token_position"][f"tok{i}"] = {
                    "position": i,
                    "correct_delta": c_delta,
                    "wrong_delta": w_delta,
                    "margin_delta": c_delta - w_delta,
                }

            final_modes = [
                "input:all",
                "output:all",
                "output:token0",
                "output:last",
                "output:random_all",
            ]
            for mode in final_modes:
                scores = final_norm_scores(model, tokenizer, device, case, values, seq_cache, mode, seed=si * 1009 + len(mode))
                row["score_modes"][f"final_{mode.replace(':', '_')}"] = {
                    "winner": winner_stats(scores, correct),
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                }

            for component in ["mlp_out", "attn_out", "layer_out"]:
                scores = cumulative_scores(
                    model, tokenizer, device, case, values, comp_cache, downstream_layers, component, False,
                    seed=si * 1009 + len(component),
                )
                row["score_modes"][f"cumulative_{component}"] = {
                    "winner": winner_stats(scores, correct),
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                }
            scores = cumulative_scores(
                model, tokenizer, device, case, values, comp_cache, downstream_layers, "layer_out", True,
                seed=si * 1009 + 999,
            )
            row["score_modes"]["cumulative_layer_out_random"] = {
                "winner": winner_stats(scores, correct),
                "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
            }
            rows.append(row)

        return {
            "phase": 626,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "result_patch_layers": patch_layers,
            "downstream_layers": downstream_layers,
            "selection_layers": sel_layers,
            "tokenization": tokenization,
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
    parser.add_argument("--downstream-layers", default="")
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
        if not args.downstream_layers:
            if args.model == "qwen3":
                args.downstream_layers = "29,30"
            elif args.model == "glm4":
                args.downstream_layers = "34,35"
            else:
                args.downstream_layers = "22,23"
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
    out_path = out_dir / f"phase626_{args.model}_multilayer_final_bridge_token_position_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
