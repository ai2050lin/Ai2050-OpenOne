#!/usr/bin/env python3
"""
Phase 627: Natural Generation Token-Position Closure
自然生成逐词元闭环

Phase 626 located value competition at the first discriminative token. This
phase moves from candidate logprob to actual greedy generation with controlled
teacher-forced patches.
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
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components  # noqa: E402
from phase618_attention_source_pattern_content import full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402
from phase620_value_token_selection_cause_audit import top_heads  # noqa: E402
from phase621_q_state_builder_backtrace import collect_q_alpha, make_patch_hook, selection_layers  # noqa: E402
from phase622_residual_state_direction_decomposition import q_backproj_direction  # noqa: E402
from phase623_selection_result_state_separation import build_patch_targets, default_specs, make_mode_specs  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase625_final_readout_bridge_mlp_causal_split import collect_final_norm_sequence  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402


OUT_ROOT = Path("results/glm5_phase627_natural_generation_token_position_closure")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def token_strings(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([tid]) for tid in ids]


def install_layer_patch_hooks(model, tokenizer, prompt: str, answer: str,
                              patches: List[Tuple[int, str, torch.Tensor]]):
    handles = []
    for li, component, target in patches:
        handle = make_patch_hook(model, tokenizer, prompt, answer, li, component, target)
        if handle is not None:
            handles.append(handle)
    return handles


def install_final_norm_generation_hook(
    model,
    prompt_len: int,
    seq_target: Dict,
    patch_kind: str,
    token_mode: str,
    seq_base: Dict | None = None,
    random_seed: int = 0,
):
    final_norm = get_final_norm(model)
    if final_norm is None:
        return None
    n_targets = len(seq_target[patch_kind])
    if token_mode == "token0":
        allowed = {0}
    elif token_mode == "last":
        allowed = {n_targets - 1}
    else:
        allowed = set(range(n_targets))

    def target_for_idx(idx: int) -> torch.Tensor:
        target = seq_target[patch_kind][idx].float().cpu()
        if "random" in token_mode and seq_base is not None:
            base = seq_base[patch_kind][idx].float().cpu()
            target = base + random_same_norm(target - base, seed=random_seed + idx * 101)
        return target

    if patch_kind == "input":
        def pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            for idx in allowed:
                pos = prompt_len + idx
                if 0 <= idx < n_targets and 0 <= pos < x_new.shape[1]:
                    target = target_for_idx(idx)
                    x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])
        return final_norm.register_forward_pre_hook(pre_hook)

    if patch_kind == "output":
        def out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            for idx in allowed:
                pos = prompt_len + idx
                if 0 <= idx < n_targets and 0 <= pos < y_new.shape[1]:
                    target = target_for_idx(idx)
                    y_new[0, pos, :] = target.to(device=y_new.device, dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        return final_norm.register_forward_hook(out_hook)
    raise ValueError(patch_kind)


def greedy_generate_ids(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    layer_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    final_patch: Dict | None = None,
    answer_for_layer_pos: str = "",
) -> Dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    gen = []
    step_top = []
    handles = []
    if layer_patches:
        handles.extend(install_layer_patch_hooks(model, tokenizer, prompt, answer_for_layer_pos, layer_patches))
    if final_patch:
        h = install_final_norm_generation_hook(
            model,
            len(prompt_ids),
            final_patch["seq_target"],
            final_patch["patch_kind"],
            final_patch["token_mode"],
            seq_base=final_patch.get("seq_base"),
            random_seed=final_patch.get("seed", 0),
        )
        if h is not None:
            handles.append(h)
    try:
        with torch.inference_mode():
            for _step in range(max_new_tokens):
                out = model(input_ids=torch.tensor([ids], device=device), return_dict=True)
                logits = out.logits[0, -1].float()
                tid = int(torch.argmax(logits).item())
                gen.append(tid)
                topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
                step_top.append([
                    {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(topv.cpu(), topi.cpu())
                ])
                ids.append(tid)
        return {
            "ids": gen,
            "tokens": token_strings(tokenizer, gen),
            "text": tokenizer.decode(gen),
            "top5": step_top,
        }
    finally:
        for h in handles:
            h.remove()


def make_cumulative_patches(comp_cache: Dict, layers: List[int], component: str,
                            random_mode: bool, seed: int) -> List[Tuple[int, str, torch.Tensor]]:
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


def generation_eval(gen: Dict, correct_ids: List[int], old_wrong_ids: List[int]) -> Dict:
    ids = gen["ids"]
    n = min(len(ids), len(correct_ids))
    exact = ids[:len(correct_ids)] == correct_ids
    wrong_exact = ids[:len(old_wrong_ids)] == old_wrong_ids
    per_pos = []
    for i in range(n):
        per_pos.append({
            "pos": i,
            "generated_id": ids[i],
            "correct_id": correct_ids[i],
            "wrong_id": old_wrong_ids[i] if i < len(old_wrong_ids) else None,
            "is_correct": ids[i] == correct_ids[i],
            "is_wrong": i < len(old_wrong_ids) and ids[i] == old_wrong_ids[i],
        })
    first_bad = None
    for i in range(n):
        if ids[i] != correct_ids[i]:
            first_bad = i
            break
    return {
        "exact_correct": exact,
        "exact_wrong": wrong_exact,
        "prefix_correct_len": 0 if first_bad == 0 else (n if first_bad is None else first_bad),
        "per_pos": per_pos,
    }


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
    best = sorted(by_mode.values(), key=lambda x: (x["exact_correct"], x["mean_prefix_correct_len"]), reverse=True)
    log("Best generation modes:")
    for item in best[:16]:
        log(
            f"  {item['mode']}: exact={item['exact_correct']}/{item['n']} "
            f"prefix={item['mean_prefix_correct_len']:.2f} pos={item['pos_correct_rate']}"
        )
    return {"by_mode": by_mode, "best": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        specs = default_specs(args.model)
        mode_specs = make_mode_specs(specs)
        result_layers = sorted({li for li, _comp, _part in mode_specs["result_only"] if 0 <= li < info.n_layers})
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        scan_layers = sorted(set(result_layers + downstream_layers))
        sel_layers = parse_layers(args.selection_layers) if args.selection_layers else selection_layers(args.model, info.n_layers)
        sel_layers = [li for li in sel_layers if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in sel_layers}
        heads_selected = {li: top_heads(args.model, heads_by_layer[li], args.top_k) for li in sel_layers}
        values = CANDIDATE_VALUES[:4]
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        log(
            f"{args.model}: result_layers={result_layers}, downstream={downstream_layers}, "
            f"max_new_tokens={max_new_tokens}, raw_cases={len(raw_cases)}, tokenization={tokenization}"
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

            comp_cache = {
                "base": collect_components(model, tokenizer, device, case["base_prompt"], case["correct"], scan_layers),
                "repair": collect_components(model, tokenizer, device, case["repair_prompt"], case["correct"], scan_layers),
            }
            base_diag = collect_q_alpha(model, tokenizer, device, case["base_prompt"], case["correct"], sel_layers)
            repair_diag = collect_q_alpha(model, tokenizer, device, case["repair_prompt"], case["correct"], sel_layers)
            direction = q_backproj_direction(model, base_diag, repair_diag, sel_layers, heads_selected, heads_by_layer)
            result_patches, _info = build_patch_targets(
                mode_specs["result_only"], {"base": comp_cache["base"], "repair": comp_cache["repair"]},
                direction, seed=si * 1009 + 17,
            )
            result_random, _info = build_patch_targets(
                mode_specs["result_random_norm"], {"base": comp_cache["base"], "repair": comp_cache["repair"]},
                direction, seed=si * 1009 + 19,
            )
            cumulative_layer = make_cumulative_patches(comp_cache, downstream_layers, "layer_out", False, si * 1009 + 23)
            cumulative_layer_random = make_cumulative_patches(comp_cache, downstream_layers, "layer_out", True, si * 1009 + 29)
            seq_base = collect_final_norm_sequence(model, tokenizer, device, case["base_prompt"], case["correct"], [])
            seq_repair = collect_final_norm_sequence(model, tokenizer, device, case["repair_prompt"], case["correct"], [])
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
            modes = {
                "base": {"prompt": case["base_prompt"], "layer": [], "final": None},
                "repair_prompt": {"prompt": case["repair_prompt"], "layer": [], "final": None},
                "result_only": {"prompt": case["base_prompt"], "layer": result_patches, "final": None},
                "result_random": {"prompt": case["base_prompt"], "layer": result_random, "final": None},
                "cumulative_layer_out": {"prompt": case["base_prompt"], "layer": cumulative_layer, "final": None},
                "cumulative_layer_out_random": {"prompt": case["base_prompt"], "layer": cumulative_layer_random, "final": None},
                "final_output_all": {
                    "prompt": case["base_prompt"],
                    "layer": [],
                    "final": {"seq_target": seq_repair, "seq_base": seq_base, "patch_kind": "output", "token_mode": "all", "seed": si},
                },
                "final_output_random_all": {
                    "prompt": case["base_prompt"],
                    "layer": [],
                    "final": {"seq_target": seq_repair, "seq_base": seq_base, "patch_kind": "output", "token_mode": "random_all", "seed": si},
                },
            }
            for mode, spec in modes.items():
                gen = greedy_generate_ids(
                    model, tokenizer, device, spec["prompt"], max_new_tokens,
                    layer_patches=spec["layer"], final_patch=spec["final"],
                    answer_for_layer_pos=case["correct"],
                )
                row["generations"][mode] = {
                    "mode": mode,
                    "generation": gen,
                    "eval": generation_eval(gen, correct_ids, old_wrong_ids),
                }
            rows.append(row)

        return {
            "phase": 627,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "result_layers": result_layers,
            "downstream_layers": downstream_layers,
            "selection_layers": sel_layers,
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
    out_path = out_dir / f"phase627_{args.model}_natural_generation_token_position_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
