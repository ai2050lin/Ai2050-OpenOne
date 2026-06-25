#!/usr/bin/env python3
"""
Phase 623: Selection State vs Result State Separation
选择状态与结果状态分离

Phase 622 split DS7B residual state into upstream Q-aligned selection state and
downstream Q-orthogonal result state. This phase tests whether those states are
additive, redundant, or competing when patched together.
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
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components  # noqa: E402
from phase618_attention_source_pattern_content import full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402
from phase620_value_token_selection_cause_audit import alpha_group_mass, top_heads  # noqa: E402
from phase621_q_state_builder_backtrace import collect_q_alpha, make_patch_hook, q_metrics, selection_layers  # noqa: E402
from phase622_residual_state_direction_decomposition import decompose_delta, q_backproj_direction  # noqa: E402


OUT_ROOT = Path("results/glm5_phase623_selection_result_state_separation")
COMPONENT = "layer_out"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_specs(model_name: str) -> Dict:
    if model_name == "qwen3":
        return {
            "selection": [(26, COMPONENT, "q_backproj_aligned"), (27, COMPONENT, "q_backproj_aligned")],
            "result": [(29, COMPONENT, "q_backproj_orthogonal")],
        }
    if model_name == "glm4":
        return {
            "selection": [(31, COMPONENT, "q_backproj_aligned"), (32, COMPONENT, "q_backproj_aligned")],
            "result": [(34, COMPONENT, "q_backproj_orthogonal")],
        }
    if model_name == "deepseek7b":
        return {
            "selection": [(20, COMPONENT, "q_backproj_aligned"), (21, COMPONENT, "q_backproj_aligned")],
            "result": [(22, COMPONENT, "q_backproj_orthogonal")],
        }
    return {"selection": [], "result": []}


def needed_layers(specs: Dict, n_layers: int) -> List[int]:
    layers = sorted({li for group in specs.values() for li, _component, _mode in group if 0 <= li < n_layers})
    return layers


def make_mode_specs(specs: Dict) -> Dict[str, List[Tuple[int, str, str]]]:
    selection = specs["selection"]
    result = specs["result"]
    return {
        "selection_early": selection[:1],
        "selection_late": selection[-1:],
        "selection_both": selection,
        "result_only": result,
        "selection_both_plus_result": selection + result,
        "selection_late_plus_result": selection[-1:] + result,
        "result_random_norm": [(li, comp, "random_same_norm") for li, comp, _mode in result],
        "selection_random_norm": [(li, comp, "random_same_norm") for li, comp, _mode in selection],
    }


def install_patch_hooks(model, tokenizer, prompt: str, answer: str, patches: List[Tuple[int, str, torch.Tensor]]):
    handles = []
    for li, component, target in patches:
        handle = make_patch_hook(model, tokenizer, prompt, answer, li, component, target)
        if handle is not None:
            handles.append(handle)
    return handles


def patch_score_multi(model, tokenizer, device, prompt: str, answer: str,
                      patches: List[Tuple[int, str, torch.Tensor]]) -> float:
    handles = install_patch_hooks(model, tokenizer, prompt, answer, patches)
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
        for h in handles:
            h.remove()


def build_patch_targets(mode_spec: List[Tuple[int, str, str]], comp_cache: Dict, direction: torch.Tensor,
                        seed: int) -> Tuple[List[Tuple[int, str, torch.Tensor]], Dict[str, float]]:
    patches = []
    norm_ratios = []
    for li, component, part_mode in mode_spec:
        base = comp_cache["base"].get(li, {}).get(component)
        repair = comp_cache["repair"].get(li, {}).get(component)
        if base is None or repair is None:
            continue
        delta = repair.float().cpu() - base.float().cpu()
        piece = decompose_delta(delta, direction, part_mode, seed=seed + li * 101 + len(component))
        target = base.float().cpu() + piece
        denom = float(delta.norm().item())
        norm_ratios.append(float(piece.norm().item()) / denom if denom > 1e-8 else 0.0)
        patches.append((li, component, target))
    return patches, {"mean_piece_norm_ratio": sum(norm_ratios) / max(1, len(norm_ratios))}


def patched_scores(model, tokenizer, device, case: Dict, values: List[str], mode_spec: List[Tuple[int, str, str]],
                   comp_cache: Dict[str, Dict], direction_cache: Dict[str, torch.Tensor], seed: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    scores = {}
    norm_info = {}
    for ai, ans in enumerate(values):
        patches, info = build_patch_targets(
            mode_spec, comp_cache[ans], direction_cache[ans], seed=seed + ai * 997
        )
        scores[ans] = patch_score_multi(model, tokenizer, device, case["base_prompt"], ans, patches)
        if ans == case["correct"]:
            norm_info = info
    return scores, norm_info


def patched_diag(model, tokenizer, device, case: Dict, answer: str, mode_spec: List[Tuple[int, str, str]],
                 comp_cache: Dict, direction: torch.Tensor, sel_layers: List[int], seed: int) -> Dict:
    patches, _info = build_patch_targets(mode_spec, comp_cache, direction, seed)
    handles = install_patch_hooks(model, tokenizer, case["base_prompt"], answer, patches)
    try:
        return collect_q_alpha(model, tokenizer, device, case["base_prompt"], answer, sel_layers)
    finally:
        for h in handles:
            h.remove()


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
            "mean_q_delta_projection": 0.0,
            "mean_correct_value_alpha_delta": 0.0,
            "mean_wrong_relation_alpha_delta": 0.0,
            "mean_piece_norm_ratio": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            metric = item["metric"]
            entry["mean_margin_gain"] += metric["margin_gain"]
            entry["mean_correct_delta"] += metric["correct_delta"]
            entry["mean_wrong_delta"] += metric["old_top_wrong_delta"]
            entry["mean_q_delta_projection"] += item["q_metrics"]["q_delta_projection"]
            entry["mean_correct_value_alpha_delta"] += item["alpha_delta"].get("correct_value_token", 0.0)
            entry["mean_wrong_relation_alpha_delta"] += item["alpha_delta"].get("wrong_same_relation_lines", 0.0)
            entry["mean_piece_norm_ratio"] += item.get("mean_piece_norm_ratio", 0.0)
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
    log("Best selection/result state patches:")
    for item in best[:18]:
        log(
            f"  {item['mode']}: switch={item['switch']}/{item['n']} "
            f"margin={item['mean_margin_gain']:.3f} qproj={item['mean_q_delta_projection']:.3f} "
            f"alpha_cv={item['mean_correct_value_alpha_delta']:.5f}"
        )
    return {"by_patch": by_patch, "best": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        specs = default_specs(args.model)
        patch_layers = parse_layers(args.patch_layers) if args.patch_layers else needed_layers(specs, info.n_layers)
        patch_layers = [li for li in patch_layers if 0 <= li < info.n_layers]
        sel_layers = parse_layers(args.selection_layers) if args.selection_layers else selection_layers(args.model, info.n_layers)
        sel_layers = [li for li in sel_layers if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in sel_layers}
        heads_selected = {li: top_heads(args.model, heads_by_layer[li], args.top_k) for li in sel_layers}
        mode_specs = make_mode_specs(specs)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        log(
            f"{args.model}: patch_layers={patch_layers}, sel_layers={sel_layers}, "
            f"heads={heads_selected}, specs={mode_specs}, raw_cases={len(raw_cases)}"
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
            diag_cache: Dict[str, Dict] = {}
            direction_cache: Dict[str, torch.Tensor] = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                comp_cache[ans] = {
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, patch_layers),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, patch_layers),
                }
                base_diag = collect_q_alpha(model, tokenizer, device, case["base_prompt"], ans, sel_layers)
                repair_diag = collect_q_alpha(model, tokenizer, device, case["repair_prompt"], ans, sel_layers)
                diag_cache[ans] = {"base": base_diag, "repair": repair_diag}
                direction_cache[ans] = q_backproj_direction(
                    model, base_diag, repair_diag, sel_layers, heads_selected, heads_by_layer
                )

            correct = case["correct"]
            base_mass = alpha_group_mass(diag_cache[correct]["base"]["alpha"], group_tokens, heads_selected)
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, correct, old_top_wrong),
                "target_case": target_case,
                "patches": {},
            }

            for mode_name, mode_spec in mode_specs.items():
                scores, norm_info = patched_scores(
                    model, tokenizer, device, case, values, mode_spec, comp_cache, direction_cache,
                    seed=si * 1009 + len(mode_name),
                )
                patched = winner_stats(scores, correct)
                diag = patched_diag(
                    model, tokenizer, device, case, correct, mode_spec,
                    comp_cache[correct], direction_cache[correct], sel_layers,
                    seed=si * 1009 + len(mode_name),
                )
                q_m = q_metrics(diag_cache[correct]["base"], diag_cache[correct]["repair"],
                                diag, heads_selected, heads_by_layer)
                mass = alpha_group_mass(diag["alpha"], group_tokens, heads_selected)
                alpha_delta = {g: mass.get(g, 0.0) - base_mass.get(g, 0.0) for g in mass}
                row["patches"][mode_name] = {
                    "mode": mode_name,
                    "winner": patched,
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                    "q_metrics": q_m,
                    "alpha_delta": alpha_delta,
                    "candidate_scores": scores,
                    "score_delta_vs_base": {v: scores[v] - base_scores[v] for v in values},
                    **norm_info,
                }
            rows.append(row)

        return {
            "phase": 623,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "patch_layers": patch_layers,
            "selection_layers": sel_layers,
            "heads_by_layer": heads_by_layer,
            "selected_heads": heads_selected,
            "mode_specs": mode_specs,
            "top_k": args.top_k,
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
    out_path = out_dir / f"phase623_{args.model}_selection_result_state_separation_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
