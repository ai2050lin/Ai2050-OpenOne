#!/usr/bin/env python3
"""
Phase 624: Result State Downstream Propagation Atlas
结果态下游传播图谱

Phase 623 showed the downstream Q-orthogonal result state can recover candidate
scores without restoring Q/alpha. This phase tracks where that state is retained,
amplified, or washed out after patching.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import n_heads_for, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import COMPONENTS, collect_components  # noqa: E402
from phase618_attention_source_pattern_content import full_ids  # noqa: E402
from phase619_rule_line_token_micro_atlas import rule_micro_groups  # noqa: E402
from phase620_value_token_selection_cause_audit import top_heads  # noqa: E402
from phase621_q_state_builder_backtrace import collect_q_alpha, make_patch_hook, selection_layers  # noqa: E402
from phase622_residual_state_direction_decomposition import q_backproj_direction  # noqa: E402
from phase623_selection_result_state_separation import (  # noqa: E402
    build_patch_targets,
    default_specs,
    make_mode_specs,
    patch_score_multi,
)


OUT_ROOT = Path("results/glm5_phase624_result_state_downstream_propagation_atlas")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_downstream_layers(model_name: str, n_layers: int) -> List[int]:
    if model_name == "qwen3":
        return [li for li in range(29, min(n_layers, 36))]
    if model_name == "glm4":
        return [li for li in range(34, min(n_layers, 40))]
    if model_name == "deepseek7b":
        return [li for li in range(22, min(n_layers, 28))]
    return list(range(max(0, n_layers - 7), n_layers))


def install_patch_hooks(model, tokenizer, prompt: str, answer: str, patches: List[Tuple[int, str, torch.Tensor]]):
    handles = []
    for li, component, target in patches:
        handle = make_patch_hook(model, tokenizer, prompt, answer, li, component, target)
        if handle is not None:
            handles.append(handle)
    return handles


def collect_components_with_patches(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layers_to_scan: List[int],
    patches: List[Tuple[int, str, torch.Tensor]],
) -> Dict[int, Dict]:
    handles = install_patch_hooks(model, tokenizer, prompt, answer, patches)
    try:
        return collect_components(model, tokenizer, device, prompt, answer, layers_to_scan)
    finally:
        for h in handles:
            h.remove()


def safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu()
    b = b.float().cpu()
    if float(a.norm().item()) <= 1e-8 or float(b.norm().item()) <= 1e-8:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0).item())


def projection(got: torch.Tensor, target: torch.Tensor) -> float:
    got = got.float().cpu()
    target = target.float().cpu()
    denom = float(torch.dot(target, target).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(got, target).item() / denom)


def vector_metrics(base: torch.Tensor, repair: torch.Tensor, patched: torch.Tensor, seed_vec: torch.Tensor) -> Dict[str, float]:
    target = repair.float().cpu() - base.float().cpu()
    got = patched.float().cpu() - base.float().cpu()
    seed = seed_vec.float().cpu()
    target_norm = float(target.norm().item())
    got_norm = float(got.norm().item())
    seed_norm = float(seed.norm().item())
    return {
        "repair_projection": projection(got, target),
        "repair_cos": safe_cos(got, target),
        "repair_norm_ratio": got_norm / target_norm if target_norm > 1e-8 else 0.0,
        "seed_projection": projection(got, seed) if seed_norm > 1e-8 else 0.0,
        "seed_cos": safe_cos(got, seed) if seed_norm > 1e-8 else 0.0,
        "target_norm": target_norm,
        "got_norm": got_norm,
    }


def patch_scores_multi(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    mode_spec: List[Tuple[int, str, str]],
    comp_cache: Dict[str, Dict],
    direction_cache: Dict[str, torch.Tensor],
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        patches, _info = build_patch_targets(
            mode_spec, comp_cache[ans], direction_cache[ans], seed=seed + ai * 997
        )
        scores[ans] = patch_score_multi(model, tokenizer, device, case["base_prompt"], ans, patches)
    return scores


def summarize(rows: List[Dict]) -> Dict:
    modes = sorted({m for r in rows for m in r["score_modes"]})
    score_summary = {}
    for mode in modes:
        items = [r["score_modes"][mode] for r in rows if mode in r["score_modes"]]
        entry = {
            "mode": mode,
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
        for key in list(entry):
            if key.startswith("mean_"):
                entry[key] /= n
        entry["switch_rate"] = entry["switch"] / n
        score_summary[mode] = entry

    prop_keys = sorted({k for r in rows for k in r["propagation"]})
    propagation = {}
    for key in prop_keys:
        items = [r["propagation"][key] for r in rows if key in r["propagation"]]
        first = items[0]
        entry = {
            "key": key,
            "mode": first["mode"],
            "layer": first["layer"],
            "component": first["component"],
            "n": len(items),
            "mean_repair_projection": 0.0,
            "mean_repair_cos": 0.0,
            "mean_repair_norm_ratio": 0.0,
            "mean_seed_projection": 0.0,
            "mean_seed_cos": 0.0,
            "mean_target_norm": 0.0,
            "mean_got_norm": 0.0,
        }
        for item in items:
            for name in [
                "repair_projection",
                "repair_cos",
                "repair_norm_ratio",
                "seed_projection",
                "seed_cos",
                "target_norm",
                "got_norm",
            ]:
                entry[f"mean_{name}"] += item[name]
        n = max(1, len(items))
        for key2 in list(entry):
            if key2.startswith("mean_"):
                entry[key2] /= n
        propagation[key] = entry

    best_prop = sorted(
        propagation.values(),
        key=lambda x: (x["mean_repair_projection"], x["mean_seed_projection"], x["mean_repair_cos"]),
        reverse=True,
    )
    log("Best propagation nodes:")
    for item in best_prop[:24]:
        log(
            f"  {item['mode']} L{item['layer']} {item['component']}: "
            f"repair_proj={item['mean_repair_projection']:.3f} "
            f"seed_proj={item['mean_seed_projection']:.3f} "
            f"cos={item['mean_repair_cos']:.3f}"
        )
    return {"score_modes": score_summary, "propagation": propagation, "best_propagation": best_prop[:240]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        specs = default_specs(args.model)
        mode_specs = make_mode_specs(specs)
        selected_modes = [
            "result_only",
            "selection_both",
            "selection_both_plus_result",
            "result_random_norm",
        ]
        patch_layers = sorted({li for mode in selected_modes for li, _comp, _part in mode_specs[mode]})
        patch_layers = [li for li in patch_layers if 0 <= li < info.n_layers]
        downstream_layers = parse_layers(args.downstream_layers) if args.downstream_layers else default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        scan_layers = sorted(set(patch_layers + downstream_layers))
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
            f"{args.model}: patch_layers={patch_layers}, downstream={downstream_layers}, "
            f"sel_layers={sel_layers}, heads={heads_selected}, raw_cases={len(raw_cases)}"
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
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, scan_layers),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, scan_layers),
                }
                base_diag = collect_q_alpha(model, tokenizer, device, case["base_prompt"], ans, sel_layers)
                repair_diag = collect_q_alpha(model, tokenizer, device, case["repair_prompt"], ans, sel_layers)
                diag_cache[ans] = {"base": base_diag, "repair": repair_diag}
                direction_cache[ans] = q_backproj_direction(
                    model, base_diag, repair_diag, sel_layers, heads_selected, heads_by_layer
                )

            correct = case["correct"]
            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base": base,
                "repair_prompt": repair,
                "target_case": target_case,
                "score_modes": {},
                "propagation": {},
            }

            patch_sets = {}
            seed_vec = torch.zeros_like(next(iter(comp_cache[correct]["base"][patch_layers[-1]].values())))
            for mode in selected_modes:
                mode_spec = mode_specs[mode]
                scores = patch_scores_multi(
                    model, tokenizer, device, case, values, mode_spec, comp_cache, direction_cache,
                    seed=si * 1009 + len(mode),
                )
                patched = winner_stats(scores, correct)
                row["score_modes"][mode] = {
                    "mode": mode,
                    "winner": patched,
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                    "candidate_scores": scores,
                    "score_delta_vs_base": {v: scores[v] - base_scores[v] for v in values},
                }
                patches, _info = build_patch_targets(
                    mode_spec, comp_cache[correct], direction_cache[correct],
                    seed=si * 1009 + len(mode),
                )
                patch_sets[mode] = patches
                if mode == "result_only" and patches:
                    base_vec = comp_cache[correct]["base"][patches[0][0]][patches[0][1]]
                    seed_vec = patches[0][2].float().cpu() - base_vec.float().cpu()

            for mode in selected_modes:
                patched_comps = collect_components_with_patches(
                    model, tokenizer, device, case["base_prompt"], correct, scan_layers, patch_sets[mode]
                )
                for li in downstream_layers:
                    for comp in COMPONENTS:
                        base_vec = comp_cache[correct]["base"].get(li, {}).get(comp)
                        repair_vec = comp_cache[correct]["repair"].get(li, {}).get(comp)
                        patch_vec = patched_comps.get(li, {}).get(comp)
                        if base_vec is None or repair_vec is None or patch_vec is None:
                            continue
                        metrics = vector_metrics(base_vec, repair_vec, patch_vec, seed_vec)
                        key = f"{mode}|L{li}|{comp}"
                        row["propagation"][key] = {
                            "mode": mode,
                            "layer": li,
                            "component": comp,
                            **metrics,
                        }
            rows.append(row)

        return {
            "phase": 624,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "patch_layers": patch_layers,
            "downstream_layers": downstream_layers,
            "selection_layers": sel_layers,
            "heads_by_layer": heads_by_layer,
            "selected_heads": heads_selected,
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
    out_path = out_dir / f"phase624_{args.model}_result_state_downstream_propagation_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
