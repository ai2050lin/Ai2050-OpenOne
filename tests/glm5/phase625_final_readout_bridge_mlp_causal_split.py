#!/usr/bin/env python3
"""
Phase 625: Final Readout Bridge and MLP Causal Split
最终读出桥接与 MLP 因果拆分

Phase 624 showed the result state propagates as a residual carrier and MLP
outputs align with the natural repair trajectory. This phase tests whether that
carrier reaches final norm / lm_head proxies, and whether a downstream MLP
output is better explained by correct-up, wrong-down, span, orthogonal, or random
candidate directions.
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
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import projection_metric, score_map  # noqa: E402
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


OUT_ROOT = Path("results/glm5_phase625_final_readout_bridge_mlp_causal_split")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_mlp_layer(model_name: str, n_layers: int) -> int:
    if model_name == "qwen3":
        return min(31, n_layers - 1)
    if model_name == "glm4":
        return min(39, n_layers - 1)
    if model_name == "deepseek7b":
        return min(23, n_layers - 1)
    return n_layers - 2


def install_patch_hooks(model, tokenizer, prompt: str, answer: str, patches: List[Tuple[int, str, torch.Tensor]]):
    handles = []
    for li, component, target in patches:
        handle = make_patch_hook(model, tokenizer, prompt, answer, li, component, target)
        if handle is not None:
            handles.append(handle)
    return handles


def collect_final_norm_sequence(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    patches: List[Tuple[int, str, torch.Tensor]],
) -> Dict:
    final_norm = get_final_norm(model)
    if final_norm is None:
        raise RuntimeError("final norm not found")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    start = len(prompt_ids) - 1
    captured = {}
    handles = install_patch_hooks(model, tokenizer, prompt, answer, patches)

    def pre_hook(_module, inputs):
        captured["input"] = inputs[0].detach().float().cpu()

    def out_hook(_module, _inputs, output):
        captured["output"] = extract_tensor(output).detach().float().cpu()

    handles.append(final_norm.register_forward_pre_hook(pre_hook))
    handles.append(final_norm.register_forward_hook(out_hook))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()

    out = {"input": [], "output": [], "positions": []}
    for i in range(len(ans_ids)):
        pos = start + i
        if pos >= captured["input"].shape[1]:
            break
        out["input"].append(captured["input"][0, pos].float().cpu())
        out["output"].append(captured["output"][0, pos].float().cpu())
        out["positions"].append(pos)
    return out


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


def mean_tensor(xs: List[torch.Tensor]) -> torch.Tensor:
    if not xs:
        return torch.zeros(1)
    return torch.stack([x.float().cpu() for x in xs]).mean(dim=0)


def final_bridge_metrics(base_seq: Dict, repair_seq: Dict, patched_seq: Dict, E: torch.Tensor,
                         correct: str, old_top_wrong: str, values: List[str]) -> Dict:
    metrics = {}
    for comp in ["input", "output"]:
        b = mean_tensor(base_seq[comp])
        r = mean_tensor(repair_seq[comp])
        p = mean_tensor(patched_seq[comp])
        target = r - b
        got = p - b
        metrics[f"{comp}_repair_projection"] = projection(got, target)
        metrics[f"{comp}_repair_cos"] = safe_cos(got, target)
        metrics[f"{comp}_norm_ratio"] = float(got.norm().item() / target.norm().clamp_min(1e-8).item())
        if comp == "output" and got.numel() == E.shape[1]:
            proj = projection_metric(got, E, correct, old_top_wrong, values)
            metrics["output_projection_margin"] = proj["projection_specific_margin"]
            metrics["output_projection_correct_specific"] = proj["projection_correct_specific"]
            metrics["output_projection_wrong_specific"] = proj["projection_old_top_wrong_specific"]
    return metrics


def normalize_dir(x: torch.Tensor) -> torch.Tensor:
    x = x.float().cpu()
    n = x.norm()
    if float(n.item()) <= 1e-8:
        return torch.zeros_like(x)
    return x / n


def project_to_span(x: torch.Tensor, dirs: List[torch.Tensor]) -> torch.Tensor:
    cols = [normalize_dir(d) for d in dirs if float(d.float().cpu().norm().item()) > 1e-8]
    if not cols:
        return torch.zeros_like(x.float().cpu())
    mat = torch.stack(cols, dim=1)
    q, _r = torch.linalg.qr(mat.float(), mode="reduced")
    return q @ (q.T @ x.float().cpu())


def mlp_split_piece(
    delta: torch.Tensor,
    E: torch.Tensor,
    values: List[str],
    correct: str,
    old_top_wrong: str,
    mode: str,
    seed: int,
) -> torch.Tensor:
    d = delta.float().cpu()
    mean_e = E.mean(dim=0)
    e_correct = E[values.index(correct)] - mean_e
    e_wrong_down = mean_e - E[values.index(old_top_wrong)]
    correct_up = normalize_dir(e_correct) * torch.dot(d, normalize_dir(e_correct))
    wrong_down = normalize_dir(e_wrong_down) * torch.dot(d, normalize_dir(e_wrong_down))
    span = project_to_span(d, [e_correct, e_wrong_down])
    if mode == "full_delta":
        return d
    if mode == "correct_up":
        return correct_up
    if mode == "wrong_down":
        return wrong_down
    if mode == "correct_plus_wrong":
        return correct_up + wrong_down
    if mode == "margin_span":
        return span
    if mode == "orthogonal":
        return d - span
    if mode == "random_same_norm":
        return random_same_norm(d, seed=seed)
    raise ValueError(mode)


def patch_scores_result_mode(
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


def patch_scores_mlp_split(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    mlp_layer: int,
    comp_cache: Dict[str, Dict],
    E: torch.Tensor,
    old_top_wrong: str,
    mode: str,
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        base = comp_cache[ans]["base"].get(mlp_layer, {}).get("mlp_out")
        repair = comp_cache[ans]["repair"].get(mlp_layer, {}).get("mlp_out")
        if base is None or repair is None:
            scores[ans] = -100.0
            continue
        delta = repair.float().cpu() - base.float().cpu()
        piece = mlp_split_piece(delta, E, values, case["correct"], old_top_wrong, mode, seed + ai * 997)
        target = base.float().cpu() + piece
        scores[ans] = patch_score_multi(
            model, tokenizer, device, case["base_prompt"], ans, [(mlp_layer, "mlp_out", target)]
        )
    return scores


def summarize(rows: List[Dict]) -> Dict:
    score_keys = sorted({k for r in rows for k in r["score_modes"]})
    scores = {}
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
        scores[key] = entry

    bridge_keys = sorted({k for r in rows for k in r["final_bridge"]})
    bridge = {}
    for key in bridge_keys:
        items = [r["final_bridge"][key] for r in rows if key in r["final_bridge"]]
        entry = {"mode": key, "n": len(items)}
        fields = sorted({f for item in items for f in item if f != "mode"})
        for field in fields:
            vals = [item.get(field, 0.0) for item in items]
            entry[f"mean_{field}"] = sum(vals) / max(1, len(vals))
        bridge[key] = entry

    best_scores = sorted(scores.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
    log("Best score modes:")
    for item in best_scores[:16]:
        log(f"  {item['mode']}: switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}")
    return {"score_modes": scores, "final_bridge": bridge, "best_scores": best_scores}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        specs = default_specs(args.model)
        mode_specs = make_mode_specs(specs)
        selected_result_modes = ["result_only", "selection_both_plus_result", "result_random_norm"]
        patch_layers = sorted({li for mode in selected_result_modes for li, _comp, _part in mode_specs[mode]})
        patch_layers = [li for li in patch_layers if 0 <= li < info.n_layers]
        mlp_layer = args.mlp_layer if args.mlp_layer >= 0 else default_mlp_layer(args.model, info.n_layers)
        mlp_layer = min(max(0, mlp_layer), info.n_layers - 1)
        scan_layers = sorted(set(patch_layers + [mlp_layer]))
        sel_layers = parse_layers(args.selection_layers) if args.selection_layers else selection_layers(args.model, info.n_layers)
        sel_layers = [li for li in sel_layers if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in sel_layers}
        heads_selected = {li: top_heads(args.model, heads_by_layer[li], args.top_k) for li in sel_layers}
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_value": 0}
        target_seen = 0
        mlp_modes = ["full_delta", "correct_up", "wrong_down", "correct_plus_wrong", "margin_span", "orthogonal", "random_same_norm"]
        log(
            f"{args.model}: patch_layers={patch_layers}, mlp_layer={mlp_layer}, "
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
                "final_bridge": {},
            }

            result_patches, _info = build_patch_targets(
                mode_specs["result_only"], comp_cache[correct], direction_cache[correct],
                seed=si * 1009 + 17,
            )
            bridge_base = collect_final_norm_sequence(model, tokenizer, device, case["base_prompt"], correct, [])
            bridge_repair = collect_final_norm_sequence(model, tokenizer, device, case["repair_prompt"], correct, [])
            for bridge_mode, patches in {
                "result_only": result_patches,
                "none": [],
            }.items():
                patched_seq = collect_final_norm_sequence(
                    model, tokenizer, device, case["base_prompt"], correct, patches
                )
                row["final_bridge"][bridge_mode] = final_bridge_metrics(
                    bridge_base, bridge_repair, patched_seq, E, correct, old_top_wrong, values
                )

            for mode in selected_result_modes:
                scores = patch_scores_result_mode(
                    model, tokenizer, device, case, values, mode_specs[mode], comp_cache, direction_cache,
                    seed=si * 1009 + len(mode),
                )
                row["score_modes"][mode] = {
                    "mode": mode,
                    "winner": winner_stats(scores, correct),
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                }
            for mode in mlp_modes:
                scores = patch_scores_mlp_split(
                    model, tokenizer, device, case, values, mlp_layer, comp_cache, E, old_top_wrong,
                    mode, seed=si * 1009 + len(mode),
                )
                row["score_modes"][f"mlp_{mode}"] = {
                    "mode": f"mlp_{mode}",
                    "winner": winner_stats(scores, correct),
                    "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                }
            rows.append(row)

        return {
            "phase": 625,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "patch_layers": patch_layers,
            "mlp_layer": mlp_layer,
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
    parser.add_argument("--mlp-layer", type=int, default=-1)
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
    out_path = out_dir / f"phase625_{args.model}_final_readout_bridge_mlp_causal_split_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
