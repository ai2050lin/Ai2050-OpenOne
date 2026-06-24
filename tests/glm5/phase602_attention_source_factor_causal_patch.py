#!/usr/bin/env python3
"""
Phase 602: Attention-Source Factor Causal Patch
注意力源因子因果修补

Phase 601 found source-resolved final attention differences between natural
correct and artificial repair trajectories. This phase tests whether adding the
natural final attention-output effect helps MLP-input repair enter readout.
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
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import (  # noqa: E402
    collect_mlp_input_output,
    get_mlp,
    mlp_parts_from_input,
    projection_metric,
    replace_input,
    score_map,
)
from phase598_downstream_trajectory_acceptance_audit import select_nodes  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm  # noqa: E402
from phase600_final_layer_acceptance_rule_audit import collect_final_block, cosine, effect_metric, get_position  # noqa: E402


OUT_ROOT = Path("results/glm5_phase602_attention_source_factor_causal_patch")
COMPONENTS = ["attn_out", "layer_out", "final_norm_input", "final_norm_output"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def random_like(vec: torch.Tensor, seed: int) -> torch.Tensor:
    return random_same_norm(vec.float().cpu(), seed=seed)


def patch_answer_logprob(model, tokenizer, device, prompt: str, answer: str,
                         source_layer: Optional[int], source_pos: Optional[int],
                         target_input: Optional[torch.Tensor],
                         probe_layer: int,
                         attn_pos: Optional[int],
                         attn_delta: Optional[torch.Tensor],
                         attn_scale: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return -100.0
    all_ids = prompt_ids + answer_ids
    layers = get_layers(model)
    handles = []

    if source_layer is not None and source_pos is not None and target_input is not None:
        if source_pos < 0 or source_pos >= len(prompt_ids):
            return -100.0
        mlp = get_mlp(layers[source_layer])
        target = target_input.to(device=device)

        def mlp_pre(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, source_pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)

        handles.append(mlp.register_forward_pre_hook(mlp_pre))

    if attn_delta is not None and attn_pos is not None:
        if attn_pos < 0 or attn_pos >= len(prompt_ids):
            return -100.0
        attn = get_attn(layers[probe_layer])
        delta = (attn_scale * attn_delta.float().cpu()).to(device=device)

        def attn_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, attn_pos, :] = y_new[0, attn_pos, :] + delta.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handles.append(attn.register_forward_hook(attn_hook))

    try:
        total = 0.0
        with torch.inference_mode():
            input_ids = torch.tensor([all_ids], device=device)
            out = model(input_ids=input_ids, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def patched_scores(model, tokenizer, device, prompt: str, values: List[str],
                   source_layer: Optional[int], source_pos: Optional[int],
                   target_input: Optional[torch.Tensor],
                   probe_layer: int,
                   attn_pos: Optional[int],
                   attn_delta: Optional[torch.Tensor],
                   attn_scale: float) -> Dict[str, float]:
    return {
        ans: patch_answer_logprob(
            model, tokenizer, device, prompt, ans,
            source_layer, source_pos, target_input,
            probe_layer, attn_pos, attn_delta, attn_scale,
        )
        for ans in values
    }


def collect_patched_final(model, tokenizer, device, prompt: str, probe_layer: int,
                          source_layer: Optional[int], source_pos: Optional[int],
                          target_input: Optional[torch.Tensor],
                          attn_pos: Optional[int], attn_delta: Optional[torch.Tensor],
                          attn_scale: float) -> Dict:
    layers = get_layers(model)
    attn = get_attn(layers[probe_layer])
    handles = []
    if attn_delta is not None and attn_pos is not None:
        delta = (attn_scale * attn_delta.float().cpu()).to(device=device)

        def attn_hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, attn_pos, :] = y_new[0, attn_pos, :] + delta.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handles.append(attn.register_forward_hook(attn_hook))
    try:
        return collect_final_block(
            model, tokenizer, device, prompt, probe_layer,
            source_layer=source_layer,
            patch_pos=source_pos,
            target_input=target_input,
            capture_attn=False,
        )
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
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "probe_layer": items[0]["node"]["probe_layer"],
            "n": len(items),
            "switch": 0,
            "mean_full_margin_gain": 0.0,
            "mean_generated_down_projection": 0.0,
            "mean_attn_delta_projection": 0.0,
            "mean_final_norm_projection": 0.0,
            "mean_final_norm_cos_to_natural": 0.0,
            "positive_full_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_full_margin_gain"] += item["full_metric"]["margin_gain"]
            entry["mean_generated_down_projection"] += item.get("generated_down_metric", {}).get("projection_specific_margin", 0.0)
            entry["mean_attn_delta_projection"] += item.get("attn_delta_metric", {}).get("projection_specific_margin", 0.0)
            fm = item.get("final_norm_metric", {})
            entry["mean_final_norm_projection"] += fm.get("projection_specific_margin", 0.0)
            entry["mean_final_norm_cos_to_natural"] += fm.get("cos_to_natural_correct", 0.0)
            entry["positive_full_margin"] += int(item["full_metric"]["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_full_margin_rate"] = entry["positive_full_margin"] / n
        by_patch[key] = entry
    best = sorted(
        by_patch.values(),
        key=lambda x: (x["switch"], x["mean_full_margin_gain"], x["mean_final_norm_cos_to_natural"]),
        reverse=True,
    )
    log("Best causal patch effects:")
    for item in best[:12]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"full={item['mean_full_margin_gain']:.3f}, cos_final={item['mean_final_norm_cos_to_natural']:.3f}"
        )
    return {"by_patch": by_patch, "best": best[:60]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = select_nodes(args.model, args.top_nodes)
        source_layers = sorted({n["layer"] for n in nodes})
        probe_layer = info.n_layers - 1
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, probe=L{probe_layer}")

        rows = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], source_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], source_layers)
            base_decomp = collect_final_block(model, tokenizer, device, case["base_prompt"], probe_layer, capture_attn=False)
            repair_decomp = collect_final_block(model, tokenizer, device, case["repair_prompt"], probe_layer, capture_attn=False)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "patches": {},
            }

            for node in nodes:
                pos_name = node["position"]
                source_layer = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                if bp is None or rp is None:
                    continue
                if source_layer not in base_cap["mlp_input"] or source_layer not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][source_layer].shape[1] or rp >= repair_cap["mlp_input"][source_layer].shape[1]:
                    continue

                base_x = base_cap["mlp_input"][source_layer][0, bp]
                repair_x = repair_cap["mlp_input"][source_layer][0, rp]
                d_repair = repair_x.float().cpu() - base_x.float().cpu()
                repair_target = base_x.float().cpu() + args.alpha * d_repair
                random_target = base_x.float().cpu() + args.alpha * random_same_norm(d_repair, seed=si * 1009 + source_layer)
                source_mlp = get_mlp(get_layers(model)[source_layer])
                generated_down = mlp_parts_from_input(source_mlp, repair_target)["down"] - mlp_parts_from_input(source_mlp, base_x)["down"]
                gen_metric = projection_metric(generated_down, E, correct, old_top_wrong, values)

                b_attn = get_position(base_decomp, "attn_out", bp)
                r_attn = get_position(repair_decomp, "attn_out", rp)
                b_final = get_position(base_decomp, "final_norm_output", bp)
                r_final = get_position(repair_decomp, "final_norm_output", rp)
                natural_final = (r_final - b_final) if b_final is not None and r_final is not None else None
                if b_attn is None or r_attn is None:
                    continue
                natural_attn_delta = (r_attn - b_attn).float().cpu()
                random_attn_delta = random_like(natural_attn_delta, seed=si * 917 + source_layer)
                attn_metric = projection_metric(natural_attn_delta, E, correct, old_top_wrong, values)
                random_attn_metric = projection_metric(random_attn_delta, E, correct, old_top_wrong, values)

                modes = [
                    {
                        "name": "mlp_repair_only",
                        "source_target": repair_target,
                        "attn_delta": None,
                        "attn_metric": {},
                    },
                    {
                        "name": "attn_effect_only",
                        "source_target": None,
                        "attn_delta": natural_attn_delta,
                        "attn_metric": attn_metric,
                    },
                    {
                        "name": "mlp_plus_attn_effect",
                        "source_target": repair_target,
                        "attn_delta": natural_attn_delta,
                        "attn_metric": attn_metric,
                    },
                    {
                        "name": "attn_random",
                        "source_target": None,
                        "attn_delta": random_attn_delta,
                        "attn_metric": random_attn_metric,
                    },
                    {
                        "name": "mlp_plus_attn_random",
                        "source_target": repair_target,
                        "attn_delta": random_attn_delta,
                        "attn_metric": random_attn_metric,
                    },
                    {
                        "name": "mlp_random_plus_attn_effect",
                        "source_target": random_target,
                        "attn_delta": natural_attn_delta,
                        "attn_metric": attn_metric,
                    },
                ]

                for mode in modes:
                    source_target = mode["source_target"]
                    use_source_layer = source_layer if source_target is not None else None
                    use_source_pos = bp if source_target is not None else None
                    scores = patched_scores(
                        model, tokenizer, device, case["base_prompt"], values,
                        use_source_layer, use_source_pos, source_target,
                        probe_layer, bp, mode["attn_delta"], args.attn_scale,
                    )
                    patched = winner_stats(scores, correct)
                    pdecomp = collect_patched_final(
                        model, tokenizer, device, case["base_prompt"], probe_layer,
                        use_source_layer, use_source_pos, source_target,
                        bp, mode["attn_delta"], args.attn_scale,
                    )
                    p_final = get_position(pdecomp, "final_norm_output", bp)
                    base_final = get_position(base_decomp, "final_norm_output", bp)
                    final_metric = {}
                    if p_final is not None and base_final is not None:
                        final_metric = effect_metric(p_final - base_final, natural_final, E, correct, old_top_wrong, values)
                    key = f"{pos_name}|L{source_layer}|{mode['name']}"
                    row["patches"][key] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "mode": mode["name"],
                        "winner": patched,
                        "full_metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                        "generated_down_metric": gen_metric if source_target is repair_target else {},
                        "attn_delta_metric": mode["attn_metric"],
                        "final_norm_metric": final_metric,
                    }
            rows.append(row)

        return {
            "phase": 602,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "attn_scale": args.attn_scale,
            "nodes": nodes,
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
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--attn-scale", type=float, default=1.0)
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
        args.top_nodes = min(args.top_nodes, 2)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 96)
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase602_{args.model}_attention_source_factor_causal_patch_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
