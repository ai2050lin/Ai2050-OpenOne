#!/usr/bin/env python3
"""
Phase 598: Downstream Trajectory Acceptance Audit
下游轨迹接收审计

Phase 597 showed that state-conditioned MLP recomputation can generate strong
candidate projection, but final candidate margin does not improve. This phase
tracks whether the injected/generated signal survives in downstream hidden
states.
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

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase595_mlp_update_causal_validation import load_phase594_mlp_nodes  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import (  # noqa: E402
    collect_mlp_input_output,
    get_mlp,
    mlp_parts_from_input,
    patched_score_map,
    projection_metric,
    replace_input,
    score_map,
)


OUT_ROOT = Path("results/glm5_phase598_downstream_trajectory_acceptance_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_nodes(model: str, top_k: int) -> List[Dict]:
    nodes = load_phase594_mlp_nodes(model, max(top_k, 3))
    if model == "deepseek7b":
        priority = [("rule_value", 26), ("prompt_last", 26), ("query_relation", 19)]
        selected = []
        by_key = {(n["position"], n["layer"]): n for n in nodes}
        for key in priority:
            if key in by_key:
                selected.append(by_key[key])
        for n in nodes:
            if n not in selected:
                selected.append(n)
        return selected[:top_k]
    return nodes[:top_k]


def hidden_projection(effect: torch.Tensor, E: torch.Tensor, correct: str, old_top_wrong: str,
                      values: List[str]) -> Dict:
    return projection_metric(effect.float().cpu(), E, correct, old_top_wrong, values)


def collect_hidden_with_mlp_input_patch(model, tokenizer, device, prompt: str,
                                        layer_idx: Optional[int] = None,
                                        patch_pos: Optional[int] = None,
                                        target_input: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    handle = None
    if layer_idx is not None and patch_pos is not None and target_input is not None:
        mlp = get_mlp(get_layers(model)[layer_idx])
        target = target_input.to(device=device)

        def pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, patch_pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)

        handle = mlp.register_forward_pre_hook(pre_hook)
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        return [h[0].detach().float().cpu() for h in out.hidden_states]
    finally:
        if handle is not None:
            handle.remove()


def build_targets(base_x: torch.Tensor, repair_x: torch.Tensor, wrong_x: Optional[torch.Tensor],
                  alpha: float, seed: int) -> List[Dict]:
    d_repair = repair_x.float().cpu() - base_x.float().cpu()
    specs = [
        {
            "name": f"repair_alpha{alpha:g}",
            "kind": "repair",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * d_repair,
        },
        {
            "name": f"random_alpha{alpha:g}",
            "kind": "random",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * random_same_norm(d_repair, seed=seed),
        },
    ]
    if wrong_x is not None:
        specs.append({
            "name": f"wrong_alpha{alpha:g}",
            "kind": "wrong",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * (wrong_x.float().cpu() - base_x.float().cpu()),
        })
    return specs


def trace_layers(layer_idx: int, n_layers: int, window: int) -> List[int]:
    start = layer_idx + 1
    stop = min(n_layers, layer_idx + 1 + window)
    layers = list(range(start, stop + 1))
    if n_layers not in layers:
        layers.append(n_layers)
    return sorted(set(layers))


def summarize(rows: List[Dict]) -> Dict:
    patch_keys = sorted({k for r in rows for k in r["patches"]})
    patch_by_key = {}
    for key in patch_keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "kind": items[0]["kind"],
            "alpha": items[0]["alpha"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_specific_margin_gain": 0.0,
            "mean_common_delta": 0.0,
            "positive_margin": 0,
            "mean_generated_down_projection": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_generated_down_projection"] += item["generated_down_metric"]["projection_specific_margin"]
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_specific_margin_gain"] += m["specific_margin_gain"]
            entry["mean_common_delta"] += m["common_delta"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        patch_by_key[key] = entry

    traj_keys = sorted({k for r in rows for k in r["trajectories"]})
    traj_by_key = {}
    for key in traj_keys:
        items = [r["trajectories"][key] for r in rows if key in r["trajectories"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "patch_layer": items[0]["node"]["layer"],
            "hidden_index": items[0]["hidden_index"],
            "kind": items[0]["kind"],
            "alpha": items[0]["alpha"],
            "n": len(items),
            "mean_projection_specific_margin": 0.0,
            "mean_projection_correct_specific": 0.0,
            "mean_projection_old_top_wrong_specific": 0.0,
            "positive_projection_margin": 0,
        }
        for item in items:
            m = item["metric"]
            entry["mean_projection_specific_margin"] += m["projection_specific_margin"]
            entry["mean_projection_correct_specific"] += m["projection_correct_specific"]
            entry["mean_projection_old_top_wrong_specific"] += m["projection_old_top_wrong_specific"]
            entry["positive_projection_margin"] += int(m["projection_specific_margin"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["positive_projection_rate"] = entry["positive_projection_margin"] / n
        traj_by_key[key] = entry

    best_patches = sorted(
        patch_by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_generated_down_projection"]),
        reverse=True,
    )[:36]
    best_traj = sorted(
        traj_by_key.values(),
        key=lambda x: (x["mean_projection_specific_margin"], x["positive_projection_rate"]),
        reverse=True,
    )[:48]
    log("Best final patch effects:")
    for item in best_patches[:12]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"mgain={item['mean_margin_gain']:.3f}, gen={item['mean_generated_down_projection']:.3f}"
        )
    log("Best downstream hidden projections:")
    for item in best_traj[:10]:
        log(
            f"  {item['key']}: hidx={item['hidden_index']}, "
            f"proj={item['mean_projection_specific_margin']:.3f}, rate={item['positive_projection_rate']:.3f}"
        )
    return {
        "patch_by_key": patch_by_key,
        "trajectory_by_key": traj_by_key,
        "best_patches": best_patches,
        "best_trajectories": best_traj,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = select_nodes(args.model, args.top_nodes)
        node_layers = sorted({n["layer"] for n in nodes})
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, alpha={args.alpha}")

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
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"]) if args.include_wrong_controls else {}
            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], node_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], node_layers)
            wrong_cap = collect_mlp_input_output(model, tokenizer, device, case["wrong_prompt"], node_layers) if args.include_wrong_controls else None
            base_hidden = collect_hidden_with_mlp_input_patch(model, tokenizer, device, case["base_prompt"])

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "patches": {},
                "trajectories": {},
            }
            for node in nodes:
                pos_name = node["position"]
                li = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name) if args.include_wrong_controls else None
                if bp is None or rp is None:
                    continue
                if li not in base_cap["mlp_input"] or li not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][li].shape[1] or rp >= repair_cap["mlp_input"][li].shape[1]:
                    continue
                base_x = base_cap["mlp_input"][li][0, bp]
                repair_x = repair_cap["mlp_input"][li][0, rp]
                wrong_x = None
                if wrong_cap is not None and wp is not None and li in wrong_cap["mlp_input"] and wp < wrong_cap["mlp_input"][li].shape[1]:
                    wrong_x = wrong_cap["mlp_input"][li][0, wp]
                specs = build_targets(base_x, repair_x, wrong_x, args.alpha, seed=si * 1009 + li)
                mlp = get_mlp(get_layers(model)[li])
                base_down = mlp_parts_from_input(mlp, base_x)["down"]
                for spec in specs:
                    key = f"{pos_name}|L{li}|{spec['name']}"
                    generated_down = mlp_parts_from_input(mlp, spec["target"])["down"] - base_down
                    gen_metric = projection_metric(generated_down, E, correct, old_top_wrong, values)
                    scores = patched_score_map(model, tokenizer, device, case["base_prompt"], values, li, bp, spec["target"])
                    patched = winner_stats(scores, correct)
                    row["patches"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "kind": spec["kind"],
                        "alpha": spec["alpha"],
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                        "generated_down_metric": gen_metric,
                    }

                    patched_hidden = collect_hidden_with_mlp_input_patch(
                        model, tokenizer, device, case["base_prompt"], li, bp, spec["target"]
                    )
                    for hidx in trace_layers(li, info.n_layers, args.window):
                        if hidx >= len(base_hidden) or hidx >= len(patched_hidden):
                            continue
                        if bp >= base_hidden[hidx].shape[0] or bp >= patched_hidden[hidx].shape[0]:
                            continue
                        effect = patched_hidden[hidx][bp] - base_hidden[hidx][bp]
                        tkey = f"{pos_name}|L{li}|{spec['name']}|H{hidx}"
                        row["trajectories"][tkey] = {
                            "node": {"position": pos_name, "layer": li},
                            "kind": spec["kind"],
                            "alpha": spec["alpha"],
                            "hidden_index": hidx,
                            "metric": hidden_projection(effect, E, correct, old_top_wrong, values),
                        }
            rows.append(row)

        return {
            "phase": 598,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "window": args.window,
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
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--include-wrong-controls", action="store_true", default=True)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_nodes = min(args.top_nodes, 2)
        args.window = min(args.window, 2)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_samples = max(args.max_samples, 64)
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase598_{args.model}_downstream_trajectory_acceptance_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
