#!/usr/bin/env python3
"""
Phase 597: State-Conditioned MLP Generation Audit
状态条件化 MLP 生成审计

Phase 596 showed that internal MLP activations have strong projection but do not
transfer as static patches. This phase patches the MLP input state itself and
lets the MLP recompute gate/up/z/down outputs.
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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase595_mlp_update_causal_validation import load_phase594_mlp_nodes  # noqa: E402
from phase596_mlp_internal_gate_path_audit import mlp_act, split_gate_up  # noqa: E402


OUT_ROOT = Path("results/glm5_phase597_state_conditioned_mlp_generation_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: List[str]) -> Dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def extract_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_input(inputs, tensor):
    return (tensor,) + tuple(inputs[1:])


def get_mlp(layer):
    return getattr(layer, "mlp", None)


def linear_cpu(module, x: torch.Tensor) -> torch.Tensor:
    weight = module.weight.detach().float().cpu()
    bias = module.bias.detach().float().cpu() if getattr(module, "bias", None) is not None else None
    return F.linear(x.float().cpu(), weight, bias)


def mlp_parts_from_input(mlp, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    x = x.float().cpu()
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
        gate = linear_cpu(mlp.gate_proj, x)
        up = linear_cpu(mlp.up_proj, x)
    elif hasattr(mlp, "gate_up_proj"):
        fused = linear_cpu(mlp.gate_up_proj, x)
        gate, up = split_gate_up(fused)
    else:
        raise RuntimeError("Unsupported MLP projection structure")
    z = mlp_act(mlp, gate) * up
    down = linear_cpu(mlp.down_proj, z)
    return {"gate": gate, "up": up, "z": z, "down": down}


def collect_mlp_input_output(model, tokenizer, device, prompt: str, layer_indices: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
    layers = get_layers(model)
    captured: Dict[str, Dict[int, torch.Tensor]] = {"mlp_input": {}, "mlp_output": {}}
    hooks = []
    for li in layer_indices:
        mlp = get_mlp(layers[li])
        if mlp is None:
            continue

        def make_pre(layer_idx):
            def hook(_module, inputs):
                captured["mlp_input"][layer_idx] = inputs[0].detach().float().cpu()
            return hook

        def make_out(layer_idx):
            def hook(_module, _inputs, output):
                captured["mlp_output"][layer_idx] = extract_tensor(output).detach().float().cpu()
            return hook

        hooks.append(mlp.register_forward_pre_hook(make_pre(li)))
        hooks.append(mlp.register_forward_hook(make_out(li)))
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            model(input_ids=input_ids, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    return captured


def projection_metric(effect: torch.Tensor, E: torch.Tensor, correct: str, old_top_wrong: str,
                      values: List[str]) -> Dict:
    scores = E @ effect.float().cpu()
    common = float(scores.mean())
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    cs = float(scores[ci] - common)
    ws = float(scores[wi] - common)
    return {
        "projection_common": common,
        "projection_correct_specific": cs,
        "projection_old_top_wrong_specific": ws,
        "projection_specific_margin": cs - ws,
    }


def build_state_targets(base_x: torch.Tensor, repair_x: torch.Tensor, wrong_x: Optional[torch.Tensor],
                        alphas: List[float], sample_seed: int) -> List[Dict]:
    d_repair = repair_x.float().cpu() - base_x.float().cpu()
    rand = random_same_norm(d_repair, seed=sample_seed)
    specs = []
    for alpha in alphas:
        specs.append({
            "name": f"repair_alpha{alpha:g}",
            "kind": "repair",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * d_repair,
        })
        if wrong_x is not None:
            specs.append({
                "name": f"wrong_alpha{alpha:g}",
                "kind": "wrong",
                "alpha": alpha,
                "target": base_x.float().cpu() + alpha * (wrong_x.float().cpu() - base_x.float().cpu()),
            })
        specs.append({
            "name": f"random_alpha{alpha:g}",
            "kind": "random",
            "alpha": alpha,
            "target": base_x.float().cpu() + alpha * rand,
        })
    return specs


def patch_full_logprob_mlp_input(model, tokenizer, device, prompt: str, answer: str,
                                 layer_idx: int, patch_pos: int,
                                 target_input: torch.Tensor, alpha_unused: float = 1.0) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0
    all_ids = prompt_ids + answer_ids
    mlp = get_mlp(get_layers(model)[layer_idx])
    target = target_input.to(device=device)

    def pre_hook(_module, inputs):
        x = inputs[0]
        x_new = x.clone()
        x_new[0, patch_pos, :] = target.to(dtype=x_new.dtype)
        return replace_input(inputs, x_new)

    handle = mlp.register_forward_pre_hook(pre_hook)
    try:
        total = 0.0
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        handle.remove()


def patched_score_map(model, tokenizer, device, prompt: str, candidates: List[str],
                      layer_idx: int, patch_pos: int, target_input: torch.Tensor) -> Dict[str, float]:
    return {
        ans: patch_full_logprob_mlp_input(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, target_input)
        for ans in candidates
    }


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
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_specific_margin_gain"] += m["specific_margin_gain"]
            entry["mean_common_delta"] += m["common_delta"]
            entry["mean_correct_specific"] += m["correct_specific"]
            entry["mean_old_top_wrong_specific"] += m["old_top_wrong_specific"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        patch_by_key[key] = entry

    proj_keys = sorted({k for r in rows for k in r["projections"]})
    proj_by_key = {}
    for key in proj_keys:
        items = [r["projections"][key] for r in rows if key in r["projections"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "source": items[0]["source"],
            "kind": items[0]["kind"],
            "alpha": items[0]["alpha"],
            "n": len(items),
            "mean_projection_common": 0.0,
            "mean_projection_correct_specific": 0.0,
            "mean_projection_old_top_wrong_specific": 0.0,
            "mean_projection_specific_margin": 0.0,
            "positive_projection_margin": 0,
        }
        for item in items:
            m = item["metric"]
            for name in [
                "projection_common",
                "projection_correct_specific",
                "projection_old_top_wrong_specific",
                "projection_specific_margin",
            ]:
                entry[f"mean_{name}"] += m[name]
            entry["positive_projection_margin"] += int(m["projection_specific_margin"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["positive_projection_rate"] = entry["positive_projection_margin"] / n
        proj_by_key[key] = entry

    best_patches = sorted(
        patch_by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_specific_margin_gain"]),
        reverse=True,
    )[:36]
    best_proj = sorted(
        proj_by_key.values(),
        key=lambda x: (x["mean_projection_specific_margin"], x["positive_projection_rate"]),
        reverse=True,
    )[:36]
    log("Best causal state patches:")
    for item in best_patches[:12]:
        log(
            f"  {item['key']}: switch={item['switch']}/{item['n']}, "
            f"mgain={item['mean_margin_gain']:.3f}, spec={item['mean_specific_margin_gain']:.3f}, "
            f"common={item['mean_common_delta']:.3f}"
        )
    log("Best generated projections:")
    for item in best_proj[:8]:
        log(
            f"  {item['key']}: proj_spec={item['mean_projection_specific_margin']:.3f}, "
            f"rate={item['positive_projection_rate']:.3f}"
        )
    return {
        "patch_by_key": patch_by_key,
        "projection_by_key": proj_by_key,
        "best_patches": best_patches,
        "best_projections": best_proj,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase594_mlp_nodes(args.model, args.top_nodes)
        if not nodes:
            raise RuntimeError(f"No Phase594 MLP nodes found for {args.model}")
        node_layers = sorted({n["layer"] for n in nodes})
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        layers = get_layers(model)
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, alphas={alphas}")

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

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "projections": {},
                "patches": {},
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
                mlp = get_mlp(layers[li])
                base_down = mlp_parts_from_input(mlp, base_x)["down"]
                specs = build_state_targets(base_x, repair_x, wrong_x, alphas, sample_seed=si * 1009 + li)
                for spec in specs:
                    parts = mlp_parts_from_input(mlp, spec["target"])
                    down_effect = parts["down"] - base_down
                    pkey = f"{pos_name}|L{li}|{spec['name']}|generated_down"
                    row["projections"][pkey] = {
                        "node": {"position": pos_name, "layer": li},
                        "source": "generated_down",
                        "kind": spec["kind"],
                        "alpha": spec["alpha"],
                        "metric": projection_metric(down_effect, E, correct, old_top_wrong, values),
                    }
                    scores = patched_score_map(model, tokenizer, device, case["base_prompt"], values, li, bp, spec["target"])
                    patched = winner_stats(scores, correct)
                    key = f"{pos_name}|L{li}|{spec['name']}"
                    row["patches"][key] = {
                        "node": {"position": pos_name, "layer": li},
                        "kind": spec["kind"],
                        "alpha": spec["alpha"],
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 597,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alphas": alphas,
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
    parser.add_argument("--alphas", default="0.25,0.5,1.0,1.5,2.0")
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
        args.alphas = "0.5,1.0"
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
    out_path = out_dir / f"phase597_{args.model}_state_conditioned_mlp_generation_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
