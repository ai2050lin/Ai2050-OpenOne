#!/usr/bin/env python3
"""
Phase 616: Residual State Cumulative Patch and Boundary Localization
残差状态累积修补与形成边界定位

Phase 615 found that answer-position layer_input/layer_out contains a much
stronger repair state than any single attn_out/mlp_out component. This phase
tests whether multiple weak component deltas can accumulate into the full
repair state, or whether the state package still requires a hidden/mixed path.
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

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, parse_layers  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402


OUT_ROOT = Path("results/glm5_phase616_residual_state_cumulative_patch")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_layers(model: str, n_layers: int) -> List[int]:
    if model == "qwen3":
        return list(range(max(0, 25), min(n_layers, 30)))
    if model == "glm4":
        return list(range(max(0, 30), min(n_layers, 35)))
    if model == "deepseek7b":
        return list(range(max(0, 18), min(n_layers, 23)))
    return list(range(max(0, n_layers - 6), n_layers))


def collect_components(model, tokenizer, device, prompt: str, answer: str, layers_to_scan: List[int]) -> Dict[int, Dict]:
    layers = get_layers(model)
    ids = full_ids(tokenizer, prompt, answer)
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    for li in layers_to_scan:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)

        def make_layer_pre(layer_idx):
            def hook(_module, inputs):
                x = inputs[0]
                if pos < x.shape[1]:
                    captured[layer_idx]["layer_input"] = x[0, pos].detach().float().cpu()
            return hook

        def make_layer_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if pos < y.shape[1]:
                    captured[layer_idx]["layer_out"] = y[0, pos].detach().float().cpu()
            return hook

        def make_attn_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if pos < y.shape[1]:
                    captured[layer_idx]["attn_out"] = y[0, pos].detach().float().cpu()
            return hook

        def make_mlp_out(layer_idx):
            def hook(_module, _inputs, output):
                y = extract_tensor(output)
                if pos < y.shape[1]:
                    captured[layer_idx]["mlp_out"] = y[0, pos].detach().float().cpu()
            return hook

        handles.append(layer.register_forward_pre_hook(make_layer_pre(li)))
        handles.append(layer.register_forward_hook(make_layer_out(li)))
        if attn is not None:
            handles.append(attn.register_forward_hook(make_attn_out(li)))
        if mlp is not None:
            handles.append(mlp.register_forward_hook(make_mlp_out(li)))

    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def make_specs(layers_to_scan: List[int]) -> List[Dict]:
    first = layers_to_scan[0]
    last = layers_to_scan[-1]
    prev = layers_to_scan[-2] if len(layers_to_scan) >= 2 else last
    mid = layers_to_scan[len(layers_to_scan) // 2]
    early = layers_to_scan[: min(3, len(layers_to_scan))]
    late = layers_to_scan[max(0, len(layers_to_scan) - 2):]
    mid_late = layers_to_scan[max(0, len(layers_to_scan) // 2):]

    specs: List[Dict] = [
        {
            "name": f"replace_L{last}_layer_input_ref",
            "mode": "replace",
            "ops": [{"layer": last, "component": "layer_input"}],
        },
        {
            "name": f"replace_L{prev}_layer_out_ref",
            "mode": "replace",
            "ops": [{"layer": prev, "component": "layer_out"}],
        },
        {
            "name": f"add_layer_out_L{prev}_L{last}_bridge",
            "mode": "add",
            "ops": [{"layer": li, "component": "layer_out"} for li in late],
        },
        {
            "name": "add_layer_out_all",
            "mode": "add",
            "ops": [{"layer": li, "component": "layer_out"} for li in layers_to_scan],
        },
        {
            "name": "add_attn_all",
            "mode": "add",
            "ops": [{"layer": li, "component": "attn_out"} for li in layers_to_scan],
        },
        {
            "name": "add_mlp_all",
            "mode": "add",
            "ops": [{"layer": li, "component": "mlp_out"} for li in layers_to_scan],
        },
        {
            "name": "add_attn_mlp_all",
            "mode": "add",
            "ops": [
                {"layer": li, "component": comp}
                for li in layers_to_scan
                for comp in ["attn_out", "mlp_out"]
            ],
        },
        {
            "name": f"add_attn_early_L{first}_L{early[-1]}",
            "mode": "add",
            "ops": [{"layer": li, "component": "attn_out"} for li in early],
        },
        {
            "name": f"add_attn_late_L{late[0]}_L{last}",
            "mode": "add",
            "ops": [{"layer": li, "component": "attn_out"} for li in late],
        },
        {
            "name": f"add_mlp_midlate_L{mid}_L{last}",
            "mode": "add",
            "ops": [{"layer": li, "component": "mlp_out"} for li in mid_late],
        },
        {
            "name": f"add_attn_mlp_midlate_L{mid}_L{last}",
            "mode": "add",
            "ops": [
                {"layer": li, "component": comp}
                for li in mid_late
                for comp in ["attn_out", "mlp_out"]
            ],
        },
    ]

    for li in layers_to_scan:
        for comp in ["attn_out", "mlp_out", "layer_out"]:
            specs.append({
                "name": f"add_L{li}_{comp}",
                "mode": "add",
                "ops": [{"layer": li, "component": comp}],
            })
    return specs


def delta_for(
    comp_cache: Dict[str, Dict],
    answer: str,
    layer_idx: int,
    component: str,
    random_mode: bool,
    seed: int,
) -> torch.Tensor | None:
    base = comp_cache[answer]["base"].get(layer_idx, {}).get(component)
    repair = comp_cache[answer]["repair"].get(layer_idx, {}).get(component)
    if base is None or repair is None:
        return None
    delta = repair.float().cpu() - base.float().cpu()
    if random_mode:
        delta = random_same_norm(delta, seed=seed)
    return delta


def patched_answer(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    spec: Dict,
    comp_cache: Dict[str, Dict],
    random_mode: bool,
    seed: int,
) -> float:
    layers = get_layers(model)
    ids = full_ids(tokenizer, prompt, answer)
    ans_ids = answer_ids(tokenizer, answer)
    pos = answer_prefix_pos(tokenizer, prompt)
    handles = []
    missing = False
    mode = spec["mode"]

    for op_idx, op in enumerate(spec["ops"]):
        li = op["layer"]
        component = op["component"]
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        delta = delta_for(
            comp_cache, answer, li, component, random_mode,
            seed=seed + op_idx * 1009 + li * 37 + len(component),
        )
        if delta is None:
            missing = True
            continue
        delta = delta.to(device=device)
        base = comp_cache[answer]["base"][li][component].float().cpu().to(device=device)
        target = base + delta

        if component == "layer_input":
            def hook(_module, inputs, d=delta, t=target):
                x = inputs[0]
                x_new = x.clone()
                if pos < x_new.shape[1]:
                    if mode == "replace":
                        x_new[0, pos, :] = t.to(dtype=x_new.dtype)
                    else:
                        x_new[0, pos, :] = x_new[0, pos, :] + d.to(dtype=x_new.dtype)
                return replace_input(inputs, x_new)
            handles.append(layer.register_forward_pre_hook(hook))
        elif component == "layer_out":
            def hook(_module, _inputs, output, d=delta, t=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if pos < y_new.shape[1]:
                    if mode == "replace":
                        y_new[0, pos, :] = t.to(dtype=y_new.dtype)
                    else:
                        y_new[0, pos, :] = y_new[0, pos, :] + d.to(dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(layer.register_forward_hook(hook))
        elif component == "attn_out":
            if attn is None:
                missing = True
                continue
            def hook(_module, _inputs, output, d=delta, t=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if pos < y_new.shape[1]:
                    if mode == "replace":
                        y_new[0, pos, :] = t.to(dtype=y_new.dtype)
                    else:
                        y_new[0, pos, :] = y_new[0, pos, :] + d.to(dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(attn.register_forward_hook(hook))
        elif component == "mlp_out":
            if mlp is None:
                missing = True
                continue
            def hook(_module, _inputs, output, d=delta, t=target):
                y = extract_tensor(output)
                y_new = y.clone()
                if pos < y_new.shape[1]:
                    if mode == "replace":
                        y_new[0, pos, :] = t.to(dtype=y_new.dtype)
                    else:
                        y_new[0, pos, :] = y_new[0, pos, :] + d.to(dtype=y_new.dtype)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(mlp.register_forward_hook(hook))
        else:
            raise ValueError(component)

    if missing and not handles:
        return -100.0

    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start = len(tokenizer.encode(prompt, add_special_tokens=False)) - 1
            for i, tid in enumerate(ans_ids):
                p = start + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def patched_scores(
    model,
    tokenizer,
    device,
    prompt: str,
    values: List[str],
    spec: Dict,
    comp_cache: Dict[str, Dict],
    random_mode: bool,
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        scores[ans] = patched_answer(
            model, tokenizer, device, prompt, ans, spec, comp_cache,
            random_mode=random_mode, seed=seed + ai * 101,
        )
    return scores


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "name": items[0]["name"],
            "mode": items[0]["mode"],
            "random": items[0]["random"],
            "n_ops": len(items[0]["ops"]),
            "ops": items[0]["ops"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_correct_delta"] += m["correct_delta"]
            entry["mean_wrong_delta"] += m["old_top_wrong_delta"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry

    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
    log("Best cumulative residual-state patches:")
    for item in best[:24]:
        flag = "random" if item["random"] else "real"
        log(
            f"  {item['name']} {flag}: switch={item['switch']}/{item['n']} "
            f"margin={item['mean_margin_gain']:.3f} ops={item['n_ops']}"
        )
    return {"by_patch": by_patch, "best": best[:200]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        specs = make_specs(layers_to_scan)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        target_seen = 0
        log(
            f"{args.model}: layers={info.n_layers}, scan_layers={layers_to_scan}, "
            f"specs={len(specs)}, raw_cases={len(raw_cases)}"
        )

        for si, case in enumerate(raw_cases):
            base_len = len(tokenizer.encode(case["base_prompt"], add_special_tokens=False))
            repair_len = len(tokenizer.encode(case["repair_prompt"], add_special_tokens=False))
            if base_len != repair_len:
                filtered["token_len_mismatch"] += 1
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
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, layers_to_scan),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, layers_to_scan),
                }

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_prompt_len": base_len,
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, case["correct"], old_top_wrong),
                "patches": {},
            }
            for spec in specs:
                for random_mode in [False, True]:
                    suffix = "random" if random_mode else "real"
                    key = f"{spec['name']}|{suffix}"
                    scores = patched_scores(
                        model, tokenizer, device, case["base_prompt"], values, spec, cache,
                        random_mode=random_mode, seed=si * 1009 + len(spec["name"]),
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "name": spec["name"],
                        "mode": spec["mode"],
                        "ops": spec["ops"],
                        "random": random_mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 616,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "n_specs": len(specs),
            "specs": specs,
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
    out_path = out_dir / f"phase616_{args.model}_residual_state_cumulative_patch_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
