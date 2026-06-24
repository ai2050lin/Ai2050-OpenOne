#!/usr/bin/env python3
"""
Phase 607: Pre-Final Residual Trajectory Layer Scan
最终层前残差轨迹层扫描

Phase 606 showed digit1 value selection is already present at final-layer input.
This phase scans late layers to find where the digit1 residual trajectory first
becomes causally sufficient.
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
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input, score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402


OUT_ROOT = Path("results/glm5_phase607_prefinal_residual_trajectory_layer_scan")
COMPONENTS = ["layer_input", "layer_out", "attn_out", "mlp_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def digit1_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_layers(model: str, n_layers: int) -> List[int]:
    if model == "qwen3":
        return list(range(max(0, n_layers - 8), n_layers))
    if model == "glm4":
        return list(range(max(0, n_layers - 8), n_layers))
    if model == "deepseek7b":
        return list(range(max(0, n_layers - 8), n_layers))
    return list(range(max(0, n_layers - 8), n_layers))


def parse_layers(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def collect_components(model, tokenizer, device, prompt: str, answer: str, layers_to_scan: List[int]) -> Dict[int, Dict]:
    layers = get_layers(model)
    ids = full_ids(tokenizer, prompt, answer)
    pos = digit1_pos(tokenizer, prompt)
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
            input_ids = torch.tensor([ids], device=device)
            model(input_ids=input_ids, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def patched_answer(model, tokenizer, device, prompt: str, answer: str, layer_idx: int,
                   component: str, target: torch.Tensor) -> float:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = get_attn(layer)
    mlp = get_mlp(layer)
    ids = full_ids(tokenizer, prompt, answer)
    ans_ids = answer_ids(tokenizer, answer)
    pos = digit1_pos(tokenizer, prompt)
    target = target.to(device=device)
    handle = None

    if component == "layer_input":
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[0, pos, :] = target.to(dtype=x_new.dtype)
            return replace_input(inputs, x_new)
        handle = layer.register_forward_pre_hook(hook)
    elif component == "layer_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = layer.register_forward_hook(hook)
    elif component == "attn_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = attn.register_forward_hook(hook)
    elif component == "mlp_out":
        def hook(_module, _inputs, output):
            y = extract_tensor(output)
            y_new = y.clone()
            y_new[0, pos, :] = target.to(dtype=y_new.dtype)
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new
        handle = mlp.register_forward_hook(hook)
    else:
        raise ValueError(component)

    try:
        total = 0.0
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids=input_ids, return_dict=True).logits[0].float()
            start = len(tokenizer.encode(prompt, add_special_tokens=False)) - 1
            for i, tid in enumerate(ans_ids):
                p = start + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        if handle is not None:
            handle.remove()


def patched_scores(model, tokenizer, device, prompt: str, values: List[str], layer_idx: int,
                   component: str, comp_cache: Dict[str, Dict], random_mode: bool, seed: int) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        base = comp_cache[ans]["base"].get(layer_idx, {}).get(component)
        repair = comp_cache[ans]["repair"].get(layer_idx, {}).get(component)
        if base is None or repair is None:
            scores[ans] = -100.0
            continue
        delta = repair.float().cpu() - base.float().cpu()
        if random_mode:
            delta = random_same_norm(delta, seed=seed + ai * 101 + layer_idx)
        target = base.float().cpu() + delta
        scores[ans] = patched_answer(model, tokenizer, device, prompt, ans, layer_idx, component, target)
    return scores


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "layer": items[0]["layer"],
            "component": items[0]["component"],
            "random": items[0]["random"],
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

    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:120]
    first_effective = {}
    for comp in COMPONENTS:
        candidates = [
            v for v in by_patch.values()
            if v["component"] == comp and not v["random"] and v["switch"] > 0
        ]
        if candidates:
            first_effective[comp] = sorted(candidates, key=lambda x: (x["layer"], -x["switch"]))[0]
    log("Best pre-final trajectory patches:")
    for item in best[:16]:
        log(f"  L{item['layer']} {item['component']}: switch={item['switch']}/{item['n']}, margin={item['mean_margin_gain']:.3f}")
    log("First effective by component:")
    for comp, item in first_effective.items():
        log(f"  {comp}: L{item['layer']} switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}")
    return {"by_patch": by_patch, "best": best, "first_effective": first_effective}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        cases = list(build_cases(args.n_tables, args.max_samples))
        rows = []
        target_seen = 0
        log(f"{args.model}: layers={info.n_layers}, scan={layers_to_scan}, cases={len(cases)}")

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

            comp_cache = {}
            for ans in values:
                comp_cache[ans] = {
                    "base": collect_components(model, tokenizer, device, case["base_prompt"], ans, layers_to_scan),
                    "repair": collect_components(model, tokenizer, device, case["repair_prompt"], ans, layers_to_scan),
                }

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, correct, old_top_wrong),
                "patches": {},
            }
            for li in layers_to_scan:
                for comp in COMPONENTS:
                    for random_mode in [False, True]:
                        if random_mode and comp not in ("layer_input", "layer_out", "attn_out", "mlp_out"):
                            continue
                        key = f"L{li}|{comp}{'|random' if random_mode else ''}"
                        scores = patched_scores(
                            model, tokenizer, device, case["base_prompt"], values,
                            li, comp, comp_cache, random_mode=random_mode, seed=si * 1009 + len(key),
                        )
                        patched = winner_stats(scores, correct)
                        row["patches"][key] = {
                            "layer": li,
                            "component": comp,
                            "random": random_mode,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                        }
            rows.append(row)

        return {
            "phase": 607,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "n_cases": len(cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
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
            args.layers = "0"
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 96)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase607_{args.model}_prefinal_residual_trajectory_layer_scan_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
