#!/usr/bin/env python3
"""
Phase 614: Answer-Position Query State Builder Audit
答案位置查询状态生成器审计

Phase 613 localized value-gate repair to answer-position Q. This phase asks
whether the repair Q is already present in q_proj input hidden state, or whether
it only appears after q_proj output replacement.

Modes:
  o_actual:         direct repair o_proj-input top-head slots
  q_output_top:     repair q_proj output for selected heads, base K/V
  q_input_full:     repair q_proj input hidden at answer position
  q_input_delta:    add repair-base delta to q_proj input at answer position
  layer_input_full: repair decoder-layer input at answer position
  layer_input_delta:add repair-base delta to decoder-layer input
  random controls: same-norm random controls for q input, layer input, and o input
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
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, default_layers, n_heads_for, parse_layers  # noqa: E402
from phase610_head_cumulative_mixture import TOP_HEADS  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402


OUT_ROOT = Path("results/glm5_phase614_query_state_builder_audit")
MODES = [
    "o_actual",
    "q_output_top",
    "q_input_full",
    "q_input_delta",
    "layer_input_full",
    "layer_input_delta",
    "q_input_random",
    "layer_input_random",
    "o_random",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def clone_out(output):
    if isinstance(output, tuple):
        return output[0].clone(), output[1:]
    return output.clone(), None


def restore_out(tensor, rest):
    if rest is None:
        return tensor
    return (tensor,) + rest


def collect_query_parts(model, tokenizer, device, prompt: str, answer: str, layer_idx: int) -> Dict:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def layer_pre(_module, inputs):
        captured["layer_input"] = inputs[0][0, pos].detach().float().cpu()

    def q_pre(_module, inputs):
        captured["q_input"] = inputs[0][0, pos].detach().float().cpu()

    def q_out(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["q_output"] = tensor[0, pos].detach().float().cpu()

    def o_pre(_module, inputs):
        captured["o_input"] = inputs[0][0, pos].detach().float().cpu()

    handles.append(layer.register_forward_pre_hook(layer_pre))
    handles.append(attn.q_proj.register_forward_pre_hook(q_pre))
    handles.append(attn.q_proj.register_forward_hook(q_out))
    handles.append(attn.o_proj.register_forward_pre_hook(o_pre))
    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def patch_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_parts: Dict,
    repair_parts: Dict,
    heads: List[int],
    n_heads: int,
    mode: str,
    seed: int,
) -> float:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    q_width = base_parts["q_output"].numel()
    q_head_dim = q_width // max(1, n_heads)
    o_width = base_parts["o_input"].numel()
    o_head_dim = o_width // max(1, n_heads)
    handles = []

    if mode in ("layer_input_full", "layer_input_delta", "layer_input_random"):
        base_vec = base_parts["layer_input"].float().cpu()
        repair_vec = repair_parts["layer_input"].float().cpu()
        if mode == "layer_input_full":
            target = repair_vec
        elif mode == "layer_input_delta":
            target = base_vec + (repair_vec - base_vec)
        else:
            target = base_vec + random_same_norm(repair_vec - base_vec, seed)

        def layer_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])

        handles.append(layer.register_forward_pre_hook(layer_hook))

    if mode in ("q_input_full", "q_input_delta", "q_input_random"):
        base_vec = base_parts["q_input"].float().cpu()
        repair_vec = repair_parts["q_input"].float().cpu()
        if mode == "q_input_full":
            target = repair_vec
        elif mode == "q_input_delta":
            target = base_vec + (repair_vec - base_vec)
        else:
            target = base_vec + random_same_norm(repair_vec - base_vec, seed)

        def q_input_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                x_new[0, pos, :] = target.to(device=x_new.device, dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.q_proj.register_forward_pre_hook(q_input_hook))

    if mode == "q_output_top":
        def q_output_hook(_module, _inputs, output):
            x, rest = clone_out(output)
            if pos < x.shape[1]:
                for hi in heads:
                    start = hi * q_head_dim
                    end = start + q_head_dim
                    x[0, pos, start:end] = repair_parts["q_output"][start:end].to(device=x.device, dtype=x.dtype)
            return restore_out(x, rest)

        handles.append(attn.q_proj.register_forward_hook(q_output_hook))

    if mode in ("o_actual", "o_random"):
        deltas = {}
        for hi in heads:
            start = hi * o_head_dim
            end = o_width if hi == n_heads - 1 else (hi + 1) * o_head_dim
            d = repair_parts["o_input"][start:end].float().cpu() - base_parts["o_input"][start:end].float().cpu()
            if mode == "o_random":
                d = random_same_norm(d, seed + hi * 101)
            deltas[hi] = d

        def o_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                for hi, d in deltas.items():
                    start = hi * o_head_dim
                    end = o_width if hi == n_heads - 1 else (hi + 1) * o_head_dim
                    x_new[0, pos, start:end] = x_new[0, pos, start:end] + d.to(x_new.device, dtype=x_new.dtype)
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(o_hook))

    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start_pos = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                p = start_pos + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
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
            "layer": items[0]["layer"],
            "mode": items[0]["mode"],
            "heads": items[0]["heads"],
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
    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:100]
    log("Best query-state builder patches:")
    for item in best[:18]:
        log(
            f"  L{item['layer']} {item['mode']} heads={item['heads']}: "
            f"switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}"
        )
    return {"by_patch": by_patch, "best": best}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        target_seen = 0
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        log(f"{args.model}: layers={info.n_layers}, probe_layers={layers_to_scan}, heads={heads_by_layer}, raw_cases={len(raw_cases)}")

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
                cache[ans] = {"base": {}, "repair": {}}
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_query_parts(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_query_parts(model, tokenizer, device, case["repair_prompt"], ans, li)

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
            for li in layers_to_scan:
                n_heads = heads_by_layer[li]
                heads = [h for h in TOP_HEADS.get(args.model, list(range(min(4, n_heads))))[: args.top_k] if h < n_heads]
                for mode in MODES:
                    key = f"L{li}|top{len(heads)}|{mode}"
                    scores = {}
                    for ans_i, ans in enumerate(values):
                        scores[ans] = patch_score(
                            model, tokenizer, device, case["base_prompt"], ans, li,
                            cache[ans]["base"][li], cache[ans]["repair"][li],
                            heads, n_heads, mode,
                            seed=si * 1009 + ans_i * 997 + len(mode),
                        )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "layer": li,
                        "mode": mode,
                        "heads": heads,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 614,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "top_k": args.top_k,
            "top_heads": TOP_HEADS.get(args.model, []),
            "n_raw_cases": len(raw_cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summarize(rows),
            "target_summary": summarize([r for r in rows if r.get("target_case")]),
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
    parser.add_argument("--top-k", type=int, default=4)
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
            args.layers = str(default_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)[0])
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
    out_path = out_dir / f"phase614_{args.model}_query_state_builder_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
