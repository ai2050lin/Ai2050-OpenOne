#!/usr/bin/env python3
"""
Phase 613: Q/K Routing Factor Split
Q/K 路由因子拆分

Phase 612 showed that alpha_repair * V_base is sufficient to reproduce the
top-head value-gate repair. This phase tests where the repair attention pattern
comes from by patching q_proj/k_proj/v_proj outputs inside the natural forward:

  q_only:  repair Q at answer position + base K/V
  k_only:  base Q + repair K at aligned source positions + base V
  qk:      repair Q + repair K + base V
  v_only:  base Q/K + repair V
  qkv:     repair Q/K/V
  o_actual: direct repair o_proj-input top-head slots, as anchor

Patching projections lets the model apply its own rotary/position handling and
attention implementation, avoiding hand-reimplementing QK attention details.
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
from typing import Dict, List, Set

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


OUT_ROOT = Path("results/glm5_phase613_qk_routing_factor_split")
MODES = ["q_only", "k_only", "qk", "v_only", "qv", "kv", "qkv", "o_actual", "random_o_actual_norm"]


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


def n_kv_heads_for(attn, n_heads: int, head_dim: int) -> int:
    for obj in [attn, getattr(attn, "config", None)]:
        if obj is not None and hasattr(obj, "num_key_value_heads"):
            val = int(getattr(obj, "num_key_value_heads"))
            if val > 0:
                return val
    if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "out_features"):
        return max(1, int(attn.k_proj.out_features) // max(1, head_dim))
    return n_heads


def kv_heads_for(q_heads: List[int], n_heads: int, n_kv: int) -> Set[int]:
    group = max(1, n_heads // max(1, n_kv))
    return {min(n_kv - 1, max(0, h // group)) for h in q_heads}


def collect_projection_parts(model, tokenizer, device, prompt: str, answer: str, layer_idx: int) -> Dict:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[0].detach().float().cpu()
        return hook

    def o_pre_hook(_module, inputs):
        captured["o_input"] = inputs[0][0, pos].detach().float().cpu()

    handles.append(attn.q_proj.register_forward_hook(make_hook("q_proj")))
    handles.append(attn.k_proj.register_forward_hook(make_hook("k_proj")))
    handles.append(attn.v_proj.register_forward_hook(make_hook("v_proj")))
    handles.append(attn.o_proj.register_forward_pre_hook(o_pre_hook))
    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def patch_projection_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_parts: Dict,
    repair_parts: Dict,
    q_heads: List[int],
    n_heads: int,
    n_kv: int,
    mode: str,
    seed: int,
) -> float:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    source_end = min(pos + 1, len(ids))
    q_dim = base_parts["q_proj"].shape[-1]
    kv_dim = base_parts["k_proj"].shape[-1]
    head_dim = q_dim // max(1, n_heads)
    kv_head_dim = kv_dim // max(1, n_kv)
    selected_kv = kv_heads_for(q_heads, n_heads, n_kv)

    handles = []

    def q_hook(_module, _inputs, output):
        x, rest = clone_out(output)
        if "q" in mode and pos < x.shape[1]:
            for hi in q_heads:
                start = hi * head_dim
                end = start + head_dim
                x[0, pos, start:end] = repair_parts["q_proj"][pos, start:end].to(device=x.device, dtype=x.dtype)
        return restore_out(x, rest)

    def k_hook(_module, _inputs, output):
        x, rest = clone_out(output)
        if "k" in mode:
            for hi in selected_kv:
                start = hi * kv_head_dim
                end = start + kv_head_dim
                x[0, :source_end, start:end] = repair_parts["k_proj"][:source_end, start:end].to(
                    device=x.device, dtype=x.dtype
                )
        return restore_out(x, rest)

    def v_hook(_module, _inputs, output):
        x, rest = clone_out(output)
        if "v" in mode:
            for hi in selected_kv:
                start = hi * kv_head_dim
                end = start + kv_head_dim
                x[0, :source_end, start:end] = repair_parts["v_proj"][:source_end, start:end].to(
                    device=x.device, dtype=x.dtype
                )
        return restore_out(x, rest)

    handles.append(attn.q_proj.register_forward_hook(q_hook))
    handles.append(attn.k_proj.register_forward_hook(k_hook))
    handles.append(attn.v_proj.register_forward_hook(v_hook))
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


def patch_o_actual_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_o: torch.Tensor,
    repair_o: torch.Tensor,
    heads: List[int],
    n_heads: int,
    seed: int,
    random_mode: bool,
) -> float:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    width = base_o.numel()
    head_dim = width // max(1, n_heads)
    deltas = {}
    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        d = repair_o[start:end].float().cpu() - base_o[start:end].float().cpu()
        if random_mode:
            d = random_same_norm(d, seed + hi * 101)
        deltas[hi] = d

    def hook(_module, inputs):
        x = inputs[0]
        x_new = x.clone()
        if pos < x_new.shape[1]:
            for hi, d in deltas.items():
                start = hi * head_dim
                end = width if hi == n_heads - 1 else (hi + 1) * head_dim
                x_new[0, pos, start:end] = x_new[0, pos, start:end] + d.to(x_new.device, dtype=x_new.dtype)
        return (x_new,) + tuple(inputs[1:])

    handle = attn.o_proj.register_forward_pre_hook(hook)
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
        handle.remove()


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
    log("Best Q/K/V projection patches:")
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
        kv_by_layer = {}
        for li in layers_to_scan:
            attn = get_layers(model)[li].self_attn
            q_dim = int(attn.q_proj.out_features)
            n_heads = heads_by_layer[li]
            head_dim = q_dim // max(1, n_heads)
            kv_by_layer[li] = n_kv_heads_for(attn, n_heads, head_dim)
        log(f"{args.model}: layers={info.n_layers}, probe_layers={layers_to_scan}, heads={heads_by_layer}, kv={kv_by_layer}, raw_cases={len(raw_cases)}")

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
                ids_base = full_ids(tokenizer, case["base_prompt"], ans)
                ids_repair = full_ids(tokenizer, case["repair_prompt"], ans)
                if len(ids_base) != len(ids_repair):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_projection_parts(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_projection_parts(model, tokenizer, device, case["repair_prompt"], ans, li)

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
                n_kv = kv_by_layer[li]
                heads = [h for h in TOP_HEADS.get(args.model, list(range(min(4, n_heads))))[: args.top_k] if h < n_heads]
                for mode in MODES:
                    key = f"L{li}|top{len(heads)}|{mode}"
                    scores = {}
                    for ans_i, ans in enumerate(values):
                        base_parts = cache[ans]["base"][li]
                        repair_parts = cache[ans]["repair"][li]
                        if mode in ("o_actual", "random_o_actual_norm"):
                            scores[ans] = patch_o_actual_score(
                                model, tokenizer, device, case["base_prompt"], ans, li,
                                base_parts["o_input"].float().cpu(),
                                repair_parts["o_input"].float().cpu(),
                                heads, n_heads,
                                seed=si * 1009 + ans_i * 997 + len(mode),
                                random_mode=(mode == "random_o_actual_norm"),
                            )
                        else:
                            scores[ans] = patch_projection_score(
                                model, tokenizer, device, case["base_prompt"], ans, li,
                                base_parts, repair_parts, heads, n_heads, n_kv, mode,
                                seed=si * 1009 + ans_i * 997 + len(mode),
                            )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "layer": li,
                        "mode": mode,
                        "heads": heads,
                        "kv_heads": sorted(kv_heads_for(heads, n_heads, n_kv)),
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 613,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "kv_heads_by_layer": kv_by_layer,
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
    out_path = out_dir / f"phase613_{args.model}_qk_routing_factor_split_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
