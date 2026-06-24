#!/usr/bin/env python3
"""
Phase 609: Query / O-Proj Input / Head-Slot Decomposition
查询、输出投影输入与逐头槽位分解

Phase 608 showed source-token K/V deltas cannot reproduce the strong L22
attention-output repair found in Phase 607. This phase tests whether the
effective unit is query state, full o_proj input mixture, or head slots.
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
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402


OUT_ROOT = Path("results/glm5_phase609_query_oproj_head_decomposition")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_layers(model: str, n_layers: int) -> List[int]:
    if model == "qwen3":
        return [li for li in [29] if li < n_layers]
    if model == "glm4":
        return [li for li in [34] if li < n_layers]
    if model == "deepseek7b":
        return [li for li in [22] if li < n_layers]
    return [max(0, n_layers - 6)]


def parse_layers(text: str) -> List[int]:
    out: List[int] = []
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


def n_heads_for(model, attn) -> int:
    cfg = getattr(model, "config", None)
    for name in ["num_attention_heads", "n_head", "n_heads"]:
        val = getattr(cfg, name, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            return val
    val = getattr(attn, "num_heads", None)
    if isinstance(val, int) and val > 0:
        return val
    return 1


def extract_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def collect_q_o(model, tokenizer, device, prompt: str, answer: str, layer_idx: int) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def q_hook(_module, _inputs, output):
        if pos < output.shape[1]:
            captured["q"] = output[0, pos].detach().float().cpu()

    def o_pre_hook(_module, inputs):
        x = inputs[0]
        if pos < x.shape[1]:
            captured["o_input"] = x[0, pos].detach().float().cpu()

    def attn_out_hook(_module, _inputs, output):
        y = extract_tensor(output)
        if pos < y.shape[1]:
            captured["attn_out"] = y[0, pos].detach().float().cpu()

    handles.append(attn.q_proj.register_forward_hook(q_hook))
    handles.append(attn.o_proj.register_forward_pre_hook(o_pre_hook))
    handles.append(attn.register_forward_hook(attn_out_hook))
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
    base_vec: torch.Tensor,
    repair_vec: torch.Tensor,
    mode: str,
    seed: int,
    head_idx: Optional[int] = None,
    n_heads: int = 1,
) -> float:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    delta = repair_vec.float().cpu() - base_vec.float().cpu()
    if mode.endswith("_random"):
        delta = random_same_norm(delta, seed=seed)

    handles = []

    if mode.startswith("q_"):
        def q_hook(_module, _inputs, output):
            y = output.clone()
            if pos < y.shape[1]:
                d = delta.to(device=y.device, dtype=y.dtype)
                y[0, pos, :] = y[0, pos, :] + d
            return y

        handles.append(attn.q_proj.register_forward_hook(q_hook))

    elif mode.startswith("o_input_"):
        def o_pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                d = delta.to(device=x_new.device, dtype=x_new.dtype)
                x_new[0, pos, :] = x_new[0, pos, :] + d
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(o_pre_hook))

    elif mode.startswith("head_"):
        if head_idx is None:
            raise ValueError("head_idx required for head patch")
        width = base_vec.numel()
        head_dim = width // max(1, n_heads)
        start = head_idx * head_dim
        end = width if head_idx == n_heads - 1 else (head_idx + 1) * head_dim

        def head_pre_hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                d = delta[start:end].to(device=x_new.device, dtype=x_new.dtype)
                x_new[0, pos, start:end] = x_new[0, pos, start:end] + d
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(head_pre_hook))
    else:
        raise ValueError(mode)

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


def patched_scores(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    layer_idx: int,
    cache: Dict[str, Dict],
    mode: str,
    seed: int,
    head_idx: Optional[int] = None,
    n_heads: int = 1,
) -> Dict[str, float]:
    scores = {}
    source = "q" if mode.startswith("q_") else "o_input"
    for ai, ans in enumerate(values):
        base_vec = cache[ans]["base"][layer_idx].get(source)
        repair_vec = cache[ans]["repair"][layer_idx].get(source)
        if base_vec is None or repair_vec is None:
            scores[ans] = -100.0
            continue
        scores[ans] = patch_score(
            model, tokenizer, device, case["base_prompt"], ans, layer_idx,
            base_vec, repair_vec, mode, seed=seed + ai * 991,
            head_idx=head_idx, n_heads=n_heads,
        )
    return scores


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "layer": items[0]["layer"],
            "mode": items[0]["mode"],
            "head": items[0].get("head"),
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
    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:160]
    log("Best Q/O/head patches:")
    for item in best[:20]:
        h = "" if item.get("head") is None else f" head={item['head']}"
        log(
            f"  L{item['layer']} {item['mode']}{h}: "
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
        cases = list(build_cases(args.n_tables, args.max_samples))
        rows = []
        target_seen = 0
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        log(f"{args.model}: layers={info.n_layers}, probe_layers={layers_to_scan}, heads={heads_by_layer}, cases={len(cases)}")

        for si, case in enumerate(cases):
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            cache: Dict[str, Dict] = {}
            for ans in values:
                cache[ans] = {"base": {}, "repair": {}}
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_q_o(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_q_o(model, tokenizer, device, case["repair_prompt"], ans, li)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, case["correct"], old_top_wrong),
                "patches": {},
            }
            for li in layers_to_scan:
                n_heads = heads_by_layer[li]
                for mode in ["q_delta", "q_random", "o_input_delta", "o_input_random"]:
                    key = f"L{li}|{mode}"
                    scores = patched_scores(
                        model, tokenizer, device, case, values, li, cache, mode,
                        seed=si * 1009 + li * 37 + len(mode), n_heads=n_heads,
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "layer": li,
                        "mode": mode,
                        "head": None,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
                for hi in range(n_heads):
                    for mode in ["head_delta", "head_random"]:
                        key = f"L{li}|{mode}|H{hi}"
                        scores = patched_scores(
                            model, tokenizer, device, case, values, li, cache, mode,
                            seed=si * 1009 + li * 41 + hi * 13 + len(mode),
                            head_idx=hi, n_heads=n_heads,
                        )
                        patched = winner_stats(scores, case["correct"])
                        row["patches"][key] = {
                            "layer": li,
                            "mode": mode,
                            "head": hi,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                        }
            rows.append(row)

        return {
            "phase": 609,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
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
            args.layers = str(default_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)[0])
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
    out_path = out_dir / f"phase609_{args.model}_query_oproj_head_decomposition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
