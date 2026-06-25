#!/usr/bin/env python3
"""
Phase 617: Attention Head Cumulative Graph
多层注意力 head 累积图谱

Phase 616 showed that DS7B multi-layer attn_out cumulative patch can fully
restore the value-gate target rows. This phase decomposes that cumulative
attention path into layer/head slots at the o_proj input mixture.
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
from phase609_query_oproj_head_decomposition import (  # noqa: E402
    answer_ids,
    collect_q_o,
    n_heads_for,
    parse_layers,
)
from phase610_head_cumulative_mixture import TOP_HEADS, WEAK_HEADS  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402


OUT_ROOT = Path("results/glm5_phase617_attention_head_cumulative_graph")


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


def known_heads(model_name: str, n_heads: int) -> List[int]:
    heads = []
    for h in TOP_HEADS.get(model_name, []):
        if 0 <= h < n_heads and h not in heads:
            heads.append(h)
    for h in WEAK_HEADS.get(model_name, []):
        if 0 <= h < n_heads and h not in heads:
            heads.append(h)
    if not heads:
        heads = list(range(min(12, n_heads)))
    return heads[:12]


def make_specs(model_name: str, layers_to_scan: List[int], heads_by_layer: Dict[int, int]) -> List[Dict]:
    specs: List[Dict] = []
    first = layers_to_scan[0]
    last = layers_to_scan[-1]
    mid_late = layers_to_scan[max(0, len(layers_to_scan) // 2):]

    specs.append({
        "name": "all_heads_all_layers",
        "kind": "all_heads",
        "ops": [{"layer": li, "heads": list(range(heads_by_layer[li]))} for li in layers_to_scan],
    })
    specs.append({
        "name": f"all_heads_midlate_L{mid_late[0]}_L{last}",
        "kind": "all_heads_midlate",
        "ops": [{"layer": li, "heads": list(range(heads_by_layer[li]))} for li in mid_late],
    })
    for li in layers_to_scan:
        specs.append({
            "name": f"all_heads_L{li}",
            "kind": "all_heads_layer",
            "ops": [{"layer": li, "heads": list(range(heads_by_layer[li]))}],
        })

    min_heads = min(heads_by_layer.values())
    model_heads = known_heads(model_name, min_heads)
    for k in [1, 2, 4, 6, 8, 12]:
        if len(model_heads) >= k:
            heads = model_heads[:k]
            specs.append({
                "name": f"known_top{k}_all_layers",
                "kind": "known_top_all_layers",
                "ops": [{"layer": li, "heads": heads} for li in layers_to_scan],
            })
            specs.append({
                "name": f"known_top{k}_midlate_L{mid_late[0]}_L{last}",
                "kind": "known_top_midlate",
                "ops": [{"layer": li, "heads": heads} for li in mid_late],
            })

    for li in layers_to_scan:
        n_heads = heads_by_layer[li]
        for h in known_heads(model_name, n_heads)[:8]:
            specs.append({
                "name": f"L{li}_H{h}",
                "kind": "single_known_head",
                "ops": [{"layer": li, "heads": [h]}],
            })

    # A small deterministic coverage set checks whether known heads miss obvious broad slots.
    for li in layers_to_scan:
        n_heads = heads_by_layer[li]
        coverage = sorted(set([0, n_heads // 4, n_heads // 2, (3 * n_heads) // 4, n_heads - 1]))
        for h in coverage:
            specs.append({
                "name": f"L{li}_coverage_H{h}",
                "kind": "single_coverage_head",
                "ops": [{"layer": li, "heads": [h]}],
            })
    return specs


def compact_specs(specs: List[Dict], layers_to_scan: List[int]) -> List[Dict]:
    first = layers_to_scan[0]
    last = layers_to_scan[-1]
    mid = layers_to_scan[len(layers_to_scan) // 2]
    keep_layers = {first, mid, last}
    compact = []
    for spec in specs:
        kind = spec["kind"]
        if kind.startswith("all_heads"):
            compact.append(spec)
        elif kind.startswith("known_top"):
            # Keep cumulative known-head curves, but drop very wide 8/12 variants.
            if any(part in spec["name"] for part in ["top1_", "top2_", "top4_", "top6_"]):
                compact.append(spec)
        elif kind == "single_known_head":
            op = spec["ops"][0]
            if op["layer"] in keep_layers:
                compact.append(spec)
        elif kind == "single_coverage_head":
            op = spec["ops"][0]
            if op["layer"] == last:
                compact.append(spec)
    return compact


def patch_answer_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    spec: Dict,
    cache: Dict[str, Dict],
    heads_by_layer: Dict[int, int],
    random_mode: bool,
    seed: int,
) -> float:
    layers = get_layers(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    handles = []

    for op_idx, op in enumerate(spec["ops"]):
        li = op["layer"]
        heads = op["heads"]
        layer = layers[li]
        attn = layer.self_attn
        base_vec = cache[answer]["base"][li].get("o_input")
        repair_vec = cache[answer]["repair"][li].get("o_input")
        if base_vec is None or repair_vec is None:
            continue
        delta = repair_vec.float().cpu() - base_vec.float().cpu()
        width = delta.numel()
        n_heads = heads_by_layer[li]
        head_dim = width // max(1, n_heads)
        slot_deltas = {}
        for hi in heads:
            if not (0 <= hi < n_heads):
                continue
            start = hi * head_dim
            end = width if hi == n_heads - 1 else (hi + 1) * head_dim
            d = delta[start:end]
            if random_mode:
                d = random_same_norm(d, seed=seed + op_idx * 1009 + li * 101 + hi)
            slot_deltas[hi] = d

        def hook(_module, inputs, slot_deltas=slot_deltas, n_heads=n_heads, width=width):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                head_dim = width // max(1, n_heads)
                for hi, d in slot_deltas.items():
                    start = hi * head_dim
                    end = width if hi == n_heads - 1 else (hi + 1) * head_dim
                    x_new[0, pos, start:end] = x_new[0, pos, start:end] + d.to(
                        device=x_new.device, dtype=x_new.dtype
                    )
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(hook))

    if not handles:
        return -100.0

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
    spec: Dict,
    cache: Dict[str, Dict],
    heads_by_layer: Dict[int, int],
    random_mode: bool,
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        scores[ans] = patch_answer_score(
            model, tokenizer, device, case["base_prompt"], ans, spec, cache,
            heads_by_layer, random_mode=random_mode, seed=seed + ai * 997,
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
            "kind": items[0]["kind"],
            "random": items[0]["random"],
            "n_ops": len(items[0]["ops"]),
            "n_slots": sum(len(op["heads"]) for op in items[0]["ops"]),
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
    log("Best attention head cumulative patches:")
    for item in best[:24]:
        flag = "random" if item["random"] else "real"
        log(
            f"  {item['name']} {flag}: switch={item['switch']}/{item['n']} "
            f"margin={item['mean_margin_gain']:.3f} slots={item['n_slots']}"
        )
    return {"by_patch": by_patch, "best": best[:220]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        specs = make_specs(args.model, layers_to_scan, heads_by_layer)
        if args.compact:
            specs = compact_specs(specs, layers_to_scan)
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        target_seen = 0
        log(
            f"{args.model}: layers={info.n_layers}, scan_layers={layers_to_scan}, "
            f"heads={heads_by_layer}, specs={len(specs)}, raw_cases={len(raw_cases)}"
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

            cache: Dict[str, Dict] = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                cache[ans] = {"base": {}, "repair": {}}
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_q_o(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_q_o(model, tokenizer, device, case["repair_prompt"], ans, li)

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
                        model, tokenizer, device, case, values, spec, cache, heads_by_layer,
                        random_mode=random_mode, seed=si * 1009 + len(spec["name"]),
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "name": spec["name"],
                        "kind": spec["kind"],
                        "ops": spec["ops"],
                        "random": random_mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 617,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "n_specs": len(specs),
            "compact": args.compact,
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
    parser.add_argument("--compact", action="store_true")
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
    out_path = out_dir / f"phase617_{args.model}_attention_head_cumulative_graph_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
