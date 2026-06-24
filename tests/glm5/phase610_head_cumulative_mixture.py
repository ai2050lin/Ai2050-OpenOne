#!/usr/bin/env python3
"""
Phase 610: Head Cumulative Mixture
逐头累积混合测试

Phase 609 localized the strong attention repair to o_proj input mixture. This
phase tests whether top head slots cumulatively approach the full o_proj-input
effect, or whether the effect requires a broad distributed head field.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
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
from phase586_distributed_value_path_patch import build_cases, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import (  # noqa: E402
    answer_ids,
    collect_q_o,
    default_layers,
    n_heads_for,
    parse_layers,
)


OUT_ROOT = Path("results/glm5_phase610_head_cumulative_mixture")

TOP_HEADS = {
    "qwen3": [11, 23, 6, 14, 5, 2],
    "glm4": [12, 8, 4, 28, 6, 7],
    "deepseek7b": [3, 1, 7, 24, 25, 13],
}

WEAK_HEADS = {
    "qwen3": [9, 0, 19, 28, 15, 8],
    "glm4": [2, 26, 3, 25, 23, 9],
    "deepseek7b": [19, 18, 26, 8, 6, 14],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cumulative_specs(model_name: str, n_heads: int) -> List[Dict]:
    top = [h for h in TOP_HEADS.get(model_name, []) if 0 <= h < n_heads]
    weak = [h for h in WEAK_HEADS.get(model_name, []) if 0 <= h < n_heads]
    if not top:
        top = list(range(min(6, n_heads)))
    if not weak:
        weak = list(range(max(0, n_heads - 6), n_heads))
    specs: List[Dict] = []
    for k in [1, 2, 3, 4, 6]:
        if len(top) >= k:
            specs.append({"name": f"top{k}_delta", "kind": "top_delta", "heads": top[:k]})
            specs.append({"name": f"top{k}_random_slots", "kind": "top_random_slots", "heads": top[:k]})
        if len(weak) >= k:
            specs.append({"name": f"weak{k}_delta", "kind": "weak_delta", "heads": weak[:k]})
    specs.append({"name": "all_delta", "kind": "all_delta", "heads": list(range(n_heads))})
    specs.append({"name": "all_random_slots", "kind": "all_random_slots", "heads": list(range(n_heads))})
    return specs


def random_heads(n_heads: int, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    heads = list(range(n_heads))
    rng.shuffle(heads)
    return sorted(heads[:k])


def patch_answer_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_vec: torch.Tensor,
    repair_vec: torch.Tensor,
    heads: List[int],
    n_heads: int,
    random_slots: bool,
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
    delta = repair_vec.float().cpu() - base_vec.float().cpu()
    width = delta.numel()
    head_dim = width // max(1, n_heads)

    slot_deltas: Dict[int, torch.Tensor] = {}
    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        d = delta[start:end]
        if random_slots:
            d = random_same_norm(d, seed=seed + hi * 101)
        slot_deltas[hi] = d

    def hook(_module, inputs):
        x = inputs[0]
        x_new = x.clone()
        if pos < x_new.shape[1]:
            for hi, d in slot_deltas.items():
                start = hi * head_dim
                end = width if hi == n_heads - 1 else (hi + 1) * head_dim
                x_new[0, pos, start:end] = x_new[0, pos, start:end] + d.to(
                    device=x_new.device, dtype=x_new.dtype
                )
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


def patched_scores(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    layer_idx: int,
    cache: Dict[str, Dict],
    heads: List[int],
    n_heads: int,
    random_slots: bool,
    seed: int,
) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        base_vec = cache[ans]["base"][layer_idx].get("o_input")
        repair_vec = cache[ans]["repair"][layer_idx].get("o_input")
        if base_vec is None or repair_vec is None:
            scores[ans] = -100.0
            continue
        scores[ans] = patch_answer_score(
            model, tokenizer, device, case["base_prompt"], ans, layer_idx,
            base_vec, repair_vec, heads, n_heads, random_slots,
            seed=seed + ai * 997,
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
            "name": items[0]["name"],
            "kind": items[0]["kind"],
            "heads": items[0]["heads"],
            "n_heads": len(items[0]["heads"]),
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
    log("Best cumulative head patches:")
    for item in best[:20]:
        log(
            f"  L{item['layer']} {item['name']} heads={item['heads']}: "
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
                specs = cumulative_specs(args.model, n_heads)
                for spec in specs:
                    heads = list(spec["heads"])
                    if spec["kind"].startswith("weak") and not heads:
                        continue
                    if spec["kind"].startswith("top") and not heads:
                        continue
                    random_slots = spec["kind"].endswith("random_slots")
                    key = f"L{li}|{spec['name']}"
                    scores = patched_scores(
                        model, tokenizer, device, case, values, li, cache, heads, n_heads,
                        random_slots=random_slots,
                        seed=si * 1009 + li * 53 + len(spec["name"]),
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "layer": li,
                        "name": spec["name"],
                        "kind": spec["kind"],
                        "heads": heads,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }

                for k in [1, 2, 3, 4, 6]:
                    if k > n_heads:
                        continue
                    heads = random_heads(n_heads, k, seed=si * 991 + li * 31 + k)
                    key = f"L{li}|randheads{k}_delta"
                    scores = patched_scores(
                        model, tokenizer, device, case, values, li, cache, heads, n_heads,
                        random_slots=False,
                        seed=si * 1009 + li * 59 + k,
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "layer": li,
                        "name": f"randheads{k}_delta",
                        "kind": "random_heads_delta",
                        "heads": heads,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 610,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "top_heads": TOP_HEADS.get(args.model, []),
            "weak_heads": WEAK_HEADS.get(args.model, []),
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
    out_path = out_dir / f"phase610_{args.model}_head_cumulative_mixture_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
