#!/usr/bin/env python3
"""
Phase 608: Attention Source K/V Decomposition
注意力源词元 K/V 因果分解

Phase 607 localized the clearest DS7B digit1 trajectory jump to L22 attention.
This phase tests which source token groups can causally carry the repair delta
through the target attention layer by patching k_proj / v_proj outputs at source
positions during real forward passes.
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
from phase586_distributed_value_path_patch import (  # noqa: E402
    build_cases,
    random_same_norm,
    token_pos_after_substring,
)
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402


OUT_ROOT = Path("results/glm5_phase608_attention_source_kv_decomposition")
SOURCE_GROUPS = [
    "rule_value",
    "rule_relation",
    "query_relation",
    "query_category",
    "query_object",
    "prompt_last",
    "answer_prefix",
    "random_position",
]
PATCH_MODES = ["v_delta", "k_delta", "kv_delta", "kv_random"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tokenizer, answer: str) -> List[int]:
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(answer, add_special_tokens=False)
    return ids


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


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


def first_answer_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def source_positions(tokenizer, case: Dict, prompt: str, relation_for_rule: str,
                     answer: str, random_seed: int) -> Dict[str, List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    full_len = len(prompt_ids) + len(ans_ids)
    ans_prefix = len(prompt_ids)
    candidates = {
        "rule_value": token_pos_after_substring(tokenizer, prompt, case["correct"], "first"),
        "rule_relation": token_pos_after_substring(tokenizer, prompt, relation_for_rule, "first"),
        "query_relation": token_pos_after_substring(tokenizer, prompt, case["relation"], "last"),
        "query_category": token_pos_after_substring(tokenizer, prompt, case["category"], "last"),
        "query_object": token_pos_after_substring(tokenizer, prompt, case["object"], "last"),
        "prompt_last": len(prompt_ids) - 1,
        "answer_prefix": ans_prefix if ans_ids else None,
    }
    rng_pos = 0
    if prompt_ids:
        rng_pos = (random_seed * 1103515245 + 12345) % len(prompt_ids)
    candidates["random_position"] = int(rng_pos)

    out: Dict[str, List[int]] = {}
    for name, pos in candidates.items():
        if pos is None:
            out[name] = []
            continue
        p = int(pos)
        out[name] = [p] if 0 <= p < full_len else []
    return out


def mean_at(tensor: torch.Tensor, positions: List[int]) -> Optional[torch.Tensor]:
    valid = [p for p in positions if 0 <= p < tensor.shape[0]]
    if not valid:
        return None
    return tensor[valid].float().mean(dim=0).cpu()


def collect_kv(model, tokenizer, device, prompt: str, answer: str, layer_idx: int) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output[0].detach().float().cpu()
        return hook

    handles.append(attn.k_proj.register_forward_hook(make_hook("k")))
    handles.append(attn.v_proj.register_forward_hook(make_hook("v")))
    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def patch_answer_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_kv: Dict[str, torch.Tensor],
    repair_kv: Dict[str, torch.Tensor],
    base_positions: List[int],
    repair_positions: List[int],
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

    base_k_mean = mean_at(base_kv["k"], base_positions)
    base_v_mean = mean_at(base_kv["v"], base_positions)
    repair_k_mean = mean_at(repair_kv["k"], repair_positions)
    repair_v_mean = mean_at(repair_kv["v"], repair_positions)
    if base_k_mean is None or base_v_mean is None or repair_k_mean is None or repair_v_mean is None:
        return -100.0

    dk = repair_k_mean - base_k_mean
    dv = repair_v_mean - base_v_mean
    if mode == "kv_random":
        dk = random_same_norm(dk, seed=seed)
        dv = random_same_norm(dv, seed=seed + 17)

    handles = []

    def make_patch(kind: str, delta: torch.Tensor):
        def hook(_module, _inputs, output):
            y = output.clone()
            d = delta.to(device=y.device, dtype=y.dtype)
            for pos in base_positions:
                if 0 <= pos < y.shape[1]:
                    y[0, pos, :] = y[0, pos, :] + d
            return y
        return hook

    if mode in ("k_delta", "kv_delta", "kv_random"):
        handles.append(attn.k_proj.register_forward_hook(make_patch("k", dk)))
    if mode in ("v_delta", "kv_delta", "kv_random"):
        handles.append(attn.v_proj.register_forward_hook(make_patch("v", dv)))

    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                p = start + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def patched_score_map(
    model,
    tokenizer,
    device,
    case: Dict,
    values: List[str],
    layer_idx: int,
    group: str,
    mode: str,
    cache: Dict[str, Dict],
    seed: int,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ai, ans in enumerate(values):
        base_pos = cache[ans]["base_pos"].get(group, [])
        repair_pos = cache[ans]["repair_pos"].get(group, [])
        scores[ans] = patch_answer_score(
            model,
            tokenizer,
            device,
            case["base_prompt"],
            ans,
            layer_idx,
            cache[ans]["base_kv"][layer_idx],
            cache[ans]["repair_kv"][layer_idx],
            base_pos,
            repair_pos,
            mode,
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
            "group": items[0]["group"],
            "mode": items[0]["mode"],
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
    log("Best source K/V patches:")
    for item in best[:18]:
        log(
            f"  L{item['layer']} {item['group']} {item['mode']}: "
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
        log(f"{args.model}: layers={info.n_layers}, source_layers={layers_to_scan}, cases={len(cases)}")

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

            cache = {}
            for ans in values:
                cache[ans] = {
                    "base_kv": {},
                    "repair_kv": {},
                    "base_pos": source_positions(tokenizer, case, case["base_prompt"], case["relation"], ans, si + 11),
                    "repair_pos": source_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"], ans, si + 11),
                }
                for li in layers_to_scan:
                    cache[ans]["base_kv"][li] = collect_kv(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair_kv"][li] = collect_kv(model, tokenizer, device, case["repair_prompt"], ans, li)

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
                for group in SOURCE_GROUPS:
                    for mode in PATCH_MODES:
                        key = f"L{li}|{group}|{mode}"
                        scores = patched_score_map(
                            model, tokenizer, device, case, values, li, group, mode, cache,
                            seed=si * 1009 + li * 31 + len(group) * 17 + len(mode),
                        )
                        patched = winner_stats(scores, case["correct"])
                        row["patches"][key] = {
                            "layer": li,
                            "group": group,
                            "mode": mode,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                        }
            rows.append(row)

        return {
            "phase": 608,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "source_groups": SOURCE_GROUPS,
            "patch_modes": PATCH_MODES,
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
    out_path = out_dir / f"phase608_{args.model}_attention_source_kv_decomposition_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
