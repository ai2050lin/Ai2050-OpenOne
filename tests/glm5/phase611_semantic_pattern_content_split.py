#!/usr/bin/env python3
"""
Phase 611: Semantic Pattern / Content Split
语义源组模式/内容拆分

Base and repair prompts are not token-index aligned. This phase avoids invalid
alpha_repair * V_base index mixing by using semantic source groups. It tests
whether top-head repair can be approximated by group-level attention pattern,
group-level V content, or their coupling.
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
from typing import Dict, List, Tuple

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
from phase609_query_oproj_head_decomposition import (  # noqa: E402
    answer_ids,
    default_layers,
    n_heads_for,
    parse_layers,
)
from phase610_head_cumulative_mixture import TOP_HEADS  # noqa: E402


OUT_ROOT = Path("results/glm5_phase611_semantic_pattern_content_split")
GROUPS = [
    "answer_prefix",
    "prompt_last",
    "rule_value",
    "rule_relation",
    "query_category",
    "query_relation",
    "query_object",
    "other",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def source_group_positions(tokenizer, case: Dict, prompt: str, relation_for_rule: str, answer: str) -> Dict[str, List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    pos = len(prompt_ids)
    max_pos = pos
    raw = {
        "answer_prefix": pos if ans_ids else None,
        "prompt_last": len(prompt_ids) - 1,
        "rule_value": token_pos_after_substring(tokenizer, prompt, case["correct"], "first"),
        "rule_relation": token_pos_after_substring(tokenizer, prompt, relation_for_rule, "first"),
        "query_category": token_pos_after_substring(tokenizer, prompt, case["category"], "last"),
        "query_relation": token_pos_after_substring(tokenizer, prompt, case["relation"], "last"),
        "query_object": token_pos_after_substring(tokenizer, prompt, case["object"], "last"),
    }
    used = set()
    groups: Dict[str, List[int]] = {}
    for name in GROUPS:
        if name == "other":
            continue
        p = raw.get(name)
        vals = []
        if p is not None and 0 <= int(p) <= max_pos and int(p) not in used:
            vals = [int(p)]
            used.add(int(p))
        groups[name] = vals
    groups["other"] = [p for p in range(max_pos + 1) if p not in used]
    return groups


def repeat_v_to_heads(v_proj: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    seq, dim = v_proj.shape
    n_kv = max(1, dim // head_dim)
    v = v_proj[:, : n_kv * head_dim].reshape(seq, n_kv, head_dim).transpose(0, 1).contiguous()
    if n_kv < n_heads:
        rep = n_heads // n_kv
        v = v.unsqueeze(1).expand(n_kv, rep, seq, head_dim).reshape(n_heads, seq, head_dim)
    elif n_kv > n_heads:
        v = v[:n_heads]
    return v.float().cpu()


def collect_parts(model, tokenizer, device, prompt: str, answer: str, layer_idx: int) -> Dict:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def v_hook(_module, _inputs, output):
        captured["v_proj"] = output[0].detach().float().cpu()

    def o_pre_hook(_module, inputs):
        x = inputs[0]
        captured["o_input"] = x[0, pos].detach().float().cpu()

    handles.append(attn.v_proj.register_forward_hook(v_hook))
    handles.append(attn.o_proj.register_forward_pre_hook(o_pre_hook))
    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                output_attentions=True,
                return_dict=True,
            )
        if out.attentions is None or out.attentions[layer_idx] is None:
            raise RuntimeError("output_attentions did not return attention weights")
        captured["alpha"] = out.attentions[layer_idx][0, :, pos, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def group_stats(alpha: torch.Tensor, v_heads: torch.Tensor, groups: Dict[str, List[int]], head: int) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    seq = alpha.shape[-1]
    for name in GROUPS:
        positions = [p for p in groups.get(name, []) if 0 <= p < seq and p < v_heads.shape[1]]
        if not positions:
            stats[name] = {
                "mass": 0.0,
                "base_weighted_v": torch.zeros(v_heads.shape[-1]),
                "mean_v": torch.zeros(v_heads.shape[-1]),
            }
            continue
        a = alpha[head, positions].float()
        mass = float(a.sum().item())
        vals = v_heads[head, positions].float()
        if mass > 1e-8:
            weighted = (a[:, None] * vals).sum(dim=0) / max(mass, 1e-8)
        else:
            weighted = vals.mean(dim=0)
        stats[name] = {
            "mass": mass,
            "base_weighted_v": weighted.cpu(),
            "mean_v": vals.mean(dim=0).cpu(),
        }
    return stats


def reconstruct_head(
    mode: str,
    head: int,
    base_alpha: torch.Tensor,
    base_v: torch.Tensor,
    repair_alpha: torch.Tensor,
    repair_v: torch.Tensor,
    base_groups: Dict[str, List[int]],
    repair_groups: Dict[str, List[int]],
    base_actual: torch.Tensor,
    repair_actual: torch.Tensor,
) -> torch.Tensor:
    if mode == "actual":
        return repair_actual
    base_stats = group_stats(base_alpha, base_v, base_groups, head)
    repair_stats = group_stats(repair_alpha, repair_v, repair_groups, head)
    z = torch.zeros_like(base_actual)
    for name in GROUPS:
        if mode == "content":
            mass = base_stats[name]["mass"]
            vec = repair_stats[name]["mean_v"]
        elif mode == "pattern":
            mass = repair_stats[name]["mass"]
            vec = base_stats[name]["base_weighted_v"]
        elif mode == "pattern_content":
            mass = repair_stats[name]["mass"]
            vec = repair_stats[name]["mean_v"]
        else:
            raise ValueError(mode)
        z = z + float(mass) * vec.to(dtype=z.dtype)
    return z.float().cpu()


def patch_answer_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    layer_idx: int,
    base_o: torch.Tensor,
    target_o: torch.Tensor,
    heads: List[int],
    n_heads: int,
    seed: int,
    random_mode: bool = False,
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
        d = target_o[start:end].float().cpu() - base_o[start:end].float().cpu()
        if random_mode:
            d = random_same_norm(d, seed=seed + hi * 101)
        deltas[hi] = d

    def hook(_module, inputs):
        x = inputs[0]
        x_new = x.clone()
        if pos < x_new.shape[1]:
            for hi, d in deltas.items():
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


def build_target_o(base_parts, repair_parts, base_groups, repair_groups, heads, n_heads, mode):
    base_o = base_parts["o_input"].float().cpu()
    target = base_o.clone()
    width = base_o.numel()
    head_dim = width // max(1, n_heads)
    base_v = repeat_v_to_heads(base_parts["v_proj"], n_heads, head_dim)
    repair_v = repeat_v_to_heads(repair_parts["v_proj"], n_heads, head_dim)
    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        z = reconstruct_head(
            mode, hi,
            base_parts["alpha"], base_v,
            repair_parts["alpha"], repair_v,
            base_groups, repair_groups,
            base_o[start:end],
            repair_parts["o_input"].float().cpu()[start:end],
        )
        target[start:end] = z
    return target


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
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_correct_delta"] += m["correct_delta"]
            entry["mean_wrong_delta"] += m["old_top_wrong_delta"]
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        by_patch[key] = entry
    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:80]
    log("Best semantic pattern/content patches:")
    for item in best[:18]:
        log(f"  L{item['layer']} {item['mode']} heads={item['heads']}: switch={item['switch']}/{item['n']} margin={item['mean_margin_gain']:.3f}")
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

            cache = {}
            for ans in values:
                cache[ans] = {"base": {}, "repair": {}, "base_groups": {}, "repair_groups": {}}
                cache[ans]["base_groups"] = source_group_positions(tokenizer, case, case["base_prompt"], case["relation"], ans)
                cache[ans]["repair_groups"] = source_group_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"], ans)
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_parts(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_parts(model, tokenizer, device, case["repair_prompt"], ans, li)

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
                heads = [h for h in TOP_HEADS.get(args.model, list(range(min(4, n_heads))))[: args.top_k] if h < n_heads]
                for mode in ["actual", "content", "pattern", "pattern_content", "random"]:
                    key = f"L{li}|top{len(heads)}|{mode}"
                    scores = {}
                    for ans_i, ans in enumerate(values):
                        base_parts = cache[ans]["base"][li]
                        repair_parts = cache[ans]["repair"][li]
                        if mode == "random":
                            target_o = repair_parts["o_input"].float().cpu()
                            random_mode = True
                        else:
                            target_o = build_target_o(
                                base_parts,
                                repair_parts,
                                cache[ans]["base_groups"],
                                cache[ans]["repair_groups"],
                                heads,
                                n_heads,
                                mode,
                            )
                            random_mode = False
                        scores[ans] = patch_answer_score(
                            model, tokenizer, device, case["base_prompt"], ans, li,
                            base_parts["o_input"].float().cpu(), target_o, heads, n_heads,
                            seed=si * 1009 + ans_i * 997 + len(mode),
                            random_mode=random_mode,
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
            "phase": 611,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "top_k": args.top_k,
            "top_heads": TOP_HEADS.get(args.model, []),
            "groups": GROUPS,
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
        args.n_tables = max(args.n_tables, 12)
        args.max_samples = max(args.max_samples, 96)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase611_{args.model}_semantic_pattern_content_split_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
