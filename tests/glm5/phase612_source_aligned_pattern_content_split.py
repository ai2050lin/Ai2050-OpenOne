#!/usr/bin/env python3
"""
Phase 612: Source-Aligned Pattern / Content Split
源序列对齐模式/内容拆分

Phase 611 showed that coarse semantic source groups cannot reconstruct the true
top-head mixture. This phase constructs base/repair prompt pairs with aligned
token positions, then directly combines attention pattern alpha and V content:

  bb = alpha_base   * V_base
  rb = alpha_repair * V_base
  br = alpha_base   * V_repair
  rr = alpha_repair * V_repair

Only prompt pairs with identical token length are used, so token-index mixing is
well-defined.
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
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import (  # noqa: E402
    CANDIDATE_CATEGORIES,
    CANDIDATE_OBJECTS,
    CANDIDATE_RELATIONS,
    CANDIDATE_VALUES,
    build_cat_rel_truth_tables,
    build_oc_truth_tables,
    load_model_flash,
)
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
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


OUT_ROOT = Path("results/glm5_phase612_source_aligned_pattern_content_split")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_aligned_cases(n_tables: int, max_samples: int):
    objects = CANDIDATE_OBJECTS[:8]
    categories = CANDIDATE_CATEGORIES[:4]
    relations = CANDIDATE_RELATIONS[:2]
    values = CANDIDATE_VALUES[:4]
    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    count = 0
    for tt_idx, oc_table in enumerate(oc_tables):
        crv_table = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)
        rng = random.Random(tt_idx * 100)
        oc_rules = list(oc_table.items())
        rng.shuffle(oc_rules)
        crv_rules = list(crv_table.items())
        rng.shuffle(crv_rules)
        oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]
        crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]
        rule_block = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
        for obj in objects:
            cat = oc_table[obj]
            for rel in relations:
                correct_val = crv_table[(cat, rel)]
                common = rule_block + f"\n\n{obj} belongs to {cat}.\nQuestion: "
                base_prompt = common + f"{obj} {rel} ?\nAnswer:"
                repair_prompt = common + f"{cat} {rel} ?\nAnswer:"
                yield {
                    "tt_idx": tt_idx,
                    "object": obj,
                    "relation": rel,
                    "category": cat,
                    "correct": correct_val,
                    "candidates": values,
                    "base_prompt": base_prompt,
                    "repair_prompt": repair_prompt,
                }
                count += 1
                if count >= max_samples:
                    return


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def repeat_v_to_heads(v_proj: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    seq, dim = v_proj.shape
    n_kv = max(1, dim // head_dim)
    v = v_proj[:, : n_kv * head_dim].reshape(seq, n_kv, head_dim).transpose(0, 1).contiguous()
    if n_kv < n_heads:
        rep = max(1, n_heads // n_kv)
        v = v.unsqueeze(1).expand(n_kv, rep, seq, head_dim).reshape(n_kv * rep, seq, head_dim)
        if v.shape[0] < n_heads:
            v = torch.cat([v, v[-1:].expand(n_heads - v.shape[0], seq, head_dim)], dim=0)
        v = v[:n_heads]
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
        captured["o_input"] = inputs[0][0, pos].detach().float().cpu()

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


def strict_head_z(alpha: torch.Tensor, v_heads: torch.Tensor, head: int, head_dim: int) -> torch.Tensor:
    seq = min(alpha.shape[-1], v_heads.shape[1])
    if seq <= 0:
        return torch.zeros(head_dim)
    a = alpha[head, :seq].float()
    v = v_heads[head, :seq].float()
    return (a[:, None] * v).sum(dim=0).float().cpu()


def build_target_o(base_parts: Dict, repair_parts: Dict, heads: List[int], n_heads: int, mode: str) -> torch.Tensor:
    base_o = base_parts["o_input"].float().cpu()
    repair_o = repair_parts["o_input"].float().cpu()
    target = base_o.clone()
    width = base_o.numel()
    head_dim = width // max(1, n_heads)
    base_v = repeat_v_to_heads(base_parts["v_proj"], n_heads, head_dim)
    repair_v = repeat_v_to_heads(repair_parts["v_proj"], n_heads, head_dim)

    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        if mode == "actual":
            z = repair_o[start:end]
        elif mode == "bb":
            z = strict_head_z(base_parts["alpha"], base_v, hi, end - start)
        elif mode == "rb_pattern":
            z = strict_head_z(repair_parts["alpha"], base_v, hi, end - start)
        elif mode == "br_content":
            z = strict_head_z(base_parts["alpha"], repair_v, hi, end - start)
        elif mode == "rr_pattern_content":
            z = strict_head_z(repair_parts["alpha"], repair_v, hi, end - start)
        else:
            raise ValueError(mode)
        target[start:end] = z[: end - start]
    return target


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
    log("Best source-aligned pattern/content patches:")
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
                ids_base = full_ids(tokenizer, case["base_prompt"], ans)
                ids_repair = full_ids(tokenizer, case["repair_prompt"], ans)
                if len(ids_base) != len(ids_repair):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                for li in layers_to_scan:
                    cache[ans]["base"][li] = collect_parts(model, tokenizer, device, case["base_prompt"], ans, li)
                    cache[ans]["repair"][li] = collect_parts(model, tokenizer, device, case["repair_prompt"], ans, li)

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
                for mode in ["bb", "rb_pattern", "br_content", "rr_pattern_content", "actual", "random_actual_norm"]:
                    key = f"L{li}|top{len(heads)}|{mode}"
                    scores = {}
                    for ans_i, ans in enumerate(values):
                        base_parts = cache[ans]["base"][li]
                        repair_parts = cache[ans]["repair"][li]
                        if mode == "random_actual_norm":
                            target_o = build_target_o(base_parts, repair_parts, heads, n_heads, "actual")
                            random_mode = True
                        else:
                            target_o = build_target_o(base_parts, repair_parts, heads, n_heads, mode)
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
            "phase": 612,
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
            "prompt_design": "same rule block and answer prefix; only query slot object/category differs; prompt token length must match",
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
    out_path = out_dir / f"phase612_{args.model}_source_aligned_pattern_content_split_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
