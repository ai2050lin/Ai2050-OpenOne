#!/usr/bin/env python3
"""
Phase 663: Projection-Specific Causal Intervention Audit.

Tests Phase 662's split diagnosis with readout-level counterfactuals:

1. norm-neutralized pair readout:
   If space wins in actual logits but loses when unembedding norms are removed,
   the failure is consistent with projection norm advantage.

2. direction correction after final_norm:
   Move final_norm output along W_correct - W_competitor and measure whether
   correct_prefix becomes top1. This is a readout-level causal intervention,
   not a full generation-path repair.
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
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS, build_site_patch, collect_caches  # noqa: E402
from phase659_final_top1_barrier_readout_audit import TASK_ORDER, load_best_combos, token_category  # noqa: E402
from phase661_last_writer_combo_generation_closure import greedy_generate, load_last_writer_specs, probe_mode  # noqa: E402
from phase662_residual_to_lmhead_projection_barrier_audit import (  # noqa: E402
    get_unembed,
    logits_from_state,
    projection_diag,
    readout_metric,
)


OUT_ROOT = Path("results/glm5_phase663_projection_specific_causal_intervention_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pair_norm_neutral_diag(
    state: torch.Tensor,
    unembed: torch.Tensor,
    prefix_id: int,
    competitor_id: int,
    prefix_logit: float,
    competitor_logit: float,
) -> Dict:
    h = state.detach().float().cpu()
    wc = unembed[prefix_id].float()
    wt = unembed[competitor_id].float()
    actual_gap = float(competitor_logit - prefix_logit)
    correct_cos = float(F.cosine_similarity(h, wc, dim=0).item())
    competitor_cos = float(F.cosine_similarity(h, wt, dim=0).item())
    neutral_gap = competitor_cos - correct_cos
    return {
        "actual_gap": actual_gap,
        "norm_neutral_cos_gap": neutral_gap,
        "norm_neutral_pair_flips_to_correct": bool(actual_gap > 0 and neutral_gap < 0),
        "actual_competitor_wins": bool(actual_gap > 0),
        "correct_cos": correct_cos,
        "competitor_cos": competitor_cos,
        "correct_w_norm": float(wc.norm().item()),
        "competitor_w_norm": float(wt.norm().item()),
        "competitor_norm_advantage": float(wt.norm().item() - wc.norm().item()),
    }


def direction_correction(
    model,
    tokenizer,
    state: torch.Tensor,
    unembed: torch.Tensor,
    prefix_id: int,
    competitor_id: int,
    value_prefix_ids,
    scales: List[float],
    top_k: int,
) -> Dict:
    h = state.detach().float().cpu()
    wc = unembed[prefix_id].float()
    wt = unembed[competitor_id].float()
    diff = wc - wt
    diff_norm = float(diff.norm().item())
    if diff_norm <= 1e-8:
        return {"diff_norm": diff_norm, "scales": []}
    base_logits = logits_from_state(model, h)
    base_gap = float(base_logits[competitor_id].item() - base_logits[prefix_id].item())
    needed_unit_delta = max(0.0, base_gap / diff_norm)
    unit = diff / diff_norm
    out = []
    for scale in scales:
        moved = h + (needed_unit_delta * scale) * unit
        logits = logits_from_state(model, moved)
        metric = readout_metric(tokenizer, logits, prefix_id, value_prefix_ids, top_k)
        out.append({
            "scale": scale,
            "move_norm": needed_unit_delta * scale,
            "prefix_rank": metric["prefix_rank"],
            "top1_category": metric["top1"]["category"],
            "top1_text": metric["top1"]["text"],
            "top1_gap": metric["top1_gap"],
            "correct_top1": bool(metric["top1"]["category"] == "correct_prefix"),
        })
    return {
        "diff_norm": diff_norm,
        "base_gap": base_gap,
        "needed_unit_delta": needed_unit_delta,
        "scales": out,
    }


def continuation_tag(row_eval: Dict, post: Dict) -> str:
    if row_eval["exact_correct"]:
        return "exact_correct"
    if post["top1"]["category"] == "correct_prefix":
        return "correct_prefix_but_generation_wrong"
    return "first_token_competition_failure"


def summarize(rows: List[Dict]) -> Dict:
    by_site: Dict[Tuple, Dict] = {}
    failures: Dict[Tuple, Dict] = {}
    scale_hits: Dict[Tuple, Dict] = {}
    continuation: Dict[Tuple, Dict] = {}

    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"])
        item = by_site.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "n": 0,
            "exact": 0,
            "correct_top1": 0,
            "sum_rank": 0.0,
            "sum_gap": 0.0,
            "top1_category": {},
            "continuation_tag": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["correct_top1"] += int(row["post"]["top1"]["category"] == "correct_prefix")
        item["sum_rank"] += row["post"]["prefix_rank"]
        item["sum_gap"] += row["post"]["top1_gap"]
        cat = row["post"]["top1"]["category"]
        tag = row["continuation_tag"]
        item["top1_category"][cat] = item["top1_category"].get(cat, 0) + 1
        item["continuation_tag"][tag] = item["continuation_tag"].get(tag, 0) + 1

        if tag == "correct_prefix_but_generation_wrong":
            ckey = (row["pair_task"], row["site"], row["combo_name"])
            c = continuation.setdefault(ckey, {
                "pair_task": row["pair_task"],
                "site": row["site"],
                "combo_name": row["combo_name"],
                "n": 0,
                "generation_text": {},
            })
            c["n"] += 1
            text = row["generation_text"].replace("\n", "\\n")
            c["generation_text"][text] = c["generation_text"].get(text, 0) + 1

        if row["eval"]["exact_correct"] or row["post"]["top1"]["category"] == "correct_prefix":
            continue

        fkey = (row["pair_task"], row["post"]["top1"]["category"])
        f = failures.setdefault(fkey, {
            "pair_task": row["pair_task"],
            "top1_category": row["post"]["top1"]["category"],
            "n": 0,
            "top1_text": {},
            "norm_neutral_flips": 0,
            "sum_actual_gap": 0.0,
            "sum_norm_neutral_gap": 0.0,
            "sum_correct_cos": 0.0,
            "sum_competitor_cos": 0.0,
            "sum_norm_adv": 0.0,
            "sum_needed_unit_delta": 0.0,
        })
        f["n"] += 1
        text = row["post"]["top1"]["text"].replace("\n", "\\n")
        f["top1_text"][text] = f["top1_text"].get(text, 0) + 1
        f["norm_neutral_flips"] += int(row["norm_neutral"]["norm_neutral_pair_flips_to_correct"])
        f["sum_actual_gap"] += row["norm_neutral"]["actual_gap"]
        f["sum_norm_neutral_gap"] += row["norm_neutral"]["norm_neutral_cos_gap"]
        f["sum_correct_cos"] += row["norm_neutral"]["correct_cos"]
        f["sum_competitor_cos"] += row["norm_neutral"]["competitor_cos"]
        f["sum_norm_adv"] += row["norm_neutral"]["competitor_norm_advantage"]
        f["sum_needed_unit_delta"] += row["direction_correction"]["needed_unit_delta"]

        for s in row["direction_correction"]["scales"]:
            skey = (row["pair_task"], row["post"]["top1"]["category"], s["scale"])
            sh = scale_hits.setdefault(skey, {
                "pair_task": row["pair_task"],
                "top1_category": row["post"]["top1"]["category"],
                "scale": s["scale"],
                "n": 0,
                "correct_top1": 0,
                "sum_rank": 0.0,
                "sum_gap": 0.0,
                "top1_after": {},
            })
            sh["n"] += 1
            sh["correct_top1"] += int(s["correct_top1"])
            sh["sum_rank"] += s["prefix_rank"]
            sh["sum_gap"] += s["top1_gap"]
            sh["top1_after"][s["top1_category"]] = sh["top1_after"].get(s["top1_category"], 0) + 1

    by_site_out = []
    for item in by_site.values():
        n = max(1, item["n"])
        r = dict(item)
        r["exact_rate"] = item["exact"] / n
        r["correct_top1_rate"] = item["correct_top1"] / n
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_gap"] = item["sum_gap"] / n
        r["top1_category"] = dict(sorted(item["top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["continuation_tag"] = dict(sorted(item["continuation_tag"].items(), key=lambda kv: kv[1], reverse=True))
        by_site_out.append(r)

    failure_out = []
    for f in failures.values():
        n = max(1, f["n"])
        r = dict(f)
        r["norm_neutral_flip_rate"] = f["norm_neutral_flips"] / n
        r["mean_actual_gap"] = f["sum_actual_gap"] / n
        r["mean_norm_neutral_cos_gap"] = f["sum_norm_neutral_gap"] / n
        r["mean_correct_cos"] = f["sum_correct_cos"] / n
        r["mean_competitor_cos"] = f["sum_competitor_cos"] / n
        r["mean_competitor_norm_advantage"] = f["sum_norm_adv"] / n
        r["mean_needed_unit_delta"] = f["sum_needed_unit_delta"] / n
        r["top1_text"] = dict(sorted(f["top1_text"].items(), key=lambda kv: kv[1], reverse=True))
        failure_out.append(r)

    scale_out = []
    for sh in scale_hits.values():
        n = max(1, sh["n"])
        r = dict(sh)
        r["correct_top1_rate"] = sh["correct_top1"] / n
        r["mean_rank"] = sh["sum_rank"] / n
        r["mean_gap"] = sh["sum_gap"] / n
        r["top1_after"] = dict(sorted(sh["top1_after"].items(), key=lambda kv: kv[1], reverse=True))
        scale_out.append(r)

    continuation_out = []
    for c in continuation.values():
        r = dict(c)
        r["generation_text"] = dict(sorted(c["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:12])
        continuation_out.append(r)

    by_site_out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"]))
    failure_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["top1_category"]))
    scale_out.sort(key=lambda r: (r["pair_task"], r["top1_category"], r["scale"]))
    continuation_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["site"], r["combo_name"]))
    return {
        "by_site": by_site_out,
        "failure_norm_neutral": failure_out,
        "direction_correction_by_scale": scale_out,
        "continuation_failures": continuation_out,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        unembed = get_unembed(model)
        combo_specs = load_best_combos(args.model, args.max_per_task)
        last_map = load_last_writer_specs(args.model, args.max_last_writers)
        site_specs = {s["name"]: s for s in SITE_SPECS[args.model]}
        needed_sites = [site_specs[c["site"]] for c in combo_specs if c["site"] in site_specs]
        site_layers = sorted({li for s in needed_sites for li in s["layers"] if 0 <= li < info.n_layers})
        site_components = sorted({c for s in needed_sites for c in s["components"]})
        site_positions = sorted({p for s in needed_sites for p in s["positions"]})
        values = CANDIDATE_VALUES[:4]
        value_prefix_ids = {answer_ids(tokenizer, v)[0] for v in values}
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        relation_pool = sorted({c["relation"] for c in raw_cases})
        selected, selection_stats = select_cases(
            model, tokenizer, device, raw_cases, values, args.max_cases, relation_pool
        )
        rows = []
        examples = []
        filtered = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0}
        log(f"{args.model}: selected={len(selected)}, combos={combo_specs}, last_map={last_map}")

        scales = [float(x) for x in args.direction_scales.split(",") if x.strip()]
        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            prefix_id = correct_ids[0]
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units_all = position_units(tokenizer, value_prompt, case, value_relation, value_intent)
            value_units = {p: value_units_all.get(p, []) for p in site_positions}
            value_caches = collect_caches(model, tokenizer, device, value_prompt, value_units, site_layers, site_components)

            for task_i, task in enumerate(TASK_ORDER):
                task_combos = [c for c in combo_specs if c["pair_task"] == task]
                if not task_combos:
                    continue
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units_all = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_units = {p: task_units_all.get(p, []) for p in site_positions}
                task_caches = collect_caches(model, tokenizer, device, task_prompt, task_units, site_layers, site_components)
                site_patch_cache = {}
                for combo_spec in task_combos:
                    site = site_specs[combo_spec["site"]]
                    if combo_spec["site"] not in site_patch_cache:
                        layers0 = [li for li in site["layers"] if 0 <= li < info.n_layers]
                        patches, stats = build_site_patch(
                            task_caches,
                            value_caches,
                            task_units,
                            value_units,
                            site,
                            layers0,
                            item["sample_idx"] * 1009 + task_i * 199,
                        )
                        for k, v in stats.items():
                            filtered[k] += v
                        site_patch_cache[combo_spec["site"]] = patches
                    patches = site_patch_cache[combo_spec["site"]]
                    if not patches:
                        continue
                    last_writers = last_map.get((task, combo_spec["site"], combo_spec["combo_name"]), [])
                    combo = combo_spec["components"]
                    probe = probe_mode(model, tokenizer, device, task_prompt, patches, combo, last_writers)
                    gen = greedy_generate(
                        model, tokenizer, device, task_prompt, args.max_new_tokens, patches, combo, last_writers
                    )
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    post_logits = probe["logits"]
                    post = readout_metric(tokenizer, post_logits, prefix_id, value_prefix_ids, args.top_k)
                    comp_id = post["top1"]["id"]
                    post_diag = projection_diag(
                        probe["final_norm_output"],
                        unembed,
                        prefix_id,
                        comp_id,
                        float(post_logits[prefix_id].item()),
                        float(post_logits[comp_id].item()),
                    )
                    norm_neutral = pair_norm_neutral_diag(
                        probe["final_norm_output"],
                        unembed,
                        prefix_id,
                        comp_id,
                        float(post_logits[prefix_id].item()),
                        float(post_logits[comp_id].item()),
                    )
                    if post["top1"]["category"] == "correct_prefix":
                        corr = {"diff_norm": 0.0, "base_gap": 0.0, "needed_unit_delta": 0.0, "scales": []}
                    else:
                        corr = direction_correction(
                            model,
                            tokenizer,
                            probe["final_norm_output"],
                            unembed,
                            prefix_id,
                            comp_id,
                            value_prefix_ids,
                            scales,
                            args.top_k,
                        )
                    row = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "site": combo_spec["site"],
                        "combo_name": combo_spec["combo_name"],
                        "combo": combo,
                        "last_writers": last_writers,
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "post": post,
                        "post_diag": post_diag,
                        "norm_neutral": norm_neutral,
                        "direction_correction": corr,
                        "eval": ev,
                        "continuation_tag": continuation_tag(ev, post),
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Failure norm-neutralized pair results:")
        for r in summary["failure_norm_neutral"]:
            log(
                f"  {r['pair_task']} top1={r['top1_category']} n={r['n']} "
                f"actual_gap={r['mean_actual_gap']:.3f} neutral_gap={r['mean_norm_neutral_cos_gap']:.4f} "
                f"flip_rate={r['norm_neutral_flip_rate']:.2f} norm_adv={r['mean_competitor_norm_advantage']:.3f}"
            )
        log("Direction correction results:")
        for r in summary["direction_correction_by_scale"]:
            log(
                f"  {r['pair_task']} top1={r['top1_category']} scale={r['scale']} n={r['n']} "
                f"correct_top1_rate={r['correct_top1_rate']:.2f} mean_rank={r['mean_rank']:.2f}"
            )
        return {
            "phase": 663,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "combo_specs": combo_specs,
            "last_writer_map": {str(k): v for k, v in last_map.items()},
            "tasks": TASK_ORDER,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "direction_scales": scales,
            "n_raw_cases": len(raw_cases),
            "n_selected_items": len(selected),
            "n_rows": len(rows),
            "max_cases": args.max_cases,
            "selection_stats": selection_stats,
            "filtered": filtered,
            "summary": summary,
            "examples": examples,
            "rows": rows if args.save_rows else examples,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--max-last-writers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--direction-scales", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=240)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 12
        args.max_cases = 1
        args.max_per_task = 1
        args.max_last_writers = 1
        args.top_k = min(args.top_k, 20)
        args.max_new_tokens = min(args.max_new_tokens, 4)
        args.direction_scales = "1.0,2.0"
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 48)
        args.max_samples = max(args.max_samples, 384)
        args.max_cases = max(args.max_cases, 32)
        args.max_per_task = max(args.max_per_task, 2)
        args.max_last_writers = min(max(args.max_last_writers, 2), 2)
        args.top_k = max(args.top_k, 30)
        args.max_new_tokens = min(max(args.max_new_tokens, 6), 6)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase663_{args.model}_projection_specific_causal_intervention_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
