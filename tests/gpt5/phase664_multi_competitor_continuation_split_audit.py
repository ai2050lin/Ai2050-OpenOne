#!/usr/bin/env python3
"""
Phase 664: Multi-Competitor Readout and Continuation Split Audit.

Extends Phase 663 from pairwise correction to:

1. multi-competitor readout margin:
   correct_prefix vs max(space, newline, word, explanation)

2. multi-competitor direction correction:
   move final_norm output against all currently winning competitor categories

3. continuation split:
   when first token is already correct_prefix but exact generation is wrong,
   inspect token1/token2 after forcing the correct first token.
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval  # noqa: E402
from phase630_distributed_format_route_multisource import install_source_patch_hooks  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS, build_site_patch, collect_caches  # noqa: E402
from phase659_final_top1_barrier_readout_audit import TASK_ORDER, load_best_combos, token_category  # noqa: E402
from phase661_last_writer_combo_generation_closure import greedy_generate, load_last_writer_specs, probe_mode  # noqa: E402
from phase662_residual_to_lmhead_projection_barrier_audit import get_unembed, logits_from_state, readout_metric  # noqa: E402


OUT_ROOT = Path("results/glm5_phase664_multi_competitor_continuation_split_audit")
TARGET_CATEGORIES = ["space", "newline", "word", "explanation"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topk_by_category(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> Dict:
    topv, topi = torch.topk(logits.float(), k=top_k)
    prefix_logit = float(logits[prefix_id].item())
    best = {}
    top_rows = []
    for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        tid = int(i)
        text = tokenizer.decode([tid])
        cat = token_category(text, tid, prefix_id, value_prefix_ids)
        row = {"rank": rank, "id": tid, "text": text, "category": cat, "logit": float(v), "gap": float(v - prefix_logit)}
        top_rows.append(row)
        if cat not in best:
            best[cat] = row
    competitors = [best[c] for c in TARGET_CATEGORIES if c in best and best[c]["gap"] > 0]
    if competitors:
        max_comp = max(competitors, key=lambda r: r["gap"])
        multi_margin = -float(max_comp["gap"])
    else:
        max_comp = None
        multi_margin = float(prefix_logit - max(float(v) for v in topv.tolist()))
    return {
        "prefix_logit": prefix_logit,
        "best_by_category": best,
        "winning_competitors": competitors,
        "max_competitor": max_comp,
        "multi_margin": multi_margin,
        "top_rows": top_rows[:12],
    }


def multi_competitor_correction(
    model,
    tokenizer,
    state: torch.Tensor,
    unembed: torch.Tensor,
    prefix_id: int,
    competitors: List[Dict],
    value_prefix_ids,
    scales: List[float],
    top_k: int,
) -> Dict:
    if not competitors:
        return {"n_competitors": 0, "competitor_categories": [], "scales": []}
    h = state.detach().float().cpu()
    wc = unembed[prefix_id].float()
    move = torch.zeros_like(h)
    used = []
    for comp in competitors:
        wt = unembed[int(comp["id"])].float()
        diff = wc - wt
        diff_norm = float(diff.norm().item())
        if diff_norm <= 1e-8:
            continue
        needed = max(0.0, float(comp["gap"]) / diff_norm)
        move = move + needed * (diff / diff_norm)
        used.append({"category": comp["category"], "text": comp["text"], "gap": comp["gap"], "needed": needed})
    move_norm = float(move.norm().item())
    out = []
    for scale in scales:
        moved = h + scale * move
        logits = logits_from_state(model, moved)
        metric = readout_metric(tokenizer, logits, prefix_id, value_prefix_ids, top_k)
        multi = topk_by_category(tokenizer, logits, prefix_id, value_prefix_ids, top_k)
        out.append({
            "scale": scale,
            "move_norm": move_norm * scale,
            "prefix_rank": metric["prefix_rank"],
            "top1_category": metric["top1"]["category"],
            "top1_text": metric["top1"]["text"],
            "top1_gap": metric["top1_gap"],
            "correct_top1": bool(metric["top1"]["category"] == "correct_prefix"),
            "multi_margin": multi["multi_margin"],
            "max_competitor_category": None if multi["max_competitor"] is None else multi["max_competitor"]["category"],
        })
    return {"n_competitors": len(used), "competitor_categories": used, "move_norm": move_norm, "scales": out}


def continuation_after_prefix(
    model,
    tokenizer,
    device,
    prompt: str,
    source_patches: Dict,
    prefix_id: int,
    correct_ids: List[int],
    steps: int,
) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False) + [int(prefix_id)]
    generated = []
    rows = []
    handles = []
    if source_patches:
        handles = install_source_patch_hooks(model, tokenizer, prompt, source_patches)
    try:
        with torch.inference_mode():
            for step in range(steps):
                logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                expected_id = int(correct_ids[step + 1]) if step + 1 < len(correct_ids) else None
                topv, topi = torch.topk(logits, k=8)
                top_rows = []
                for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
                    tid = int(i)
                    top_rows.append({"rank": rank, "id": tid, "text": tokenizer.decode([tid]), "logit": float(v)})
                if expected_id is None:
                    expected_rank = None
                    expected_logit = None
                    expected_text = None
                else:
                    expected_rank = 1 + int((logits > logits[expected_id]).sum().item())
                    expected_logit = float(logits[expected_id].item())
                    expected_text = tokenizer.decode([expected_id])
                tid = int(torch.argmax(logits).item())
                generated.append(tid)
                ids.append(tid)
                rows.append({
                    "step": step + 1,
                    "expected_id": expected_id,
                    "expected_text": expected_text,
                    "expected_rank": expected_rank,
                    "expected_logit": expected_logit,
                    "top1_id": tid,
                    "top1_text": tokenizer.decode([tid]),
                    "top1_matches_expected": bool(expected_id is not None and tid == expected_id),
                    "top_rows": top_rows,
                })
        return {"rows": rows, "generated_tokens": [tokenizer.decode([x]) for x in generated], "generated_text": tokenizer.decode(generated)}
    finally:
        for h in handles:
            h.remove()


def continuation_tag(row_eval: Dict, post: Dict) -> str:
    if row_eval["exact_correct"]:
        return "exact_correct"
    if post["top1"]["category"] == "correct_prefix":
        return "correct_prefix_but_generation_wrong"
    return "first_token_competition_failure"


def summarize(rows: List[Dict]) -> Dict:
    by_site: Dict[Tuple, Dict] = {}
    multi_failures: Dict[Tuple, Dict] = {}
    multi_scale: Dict[Tuple, Dict] = {}
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
            "sum_multi_margin": 0.0,
            "top1_category": {},
            "max_competitor_category": {},
            "continuation_tag": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["correct_top1"] += int(row["post"]["top1"]["category"] == "correct_prefix")
        item["sum_rank"] += row["post"]["prefix_rank"]
        item["sum_gap"] += row["post"]["top1_gap"]
        item["sum_multi_margin"] += row["multi"]["multi_margin"]
        topcat = row["post"]["top1"]["category"]
        item["top1_category"][topcat] = item["top1_category"].get(topcat, 0) + 1
        maxcat = "none" if row["multi"]["max_competitor"] is None else row["multi"]["max_competitor"]["category"]
        item["max_competitor_category"][maxcat] = item["max_competitor_category"].get(maxcat, 0) + 1
        tag = row["continuation_tag"]
        item["continuation_tag"][tag] = item["continuation_tag"].get(tag, 0) + 1

        if row["multi"]["winning_competitors"] and not row["eval"]["exact_correct"]:
            fkey = (row["pair_task"], maxcat)
            f = multi_failures.setdefault(fkey, {
                "pair_task": row["pair_task"],
                "max_competitor_category": maxcat,
                "n": 0,
                "sum_multi_margin": 0.0,
                "winner_sets": {},
            })
            f["n"] += 1
            f["sum_multi_margin"] += row["multi"]["multi_margin"]
            cats = "+".join(c["category"] for c in row["multi"]["winning_competitors"])
            f["winner_sets"][cats] = f["winner_sets"].get(cats, 0) + 1
            for s in row["multi_correction"]["scales"]:
                skey = (row["pair_task"], maxcat, s["scale"])
                sh = multi_scale.setdefault(skey, {
                    "pair_task": row["pair_task"],
                    "max_competitor_category": maxcat,
                    "scale": s["scale"],
                    "n": 0,
                    "correct_top1": 0,
                    "sum_rank": 0.0,
                    "sum_gap": 0.0,
                    "sum_multi_margin": 0.0,
                    "top1_after": {},
                    "max_comp_after": {},
                })
                sh["n"] += 1
                sh["correct_top1"] += int(s["correct_top1"])
                sh["sum_rank"] += s["prefix_rank"]
                sh["sum_gap"] += s["top1_gap"]
                sh["sum_multi_margin"] += s["multi_margin"]
                sh["top1_after"][s["top1_category"]] = sh["top1_after"].get(s["top1_category"], 0) + 1
                mc = "none" if s["max_competitor_category"] is None else s["max_competitor_category"]
                sh["max_comp_after"][mc] = sh["max_comp_after"].get(mc, 0) + 1

        if row["continuation_audit"] is not None:
            ckey = (row["pair_task"], row["site"], row["combo_name"])
            c = continuation.setdefault(ckey, {
                "pair_task": row["pair_task"],
                "site": row["site"],
                "combo_name": row["combo_name"],
                "n": 0,
                "token1_match": 0,
                "token2_match": 0,
                "token1_expected_rank_sum": 0.0,
                "token2_expected_rank_sum": 0.0,
                "generated_text": {},
            })
            c["n"] += 1
            rows2 = row["continuation_audit"]["rows"]
            if len(rows2) >= 1:
                c["token1_match"] += int(rows2[0]["top1_matches_expected"])
                c["token1_expected_rank_sum"] += rows2[0]["expected_rank"] or 0
            if len(rows2) >= 2:
                c["token2_match"] += int(rows2[1]["top1_matches_expected"])
                c["token2_expected_rank_sum"] += rows2[1]["expected_rank"] or 0
            text = row["continuation_audit"]["generated_text"].replace("\n", "\\n")
            c["generated_text"][text] = c["generated_text"].get(text, 0) + 1

    by_site_out = []
    for item in by_site.values():
        n = max(1, item["n"])
        r = dict(item)
        r["exact_rate"] = item["exact"] / n
        r["correct_top1_rate"] = item["correct_top1"] / n
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_gap"] = item["sum_gap"] / n
        r["mean_multi_margin"] = item["sum_multi_margin"] / n
        for k in ["top1_category", "max_competitor_category", "continuation_tag"]:
            r[k] = dict(sorted(item[k].items(), key=lambda kv: kv[1], reverse=True))
        by_site_out.append(r)

    multi_failures_out = []
    for f in multi_failures.values():
        n = max(1, f["n"])
        r = dict(f)
        r["mean_multi_margin"] = f["sum_multi_margin"] / n
        r["winner_sets"] = dict(sorted(f["winner_sets"].items(), key=lambda kv: kv[1], reverse=True))
        multi_failures_out.append(r)

    multi_scale_out = []
    for sh in multi_scale.values():
        n = max(1, sh["n"])
        r = dict(sh)
        r["correct_top1_rate"] = sh["correct_top1"] / n
        r["mean_rank"] = sh["sum_rank"] / n
        r["mean_gap"] = sh["sum_gap"] / n
        r["mean_multi_margin"] = sh["sum_multi_margin"] / n
        r["top1_after"] = dict(sorted(sh["top1_after"].items(), key=lambda kv: kv[1], reverse=True))
        r["max_comp_after"] = dict(sorted(sh["max_comp_after"].items(), key=lambda kv: kv[1], reverse=True))
        multi_scale_out.append(r)

    continuation_out = []
    for c in continuation.values():
        n = max(1, c["n"])
        r = dict(c)
        r["token1_match_rate"] = c["token1_match"] / n
        r["token2_match_rate"] = c["token2_match"] / n
        r["mean_token1_expected_rank"] = c["token1_expected_rank_sum"] / n
        r["mean_token2_expected_rank"] = c["token2_expected_rank_sum"] / n
        r["generated_text"] = dict(sorted(c["generated_text"].items(), key=lambda kv: kv[1], reverse=True)[:12])
        continuation_out.append(r)

    by_site_out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"]))
    multi_failures_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["max_competitor_category"]))
    multi_scale_out.sort(key=lambda r: (r["pair_task"], r["max_competitor_category"], r["scale"]))
    continuation_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["site"]))
    return {
        "by_site": by_site_out,
        "multi_competitor_failures": multi_failures_out,
        "multi_correction_by_scale": multi_scale_out,
        "continuation_audit": continuation_out,
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
        scales = [float(x) for x in args.direction_scales.split(",") if x.strip()]
        log(f"{args.model}: selected={len(selected)}, combos={combo_specs}, last_map={last_map}")

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
                    multi = topk_by_category(tokenizer, post_logits, prefix_id, value_prefix_ids, args.top_k)
                    multi_corr = multi_competitor_correction(
                        model,
                        tokenizer,
                        probe["final_norm_output"],
                        unembed,
                        prefix_id,
                        multi["winning_competitors"],
                        value_prefix_ids,
                        scales,
                        args.top_k,
                    )
                    tag = continuation_tag(ev, post)
                    cont = None
                    if tag == "correct_prefix_but_generation_wrong":
                        cont = continuation_after_prefix(
                            model,
                            tokenizer,
                            device,
                            task_prompt,
                            patches,
                            prefix_id,
                            correct_ids,
                            args.continuation_steps,
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
                        "multi": multi,
                        "multi_correction": multi_corr,
                        "eval": ev,
                        "continuation_tag": tag,
                        "continuation_audit": cont,
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                    }
                    rows.append(row)
                    if len(examples) < args.example_limit:
                        examples.append(row)

        summary = summarize(rows)
        log("Multi-competitor failures:")
        for r in summary["multi_competitor_failures"]:
            log(f"  {r['pair_task']} max={r['max_competitor_category']} n={r['n']} margin={r['mean_multi_margin']:.3f} sets={r['winner_sets']}")
        log("Multi-correction:")
        for r in summary["multi_correction_by_scale"]:
            log(
                f"  {r['pair_task']} max={r['max_competitor_category']} scale={r['scale']} n={r['n']} "
                f"correct_top1_rate={r['correct_top1_rate']:.2f} margin={r['mean_multi_margin']:.3f} after={r['top1_after']}"
            )
        log("Continuation audit:")
        for r in summary["continuation_audit"]:
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} n={r['n']} "
                f"tok1={r['token1_match_rate']:.2f} tok2={r['token2_match_rate']:.2f}"
            )
        return {
            "phase": 664,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "combo_specs": combo_specs,
            "last_writer_map": {str(k): v for k, v in last_map.items()},
            "tasks": TASK_ORDER,
            "target_categories": TARGET_CATEGORIES,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "direction_scales": scales,
            "continuation_steps": args.continuation_steps,
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
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--direction-scales", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--continuation-steps", type=int, default=3)
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
        args.top_k = min(args.top_k, 30)
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
        args.top_k = max(args.top_k, 50)
        args.max_new_tokens = min(max(args.max_new_tokens, 6), 6)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase664_{args.model}_multi_competitor_continuation_split_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
