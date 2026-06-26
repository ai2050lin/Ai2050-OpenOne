#!/usr/bin/env python3
"""
Phase 662: Residual-to-LMHead Projection Barrier Audit.

Audits remaining failures after Phase 661 by measuring whether correct_prefix
still loses because final_norm output aligns more with space/newline tokens or
because those tokens have a projection advantage in the lm_head.
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
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS, build_site_patch, collect_caches  # noqa: E402
from phase659_final_top1_barrier_readout_audit import TASK_ORDER, load_best_combos, token_category  # noqa: E402
from phase661_last_writer_combo_generation_closure import (  # noqa: E402
    greedy_generate,
    load_last_writer_specs,
    probe_mode,
)


OUT_ROOT = Path("results/glm5_phase662_residual_to_lmhead_projection_barrier_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_unembed(model) -> torch.Tensor:
    emb = model.get_output_embeddings()
    return emb.weight.detach().float().cpu()


def logits_from_state(model, state: torch.Tensor) -> torch.Tensor:
    emb = model.get_output_embeddings()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        logits = emb(state.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0).float()
    return logits.detach().cpu()


def top_tokens(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> List[Dict]:
    topv, topi = torch.topk(logits.float(), k=top_k)
    out = []
    for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        tid = int(i)
        text = tokenizer.decode([tid])
        out.append({
            "rank": rank,
            "id": tid,
            "text": text,
            "category": token_category(text, tid, prefix_id, value_prefix_ids),
            "logit": float(v),
        })
    return out


def readout_metric(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> Dict:
    toks = top_tokens(tokenizer, logits, prefix_id, value_prefix_ids, top_k)
    prefix_logit = float(logits[prefix_id].item())
    prefix_rank = 1 + int((logits > logits[prefix_id]).sum().item())
    top1 = toks[0]
    return {
        "prefix_rank": prefix_rank,
        "prefix_logit": prefix_logit,
        "top1": top1,
        "top1_gap": float(top1["logit"] - prefix_logit),
        "top_tokens": toks[:10],
    }


def projection_diag(
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
    diff = wt - wc
    gap = float(competitor_logit - prefix_logit)
    h_norm = float(h.norm().item())
    wc_norm = float(wc.norm().item())
    wt_norm = float(wt.norm().item())
    diff_norm = float(diff.norm().item())
    denom = max(h_norm * diff_norm, 1e-8)
    return {
        "gap": gap,
        "state_norm": h_norm,
        "correct_w_norm": wc_norm,
        "competitor_w_norm": wt_norm,
        "competitor_norm_advantage": wt_norm - wc_norm,
        "diff_norm": diff_norm,
        "diff_alignment": float(torch.dot(h, diff).item() / denom),
        "needed_unit_delta": gap / max(diff_norm, 1e-8),
        "correct_cos": float(F.cosine_similarity(h, wc, dim=0).item()),
        "competitor_cos": float(F.cosine_similarity(h, wt, dim=0).item()),
    }


def summarize(rows: List[Dict]) -> Dict:
    by_mode: Dict[Tuple, Dict] = {}
    failure_by_category: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"], row["mode"])
        item = by_mode.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "mode": row["mode"],
            "n": 0,
            "exact": 0,
            "correct_top1": 0,
            "sum_rank": 0.0,
            "sum_gap": 0.0,
            "top1_category": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["correct_top1"] += int(row["post"]["top1"]["category"] == "correct_prefix")
        item["sum_rank"] += row["post"]["prefix_rank"]
        item["sum_gap"] += row["post"]["top1_gap"]
        cat = row["post"]["top1"]["category"]
        item["top1_category"][cat] = item["top1_category"].get(cat, 0) + 1

        if row["mode"] == "plus_last_writers" and not row["eval"]["exact_correct"]:
            fkey = (row["pair_task"], row["post"]["top1"]["category"])
            f = failure_by_category.setdefault(fkey, {
                "pair_task": row["pair_task"],
                "top1_category": row["post"]["top1"]["category"],
                "n": 0,
                "top1_text": {},
                "sum_post_gap": 0.0,
                "sum_pre_gap": 0.0,
                "sum_norm_gap_change": 0.0,
                "sum_needed_delta": 0.0,
                "sum_diff_alignment": 0.0,
                "sum_correct_cos": 0.0,
                "sum_competitor_cos": 0.0,
                "sum_norm_adv": 0.0,
            })
            f["n"] += 1
            text = row["post"]["top1"]["text"].replace("\n", "\\n")
            f["top1_text"][text] = f["top1_text"].get(text, 0) + 1
            f["sum_post_gap"] += row["post"]["top1_gap"]
            f["sum_pre_gap"] += row["pre"]["top1_gap"]
            f["sum_norm_gap_change"] += row["post"]["top1_gap"] - row["pre"]["top1_gap"]
            f["sum_needed_delta"] += row["post_diag"]["needed_unit_delta"]
            f["sum_diff_alignment"] += row["post_diag"]["diff_alignment"]
            f["sum_correct_cos"] += row["post_diag"]["correct_cos"]
            f["sum_competitor_cos"] += row["post_diag"]["competitor_cos"]
            f["sum_norm_adv"] += row["post_diag"]["competitor_norm_advantage"]

    by_mode_out = []
    for item in by_mode.values():
        n = max(1, item["n"])
        r = dict(item)
        r["exact_rate"] = item["exact"] / n
        r["correct_top1_rate"] = item["correct_top1"] / n
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_gap"] = item["sum_gap"] / n
        r["top1_category"] = dict(sorted(item["top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        by_mode_out.append(r)

    failure_out = []
    for f in failure_by_category.values():
        n = max(1, f["n"])
        r = dict(f)
        r["mean_post_gap"] = f["sum_post_gap"] / n
        r["mean_pre_gap"] = f["sum_pre_gap"] / n
        r["mean_norm_gap_change"] = f["sum_norm_gap_change"] / n
        r["mean_needed_unit_delta"] = f["sum_needed_delta"] / n
        r["mean_diff_alignment"] = f["sum_diff_alignment"] / n
        r["mean_correct_cos"] = f["sum_correct_cos"] / n
        r["mean_competitor_cos"] = f["sum_competitor_cos"] / n
        r["mean_competitor_norm_advantage"] = f["sum_norm_adv"] / n
        r["top1_text"] = dict(sorted(f["top1_text"].items(), key=lambda kv: kv[1], reverse=True))
        failure_out.append(r)

    by_mode_out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["mode"]))
    failure_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["top1_category"]))
    return {"by_mode": by_mode_out, "plus_failure_projection": failure_out}


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
                    for mode, combo, extra in [
                        ("phase658_combo", combo_spec["components"], []),
                        ("plus_last_writers", combo_spec["components"], last_writers),
                    ]:
                        probe = probe_mode(model, tokenizer, device, task_prompt, patches, combo, extra)
                        gen = greedy_generate(
                            model, tokenizer, device, task_prompt, args.max_new_tokens, patches, combo, extra
                        )
                        ev = generation_eval(gen, correct_ids, old_wrong_ids)
                        post_logits = probe["logits"]
                        pre_logits = logits_from_state(model, probe["final_norm_input"])
                        post = readout_metric(tokenizer, post_logits, prefix_id, value_prefix_ids, args.top_k)
                        pre = readout_metric(tokenizer, pre_logits, prefix_id, value_prefix_ids, args.top_k)
                        comp_id = post["top1"]["id"]
                        post_diag = projection_diag(
                            probe["final_norm_output"],
                            unembed,
                            prefix_id,
                            comp_id,
                            float(post_logits[prefix_id].item()),
                            float(post_logits[comp_id].item()),
                        )
                        pre_diag = projection_diag(
                            probe["final_norm_input"],
                            unembed,
                            prefix_id,
                            comp_id,
                            float(pre_logits[prefix_id].item()),
                            float(pre_logits[comp_id].item()),
                        )
                        row = {
                            "sample_idx": item["sample_idx"],
                            "item_idx": item_i,
                            "pair_task": task,
                            "site": combo_spec["site"],
                            "combo_name": combo_spec["combo_name"],
                            "combo": combo,
                            "last_writers": extra,
                            "mode": mode,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "prefix_id": prefix_id,
                            "prefix_text": tokenizer.decode([prefix_id]),
                            "post": post,
                            "pre": pre,
                            "post_diag": post_diag,
                            "pre_diag": pre_diag,
                            "eval": ev,
                            "generation_text": gen["text"],
                            "generation_tokens": gen["tokens"],
                        }
                        rows.append(row)
                        if len(examples) < args.example_limit:
                            examples.append(row)

        summary = summarize(rows)
        log("Remaining failure projection:")
        for r in summary["plus_failure_projection"]:
            log(
                f"  {r['pair_task']} top1={r['top1_category']} n={r['n']} "
                f"post_gap={r['mean_post_gap']:.2f} pre_gap={r['mean_pre_gap']:.2f} "
                f"norm_change={r['mean_norm_gap_change']:.2f} need_delta={r['mean_needed_unit_delta']:.3f} "
                f"cos c/t={r['mean_correct_cos']:.3f}/{r['mean_competitor_cos']:.3f}"
            )
        return {
            "phase": 662,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "combo_specs": combo_specs,
            "last_writer_map": {str(k): v for k, v in last_map.items()},
            "tasks": TASK_ORDER,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_selected_items": len(selected),
            "n_mode_rows": len(rows),
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
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.max_per_task = max(args.max_per_task, 2)
        args.max_last_writers = min(max(args.max_last_writers, 2), 2)
        args.top_k = max(args.top_k, 30)
        args.max_new_tokens = min(max(args.max_new_tokens, 6), 6)
        args.example_limit = max(args.example_limit, 240)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase662_{args.model}_residual_to_lmhead_projection_barrier_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
