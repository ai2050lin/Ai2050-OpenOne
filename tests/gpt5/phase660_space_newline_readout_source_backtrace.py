#!/usr/bin/env python3
"""
Phase 660: Space/Newline Residual Readout Source Backtrace.

Backtraces the remaining top1 space/newline barrier after Phase 659 by
separating pre-final-norm projection, post-final-norm lm_head projection, and
last residual writer ablation effects.
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
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import (  # noqa: E402
    SITE_SPECS,
    build_site_patch,
    collect_caches,
    install_component_ablation_hook,
)
from phase658_combined_format_prior_suppression_generation_audit import install_combo_ablation_hooks  # noqa: E402
from phase659_final_top1_barrier_readout_audit import (  # noqa: E402
    TASK_ORDER,
    load_best_combos,
    token_category,
)


OUT_ROOT = Path("results/glm5_phase660_space_newline_readout_source_backtrace")
SCAN_COMPONENTS = ["attn_out", "mlp_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def final_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def output_logits_from_state(model, state: torch.Tensor) -> torch.Tensor:
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


def metrics_from_logits(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> Dict:
    toks = top_tokens(tokenizer, logits, prefix_id, value_prefix_ids, min(top_k, 20))
    prefix_logit = float(logits[prefix_id].item())
    prefix_rank = 1 + int((logits > logits[prefix_id]).sum().item())
    top1 = toks[0]
    best_by_group = {}
    for t in toks:
        g = t["category"]
        if g not in best_by_group:
            best_by_group[g] = t
    return {
        "prefix_rank": prefix_rank,
        "prefix_logit": prefix_logit,
        "top1_id": top1["id"],
        "top1_text": top1["text"],
        "top1_category": top1["category"],
        "top1_gap": float(top1["logit"] - prefix_logit),
        "top_tokens": toks[:10],
        "space_gap": (
            float(best_by_group["space"]["logit"] - prefix_logit)
            if "space" in best_by_group else None
        ),
        "newline_gap": (
            float(best_by_group["newline"]["logit"] - prefix_logit)
            if "newline" in best_by_group else None
        ),
        "explanation_gap": (
            float(best_by_group["explanation"]["logit"] - prefix_logit)
            if "explanation" in best_by_group else None
        ),
    }


def probe_with_hooks(model, tokenizer, device, prompt, source_patches, combo, ablate):
    handles = []
    try:
        if combo:
            handles.extend(install_combo_ablation_hooks(model, combo, final_pos(tokenizer, prompt)))
        if ablate:
            h = install_component_ablation_hook(model, ablate["layer"], ablate["component"], final_pos(tokenizer, prompt))
            if h is not None:
                handles.append(h)
        return final_state_probe(model, tokenizer, device, prompt, source_patches=source_patches)
    finally:
        for h in handles:
            h.remove()


def summarize(rows: List[Dict]) -> Dict:
    by_mode: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"], row["mode"])
        item = by_mode.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "mode": row["mode"],
            "components": row["components"],
            "n": 0,
            "sum_post_rank": 0.0,
            "sum_post_gap": 0.0,
            "sum_pre_gap": 0.0,
            "sum_norm_gap_shift": 0.0,
            "post_top1_category": {},
            "pre_top1_category": {},
            "correct_top1": 0,
        })
        item["n"] += 1
        item["sum_post_rank"] += row["post"]["prefix_rank"]
        item["sum_post_gap"] += row["post"]["top1_gap"]
        item["sum_pre_gap"] += row["pre"]["top1_gap"]
        item["sum_norm_gap_shift"] += row["post"]["top1_gap"] - row["pre"]["top1_gap"]
        item["post_top1_category"][row["post"]["top1_category"]] = item["post_top1_category"].get(row["post"]["top1_category"], 0) + 1
        item["pre_top1_category"][row["pre"]["top1_category"]] = item["pre_top1_category"].get(row["pre"]["top1_category"], 0) + 1
        item["correct_top1"] += int(row["post"]["top1_category"] == "correct_prefix")

    modes = []
    for item in by_mode.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_post_rank"] = item["sum_post_rank"] / n
        r["mean_post_gap"] = item["sum_post_gap"] / n
        r["mean_pre_gap"] = item["sum_pre_gap"] / n
        r["mean_norm_gap_shift"] = item["sum_norm_gap_shift"] / n
        r["correct_top1_rate"] = item["correct_top1"] / n
        r["post_top1_category"] = dict(sorted(item["post_top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["pre_top1_category"] = dict(sorted(item["pre_top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        modes.append(r)

    by_combo = {}
    for r in modes:
        key = (r["pair_task"], r["site"], r["combo_name"])
        by_combo.setdefault(key, {})[r["mode"]] = r

    source_effects = []
    writer_effects = []
    for key, vals in by_combo.items():
        site = vals.get("site_restore")
        combo = vals.get("combo_ablation")
        if site and combo:
            source_effects.append({
                "pair_task": key[0],
                "site": key[1],
                "combo_name": key[2],
                "components": combo["components"],
                "n": combo["n"],
                "site_post_gap": site["mean_post_gap"],
                "combo_post_gap": combo["mean_post_gap"],
                "gap_reduction": site["mean_post_gap"] - combo["mean_post_gap"],
                "site_post_rank": site["mean_post_rank"],
                "combo_post_rank": combo["mean_post_rank"],
                "rank_improvement": site["mean_post_rank"] - combo["mean_post_rank"],
                "combo_norm_gap_shift": combo["mean_norm_gap_shift"],
                "site_post_top1_category": site["post_top1_category"],
                "combo_post_top1_category": combo["post_top1_category"],
            })
        if combo:
            for mode, r in vals.items():
                if not mode.startswith("last4_"):
                    continue
                writer_effects.append({
                    "pair_task": key[0],
                    "site": key[1],
                    "combo_name": key[2],
                    "mode": mode,
                    "components": r["components"],
                    "n": r["n"],
                    "combo_post_gap": combo["mean_post_gap"],
                    "ablated_post_gap": r["mean_post_gap"],
                    "gap_delta_vs_combo": combo["mean_post_gap"] - r["mean_post_gap"],
                    "combo_rank": combo["mean_post_rank"],
                    "ablated_rank": r["mean_post_rank"],
                    "rank_delta_vs_combo": combo["mean_post_rank"] - r["mean_post_rank"],
                    "ablated_top1_category": r["post_top1_category"],
                    "ablated_norm_gap_shift": r["mean_norm_gap_shift"],
                })
    source_effects.sort(key=lambda r: (-r["gap_reduction"], -r["rank_improvement"]))
    writer_effects.sort(key=lambda r: (-r["gap_delta_vs_combo"], -r["rank_delta_vs_combo"]))
    modes.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["mode"]))
    return {"by_mode": modes, "source_effects": source_effects, "last_writer_effects": writer_effects}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        combo_specs = load_best_combos(args.model, args.max_per_task)
        site_specs = {s["name"]: s for s in SITE_SPECS[args.model]}
        needed_sites = [site_specs[c["site"]] for c in combo_specs if c["site"] in site_specs]
        site_layers = sorted({li for s in needed_sites for li in s["layers"] if 0 <= li < info.n_layers})
        site_components = sorted({c for s in needed_sites for c in s["components"]})
        site_positions = sorted({p for s in needed_sites for p in s["positions"]})
        last_layers = [li for li in range(max(0, info.n_layers - args.last_n_layers), info.n_layers)]
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
        log(f"{args.model}: selected={len(selected)}, combos={combo_specs}, last_layers={last_layers}")

        for item_i, item in enumerate(selected):
            case = item["case"]
            prefix_id = answer_ids(tokenizer, case["correct"])[0]
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
                    mode_specs = [
                        ("baseline_task", None, [], None),
                        ("site_restore", patches, [], None),
                        ("combo_ablation", patches, combo_spec["components"], None),
                    ]
                    for li in last_layers:
                        for comp in SCAN_COMPONENTS:
                            mode_specs.append((
                                f"last4_L{li}_{comp}",
                                patches,
                                combo_spec["components"],
                                {"layer": li, "component": comp},
                            ))
                    for mode, source_patches, combo, ablate in mode_specs:
                        probe = probe_with_hooks(model, tokenizer, device, task_prompt, source_patches, combo, ablate)
                        post = metrics_from_logits(tokenizer, probe["logits"], prefix_id, value_prefix_ids, args.top_k)
                        pre = metrics_from_logits(
                            tokenizer,
                            output_logits_from_state(model, probe["final_norm_input"]),
                            prefix_id,
                            value_prefix_ids,
                            args.top_k,
                        )
                        row = {
                            "sample_idx": item["sample_idx"],
                            "item_idx": item_i,
                            "pair_task": task,
                            "site": combo_spec["site"],
                            "combo_name": combo_spec["combo_name"],
                            "components": combo_spec["components"],
                            "mode": mode,
                            "ablate": ablate,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "prefix_id": prefix_id,
                            "prefix_text": tokenizer.decode([prefix_id]),
                            "pre": pre,
                            "post": post,
                            "norm_gap_shift": post["top1_gap"] - pre["top1_gap"],
                        }
                        rows.append(row)
                        if len(examples) < args.example_limit:
                            examples.append(row)

        summary = summarize(rows)
        log("Source effects:")
        for r in summary["source_effects"][:12]:
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']}: "
                f"gap {r['site_post_gap']:.2f}->{r['combo_post_gap']:.2f} "
                f"rank {r['site_post_rank']:.1f}->{r['combo_post_rank']:.1f} "
                f"top1={r['combo_post_top1_category']} norm_shift={r['combo_norm_gap_shift']:.2f}"
            )
        log("Last-writer effects:")
        for r in summary["last_writer_effects"][:18]:
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} {r['mode']}: "
                f"gap_delta={r['gap_delta_vs_combo']:.2f} rank_delta={r['rank_delta_vs_combo']:.2f} "
                f"top1={r['ablated_top1_category']}"
            )
        return {
            "phase": 660,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "combo_specs": combo_specs,
            "last_layers": last_layers,
            "tasks": TASK_ORDER,
            "top_k": args.top_k,
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
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=30)
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
        args.last_n_layers = 2
        args.top_k = min(args.top_k, 20)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.max_per_task = max(args.max_per_task, 2)
        args.last_n_layers = min(max(args.last_n_layers, 4), 4)
        args.top_k = max(args.top_k, 30)
        args.example_limit = max(args.example_limit, 240)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase660_{args.model}_space_newline_readout_source_backtrace_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
