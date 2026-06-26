#!/usr/bin/env python3
"""
Phase 659: Final Top1 Barrier and Readout Audit.

Uses the best Phase 658 combo per task/site and records the remaining top1
competitor after intent restore plus format-prior suppression.
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
from phase651_task_intent_gate_protocol_boundary_audit import (  # noqa: E402
    ladder_row,
    make_prompt,
    position_units,
    select_cases,
)
from phase656_format_prior_writer_localization_audit import (  # noqa: E402
    POLICY_GROUPS,
    SITE_SPECS,
    build_site_patch,
    collect_caches,
    group_margin,
)
from phase658_combined_format_prior_suppression_generation_audit import (  # noqa: E402
    OUT_ROOT as PHASE658_OUT,
    install_combo_ablation_hooks,
)


OUT_ROOT = Path("results/glm5_phase659_final_top1_barrier_readout_audit")
TASK_ORDER = ["explanation_required", "yes_no_required"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def final_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def load_best_combos(model_key: str, max_per_task: int) -> List[Dict]:
    path = PHASE658_OUT / f"phase658_{model_key}_combined_format_prior_suppression_generation_audit_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_task: Dict[str, List[Dict]] = {}
    for row in data["summary"]["combo_effects"]:
        by_task.setdefault(row["pair_task"], []).append(row)
    specs = []
    for task, rows in by_task.items():
        rows = sorted(
            rows,
            key=lambda r: (-r["delta_exact"], -r["delta_tok0"], -r["rank_improvement"]),
        )
        seen = set()
        for r in rows:
            key = (r["site"], r["combo_name"])
            if key in seen:
                continue
            seen.add(key)
            specs.append({
                "pair_task": task,
                "site": r["site"],
                "combo_name": r["combo_name"],
                "components": r["components"],
                "phase658_delta_exact": r["delta_exact"],
                "phase658_delta_tok0": r["delta_tok0"],
                "phase658_rank_improvement": r["rank_improvement"],
            })
            if len([x for x in specs if x["pair_task"] == task]) >= max_per_task:
                break
    return specs


def token_category(text: str, tid: int, prefix_id: int, value_prefix_ids) -> str:
    if tid == prefix_id:
        return "correct_prefix"
    if tid in value_prefix_ids:
        return "other_value_prefix"
    if "\n" in text:
        return "newline"
    if text == " " or text.strip() == "":
        return "space"
    low = text.strip().lower()
    if low in {"yes", "no", "true", "false"}:
        return "explanation"
    if low in {".", ",", ":", ";", "(", ")", "[", "]", "{", "}"}:
        return "punctuation"
    if any(ch.isalpha() for ch in low):
        return "word"
    return "symbol"


def top_tokens(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> List[Dict]:
    topv, topi = torch.topk(logits.float(), k=top_k)
    out = []
    for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        text = tokenizer.decode([int(i)])
        out.append({
            "rank": rank,
            "id": int(i),
            "text": text,
            "category": token_category(text, int(i), prefix_id, value_prefix_ids),
            "logit": float(v),
        })
    return out


def probe_mode(model, tokenizer, device, prompt, source_patches, combo):
    handles = []
    try:
        if combo:
            handles = install_combo_ablation_hooks(model, combo, final_pos(tokenizer, prompt))
        return final_state_probe(model, tokenizer, device, prompt, source_patches=source_patches)
    finally:
        for h in handles:
            h.remove()


def metric_row(tokenizer, probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k):
    row = ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k)
    row["policy_margins"] = {g: group_margin(row, g) for g in POLICY_GROUPS}
    row["top_tokens"] = top_tokens(tokenizer, probe["logits"], prefix_id, value_prefix_ids, min(top_k, 10))
    row["correct_logit"] = float(probe["logits"][prefix_id].item())
    row["top1_gap"] = float(row["top_tokens"][0]["logit"] - row["correct_logit"])
    return row


def summarize(rows: List[Dict]) -> Dict:
    by_mode: Dict[Tuple, Dict] = {}
    transitions: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"], row["mode"])
        item = by_mode.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "mode": row["mode"],
            "components": row["components"],
            "n": 0,
            "sum_rank": 0.0,
            "sum_gap": 0.0,
            "top1_category": {},
            "top1_text": {},
            "correct_top1": 0,
        })
        item["n"] += 1
        item["sum_rank"] += row["prefix_rank"]
        item["sum_gap"] += row["top1_gap"]
        top1 = row["top_tokens"][0]
        item["top1_category"][top1["category"]] = item["top1_category"].get(top1["category"], 0) + 1
        text = top1["text"].replace("\n", "\\n")
        item["top1_text"][text] = item["top1_text"].get(text, 0) + 1
        item["correct_top1"] += int(top1["category"] == "correct_prefix")

    out = []
    for item in by_mode.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_top1_gap"] = item["sum_gap"] / n
        r["correct_top1_rate"] = item["correct_top1"] / n
        r["top1_category"] = dict(sorted(item["top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["top1_text"] = dict(sorted(item["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        out.append(r)

    for r in out:
        key = (r["pair_task"], r["site"], r["combo_name"])
        transitions.setdefault(key, {})[r["mode"]] = r
    effects = []
    for key, vals in transitions.items():
        base = vals.get("site_restore")
        combo = vals.get("combo_ablation")
        if not base or not combo:
            continue
        effects.append({
            "pair_task": key[0],
            "site": key[1],
            "combo_name": key[2],
            "components": combo["components"],
            "n": combo["n"],
            "site_rank": base["mean_rank"],
            "combo_rank": combo["mean_rank"],
            "rank_improvement": base["mean_rank"] - combo["mean_rank"],
            "site_gap": base["mean_top1_gap"],
            "combo_gap": combo["mean_top1_gap"],
            "gap_reduction": base["mean_top1_gap"] - combo["mean_top1_gap"],
            "site_correct_top1": base["correct_top1"],
            "combo_correct_top1": combo["correct_top1"],
            "delta_correct_top1": combo["correct_top1"] - base["correct_top1"],
            "site_top1_category": base["top1_category"],
            "combo_top1_category": combo["top1_category"],
            "combo_top1_text": combo["top1_text"],
        })
    effects.sort(key=lambda r: (-r["delta_correct_top1"], -r["gap_reduction"], -r["rank_improvement"]))
    out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["mode"]))
    return {"by_mode": out, "barrier_effects": effects}


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
        log(f"{args.model}: selected={len(selected)}, combo_specs={combo_specs}")

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_prefix_id = answer_ids(tokenizer, old_wrong)[0]
            prefix_id = correct_ids[0]
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
                    for mode, source_patches, combo in [
                        ("baseline_task", None, []),
                        ("site_restore", patches, []),
                        ("combo_ablation", patches, combo_spec["components"]),
                    ]:
                        probe = probe_mode(model, tokenizer, device, task_prompt, source_patches, combo)
                        metric = metric_row(tokenizer, probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k)
                        row = {
                            "sample_idx": item["sample_idx"],
                            "item_idx": item_i,
                            "pair_task": task,
                            "site": combo_spec["site"],
                            "combo_name": combo_spec["combo_name"],
                            "components": combo_spec["components"],
                            "mode": mode,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "prefix_id": prefix_id,
                            "prefix_text": tokenizer.decode([prefix_id]),
                            **metric,
                        }
                        rows.append(row)
                        if len(examples) < args.example_limit:
                            examples.append(row)

        summary = summarize(rows)
        log("Barrier effects:")
        for r in summary["barrier_effects"][:20]:
            comps = ",".join(f"L{x['layer']}-{x['component']}" for x in r["components"])
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} [{comps}]: "
                f"rank {r['site_rank']:.1f}->{r['combo_rank']:.1f} "
                f"gap {r['site_gap']:.2f}->{r['combo_gap']:.2f} "
                f"correct_top1 {r['site_correct_top1']}->{r['combo_correct_top1']} "
                f"combo_top1={r['combo_top1_category']}"
            )
        return {
            "phase": 659,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "combo_specs": combo_specs,
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
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=320)
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
        args.top_k = min(args.top_k, 20)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.max_per_task = max(args.max_per_task, 2)
        args.top_k = max(args.top_k, 30)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase659_{args.model}_final_top1_barrier_readout_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
