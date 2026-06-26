#!/usr/bin/env python3
"""
Phase 661: Last-Writer Combo Generation Closure.

Uses Phase 660's strongest last-writer ablation candidates and combines them
with the Phase 658 format-prior suppression combo to test whether the remaining
top1 barrier can be pushed into generation closure.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
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
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import (  # noqa: E402
    SITE_SPECS,
    build_site_patch,
    collect_caches,
    install_component_ablation_hook,
)
from phase658_combined_format_prior_suppression_generation_audit import install_combo_ablation_hooks  # noqa: E402
from phase659_final_top1_barrier_readout_audit import TASK_ORDER, load_best_combos, token_category  # noqa: E402
from phase660_space_newline_readout_source_backtrace import OUT_ROOT as PHASE660_OUT  # noqa: E402


OUT_ROOT = Path("results/glm5_phase661_last_writer_combo_generation_closure")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def final_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def parse_last_mode(mode: str) -> Dict | None:
    m = re.match(r"last4_L(\d+)_(attn_out|mlp_out)$", mode)
    if not m:
        return None
    return {"layer": int(m.group(1)), "component": m.group(2)}


def load_last_writer_specs(model_key: str, max_last_writers: int) -> Dict[Tuple[str, str, str], List[Dict]]:
    path = PHASE660_OUT / f"phase660_{model_key}_space_newline_readout_source_backtrace_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in data["summary"]["last_writer_effects"]:
        if row["gap_delta_vs_combo"] <= 0:
            continue
        spec = parse_last_mode(row["mode"])
        if spec is None:
            continue
        key = (row["pair_task"], row["site"], row["combo_name"])
        item = dict(spec)
        item["phase660_gap_delta"] = row["gap_delta_vs_combo"]
        item["phase660_rank_delta"] = row["rank_delta_vs_combo"]
        out.setdefault(key, []).append(item)
    for key, items in list(out.items()):
        seen = set()
        uniq = []
        for item in sorted(items, key=lambda x: (-x["phase660_gap_delta"], -x["phase660_rank_delta"])):
            k = (item["layer"], item["component"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(item)
            if len(uniq) >= max_last_writers:
                break
        out[key] = uniq
    return out


def top_metric(tokenizer, logits: torch.Tensor, prefix_id: int, value_prefix_ids, top_k: int) -> Dict:
    topv, topi = torch.topk(logits.float(), k=top_k)
    toks = []
    for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        tid = int(i)
        text = tokenizer.decode([tid])
        toks.append({
            "rank": rank,
            "id": tid,
            "text": text,
            "category": token_category(text, tid, prefix_id, value_prefix_ids),
            "logit": float(v),
        })
    prefix_rank = 1 + int((logits > logits[prefix_id]).sum().item())
    prefix_logit = float(logits[prefix_id].item())
    return {
        "prefix_rank": prefix_rank,
        "top1_gap": float(toks[0]["logit"] - prefix_logit),
        "top1_category": toks[0]["category"],
        "top1_text": toks[0]["text"],
        "top_tokens": toks[:10],
    }


def install_all_ablation_hooks(model, tokenizer, prompt: str, combo: List[Dict], last_writers: List[Dict]):
    pos = final_pos(tokenizer, prompt)
    handles = []
    if combo:
        handles.extend(install_combo_ablation_hooks(model, combo, pos))
    for item in last_writers:
        h = install_component_ablation_hook(model, item["layer"], item["component"], pos)
        if h is not None:
            handles.append(h)
    return handles


def probe_mode(model, tokenizer, device, prompt, source_patches, combo, last_writers):
    handles = []
    try:
        handles = install_all_ablation_hooks(model, tokenizer, prompt, combo, last_writers)
        return final_state_probe(model, tokenizer, device, prompt, source_patches=source_patches)
    finally:
        for h in handles:
            h.remove()


def greedy_generate(model, tokenizer, device, prompt, max_new_tokens, source_patches, combo, last_writers):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    gen = []
    top5 = []
    source_handles = []
    if source_patches:
        source_handles = install_source_patch_hooks(model, tokenizer, prompt, source_patches)
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                handles = []
                if step == 0:
                    handles = install_all_ablation_hooks(model, tokenizer, prompt, combo, last_writers)
                try:
                    logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                finally:
                    for h in handles:
                        h.remove()
                topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
                top5.append([
                    {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(topv.cpu(), topi.cpu())
                ])
                tid = int(torch.argmax(logits).item())
                gen.append(tid)
                ids.append(tid)
        return {"ids": gen, "tokens": [tokenizer.decode([x]) for x in gen], "text": tokenizer.decode(gen), "top5": top5}
    finally:
        for h in source_handles:
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
            "combo": row["combo"],
            "last_writers": row["last_writers"],
            "n": 0,
            "exact": 0,
            "tok0": 0,
            "sum_rank": 0.0,
            "sum_gap": 0.0,
            "top1_category": {},
            "generation_text": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["tok0"] += int(row["metric"]["top1_category"] == "correct_prefix")
        item["sum_rank"] += row["metric"]["prefix_rank"]
        item["sum_gap"] += row["metric"]["top1_gap"]
        item["top1_category"][row["metric"]["top1_category"]] = item["top1_category"].get(row["metric"]["top1_category"], 0) + 1
        text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"][text] = item["generation_text"].get(text, 0) + 1

    out = []
    for item in by_mode.values():
        n = max(1, item["n"])
        r = dict(item)
        r["exact_rate"] = item["exact"] / n
        r["tok0_rate"] = item["tok0"] / n
        r["mean_rank"] = item["sum_rank"] / n
        r["mean_gap"] = item["sum_gap"] / n
        r["top1_category"] = dict(sorted(item["top1_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["generation_text"] = dict(sorted(item["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        out.append(r)

    paired = {}
    for r in out:
        key = (r["pair_task"], r["site"], r["combo_name"])
        paired.setdefault(key, {})[r["mode"]] = r
    effects = []
    for key, vals in paired.items():
        base = vals.get("phase658_combo")
        ext = vals.get("plus_last_writers")
        if not base or not ext:
            continue
        effects.append({
            "pair_task": key[0],
            "site": key[1],
            "combo_name": key[2],
            "combo": ext["combo"],
            "last_writers": ext["last_writers"],
            "n": ext["n"],
            "base_exact": base["exact"],
            "ext_exact": ext["exact"],
            "delta_exact": ext["exact"] - base["exact"],
            "base_tok0": base["tok0"],
            "ext_tok0": ext["tok0"],
            "delta_tok0": ext["tok0"] - base["tok0"],
            "base_rank": base["mean_rank"],
            "ext_rank": ext["mean_rank"],
            "rank_improvement": base["mean_rank"] - ext["mean_rank"],
            "base_gap": base["mean_gap"],
            "ext_gap": ext["mean_gap"],
            "gap_reduction": base["mean_gap"] - ext["mean_gap"],
            "base_top1": base["top1_category"],
            "ext_top1": ext["top1_category"],
        })
    effects.sort(key=lambda r: (-r["delta_exact"], -r["delta_tok0"], -r["gap_reduction"]))
    out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["mode"]))
    return {"by_mode": out, "closure_effects": effects}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
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
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
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
                    last_writers = last_map.get((task, combo_spec["site"], combo_spec["combo_name"]), [])
                    for mode, combo, extra in [
                        ("site_restore", [], []),
                        ("phase658_combo", combo_spec["components"], []),
                        ("plus_last_writers", combo_spec["components"], last_writers),
                    ]:
                        probe = probe_mode(model, tokenizer, device, task_prompt, patches, combo, extra)
                        metric = top_metric(tokenizer, probe["logits"], prefix_id, value_prefix_ids, args.top_k)
                        gen = greedy_generate(
                            model, tokenizer, device, task_prompt, args.max_new_tokens, patches, combo, extra
                        )
                        ev = generation_eval(gen, correct_ids, old_wrong_ids)
                        row = {
                            "sample_idx": item["sample_idx"],
                            "item_idx": item_i,
                            "pair_task": task,
                            "site": combo_spec["site"],
                            "combo_name": combo_spec["combo_name"],
                            "combo": combo_spec["components"],
                            "last_writers": extra,
                            "mode": mode,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "prefix_id": prefix_id,
                            "prefix_text": tokenizer.decode([prefix_id]),
                            "metric": metric,
                            "eval": ev,
                            "generation_text": gen["text"],
                            "generation_tokens": gen["tokens"],
                        }
                        rows.append(row)
                        if len(examples) < args.example_limit:
                            examples.append(row)

        summary = summarize(rows)
        log("Closure effects:")
        for r in summary["closure_effects"][:16]:
            last = ",".join(f"L{x['layer']}-{x['component']}" for x in r["last_writers"])
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} + [{last}]: "
                f"exact {r['base_exact']}->{r['ext_exact']} d={r['delta_exact']} "
                f"tok0 {r['base_tok0']}->{r['ext_tok0']} d={r['delta_tok0']} "
                f"gap {r['base_gap']:.2f}->{r['ext_gap']:.2f}"
            )
        return {
            "phase": 661,
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
    out_path = out_dir / f"phase661_{args.model}_last_writer_combo_generation_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
