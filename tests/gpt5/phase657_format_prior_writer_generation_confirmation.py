#!/usr/bin/env python3
"""
Phase 657: Format-Prior Writer Candidate Generation Confirmation.

Uses top Phase 656 component-ablation candidates and tests whether the margin
improvement also changes short greedy generation under the fixed intent-gate
restore patch.
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
from phase630_distributed_format_route_multisource import collect_positions_components  # noqa: E402
from phase635_final_readout_projection_bridge_audit import final_state_probe  # noqa: E402
from phase647_protocol_writer_graph_audit import make_multi_patch  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import (  # noqa: E402
    ladder_row,
    make_prompt,
    position_units,
    select_cases,
)
from phase656_format_prior_writer_localization_audit import (  # noqa: E402
    OUT_ROOT as PHASE656_OUT,
    POLICY_GROUPS,
    SITE_SPECS,
    build_site_patch,
    collect_caches,
    group_margin,
    install_component_ablation_hook,
)
from phase630_distributed_format_route_multisource import install_source_patch_hooks  # noqa: E402


OUT_ROOT = Path("results/glm5_phase657_format_prior_writer_generation_confirmation")
TASK_ORDER = ["explanation_required", "yes_no_required"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def final_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False)) - 1


def load_candidate_specs(model_key: str, max_specs: int, min_n: int) -> List[Dict]:
    path = PHASE656_OUT / f"phase656_{model_key}_format_prior_writer_localization_audit_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    seen = set()
    for row in data["summary"]["format_prior_writer_candidates"]:
        if row["n"] < min_n:
            continue
        if row["mean_top_margin_delta"] <= 0:
            continue
        key = (row["pair_task"], row["site"], row["layer"], row["component"])
        if key in seen:
            continue
        seen.add(key)
        specs.append({
            "pair_task": row["pair_task"],
            "site": row["site"],
            "layer": row["layer"],
            "component": row["component"],
            "baseline_top0_category": row["baseline_top0_category"],
            "phase656_n": row["n"],
            "phase656_dtop": row["mean_top_margin_delta"],
            "phase656_drank": row["mean_rank_improvement"],
            "phase656_flip": row["flipped_to_correct"],
        })
        if len(specs) >= max_specs:
            break
    return specs


def greedy_generate_with_optional_ablation(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    source_patches,
    ablate_layer: int | None,
    ablate_component: str | None,
) -> Dict:
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
                ablate_handle = None
                if step == 0 and ablate_layer is not None and ablate_component is not None:
                    ablate_handle = install_component_ablation_hook(
                        model, ablate_layer, ablate_component, len(ids) - 1
                    )
                try:
                    logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                finally:
                    if ablate_handle is not None:
                        ablate_handle.remove()
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


def probe_with_optional_ablation(model, tokenizer, device, prompt, source_patches, layer, component):
    handles = []
    try:
        if layer is not None and component is not None:
            h = install_component_ablation_hook(model, layer, component, final_pos(tokenizer, prompt))
            if h is not None:
                handles.append(h)
        return final_state_probe(model, tokenizer, device, prompt, source_patches=source_patches)
    finally:
        for h in handles:
            h.remove()


def metric_row(tokenizer, probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k):
    row = ladder_row(tokenizer, probe["logits"], prefix_id, old_wrong_prefix_id, value_prefix_ids, top_k)
    row["policy_margins"] = {g: group_margin(row, g) for g in POLICY_GROUPS}
    return row


def summarize(rows: List[Dict]) -> Dict:
    by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["candidate_layer"], row["candidate_component"], row["mode"])
        item = by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "candidate_layer": row["candidate_layer"],
            "candidate_component": row["candidate_component"],
            "mode": row["mode"],
            "phase656_dtop": row["phase656_dtop"],
            "phase656_n": row["phase656_n"],
            "n": 0,
            "exact": 0,
            "tok0": 0,
            "sum_rank": 0.0,
            "top0_category": {},
            "generation_text": {},
        })
        item["n"] += 1
        item["exact"] += int(row["eval"]["exact_correct"])
        item["tok0"] += int(row["top0_id"] == row["prefix_id"])
        item["sum_rank"] += row["prefix_rank"]
        item["top0_category"][row["top0_category"]] = item["top0_category"].get(row["top0_category"], 0) + 1
        text = row["generation_text"].replace("\n", "\\n")
        item["generation_text"][text] = item["generation_text"].get(text, 0) + 1
    out = []
    for item in by_key.values():
        n = max(1, item["n"])
        r = dict(item)
        r["mean_rank"] = item["sum_rank"] / n
        r["exact_rate"] = item["exact"] / n
        r["tok0_rate"] = item["tok0"] / n
        r["top0_category"] = dict(sorted(item["top0_category"].items(), key=lambda kv: kv[1], reverse=True))
        r["generation_text"] = dict(sorted(item["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        out.append(r)

    paired = {}
    for r in out:
        key = (r["pair_task"], r["site"], r["candidate_layer"], r["candidate_component"])
        paired.setdefault(key, {})[r["mode"]] = r
    effects = []
    for key, vals in paired.items():
        base = vals.get("site_restore")
        ab = vals.get("candidate_ablation")
        if not base or not ab:
            continue
        effects.append({
            "pair_task": key[0],
            "site": key[1],
            "candidate_layer": key[2],
            "candidate_component": key[3],
            "n": ab["n"],
            "phase656_dtop": ab["phase656_dtop"],
            "base_exact": base["exact"],
            "ablation_exact": ab["exact"],
            "delta_exact": ab["exact"] - base["exact"],
            "base_tok0": base["tok0"],
            "ablation_tok0": ab["tok0"],
            "delta_tok0": ab["tok0"] - base["tok0"],
            "base_rank": base["mean_rank"],
            "ablation_rank": ab["mean_rank"],
            "rank_improvement": base["mean_rank"] - ab["mean_rank"],
            "base_top0": base["top0_category"],
            "ablation_top0": ab["top0_category"],
        })
    effects.sort(key=lambda r: (-r["delta_exact"], -r["delta_tok0"], -r["rank_improvement"]))
    out.sort(key=lambda r: (r["pair_task"], r["site"], r["candidate_layer"], r["candidate_component"], r["mode"]))
    return {"by_mode": out, "generation_effects": effects}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        candidates = load_candidate_specs(args.model, args.max_candidate_specs, args.min_candidate_n)
        site_specs = {s["name"]: s for s in SITE_SPECS[args.model]}
        needed_sites = [site_specs[c["site"]] for c in candidates if c["site"] in site_specs]
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
        log(f"{args.model}: selected={len(selected)}, candidates={candidates}")

        for item_i, item in enumerate(selected):
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong = item["base_top_wrong"] or item["repair_top_wrong"] or item["mode_v_top_wrong"] or values[0]
            old_wrong_ids = answer_ids(tokenizer, old_wrong)
            prefix_id = correct_ids[0]
            old_wrong_prefix_id = old_wrong_ids[0]
            value_prompt, value_relation, value_intent = make_prompt(
                case, "short_value_allowed", relation_pool, tokenizer, item["sample_idx"]
            )
            value_units_all = position_units(tokenizer, value_prompt, case, value_relation, value_intent)
            value_units = {p: value_units_all.get(p, []) for p in site_positions}
            value_caches = collect_caches(model, tokenizer, device, value_prompt, value_units, site_layers, site_components)

            for task_i, task in enumerate(TASK_ORDER):
                task_candidates = [c for c in candidates if c["pair_task"] == task]
                if not task_candidates:
                    continue
                task_prompt, task_relation, task_intent = make_prompt(
                    case, task, relation_pool, tokenizer, item["sample_idx"] + task_i * 17
                )
                task_units_all = position_units(tokenizer, task_prompt, case, task_relation, task_intent)
                task_units = {p: task_units_all.get(p, []) for p in site_positions}
                task_caches = collect_caches(model, tokenizer, device, task_prompt, task_units, site_layers, site_components)

                site_patch_cache = {}
                for cand in task_candidates:
                    site = site_specs[cand["site"]]
                    if cand["site"] not in site_patch_cache:
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
                        site_patch_cache[cand["site"]] = patches
                    patches = site_patch_cache[cand["site"]]
                    if not patches:
                        continue
                    for mode, layer, component in [
                        ("site_restore", None, None),
                        ("candidate_ablation", cand["layer"], cand["component"]),
                    ]:
                        probe = probe_with_optional_ablation(model, tokenizer, device, task_prompt, patches, layer, component)
                        gen = greedy_generate_with_optional_ablation(
                            model, tokenizer, device, task_prompt, args.max_new_tokens, patches, layer, component
                        )
                        ev = generation_eval(gen, correct_ids, old_wrong_ids)
                        metric = metric_row(tokenizer, probe, prefix_id, old_wrong_prefix_id, value_prefix_ids, args.top_k)
                        row = {
                            "sample_idx": item["sample_idx"],
                            "item_idx": item_i,
                            "pair_task": task,
                            "site": cand["site"],
                            "candidate_layer": cand["layer"],
                            "candidate_component": cand["component"],
                            "phase656_dtop": cand["phase656_dtop"],
                            "phase656_n": cand["phase656_n"],
                            "mode": mode,
                            "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                            "prefix_id": prefix_id,
                            "prefix_text": tokenizer.decode([prefix_id]),
                            "old_wrong_prefix_id": old_wrong_prefix_id,
                            "old_wrong_prefix_text": tokenizer.decode([old_wrong_prefix_id]),
                            "eval": ev,
                            "generation_text": gen["text"],
                            "generation_tokens": gen["tokens"],
                            **metric,
                        }
                        rows.append(row)
                        if len(examples) < args.example_limit:
                            examples.append(row)

        summary = summarize(rows)
        log("Generation effects:")
        for r in summary["generation_effects"][:18]:
            log(
                f"  {r['pair_task']} {r['site']} L{r['candidate_layer']} {r['candidate_component']}: "
                f"exact {r['base_exact']}->{r['ablation_exact']} d={r['delta_exact']} "
                f"tok0 {r['base_tok0']}->{r['ablation_tok0']} d={r['delta_tok0']} "
                f"rank {r['base_rank']:.1f}->{r['ablation_rank']:.1f}"
            )
        return {
            "phase": 657,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "candidate_specs": candidates,
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
    parser.add_argument("--max-candidate-specs", type=int, default=6)
    parser.add_argument("--min-candidate-n", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=280)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 12
        args.max_cases = 1
        args.max_candidate_specs = 2
        args.min_candidate_n = 1
        args.top_k = min(args.top_k, 20)
        args.max_new_tokens = min(args.max_new_tokens, 4)
        args.example_limit = 120
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 40)
        args.max_samples = max(args.max_samples, 320)
        args.max_cases = max(args.max_cases, 20)
        args.max_candidate_specs = max(args.max_candidate_specs, 6)
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
    out_path = out_dir / f"phase657_{args.model}_format_prior_writer_generation_confirmation_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
