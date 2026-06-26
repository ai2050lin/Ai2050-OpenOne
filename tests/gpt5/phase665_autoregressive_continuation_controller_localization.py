#!/usr/bin/env python3
"""
Phase 665: Autoregressive Continuation Controller Localization.

Moves beyond first-token readout. It selects cases where correct_prefix is
already top1 but exact generation is still wrong, then audits token1/token2
continuation under the true autoregressive input:

  prompt + forced correct previous tokens

For each continuation step, it patches the hidden state at the last generated
token position from the short_value_allowed source path into the task path,
scanning layer_input/layer_out/attn_out/mlp_out to localize continuation
controller candidates.
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

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import get_mlp, replace_input  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval  # noqa: E402
from phase630_distributed_format_route_multisource import install_source_patch_hooks  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import make_prompt, position_units, select_cases  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS, build_site_patch, collect_caches  # noqa: E402
from phase659_final_top1_barrier_readout_audit import TASK_ORDER, load_best_combos, token_category  # noqa: E402
from phase661_last_writer_combo_generation_closure import (  # noqa: E402
    greedy_generate,
    install_all_ablation_hooks,
    load_last_writer_specs,
    probe_mode,
)
from phase662_residual_to_lmhead_projection_barrier_audit import readout_metric  # noqa: E402


OUT_ROOT = Path("results/glm5_phase665_autoregressive_continuation_controller_localization")
SCAN_COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_scan_layers(model_key: str, n_layers: int) -> List[int]:
    if model_key == "qwen3":
        raw = range(max(0, n_layers - 16), n_layers)
    elif model_key == "glm4":
        raw = range(max(0, n_layers - 18), n_layers)
    else:
        raw = range(max(0, n_layers - 14), n_layers)
    return list(raw)


def token_top_metric(tokenizer, logits: torch.Tensor, expected_id: int, top_k: int) -> Dict:
    topv, topi = torch.topk(logits.float(), k=top_k)
    expected_logit = float(logits[expected_id].item())
    expected_rank = 1 + int((logits > logits[expected_id]).sum().item())
    top_rows = []
    for rank, (v, i) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        tid = int(i)
        top_rows.append({"rank": rank, "id": tid, "text": tokenizer.decode([tid]), "logit": float(v)})
    top1 = top_rows[0]
    return {
        "expected_id": int(expected_id),
        "expected_text": tokenizer.decode([int(expected_id)]),
        "expected_rank": expected_rank,
        "expected_logit": expected_logit,
        "top1": top1,
        "expected_minus_top1": float(expected_logit - top1["logit"]),
        "top_rows": top_rows[:10],
    }


def install_id_patch_hooks(
    model,
    patches: List[Tuple[int, str, List[int], List[torch.Tensor]]],
):
    layers = get_layers(model)
    handles = []
    for li, component, positions, targets in patches:
        if not positions or len(positions) != len(targets):
            continue
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        pos_targets = [(int(p), t.float().cpu()) for p, t in zip(positions, targets)]

        def patch_tensor(tensor):
            y = tensor.clone()
            for p, target in pos_targets:
                if 0 <= p < y.shape[1]:
                    y[0, p, :] = target.to(device=y.device, dtype=y.dtype)
            return y

        if component == "layer_input":
            def hook(_module, inputs, patch_tensor=patch_tensor):
                return replace_input(inputs, patch_tensor(inputs[0]))
            handles.append(layer.register_forward_pre_hook(hook))
        elif component == "layer_out":
            def hook(_module, _inputs, output, patch_tensor=patch_tensor):
                y_new = patch_tensor(extract_tensor(output))
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(layer.register_forward_hook(hook))
        elif component == "attn_out" and attn is not None:
            def hook(_module, _inputs, output, patch_tensor=patch_tensor):
                y_new = patch_tensor(extract_tensor(output))
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(attn.register_forward_hook(hook))
        elif component == "mlp_out" and mlp is not None:
            def hook(_module, _inputs, output, patch_tensor=patch_tensor):
                y_new = patch_tensor(extract_tensor(output))
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new
            handles.append(mlp.register_forward_hook(hook))
    return handles


def collect_id_components(
    model,
    device,
    ids: List[int],
    pos: int,
    layers_to_scan: List[int],
    components: List[str],
) -> Dict[int, Dict[str, torch.Tensor]]:
    layers = get_layers(model)
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    def save(layer_idx, comp, tensor):
        if comp not in components:
            return
        if 0 <= pos < tensor.shape[1]:
            captured[layer_idx][comp] = tensor[0, pos].detach().float().cpu()

    for li in layers_to_scan:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)

        def make_layer_pre(layer_idx):
            def hook(_module, inputs):
                save(layer_idx, "layer_input", inputs[0])
            return hook

        def make_layer_out(layer_idx):
            def hook(_module, _inputs, output):
                save(layer_idx, "layer_out", extract_tensor(output))
            return hook

        def make_attn_out(layer_idx):
            def hook(_module, _inputs, output):
                save(layer_idx, "attn_out", extract_tensor(output))
            return hook

        def make_mlp_out(layer_idx):
            def hook(_module, _inputs, output):
                save(layer_idx, "mlp_out", extract_tensor(output))
            return hook

        if "layer_input" in components:
            handles.append(layer.register_forward_pre_hook(make_layer_pre(li)))
        if "layer_out" in components:
            handles.append(layer.register_forward_hook(make_layer_out(li)))
        if attn is not None and "attn_out" in components:
            handles.append(attn.register_forward_hook(make_attn_out(li)))
        if mlp is not None and "mlp_out" in components:
            handles.append(mlp.register_forward_hook(make_mlp_out(li)))

    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captured


def logits_with_task_hooks(
    model,
    tokenizer,
    device,
    ids: List[int],
    original_prompt: str,
    source_patches,
    combo,
    last_writers,
    extra_patches=None,
) -> torch.Tensor:
    handles = []
    try:
        if source_patches:
            handles.extend(install_source_patch_hooks(model, tokenizer, original_prompt, source_patches))
        if combo or last_writers:
            handles.extend(install_all_ablation_hooks(model, tokenizer, original_prompt, combo, last_writers))
        if extra_patches:
            handles.extend(install_id_patch_hooks(model, extra_patches))
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return logits.detach().cpu()
    finally:
        for h in handles:
            h.remove()


def continuation_tag(row_eval: Dict, post: Dict) -> str:
    if row_eval["exact_correct"]:
        return "exact_correct"
    if post["top1"]["category"] == "correct_prefix":
        return "correct_prefix_but_generation_wrong"
    return "first_token_competition_failure"


def summarize(rows: List[Dict], selected_failures: List[Dict]) -> Dict:
    selected_by_mode: Dict[Tuple, Dict] = {}
    for item in selected_failures:
        key = (item["pair_task"], item["site"], item["combo_name"])
        s = selected_by_mode.setdefault(key, {
            "pair_task": item["pair_task"],
            "site": item["site"],
            "combo_name": item["combo_name"],
            "n": 0,
            "generation_text": {},
        })
        s["n"] += 1
        text = item["generation_text"].replace("\n", "\\n")
        s["generation_text"][text] = s["generation_text"].get(text, 0) + 1

    baseline_by_step: Dict[Tuple, Dict] = {}
    component_by_key: Dict[Tuple, Dict] = {}
    for row in rows:
        bkey = (row["pair_task"], row["site"], row["combo_name"], row["step"])
        base = baseline_by_step.setdefault(bkey, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "step": row["step"],
            "n": 0,
            "expected_top1": 0,
            "sum_rank": 0.0,
            "sum_margin": 0.0,
            "top1_text": {},
        })
        if row["kind"] == "baseline":
            base["n"] += 1
            base["expected_top1"] += int(row["expected_rank"] == 1)
            base["sum_rank"] += row["expected_rank"]
            base["sum_margin"] += row["expected_minus_top1"]
            text = row["top1_text"].replace("\n", "\\n")
            base["top1_text"][text] = base["top1_text"].get(text, 0) + 1
            continue
        if row["kind"] != "component_patch":
            continue
        key = (row["pair_task"], row["site"], row["combo_name"], row["step"], row["layer"], row["component"])
        item = component_by_key.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "step": row["step"],
            "layer": row["layer"],
            "component": row["component"],
            "n": 0,
            "flipped_to_expected": 0,
            "sum_rank_delta": 0.0,
            "sum_margin_delta": 0.0,
            "patched_top1": {},
        })
        item["n"] += 1
        item["flipped_to_expected"] += int(row["expected_rank"] == 1 and row["baseline_expected_rank"] != 1)
        item["sum_rank_delta"] += row["baseline_expected_rank"] - row["expected_rank"]
        item["sum_margin_delta"] += row["expected_minus_top1"] - row["baseline_expected_minus_top1"]
        text = row["top1_text"].replace("\n", "\\n")
        item["patched_top1"][text] = item["patched_top1"].get(text, 0) + 1

    selected_out = []
    for s in selected_by_mode.values():
        r = dict(s)
        r["generation_text"] = dict(sorted(s["generation_text"].items(), key=lambda kv: kv[1], reverse=True)[:10])
        selected_out.append(r)

    baseline_out = []
    for b in baseline_by_step.values():
        n = max(1, b["n"])
        if b["n"] == 0:
            continue
        r = dict(b)
        r["expected_top1_rate"] = b["expected_top1"] / n
        r["mean_expected_rank"] = b["sum_rank"] / n
        r["mean_expected_minus_top1"] = b["sum_margin"] / n
        r["top1_text"] = dict(sorted(b["top1_text"].items(), key=lambda kv: kv[1], reverse=True))
        baseline_out.append(r)

    component_out = []
    for item in component_by_key.values():
        n = max(1, item["n"])
        r = dict(item)
        r["flip_rate"] = item["flipped_to_expected"] / n
        r["mean_rank_improvement"] = item["sum_rank_delta"] / n
        r["mean_margin_delta"] = item["sum_margin_delta"] / n
        r["patched_top1"] = dict(sorted(item["patched_top1"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        component_out.append(r)

    component_out.sort(key=lambda r: (-r["mean_margin_delta"], -r["mean_rank_improvement"], -r["flip_rate"]))
    baseline_out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["step"]))
    selected_out.sort(key=lambda r: (-r["n"], r["pair_task"], r["site"], r["combo_name"]))
    return {
        "selected_continuation_failures": selected_out,
        "continuation_baselines": baseline_out,
        "component_patch_candidates": component_out[:160],
        "component_patch_all": component_out,
    }


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        scan_layers = [li for li in (args.layers or default_scan_layers(args.model, info.n_layers)) if 0 <= li < info.n_layers]
        components = [c.strip() for c in args.components.split(",") if c.strip()]
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
        selected_failures = []
        filtered = {"position_missing": 0, "position_len_mismatch": 0, "empty_patch": 0, "short_answer_ids": 0}
        log(f"{args.model}: selected={len(selected)}, scan_layers={scan_layers}, combo_specs={combo_specs}")

        for item_i, item in enumerate(selected):
            if len(selected_failures) >= args.max_continuation_cases:
                break
            case = item["case"]
            correct_ids = answer_ids(tokenizer, case["correct"])
            if len(correct_ids) < 2:
                filtered["short_answer_ids"] += 1
                continue
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
                if len(selected_failures) >= args.max_continuation_cases:
                    break
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
                    if len(selected_failures) >= args.max_continuation_cases:
                        break
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
                    gen = greedy_generate(model, tokenizer, device, task_prompt, args.max_new_tokens, patches, combo, last_writers)
                    ev = generation_eval(gen, correct_ids, old_wrong_ids)
                    post = readout_metric(tokenizer, probe["logits"], prefix_id, value_prefix_ids, args.top_k)
                    tag = continuation_tag(ev, post)
                    if tag != "correct_prefix_but_generation_wrong":
                        continue
                    failure = {
                        "sample_idx": item["sample_idx"],
                        "item_idx": item_i,
                        "pair_task": task,
                        "site": combo_spec["site"],
                        "combo_name": combo_spec["combo_name"],
                        "combo": combo,
                        "last_writers": last_writers,
                        "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                        "task_prompt": task_prompt,
                        "value_prompt": value_prompt,
                        "prefix_id": prefix_id,
                        "prefix_text": tokenizer.decode([prefix_id]),
                        "correct_ids": correct_ids,
                        "generation_text": gen["text"],
                        "generation_tokens": gen["tokens"],
                        "post": post,
                    }
                    selected_failures.append(failure)

                    for step in range(1, min(args.steps, len(correct_ids) - 1) + 1):
                        forced_prev = correct_ids[:step]
                        expected_id = correct_ids[step]
                        task_ids = tokenizer.encode(task_prompt, add_special_tokens=False) + forced_prev
                        value_ids = tokenizer.encode(value_prompt, add_special_tokens=False) + forced_prev
                        pos = len(task_ids) - 1
                        source_cache = collect_id_components(model, device, value_ids, pos, scan_layers, components)
                        baseline_logits = logits_with_task_hooks(
                            model, tokenizer, device, task_ids, task_prompt, patches, combo, last_writers
                        )
                        base = token_top_metric(tokenizer, baseline_logits, expected_id, args.top_k)
                        base_row = {
                            "kind": "baseline",
                            "sample_idx": item["sample_idx"],
                            "pair_task": task,
                            "site": combo_spec["site"],
                            "combo_name": combo_spec["combo_name"],
                            "step": step,
                            "forced_prev_text": tokenizer.decode(forced_prev),
                            "expected_id": expected_id,
                            "expected_text": base["expected_text"],
                            "expected_rank": base["expected_rank"],
                            "expected_minus_top1": base["expected_minus_top1"],
                            "top1_id": base["top1"]["id"],
                            "top1_text": base["top1"]["text"],
                            "top_rows": base["top_rows"],
                        }
                        rows.append(base_row)
                        for li in scan_layers:
                            for comp in components:
                                target = source_cache.get(li, {}).get(comp)
                                if target is None:
                                    continue
                                extra = [(li, comp, [pos], [target])]
                                logits = logits_with_task_hooks(
                                    model, tokenizer, device, task_ids, task_prompt, patches, combo, last_writers, extra_patches=extra
                                )
                                met = token_top_metric(tokenizer, logits, expected_id, args.top_k)
                                row = {
                                    "kind": "component_patch",
                                    "sample_idx": item["sample_idx"],
                                    "pair_task": task,
                                    "site": combo_spec["site"],
                                    "combo_name": combo_spec["combo_name"],
                                    "step": step,
                                    "layer": li,
                                    "component": comp,
                                    "forced_prev_text": tokenizer.decode(forced_prev),
                                    "expected_id": expected_id,
                                    "expected_text": met["expected_text"],
                                    "expected_rank": met["expected_rank"],
                                    "expected_minus_top1": met["expected_minus_top1"],
                                    "baseline_expected_rank": base["expected_rank"],
                                    "baseline_expected_minus_top1": base["expected_minus_top1"],
                                    "top1_id": met["top1"]["id"],
                                    "top1_text": met["top1"]["text"],
                                }
                                rows.append(row)
                                if len(examples) < args.example_limit and (
                                    row["expected_rank"] < base["expected_rank"] or row["expected_minus_top1"] > base["expected_minus_top1"]
                                ):
                                    examples.append(row)

        summary = summarize(rows, selected_failures)
        log("Selected continuation failures:")
        for r in summary["selected_continuation_failures"]:
            log(f"  {r['pair_task']} {r['site']} {r['combo_name']} n={r['n']} gen={r['generation_text']}")
        log("Continuation baselines:")
        for r in summary["continuation_baselines"]:
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} step={r['step']} n={r['n']} "
                f"top1_rate={r['expected_top1_rate']:.2f} rank={r['mean_expected_rank']:.2f} margin={r['mean_expected_minus_top1']:.3f}"
            )
        log("Top component patch candidates:")
        for r in summary["component_patch_candidates"][:20]:
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} step={r['step']} "
                f"L{r['layer']} {r['component']} n={r['n']} margin_delta={r['mean_margin_delta']:.3f} "
                f"rank_imp={r['mean_rank_improvement']:.2f} flip={r['flip_rate']:.2f}"
            )
        return {
            "phase": 665,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "scan_layers": scan_layers,
            "scan_components": components,
            "combo_specs": combo_specs,
            "last_writer_map": {str(k): v for k, v in last_map.items()},
            "tasks": TASK_ORDER,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "steps": args.steps,
            "n_raw_cases": len(raw_cases),
            "n_selected_items": len(selected),
            "n_continuation_failures": len(selected_failures),
            "n_rows": len(rows),
            "max_cases": args.max_cases,
            "max_continuation_cases": args.max_continuation_cases,
            "selection_stats": selection_stats,
            "filtered": filtered,
            "summary": summary,
            "examples": examples,
            "selected_failures": selected_failures if args.save_rows else selected_failures[: args.example_limit],
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
    parser.add_argument("--n-tables", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--max-continuation-cases", type=int, default=8)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--max-last-writers", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--components", default="layer_input,attn_out,mlp_out,layer_out")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2)
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
        args.max_cases = 2
        args.max_continuation_cases = 1
        args.max_per_task = 1
        args.max_last_writers = 1
        args.layers = None
        args.components = "layer_input,mlp_out"
        args.top_k = min(args.top_k, 20)
        args.max_new_tokens = min(args.max_new_tokens, 4)
        args.steps = 1
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 64)
        args.max_samples = max(args.max_samples, 512)
        args.max_cases = max(args.max_cases, 64)
        args.max_continuation_cases = max(args.max_continuation_cases, 12)
        args.max_per_task = max(args.max_per_task, 2)
        args.max_last_writers = min(max(args.max_last_writers, 2), 2)
        args.top_k = max(args.top_k, 30)
        args.max_new_tokens = min(max(args.max_new_tokens, 6), 6)
        args.steps = min(max(args.steps, 2), 2)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase665_{args.model}_autoregressive_continuation_controller_localization_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
