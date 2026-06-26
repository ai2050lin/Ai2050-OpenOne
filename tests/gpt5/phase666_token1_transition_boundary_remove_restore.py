#!/usr/bin/env python3
"""
Phase 666: Token0-to-Token1 Transition Boundary Remove/Restore Audit.

Phase 665 localized continuation failures to the first transition after a
correct answer prefix. This phase stops broad scanning and audits only the
earliest effective boundary candidates for each model.

For selected Phase 665 failures:
  baseline:      task path with the same protocol patches and last-writer ablations
  self_restore:  task boundary state restored into itself, a no-op control
  zero_remove:   boundary state replaced by zero at the continuation position
  mismatch_restore: boundary state from another value prompt, semantic mismatch control
  correct_restore:  boundary state from the matching short_value_allowed value prompt

The target is token1, under input:
  task_prompt + correct_prefix
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
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase651_task_intent_gate_protocol_boundary_audit import TASKS, position_units  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS, build_site_patch, collect_caches  # noqa: E402
from phase661_last_writer_combo_generation_closure import install_all_ablation_hooks  # noqa: E402
from phase665_autoregressive_continuation_controller_localization import (  # noqa: E402
    collect_id_components,
    install_id_patch_hooks,
    token_top_metric,
)
from phase630_distributed_format_route_multisource import install_source_patch_hooks  # noqa: E402


PHASE665_ROOT = Path("results/glm5_phase665_autoregressive_continuation_controller_localization")
OUT_ROOT = Path("results/glm5_phase666_token1_transition_boundary_remove_restore")
BOUNDARIES = {
    "qwen3": [
        {"layer": 22, "component": "attn_out", "label": "L22_attn_out"},
        {"layer": 23, "component": "layer_input", "label": "L23_layer_input"},
    ],
    "glm4": [
        {"layer": 22, "component": "layer_input", "label": "L22_layer_input"},
        {"layer": 22, "component": "attn_out", "label": "L22_attn_out"},
        {"layer": 22, "component": "layer_out", "label": "L22_layer_out"},
    ],
    "deepseek7b": [
        {"layer": 21, "component": "layer_out", "label": "L21_layer_out"},
        {"layer": 22, "component": "layer_input", "label": "L22_layer_input"},
        {"layer": 22, "component": "layer_out", "label": "L22_layer_out"},
    ],
}
INTERVENTIONS = ["baseline", "self_restore", "zero_remove", "mismatch_restore", "correct_restore"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase665_path(model_key: str) -> Path:
    return PHASE665_ROOT / f"phase665_{model_key}_autoregressive_continuation_controller_localization_confirm.json"


def parse_question_relation(prompt: str, category: str, fallback: str) -> str:
    match = re.search(rf"Question:\s*{re.escape(category)}\s+(.+?)\s+\?", prompt)
    if match:
        return match.group(1)
    return fallback


def prompt_intent(task: str) -> str:
    return "value" if task == "short_value_allowed" else TASKS.get(task, "value")


def compute_source_patches(
    model,
    tokenizer,
    device,
    info,
    failure: Dict,
    site_specs: Dict,
) -> Tuple[List, Dict]:
    site_name = failure["site"]
    site = site_specs[site_name]
    site_layers = [li for li in site["layers"] if 0 <= li < info.n_layers]
    site_components = list(site["components"])
    site_positions = list(site["positions"])
    case = failure["case"]
    value_relation = case["relation"]
    task_relation = parse_question_relation(failure["task_prompt"], case["category"], case["relation"])
    value_units_all = position_units(tokenizer, failure["value_prompt"], case, value_relation, "value")
    task_units_all = position_units(tokenizer, failure["task_prompt"], case, task_relation, prompt_intent(failure["pair_task"]))
    value_units = {p: value_units_all.get(p, []) for p in site_positions}
    task_units = {p: task_units_all.get(p, []) for p in site_positions}
    value_caches = collect_caches(model, tokenizer, device, failure["value_prompt"], value_units, site_layers, site_components)
    task_caches = collect_caches(model, tokenizer, device, failure["task_prompt"], task_units, site_layers, site_components)
    patches, stats = build_site_patch(
        task_caches,
        value_caches,
        task_units,
        value_units,
        site,
        site_layers,
        int(failure["sample_idx"]) * 1009 + len(failure["pair_task"]) * 199,
    )
    return patches, stats


def logits_with_hooks(
    model,
    tokenizer,
    device,
    ids: List[int],
    prompt: str,
    source_patches,
    combo,
    last_writers,
    extra_patches=None,
) -> torch.Tensor:
    handles = []
    try:
        if source_patches:
            handles.extend(install_source_patch_hooks(model, tokenizer, prompt, source_patches))
        if combo or last_writers:
            handles.extend(install_all_ablation_hooks(model, tokenizer, prompt, combo, last_writers))
        if extra_patches:
            handles.extend(install_id_patch_hooks(model, extra_patches))
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return logits.detach().cpu()
    finally:
        for h in handles:
            h.remove()


def pick_mismatch(failures: List[Dict], i: int) -> Dict:
    target = failures[i]
    for off in range(1, len(failures) + 1):
        cand = failures[(i + off) % len(failures)]
        if cand["case"]["correct"] != target["case"]["correct"]:
            return cand
    return failures[(i + 1) % len(failures)]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: List[Dict], failures: List[Dict]) -> Dict:
    groups: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"], row["boundary"], row["intervention"])
        g = groups.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "boundary": row["boundary"],
            "intervention": row["intervention"],
            "n": 0,
            "top1": 0,
            "ranks": [],
            "margins": [],
            "rank_delta_vs_baseline": [],
            "margin_delta_vs_baseline": [],
            "top1_text": {},
        })
        g["n"] += 1
        g["top1"] += int(row["expected_rank"] == 1)
        g["ranks"].append(float(row["expected_rank"]))
        g["margins"].append(float(row["expected_minus_top1"]))
        g["rank_delta_vs_baseline"].append(float(row["baseline_expected_rank"] - row["expected_rank"]))
        g["margin_delta_vs_baseline"].append(float(row["expected_minus_top1"] - row["baseline_expected_minus_top1"]))
        text = row["top1_text"].replace("\n", "\\n")
        g["top1_text"][text] = g["top1_text"].get(text, 0) + 1

    out = []
    for g in groups.values():
        r = dict(g)
        r["expected_top1_rate"] = g["top1"] / max(1, g["n"])
        r["mean_expected_rank"] = mean(g["ranks"])
        r["mean_expected_minus_top1"] = mean(g["margins"])
        r["mean_rank_delta_vs_baseline"] = mean(g["rank_delta_vs_baseline"])
        r["mean_margin_delta_vs_baseline"] = mean(g["margin_delta_vs_baseline"])
        r["top1_text"] = dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        for k in ["ranks", "margins", "rank_delta_vs_baseline", "margin_delta_vs_baseline", "top1"]:
            r.pop(k, None)
        out.append(r)

    by_boundary: Dict[Tuple, Dict] = {}
    for item in out:
        key = (item["pair_task"], item["site"], item["combo_name"], item["boundary"])
        b = by_boundary.setdefault(key, {
            "pair_task": item["pair_task"],
            "site": item["site"],
            "combo_name": item["combo_name"],
            "boundary": item["boundary"],
            "n": item["n"],
            "interventions": {},
        })
        b["interventions"][item["intervention"]] = item
    boundary_out = []
    for b in by_boundary.values():
        correct = b["interventions"].get("correct_restore", {})
        mismatch = b["interventions"].get("mismatch_restore", {})
        zero = b["interventions"].get("zero_remove", {})
        self_r = b["interventions"].get("self_restore", {})
        b["correct_minus_mismatch_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - mismatch.get("mean_margin_delta_vs_baseline", 0.0)
        )
        b["correct_minus_zero_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - zero.get("mean_margin_delta_vs_baseline", 0.0)
        )
        b["self_control_abs_margin_delta"] = abs(self_r.get("mean_margin_delta_vs_baseline", 0.0))
        boundary_out.append(b)
    boundary_out.sort(
        key=lambda r: (
            -r["correct_minus_mismatch_margin_delta"],
            -r["correct_minus_zero_margin_delta"],
            r["self_control_abs_margin_delta"],
        )
    )
    out.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["boundary"], INTERVENTIONS.index(r["intervention"])))
    return {
        "selected_failures": [
            {
                "pair_task": f["pair_task"],
                "site": f["site"],
                "combo_name": f["combo_name"],
                "correct": f["case"]["correct"],
                "generation_text": f["generation_text"],
                "correct_ids": f["correct_ids"],
            }
            for f in failures
        ],
        "intervention_summary": out,
        "boundary_specificity": boundary_out,
    }


def run_model(args) -> Dict:
    data_path = phase665_path(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Phase 665 confirm result: {data_path}")
    phase665 = json.loads(data_path.read_text(encoding="utf-8"))
    failures = phase665["selected_failures"][: args.max_failures]
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        site_specs = {s["name"]: s for s in SITE_SPECS[args.model]}
        boundaries = [b for b in BOUNDARIES[args.model] if 0 <= b["layer"] < info.n_layers]
        rows = []
        patch_stats = []
        log(f"{args.model}: failures={len(failures)}, boundaries={boundaries}")
        for i, failure in enumerate(failures):
            correct_ids = failure["correct_ids"]
            if len(correct_ids) < 2:
                continue
            expected_id = int(correct_ids[1])
            forced_prev = [int(correct_ids[0])]
            task_ids = tokenizer.encode(failure["task_prompt"], add_special_tokens=False) + forced_prev
            value_ids = tokenizer.encode(failure["value_prompt"], add_special_tokens=False) + forced_prev
            pos = len(task_ids) - 1
            mismatch = pick_mismatch(failures, i)
            mismatch_ids = tokenizer.encode(mismatch["value_prompt"], add_special_tokens=False) + forced_prev
            source_patches, stats = compute_source_patches(model, tokenizer, device, info, failure, site_specs)
            patch_stats.append({"sample_idx": failure["sample_idx"], "site": failure["site"], "stats": stats, "n_patches": len(source_patches)})
            combo = failure["combo"]
            last_writers = failure["last_writers"]
            baseline_logits = logits_with_hooks(
                model, tokenizer, device, task_ids, failure["task_prompt"], source_patches, combo, last_writers
            )
            baseline = token_top_metric(tokenizer, baseline_logits, expected_id, args.top_k)

            for boundary in boundaries:
                layer = int(boundary["layer"])
                component = boundary["component"]
                correct_cache = collect_id_components(model, device, value_ids, pos, [layer], [component])
                task_cache = collect_id_components(model, device, task_ids, pos, [layer], [component])
                mismatch_cache = collect_id_components(model, device, mismatch_ids, pos, [layer], [component])
                correct_vec = correct_cache.get(layer, {}).get(component)
                task_vec = task_cache.get(layer, {}).get(component)
                mismatch_vec = mismatch_cache.get(layer, {}).get(component)
                if correct_vec is None or task_vec is None or mismatch_vec is None:
                    continue
                vectors = {
                    "baseline": None,
                    "self_restore": task_vec,
                    "zero_remove": torch.zeros_like(task_vec),
                    "mismatch_restore": mismatch_vec,
                    "correct_restore": correct_vec,
                }
                for intervention, target in vectors.items():
                    extra = None if target is None else [(layer, component, [pos], [target])]
                    logits = (
                        baseline_logits
                        if intervention == "baseline"
                        else logits_with_hooks(
                            model,
                            tokenizer,
                            device,
                            task_ids,
                            failure["task_prompt"],
                            source_patches,
                            combo,
                            last_writers,
                            extra,
                        )
                    )
                    met = token_top_metric(tokenizer, logits, expected_id, args.top_k)
                    rows.append({
                        "sample_idx": failure["sample_idx"],
                        "pair_task": failure["pair_task"],
                        "site": failure["site"],
                        "combo_name": failure["combo_name"],
                        "boundary": boundary["label"],
                        "layer": layer,
                        "component": component,
                        "intervention": intervention,
                        "forced_prev_text": tokenizer.decode(forced_prev),
                        "expected_id": expected_id,
                        "expected_text": met["expected_text"],
                        "expected_rank": met["expected_rank"],
                        "expected_minus_top1": met["expected_minus_top1"],
                        "baseline_expected_rank": baseline["expected_rank"],
                        "baseline_expected_minus_top1": baseline["expected_minus_top1"],
                        "top1_id": met["top1"]["id"],
                        "top1_text": met["top1"]["text"],
                        "mismatch_correct": mismatch["case"]["correct"],
                    })
        summary = summarize(rows, failures)
        log("Boundary specificity:")
        for r in summary["boundary_specificity"][:20]:
            cor = r["interventions"].get("correct_restore", {})
            mis = r["interventions"].get("mismatch_restore", {})
            zer = r["interventions"].get("zero_remove", {})
            log(
                f"  {r['pair_task']} {r['site']} {r['combo_name']} {r['boundary']} "
                f"correct_delta={cor.get('mean_margin_delta_vs_baseline', 0):.3f} "
                f"mismatch_delta={mis.get('mean_margin_delta_vs_baseline', 0):.3f} "
                f"zero_delta={zer.get('mean_margin_delta_vs_baseline', 0):.3f} "
                f"correct_minus_mismatch={r['correct_minus_mismatch_margin_delta']:.3f}"
            )
        return {
            "phase": 666,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_phase665": str(data_path),
            "n_layers": info.n_layers,
            "boundaries": boundaries,
            "interventions": INTERVENTIONS,
            "top_k": args.top_k,
            "n_phase665_failures_loaded": len(phase665["selected_failures"]),
            "n_failures_tested": len(failures),
            "n_rows": len(rows),
            "patch_stats": patch_stats,
            "summary": summary,
            "rows": rows if args.save_rows else rows[: args.example_limit],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-failures", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=240)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_failures = min(args.max_failures, 1)
        args.top_k = min(args.top_k, 20)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.max_failures = max(args.max_failures, 12)
        args.top_k = max(args.top_k, 30)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase666_{args.model}_token1_transition_boundary_remove_restore_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
