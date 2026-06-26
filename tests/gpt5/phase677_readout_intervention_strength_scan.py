#!/usr/bin/env python3
"""
Phase 677: Readout Intervention Strength Scan.

Extends Phase 676 by scanning intervention strength. This checks whether the
partial DS7B repair is a real readout-direction effect or an alpha-specific
artifact.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase676_late_readout_competitor_causal_suppression_audit import (  # noqa: E402
    forward_with_condition,
    load_rows,
    normalize,
    prompt_map,
    random_unit_like,
)


OUT_ROOT = Path("results/glm5_phase677_readout_intervention_strength_scan")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_conditions() -> list[dict]:
    out = [{"name": "baseline", "site": "none", "mode": "none"}]
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        name = str(alpha).replace(".", "p")
        out.append({
            "name": f"final_cancel_gap_a{name}",
            "site": "final_norm_output",
            "mode": "cancel_gap",
            "direction": "comp",
            "alpha": alpha,
        })
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        name = str(alpha).replace(".", "p")
        out.append({
            "name": f"final_remove_comp_a{name}",
            "site": "final_norm_output",
            "mode": "remove_projection",
            "direction": "comp",
            "alpha": alpha,
        })
    out.append({
        "name": "final_remove_random_a1p0",
        "site": "final_norm_output",
        "mode": "remove_projection",
        "direction": "random",
        "alpha": 1.0,
    })
    return out


def summarize(rows: list[dict]) -> dict:
    baseline = {r["case_id"]: r for r in rows if r["condition"] == "baseline"}
    groups = defaultdict(lambda: {
        "n": 0,
        "baseline_failures": 0,
        "baseline_successes": 0,
        "top1": 0,
        "rank_sum": 0.0,
        "gap_sum": 0.0,
        "gap_delta_sum": 0.0,
        "switches": 0,
        "damages": 0,
        "top1_text": {},
    })
    for row in rows:
        base = baseline[row["case_id"]]
        base_failure = not base["expected_top1"]
        base_success = base["expected_top1"]
        for scope in ["overall", row["base_top1_category"], row["relation"]]:
            key = f"{scope}|{row['condition']}"
            g = groups[key]
            g["n"] += 1
            g["baseline_failures"] += int(base_failure)
            g["baseline_successes"] += int(base_success)
            g["top1"] += int(row["expected_top1"])
            g["rank_sum"] += row["expected_rank"]
            g["gap_sum"] += row["gap"]
            g["gap_delta_sum"] += row["gap"] - base["gap"]
            g["switches"] += int(base_failure and row["expected_top1"])
            g["damages"] += int(base_success and not row["expected_top1"])
            text = row["top1_text"].replace("\n", "\\n")
            g["top1_text"][text] = g["top1_text"].get(text, 0) + 1
    out = {}
    for key, g in groups.items():
        scope, condition = key.split("|", 1)
        n = max(1, g["n"])
        failures = max(1, g["baseline_failures"])
        successes = max(1, g["baseline_successes"])
        out[key] = {
            "scope": scope,
            "condition": condition,
            "n": g["n"],
            "baseline_failures": g["baseline_failures"],
            "baseline_successes": g["baseline_successes"],
            "expected_top1_rate": g["top1"] / n,
            "mean_expected_rank": g["rank_sum"] / n,
            "mean_gap": g["gap_sum"] / n,
            "mean_gap_delta_vs_baseline": g["gap_delta_sum"] / n,
            "failure_switch_rate": g["switches"] / failures,
            "success_damage_rate": g["damages"] / successes,
            "top1_text": dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:8]),
        }
    return out


def run_model(args) -> dict:
    phase674_rows = load_rows(args.model, args.max_cases)
    prompts = prompt_map()
    conds = scan_conditions()
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        layers = get_layers(model)
        target_layer = len(layers) - 1
        prev_layer = max(0, target_layer - 1)
        unembed = model.get_output_embeddings().weight.detach().float().cpu()
        for i, row674 in enumerate(phase674_rows):
            expected_id = int(row674["expected_id"])
            competitor_id = int(row674["competitor"]["id"])
            readout_dir = normalize(unembed[competitor_id] - unembed[expected_id])
            random_dir = random_unit_like(readout_dir, seed=677000 + i)
            prompt = prompts[row674["case_id"]]
            baseline = forward_with_condition(
                model, tokenizer, device, prompt, expected_id, competitor_id,
                conds[0], readout_dir, random_dir, 0.0, target_layer, prev_layer
            )
            baseline_gap = baseline["gap"]
            for cond in conds:
                stats = baseline if cond["name"] == "baseline" else forward_with_condition(
                    model, tokenizer, device, prompt, expected_id, competitor_id,
                    cond, readout_dir, random_dir, baseline_gap, target_layer, prev_layer
                )
                rows.append({
                    "case_id": row674["case_id"],
                    "relation": row674["relation"],
                    "base_top1_category": row674["top1_category"],
                    "condition": cond["name"],
                    "expected_id": expected_id,
                    "competitor_id": competitor_id,
                    "competitor_text": row674["competitor"]["text"],
                    **stats,
                })
            if (i + 1) % 12 == 0 or i + 1 == len(phase674_rows):
                log(f"{args.model}: {i + 1}/{len(phase674_rows)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_ROOT / f"phase677_{args.model}_strength_scan_rows.jsonl"
    rows_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = {
        "phase": 677,
        "title": "Readout Intervention Strength Scan",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_cases": len(phase674_rows),
        "n_rows": len(rows),
        "conditions": [c["name"] for c in conds],
        "summary": summarize(rows),
    }
    out_path = OUT_ROOT / f"phase677_{args.model}_strength_scan_summary.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Wrote {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_ROOT.glob("phase677_*_strength_scan_summary.json"))]
    result = {
        "phase": 677,
        "title": "Readout Intervention Strength Scan Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase677_cross_model_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 677 Readout Intervention Strength Scan",
        "",
        f"- generated: `{result['timestamp']}`",
        "",
        "| model | condition | top1_rate | mean_rank | mean_gap | gap_delta | failure_switch | success_damage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        for cond in item["conditions"]:
            row = item["summary"].get(f"overall|{cond}")
            if not row:
                continue
            lines.append(
                f"| {item['model']} | {cond} | {row['expected_top1_rate']:.3f} | "
                f"{row['mean_expected_rank']:.2f} | {row['mean_gap']:.3f} | "
                f"{row['mean_gap_delta_vs_baseline']:.3f} | {row['failure_switch_rate']:.3f} | "
                f"{row['success_damage_rate']:.3f} |"
            )
    (OUT_ROOT / "phase677_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-cases", type=int, default=72)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
