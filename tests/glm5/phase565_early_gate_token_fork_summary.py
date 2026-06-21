#!/usr/bin/env python3
"""Cross-model summary for Phase565 early gate / token fork audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase565_early_gate_token_fork")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
INTERVENTIONS = [
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_random",
    "add_normal_repeat2",
]
LABELS = {
    "one_shot_repeat2": "r2",
    "one_shot_repeat4": "r4",
    "one_shot_random": "rand",
    "add_normal_repeat2": "norm_r2",
}


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase565_{model}_early_gate_token_fork.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row(data: dict[str, Any], route: str, condition: str) -> dict[str, Any] | None:
    combo = list(data["audit"].keys())[0]
    return data["audit"].get(combo, {}).get("rows", {}).get(route, {}).get(condition)


def metric(data: dict[str, Any], route: str, condition: str, field: str) -> float | None:
    r = row(data, route, condition)
    if not r:
        return None
    cur: Any = r
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def fmt(v: float | None, sign: bool = False) -> str:
    if v is None:
        return ""
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase565 Early Gate and Token Fork Cross-Model Summary", ""]
    lines.append("bfi = baseline forced to use intervention first token.")
    lines.append("ifb = intervention forced to use baseline first token.")
    lines.append("")

    for route in ROUTES:
        lines.append(f"## Route: {route}")
        lines.append("")
        lines.append("### Clean Rates")
        lines.append("")
        lines.append("| model | base | intervention | free | bfi | ifb | bfi_transfer | ifb_transfer | first_div |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = metric(data[model], route, "baseline_free", "clean_non_object_rate") or 0.0
            for intervention in INTERVENTIONS:
                free = metric(data[model], route, f"{intervention}_free", "clean_non_object_rate")
                bfi = metric(data[model], route, f"baseline_force_{intervention}_first", "clean_non_object_rate")
                ifb = metric(data[model], route, f"{intervention}_force_baseline_first", "clean_non_object_rate")
                btr = metric(data[model], route, f"{intervention}_free",
                             "forced_first_transfer.baseline_to_intervention_transfer_ratio")
                itr = metric(data[model], route, f"{intervention}_free",
                             "forced_first_transfer.intervention_to_baseline_transfer_ratio")
                div = metric(data[model], route, f"{intervention}_free", "avg_first_divergence_step")
                lines.append(
                    f"| {model} | {fmt(base)} | {LABELS[intervention]} | {fmt(free)} | "
                    f"{fmt(bfi)} | {fmt(ifb)} | {fmt(btr)} | {fmt(itr)} | {fmt(div)} |"
                )
        lines.append("")

        lines.append("### Step0 Target-Competitor Margin")
        lines.append("")
        lines.append("| model | base | r2 | r4 | rand | norm_r2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = metric(data[model], route, "baseline_free", "step0_avg_target_minus_competitor")
            vals = [metric(data[model], route, f"{i}_free", "step0_avg_target_minus_competitor") for i in INTERVENTIONS]
            lines.append(f"| {model} | {fmt(base)} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")

        lines.append("### Step1 Target-Competitor Margin")
        lines.append("")
        lines.append("| model | base | r2 | r4 | rand | norm_r2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = metric(data[model], route, "baseline_free", "step1_avg_target_minus_competitor")
            vals = [metric(data[model], route, f"{i}_free", "step1_avg_target_minus_competitor") for i in INTERVENTIONS]
            lines.append(f"| {model} | {fmt(base)} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")

        lines.append("### Fork Buckets For Free Intervention")
        lines.append("")
        lines.append("| model | intervention | early_rate | early_clean | middle_rate | middle_clean | late_rate | late_clean |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            for intervention in INTERVENTIONS:
                r = row(data[model], route, f"{intervention}_free") or {}
                dist = r.get("fork_bucket_distribution", {})
                buckets = r.get("fork_bucket_metrics", {})
                early = buckets.get("early_0_1", {})
                middle = buckets.get("middle_2_5", {})
                late = buckets.get("late_or_none_6p", {})
                lines.append(
                    f"| {model} | {LABELS[intervention]} | "
                    f"{fmt(float(dist.get('early_0_1', 0.0)))} | {fmt(float(early.get('clean_non_object_rate', 0.0)))} | "
                    f"{fmt(float(dist.get('middle_2_5', 0.0)))} | {fmt(float(middle.get('clean_non_object_rate', 0.0)))} | "
                    f"{fmt(float(dist.get('late_or_none_6p', 0.0)))} | {fmt(float(late.get('clean_non_object_rate', 0.0)))} |"
                )
        lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("| model | minutes | seeds | max tokens |")
    lines.append("|---|---:|---|---:|")
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        lines.append(f"| {model} | {d.get('total_time_min')} | {d.get('sample_seeds')} | {d.get('max_new_tokens')} |")
    lines.append("")

    out = root / "phase565_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
