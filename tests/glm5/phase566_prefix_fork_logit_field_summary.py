#!/usr/bin/env python3
"""Cross-model summary for Phase566 prefix fork and logit field audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase566_prefix_fork_logit_field")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTE = "forbidden_sentence_completion:temperature<-forbidden_definition"
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
PREFIXES = [1, 2, 3]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase566_{model}_prefix_fork_logit_field.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row(data: dict[str, Any], condition: str) -> dict[str, Any] | None:
    combo = list(data["audit"].keys())[0]
    return data["audit"].get(combo, {}).get("rows", {}).get(ROUTE, {}).get(condition)


def metric(data: dict[str, Any], condition: str, field: str) -> float | None:
    r = row(data, condition)
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

    lines = ["# Phase566 Prefix Fork and Logit Field Cross-Model Summary", ""]
    lines.append(f"Route: {ROUTE}")
    lines.append("")

    lines.append("## Free Clean and Step Margins")
    lines.append("")
    lines.append("| model | condition | clean | step0 | step1 | step2 | step3 | step4 | step5 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        if model not in data:
            continue
        for condition in ["baseline_free"] + [f"{i}_free" for i in INTERVENTIONS]:
            label = "base" if condition == "baseline_free" else LABELS[condition[:-5]]
            vals = [metric(data[model], condition, f"step{s}_avg_target_minus_competitor") for s in range(6)]
            clean = metric(data[model], condition, "clean_non_object_rate")
            lines.append(f"| {model} | {label} | {fmt(clean)} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")

    lines.append("## Prefix Transfer")
    lines.append("")
    lines.append("| model | intervention | free | bfi_p1 | bfi_p2 | bfi_p3 | ifb_p1 | ifb_p2 | ifb_p3 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        if model not in data:
            continue
        for intervention in INTERVENTIONS:
            free = metric(data[model], f"{intervention}_free", "clean_non_object_rate")
            bfi = [
                metric(data[model], f"baseline_force_{intervention}_prefix{p}", "clean_non_object_rate")
                for p in PREFIXES
            ]
            ifb = [
                metric(data[model], f"{intervention}_force_baseline_prefix{p}", "clean_non_object_rate")
                for p in PREFIXES
            ]
            lines.append(
                f"| {model} | {LABELS[intervention]} | {fmt(free)} | "
                + " | ".join(fmt(v) for v in bfi + ifb)
                + " |"
            )
    lines.append("")

    lines.append("## Prefix Transfer Ratios")
    lines.append("")
    lines.append("| model | intervention | bfi_r1 | bfi_r2 | bfi_r3 | ifb_r1 | ifb_r2 | ifb_r3 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        if model not in data:
            continue
        for intervention in INTERVENTIONS:
            r = row(data[model], f"{intervention}_free") or {}
            pt = r.get("prefix_transfer", {})
            bfi = [float(pt.get(str(p), {}).get("baseline_to_intervention_transfer_ratio", 0.0)) for p in PREFIXES]
            ifb = [float(pt.get(str(p), {}).get("intervention_to_baseline_transfer_ratio", 0.0)) for p in PREFIXES]
            lines.append(
                f"| {model} | {LABELS[intervention]} | "
                + " | ".join(fmt(v) for v in bfi + ifb)
                + " |"
            )
    lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("| model | minutes | seeds | max tokens | prefixes |")
    lines.append("|---|---:|---|---:|---|")
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        lines.append(f"| {model} | {d.get('total_time_min')} | {d.get('sample_seeds')} | {d.get('max_new_tokens')} | {d.get('prefix_lengths')} |")
    lines.append("")

    out = root / "phase566_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
