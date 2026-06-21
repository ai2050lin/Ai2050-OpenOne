#!/usr/bin/env python3
"""Cross-model summary for Phase563 hidden trajectory distance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase563_hidden_trajectory_distance")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
CONDITIONS = [
    "baseline",
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_mean",
    "one_shot_random",
    "add_tangent_repeat2",
    "add_normal_repeat2",
]
LABELS = {
    "baseline": "base",
    "one_shot_repeat2": "r2",
    "one_shot_repeat4": "r4",
    "one_shot_mean": "mean",
    "one_shot_random": "rand",
    "add_tangent_repeat2": "tan_r2",
    "add_normal_repeat2": "norm_r2",
}


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase563_{model}_hidden_trajectory_distance.json"
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
    if field.startswith("hidden."):
        return float(r.get("hidden_metrics", {}).get(field.split(".", 1)[1], 0.0))
    return float(r.get(field, 0.0))


def fmt(v: float | None, sign: bool = False) -> str:
    if v is None:
        return ""
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def table(lines: list[str], title: str, data: dict[str, dict[str, Any]], route: str, field: str) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| model | " + " | ".join(LABELS[c] for c in CONDITIONS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(CONDITIONS)) + "|")
    for model in MODELS:
        if model not in data:
            continue
        vals = [fmt(metric(data[model], route, c, field)) for c in CONDITIONS]
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase563 Hidden Trajectory Distance Cross-Model Summary", ""]
    lines.append("Metrics: hidden deltas are measured against same-seed baseline trajectories.")
    lines.append("hidden_relax_step uses first step where delta_ratio <= epsilon_ratio.")
    lines.append("")

    for route in ROUTES:
        lines.append(f"## Route: {route}")
        lines.append("")
        table(lines, "clean_non_object_rate", data, route, "clean_non_object_rate")
        table(lines, "hidden_relax_step", data, route, "hidden.avg_hidden_relax_step")
        table(lines, "finite_time_log_growth", data, route, "hidden.avg_finite_time_log_growth")
        table(lines, "trajectory_distance", data, route, "hidden.avg_trajectory_distance")
        table(lines, "last_delta_ratio", data, route, "hidden.avg_delta_ratio_last")

        lines.append("### tangent vs normal")
        lines.append("")
        lines.append("| model | baseline | tangent clean | normal clean | tangent growth | normal growth | tangent traj | normal traj |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = metric(data[model], route, "baseline", "clean_non_object_rate")
            tc = metric(data[model], route, "add_tangent_repeat2", "clean_non_object_rate")
            nc = metric(data[model], route, "add_normal_repeat2", "clean_non_object_rate")
            tg = metric(data[model], route, "add_tangent_repeat2", "hidden.avg_finite_time_log_growth")
            ng = metric(data[model], route, "add_normal_repeat2", "hidden.avg_finite_time_log_growth")
            tt = metric(data[model], route, "add_tangent_repeat2", "hidden.avg_trajectory_distance")
            nt = metric(data[model], route, "add_normal_repeat2", "hidden.avg_trajectory_distance")
            lines.append(f"| {model} | {fmt(base)} | {fmt(tc)} | {fmt(nc)} | {fmt(tg)} | {fmt(ng)} | {fmt(tt)} | {fmt(nt)} |")
        lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("| model | minutes | seeds | max tokens | epsilon |")
    lines.append("|---|---:|---|---:|---:|")
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        lines.append(f"| {model} | {d.get('total_time_min')} | {d.get('sample_seeds')} | {d.get('max_new_tokens')} | {d.get('epsilon_ratio')} |")
    lines.append("")

    out = root / "phase563_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
