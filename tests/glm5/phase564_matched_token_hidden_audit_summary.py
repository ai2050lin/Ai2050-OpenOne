#!/usr/bin/env python3
"""Cross-model summary for Phase564 matched-token hidden audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase564_matched_token_hidden_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
CONDITIONS = [
    "baseline",
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_random",
    "add_normal_repeat2",
]
LABELS = {
    "baseline": "base",
    "one_shot_repeat2": "r2",
    "one_shot_repeat4": "r4",
    "one_shot_random": "rand",
    "add_normal_repeat2": "norm_r2",
}


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase564_{model}_matched_token_hidden_audit.json"
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
    if "." in field:
        head, tail = field.split(".", 1)
        return float(r.get(head, {}).get(tail, 0.0))
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

    lines = ["# Phase564 Matched-Token Hidden Audit Cross-Model Summary", ""]
    lines.append("free = baseline free generation vs intervention free generation.")
    lines.append("matched_base = both runs forced through baseline generated tokens.")
    lines.append("matched_condition = both runs forced through intervention generated tokens.")
    lines.append("")

    for route in ROUTES:
        lines.append(f"## Route: {route}")
        lines.append("")
        table(lines, "clean_non_object_rate", data, route, "clean_non_object_rate")
        table(lines, "exact_sequence_match_rate", data, route, "token_divergence.exact_sequence_match_rate")
        table(lines, "avg_first_divergence_step", data, route, "token_divergence.avg_first_divergence_step")
        table(lines, "free trajectory_distance", data, route, "hidden_free.avg_trajectory_distance")
        table(lines, "matched_base trajectory_distance", data, route, "hidden_matched_base.avg_trajectory_distance")
        table(lines, "matched_condition trajectory_distance", data, route, "hidden_matched_condition.avg_trajectory_distance")
        table(lines, "free hidden_relax_step", data, route, "hidden_free.avg_hidden_relax_step")
        table(lines, "matched_base hidden_relax_step", data, route, "hidden_matched_base.avg_hidden_relax_step")
        table(lines, "matched_condition hidden_relax_step", data, route, "hidden_matched_condition.avg_hidden_relax_step")
        table(lines, "free finite_time_growth", data, route, "hidden_free.avg_finite_time_log_growth")
        table(lines, "matched_base finite_time_growth", data, route, "hidden_matched_base.avg_finite_time_log_growth")
        table(lines, "matched_condition finite_time_growth", data, route, "hidden_matched_condition.avg_finite_time_log_growth")

        lines.append("### free vs matched_base retention")
        lines.append("")
        lines.append("| model | condition | clean_delta | free_traj | matched_base_traj | retention | seq_match | first_div |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = metric(data[model], route, "baseline", "clean_non_object_rate") or 0.0
            for cond in CONDITIONS:
                if cond == "baseline":
                    continue
                clean = metric(data[model], route, cond, "clean_non_object_rate") or 0.0
                free = metric(data[model], route, cond, "hidden_free.avg_trajectory_distance") or 0.0
                matched = metric(data[model], route, cond, "hidden_matched_base.avg_trajectory_distance") or 0.0
                retention = matched / free if free > 1e-8 else 0.0
                seq = metric(data[model], route, cond, "token_divergence.exact_sequence_match_rate") or 0.0
                div = metric(data[model], route, cond, "token_divergence.avg_first_divergence_step") or 0.0
                lines.append(
                    f"| {model} | {LABELS[cond]} | {fmt(clean - base, True)} | "
                    f"{fmt(free)} | {fmt(matched)} | {fmt(retention)} | {fmt(seq)} | {fmt(div)} |"
                )
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

    out = root / "phase564_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
