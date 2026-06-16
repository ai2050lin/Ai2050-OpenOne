#!/usr/bin/env python3
"""Summarize Phase159 gain-readout / trajectory bridge results."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("results/gpt5_phase159_gain_readout_trajectory_bridge")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def top_counter(rows: list[dict[str, Any]], key: str) -> str:
    c: Counter[str] = Counter()
    for row in rows:
        rates = row["trajectory"]["trajectory_rates"]
        if not rates:
            continue
        name, _val = max(rates.items(), key=lambda kv: kv[1])
        c[name] += 1
    if not c:
        return ""
    name, n = c.most_common(1)[0]
    return f"{name}:{n}"


def collect_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for case_key, item in result["results"].items():
        row = dict(item)
        row["case_key"] = case_key
        out.append(row)
    return out


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[tuple[str, dict[str, float | str | int]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[group_key])].append(row)
    out = []
    for name, bucket in sorted(buckets.items()):
        hits = [float(r["trajectory"]["hit_rate"]) for r in bucket]
        out.append((name, {
            "n": len(bucket),
            "hit": mean(hits),
            "dcf": mean([float(r["readout"]["dcf_mean"]) for r in bucket]),
            "dcf_delta": mean([float(r["readout"]["dcf_delta"]) for r in bucket]),
            "proj_q_over_rms": mean([float(r["readout"]["proj_q_over_rms"]) for r in bucket]),
            "step1_margin": mean([float(r["trajectory"]["step_margins"].get("step1_correct_vs_competitor", 0.0)) for r in bucket]),
            "step2_margin": mean([float(r["trajectory"]["step_margins"].get("step2_correct_vs_competitor", 0.0)) for r in bucket]),
            "step3_margin": mean([float(r["trajectory"]["step_margins"].get("step3_correct_vs_competitor", 0.0)) for r in bucket]),
            "top_traj": top_counter(bucket, "trajectory"),
        }))
    return out


def model_block(model: str, result: dict[str, Any]) -> list[str]:
    rows = collect_rows(result)
    hit = [float(r["trajectory"]["hit_rate"]) for r in rows]
    metrics = {
        "dcf_mean": [float(r["readout"]["dcf_mean"]) for r in rows],
        "dcf_delta": [float(r["readout"]["dcf_delta"]) for r in rows],
        "dcf_max_delta": [float(r["readout"]["dcf_max_delta"]) for r in rows],
        "proj_q_over_rms": [float(r["readout"]["proj_q_over_rms"]) for r in rows],
        "cos_v_q": [float(r["readout"]["cos_v_q"]) for r in rows],
        "target_delta": [float(r["readout"]["target_delta"]) for r in rows],
        "competitor_delta": [float(r["readout"]["competitor_delta"]) for r in rows],
        "step1_margin": [float(r["trajectory"]["step_margins"].get("step1_correct_vs_competitor", 0.0)) for r in rows],
        "step2_margin": [float(r["trajectory"]["step_margins"].get("step2_correct_vs_competitor", 0.0)) for r in rows],
        "step3_margin": [float(r["trajectory"]["step_margins"].get("step3_correct_vs_competitor", 0.0)) for r in rows],
    }
    lines = [f"## {model}", ""]
    lines.append(f"cases={len(rows)}, mean_hit={mean(hit):.4f}, top_traj={top_counter(rows, 'trajectory')}")
    lines.append("")
    lines.append("| metric | mean | corr_with_hit |")
    lines.append("|---|---:|---:|")
    for name, vals in metrics.items():
        lines.append(f"| {name} | {mean(vals):.4f} | {corr(vals, hit):.4f} |")
    lines.append("")

    for gkey in ["category", "format", "family", "split"]:
        lines.append(f"### by {gkey}")
        lines.append("")
        lines.append("| group | n | hit | dcf | dcf_delta | proj | s1 | s2 | s3 | top_traj |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for name, stats in group_summary(rows, gkey):
            lines.append(
                f"| {name} | {stats['n']} | {stats['hit']:.4f} | {stats['dcf']:.4f} | "
                f"{stats['dcf_delta']:.4f} | {stats['proj_q_over_rms']:.4f} | "
                f"{stats['step1_margin']:.4f} | {stats['step2_margin']:.4f} | {stats['step3_margin']:.4f} | {stats['top_traj']} |"
            )
        lines.append("")

    tc_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tc_buckets[str(row["readout"]["tc_mode"])].append(row)
    lines.append("### by tc_mode")
    lines.append("")
    lines.append("| mode | n | hit | dcf_delta | competitor_delta | top_traj |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for mode, bucket in sorted(tc_buckets.items()):
        lines.append(
            f"| {mode} | {len(bucket)} | {mean([float(r['trajectory']['hit_rate']) for r in bucket]):.4f} | "
            f"{mean([float(r['readout']['dcf_delta']) for r in bucket]):.4f} | "
            f"{mean([float(r['readout']['competitor_delta']) for r in bucket]):.4f} | {top_counter(bucket, 'trajectory')} |"
        )
    lines.append("")
    return lines


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase159 Cross-model Summary", ""]
    all_rows = []
    for model in MODELS:
        path = ROOT / f"phase159_{model}_gain_readout_trajectory_bridge.json"
        if not path.exists():
            lines.append(f"## {model}")
            lines.append("")
            lines.append(f"Missing: {path}")
            lines.append("")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = collect_rows(result)
        for row in rows:
            row["model"] = model
        all_rows.extend(rows)
        lines.extend(model_block(model, result))

    if all_rows:
        lines.append("## cross_model")
        lines.append("")
        hit = [float(r["trajectory"]["hit_rate"]) for r in all_rows]
        lines.append(f"cases={len(all_rows)}, mean_hit={mean(hit):.4f}")
        lines.append("")
        lines.append("| metric | mean | corr_with_hit |")
        lines.append("|---|---:|---:|")
        for name, vals in {
            "dcf_mean": [float(r["readout"]["dcf_mean"]) for r in all_rows],
            "dcf_delta": [float(r["readout"]["dcf_delta"]) for r in all_rows],
            "proj_q_over_rms": [float(r["readout"]["proj_q_over_rms"]) for r in all_rows],
            "step1_margin": [float(r["trajectory"]["step_margins"].get("step1_correct_vs_competitor", 0.0)) for r in all_rows],
            "step2_margin": [float(r["trajectory"]["step_margins"].get("step2_correct_vs_competitor", 0.0)) for r in all_rows],
            "step3_margin": [float(r["trajectory"]["step_margins"].get("step3_correct_vs_competitor", 0.0)) for r in all_rows],
        }.items():
            lines.append(f"| {name} | {mean(vals):.4f} | {corr(vals, hit):.4f} |")
        lines.append("")

    out_path = ROOT / "phase159_cross_model_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
