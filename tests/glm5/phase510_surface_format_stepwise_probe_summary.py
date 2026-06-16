#!/usr/bin/env python3
"""Summary for Phase510 surface-format stepwise probe."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/glm5_phase510_surface_format_stepwise_probe")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["remove_support", "add_support", "remove_release", "add_release"]


def avg(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def load(model: str) -> dict[str, Any] | None:
    path = ROOT / f"phase510_{model}_surface_format_stepwise_probe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def step_metric(trace: dict[str, Any], step: int, key: str) -> float:
    return float(trace["step_metrics"][step - 1].get(key, 0.0))


def delta_vs_clean(cat_data: dict[str, Any], cond: str, step: int, key: str) -> float:
    return step_metric(cat_data["traces"][cond], step, key) - step_metric(cat_data["traces"]["clean"], step, key)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase510 Surface-format Stepwise Probe Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"L={d['L']}, categories={','.join(d['categories'])}, train={d['train_objects']}, "
            f"test={d['test_objects']}, templates={len(d['templates'])}, steps={d['steps']}"
        )
        lines.append("")
        lines.append("| category | support axis | release axis | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        model_rows = []
        for cat, cd in d["category_results"].items():
            support = cd["chosen_axes"]["support"]
            release = cd["chosen_axes"]["release"]
            support_axis = f"L{support['layer']} {support['name']} {support['delta_D']:+.3f}"
            release_axis = f"L{release['layer']} {release['name']} {release['delta_D']:+.3f}"
            clean_hit = float(cd["traces"]["clean"]["category_hit_rate"])
            for cond in CONDITIONS:
                tr = cd["traces"][cond]
                hit_delta = float(tr["category_hit_rate"]) - clean_hit
                s1 = delta_vs_clean(cd, cond, 1, "category_vs_competitor")
                s2 = delta_vs_clean(cd, cond, 2, "category_vs_competitor")
                s3 = delta_vs_clean(cd, cond, 3, "category_vs_competitor")
                p1 = delta_vs_clean(cd, cond, 1, "category_vs_punctuation")
                g1 = delta_vs_clean(cd, cond, 1, "category_vs_generic")
                top1 = delta_vs_clean(cd, cond, 1, "category_top1_rate")
                model_rows.append({
                    "condition": cond,
                    "hit_delta": hit_delta,
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "p1": p1,
                    "g1": g1,
                    "top1": top1,
                })
                lines.append(
                    f"| {cat} | {support_axis} | {release_axis} | {cond} | "
                    f"{hit_delta:+.3f} | {s1:+.3f} | {s2:+.3f} | {s3:+.3f} | "
                    f"{p1:+.3f} | {g1:+.3f} | {top1:+.3f} |"
                )
        lines.append("")
        for cond in CONDITIONS:
            rows = [r for r in model_rows if r["condition"] == cond]
            compact.append({
                "model": model,
                "condition": cond,
                "hit_delta": avg([r["hit_delta"] for r in rows]),
                "s1": avg([r["s1"] for r in rows]),
                "s2": avg([r["s2"] for r in rows]),
                "s3": avg([r["s3"] for r in rows]),
                "p1": avg([r["p1"] for r in rows]),
                "g1": avg([r["g1"] for r in rows]),
                "top1": avg([r["top1"] for r in rows]),
            })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | condition | hit Δ | s1 cat-comp Δ | s2 cat-comp Δ | s3 cat-comp Δ | s1 cat-punct Δ | s1 cat-generic Δ | s1 top category Δ |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['condition']} | {r['hit_delta']:+.4f} | "
                f"{r['s1']:+.4f} | {r['s2']:+.4f} | {r['s3']:+.4f} | "
                f"{r['p1']:+.4f} | {r['g1']:+.4f} | {r['top1']:+.4f} |"
            )
        lines.append("")

    out = ROOT / "phase510_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
