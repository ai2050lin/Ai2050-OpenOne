#!/usr/bin/env python3
"""Cross-model summary for Phase508 orthogonal field basis decomposition."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/glm5_phase508_orthogonal_field_basis")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def avg(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def load(model: str) -> dict[str, Any] | None:
    path = ROOT / f"phase508_{model}_orthogonal_field_basis.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[0]


def fmt_row(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    idx = row.get("basis_index", row.get("basis_indices", ""))
    label = row.get("label", row.get("set_name", ""))
    return f"{idx} {float(row.get('delta_D', 0.0)):+.3f} {label}"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase508 Orthogonal Field Basis Decomposition Summary", ""]

    all_model_stats = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"L={d['L']}, d={d['d_model']}, train={d['train_objects']}, "
            f"test={d['test_objects']}, templates={len(d['templates'])}, rank={d['rank']}, scale={d['scale']}"
        )
        lines.append("")
        lines.append("| category | layer | ratio | best component | strongest positive | best random | support_top4 | suppressor_top2 | format_top2 |")
        lines.append("|---|---:|---:|---|---|---|---|---|---|")

        model_ratios, best_effects, pos_effects, rand_effects = [], [], [], []
        support_labels = 0
        positive_labels = 0
        total_components = 0
        for cat, cat_data in d["category_results"].items():
            for layer, item in cat_data["layers"].items():
                comps = item["components"]
                rands = item["random_components"]
                sets = item["sets"]
                b = best(comps, "delta_D")
                p = best(comps, "delta_D", reverse=True)
                rb = best(rands, "delta_D")
                support_top4 = next((x for x in sets if x["set_name"] == "support_top4"), None)
                suppressor_top2 = next((x for x in sets if x["set_name"] == "suppressor_top2"), None)
                format_top2 = next((x for x in sets if x["set_name"] == "format_aligned_top2"), None)
                ratio = float(item["perp_para_ratio"])
                model_ratios.append(ratio)
                if b:
                    best_effects.append(float(b["delta_D"]))
                if p:
                    pos_effects.append(float(p["delta_D"]))
                if rb:
                    rand_effects.append(float(rb["delta_D"]))
                total_components += len(comps)
                support_labels += sum(1 for x in comps if x.get("label") == "support")
                positive_labels += sum(1 for x in comps if x.get("label") in {
                    "competitor_suppressor", "target_release", "suppressor_or_interface"
                })
                lines.append(
                    f"| {cat} | L{layer} | {ratio:.2f} | {fmt_row(b)} | {fmt_row(p)} | "
                    f"{fmt_row(rb)} | {fmt_row(support_top4)} | {fmt_row(suppressor_top2)} | {fmt_row(format_top2)} |"
                )
        lines.append("")
        model_summary = {
            "model": model,
            "mean_ratio": avg(model_ratios),
            "mean_best_delta": avg(best_effects),
            "mean_pos_delta": avg(pos_effects),
            "mean_random_best": avg(rand_effects),
            "support_label_rate": support_labels / total_components if total_components else 0.0,
            "positive_label_rate": positive_labels / total_components if total_components else 0.0,
        }
        all_model_stats.append(model_summary)
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        for k, v in model_summary.items():
            if k == "model":
                continue
            lines.append(f"| {k} | {float(v):.4f} |")
        lines.append("")

    if all_model_stats:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | mean ratio | mean best ΔD | mean strongest positive ΔD | mean random best ΔD | support label rate | positive label rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in all_model_stats:
            lines.append(
                f"| {row['model']} | {row['mean_ratio']:.4f} | {row['mean_best_delta']:.4f} | "
                f"{row['mean_pos_delta']:.4f} | {row['mean_random_best']:.4f} | "
                f"{row['support_label_rate']:.4f} | {row['positive_label_rate']:.4f} |"
            )
        lines.append("")

    out = ROOT / "phase508_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
