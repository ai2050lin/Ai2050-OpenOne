#!/usr/bin/env python3
"""Summary for Phase509 rotation-stable orthogonal field factor audit."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/glm5_phase509_rotation_stable_orthogonal_field")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def avg(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def load(model: str) -> dict[str, Any] | None:
    path = ROOT / f"phase509_{model}_rotation_stable_orthogonal_field.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[0]


def fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row.get('name', row.get('basis_index', ''))} {float(row.get('delta_D', 0.0)):+.3f} {row.get('label', '')}"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    all_data = {m: load(m) for m in MODELS}
    all_data = {m: d for m, d in all_data.items() if d is not None}
    lines: list[str] = ["# Phase509 Rotation-stable Orthogonal Field Factor Audit Summary", ""]
    compact = []

    for model, d in all_data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"L={d['L']}, d={d['d_model']}, categories={','.join(d['categories'])}, "
            f"train={d['train_objects']}, test={d['test_objects']}, templates={len(d['templates'])}, "
            f"rank={d['rank']}, candidate_random_axes={d['candidate_random_axes']}"
        )
        lines.append("")
        lines.append("| category | layer | ratio | svd best | rotated best | causal best | causal positive | outside best | causal format | causal surface cat | causal surface punct |")
        lines.append("|---|---:|---:|---|---|---|---|---|---:|---:|---:|")

        svd_vals, rot_vals, causal_vals, pos_vals, outside_vals = [], [], [], [], []
        same_sign = 0
        n_layers = 0
        surface_cat_vals, surface_punc_vals = [], []
        for cat, cd in d["category_results"].items():
            for layer, item in cd["layers"].items():
                svd_best = best(item["svd_components"], "delta_D")
                rot_best = best(item["rotated_components"], "delta_D")
                causal_best = best(item["causal_candidates"], "delta_D")
                causal_pos = best(item["causal_candidates"], "delta_D", reverse=True)
                outside_best = best(item["outside_random_components"], "delta_D")
                ratio = float(item["perp_para_ratio"])
                sv = float(svd_best["delta_D"])
                rv = float(rot_best["delta_D"])
                cv = float(causal_best["delta_D"])
                pv = float(causal_pos["delta_D"])
                ov = float(outside_best["delta_D"])
                svd_vals.append(sv)
                rot_vals.append(rv)
                causal_vals.append(cv)
                pos_vals.append(pv)
                outside_vals.append(ov)
                if sv < -0.25 and rv < -0.25:
                    same_sign += 1
                n_layers += 1
                surface_cat_vals.append(float(causal_best.get("surface_delta_category", 0.0)))
                surface_punc_vals.append(float(causal_best.get("surface_delta_punctuation", 0.0)))
                lines.append(
                    f"| {cat} | L{layer} | {ratio:.2f} | {fmt(svd_best)} | {fmt(rot_best)} | "
                    f"{fmt(causal_best)} | {fmt(causal_pos)} | {fmt(outside_best)} | "
                    f"{float(causal_best.get('format_abs_cos', 0.0)):.3f} | "
                    f"{float(causal_best.get('surface_delta_category', 0.0)):+.3f} | "
                    f"{float(causal_best.get('surface_delta_punctuation', 0.0)):+.3f} |"
                )
        lines.append("")
        row = {
            "model": model,
            "mean_svd_best": avg(svd_vals),
            "mean_rotated_best": avg(rot_vals),
            "mean_causal_best": avg(causal_vals),
            "mean_causal_positive": avg(pos_vals),
            "mean_outside_best": avg(outside_vals),
            "support_rotation_match_rate": same_sign / n_layers if n_layers else 0.0,
            "mean_causal_surface_category_delta": avg(surface_cat_vals),
            "mean_causal_surface_punctuation_delta": avg(surface_punc_vals),
        }
        compact.append(row)
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        for k, v in row.items():
            if k != "model":
                lines.append(f"| {k} | {float(v):.4f} |")
        lines.append("")

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | svd best | rotated best | causal best | causal positive | outside best | rotation support match | surface category | surface punctuation |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['mean_svd_best']:.4f} | {r['mean_rotated_best']:.4f} | "
                f"{r['mean_causal_best']:.4f} | {r['mean_causal_positive']:.4f} | "
                f"{r['mean_outside_best']:.4f} | {r['support_rotation_match_rate']:.4f} | "
                f"{r['mean_causal_surface_category_delta']:.4f} | "
                f"{r['mean_causal_surface_punctuation_delta']:.4f} |"
            )
        lines.append("")

    out = ROOT / "phase509_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
