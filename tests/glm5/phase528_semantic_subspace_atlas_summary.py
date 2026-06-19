#!/usr/bin/env python3
"""Cross-model summary for Phase528 semantic subspace atlas."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/glm5_phase528_semantic_subspace_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]
VARS = ["category", "color", "object"]


def avg(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def load(model: str) -> dict[str, Any] | None:
    path = ROOT / f"phase528_{model}_semantic_subspace_atlas.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase528 Semantic Subspace Atlas Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, alpha={d['alpha']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("### Direction Cosine Matrix")
        lines.append("")
        lines.append("| dir | category | color | object |")
        lines.append("|---|---:|---:|---:|")
        off_diag = []
        for a in VARS:
            vals = []
            for b in VARS:
                v = float(d["cos_matrix"][a][b])
                vals.append(v)
                if a < b:
                    off_diag.append(abs(v))
            lines.append(f"| {a} | " + " | ".join(f"{x:+.4f}" for x in vals) + " |")
        lines.append("")

        lines.append("### Readout Alignment")
        lines.append("")
        lines.append("| dir | norm | readout norm % | semantic norm % | cos to readout |")
        lines.append("|---|---:|---:|---:|---:|")
        for var in VARS:
            r = d["readout_alignment"][var]
            lines.append(
                f"| {var} | {float(r['norm']):.4f} | {float(r['readout_norm_pct']):.2f} | "
                f"{float(r['semantic_norm_pct']):.2f} | {float(r['cos_to_readout']):+.5f} |"
            )
        lines.append("")

        lines.append("### Selectivity Matrix: Δmargin")
        lines.append("")
        lines.append("| direction -> task | category | color | object |")
        lines.append("|---|---:|---:|---:|")
        own_deltas, off_deltas = [], []
        for var in VARS:
            vals = []
            for task in VARS:
                v = float(d["selectivity"][var][task]["delta_margin"])
                vals.append(v)
                if var == task:
                    own_deltas.append(v)
                else:
                    off_deltas.append(abs(v))
            lines.append(f"| {var} | " + " | ".join(f"{x:+.4f}" for x in vals) + " |")
        lines.append("")

        lines.append("### Positive Control")
        lines.append("")
        lines.append("| dir | own Δmargin | own Δtop1 | max other abs Δmargin | selectivity ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        ratios = []
        for var in VARS:
            pc = d["positive_control"][var]
            own = float(pc["own_task_delta_margin"])
            other = float(pc["max_other_abs_delta_margin"])
            ratio = abs(own) / (other + 1e-8)
            ratios.append(ratio)
            lines.append(
                f"| {var} | {own:+.4f} | {float(pc['own_task_delta_top1']):+.4f} | "
                f"{other:.4f} | {ratio:.4f} |"
            )
        lines.append("")

        row = {
            "model": model,
            "mean_abs_offdiag_cos": avg(off_diag),
            "mean_own_delta": avg(own_deltas),
            "mean_off_abs_delta": avg(off_deltas),
            "mean_selectivity_ratio": avg(ratios),
            "color_positive_delta": float(d["positive_control"]["color"]["own_task_delta_margin"]),
            "category_readout_pct": float(d["readout_alignment"]["category"]["readout_norm_pct"]),
            "color_readout_pct": float(d["readout_alignment"]["color"]["readout_norm_pct"]),
        }
        compact.append(row)

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | mean abs offdiag cos | mean own Δ | mean off abs Δ | selectivity ratio | color positive Δ | category readout % | color readout % |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['mean_abs_offdiag_cos']:.4f} | {r['mean_own_delta']:+.4f} | "
                f"{r['mean_off_abs_delta']:.4f} | {r['mean_selectivity_ratio']:.4f} | "
                f"{r['color_positive_delta']:+.4f} | {r['category_readout_pct']:.2f} | {r['color_readout_pct']:.2f} |"
            )
        lines.append("")

    out = ROOT / "phase528_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
