#!/usr/bin/env python3
"""Summarize Phase 105 atlas outputs across models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def r2(x: Any) -> str:
    if isinstance(x, (float, int)):
        return f"{float(x):.2f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase105_global_category_atlas")
    parser.add_argument("--output", default="results/gpt5_phase105_global_category_atlas/phase105_cross_model_summary.md")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    data = {m: load(input_dir / f"phase105_{m}_atlas.json") for m in MODELS}
    cats = data["qwen3"]["categories"]

    lines: list[str] = []
    lines.append("# Phase 105 Cross-Model Global Category Atlas Summary")
    lines.append("")
    lines.append("## Global Layer Distribution")
    lines.append("| model | layers | best top1 layer | top1 count | best mean margin layer | best mean boundary layer |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in MODELS:
        d = data[m]
        top1 = max(d["layer_global"], key=lambda x: x["top1_count"])
        margin = max(d["layer_global"], key=lambda x: x["mean_target_margin"])
        boundary = max(d["layer_global"], key=lambda x: x["mean_boundary_norm"])
        lines.append(
            f"| {m} | {d['n_layers']} | L{top1['layer']} | {top1['top1_count']}/{len(cats)} | "
            f"L{margin['layer']} ({margin['mean_target_margin']:.2f}) | "
            f"L{boundary['layer']} ({boundary['mean_boundary_norm']:.2f}) |"
        )

    lines.append("")
    lines.append("## Category Relative Map")
    lines.append("| category | qwen3 | glm4 | deepseek7b | stable reading |")
    lines.append("|---|---|---|---|---|")
    for cat in cats:
        cells = []
        stable_layers = []
        type_classes = []
        for m in MODELS:
            x = data[m]["category_summary"][cat]
            cells.append(
                f"M{x['best_layer_by_margin']}/B{x['best_layer_by_boundary_norm']} "
                f"margin={r2(x['best_margin'])} rank={x['best_rank']} {x['type_class']}"
            )
            stable_layers.append(x["best_layer_by_margin"])
            type_classes.append(x["type_class"])
        if len(set(type_classes)) == 1:
            stable = f"same class: {type_classes[0]}"
        elif data["qwen3"]["category_summary"][cat]["best_margin"] > 4:
            stable = "Qwen3 readable; cross-model weak/variant"
        elif all(data[m]["category_summary"][cat]["best_rank"] == 1 for m in MODELS):
            stable = "rank-stable but margin weak"
        else:
            stable = "model-specific or diffuse"
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} | {stable} |")

    lines.append("")
    lines.append("## Qwen3 Strong Readout Types")
    strong = []
    for cat, x in data["qwen3"]["category_summary"].items():
        if x["best_margin"] > 8:
            neigh = ", ".join(n["category"] for n in x["nearest_neighbors_at_best_margin_layer"][:3])
            strong.append((x["best_margin"], cat, x, neigh))
    for _margin, cat, x, neigh in sorted(strong, reverse=True):
        lines.append(
            f"- {cat}: margin={x['best_margin']:.2f}, marginL=L{x['best_layer_by_margin']}, "
            f"boundaryL=L{x['best_layer_by_boundary_norm']}, neighbors={neigh}"
        )

    lines.append("")
    lines.append("## Diffuse Or Weak Readout Types")
    for m in MODELS:
        weak = [
            cat for cat, x in data[m]["category_summary"].items()
            if x["best_margin"] < 1 and x["best_rank"] > 1
        ]
        lines.append(f"- {m}: {', '.join(weak) if weak else 'none by this criterion'}")

    lines.append("")
    lines.append("## Main Interpretation")
    lines.append("- Qwen3 shows the clearest late-layer category readout: concrete object classes and sensory/event classes often peak at L32-L36.")
    lines.append("- GLM4 shows very small DCF margins in this readout basis; rank can be correct while amplitude remains weak, so GLM4 needs a better calibrated readout or stronger templates.")
    lines.append("- DS7B shows strong late boundary norms and cohesion but weak category-label margins; its internal category structure is present but not cleanly decoded by this DCF word basis.")
    lines.append("- Across models, boundary norm usually peaks very late, while margin peaks can be category-specific; this supports a layer-development view rather than a single universal category layer.")
    lines.append("- Local boundary removal is only logit-lens evidence; any high-value release edge must be followed by downstream causal patching before being treated as mechanism.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output}")


if __name__ == "__main__":
    main()
