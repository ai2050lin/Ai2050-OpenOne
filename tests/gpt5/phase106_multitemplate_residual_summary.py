#!/usr/bin/env python3
"""Cross-model summary for Phase106 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]
POSITIONS = ["answer_last", "object_last"]
BASES = ["raw", "template_residual"]


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_global(d: dict[str, Any], pos: str, basis: str) -> dict[str, Any]:
    g = d["global"][pos][basis]
    return {
        "top1": max(g, key=lambda x: x["top1_count"]),
        "margin": max(g, key=lambda x: x["mean_margin"]),
        "boundary": max(g, key=lambda x: x["mean_boundary_norm"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase106_multitemplate_residual")
    parser.add_argument("--output", default="results/gpt5_phase106_multitemplate_residual/phase106_cross_model_summary.md")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    data = {m: load(input_dir / f"phase106_{m}_multitemplate_residual.json") for m in MODELS}
    cats = data["qwen3"]["categories"]

    lines: list[str] = []
    lines.append("# Phase 106 Cross-Model Multi-template Residual Summary")
    lines.append("")
    lines.append("## Global Objective Results")
    lines.append("| model | position | basis | top1 layer/count | best mean margin | best boundary |")
    lines.append("|---|---|---|---:|---:|---:|")
    for m in MODELS:
        d = data[m]
        for pos in POSITIONS:
            for basis in BASES:
                b = best_global(d, pos, basis)
                lines.append(
                    f"| {m} | {pos} | {basis} | L{b['top1']['layer']} {b['top1']['top1_count']}/32 | "
                    f"L{b['margin']['layer']} {b['margin']['mean_margin']:.3f} | "
                    f"L{b['boundary']['layer']} {b['boundary']['mean_boundary_norm']:.3f} |"
                )

    lines.append("")
    lines.append("## Answer Slot: Raw vs Template Residual Category Margins")
    lines.append("| category | qwen3 raw/resid | glm4 raw/resid | deepseek7b raw/resid | objective reading |")
    lines.append("|---|---:|---:|---:|---|")
    for cat in cats:
        cells = []
        readings = []
        for m in MODELS:
            raw = data[m]["category_summary"]["answer_last"]["raw"][cat]
            res = data[m]["category_summary"]["answer_last"]["template_residual"][cat]
            cells.append(
                f"{raw['best_margin']:.2f}/{res['best_margin']:.2f} "
                f"L{raw['best_margin_layer']}->L{res['best_margin_layer']}"
            )
            if res["best_margin"] - raw["best_margin"] > 3:
                readings.append(f"{m}+resid")
        if readings:
            reading = "residual improves: " + ",".join(readings)
        elif all(data[m]["category_summary"]["answer_last"]["template_residual"][cat]["best_margin"] < 1 for m in MODELS):
            reading = "still weak across models"
        else:
            reading = "mostly stable or model-specific"
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} | {reading} |")

    lines.append("")
    lines.append("## Object Position Survival")
    lines.append("| category | qwen3 object resid | glm4 object resid | deepseek7b object resid |")
    lines.append("|---|---:|---:|---:|")
    for cat in cats:
        vals = []
        for m in MODELS:
            x = data[m]["category_summary"]["object_last"]["template_residual"][cat]
            vals.append(f"{x['best_margin']:.2f} L{x['best_margin_layer']} rank{x['best_rank']}")
        lines.append(f"| {cat} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines.append("")
    lines.append("## Direct Corrections To Phase105")
    lines.append("- Phase105 single-template Qwen3 conclusion is mostly retained at answer_last, but several weak categories become strong after template residual subtraction: clothing, furniture, body, place, action, time, number, container, communication, property.")
    lines.append("- Qwen3 object_last has much weaker margins than answer_last, but many categories survive after template residual subtraction; category information exists before the answer slot, yet is amplified at the answer slot.")
    lines.append("- GLM4 remains near-zero margin after template residual subtraction at both positions; this points to readout-token/model-format calibration, not just template common-vector contamination.")
    lines.append("- DS7B answer_last changes strongly after template residual subtraction: best mean margin rises to 4.723 at L27, supporting that Phase105 understated DS7B because common template/format components masked category directions.")
    lines.append("- Boundary norm peaks are unchanged by subtracting same-template mean because the same vector is subtracted from every category in a template; boundary-layer conclusions remain stable.")
    lines.append("- Top1 counts after residual subtraction can be inflated when margins are tiny; margin magnitude is the stricter objective signal.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
