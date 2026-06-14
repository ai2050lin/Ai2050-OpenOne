#!/usr/bin/env python3
"""Cross-model summary for Phase107 causal boundary removal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify(rem: dict[str, Any], ctl: dict[str, Any]) -> str:
    rt = rem["target_delta"]
    ct = ctl["target_delta"]
    release = rem["max_other_delta"]
    crelease = ctl["max_other_delta"]
    if rt < -0.5 and release > crelease + 0.2:
        return "target_down_competitor_release"
    if rt < -0.5:
        return "target_down_only"
    if release > crelease + 0.5:
        return "competitor_release_only"
    if rt > 0.5:
        return "target_up_opposed"
    return "weak_or_control_like"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase107_causal_boundary_removal")
    parser.add_argument("--output", default="results/gpt5_phase107_causal_boundary_removal/phase107_cross_model_summary.md")
    args = parser.parse_args()
    base = Path(args.input_dir)
    data = {m: load(base / f"phase107_{m}_causal_boundary_removal.json") for m in MODELS}

    cats = data["qwen3"]["test_categories"]
    lines: list[str] = []
    lines.append("# Phase 107 Cross-Model Causal Boundary Removal Summary")
    lines.append("")
    lines.append("## Global Setup")
    lines.append("| model | boundary layer | train/category | test/category | prompts/category |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in MODELS:
        d = data[m]
        first = next(iter(d["category_results"].values()))
        lines.append(
            f"| {m} | L{d['boundary_layer']} | {d['train_objects_per_category']} | "
            f"{d['test_objects_per_category']} | {first['n_prompts']} |"
        )

    lines.append("")
    lines.append("## Category Effects")
    lines.append("| category | qwen3 | glm4 | deepseek7b | objective reading |")
    lines.append("|---|---|---|---|---|")
    for cat in cats:
        cells = []
        classes = []
        for m in MODELS:
            item = data[m]["category_results"][cat]
            rem = item["remove_boundary"]
            ctl = item["random_same_norm"]
            top = rem["top_releases"][0] if rem["top_releases"] else {"category": "none", "delta": 0.0}
            cls = classify(rem, ctl)
            classes.append(f"{m}:{cls}")
            cells.append(
                f"TΔ={rem['target_delta']:.2f} ctl={ctl['target_delta']:.2f}; "
                f"rel={top['category']}+{top['delta']:.2f}; {cls}"
            )
        if any("target_down_competitor_release" in c for c in classes):
            reading = "causal-like target decrease with release in at least one model"
        elif any("target_down_only" in c for c in classes):
            reading = "target decrease without clean release"
        elif any("competitor_release_only" in c for c in classes):
            reading = "release without target decrease"
        elif any("target_up_opposed" in c for c in classes):
            reading = "boundary direction may be opposed/suppressive"
        else:
            reading = "weak or control-like"
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} | {reading} |")

    lines.append("")
    lines.append("## Objective Facts")
    lines.append("- Qwen3 L35 boundary removal produced clear target decreases for time (-0.51) and number (-1.41), while most concrete categories showed target increases or release-only behavior.")
    lines.append("- Qwen3 clothing/furniture/body/container showed competitor releases beyond random control, but target DCF increased rather than decreased; these are not clean 'remove category boundary suppresses target' effects.")
    lines.append("- GLM4 had to be rerun with PROBE_TORCH_DTYPE=bfloat16; fp16 logits produced NaN. In bf16, effects were small but finite.")
    lines.append("- DS7B L27 boundary removal produced strong target decreases for number (-2.58) and container (-2.28), while several concrete categories showed target-up/opposed effects.")
    lines.append("- Random same-norm controls were usually much smaller than boundary removal for release magnitude, so many release effects are direction-specific even when target decrease is absent.")
    lines.append("- This phase confirms that atlas boundary vectors can affect final logits in real forward passes, but the sign is category/model-specific; a boundary vector is not always a simple positive support direction for its category.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
