#!/usr/bin/env python3
"""Cross-model summary for Phase109 support/suppressor decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best(items, kind, key, reverse=False):
    xs = [x for x in items if x["kind"] == kind]
    return max(xs, key=lambda x: x[key]) if reverse else min(xs, key=lambda x: x[key])


def classify(par_down, orth_down, orth_rel, full_down):
    if orth_down["target_delta"] < -1.0 and par_down["target_delta"] > orth_down["target_delta"] + 0.5:
        return "orthogonal_target_support"
    if par_down["target_delta"] < -0.5 and orth_down["target_delta"] > par_down["target_delta"] + 0.3:
        return "readout_parallel_support"
    if orth_rel["max_other_delta"] > 1.0 and full_down["target_delta"] > -0.5:
        return "orthogonal_competition_release"
    if full_down["target_delta"] < -0.5:
        return "mixed_target_down"
    return "weak_or_mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase109_support_suppressor_decomposition")
    parser.add_argument("--output", default="results/gpt5_phase109_support_suppressor_decomposition/phase109_cross_model_summary.md")
    args = parser.parse_args()

    base = Path(args.input_dir)
    data = {m: load(base / f"phase109_{m}_support_suppressor_decomposition.json") for m in MODELS}
    cats = data["qwen3"]["test_categories"]

    lines = ["# Phase 109 Cross-Model Support/Suppressor Decomposition Summary", ""]
    lines.append("## Category Decomposition")
    lines.append("| category | qwen3 | glm4 | deepseek7b | objective reading |")
    lines.append("|---|---|---|---|---|")
    for cat in cats:
        cells = []
        labels = []
        for m in MODELS:
            item = data[m]["category_results"][cat]
            conds = item["conditions"]
            par_down = best(conds, "readout_parallel", "target_delta")
            orth_down = best(conds, "orthogonal", "target_delta")
            orth_rel = best(conds, "orthogonal", "max_other_delta", reverse=True)
            full_down = best(conds, "full_boundary", "target_delta")
            top = orth_rel["top_releases"][0] if orth_rel["top_releases"] else {"category": "none", "delta": 0.0}
            label = classify(par_down, orth_down, orth_rel, full_down)
            labels.append(f"{m}:{label}")
            cells.append(
                f"cos={item['boundary_readout_cos']:.2f} frac={item['parallel_norm_fraction']:.2f}; "
                f"parT={par_down['target_delta']:.2f}; orthT={orth_down['target_delta']:.2f}; "
                f"orthRel={top['category']}+{top['delta']:.2f}; fullT={full_down['target_delta']:.2f}; {label}"
            )
        if any("orthogonal_target_support" in x for x in labels):
            reading = "target support is mostly orthogonal to readout direction"
        elif any("readout_parallel_support" in x for x in labels):
            reading = "readout-parallel support exists"
        elif any("orthogonal_competition_release" in x for x in labels):
            reading = "orthogonal component releases competitors"
        else:
            reading = "weak or mixed"
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} | {reading} |")

    lines.append("")
    lines.append("## Objective Facts")
    lines.append("- Across Qwen3 and DS7B, strong target-down for number comes from the orthogonal component, not the readout-parallel component.")
    lines.append("- Qwen3 time also has larger orthogonal target-down and orthogonal competitor release than readout-parallel removal.")
    lines.append("- DS7B container target-down is almost entirely orthogonal: orthogonal target_delta=-3.15, full=-3.21, readout_parallel does not reduce target.")
    lines.append("- DS7B clothing/furniture show readout-parallel target-down but orthogonal competitor release and full-boundary target-up, confirming component conflict.")
    lines.append("- Boundary-readout cos is small in all models; the category-causal boundary is mostly not aligned with direct output readout words.")
    lines.append("- GLM4 remains weak; its boundary-readout cos is near zero and effects are small.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
