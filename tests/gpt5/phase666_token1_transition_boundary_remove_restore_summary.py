#!/usr/bin/env python3
"""Summarize Phase 666 cross-model token1 boundary remove/restore results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase666_token1_transition_boundary_remove_restore")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(model: str):
    path = OUT_ROOT / f"phase666_{model}_token1_transition_boundary_remove_restore_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dtext(d: dict) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = [
        "# Phase 666 Cross-Model Summary",
        "",
        "目标：在 Phase 665 找到的最早 token0->token1 边界上，比较 baseline / self_restore / zero_remove / mismatch_restore / correct_restore，审计 token1 转移门是否具有语义特异性。",
        "",
    ]
    for model in MODELS:
        data = load_model(model)
        if data is None:
            lines += [f"## {model}", "", "No confirm result found.", ""]
            continue
        lines += [
            f"## {model}",
            "",
            f"- source_phase665: `{data['source_phase665']}`",
            f"- failures_tested: {data['n_failures_tested']} / rows: {data['n_rows']} / total_time_min: {data['total_time_min']:.2f}",
            f"- boundaries: `{data['boundaries']}`",
            f"- interventions: `{data['interventions']}`",
            "",
            "### Boundary Specificity",
            "",
            "| pair_task | site | combo | boundary | n | correct_delta | mismatch_delta | zero_delta | correct_minus_mismatch | correct_minus_zero |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["boundary_specificity"]:
            inter = r["interventions"]
            correct = inter.get("correct_restore", {})
            mismatch = inter.get("mismatch_restore", {})
            zero = inter.get("zero_remove", {})
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['boundary']} | {r['n']} | "
                f"{correct.get('mean_margin_delta_vs_baseline', 0.0):.3f} | "
                f"{mismatch.get('mean_margin_delta_vs_baseline', 0.0):.3f} | "
                f"{zero.get('mean_margin_delta_vs_baseline', 0.0):.3f} | "
                f"{r['correct_minus_mismatch_margin_delta']:.3f} | "
                f"{r['correct_minus_zero_margin_delta']:.3f} |"
            )
        lines += [
            "",
            "### Intervention Summary",
            "",
            "| pair_task | site | combo | boundary | intervention | n | top1_rate | mean_rank | mean_margin | mean_rank_delta | mean_margin_delta | top1_text |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in data["summary"]["intervention_summary"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['boundary']} | {r['intervention']} | "
                f"{r['n']} | {r['expected_top1_rate']:.3f} | {r['mean_expected_rank']:.2f} | "
                f"{r['mean_expected_minus_top1']:.3f} | {r['mean_rank_delta_vs_baseline']:.2f} | "
                f"{r['mean_margin_delta_vs_baseline']:.3f} | {dtext(r['top1_text'])} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase666_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
