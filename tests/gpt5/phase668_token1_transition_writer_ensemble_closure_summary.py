#!/usr/bin/env python3
"""Summarize Phase 668 cross-model writer ensemble closure results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase668_token1_transition_writer_ensemble_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(model: str):
    path = OUT_ROOT / f"phase668_{model}_token1_transition_writer_ensemble_closure_confirm.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lines = [
        "# Phase 668 Cross-Model Summary",
        "",
        "目标：比较 full boundary state、top-head ensemble、component ensemble 对 token1 transition 的闭合能力。",
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
            f"- ensembles: `{data['ensembles']}`",
            "",
            "### Ensemble Specificity",
            "",
            "| pair_task | site | combo | ensemble | kind | n | correct_top1 | mismatch_top1 | correct_minus_mismatch | correct_minus_zero |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["ensemble_specificity"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['ensemble']} | {r['kind']} | "
                f"{r['n']} | {r['correct_top1_rate']:.3f} | {r['mismatch_top1_rate']:.3f} | "
                f"{r['correct_minus_mismatch_margin_delta']:.3f} | {r['correct_minus_zero_margin_delta']:.3f} |"
            )
        lines += [
            "",
            "### Intervention Summary",
            "",
            "| pair_task | site | combo | ensemble | intervention | n | top1_rate | rank_delta | margin_delta |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
        for r in data["summary"]["intervention_summary"]:
            lines.append(
                f"| {r['pair_task']} | {r['site']} | {r['combo_name']} | {r['ensemble']} | {r['intervention']} | "
                f"{r['n']} | {r['expected_top1_rate']:.3f} | {r['mean_rank_delta_vs_baseline']:.2f} | "
                f"{r['mean_margin_delta_vs_baseline']:.3f} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase668_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
