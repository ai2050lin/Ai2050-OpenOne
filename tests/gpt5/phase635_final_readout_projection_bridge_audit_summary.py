#!/usr/bin/env python3
"""Summarize Phase 635 cross-model final readout bridge audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase635_final_readout_projection_bridge_audit")
MODE_ORDER = [
    "base",
    "repair_prompt",
    "semantic_cumulative",
    "source_all6",
    "source_all6_semantic",
    "final_input_repair",
    "final_input_repair_semantic",
    "final_output_repair",
    "final_output_repair_semantic",
    "final_output_source",
    "final_output_source_semantic",
    "readout_delta",
    "readout_delta_semantic",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_top(top0):
    return ", ".join(f"{k}:{v}" for k, v in top0.items())


def row_for_mode(summary, mode):
    for row in summary["by_mode"]:
        if row["mode"] == mode:
            return row
    return None


def main() -> None:
    lines = []
    lines.append("# Phase 635 Cross-Model Summary\n")
    lines.append("目标：审计 source/format state 到 final_norm/lm_head 读出竞争之间的 final readout bridge。\n")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = OUT_ROOT / f"phase635_{model}_final_readout_projection_bridge_audit_confirm.json"
        data = load_model(path)
        if data is None:
            lines.append(f"## {model}\n\nMissing: `{path}`\n")
            continue
        lines.append(f"## {model}\n")
        lines.append(f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}")
        lines.append(f"- readout_scale: {data['readout_scale']}")
        lines.append(f"- downstream_layers: {data['downstream_layers']}")
        lines.append(f"- source_layer_map: {data['source_layer_map']}\n")
        lines.append("| mode | tok0 | exact | wrong_exact | mean_rank | mean_margin | out_proj | out_cos | top0_text |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for mode in MODE_ORDER:
            row = row_for_mode(data["summary"], mode)
            if row is None:
                continue
            n = row["n"]
            lines.append(
                f"| {mode} | {row['tok0_hit']}/{n} | {row['exact']}/{n} | {row['wrong_exact']}/{n} | "
                f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_margin']:.3f} | "
                f"{row['mean_final_output_readout_projection']:.3f} | {row['mean_final_output_readout_cos']:.3f} | "
                f"{fmt_top(row['top0_text'])} |"
            )
        lines.append("\n### Examples\n")
        for item in data["rows"][:10]:
            lines.append(
                f"- sample={item['sample_idx']} mode={item['mode']} tok0={item['tok0_text']!r} "
                f"rank={item['prefix_rank']} exact={item['eval']['exact_correct']} "
                f"margin={item['prefix_margin']:.3f} text={item['generation_text']!r}"
            )
        lines.append("")
    out = OUT_ROOT / "phase635_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
