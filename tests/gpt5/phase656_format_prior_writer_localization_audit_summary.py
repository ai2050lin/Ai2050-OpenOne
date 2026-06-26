#!/usr/bin/env python3
"""Summarize Phase 656 format-prior writer localization audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase656_format_prior_writer_localization_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_line(r) -> str:
    group_delta = r.get("mean_group_margin_delta", {})
    gd = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(group_delta.items()))
    return (
        f"| {r['pair_task']} | {r['site']} | {r['baseline_top0_category']} | "
        f"{r['layer']} | {r['component']} | {r['n']} | "
        f"{r['mean_top_margin_delta']:.3f} | {r['mean_format_margin_delta']:.3f} | "
        f"{r['mean_rank_improvement']:.2f} | {r['flipped_to_correct']} | "
        f"{r['baseline_top0']} | {r['ablated_top0']} | {gd} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 656 Cross-Model Summary\n")
    lines.append(
        "目标：固定 Phase 653/654 的 intent-gate restore patch，"
        "在最终读出位置消融 attn_out / mlp_out，定位 space/newline/explanation 等格式先验写入候选。\n"
    )
    header = (
        "| pair_task | site | baseline_top0_category | layer | component | n | "
        "mean_top_margin_delta | mean_format_margin_delta | mean_rank_improvement | "
        "flipped_to_correct | baseline_top0 | ablated_top0 | group_margin_delta |"
    )
    sep = "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|"
    for model in MODELS:
        path = OUT_ROOT / f"phase656_{model}_format_prior_writer_localization_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / selected_items: {data['n_selected_items']} / "
            f"mode_rows: {data['n_mode_rows']} / time: {data.get('total_time_min', 0):.2f} min"
        )
        lines.append(f"- scan_layers: `{data['scan_layers']}` / components: `{data['scan_components']}`")
        lines.append(f"- selection: `{data['selection_stats']}` / filtered: `{data['filtered']}`\n")

        lines.append("### Format-Prior Writer Candidates\n")
        lines.append(header)
        lines.append(sep)
        for r in data["summary"]["format_prior_writer_candidates"][:60]:
            lines.append(row_line(r))
        lines.append("")

        lines.append("### Value-Support Writer Candidates / Ablation Hurts Correct Prefix\n")
        lines.append(header)
        lines.append(sep)
        for r in data["summary"]["value_support_writer_candidates"][:40]:
            lines.append(row_line(r))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase656_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
