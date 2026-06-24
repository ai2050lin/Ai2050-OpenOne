#!/usr/bin/env python3
"""Summarize Phase 613 Q/K routing factor split."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase613_qk_routing_factor_split")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODES = ["o_actual", "q_only", "k_only", "qk", "v_only", "qv", "kv", "qkv", "random_o_actual_norm"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def emit_table(lines, data, summary_key):
    layer = data["layers_to_scan"][0] if data["layers_to_scan"] else None
    lines.append("| mode | switch | margin | correct_delta | wrong_delta | pos_margin | heads |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    by_patch = data.get(summary_key, {}).get("by_patch", {})
    for mode in MODES:
        key = f"L{layer}|top{data['top_k']}|{mode}"
        item = by_patch.get(key)
        if not item:
            lines.append(f"| `{mode}` | missing | | | | | |")
            continue
        lines.append(
            f"| `{mode}` | {item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
            f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
            f"{item['positive_margin']}/{item['n']} | `{item['heads']}` |"
        )


def main() -> None:
    lines = [
        "# Phase 613 Cross Model Summary",
        "",
        "Projection-level Q/K/V patch inside natural forward. Source-aligned prompt pairs.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase613_{model}_qk_routing_factor_split_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append("missing")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"rows={data['n_rows']}, target_seen={data['n_target_cases_seen']}, "
            f"raw={data['n_raw_cases']}, filtered={data.get('filtered')}, "
            f"layers={data['layers_to_scan']}, top_k={data['top_k']}, "
            f"top_heads={data.get('top_heads')}, kv={data.get('kv_heads_by_layer')}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### all_rows")
        lines.append("")
        emit_table(lines, data, "summary")
        lines.append("")
        lines.append("### target_rows")
        lines.append("")
        emit_table(lines, data, "target_summary")
        lines.append("")
    out = ROOT / "phase613_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
