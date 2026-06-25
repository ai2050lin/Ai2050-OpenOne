#!/usr/bin/env python3
"""Summarize Phase 620 value-token selection cause audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase620_value_token_selection_cause_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.5f}" if abs(x) < 0.1 else f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 620 Cross Model Summary",
        "",
        "Q/K cause audit for correct value-token attention selection.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase620_{model}_value_token_selection_cause_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append("missing")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"rows={data['n_rows']}, target_seen={data['n_target_cases_seen']}, "
            f"raw={data['n_raw_cases']}, filtered={data['filtered']}, "
            f"layers={data['layers_to_scan']}, heads={data['selected_heads']}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")

        lines.append("### causal_patch")
        lines.append("")
        lines.append("| mode | switch | margin | correct_delta | wrong_delta | positive_margin |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best"]:
            lines.append(
                f"| `{item['mode']}` | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} | {item['positive_margin']}/{item['n']} |"
            )
        lines.append("")

        lines.append("### alpha_mass")
        lines.append("")
        lines.append("| group | base | repair | q_only | q_random | repair-base | q-base | random-base |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        alpha = data["summary"]["alpha"]
        for group, item in alpha.items():
            lines.append(
                f"| {group} | {fmt(item['base'])} | {fmt(item['repair'])} | "
                f"{fmt(item['q_only'])} | {fmt(item['q_random_same_norm'])} | "
                f"{fmt(item['repair_minus_base'])} | {fmt(item['q_only_minus_base'])} | "
                f"{fmt(item['q_random_minus_base'])} |"
            )
        lines.append("")
    out = ROOT / "phase620_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
