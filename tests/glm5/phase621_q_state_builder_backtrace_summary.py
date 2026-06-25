#!/usr/bin/env python3
"""Summarize Phase 621 Q state builder backtrace."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase621_q_state_builder_backtrace")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.5f}" if abs(x) < 0.1 else f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 621 Cross Model Summary",
        "",
        "Backtrace upstream residual components that regenerate Q state, value-token attention, and candidate switch.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase621_{model}_q_state_builder_backtrace_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append("missing")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"rows={data['n_rows']}, target_seen={data['n_target_cases_seen']}, raw={data['n_raw_cases']}, "
            f"patch_layers={data['patch_layers']}, selection_layers={data['selection_layers']}, "
            f"heads={data['selected_heads']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("| rank | layer | component | random | switch | margin | q_proj | q_cos | q_norm | alpha_cv | alpha_rule | alpha_wrong_rel |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:60], 1):
            lines.append(
                f"| {rank} | {item['layer']} | {item['component']} | {item['random']} | "
                f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_q_delta_projection'])} | {fmt(item['mean_q_delta_cos'])} | "
                f"{fmt(item['mean_q_delta_norm_ratio'])} | {fmt(item['mean_correct_value_alpha_delta'])} | "
                f"{fmt(item['mean_correct_rule_alpha_delta'])} | {fmt(item['mean_wrong_relation_alpha_delta'])} |"
            )
        lines.append("")
    out = ROOT / "phase621_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
