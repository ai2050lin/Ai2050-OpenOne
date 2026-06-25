#!/usr/bin/env python3
"""Summarize Phase 622 residual state direction decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase622_residual_state_direction_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.5f}" if abs(x) < 0.1 else f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 622 Cross Model Summary",
        "",
        "Residual carried-state decomposition into Q-backprojected aligned and orthogonal components.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase622_{model}_residual_state_direction_decomposition_confirm.json"
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
        lines.append("| rank | layer | component | mode | switch | margin | q_proj | q_cos | alpha_cv | alpha_wrong_rel | norm_ratio |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:80], 1):
            lines.append(
                f"| {rank} | {item['layer']} | {item['component']} | {item['mode']} | "
                f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_q_delta_projection'])} | {fmt(item['mean_q_delta_cos'])} | "
                f"{fmt(item['mean_correct_value_alpha_delta'])} | "
                f"{fmt(item['mean_wrong_relation_alpha_delta'])} | {fmt(item['mean_piece_norm_ratio'])} |"
            )
        lines.append("")
    out = ROOT / "phase622_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
