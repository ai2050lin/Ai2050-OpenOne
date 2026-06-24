#!/usr/bin/env python3
"""Summarize Phase 606 digit1 upstream source decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase606_digit1_upstream_source_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCH = [
    "layer_input",
    "attn_out",
    "mlp_out",
    "final_norm_input",
    "final_norm_output",
    "layer_input_random",
    "attn_out_random",
    "mlp_out_random",
    "final_norm_input_random",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['component']} | {item['random']} | {item['n']} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
        f"{fmt(item['positive_margin_rate'])} |"
    )


def source_line(item: dict) -> str:
    return f"| `{item['source']}` | {item['n']} | {fmt(item['mean_delta'])} |"


def main() -> None:
    lines = ["# Phase606 Cross-Model Summary", "", "Digit1 upstream source decomposition.", ""]
    for model in MODELS:
        path = ROOT / f"phase606_{model}_digit1_upstream_source_decomposition_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Component Patches")
        lines.append("")
        lines.append("| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_patch"][:20]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Watched Component Patches")
        lines.append("")
        lines.append("| key | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta | positive_margin_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        by_patch = data["summary"]["by_patch"]
        for key in WATCH:
            if key in by_patch:
                lines.append(patch_line(by_patch[key]))
        lines.append("")
        lines.append("### Attention Source Mass Delta")
        lines.append("")
        lines.append("| source | n | repair_minus_base_mass |")
        lines.append("|---|---:|---:|")
        for item in data["summary"]["source_best"]:
            lines.append(source_line(item))
        lines.append("")
    out = ROOT / "phase606_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
