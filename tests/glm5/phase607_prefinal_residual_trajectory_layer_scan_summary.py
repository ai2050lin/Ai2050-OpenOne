#!/usr/bin/env python3
"""Summarize Phase 607 pre-final residual trajectory layer scan."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase607_prefinal_residual_trajectory_layer_scan")
MODELS = ["qwen3", "glm4", "deepseek7b"]
COMPONENTS = ["layer_input", "layer_out", "attn_out", "mlp_out"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | L{item['layer']} | {item['component']} | {item['random']} | {item['n']} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
    )


def main() -> None:
    lines = ["# Phase607 Cross-Model Summary", "", "Pre-final residual trajectory layer scan.", ""]
    for model in MODELS:
        path = ROOT / f"phase607_{model}_prefinal_residual_trajectory_layer_scan_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"layers={data['layers_to_scan']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### First Effective")
        lines.append("")
        lines.append("| component | key | layer | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        first = data["summary"].get("first_effective", {})
        for comp in COMPONENTS:
            item = first.get(comp)
            if item:
                lines.append(
                    f"| {comp} | `{item['key']}` | L{item['layer']} | {item['switch']}/{item['n']} | "
                    f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
                )
        lines.append("")
        lines.append("### Best Patches")
        lines.append("")
        lines.append("| key | layer | component | random | n | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:32]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Layer Component Grid")
        lines.append("")
        lines.append("| layer | layer_input | layer_out | attn_out | mlp_out |")
        lines.append("|---:|---:|---:|---:|---:|")
        by_patch = data["summary"]["by_patch"]
        for li in data["layers_to_scan"]:
            vals = []
            for comp in COMPONENTS:
                item = by_patch.get(f"L{li}|{comp}")
                if item:
                    vals.append(f"{item['switch']}/{item['n']} ({fmt(item['mean_margin_gain'])})")
                else:
                    vals.append("")
            lines.append(f"| L{li} | " + " | ".join(vals) + " |")
        lines.append("")
    out = ROOT / "phase607_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
