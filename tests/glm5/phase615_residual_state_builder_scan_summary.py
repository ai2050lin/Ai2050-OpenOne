#!/usr/bin/env python3
"""Summarize Phase 615 residual state builder scan."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase615_residual_state_builder_scan")
MODELS = ["qwen3", "glm4", "deepseek7b"]
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 615 Cross Model Summary",
        "",
        "Residual-state builder layer/component scan on source-aligned target rows.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase615_{model}_residual_state_builder_scan_confirm.json"
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
            f"layers={data['layers_to_scan']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### best")
        lines.append("")
        lines.append("| rank | layer | component | random | switch | margin | correct_delta | wrong_delta | pos_margin |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:24], 1):
            lines.append(
                f"| {rank} | L{item['layer']} | `{item['component']}` | {item['random']} | "
                f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
                f"{item['positive_margin']}/{item['n']} |"
            )
        lines.append("")
        lines.append("### by_layer_real")
        lines.append("")
        header = "| layer | " + " | ".join(COMPONENTS) + " |"
        lines.append(header)
        lines.append("|---:|" + "|".join(["---:" for _ in COMPONENTS]) + "|")
        by = data["summary"]["by_patch"]
        for li in data["layers_to_scan"]:
            vals = []
            for comp in COMPONENTS:
                item = by.get(f"L{li}|{comp}|real")
                if item:
                    vals.append(f"{item['switch']}/{item['n']} {fmt(item['mean_margin_gain'])}")
                else:
                    vals.append("missing")
            lines.append(f"| L{li} | " + " | ".join(vals) + " |")
        lines.append("")
    out = ROOT / "phase615_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
