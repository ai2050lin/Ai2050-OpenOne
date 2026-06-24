#!/usr/bin/env python3
"""Summarize Phase 604 final norm / readout acceptance audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase604_final_norm_readout_acceptance_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCH = [
    "input_interp|beta1",
    "output_interp|beta1",
    "input_interp|beta2",
    "output_interp|beta2",
    "seq_input_interp|beta1",
    "seq_output_interp|beta1",
    "seq_input_interp|beta2",
    "seq_output_interp|beta2",
    "input_random|beta1",
    "output_random|beta1",
    "seq_input_random|beta1",
    "seq_output_random|beta1",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def interp_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {fmt(item['beta'])} | {item['n']} | "
        f"{item['first_switch']}/{item['n']} | {fmt(item['mean_first_margin_gain'])} | "
        f"{item['full_switch']}/{item['n']} | {fmt(item['mean_full_margin_gain'])} | "
        f"{fmt(item['mean_correct_full_delta'])} | {fmt(item['mean_wrong_full_delta'])} |"
    )


def local_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['position']} | {item['component']} | {item['n']} | "
        f"{fmt(item['mean_projection_margin'])} | {fmt(item['mean_effect_norm'])} | "
        f"{fmt(item['mean_base_norm'])} | {fmt(item['mean_repair_norm'])} | "
        f"{fmt(item['mean_cos_base_repair'])} |"
    )


def main() -> None:
    lines = ["# Phase604 Cross-Model Summary", "", "Final norm and readout acceptance audit.", ""]
    for model in MODELS:
        path = ROOT / f"phase604_{model}_final_norm_readout_acceptance_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, betas={data['betas']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Interpolations")
        lines.append("")
        lines.append("| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_interpolation"][:24]:
            lines.append(interp_line(item))
        lines.append("")
        lines.append("### Watched Interpolations")
        lines.append("")
        lines.append("| key | kind | beta | n | first_switch | first_margin_gain | full_switch | full_margin_gain | correct_full_delta | old_wrong_full_delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        by_i = data["summary"]["by_interpolation"]
        for key in WATCH:
            if key in by_i:
                lines.append(interp_line(by_i[key]))
        lines.append("")
        lines.append("### Best Local Readout Deltas")
        lines.append("")
        lines.append("| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_local"][:20]:
            lines.append(local_line(item))
        lines.append("")
        if model == "deepseek7b":
            lines.append("### DS7B prompt_last local readout")
            lines.append("")
            lines.append("| key | position | component | n | projection_margin | effect_norm | base_norm | repair_norm | cos_base_repair |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
            by_l = data["summary"]["by_local"]
            for key in ["prompt_last|final_norm_input", "prompt_last|final_norm_output"]:
                if key in by_l:
                    lines.append(local_line(by_l[key]))
            lines.append("")
    out = ROOT / "phase604_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
