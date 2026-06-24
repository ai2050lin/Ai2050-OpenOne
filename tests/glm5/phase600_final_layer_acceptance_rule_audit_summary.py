#!/usr/bin/env python3
"""Summarize Phase 600 final-layer acceptance rule audit results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase600_final_layer_acceptance_rule_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCHED_DS7B = [
    "rule_value|L26",
    "prompt_last|L26",
    "query_relation|L19",
]
WATCHED_TRAJ = [
    "natural_correct",
    "natural_wrong",
    "artificial_repair",
    "artificial_random",
    "artificial_wrong",
]
WATCHED_COMPONENTS = [
    "layer_input",
    "attn_out",
    "mlp_input",
    "mlp_out",
    "layer_out",
    "final_norm_output",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def comp_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['trajectory']} | `{item['component']}` | {item['n']} | "
        f"{fmt(item['mean_projection_specific_margin'])} | "
        f"{fmt(item['mean_effect_norm'])} | "
        f"{fmt(item['mean_cos_to_natural_correct'])} | "
        f"{fmt(item['mean_norm_ratio_to_natural_correct'])} | "
        f"{fmt(item['positive_projection_rate'])} | "
        f"{fmt(item['mean_attn_l1_to_base'])} |"
    )


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['trajectory']} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_generated_down_projection'])} | {fmt(item['mean_full_margin_gain'])} |"
    )


def main() -> None:
    lines = [
        "# Phase600 Cross-Model Summary",
        "",
        "Final-layer acceptance rule audit: natural correct/wrong trajectories vs artificial repair/random/wrong trajectories.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase600_{model}_final_layer_acceptance_rule_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, alpha={data['alpha']}, capture_attn={data['capture_attn']}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Artificial Final Effects")
        lines.append("")
        lines.append("| key | trajectory | n | switch | generated_down_projection | full_margin_gain |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["final_effects"][:14]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Best Projection Components")
        lines.append("")
        lines.append("| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_projection"][:28]:
            lines.append(comp_line(item))
        lines.append("")
        lines.append("### Best Natural Alignment Components")
        lines.append("")
        lines.append("| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["natural_alignment"][:28]:
            lines.append(comp_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_comp = data["summary"]["by_component"]
            lines.append("### DS7B watched acceptance path")
            lines.append("")
            lines.append("| key | trajectory | component | n | projection_margin | effect_norm | cos_to_natural_correct | norm_ratio_to_natural | positive_rate | attn_l1_to_base |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for base in WATCHED_DS7B:
                for traj in WATCHED_TRAJ:
                    tname = traj
                    if traj.startswith("artificial_"):
                        tname = traj
                    for comp in WATCHED_COMPONENTS:
                        key = f"{base}|{tname}|{comp}"
                        if key in by_comp:
                            lines.append(comp_line(by_comp[key]))
            lines.append("")
    out = ROOT / "phase600_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
