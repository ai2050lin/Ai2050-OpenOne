#!/usr/bin/env python3
"""Summarize Phase 603 post-attention MLP compensation audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase603_post_attention_mlp_compensation_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCHED = ["rule_value|L26", "prompt_last|L26", "query_relation|L19"]
MODES = [
    "mlp_repair_only",
    "mlp_plus_attn_effect",
    "mlp_plus_attn_random",
    "mlp_random_plus_attn_effect",
]
PARTS = ["mlp_input", "gate", "up", "z", "down", "mlp_out", "layer_out", "final_norm_output"]
PATCH_MODES = [
    "mlp_repair_only",
    "mlp_plus_attn_effect",
    "mlpout_effect_only",
    "mlp_plus_mlpout_effect",
    "mlp_plus_attn_plus_mlpout_effect",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def diag_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['mode']} | `{item['part']}` | {item['n']} | "
        f"{fmt(item['mean_cos_to_natural'])} | {fmt(item['mean_norm_ratio'])} | "
        f"{fmt(item['mean_projection_specific_margin'])} |"
    )


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['mode']} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_full_margin_gain'])} | {fmt(item['positive_margin_rate'])} |"
    )


def main() -> None:
    lines = ["# Phase603 Cross-Model Summary", "", "Post-attention MLP compensation audit.", ""]
    for model in MODELS:
        path = ROOT / f"phase603_{model}_post_attention_mlp_compensation_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, alpha={data['alpha']}, attn_scale={data['attn_scale']}, "
            f"mlpout_scale={data['mlpout_scale']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Diagnostics")
        lines.append("")
        lines.append("| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best_diag"][:30]:
            lines.append(diag_line(item))
        lines.append("")
        lines.append("### Patch Effects")
        lines.append("")
        lines.append("| key | mode | n | switch | full_margin_gain | positive_margin_rate |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best_patch"][:24]:
            lines.append(patch_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_diag = data["summary"]["by_diag"]
            by_patch = data["summary"]["by_patch"]
            lines.append("### DS7B watched diagnostics")
            lines.append("")
            lines.append("| key | mode | part | n | cos_to_natural | norm_ratio | projection_margin |")
            lines.append("|---|---|---|---:|---:|---:|---:|")
            for base in WATCHED:
                for mode in MODES:
                    for part in PARTS:
                        key = f"{base}|{mode}|{part}"
                        if key in by_diag:
                            lines.append(diag_line(by_diag[key]))
            lines.append("")
            lines.append("### DS7B watched patch effects")
            lines.append("")
            lines.append("| key | mode | n | switch | full_margin_gain | positive_margin_rate |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for base in WATCHED:
                for mode in PATCH_MODES:
                    key = f"{base}|{mode}"
                    if key in by_patch:
                        lines.append(patch_line(by_patch[key]))
            lines.append("")
    out = ROOT / "phase603_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
