#!/usr/bin/env python3
"""Summarize Phase 596 cross-model MLP internal gate path audit results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase596_mlp_internal_gate_path_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_specific_margin_gain'])} | "
        f"{fmt(item['mean_common_delta'])} | {fmt(item['mean_correct_specific'])} | "
        f"{fmt(item['mean_old_top_wrong_specific'])} |"
    )


def proj_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['n']} | {fmt(item['mean_projection_specific_margin'])} | "
        f"{fmt(item['mean_projection_correct_specific'])} | "
        f"{fmt(item['mean_projection_old_top_wrong_specific'])} | "
        f"{fmt(item['positive_projection_rate'])} |"
    )


def main() -> None:
    lines = [
        "# Phase596 Cross-Model Summary",
        "",
        "MLP internal gate/up/z/down path audit.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase596_{model}_mlp_internal_gate_path_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"alpha={data['alpha']}, topks={data['topks']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best causal patches")
        lines.append("")
        lines.append("| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_patches"][:14]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Best internal projections")
        lines.append("")
        lines.append("| key | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_projections"][:12]:
            lines.append(proj_line(item))
        lines.append("")

        if model == "deepseek7b":
            lines.append("### DS7B watched keys")
            lines.append("")
            watched_patches = [
                "rule_value|L26|gate_raw",
                "rule_value|L26|up_raw",
                "rule_value|L26|gate_up_pair_raw",
                "rule_value|L26|z_pair_raw",
                "rule_value|L26|z_pair_top32",
                "rule_value|L26|z_pair_top128",
                "rule_value|L26|wrong_z_pair_raw",
                "query_relation|L19|gate_up_pair_raw",
                "query_relation|L19|z_pair_raw",
                "rule_relation|L18|gate_up_pair_raw",
                "rule_relation|L18|z_pair_raw",
            ]
            by_patch = data["summary"]["patch_by_key"]
            lines.append("| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for key in watched_patches:
                if key in by_patch:
                    lines.append(patch_line(by_patch[key]))
            lines.append("")
    out = ROOT / "phase596_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
