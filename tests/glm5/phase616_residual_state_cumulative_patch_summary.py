#!/usr/bin/env python3
"""Summarize Phase 616 residual-state cumulative patch results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase616_residual_state_cumulative_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]
KEY_PATTERNS = [
    "replace_",
    "add_layer_out_",
    "add_attn_all",
    "add_mlp_all",
    "add_attn_mlp_all",
    "add_attn_early",
    "add_attn_late",
    "add_mlp_midlate",
    "add_attn_mlp_midlate",
]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def keep_name(name: str) -> bool:
    return any(name.startswith(p) for p in KEY_PATTERNS)


def main() -> None:
    lines = [
        "# Phase 616 Cross Model Summary",
        "",
        "Residual-state cumulative additive patch with single-replace references.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase616_{model}_residual_state_cumulative_patch_confirm.json"
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
            f"layers={data['layers_to_scan']}, specs={data['n_specs']}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### best")
        lines.append("")
        lines.append("| rank | name | mode | random | ops | switch | margin | correct_delta | wrong_delta | pos_margin |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:28], 1):
            lines.append(
                f"| {rank} | `{item['name']}` | {item['mode']} | {item['random']} | "
                f"{item['n_ops']} | {item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
                f"{item['positive_margin']}/{item['n']} |"
            )
        lines.append("")
        lines.append("### key_real")
        lines.append("")
        lines.append("| name | mode | ops | switch | margin | correct_delta | wrong_delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        items = [
            item for item in data["summary"]["by_patch"].values()
            if (not item["random"]) and keep_name(item["name"])
        ]
        items = sorted(items, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
        for item in items[:40]:
            lines.append(
                f"| `{item['name']}` | {item['mode']} | {item['n_ops']} | "
                f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
            )
        lines.append("")
        lines.append("### random_controls")
        lines.append("")
        lines.append("| name | mode | ops | switch | margin |")
        lines.append("|---|---|---:|---:|---:|")
        random_items = [
            item for item in data["summary"]["by_patch"].values()
            if item["random"] and keep_name(item["name"])
        ]
        random_items = sorted(random_items, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
        for item in random_items[:20]:
            lines.append(
                f"| `{item['name']}` | {item['mode']} | {item['n_ops']} | "
                f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} |"
            )
        lines.append("")
    out = ROOT / "phase616_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
