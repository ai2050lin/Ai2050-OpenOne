#!/usr/bin/env python3
"""
Phase 626 summary: Multi-Layer Final Bridge and Token-Position Readout Audit
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase626_multilayer_final_bridge_token_position_audit")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OUT_ROOT.glob("phase626_*_multilayer_final_bridge_token_position_audit_confirm.json"))
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 626 Cross-Model Summary",
        "",
        "Multi-layer final bridge and token-position readout audit.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")

    preferred = [
        "result_only",
        "final_input_all",
        "final_output_all",
        "final_output_token0",
        "final_output_last",
        "final_output_random_all",
        "cumulative_mlp_out",
        "cumulative_attn_out",
        "cumulative_layer_out",
        "cumulative_layer_out_random",
    ]
    for data in results:
        lines.extend(
            [
                f"## {data['model']}",
                "",
                f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
                f"- target cases seen: {data['n_target_cases_seen']}",
                f"- result patch layers: {data['result_patch_layers']}",
                f"- downstream layers: {data['downstream_layers']}",
                f"- tokenization: `{data['tokenization']}`",
                "",
                "### Score Modes",
                "",
                "| mode | switch | margin | correct_delta | wrong_delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        modes = data.get("summary", {}).get("score_modes", {})
        ordered = [m for m in preferred if m in modes] + [m for m in sorted(modes) if m not in preferred]
        for mode in ordered:
            item = modes[mode]
            lines.append(
                f"| {mode} | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | "
                f"{fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} |"
            )
        lines.extend(
            [
                "",
                "### Token Position Deltas Under Result-Only",
                "",
                "| token_pos | correct_delta | wrong_delta | margin_delta |",
                "|---:|---:|---:|---:|",
            ]
        )
        token_pos = data.get("summary", {}).get("token_position", {})
        for key in sorted(token_pos, key=lambda x: int(x.replace("tok", ""))):
            item = token_pos[key]
            lines.append(
                f"| {key} | {fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} | {fmt(item['mean_margin_delta'])} |"
            )
        lines.append("")

    out_path = OUT_ROOT / "phase626_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
