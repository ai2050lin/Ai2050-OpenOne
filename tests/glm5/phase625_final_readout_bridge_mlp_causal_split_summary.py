#!/usr/bin/env python3
"""
Phase 625 summary: Final Readout Bridge and MLP Causal Split
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase625_final_readout_bridge_mlp_causal_split")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(OUT_ROOT.glob("phase625_*_final_readout_bridge_mlp_causal_split_confirm.json"))
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 625 Cross-Model Summary",
        "",
        "Final readout bridge and MLP causal split.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")

    preferred = [
        "result_only",
        "selection_both_plus_result",
        "result_random_norm",
        "mlp_full_delta",
        "mlp_correct_up",
        "mlp_wrong_down",
        "mlp_correct_plus_wrong",
        "mlp_margin_span",
        "mlp_orthogonal",
        "mlp_random_same_norm",
    ]
    for data in results:
        lines.extend(
            [
                f"## {data['model']}",
                "",
                f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
                f"- target cases seen: {data['n_target_cases_seen']}",
                f"- patch layers: {data['patch_layers']}",
                f"- MLP split layer: L{data['mlp_layer']}",
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
                "### Final Bridge",
                "",
                "| mode | input_proj | input_cos | output_proj | output_cos | output_margin_proxy | correct_proxy | wrong_proxy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        bridge = data.get("summary", {}).get("final_bridge", {})
        for mode in sorted(bridge):
            item = bridge[mode]
            lines.append(
                f"| {mode} | "
                f"{fmt(item.get('mean_input_repair_projection', 0.0))} | "
                f"{fmt(item.get('mean_input_repair_cos', 0.0))} | "
                f"{fmt(item.get('mean_output_repair_projection', 0.0))} | "
                f"{fmt(item.get('mean_output_repair_cos', 0.0))} | "
                f"{fmt(item.get('mean_output_projection_margin', 0.0))} | "
                f"{fmt(item.get('mean_output_projection_correct_specific', 0.0))} | "
                f"{fmt(item.get('mean_output_projection_wrong_specific', 0.0))} |"
            )
        lines.append("")

    out_path = OUT_ROOT / "phase625_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
