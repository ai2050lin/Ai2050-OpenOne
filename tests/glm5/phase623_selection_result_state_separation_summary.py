#!/usr/bin/env python3
"""
Phase 623 summary: Selection State vs Result State Separation
"""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase623_selection_result_state_separation")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def load_results() -> list[dict]:
    rows = []
    for path in sorted(OUT_ROOT.glob("phase623_*_selection_result_state_separation_confirm.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)
    if not rows:
        for path in sorted(OUT_ROOT.glob("phase623_*_selection_result_state_separation_*.json")):
            if "smoke" in path.name:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(data)
    return rows


def best_by_mode(data: dict) -> list[dict]:
    best = data.get("summary", {}).get("best", [])
    return sorted(best, key=lambda x: x["mode"])


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = load_results()
    lines = [
        "# Phase 623 Cross-Model Summary",
        "",
        "Selection state and result state patch combinations.",
        "",
    ]
    if not results:
        lines.append("No confirm results found.")
    for data in results:
        lines.extend(
            [
                f"## {data['model']}",
                "",
                f"- rows: {data['n_rows']} / raw {data['n_raw_cases']}",
                f"- target cases seen: {data['n_target_cases_seen']}",
                f"- patch layers: {data['patch_layers']}",
                f"- selection layers: {data['selection_layers']}",
                "",
                "| mode | switch | margin | correct_delta | wrong_delta | qproj | alpha_cv | alpha_wrong_rel | norm_ratio |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in best_by_mode(data):
            lines.append(
                "| {mode} | {sw}/{n} | {margin} | {cd} | {wd} | {qp} | {acv} | {awr} | {nr} |".format(
                    mode=item["mode"],
                    sw=item["switch"],
                    n=item["n"],
                    margin=fmt(item["mean_margin_gain"]),
                    cd=fmt(item["mean_correct_delta"]),
                    wd=fmt(item["mean_wrong_delta"]),
                    qp=fmt(item["mean_q_delta_projection"]),
                    acv=fmt(item["mean_correct_value_alpha_delta"], 5),
                    awr=fmt(item["mean_wrong_relation_alpha_delta"], 5),
                    nr=fmt(item["mean_piece_norm_ratio"]),
                )
            )
        lines.append("")
        top = data.get("summary", {}).get("best", [])[:6]
        lines.append("Top modes:")
        for item in top:
            lines.append(
                f"- {item['mode']}: switch={item['switch']}/{item['n']}, "
                f"margin={fmt(item['mean_margin_gain'])}, "
                f"qproj={fmt(item['mean_q_delta_projection'])}, "
                f"alpha_cv={fmt(item['mean_correct_value_alpha_delta'], 5)}"
            )
        lines.append("")

    out_path = OUT_ROOT / "phase623_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
