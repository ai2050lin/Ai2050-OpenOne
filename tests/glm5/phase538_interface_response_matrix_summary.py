#!/usr/bin/env python3
"""Summary for Phase538 category interface response matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase538_interface_response_matrix")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase538_{model}_interface_response_matrix.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{float(x):+.3f}"


def condition_cell(row: dict[str, Any]) -> str:
    return (
        f"{fmt(row['self_min_delta'])}/"
        f"{fmt(row['self_mean_delta'])}/"
        f"{float(row['off_pair_max_abs']):.3f}/"
        f"{float(row['specificity']):.2f}/"
        f"{row['top_off_pair']}"
    )


def best_window_for_source(d: dict[str, Any], source: str, condition: str = "common") -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["sources"][source][condition]
        for win in d["audit"]
    }
    best_win = max(rows, key=lambda w: rows[w]["self_min_delta"])
    return best_win, rows[best_win]


def strict_pass(row: dict[str, Any], rand: dict[str, Any], direct: dict[str, Any], shuffled: dict[str, Any]) -> bool:
    return (
        float(row["self_min_delta"]) > 0.25
        and float(row["specificity"]) > 1.0
        and float(row["self_min_delta"]) > float(rand["max_self_min_delta"])
        and int(rand["pass_like_count"]) == 0
        and float(row["self_min_delta"]) >= float(direct["self_min_delta"])
        and float(row["self_min_delta"]) >= float(shuffled["self_min_delta"])
    )


def leakage_edges(row: dict[str, Any], source: str) -> list[tuple[str, float, float]]:
    edges = []
    matrix = row["matrix_at_best_alpha"]
    for target, cell in matrix.items():
        if target == source:
            continue
        edges.append((target, float(cell["max_abs_delta"]), float(cell["mean_delta"])))
    return sorted(edges, key=lambda x: x[1], reverse=True)


def compact_matrix(row: dict[str, Any], pairs: list[str]) -> list[list[float]]:
    matrix = row["matrix_at_best_alpha"]
    return [[float(matrix[t]["mean_delta"]) for t in pairs]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase538 Interface Response Matrix Summary", ""]
    cross = []
    for model, d in data.items():
        pairs = d["pairs"]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pairs={pairs}, windows={d['windows']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, seeds={len(d['random_seeds'])}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Cell format: self min / self mean / off max abs / specificity / top off-pair.")
        lines.append("")
        lines.append("| source pair | best win | common | direct | shuffled | random self max | random pass-like | strict pass |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for source in pairs:
            win, common = best_window_for_source(d, source, "common")
            source_row = d["audit"][win]["sources"][source]
            direct = source_row["direct"]
            shuffled = source_row["shuffled"]
            rand = source_row["random"]
            passed = strict_pass(common, rand, direct, shuffled)
            cross.append({
                "model": model,
                "source": source,
                "best_win": win,
                "self_min": common["self_min_delta"],
                "specificity": common["specificity"],
                "top_off_pair": common["top_off_pair"],
                "top_off_abs": common["top_off_abs"],
                "strict_pass": passed,
            })
            lines.append(
                f"| {source} | {win} | {condition_cell(common)} | {condition_cell(direct)} | "
                f"{condition_cell(shuffled)} | {fmt(rand['max_self_min_delta'])} | "
                f"{int(rand['pass_like_count'])} | {passed} |"
            )
        lines.append("")

        lines.append("### Common Mean Response Matrices")
        lines.append("")
        lines.append("Each matrix uses the best common alpha/window for that source. Values are mean delta over templates.")
        lines.append("")
        lines.append("| source \\ target | " + " | ".join(pairs) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(pairs)) + "|")
        for source in pairs:
            _win, row = best_window_for_source(d, source, "common")
            matrix = row["matrix_at_best_alpha"]
            vals = [fmt(matrix[target]["mean_delta"]) for target in pairs]
            lines.append(f"| {source} | " + " | ".join(vals) + " |")
        lines.append("")

        lines.append("### Top Leakage Edges")
        lines.append("")
        lines.append("| source | best win | top1 target/max/mean | top2 target/max/mean | top3 target/max/mean |")
        lines.append("|---|---|---:|---:|---:|")
        for source in pairs:
            win, row = best_window_for_source(d, source, "common")
            edges = leakage_edges(row, source)[:3]
            cells = [f"{t}/{a:.3f}/{fmt(m)}" for t, a, m in edges]
            while len(cells) < 3:
                cells.append("")
            lines.append(f"| {source} | {win} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

        vf_ct = []
        for win, audit in d["audit"].items():
            if "vehicle_furniture" in audit["sources"]:
                c = audit["sources"]["vehicle_furniture"]["common"]
                cell = c["matrix_at_best_alpha"]["clothing_tool"]
                vf_ct.append(
                    f"{win}: mean {fmt(cell['mean_delta'])}, max_abs {float(cell['max_abs_delta']):.3f}, "
                    f"source_min {fmt(c['self_min_delta'])}, spec {float(c['specificity']):.2f}"
                )
        lines.append("### Vehicle/Furniture -> Clothing/Tool")
        lines.append("")
        lines.extend(f"- {x}" for x in vf_ct)
        lines.append("")

    if cross:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | source | best win | self min | specificity | top off-pair | top off abs | strict pass |")
        lines.append("|---|---|---|---:|---:|---|---:|---|")
        for row in cross:
            lines.append(
                f"| {row['model']} | {row['source']} | {row['best_win']} | {fmt(row['self_min'])} | "
                f"{float(row['specificity']):.2f} | {row['top_off_pair']} | "
                f"{float(row['top_off_abs']):.3f} | {row['strict_pass']} |"
            )
        lines.append("")

    out = root / "phase538_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
