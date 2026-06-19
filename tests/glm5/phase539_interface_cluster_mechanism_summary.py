#!/usr/bin/env python3
"""Summary for Phase539 interface cluster mechanism decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase539_interface_cluster_mechanism")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["residual_full", "residual_perp", "residual_parallel", "attention_perp", "mlp_perp"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase539_{model}_interface_cluster_mechanism.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{float(x):+.3f}"


def cell(row: dict[str, Any]) -> str:
    return (
        f"{fmt(row['self_min_delta'])}/"
        f"{fmt(row['self_mean_delta'])}/"
        f"{float(row['off_pair_max_abs']):.3f}/"
        f"{float(row['specificity']):.2f}/"
        f"{row['top_off_pair']}"
    )


def best_condition(d: dict[str, Any], source: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {win: d["audit"][win]["sources"][source][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w]["self_min_delta"])
    return win, rows[win]


def strongest_condition(d: dict[str, Any], source: str) -> tuple[str, str, dict[str, Any]]:
    rows = []
    for cond in CONDITIONS:
        win, row = best_condition(d, source, cond)
        rows.append((cond, win, row))
    return max(rows, key=lambda x: x[2]["self_min_delta"])


def key_edges(row: dict[str, Any]) -> dict[str, tuple[float, float]]:
    m = row["matrix_at_best_alpha"]
    out = {}
    for k in ["vehicle_furniture", "vehicle_tool", "vehicle_clothing", "clothing_tool", "fruit_tool", "animal_tool"]:
        if k in m:
            out[k] = (float(m[k]["mean_delta"]), float(m[k]["max_abs_delta"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase539 Interface Cluster Mechanism Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"core={d['core_sources']}, targets={len(d['pairs'])}, windows={d['windows']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, alphas={d['alphas']}, "
            f"seeds={len(d['random_seeds'])}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Cell format: self min / self mean / off max abs / specificity / top off-pair.")
        lines.append("")
        lines.append("| source | condition | best win | response |")
        lines.append("|---|---|---|---:|")
        for source in d["core_sources"]:
            for cond in CONDITIONS:
                win, row = best_condition(d, source, cond)
                lines.append(f"| {source} | {cond} | {win} | {cell(row)} |")
            cond, win, row = strongest_condition(d, source)
            compact.append({
                "model": model,
                "source": source,
                "condition": cond,
                "win": win,
                "self_min": row["self_min_delta"],
                "off": row["off_pair_max_abs"],
                "specificity": row["specificity"],
                "top_off": row["top_off_pair"],
            })
        lines.append("")

        lines.append("### Core Cosines")
        lines.append("")
        for win, audit in d["audit"].items():
            lines.append(f"#### {win}")
            lines.append("")
            lines.append("Common-perp cosine:")
            pairs = d["core_sources"]
            lines.append("| pair | " + " | ".join(pairs) + " |")
            lines.append("|---|" + "|".join(["---:"] * len(pairs)) + "|")
            for p in pairs:
                vals = [fmt(audit["cosine"]["common_perp_cos"][p][q]) for q in pairs]
                lines.append(f"| {p} | " + " | ".join(vals) + " |")
            lines.append("")
            lines.append("Readout cosine:")
            lines.append("| pair | " + " | ".join(pairs) + " |")
            lines.append("|---|" + "|".join(["---:"] * len(pairs)) + "|")
            for p in pairs:
                vals = [fmt(audit["cosine"]["readout_cos"][p][q]) for q in pairs]
                lines.append(f"| {p} | " + " | ".join(vals) + " |")
            lines.append("")

        lines.append("### Key Edge Snapshots")
        lines.append("")
        lines.append("| source | strongest condition | win | vehicle_furniture | vehicle_tool | vehicle_clothing | clothing_tool | fruit_tool | animal_tool |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for source in d["core_sources"]:
            cond, win, row = strongest_condition(d, source)
            edges = key_edges(row)
            def edge_text(k: str) -> str:
                if k not in edges:
                    return ""
                mean, mx = edges[k]
                return f"{fmt(mean)}/{mx:.3f}"
            lines.append(
                f"| {source} | {cond} | {win} | {edge_text('vehicle_furniture')} | "
                f"{edge_text('vehicle_tool')} | {edge_text('vehicle_clothing')} | "
                f"{edge_text('clothing_tool')} | {edge_text('fruit_tool')} | {edge_text('animal_tool')} |"
            )
        lines.append("")

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | source | strongest condition | win | self min | off max | specificity | top off |")
        lines.append("|---|---|---|---|---:|---:|---:|---|")
        for row in compact:
            lines.append(
                f"| {row['model']} | {row['source']} | {row['condition']} | {row['win']} | "
                f"{fmt(row['self_min'])} | {float(row['off']):.3f} | "
                f"{float(row['specificity']):.2f} | {row['top_off']} |"
            )
        lines.append("")

    out = root / "phase539_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
