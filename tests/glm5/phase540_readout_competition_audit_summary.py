#!/usr/bin/env python3
"""Summary for Phase540 readout-competition control audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase540_readout_competition_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["residual_perp", "residual_parallel", "residual_full"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase540_{model}_readout_competition_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{float(x):+.3f}"


def best_window(d: dict[str, Any], source: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {win: d["audit"][win]["sources"][source][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w]["margin_delta"])
    return win, rows[win]


def cell(row: dict[str, Any]) -> str:
    return (
        f"margin {fmt(row['margin_delta'])}, "
        f"target {fmt(row['target_max_delta'])}, "
        f"comp {fmt(row['competitor_max_delta'])}, "
        f"supp {fmt(row['competitor_suppression'])}, "
        f"cluster {fmt(row['cluster_other_max_delta'])}, "
        f"off {fmt(row['off_cluster_max_delta'])}, "
        f"shortcut {fmt(row['shortcut_index'])}"
    )


def classify(row: dict[str, Any]) -> str:
    target = float(row["target_max_delta"])
    comp = float(row["competitor_max_delta"])
    cluster = float(row["cluster_other_max_delta"])
    off = float(row["off_cluster_max_delta"])
    supp = float(row["competitor_suppression"])
    if target > 0.5 and supp < 0.15 and target > max(cluster, off) + 0.25:
        return "target_push_shortcut"
    if target > 0.5 and supp > 0.25 and cluster > off:
        return "competition_cluster"
    if cluster > target and cluster > off:
        return "cluster_broadening"
    if off >= max(target, cluster) * 0.75:
        return "global_readout_spill"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase540 Readout Competition Audit Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"core={d['core_sources']}, windows={d['windows']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, alphas={d['alphas']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| source | condition | best win | alpha | token deltas | class |")
        lines.append("|---|---|---|---:|---|---|")
        for source in d["core_sources"]:
            for condition in CONDITIONS:
                win, row = best_window(d, source, condition)
                cls = classify(row)
                if condition == "residual_parallel":
                    compact.append({
                        "model": model,
                        "source": source,
                        "win": win,
                        "margin": row["margin_delta"],
                        "target": row["target_max_delta"],
                        "competitor": row["competitor_max_delta"],
                        "suppression": row["competitor_suppression"],
                        "cluster": row["cluster_other_max_delta"],
                        "off": row["off_cluster_max_delta"],
                        "shortcut": row["shortcut_index"],
                        "class": cls,
                    })
                lines.append(
                    f"| {source} | {condition} | {win} | {float(row['best_alpha']):.1f} | "
                    f"{cell(row)} | {cls} |"
                )
        lines.append("")

    if compact:
        lines.append("## Residual Parallel Compact")
        lines.append("")
        lines.append("| model | source | win | margin | target | competitor | suppression | cluster_other | off_cluster | shortcut | class |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['source']} | {r['win']} | {fmt(r['margin'])} | "
                f"{fmt(r['target'])} | {fmt(r['competitor'])} | {fmt(r['suppression'])} | "
                f"{fmt(r['cluster'])} | {fmt(r['off'])} | {fmt(r['shortcut'])} | {r['class']} |"
            )
        lines.append("")

    out = root / "phase540_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
