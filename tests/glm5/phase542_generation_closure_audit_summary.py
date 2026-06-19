#!/usr/bin/env python3
"""Summary for Phase542 generation closure audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase542_generation_closure_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["baseline", "residual_perp", "residual_parallel", "residual_full"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase542_{model}_generation_closure_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], source: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {win: d["audit"][win]["sources"][source][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w]["hit_rates"]["target"])
    return win, rows[win]


def cell(row: dict[str, Any]) -> str:
    h = row["hit_rates"]
    f = row["first_type_rates"]
    return (
        f"hit target {h['target']:.2f}, comp {h['competitor']:.2f}, cluster {h['cluster_other']:.2f}, "
        f"off {h['off_cluster']:.2f}, first target {f['target']:.2f}, first comp {f['competitor']:.2f}, "
        f"rankT {float(row['mean_first_target_rank']):.1f}"
    )


def closure_gain(base: dict[str, Any], row: dict[str, Any]) -> float:
    return float(row["hit_rates"]["target"] - base["hit_rates"]["target"])


def classify(base: dict[str, Any], row: dict[str, Any]) -> str:
    gain = closure_gain(base, row)
    comp = float(row["hit_rates"]["competitor"])
    first_target = float(row["first_type_rates"]["target"])
    if gain >= 0.20 and comp <= base["hit_rates"]["competitor"] + 0.05:
        return "generation_closure_positive"
    if first_target > base["first_type_rates"]["target"] and gain < 0.10:
        return "first_step_only"
    if comp > base["hit_rates"]["competitor"] + 0.10:
        return "competitor_leak"
    return "no_closure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase542 Generation Closure Audit Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"core={d['core_sources']}, windows={d['windows']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, max_new_tokens={d['max_new_tokens']}, alpha={d['alpha']}, "
            f"attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| source | condition | best win | generation metrics | class |")
        lines.append("|---|---|---|---|---|")
        for source in d["core_sources"]:
            base_win, base = best_window(d, source, "baseline")
            for condition in CONDITIONS:
                win, row = best_window(d, source, condition)
                cls = "baseline" if condition == "baseline" else classify(base, row)
                if condition != "baseline":
                    compact.append({
                        "model": model,
                        "source": source,
                        "condition": condition,
                        "win": win,
                        "target_hit": row["hit_rates"]["target"],
                        "base_hit": base["hit_rates"]["target"],
                        "gain": closure_gain(base, row),
                        "competitor_hit": row["hit_rates"]["competitor"],
                        "first_target": row["first_type_rates"]["target"],
                        "class": cls,
                    })
                lines.append(f"| {source} | {condition} | {win} | {cell(row)} | {cls} |")
        lines.append("")

    if compact:
        lines.append("## Intervention Compact")
        lines.append("")
        lines.append("| model | source | condition | win | base target hit | target hit | gain | competitor hit | first target | class |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['source']} | {r['condition']} | {r['win']} | "
                f"{float(r['base_hit']):.2f} | {float(r['target_hit']):.2f} | {float(r['gain']):+.2f} | "
                f"{float(r['competitor_hit']):.2f} | {float(r['first_target']):.2f} | {r['class']} |"
            )
        lines.append("")

    out = root / "phase542_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
