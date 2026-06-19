#!/usr/bin/env python3
"""Summary for Phase543 policy gate and scaffold sensitivity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase543_policy_gate_scaffold_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["baseline", "residual_perp", "residual_parallel", "residual_full"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase543_{model}_policy_gate_scaffold_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], source: str, scaffold: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["sources"][source][scaffold][condition]
        for win in d["audit"]
    }
    final_k = str(d["max_new_tokens"])
    win = max(rows, key=lambda w: rows[w]["hit_at_k"][final_k]["target"])
    return win, rows[win]


def gain(base: dict[str, Any], row: dict[str, Any], k: int) -> float:
    key = str(k)
    return float(row["hit_at_k"][key]["target"] - base["hit_at_k"][key]["target"])


def policy_gate(base: dict[str, Any], row: dict[str, Any], k: int) -> float:
    rank_improve = float(base["mean_first_target_rank"] - row["mean_first_target_rank"])
    if rank_improve <= 1e-6:
        return 0.0
    return gain(base, row, k) / rank_improve


def classify(base: dict[str, Any], row: dict[str, Any], k: int) -> str:
    g = gain(base, row, k)
    first_gain = float(row["first_type_rates"]["target"] - base["first_type_rates"]["target"])
    rank_improve = float(base["mean_first_target_rank"] - row["mean_first_target_rank"])
    if g >= 0.25:
        return "gate_open"
    if g >= 0.10:
        return "weak_gate_open"
    if rank_improve > 50 and g < 0.05:
        return "rank_only"
    if first_gain > 0 and g < 0.10:
        return "first_step_only"
    return "no_gate"


def hit_curve(row: dict[str, Any], checkpoints: list[int]) -> str:
    vals = []
    for k in checkpoints:
        key = str(k)
        vals.append(f"k{k}={row['hit_at_k'][key]['target']:.2f}")
    return ", ".join(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase543 Policy Gate and Scaffold Sensitivity Summary", ""]
    compact = []
    for model, d in data.items():
        final_k = int(d["max_new_tokens"])
        checkpoints = [int(x) for x in d["checkpoints"]]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"core={d['core_sources']}, scaffolds={d['scaffolds']}, windows={d['windows']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, max_new_tokens={final_k}, "
            f"checkpoints={checkpoints}, alpha={d['alpha']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| source | scaffold | condition | best win | target hit curve | first target | rankT | gate class |")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for source in d["core_sources"]:
            for scaffold in d["scaffolds"]:
                base_win, base = best_window(d, source, scaffold, "baseline")
                for condition in CONDITIONS:
                    win, row = best_window(d, source, scaffold, condition)
                    cls = "baseline" if condition == "baseline" else classify(base, row, final_k)
                    if condition != "baseline":
                        compact.append({
                            "model": model,
                            "source": source,
                            "scaffold": scaffold,
                            "condition": condition,
                            "win": win,
                            "base_hit": base["hit_at_k"][str(final_k)]["target"],
                            "target_hit": row["hit_at_k"][str(final_k)]["target"],
                            "gain": gain(base, row, final_k),
                            "first_target": row["first_type_rates"]["target"],
                            "rank_improve": base["mean_first_target_rank"] - row["mean_first_target_rank"],
                            "gate": policy_gate(base, row, final_k),
                            "class": cls,
                        })
                    lines.append(
                        f"| {source} | {scaffold} | {condition} | {win} | {hit_curve(row, checkpoints)} | "
                        f"{row['first_type_rates']['target']:.2f} | {float(row['mean_first_target_rank']):.1f} | {cls} |"
                    )
        lines.append("")

    if compact:
        lines.append("## Best Positive Rows")
        lines.append("")
        lines.append("| model | source | scaffold | condition | win | base hit | hit | gain | rank improve | gate ratio | class |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in sorted(compact, key=lambda x: (x["gain"], x["rank_improve"]), reverse=True)[:36]:
            lines.append(
                f"| {r['model']} | {r['source']} | {r['scaffold']} | {r['condition']} | {r['win']} | "
                f"{float(r['base_hit']):.2f} | {float(r['target_hit']):.2f} | {float(r['gain']):+.2f} | "
                f"{float(r['rank_improve']):.1f} | {float(r['gate']):.5f} | {r['class']} |"
            )
        lines.append("")

    out = root / "phase543_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
