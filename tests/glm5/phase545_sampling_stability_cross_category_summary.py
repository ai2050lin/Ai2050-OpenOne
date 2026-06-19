#!/usr/bin/env python3
"""Summary for Phase545 sampling stability and cross-category closure audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase545_sampling_stability_cross_category")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase545_{model}_sampling_stability_cross_category.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], pair: str, scaffold: str, mode: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["sources"][pair][scaffold][mode][condition]
        for win in d["audit"]
    }
    final_k = str(d["max_new_tokens"])
    win = max(rows, key=lambda w: rows[w]["hit_at_k_mean"][final_k]["family_target"])
    return win, rows[win]


def fam(row: dict[str, Any], k: int) -> float:
    return float(row["hit_at_k_mean"][str(k)]["family_target"])


def fam_std(row: dict[str, Any], k: int) -> float:
    return float(row["hit_at_k_std"][str(k)]["family_target"])


def exact(row: dict[str, Any], k: int) -> float:
    return float(row["hit_at_k_mean"][str(k)]["exact_target"])


def comp(row: dict[str, Any], k: int) -> float:
    return float(row["hit_at_k_mean"][str(k)]["family_competitor"])


def classify(base: dict[str, Any], row: dict[str, Any], k: int) -> str:
    gain = fam(row, k) - fam(base, k)
    std = fam_std(row, k)
    competitor = comp(row, k)
    if gain >= 0.20 and gain / (std + 1e-6) >= 1.5 and competitor <= comp(base, k) + 0.20:
        return "stable_positive"
    if gain >= 0.20:
        return "positive_high_var"
    if gain >= 0.10:
        return "weak_positive"
    if gain <= -0.10:
        return "negative"
    return "flat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase545 Sampling Stability and Cross-Category Summary", ""]
    compact = []
    for model, d in data.items():
        final_k = int(d["max_new_tokens"])
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pairs={d['pairs']}, scaffolds={d['scaffolds']}, modes={d['decode_modes']}, "
            f"conditions={d['conditions']}, windows={d['windows']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, alpha={d['alpha']}"
        )
        lines.append("")
        lines.append("| pair | scaffold | mode | condition | win | base family | family mean | std | gain | exact | comp family | stability | class |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for pair in d["pairs"]:
            for scaffold in d["scaffolds"]:
                for mode in d["decode_modes"]:
                    base_win, base = best_window(d, pair, scaffold, mode, "baseline")
                    for condition in d["conditions"]:
                        win, row = best_window(d, pair, scaffold, mode, condition)
                        gain = 0.0 if condition == "baseline" else fam(row, final_k) - fam(base, final_k)
                        stability = 0.0 if condition == "baseline" else gain / (fam_std(row, final_k) + 1e-6)
                        cls = "baseline" if condition == "baseline" else classify(base, row, final_k)
                        if condition != "baseline":
                            compact.append({
                                "model": model,
                                "pair": pair,
                                "scaffold": scaffold,
                                "mode": mode,
                                "condition": condition,
                                "win": win,
                                "base": fam(base, final_k),
                                "family": fam(row, final_k),
                                "std": fam_std(row, final_k),
                                "gain": gain,
                                "exact": exact(row, final_k),
                                "comp": comp(row, final_k),
                                "stability": stability,
                                "class": cls,
                            })
                        lines.append(
                            f"| {pair} | {scaffold} | {mode} | {condition} | {win} | "
                            f"{fam(base, final_k):.2f} | {fam(row, final_k):.2f} | {fam_std(row, final_k):.2f} | "
                            f"{gain:+.2f} | {exact(row, final_k):.2f} | {comp(row, final_k):.2f} | "
                            f"{stability:.2f} | {cls} |"
                        )
        lines.append("")

    if compact:
        lines.append("## Best Stable Positive Rows")
        lines.append("")
        lines.append("| model | pair | scaffold | mode | condition | win | base | family | std | gain | stability | exact | comp | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        rows = sorted(compact, key=lambda x: (x["class"] == "stable_positive", x["gain"], x["stability"]), reverse=True)
        for r in rows[:60]:
            lines.append(
                f"| {r['model']} | {r['pair']} | {r['scaffold']} | {r['mode']} | {r['condition']} | {r['win']} | "
                f"{r['base']:.2f} | {r['family']:.2f} | {r['std']:.2f} | {r['gain']:+.2f} | "
                f"{r['stability']:.2f} | {r['exact']:.2f} | {r['comp']:.2f} | {r['class']} |"
            )
        lines.append("")

        lines.append("## Pair Max Gain")
        lines.append("")
        lines.append("| model | pair | max gain | row |")
        lines.append("|---|---|---:|---|")
        for model in sorted(set(r["model"] for r in compact)):
            for pair in sorted(set(r["pair"] for r in compact if r["model"] == model)):
                rows = [r for r in compact if r["model"] == model and r["pair"] == pair]
                best = max(rows, key=lambda x: x["gain"])
                desc = f"{best['scaffold']} {best['mode']} {best['condition']} {best['win']} std={best['std']:.2f} cls={best['class']}"
                lines.append(f"| {model} | {pair} | {best['gain']:+.2f} | {desc} |")
        lines.append("")

    out = root / "phase545_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
