#!/usr/bin/env python3
"""Summary for Phase536 pair quality and selectivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase536_pair_quality_selectivity")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase536_{model}_pair_quality_selectivity.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tr_cell(tr: dict[str, Any]) -> str:
    return f"{float(tr['best_transfer_min']):+.3f}/{float(tr['best_transfer_mean']):+.3f}/{float(tr['pair_specificity']):.2f}"


def best_condition(d: dict[str, Any], pair: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][pair][win]["conditions"][condition]["transfer"]
        for win in d["audit"][pair]
    }
    win = max(rows, key=lambda w: rows[w]["best_transfer_min"])
    return win, rows[win]


def pair_verdict(d: dict[str, Any], pair: str) -> str:
    _win, common = best_condition(d, pair, "common")
    _dwin, direct = best_condition(d, pair, "direct")
    _swin, shuffled = best_condition(d, pair, "shuffled")
    rand_max = max(
        float(d["audit"][pair][win]["conditions"]["random"]["max_transfer_min"])
        for win in d["audit"][pair]
    )
    baseline = d["pair_baseline"][pair]
    moderate = -1.0 <= float(baseline["mean_margin"]) <= 4.0 and 5.0 <= float(baseline["mean_rank"]) <= 1000.0
    if (
        moderate
        and float(common["best_transfer_min"]) > 0.25
        and float(common["pair_specificity"]) > 1.0
        and float(common["best_transfer_min"]) > rand_max
        and float(common["best_transfer_min"]) >= float(direct["best_transfer_min"])
        and float(common["best_transfer_min"]) >= float(shuffled["best_transfer_min"])
    ):
        return "candidate_common_pair"
    if float(common["best_transfer_min"]) > 0.25 and float(common["best_transfer_min"]) > rand_max:
        return "strong_but_not_specific"
    if not moderate:
        return "baseline_not_ideal"
    return "weak"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase536 Pair Quality and Selectivity Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"windows={d['windows']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, seeds={d['random_seeds']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Transfer format: min / mean / specificity.")
        lines.append("")
        lines.append("| pair | base margin | base rank | top1 | template cos avg | best common | best direct | best shuffled | random max | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        candidates = []
        for pair in d["pairs"]:
            b = d["pair_baseline"][pair]
            cw, common = best_condition(d, pair, "common")
            _dw, direct = best_condition(d, pair, "direct")
            _sw, shuffled = best_condition(d, pair, "shuffled")
            rand_max = max(
                float(d["audit"][pair][win]["conditions"]["random"]["max_transfer_min"])
                for win in d["audit"][pair]
            )
            # Use first layer's cosine avg as a simple quality marker.
            first_layer = str(d["all_layers"][0])
            cos_row = d["layer_stats"][first_layer][pair]
            cos_avg = sum(float(v) for v in cos_row.values()) / len(cos_row)
            v = pair_verdict(d, pair)
            if v == "candidate_common_pair":
                candidates.append(pair)
            lines.append(
                f"| {pair} | {float(b['mean_margin']):+.3f} | {float(b['mean_rank']):.1f} | "
                f"{float(b['mean_top1']):.2f} | {cos_avg:+.3f} | {cw}:{tr_cell(common)} | "
                f"{tr_cell(direct)} | {tr_cell(shuffled)} | {rand_max:+.3f} | {v} |"
            )
        lines.append("")
        compact.append({"model": model, "candidate_pairs": ",".join(candidates) if candidates else "none"})

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | candidate common pairs |")
        lines.append("|---|---|")
        for row in compact:
            lines.append(f"| {row['model']} | {row['candidate_pairs']} |")
        lines.append("")

    out = root / "phase536_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
