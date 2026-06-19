#!/usr/bin/env python3
"""Summary for Phase537 vehicle/furniture clean-common candidate audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase537_vehicle_furniture_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase537_{model}_vehicle_furniture_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tr_cell(tr: dict[str, Any]) -> str:
    return (
        f"{float(tr['best_transfer_min']):+.3f}/"
        f"{float(tr['best_transfer_mean']):+.3f}/"
        f"{float(tr['best_off_pair_max_abs']):.3f}/"
        f"{float(tr['pair_specificity']):.2f}"
    )


def condition_best(d: dict[str, Any], condition: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["conditions"][condition]["transfer"]
        for win in d["audit"]
    }
    best_win = max(rows, key=lambda w: rows[w]["best_transfer_min"])
    return best_win, rows[best_win]


def common_pass(d: dict[str, Any], win: str) -> bool:
    common = d["audit"][win]["conditions"]["common"]["transfer"]
    direct = d["audit"][win]["conditions"]["direct"]["transfer"]
    shuffled = d["audit"][win]["conditions"]["shuffled"]["transfer"]
    random = d["audit"][win]["conditions"]["random"]
    return (
        float(common["best_transfer_min"]) > 0.25
        and float(common["pair_specificity"]) > 1.0
        and float(common["best_transfer_min"]) > float(random["max_transfer_min"])
        and int(random["pass_like_count"]) == 0
        and float(common["best_transfer_min"]) >= float(direct["best_transfer_min"])
        and float(common["best_transfer_min"]) >= float(shuffled["best_transfer_min"])
    )


def verdict(d: dict[str, Any]) -> str:
    pass_windows = [win for win in d["audit"] if common_pass(d, win)]
    if len(pass_windows) >= 2:
        return "clean_candidate_replicated"
    if len(pass_windows) == 1:
        return "single_window_candidate"
    cw, common = condition_best(d, "common")
    random_max = max(
        float(d["audit"][win]["conditions"]["random"]["max_transfer_min"])
        for win in d["audit"]
    )
    if float(common["best_transfer_min"]) > 0.25 and float(common["best_transfer_min"]) > random_max:
        return f"strong_but_not_clean:{cw}"
    return "not_clean"


def gen_cell(row: dict[str, Any], base: dict[str, Any] | None = None) -> str:
    hit = float(row["target_hit_rate"])
    rank = float(row["mean_best_target_rank"])
    margin = float(row["mean_first_step_margin"])
    if base is None:
        return f"{hit:.2f}/{rank:.1f}/{margin:+.3f}"
    return (
        f"{hit:.2f}/{rank:.1f}/{margin:+.3f} "
        f"(Δrank {rank - float(base['mean_best_target_rank']):+.1f}, "
        f"Δmargin {margin - float(base['mean_first_step_margin']):+.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase537 Vehicle/Furniture Audit Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"source={d['source_pair']}, offpairs={d['offpairs']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, bridge_n={d['bridge_n']}, "
            f"alphas={d['alphas']}, seeds={len(d['random_seeds'])}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Transfer format: source min / source mean / off-pair max abs / specificity.")
        lines.append("")
        lines.append("| window | layers | common | direct | shuffled | random max | random pass-like | common pass |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for win, row in d["audit"].items():
            common = row["conditions"]["common"]["transfer"]
            direct = row["conditions"]["direct"]["transfer"]
            shuffled = row["conditions"]["shuffled"]["transfer"]
            random = row["conditions"]["random"]
            lines.append(
                f"| {win} | {row['window']} | {tr_cell(common)} | {tr_cell(direct)} | "
                f"{tr_cell(shuffled)} | {float(random['max_transfer_min']):+.3f} | "
                f"{int(random['pass_like_count'])} | {common_pass(d, win)} |"
            )
        lines.append("")

        cw, common = condition_best(d, "common")
        lines.append(f"Best common window: {cw}, transfer={tr_cell(common)}")
        lines.append("")
        lines.append("| off-pair | max abs delta at best common |")
        lines.append("|---|---:|")
        for pair, val in common["best_off_by_pair"].items():
            lines.append(f"| {pair} | {float(val):.3f} |")
        lines.append("")

        gen = d["generation"]
        base = gen["baseline"]
        lines.append(f"Generation bridge window: {gen['best_window']}")
        lines.append("")
        lines.append("| condition | hit / best rank / first margin |")
        lines.append("|---|---:|")
        lines.append(f"| baseline | {gen_cell(base)} |")
        for cond in ["common", "direct", "shuffled", "random"]:
            lines.append(f"| {cond} | {gen_cell(gen[cond], base)} |")
        lines.append("")
        v = verdict(d)
        lines.append(f"Verdict: {v}")
        lines.append("")
        compact.append({
            "model": model,
            "best_common_window": cw,
            "best_common_min": common["best_transfer_min"],
            "best_common_specificity": common["pair_specificity"],
            "verdict": v,
        })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | best common window | source min | specificity | verdict |")
        lines.append("|---|---|---:|---:|---|")
        for row in compact:
            lines.append(
                f"| {row['model']} | {row['best_common_window']} | "
                f"{float(row['best_common_min']):+.3f} | {float(row['best_common_specificity']):.2f} | "
                f"{row['verdict']} |"
            )
        lines.append("")

    out = root / "phase537_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
