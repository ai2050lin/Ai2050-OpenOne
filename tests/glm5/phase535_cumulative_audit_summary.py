#!/usr/bin/env python3
"""Summary for Phase535 cumulative audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase535_cumulative_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
PAIRS = ["fruit_nonfruit", "animal_vehicle"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase535_{model}_cumulative_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tr_cell(tr: dict[str, Any]) -> str:
    return (
        f"{float(tr['best_transfer_min']):+.3f}/"
        f"{float(tr['best_transfer_mean']):+.3f}/"
        f"{float(tr['best_transfer_ratio']):.2f}/"
        f"{'Y' if tr['passes_transfer_gate'] else 'n'}"
    )


def gen_cell(row: dict[str, Any]) -> str:
    return (
        f"hit={float(row['target_hit_rate']):.2f}, "
        f"rank={float(row['mean_best_target_rank']):.1f}, "
        f"m1={float(row['mean_first_step_margin']):+.3f}"
    )


def best_common(d: dict[str, Any], pair: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][pair][win]["conditions"]["common"]["transfer"]
        for win in d["audit"][pair]
    }
    win = max(rows, key=lambda w: rows[w]["best_transfer_min"])
    return win, rows[win]


def verdict(d: dict[str, Any]) -> str:
    q = []
    for pair in PAIRS:
        win, common = best_common(d, pair)
        random_ctrl = d["audit"][pair][win]["conditions"]["random"]
        direct = d["audit"][pair][win]["conditions"]["direct"]["transfer"]
        shuffled = d["audit"][pair][win]["conditions"]["shuffled_template"]["transfer"]
        clean = (
            common["passes_transfer_gate"]
            and float(common["best_transfer_min"]) > float(random_ctrl["max_transfer_min"])
            and int(random_ctrl["pass_count"]) == 0
            and float(common["best_transfer_min"]) >= float(direct["best_transfer_min"])
            and float(common["best_transfer_min"]) >= float(shuffled["best_transfer_min"])
        )
        q.append(clean)
    if all(q):
        return "two_pair_clean_common"
    if any(q):
        return "one_pair_clean_common"
    return "no_clean_common"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase535 Cumulative Audit Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"windows={d['windows']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"bridge_n={d['bridge_n']}, max_new_tokens={d['max_new_tokens']}, "
            f"alphas={d['alphas']}, seeds={d['random_seeds']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        for pair in PAIRS:
            lines.append(f"### {pair}")
            lines.append("")
            lines.append("Transfer format: min / mean / ratio / pass.")
            lines.append("")
            lines.append("| window | common | direct | shuffled_template | random max min | random pass count |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for win, payload in d["audit"][pair].items():
                common = payload["conditions"]["common"]["transfer"]
                direct = payload["conditions"]["direct"]["transfer"]
                shuffled = payload["conditions"]["shuffled_template"]["transfer"]
                rand = payload["conditions"]["random"]
                lines.append(
                    f"| {win} {payload['window']} | {tr_cell(common)} | {tr_cell(direct)} | "
                    f"{tr_cell(shuffled)} | {float(rand['max_transfer_min']):+.3f} | {int(rand['pass_count'])} |"
                )
            lines.append("")

            gen = d["generation"][pair]
            lines.append(f"Generation best_window={gen['best_window']}")
            lines.append("")
            lines.append("| condition | trace |")
            lines.append("|---|---|")
            for cond in ["baseline", "common", "direct", "random"]:
                lines.append(f"| {cond} | {gen_cell(gen[cond])} |")
            lines.append("")

        compact.append({"model": model, "verdict": verdict(d)})

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | verdict |")
        lines.append("|---|---|")
        for row in compact:
            lines.append(f"| {row['model']} | {row['verdict']} |")
        lines.append("")

    out = root / "phase535_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
