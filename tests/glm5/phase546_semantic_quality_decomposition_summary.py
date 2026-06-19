#!/usr/bin/env python3
"""Summary for Phase546 semantic quality decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase546_semantic_quality_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase546_{model}_semantic_quality_decomposition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], pair: str, scaffold: str, mode: str, condition: str, metric: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["sources"][pair][scaffold][mode][condition]
        for win in d["audit"]
    }
    win = max(rows, key=lambda w: rows[w][metric])
    return win, rows[win]


def classify_gain(base: dict[str, Any], row: dict[str, Any]) -> str:
    sem_gain = row["semantic_ok_rate"] - base["semantic_ok_rate"]
    exact_gain = row["exact_label_rate"] - base["exact_label_rate"]
    para_gain = row["family_non_exact_rate"] - base["family_non_exact_rate"]
    wrong_gain = row["wrong_family_rate"] - base["wrong_family_rate"]
    if sem_gain >= 0.20 and para_gain >= exact_gain and para_gain >= 0.10:
        return "semantic_paraphrase_positive"
    if sem_gain >= 0.20 and exact_gain > para_gain:
        return "label_driven_positive"
    if sem_gain >= 0.10:
        return "weak_positive"
    if wrong_gain >= 0.10 and sem_gain < 0.10:
        return "wrong_family_shift"
    if sem_gain <= -0.10:
        return "negative"
    return "flat"


def row_line(model: str, pair: str, scaffold: str, mode: str, condition: str, win: str, base: dict[str, Any], row: dict[str, Any]) -> str:
    sem_gain = row["semantic_ok_rate"] - base["semantic_ok_rate"]
    exact_gain = row["exact_label_rate"] - base["exact_label_rate"]
    para_gain = row["family_non_exact_rate"] - base["family_non_exact_rate"]
    return (
        f"| {model} | {pair} | {scaffold} | {mode} | {condition} | {win} | "
        f"{base['semantic_ok_rate']:.2f} | {row['semantic_ok_rate']:.2f} | {sem_gain:+.2f} | "
        f"{row['exact_label_rate']:.2f} | {exact_gain:+.2f} | "
        f"{row['family_non_exact_rate']:.2f} | {para_gain:+.2f} | "
        f"{row['label_share_of_semantic_ok']:.2f} | {row['wrong_family_rate']:.2f} | "
        f"{row['generic_only_rate']:.2f} | {classify_gain(base, row)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase546 Semantic Quality Decomposition Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pairs={d['pairs']}, scaffolds={d['scaffolds']}, modes={d['decode_modes']}, "
            f"conditions={d['conditions']}, windows={d['windows']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, alpha={d['alpha']}"
        )
        lines.append("")
        lines.append("| pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for pair in d["pairs"]:
            for scaffold in d["scaffolds"]:
                for mode in d["decode_modes"]:
                    _, base = best_window(d, pair, scaffold, mode, "baseline", "semantic_ok_rate")
                    for condition in d["conditions"]:
                        win, row = best_window(d, pair, scaffold, mode, condition, "semantic_ok_rate")
                        lines.append(row_line(model, pair, scaffold, mode, condition, win, base, row))
                        if condition != "baseline":
                            compact.append({
                                "model": model,
                                "pair": pair,
                                "scaffold": scaffold,
                                "mode": mode,
                                "condition": condition,
                                "win": win,
                                "base": base,
                                "row": row,
                                "semantic_gain": row["semantic_ok_rate"] - base["semantic_ok_rate"],
                                "exact_gain": row["exact_label_rate"] - base["exact_label_rate"],
                                "non_exact_gain": row["family_non_exact_rate"] - base["family_non_exact_rate"],
                                "class": classify_gain(base, row),
                            })
        lines.append("")

    if compact:
        lines.append("## Best Semantic Gains")
        lines.append("")
        lines.append("| model | pair | scaffold | mode | condition | win | base semantic | semantic | gain | exact | exact gain | non-exact | non-exact gain | label share | wrong | generic | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(compact, key=lambda x: (x["semantic_gain"], x["non_exact_gain"]), reverse=True)[:60]:
            lines.append(row_line(r["model"], r["pair"], r["scaffold"], r["mode"], r["condition"], r["win"], r["base"], r["row"]))
        lines.append("")

        lines.append("## Pair Max Semantic Gain")
        lines.append("")
        lines.append("| model | pair | max semantic gain | exact gain | non-exact gain | row | class |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for model in sorted(set(r["model"] for r in compact)):
            for pair in sorted(set(r["pair"] for r in compact if r["model"] == model)):
                rows = [r for r in compact if r["model"] == model and r["pair"] == pair]
                best = max(rows, key=lambda x: x["semantic_gain"])
                desc = f"{best['scaffold']} {best['mode']} {best['condition']} {best['win']}"
                lines.append(
                    f"| {model} | {pair} | {best['semantic_gain']:+.2f} | "
                    f"{best['exact_gain']:+.2f} | {best['non_exact_gain']:+.2f} | {desc} | {best['class']} |"
                )
        lines.append("")

        lines.append("## Representative Samples")
        lines.append("")
        lines.append("| model | window | pair | scaffold | mode | condition | seed | quality | prompt | suffix |")
        lines.append("|---|---|---|---|---|---|---:|---|---|---|")
        for model, d in data.items():
            wanted = {"exact_label", "family_non_exact", "wrong_family", "generic_only", "other"}
            used = 0
            for rec in d.get("sample_records", []):
                if rec["quality"] not in wanted:
                    continue
                prompt = str(rec["prompt"]).replace("|", "\\|").replace("\n", " ")[:90]
                suffix = str(rec["generated_suffix"]).replace("|", "\\|").replace("\n", " ")[:90]
                lines.append(
                    f"| {model} | {rec['window']} | {rec['pair']} | {rec['scaffold']} | {rec['mode']} | "
                    f"{rec['condition']} | {rec['seed']} | {rec['quality']} | {prompt} | {suffix} |"
                )
                used += 1
                if used >= 18:
                    break
            lines.append("")

    out = root / "phase546_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
