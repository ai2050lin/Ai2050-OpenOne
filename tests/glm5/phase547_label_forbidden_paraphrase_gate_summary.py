#!/usr/bin/env python3
"""Summary for Phase547 label-forbidden paraphrase gate split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase547_label_forbidden_paraphrase_gate")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase547_{model}_label_forbidden_paraphrase_gate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], pair: str, scaffold: str, mode: str, condition: str, metric: str) -> tuple[str, dict[str, Any]]:
    rows = {win: d["audit"][win]["sources"][pair][scaffold][mode][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w][metric])
    return win, rows[win]


def classify(base: dict[str, Any], row: dict[str, Any]) -> str:
    clean_gain = row["clean_synonym_rate"] - base["clean_synonym_rate"]
    label_gain = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    wrong_gain = row["wrong_synonym_rate"] - base["wrong_synonym_rate"]
    score_gain = row["clean_paraphrase_score"] - base["clean_paraphrase_score"]
    if clean_gain >= 0.15 and label_gain <= 0.05 and wrong_gain <= 0.05 and score_gain >= 0.10:
        return "clean_paraphrase_positive"
    if clean_gain >= 0.15 and label_gain > 0.05:
        return "synonym_with_label_leak"
    if label_gain >= 0.15 and clean_gain < 0.10:
        return "label_gate_leak"
    if clean_gain >= 0.08:
        return "weak_clean_positive"
    if score_gain <= -0.10:
        return "negative"
    return "flat"


def row_line(model: str, pair: str, scaffold: str, mode: str, condition: str, win: str, base: dict[str, Any], row: dict[str, Any]) -> str:
    clean_gain = row["clean_synonym_rate"] - base["clean_synonym_rate"]
    label_gain = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    score_gain = row["clean_paraphrase_score"] - base["clean_paraphrase_score"]
    return (
        f"| {model} | {pair} | {scaffold} | {mode} | {condition} | {win} | "
        f"{base['clean_synonym_rate']:.2f} | {row['clean_synonym_rate']:.2f} | {clean_gain:+.2f} | "
        f"{base['any_label_violation_rate']:.2f} | {row['any_label_violation_rate']:.2f} | {label_gain:+.2f} | "
        f"{row['wrong_synonym_rate']:.2f} | {row['generic_only_rate']:.2f} | "
        f"{row['clean_paraphrase_score']:.2f} | {score_gain:+.2f} | {classify(base, row)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase547 Label-Forbidden Paraphrase Gate Summary", ""]
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
        lines.append("| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for pair in d["pairs"]:
            for scaffold in d["scaffolds"]:
                for mode in d["decode_modes"]:
                    _, base = best_window(d, pair, scaffold, mode, "baseline", "clean_synonym_rate")
                    for condition in d["conditions"]:
                        win, row = best_window(d, pair, scaffold, mode, condition, "clean_synonym_rate")
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
                                "clean_gain": row["clean_synonym_rate"] - base["clean_synonym_rate"],
                                "label_gain": row["any_label_violation_rate"] - base["any_label_violation_rate"],
                                "score_gain": row["clean_paraphrase_score"] - base["clean_paraphrase_score"],
                                "class": classify(base, row),
                            })
        lines.append("")

    if compact:
        lines.append("## Best Clean Paraphrase Gains")
        lines.append("")
        lines.append("| model | pair | scaffold | mode | condition | win | base clean | clean | clean gain | base label | label | label gain | wrong | generic | score | score gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(compact, key=lambda x: (x["class"] == "clean_paraphrase_positive", x["clean_gain"], x["score_gain"]), reverse=True)[:80]:
            lines.append(row_line(r["model"], r["pair"], r["scaffold"], r["mode"], r["condition"], r["win"], r["base"], r["row"]))
        lines.append("")

        lines.append("## Pair Max Clean Gain")
        lines.append("")
        lines.append("| model | pair | max clean gain | label gain | score gain | row | class |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for model in sorted(set(r["model"] for r in compact)):
            for pair in sorted(set(r["pair"] for r in compact if r["model"] == model)):
                rows = [r for r in compact if r["model"] == model and r["pair"] == pair]
                best = max(rows, key=lambda x: x["clean_gain"])
                desc = f"{best['scaffold']} {best['mode']} {best['condition']} {best['win']}"
                lines.append(
                    f"| {model} | {pair} | {best['clean_gain']:+.2f} | {best['label_gain']:+.2f} | "
                    f"{best['score_gain']:+.2f} | {desc} | {best['class']} |"
                )
        lines.append("")

        lines.append("## Representative Samples")
        lines.append("")
        lines.append("| model | window | pair | scaffold | mode | condition | seed | quality | prompt | suffix |")
        lines.append("|---|---|---|---|---|---|---:|---|---|---|")
        for model, d in data.items():
            used = 0
            for rec in d.get("sample_records", []):
                if rec["quality"] not in {"clean_synonym", "synonym_with_label_violation", "label_violation", "wrong_synonym", "generic_only", "other"}:
                    continue
                prompt = str(rec["prompt"]).replace("|", "\\|").replace("\n", " ")[:110]
                suffix = str(rec["generated_suffix"]).replace("|", "\\|").replace("\n", " ")[:110]
                lines.append(
                    f"| {model} | {rec['window']} | {rec['pair']} | {rec['scaffold']} | {rec['mode']} | "
                    f"{rec['condition']} | {rec['seed']} | {rec['quality']} | {prompt} | {suffix} |"
                )
                used += 1
                if used >= 22:
                    break
            lines.append("")

    out = root / "phase547_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
