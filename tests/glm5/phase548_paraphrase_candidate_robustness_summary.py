#!/usr/bin/env python3
"""Summary for Phase548 paraphrase candidate robustness audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase548_paraphrase_candidate_robustness")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase548_{model}_paraphrase_candidate_robustness.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], scaffold: str, mode: str, condition: str, metric: str) -> tuple[str, dict[str, Any]]:
    pair = d["pair"]
    rows = {win: d["audit"][win]["sources"][pair][scaffold][mode][condition] for win in d["audit"]}
    win = max(rows, key=lambda w: rows[w][metric])
    return win, rows[win]


def classify(base: dict[str, Any], row: dict[str, Any], random_best: dict[str, Any]) -> str:
    clean_gain = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    label_gain = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    score_gain = row["clean_non_object_score"] - base["clean_non_object_score"]
    rand_gain = random_best["clean_non_object_rate"] - base["clean_non_object_rate"]
    above_random = clean_gain - rand_gain
    if clean_gain >= 0.15 and score_gain >= 0.10 and label_gain <= 0.05 and above_random >= 0.08:
        return "robust_clean_positive"
    if clean_gain >= 0.15 and label_gain <= 0.05:
        return "clean_positive_not_above_random"
    if label_gain >= 0.15:
        return "label_leak"
    if clean_gain >= 0.08:
        return "weak_clean"
    if score_gain <= -0.10:
        return "negative"
    return "flat"


def row_line(
    model: str,
    scaffold: str,
    mode: str,
    condition: str,
    win: str,
    base: dict[str, Any],
    row: dict[str, Any],
    random_best: dict[str, Any],
) -> str:
    clean_gain = row["clean_non_object_rate"] - base["clean_non_object_rate"]
    label_gain = row["any_label_violation_rate"] - base["any_label_violation_rate"]
    score_gain = row["clean_non_object_score"] - base["clean_non_object_score"]
    rand_gain = random_best["clean_non_object_rate"] - base["clean_non_object_rate"]
    return (
        f"| {model} | {scaffold} | {mode} | {condition} | {win} | "
        f"{base['clean_non_object_rate']:.2f} | {row['clean_non_object_rate']:.2f} | {clean_gain:+.2f} | "
        f"{base['any_label_violation_rate']:.2f} | {row['any_label_violation_rate']:.2f} | {label_gain:+.2f} | "
        f"{row['object_echo_rate']:.2f} | {row['prompt_echo_rate']:.2f} | "
        f"{row['clean_non_object_score']:.2f} | {score_gain:+.2f} | {rand_gain:+.2f} | "
        f"{classify(base, row, random_best)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase548 Paraphrase Candidate Robustness Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, scaffolds={d['scaffolds']}, modes={d['decode_modes']}, "
            f"conditions={d['conditions']}, windows={d['windows']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, alpha={d['alpha']}"
        )
        lines.append("")
        lines.append("| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for scaffold in d["scaffolds"]:
            for mode in d["decode_modes"]:
                _, base = best_window(d, scaffold, mode, "baseline", "clean_non_object_rate")
                random_rows = []
                for rc in ["random_full", "random_perp"]:
                    if rc in d["conditions"]:
                        _, rr = best_window(d, scaffold, mode, rc, "clean_non_object_rate")
                        random_rows.append(rr)
                random_best = max(random_rows, key=lambda x: x["clean_non_object_rate"]) if random_rows else base
                for condition in d["conditions"]:
                    win, row = best_window(d, scaffold, mode, condition, "clean_non_object_rate")
                    lines.append(row_line(model, scaffold, mode, condition, win, base, row, random_best))
                    if condition != "baseline":
                        compact.append({
                            "model": model,
                            "pair": d["pair"],
                            "scaffold": scaffold,
                            "mode": mode,
                            "condition": condition,
                            "win": win,
                            "base": base,
                            "row": row,
                            "random_best": random_best,
                            "clean_gain": row["clean_non_object_rate"] - base["clean_non_object_rate"],
                            "score_gain": row["clean_non_object_score"] - base["clean_non_object_score"],
                            "random_gain": random_best["clean_non_object_rate"] - base["clean_non_object_rate"],
                            "class": classify(base, row, random_best),
                        })
        lines.append("")

    if compact:
        lines.append("## Best Strict Clean Non-Object Gains")
        lines.append("")
        lines.append("| model | scaffold | mode | condition | win | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score | score gain | random gain | class |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        rows = sorted(compact, key=lambda x: (x["class"] == "robust_clean_positive", x["clean_gain"], x["score_gain"]), reverse=True)
        for r in rows[:80]:
            lines.append(row_line(r["model"], r["scaffold"], r["mode"], r["condition"], r["win"], r["base"], r["row"], r["random_best"]))
        lines.append("")

        lines.append("## Model Max Strict Clean Gain")
        lines.append("")
        lines.append("| model | max clean gain | random gain | score gain | row | class |")
        lines.append("|---|---:|---:|---:|---|---|")
        for model in sorted(set(r["model"] for r in compact)):
            rows_m = [r for r in compact if r["model"] == model and not r["condition"].startswith("random")]
            best = max(rows_m, key=lambda x: x["clean_gain"])
            desc = f"{best['scaffold']} {best['mode']} {best['condition']} {best['win']}"
            lines.append(
                f"| {model} | {best['clean_gain']:+.2f} | {best['random_gain']:+.2f} | "
                f"{best['score_gain']:+.2f} | {desc} | {best['class']} |"
            )
        lines.append("")

        lines.append("## Readable Samples")
        lines.append("")
        lines.append("| model | window | scaffold | mode | condition | seed | object | quality | clean-no | labels | target terms | suffix |")
        lines.append("|---|---|---|---|---|---:|---|---|---|---|---|---|")
        for model, d in data.items():
            used = 0
            for rec in d.get("sample_records", []):
                if rec["quality"] not in {"clean_synonym", "synonym_with_label_violation", "label_violation", "wrong_synonym", "generic_only", "other"}:
                    continue
                suffix = str(rec["generated_suffix"]).replace("|", "\\|").replace("\n", " ")[:130]
                labels = ",".join(rec.get("target_label_matches", []) + rec.get("competitor_label_matches", []))
                terms = ",".join(rec.get("target_non_object_matches", []))
                lines.append(
                    f"| {model} | {rec['window']} | {rec['scaffold']} | {rec['mode']} | {rec['condition']} | "
                    f"{rec['seed']} | {rec['object']} | {rec['quality']} | {rec['clean_non_object']} | "
                    f"{labels} | {terms} | {suffix} |"
                )
                used += 1
                if used >= 36:
                    break
            lines.append("")

    out = root / "phase548_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
