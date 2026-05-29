from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def mean(values: list[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else 0.0


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(path: Path) -> dict[str, Any]:
    data = load(path)
    rows = data["results"]

    alpha1 = [row for row in rows if abs(float(row["alpha"]) - 1.0) < 1e-9]
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    categories: dict[str, str] = {}
    for row in alpha1:
        subtype = str(row["subtype"])
        categories[subtype] = str(row["category"])
        grouped[(subtype, int(row["layer"]), str(row["patch_type"]))].append(float(row["progress"]))

    best_by_subtype = []
    for subtype in sorted(categories):
        candidates = []
        for (sub, layer, patch), vals in grouped.items():
            if sub == subtype:
                candidates.append((mean(vals), layer, patch))
        if not candidates:
            continue
        progress, layer, patch = max(candidates, key=lambda item: item[0])
        best_by_subtype.append({
            "subtype": subtype,
            "category": categories[subtype],
            "best_patch_type": patch,
            "best_layer": layer,
            "best_progress": progress,
        })

    by_category: dict[str, list[float]] = defaultdict(list)
    for row in best_by_subtype:
        by_category[row["category"]].append(float(row["best_progress"]))

    return {
        "model": data["model"],
        "complete": data.get("complete"),
        "num_pairs": data.get("num_pairs"),
        "num_results": data.get("num_results"),
        "target_layers": data.get("target_layers"),
        "categories": data.get("categories"),
        "subtypes": data.get("subtypes"),
        "nonfinite_rows": data.get("summary", {}).get("nonfinite_rows"),
        "best_by_patch_type": data.get("summary", {}).get("best_by_patch_type", {}),
        "best_patch_counts": dict(Counter(row["best_patch_type"] for row in best_by_subtype)),
        "best_layer_counts": dict(sorted(Counter(row["best_layer"] for row in best_by_subtype).items())),
        "best_category_mean": {
            category: mean(values)
            for category, values in sorted(by_category.items())
        },
        "top_best_subtypes": sorted(best_by_subtype, key=lambda row: row["best_progress"], reverse=True)[:10],
        "bottom_best_subtypes": sorted(best_by_subtype, key=lambda row: row["best_progress"])[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase298_expanded_dynamic_normal")
    parser.add_argument("--output-dir", default="results/gpt5_phase298_expanded_dynamic_normal")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = ["# Phase 298 Expanded Dynamic Recompute Summary\n\n"]
    summaries = {}
    for model in MODELS:
        path = input_dir / f"{model}_phase294_dynamic_recompute_pilot.json"
        summary = summarize_model(path)
        summaries[model] = summary
        report.append(f"## {model}\n")
        report.append(f"- complete: {summary['complete']}\n")
        report.append(f"- pairs: {summary['num_pairs']}\n")
        report.append(f"- rows: {summary['num_results']}\n")
        report.append(f"- nonfinite_rows: {summary['nonfinite_rows']}\n")
        report.append(f"- target_layers: {summary['target_layers']}\n")
        report.append(f"- best_patch_counts: {summary['best_patch_counts']}\n")
        report.append(f"- best_layer_counts: {summary['best_layer_counts']}\n")
        report.append("- best_by_patch_type:\n")
        for patch, value in sorted(summary["best_by_patch_type"].items()):
            report.append(f"  - {patch}: layer={value['layer']}, progress={value['progress']:.6f}\n")
        report.append("- top_best_subtypes:\n")
        for row in summary["top_best_subtypes"][:5]:
            report.append(
                f"  - {row['subtype']} ({row['category']}): {row['best_patch_type']} "
                f"L{row['best_layer']} progress={row['best_progress']:.6f}\n"
            )
        report.append("- bottom_best_subtypes:\n")
        for row in summary["bottom_best_subtypes"][:5]:
            report.append(
                f"  - {row['subtype']} ({row['category']}): {row['best_patch_type']} "
                f"L{row['best_layer']} progress={row['best_progress']:.6f}\n"
            )
        report.append("\n")

    (output_dir / "expanded_dynamic_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output_dir / "EXPANDED_DYNAMIC_SUMMARY.md").write_text("".join(report), encoding="utf-8")
    print(f"saved {output_dir / 'expanded_dynamic_summary.json'}")
    print(f"saved {output_dir / 'EXPANDED_DYNAMIC_SUMMARY.md'}")


if __name__ == "__main__":
    main()
