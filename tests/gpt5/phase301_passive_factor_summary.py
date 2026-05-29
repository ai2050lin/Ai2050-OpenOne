from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def mean(values: list[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    probe_rows = data["probe_rows"]
    best_probe = {}
    for variable in sorted({row["variable"] for row in probe_rows}):
        items = [row for row in probe_rows if row["variable"] == variable]
        best_probe[variable] = max(items, key=lambda row: row["test_accuracy"])
    return {
        "model": data["model"],
        "complete": data["complete"],
        "num_bases": data["num_bases"],
        "num_train_bases": data["num_train_bases"],
        "num_test_bases": data["num_test_bases"],
        "num_results": data["num_results"],
        "nonfinite_rows": data["summary"]["nonfinite_rows"],
        "probe_best": best_probe,
        "best_by_variable_direction": data["summary"]["best_by_variable_direction"],
        "variable_direction_curve": data["summary"]["variable_direction_curve"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase301_passive_factor_closure")
    parser.add_argument("--output-dir", default="results/gpt5_phase301_passive_factor_closure")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        model: summarize(input_dir / f"{model}_phase301_passive_factor_closure.json")
        for model in MODELS
    }
    lines = ["# Phase 301 Passive Factor Closure Summary\n\n"]
    for model, summary in summaries.items():
        lines.append(f"## {model}\n")
        lines.append(f"- complete: {summary['complete']}\n")
        lines.append(f"- bases/train/test: {summary['num_bases']} / {summary['num_train_bases']} / {summary['num_test_bases']}\n")
        lines.append(f"- rows: {summary['num_results']}\n")
        lines.append(f"- nonfinite_rows: {summary['nonfinite_rows']}\n")
        lines.append("- probe_best:\n")
        for variable, row in summary["probe_best"].items():
            lines.append(
                f"  - {variable}: L{row['layer']} {row['module']} "
                f"acc={row['test_accuracy']:.6f} margin={row['mean_signed_margin']:.6f}\n"
            )
        lines.append("- best_by_variable_direction:\n")
        for key, row in sorted(summary["best_by_variable_direction"].items()):
            lines.append(
                f"  - {key}: L{row['layer']} {row['module']} "
                f"progress={row['mean_progress']:.6f} kl={row['mean_kl_ratio']:.6f} "
                f"delta={row['mean_logit_delta_ratio']:.6f}\n"
            )
        lines.append("- variable_direction_curve:\n")
        for row in summary["variable_direction_curve"]:
            lines.append(
                f"  - {row['variable']} {row['direction']}: "
                f"progress={row['mean_progress']:.6f} kl={row['mean_kl_ratio']:.6f}\n"
            )
        lines.append("\n")

    (output_dir / "passive_factor_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output_dir / "PASSIVE_FACTOR_SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print(f"saved {output_dir / 'passive_factor_summary.json'}")
    print(f"saved {output_dir / 'PASSIVE_FACTOR_SUMMARY.md'}")


if __name__ == "__main__":
    main()
