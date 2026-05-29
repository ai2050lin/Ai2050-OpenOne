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
    best_probe = max(probe_rows, key=lambda row: row["test_accuracy"]) if probe_rows else {}
    acc_values = [float(row["test_accuracy"]) for row in probe_rows]
    return {
        "model": data["model"],
        "complete": data["complete"],
        "num_pairs": data["num_pairs"],
        "num_train_pairs": data["num_train_pairs"],
        "num_test_pairs": data["num_test_pairs"],
        "num_results": data["num_results"],
        "nonfinite_rows": data["summary"]["nonfinite_rows"],
        "probe_mean_accuracy": mean(acc_values),
        "probe_best": best_probe,
        "best_by_direction": data["summary"]["best_by_direction"],
        "top_active_to_passive": sorted(
            [row for row in data["summary"]["layer_module_curve"] if row["direction"] == "active_to_passive"],
            key=lambda row: row["mean_progress"],
            reverse=True,
        )[:5],
        "top_passive_to_active": sorted(
            [row for row in data["summary"]["layer_module_curve"] if row["direction"] == "passive_to_active"],
            key=lambda row: row["mean_progress"],
            reverse=True,
        )[:5],
        "subtype_curve": data["summary"]["subtype_curve"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase300_voice_closure_pilot")
    parser.add_argument("--output-dir", default="results/gpt5_phase300_voice_closure_pilot")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        model: summarize(input_dir / f"{model}_phase300_voice_closure_pilot.json")
        for model in MODELS
    }
    lines = ["# Phase 300 Voice Closure Pilot Summary\n\n"]
    for model, summary in summaries.items():
        lines.append(f"## {model}\n")
        lines.append(f"- complete: {summary['complete']}\n")
        lines.append(f"- pairs/train/test: {summary['num_pairs']} / {summary['num_train_pairs']} / {summary['num_test_pairs']}\n")
        lines.append(f"- rows: {summary['num_results']}\n")
        lines.append(f"- nonfinite_rows: {summary['nonfinite_rows']}\n")
        lines.append(f"- probe_mean_accuracy: {summary['probe_mean_accuracy']:.6f}\n")
        bp = summary["probe_best"]
        lines.append(
            f"- best_probe: L{bp.get('layer')} {bp.get('module')} "
            f"acc={float(bp.get('test_accuracy', 0.0)):.6f} margin={float(bp.get('mean_signed_margin', 0.0)):.6f}\n"
        )
        lines.append("- best_by_direction:\n")
        for direction, row in summary["best_by_direction"].items():
            lines.append(
                f"  - {direction}: L{row['layer']} {row['module']} "
                f"progress={row['mean_progress']:.6f} kl={row['mean_kl_ratio']:.6f} "
                f"delta={row['mean_logit_delta_ratio']:.6f}\n"
            )
        lines.append("- subtype_curve:\n")
        for row in summary["subtype_curve"]:
            lines.append(
                f"  - {row['subtype']} {row['direction']}: "
                f"progress={row['mean_progress']:.6f} kl={row['mean_kl_ratio']:.6f}\n"
            )
        lines.append("\n")

    (output_dir / "voice_closure_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output_dir / "VOICE_CLOSURE_SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    print(f"saved {output_dir / 'voice_closure_summary.json'}")
    print(f"saved {output_dir / 'VOICE_CLOSURE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
