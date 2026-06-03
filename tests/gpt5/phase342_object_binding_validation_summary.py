from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE_BLOCKS = {
    "qwen3": ("L21-29_mlp", "L21-29_attn", "L21-29_full"),
    "glm4": ("L30-38_mlp", "L30-38_attn", "L30-38_full"),
    "deepseek7b": ("L19-24_mlp", "L19-24_attn", "L19-24_full"),
}

IDENTITY_BLOCK = {
    "qwen3": "identity_L0-2_full",
    "glm4": "identity_L0-4_full",
    "deepseek7b": "identity_L0-2_full",
}


def load_result(input_dir: Path, model: str) -> dict[str, Any]:
    path = input_dir / f"{model}_phase339.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def summarize_model(data: dict[str, Any]) -> dict[str, Any]:
    model = data["model"]
    mlp_name, attn_name, full_name = BASELINE_BLOCKS[model]
    baseline_rows = []
    mlp_gt_attn = []
    mlp_values = []
    attn_values = []
    full_values = []
    valid_counts = []
    for baseline, bdata in sorted(data["phase339_multibaseline"].items()):
        aggs = bdata.get("aggs", {})
        mlp = fnum(aggs.get(mlp_name, {}).get("mean"))
        attn = fnum(aggs.get(attn_name, {}).get("mean"))
        full = fnum(aggs.get(full_name, {}).get("mean"))
        n_valid = int(bdata.get("n_valid", 0))
        baseline_rows.append({
            "baseline": baseline,
            "mlp": mlp,
            "attn": attn,
            "full": full,
            "n_valid": n_valid,
            "mlp_gt_attn": bool(mlp > attn),
        })
        mlp_gt_attn.append(mlp > attn)
        mlp_values.append(mlp)
        attn_values.append(attn)
        full_values.append(full)
        valid_counts.append(n_valid)

    pipeline = data["phase340_pipeline"]["aggs"]
    identity_name = IDENTITY_BLOCK[model]
    identity = fnum(pipeline.get(identity_name, {}).get("mean"))
    identity_std = fnum(pipeline.get(identity_name, {}).get("std"))
    combined = fnum(pipeline.get("identity+compute", {}).get("mean"))
    combined_std = fnum(pipeline.get("identity+compute", {}).get("std"))

    category = data["phase341b_category_probe"]
    category_accs = [fnum(row.get("accuracy")) for row in category.values()]
    category_chances = [fnum(row.get("chance")) for row in category.values()]
    max_category_acc = max(category_accs) if category_accs else float("nan")
    mean_category_acc = mean(category_accs) if category_accs else float("nan")
    mean_chance = mean(category_chances) if category_chances else float("nan")

    validity = {
        "all_baselines_mlp_gt_attn": all(mlp_gt_attn),
        "identity_recovery_ge_95": identity >= 95.0,
        "min_valid_pairs_ge_12": min(valid_counts) >= 12 if valid_counts else False,
        "category_probe_above_chance": mean_category_acc > mean_chance if category_accs else False,
    }
    validity["passes_core_validation"] = all(validity.values())

    return {
        "model": model,
        "baseline_rows": baseline_rows,
        "mean_mlp": mean(mlp_values),
        "mean_attn": mean(attn_values),
        "mean_full": mean(full_values),
        "identity_block": identity_name,
        "identity_recovery": identity,
        "identity_std": identity_std,
        "identity_plus_compute": combined,
        "identity_plus_compute_std": combined_std,
        "mean_category_accuracy": mean_category_acc,
        "max_category_accuracy": max_category_acc,
        "mean_category_chance": mean_chance,
        "validity": validity,
    }


def write_markdown(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# Phase342 Object-Property Binding Validation Summary", ""]
    for summary in summaries:
        lines.extend([
            f"## {summary['model']}",
            "",
            f"core_validation={summary['validity']['passes_core_validation']}",
            "",
            "| baseline | MLP | Attention | Full | n_valid | MLP>Attn |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in summary["baseline_rows"]:
            lines.append(
                f"| {row['baseline']} | {row['mlp']:.1f} | {row['attn']:.1f} | "
                f"{row['full']:.1f} | {row['n_valid']} | {row['mlp_gt_attn']} |"
            )
        lines.extend([
            "",
            f"mean_mlp={summary['mean_mlp']:.2f}",
            f"mean_attn={summary['mean_attn']:.2f}",
            f"mean_full={summary['mean_full']:.2f}",
            f"identity_block={summary['identity_block']}",
            f"identity_recovery={summary['identity_recovery']:.2f}",
            f"identity_plus_compute={summary['identity_plus_compute']:.2f}",
            f"mean_category_accuracy={summary['mean_category_accuracy']:.4f}",
            f"mean_category_chance={summary['mean_category_chance']:.4f}",
            "",
            "### Validity",
            "",
        ])
        for key, value in summary["validity"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    (output_dir / "PHASE342_OBJECT_BINDING_VALIDATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_model(load_result(input_dir, model)) for model in ["qwen3", "glm4", "deepseek7b"]]
    out = {
        "models": {item["model"]: item for item in summaries},
        "all_models_pass_core_validation": all(item["validity"]["passes_core_validation"] for item in summaries),
    }
    (output_dir / "phase342_object_binding_validation_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir, summaries)
    print(f"wrote {output_dir / 'phase342_object_binding_validation_summary.json'}")
    print(f"wrote {output_dir / 'PHASE342_OBJECT_BINDING_VALIDATION_SUMMARY.md'}")


if __name__ == "__main__":
    main()
