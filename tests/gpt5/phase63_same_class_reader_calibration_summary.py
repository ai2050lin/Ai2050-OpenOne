from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_model(input_dir: Path, model: str) -> dict[str, Any] | None:
    path = input_dir / f"{model}_phase63_same_class_reader_calibration.json"
    if not path.exists():
        shard_paths = sorted(input_dir.glob(f"{model}_phase63_same_class_reader_calibration_shard*.json"))
        if not shard_paths:
            return None
        shards = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths]
        rows = []
        for shard in shards:
            rows.extend(shard.get("rows", []))
        if not rows:
            return None
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from phase63_same_class_reader_calibration import summarize  # type: ignore

        base = shards[0]
        return {
            "phase": 63,
            "model": model,
            "max_cases": base.get("max_cases"),
            "num_cases": sum(int(s.get("num_cases", 0)) for s in shards),
            "num_rows": len(rows),
            "merged_shards": [str(p) for p in shard_paths],
            "min_accuracy": base.get("min_accuracy", 0.9),
            "min_group_accuracy": base.get("min_group_accuracy", 0.85),
            "summary": summarize(rows, base.get("min_accuracy", 0.9), base.get("min_group_accuracy", 0.85)),
            "rows": rows,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {m: load_model(input_dir, m) for m in MODELS}
    summary: dict[str, Any] = {"models": {}}
    lines: list[str] = ["# Phase63 Same Class Reader Calibration Summary", ""]
    for model in MODELS:
        data = loaded[model]
        if data is None:
            lines += [f"## {model}", "", "missing", ""]
            continue
        readers = data["summary"]["by_reader"]
        summary["models"][model] = {
            "num_cases": data["num_cases"],
            "num_rows": data["num_rows"],
            "by_reader": readers,
            "passed_readers": [r for r in readers if r.get("passes_gate")],
        }
        lines += [f"## {model}", "", f"cases={data['num_cases']}, rows={data['num_rows']}", ""]
        lines.append("| rank | reader | acc | min_ctx | min_variant | margin | abs_margin | pass |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(readers, 1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        r["reader_id"],
                        fmt(r["accuracy"]),
                        fmt(r["min_context_accuracy"]),
                        fmt(r["min_variant_accuracy"]),
                        fmt(r["mean_margin"]),
                        fmt(r["mean_abs_margin"]),
                        "yes" if r.get("passes_gate") else "no",
                    ]
                )
                + " |"
            )
        lines.append("")

    cross: dict[str, list[dict[str, Any]]] = {}
    for model, data in loaded.items():
        if data is None:
            continue
        for r in data["summary"]["by_reader"]:
            cross.setdefault(r["reader_id"], []).append({"model": model, **r})
    cross_rows = []
    for reader_id, rows in sorted(cross.items()):
        if len(rows) != 3:
            continue
        cross_rows.append(
            {
                "reader_id": reader_id,
                "mean_accuracy": sum(r["accuracy"] for r in rows) / 3.0,
                "min_accuracy": min(r["accuracy"] for r in rows),
                "min_context_accuracy": min(r["min_context_accuracy"] for r in rows),
                "min_variant_accuracy": min(r["min_variant_accuracy"] for r in rows),
                "all_pass": all(r.get("passes_gate") for r in rows),
                "models": rows,
            }
        )
    cross_rows.sort(key=lambda x: (x["all_pass"], x["min_accuracy"], x["mean_accuracy"]), reverse=True)
    summary["cross_model_readers"] = cross_rows
    lines += ["## Cross Model", ""]
    lines.append("| rank | reader | mean_acc | min_acc | min_ctx | min_variant | all_pass |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(cross_rows, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    r["reader_id"],
                    fmt(r["mean_accuracy"]),
                    fmt(r["min_accuracy"]),
                    fmt(r["min_context_accuracy"]),
                    fmt(r["min_variant_accuracy"]),
                    "yes" if r["all_pass"] else "no",
                ]
            )
            + " |"
        )

    (output_dir / "phase63_same_class_reader_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "PHASE63_SAME_CLASS_READER_CALIBRATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'phase63_same_class_reader_calibration_summary.json'}")
    print(f"wrote {output_dir / 'PHASE63_SAME_CLASS_READER_CALIBRATION_SUMMARY.md'}")


if __name__ == "__main__":
    main()
