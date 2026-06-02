from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
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

    files = sorted(input_dir.glob("*_phase304_role_query_template_calibration.json"))
    lines = ["# Phase304 Role Query Template Calibration Summary", ""]
    summary: dict[str, Any] = {"models": {}}
    for path in files:
        data = load_json(path)
        model = data["model"]
        summary["models"][model] = data["summary"]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"rows={data['num_rows']} bases={data['num_bases']} "
            f"reliable_templates={data['summary']['num_reliable']} "
            f"nonfinite={data['summary']['nonfinite_rows']}"
        )
        lines.append("")

        reliable = sorted(
            data["summary"]["reliable_templates"],
            key=lambda row: (
                row["query_type"],
                not row["passes"],
                -float(row["min_state_accuracy"]),
                -float(row["min_state_mean_margin"]),
                row["template_id"],
            ),
        )
        lines.append("### Reliable Template Candidates")
        lines.append("")
        lines.append("| query | template | pass | min_state_acc | min_state_margin | min_order_acc |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in reliable:
            if row["passes"] or float(row["min_state_accuracy"]) >= 0.75:
                lines.append(
                    f"| {row['query_type']} | {row['template_id']} | {row['passes']} | "
                    f"{fmt(row['min_state_accuracy'])} | {fmt(row['min_state_mean_margin'])} | "
                    f"{fmt(row['min_order_accuracy'])} |"
                )
        lines.append("")

        bad_state = sorted(
            data["summary"]["by_state"],
            key=lambda row: (float(row["accuracy"]), float(row["mean_margin"])),
        )[:12]
        lines.append("### Weakest State Rows")
        lines.append("")
        lines.append("| query | template | state | acc | margin | n |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in bad_state:
            lines.append(
                f"| {row['query_type']} | {row['template_id']} | {row['state']} | "
                f"{fmt(row['accuracy'])} | {fmt(row['mean_margin'])} | {row['n']} |"
            )
        lines.append("")

    out_md = output_dir / "ROLE_QUERY_TEMPLATE_CALIBRATION_SUMMARY.md"
    out_json = output_dir / "role_query_template_calibration_summary.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
