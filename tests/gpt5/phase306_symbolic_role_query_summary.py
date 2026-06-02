from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*_phase306_symbolic_role_query_calibration.json"))
    lines = ["# Phase306 Symbolic Role Query Calibration Summary", ""]
    summary = {"models": {}}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
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
                row["entity_style"],
                row["answer_style"],
                row["query_type"],
                not row["passes"],
                -float(row["min_state_accuracy"]),
                -float(row["min_option_accuracy"]),
                -float(row["min_state_mean_margin"]),
            ),
        )
        lines.append("### Template Candidates")
        lines.append("")
        lines.append("| entity | answer | query | template | pass | min_state_acc | min_option_acc | min_state_margin |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in reliable:
            if row["passes"] or float(row["min_state_accuracy"]) >= 0.75:
                lines.append(
                    f"| {row['entity_style']} | {row['answer_style']} | {row['query_type']} | "
                    f"{row['template_id']} | {row['passes']} | {fmt(row['min_state_accuracy'])} | "
                    f"{fmt(row['min_option_accuracy'])} | {fmt(row['min_state_mean_margin'])} |"
                )
        lines.append("")

        weak = sorted(
            data["summary"]["by_state"],
            key=lambda row: (float(row["accuracy"]), float(row["mean_margin"])),
        )[:16]
        lines.append("### Weakest State Rows")
        lines.append("")
        lines.append("| entity | answer | query | template | state | acc | margin | n |")
        lines.append("|---|---|---|---|---|---:|---:|---:|")
        for row in weak:
            lines.append(
                f"| {row['entity_style']} | {row['answer_style']} | {row['query_type']} | "
                f"{row['template_id']} | {row['state']} | {fmt(row['accuracy'])} | "
                f"{fmt(row['mean_margin'])} | {row['n']} |"
            )
        lines.append("")

    out_md = output_dir / "SYMBOLIC_ROLE_QUERY_CALIBRATION_SUMMARY.md"
    out_json = output_dir / "symbolic_role_query_calibration_summary.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
