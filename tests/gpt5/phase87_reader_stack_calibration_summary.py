from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sort_entries(entries: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"key": k, **v} for k, v in entries.items()]
    rows.sort(
        key=lambda x: (
            -float(x.get("closed_top1", 0.0)),
            -float(x.get("choice_top1", 0.0)),
            -float(x.get("open_family_overlap_hit", 0.0)),
            -float(x.get("closed_margin", 0.0)),
        )
    )
    return rows


def table(entries: list[dict[str, Any]], limit: int = 80) -> str:
    lines = [
        "| rank | key | n | closed_top1 | closed_margin | choice_top1 | choice_no_first | choice_rot | choice_last | choice_valid | open_subset | open_family | open_format_bad |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {int(e.get('n', 0))} | "
            f"{float(e.get('closed_top1', 0.0)):.4f} | "
            f"{float(e.get('closed_margin', 0.0)):.4f} | "
            f"{float(e.get('choice_top1', 0.0)):.4f} | "
            f"{float(e.get('choice_no_target_first_top1', 0.0)):.4f} | "
            f"{float(e.get('choice_rotating_top1', 0.0)):.4f} | "
            f"{float(e.get('choice_target_last_top1', 0.0)):.4f} | "
            f"{float(e.get('choice_valid', 0.0)):.4f} | "
            f"{float(e.get('open_word_subset_hit', 0.0)):.4f} | "
            f"{float(e.get('open_family_overlap_hit', 0.0)):.4f} | "
            f"{float(e.get('open_format_violation', 0.0)):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"models": {}, "cross_model": {}}
    md_lines = ["# Phase87 Reader Stack Calibration Summary"]
    cross_reader: dict[str, list[dict[str, Any]]] = {}
    cross_template: dict[str, list[dict[str, Any]]] = {}

    for model in MODELS:
        path = input_dir / f"{model}_phase87_reader_stack_calibration.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("summary", {})
        model_summary: dict[str, Any] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "relations": data.get("relations"),
            "choice_templates": data.get("choice_templates"),
            "open_templates": data.get("open_templates"),
            "by_reader": sort_entries(s.get("by_reader", {})),
            "by_relation": sort_entries(s.get("by_relation", {})),
            "by_reader_template": sort_entries(s.get("by_reader_template", {})),
            "by_choice_order": sort_entries(s.get("by_choice_order", {})),
            "top_template_relation": sort_entries(s.get("by_template_relation", {}))[:160],
            "samples": data.get("samples", [])[:32],
        }
        summary["models"][model] = model_summary
        for e in model_summary["by_reader"]:
            cross_reader.setdefault(e["key"], []).append({"model": model, **e})
        for e in model_summary["by_reader_template"]:
            cross_template.setdefault(e["key"], []).append({"model": model, **e})

        md_lines.append(f"\n## {model}\n")
        md_lines.append(
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, "
            f"relations={data.get('relations')}\n"
        )
        md_lines.append("### By reader\n")
        md_lines.append(table(model_summary["by_reader"], 40))
        md_lines.append("\n### By reader template\n")
        md_lines.append(table(model_summary["by_reader_template"], 80))
        md_lines.append("\n### By choice order\n")
        md_lines.append(table(model_summary["by_choice_order"], 80))
        md_lines.append("\n### By relation\n")
        md_lines.append(table(model_summary["by_relation"], 80))

    def avg_entries(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        out = []
        metrics = [
            "closed_top1", "closed_margin", "choice_top1", "choice_valid",
            "choice_target_letter_rate", "open_word_subset_hit",
            "choice_no_target_first_top1", "choice_rotating_top1",
            "choice_target_last_top1",
            "open_family_overlap_hit", "open_format_violation",
        ]
        for key, vals in groups.items():
            item = {"key": key, "n": sum(int(v.get("n", 0)) for v in vals), "models": ",".join(v["model"] for v in vals)}
            for metric in metrics:
                item[metric] = sum(float(v.get(metric, 0.0)) for v in vals) / len(vals)
            out.append(item)
        out.sort(key=lambda x: (-x.get("closed_top1", 0.0), -x.get("choice_top1", 0.0), -x.get("open_family_overlap_hit", 0.0)))
        return out

    summary["cross_model"]["by_reader"] = avg_entries(cross_reader)
    summary["cross_model"]["by_reader_template"] = avg_entries(cross_template)
    md_lines.append("\n## Cross Model By Reader\n")
    md_lines.append(table(summary["cross_model"]["by_reader"], 40))
    md_lines.append("\n## Cross Model By Reader Template\n")
    md_lines.append(table(summary["cross_model"]["by_reader_template"], 120))

    out_json = output_dir / "phase87_reader_stack_calibration_summary.json"
    out_md = output_dir / "PHASE87_READER_STACK_CALIBRATION_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
