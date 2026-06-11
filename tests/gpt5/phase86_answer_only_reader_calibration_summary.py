from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sorted_entries(d: dict[str, Any], key_metric: str = "family_overlap_hit") -> list[dict[str, Any]]:
    rows = [{"key": k, **v} for k, v in d.items()]
    rows.sort(
        key=lambda x: (
            -float(x.get(key_metric, 0.0)),
            float(x.get("format_violation", 1.0)),
            -float(x.get("prefix_hit", 0.0)),
        )
    )
    return rows


def table(entries: list[dict[str, Any]], limit: int = 80) -> str:
    lines = [
        "| rank | key | n | exact | prefix | contains | word_subset | family_overlap | coverage | precision | short | format_violation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {int(e.get('n', 0))} | "
            f"{float(e.get('exact_hit', 0.0)):.4f} | "
            f"{float(e.get('prefix_hit', 0.0)):.4f} | "
            f"{float(e.get('contains_hit', 0.0)):.4f} | "
            f"{float(e.get('word_subset_hit', 0.0)):.4f} | "
            f"{float(e.get('family_overlap_hit', 0.0)):.4f} | "
            f"{float(e.get('target_word_coverage', 0.0)):.4f} | "
            f"{float(e.get('first_word_precision', 0.0)):.4f} | "
            f"{float(e.get('short_output', 0.0)):.4f} | "
            f"{float(e.get('format_violation', 0.0)):.4f} |"
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

    summary: dict[str, Any] = {"models": {}, "cross_model_template": {}}
    md_lines = ["# Phase86 Answer-Only Reader Calibration Summary"]
    template_accumulator: dict[str, list[dict[str, Any]]] = {}

    for model in MODELS:
        path = input_dir / f"{model}_phase86_answer_only_reader_calibration.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("summary", {})
        by_template = sorted_entries(s.get("by_template", {}))
        by_relation = sorted_entries(s.get("by_relation", {}))
        by_template_relation = sorted_entries(s.get("by_template_relation", {}))
        for entry in by_template:
            template_accumulator.setdefault(entry["key"], []).append({"model": model, **entry})
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "max_new_tokens": data.get("max_new_tokens"),
            "relations": data.get("relations"),
            "templates": data.get("templates"),
            "by_template": by_template,
            "by_relation": by_relation,
            "top_template_relation": by_template_relation[:160],
            "samples": data.get("samples", [])[:24],
        }
        md_lines.append(f"\n## {model}\n")
        md_lines.append(
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, "
            f"max_new_tokens={data.get('max_new_tokens')}, relations={data.get('relations')}\n"
        )
        md_lines.append("### By template\n")
        md_lines.append(table(by_template, 80))
        md_lines.append("\n### By relation\n")
        md_lines.append(table(by_relation, 80))
        md_lines.append("\n### Top template relation\n")
        md_lines.append(table(by_template_relation, 160))

    cross_rows = []
    for template, vals in template_accumulator.items():
        if not vals:
            continue
        cross_rows.append(
            {
                "key": template,
                "models": ",".join(v["model"] for v in vals),
                "n": sum(int(v.get("n", 0)) for v in vals),
                "exact_hit": sum(float(v.get("exact_hit", 0.0)) for v in vals) / len(vals),
                "prefix_hit": sum(float(v.get("prefix_hit", 0.0)) for v in vals) / len(vals),
                "contains_hit": sum(float(v.get("contains_hit", 0.0)) for v in vals) / len(vals),
                "word_subset_hit": sum(float(v.get("word_subset_hit", 0.0)) for v in vals) / len(vals),
                "family_overlap_hit": sum(float(v.get("family_overlap_hit", 0.0)) for v in vals) / len(vals),
                "target_word_coverage": sum(float(v.get("target_word_coverage", 0.0)) for v in vals) / len(vals),
                "first_word_precision": sum(float(v.get("first_word_precision", 0.0)) for v in vals) / len(vals),
                "short_output": sum(float(v.get("short_output", 0.0)) for v in vals) / len(vals),
                "format_violation": sum(float(v.get("format_violation", 0.0)) for v in vals) / len(vals),
                "per_model": vals,
            }
        )
    cross_rows.sort(key=lambda x: (-x["family_overlap_hit"], x["format_violation"], -x["word_subset_hit"]))
    summary["cross_model_template"] = cross_rows
    md_lines.append("\n## Cross Model Template Ranking\n")
    md_lines.append(table(cross_rows, 80))

    out_json = output_dir / "phase86_answer_only_reader_calibration_summary.json"
    out_md = output_dir / "PHASE86_ANSWER_ONLY_READER_CALIBRATION_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
