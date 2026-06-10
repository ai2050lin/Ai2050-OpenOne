from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sorted_entries(d: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"key": k, **v} for k, v in d.items()]
    rows.sort(key=lambda x: (-float(x.get("eligible_clean_drop", 0.0)), -float(x.get("eligible_restore_gain", 0.0))))
    return rows


def table(entries: list[dict[str, Any]], limit: int = 80) -> str:
    lines = [
        "| rank | key | n | eligible | elig_drop | elig_restore_gain | elig_restore_gap | base_margin | erase_margin | restore_margin | erase_top1 | restore_top1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {int(e.get('n', 0))} | {int(e.get('eligible_n', 0))} | "
            f"{float(e.get('eligible_clean_drop', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_gain', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_gap', 0.0)):.4f} | "
            f"{float(e.get('eligible_base_margin', 0.0)):.4f} | "
            f"{float(e.get('eligible_erase_margin', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_margin', 0.0)):.4f} | "
            f"{float(e.get('eligible_erase_top1', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_top1', 0.0)):.4f} |"
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
    summary: dict[str, Any] = {"models": {}}
    md_lines = ["# Phase84 Clean Suffix Erase/Restore Summary"]
    for model in MODELS:
        path = input_dir / f"{model}_phase84_clean_suffix_erase_restore.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data.get("summary", {})
        by_condition = sorted_entries(s.get("by_condition", {}))
        by_path = sorted_entries(s.get("by_condition_path", {}))
        by_relation = sorted_entries(s.get("by_condition_relation", {}))
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "max_basis_items": data.get("max_basis_items"),
            "rows": len(data.get("rows", [])),
            "layer_pairs": data.get("layer_pairs"),
            "contrast_rank": data.get("contrast_rank"),
            "component_rank": data.get("component_rank"),
            "module": data.get("module"),
            "relations": data.get("relations"),
            "by_condition": by_condition,
            "top_condition_paths": by_path[:80],
            "top_condition_relations": by_relation[:120],
        }
        md_lines.append(f"\n## {model}\n")
        md_lines.append(
            f"items={data.get('num_items')}, basis_items={data.get('max_basis_items')}, "
            f"rows={len(data.get('rows', []))}, layer_pairs={data.get('layer_pairs')}\n"
            f"module={data.get('module')}, contrast_rank={data.get('contrast_rank')}, component_rank={data.get('component_rank')}, "
            f"relations={data.get('relations')}\n"
        )
        md_lines.append("### By condition\n")
        md_lines.append(table(by_condition, 80))
        md_lines.append("\n### Top condition paths\n")
        md_lines.append(table(by_path, 80))
        md_lines.append("\n### Top condition relations\n")
        md_lines.append(table(by_relation, 120))
    out_json = output_dir / "phase84_clean_suffix_erase_restore_summary.json"
    out_md = output_dir / "PHASE84_CLEAN_SUFFIX_ERASE_RESTORE_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
