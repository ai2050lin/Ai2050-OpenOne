from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sorted_entries(mapping: dict[str, Any], limit: int = 120) -> list[dict[str, Any]]:
    entries = [{"key": k, **v} for k, v in mapping.items()]
    entries.sort(
        key=lambda x: (
            float(x.get("eligible_clean_drop", 0.0)),
            float(x.get("eligible_matched_gain", 0.0)),
            float(x.get("eligible_patched_matched_top1", 0.0)),
        ),
        reverse=True,
    )
    return entries[:limit]


def add_table(lines: list[str], title: str, entries: list[dict[str, Any]], limit: int = 40) -> None:
    lines += [
        f"### {title}",
        "",
        "| rank | key | n | eligible | elig_clean_drop | elig_matched_gain | elig_clean_after | elig_matched_after | elig_clean_top1 | elig_matched_top1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {e.get('n', 0)} | {e.get('eligible_n', 0)} | "
            f"{float(e.get('eligible_clean_drop', 0.0)):.4f} | "
            f"{float(e.get('eligible_matched_gain', 0.0)):.4f} | "
            f"{float(e.get('eligible_clean_margin_after', 0.0)):.4f} | "
            f"{float(e.get('eligible_matched_margin_after', 0.0)):.4f} | "
            f"{float(e.get('eligible_patched_clean_top1', 0.0)):.4f} | "
            f"{float(e.get('eligible_patched_matched_top1', 0.0)):.4f} |"
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"models": {}}
    lines = ["# Phase76 Object-Frame Joint Closure Summary", ""]
    for model in MODELS:
        path = input_dir / f"{model}_phase76_object_frame_joint_closure.json"
        lines += [f"## {model}", ""]
        if not path.exists():
            lines += ["missing", ""]
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sm = data.get("summary", {})
        by_condition = sorted_entries(sm.get("by_condition", {}), 40)
        by_condition_path = sorted_entries(sm.get("by_condition_path", {}), 80)
        by_condition_relation = sorted_entries(sm.get("by_condition_relation", {}), 120)
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "layer_pairs": data.get("layer_pairs"),
            "by_condition": by_condition,
            "top_condition_paths": by_condition_path,
            "top_condition_relations": by_condition_relation,
        }
        lines += [
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, layer_pairs={data.get('layer_pairs')}",
            "",
        ]
        add_table(lines, "By condition", by_condition, 40)
        add_table(lines, "Top condition paths", by_condition_path, 40)
        add_table(lines, "Top condition relations", by_condition_relation, 60)

    out_json = output_dir / "phase76_object_frame_joint_closure_summary.json"
    out_md = output_dir / "PHASE76_OBJECT_FRAME_JOINT_CLOSURE_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
