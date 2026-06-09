from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sorted_entries(mapping: dict[str, Any], limit: int = 80) -> list[dict[str, Any]]:
    entries = [{"key": k, **v} for k, v in mapping.items()]
    entries.sort(
        key=lambda x: (
            float(x.get("eligible_destroy_drop", 0.0)),
            float(x.get("eligible_restore_gain", 0.0)),
            float(x.get("eligible_restore_top1", 0.0)),
        ),
        reverse=True,
    )
    return entries[:limit]


def add_table(lines: list[str], title: str, entries: list[dict[str, Any]], limit: int = 40) -> None:
    lines += [
        f"### {title}",
        "",
        "| rank | key | n | eligible | elig_destroy_drop | elig_restore_gain | elig_restore_gap | elig_destroy_top1 | elig_restore_top1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {e.get('n', 0)} | {e.get('eligible_n', 0)} | "
            f"{float(e.get('eligible_destroy_drop', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_gain', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_to_clean_gap', 0.0)):.4f} | "
            f"{float(e.get('eligible_destroy_top1', 0.0)):.4f} | "
            f"{float(e.get('eligible_restore_top1', 0.0)):.4f} |"
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
    lines = ["# Phase75 Relation-Frame Token Intervention Summary", ""]
    for model in MODELS:
        path = input_dir / f"{model}_phase75_relation_frame_token_intervention.json"
        lines += [f"## {model}", ""]
        if not path.exists():
            lines += ["missing", ""]
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sm = data.get("summary", {})
        by_control = sorted_entries(sm.get("by_control", {}), 20)
        by_control_path = sorted_entries(sm.get("by_control_path", {}), 80)
        by_control_relation = sorted_entries(sm.get("by_control_relation", {}), 80)
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "layer_pairs": data.get("layer_pairs"),
            "control_types": data.get("control_types"),
            "by_control": by_control,
            "top_control_paths": by_control_path,
            "top_control_relations": by_control_relation,
        }
        lines += [
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, layer_pairs={data.get('layer_pairs')}",
            f"control_types={data.get('control_types')}",
            "",
        ]
        add_table(lines, "By control type", by_control, 20)
        add_table(lines, "Top control paths", by_control_path, 30)
        add_table(lines, "Top control relations", by_control_relation, 40)

    out_json = output_dir / "phase75_relation_frame_token_intervention_summary.json"
    out_md = output_dir / "PHASE75_RELATION_FRAME_TOKEN_INTERVENTION_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
