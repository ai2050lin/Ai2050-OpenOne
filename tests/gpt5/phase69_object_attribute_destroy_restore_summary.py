from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"models": {}}
    lines = ["# Phase69 Object-Attribute Destroy-Restore Summary", ""]
    for model in MODELS:
        path = input_dir / f"{model}_phase69_object_attribute_destroy_restore.json"
        lines += [f"## {model}", ""]
        if not path.exists():
            lines += ["missing", ""]
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [{"path": k, **v} for k, v in data.get("summary", {}).items()]
        entries.sort(
            key=lambda x: (
                float(x.get("eligible_restore_gain", 0.0)),
                float(x.get("eligible_destroy_drop", 0.0)),
                float(x.get("eligible_restore_top1", 0.0)) - float(x.get("eligible_destroy_top1", 0.0)),
            ),
            reverse=True,
        )
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "layer_pairs": data.get("layer_pairs"),
            "top_paths": entries[:30],
        }
        lines += [
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, layer_pairs={data.get('layer_pairs')}",
            "",
            "| rank | path | n | eligible | destroy_drop | restore_gain | restore_gap | elig_destroy_drop | elig_restore_gain | elig_restore_gap | clean_top1 | destroy_top1 | restore_top1 | elig_destroy_top1 | elig_restore_top1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, e in enumerate(entries[:25], 1):
            lines.append(
                f"| {i} | {e['path']} | {e.get('n', 0)} | {e.get('eligible_n', 0)} | "
                f"{float(e.get('destroy_drop', 0.0)):.4f} | "
                f"{float(e.get('restore_gain', 0.0)):.4f} | "
                f"{float(e.get('restore_to_clean_gap', 0.0)):.4f} | "
                f"{float(e.get('eligible_destroy_drop', 0.0)):.4f} | "
                f"{float(e.get('eligible_restore_gain', 0.0)):.4f} | "
                f"{float(e.get('eligible_restore_to_clean_gap', 0.0)):.4f} | "
                f"{float(e.get('clean_top1', 0.0)):.4f} | "
                f"{float(e.get('destroy_top1', 0.0)):.4f} | "
                f"{float(e.get('restore_top1', 0.0)):.4f} | "
                f"{float(e.get('eligible_destroy_top1', 0.0)):.4f} | "
                f"{float(e.get('eligible_restore_top1', 0.0)):.4f} |"
            )
        lines.append("")

    out_json = output_dir / "phase69_object_attribute_destroy_restore_summary.json"
    out_md = output_dir / "PHASE69_OBJECT_ATTRIBUTE_DESTROY_RESTORE_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
