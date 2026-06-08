from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"models": {}}
    lines = ["# Phase67 Object-Attribute Path Map Summary", ""]

    for model in MODELS:
        data = load(input_dir / f"{model}_phase67_object_attribute_path_map.json")
        lines += [f"## {model}", ""]
        if data is None:
            lines += ["missing", ""]
            continue
        rows = list(data.get("rows", []))
        entries = []
        for key, val in data.get("summary", {}).items():
            entry = {"path": key, **val}
            entries.append(entry)
        entries.sort(
            key=lambda x: (
                float(x.get("rank_flip_rate", 0.0)),
                float(x.get("mean_progress", 0.0)),
                float(x.get("improve_rate", 0.0)),
            ),
            reverse=True,
        )
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(rows),
            "layers": data.get("layers"),
            "top_paths": entries[:20],
        }
        lines += [
            f"items={data.get('num_items')}, rows={len(rows)}, layers={data.get('layers')}",
            "",
            "| rank | path | n | mean_progress | rank_flip_rate | improve_rate | clean_top1 | corrupt_not_top1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for i, e in enumerate(entries[:20], 1):
            lines.append(
                f"| {i} | {e['path']} | {e.get('n', 0)} | "
                f"{float(e.get('mean_progress', 0.0)):.4f} | "
                f"{float(e.get('rank_flip_rate', 0.0)):.4f} | "
                f"{float(e.get('improve_rate', 0.0)):.4f} | "
                f"{float(e.get('clean_top1_rate', 0.0)):.4f} | "
                f"{float(e.get('corrupt_not_top1_rate', 0.0)):.4f} |"
            )
        lines.append("")

    out_json = output_dir / "phase67_object_attribute_path_summary.json"
    out_md = output_dir / "PHASE67_OBJECT_ATTRIBUTE_PATH_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
