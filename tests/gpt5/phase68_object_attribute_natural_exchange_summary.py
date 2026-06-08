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
    lines = ["# Phase68 Object-Attribute Natural Exchange Summary", ""]
    for model in MODELS:
        path = input_dir / f"{model}_phase68_object_attribute_natural_exchange.json"
        lines += [f"## {model}", ""]
        if not path.exists():
            lines += ["missing", ""]
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [{"path": k, **v} for k, v in data.get("summary", {}).items()]
        entries.sort(
            key=lambda x: (
                float(x.get("eligible_net_delta", 0.0)),
                float(x.get("eligible_correct_top1", 0.0)) - float(x.get("eligible_control_top1", 0.0)),
                float(x.get("net_delta", 0.0)),
            ),
            reverse=True,
        )
        summary["models"][model] = {
            "num_items": data.get("num_items"),
            "rows": len(data.get("rows", [])),
            "layers": data.get("layers"),
            "top_paths": entries[:30],
        }
        lines += [
            f"items={data.get('num_items')}, rows={len(data.get('rows', []))}, layers={data.get('layers')}",
            "",
            "| rank | path | n | eligible | correct_delta | control_delta | net_delta | correct_flip | control_flip | elig_correct_top1 | elig_control_top1 | elig_net_delta |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, e in enumerate(entries[:25], 1):
            lines.append(
                f"| {i} | {e['path']} | {e.get('n', 0)} | {e.get('eligible_n', 0)} | "
                f"{float(e.get('correct_delta', 0.0)):.4f} | "
                f"{float(e.get('control_delta', 0.0)):.4f} | "
                f"{float(e.get('net_delta', 0.0)):.4f} | "
                f"{float(e.get('correct_flip_rate', 0.0)):.4f} | "
                f"{float(e.get('control_flip_rate', 0.0)):.4f} | "
                f"{float(e.get('eligible_correct_top1', 0.0)):.4f} | "
                f"{float(e.get('eligible_control_top1', 0.0)):.4f} | "
                f"{float(e.get('eligible_net_delta', 0.0)):.4f} |"
            )
        lines.append("")

    out_json = output_dir / "phase68_object_attribute_natural_exchange_summary.json"
    out_md = output_dir / "PHASE68_OBJECT_ATTRIBUTE_NATURAL_EXCHANGE_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
