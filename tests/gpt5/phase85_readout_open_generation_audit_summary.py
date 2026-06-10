from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def sorted_entries(d: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"key": k, **v} for k, v in d.items()]
    rows.sort(key=lambda x: (-float(x.get("eligible_prefix_drop", 0.0)), -float(x.get("eligible_changed", 0.0))))
    return rows


def table(entries: list[dict[str, Any]], limit: int = 80) -> str:
    lines = [
        "| rank | key | n | eligible | prefix_hit | eligible_prefix_hit | eligible_prefix_drop | changed | eligible_changed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, e in enumerate(entries[:limit], 1):
        lines.append(
            f"| {i} | {e['key']} | {int(e.get('n', 0))} | {int(e.get('eligible_n', 0))} | "
            f"{float(e.get('prefix_hit', 0.0)):.4f} | "
            f"{float(e.get('eligible_prefix_hit', 0.0)):.4f} | "
            f"{float(e.get('eligible_prefix_drop', 0.0)):.4f} | "
            f"{float(e.get('changed', 0.0)):.4f} | "
            f"{float(e.get('eligible_changed', 0.0)):.4f} |"
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
    md_lines = ["# Phase85 Readout Open Generation Audit Summary"]
    for model in MODELS:
        path = input_dir / f"{model}_phase85_readout_open_generation_audit.json"
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
            "audit_layers": data.get("audit_layers"),
            "source_layer_pairs": data.get("source_layer_pairs"),
            "component_rank": data.get("component_rank"),
            "module": data.get("module"),
            "max_new_tokens": data.get("max_new_tokens"),
            "relations": data.get("relations"),
            "by_condition": by_condition,
            "top_condition_paths": by_path[:80],
            "top_condition_relations": by_relation[:120],
        }
        md_lines.append(f"\n## {model}\n")
        md_lines.append(
            f"items={data.get('num_items')}, basis_items={data.get('max_basis_items')}, "
            f"rows={len(data.get('rows', []))}, audit_layers={data.get('audit_layers')}\n"
            f"module={data.get('module')}, component_rank={data.get('component_rank')}, "
            f"max_new_tokens={data.get('max_new_tokens')}, relations={data.get('relations')}\n"
        )
        md_lines.append("### By condition\n")
        md_lines.append(table(by_condition, 80))
        md_lines.append("\n### Top condition paths\n")
        md_lines.append(table(by_path, 80))
        md_lines.append("\n### Top condition relations\n")
        md_lines.append(table(by_relation, 120))
    out_json = output_dir / "phase85_readout_open_generation_audit_summary.json"
    out_md = output_dir / "PHASE85_READOUT_OPEN_GENERATION_AUDIT_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
