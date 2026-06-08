from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(input_dir: Path, model: str) -> dict[str, Any] | None:
    path = input_dir / f"{model}_phase65_object_attribute_compat_decomposition.json"
    if not path.exists():
        matches = sorted(input_dir.glob(f"{model}_phase65_object_attribute_compat_decomposition*.json"))
        matches = [p for p in matches if not p.name.endswith(".partial.json")]
        if not matches:
            return None
        path = matches[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {m: load(input_dir, m) for m in MODELS}
    summary: dict[str, Any] = {"models": {}}
    lines = ["# Phase65 Object-Attribute Compatibility Decomposition Summary", ""]
    for model, data in loaded.items():
        if data is None:
            lines += [f"## {model}", "", "missing", ""]
            continue
        lines += [f"## {model}", "", f"layers={data['layers']}, pairs={data['num_pairs']}", ""]
        lines.append("| layer | full | neutral_ideal | L1_FULL | L2cf_FULL | OBJcf_FULL |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        model_rows = []
        for layer, layer_sum in data["layer_summaries"].items():
            counts = layer_sum["counts"]
            row = {
                "layer": int(layer),
                "full": int(layer_sum["full_symmetric_count"]),
                "neutral_ideal": int(layer_sum["neutral_ideal_count"]),
                "L1_FULL": int(counts.get("L1:FULL", 0)),
                "L2cf_FULL": int(counts.get("L2_cf:FULL", 0)),
                "OBJcf_FULL": int(counts.get("OBJ_cf:FULL", 0)),
            }
            model_rows.append(row)
        model_rows.sort(key=lambda x: x["layer"])
        for row in model_rows:
            lines.append(
                f"| {row['layer']} | {row['full']} | {row['neutral_ideal']} | "
                f"{row['L1_FULL']} | {row['L2cf_FULL']} | {row['OBJcf_FULL']} |"
            )
        lines.append("")
        best = sorted(model_rows, key=lambda x: (x["full"], -x["neutral_ideal"]), reverse=True)[:3]
        summary["models"][model] = {
            "layers": data["layers"],
            "num_pairs": data["num_pairs"],
            "rows": model_rows,
            "best_layers": best,
        }

        by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for layer, obj_results in data["per_layer"].items():
            for row in obj_results.values():
                by_cat[row["cat"]][f"{row['version']}:{row['symmetric']}"] += 1
        summary["models"][model]["by_cat"] = {k: dict(v) for k, v in by_cat.items()}

    out_json = output_dir / "phase65_object_attribute_compat_summary.json"
    out_md = output_dir / "PHASE65_OBJECT_ATTRIBUTE_COMPAT_SUMMARY.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
