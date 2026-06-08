from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    ranked = data.get("token_path_summary", {}).get("ranked_paths", [])
    return {
        "model": data.get("model"),
        "attn_implementations": data.get("attn_implementations"),
        "n_cases": data.get("n_cases"),
        "accuracy": data.get("baseline", {}).get("accuracy"),
        "mean_abs_margin": data.get("baseline", {}).get("mean_abs_margin"),
        "top_paths": ranked[:20],
        "elapsed_sec": data.get("elapsed_sec"),
    }


def write_markdown(out_dir: Path, out: dict[str, Any]) -> None:
    lines = ["# Phase59 Temporal Order Token Path Summary", ""]
    for model, s in out["models"].items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"accuracy={s['accuracy']:.4f}, mean_abs_margin={s['mean_abs_margin']:.4f}, n_cases={s['n_cases']}")
        lines.append("")
        lines.append("| rank | path | mean_signed_projection | std | n |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, row in enumerate(s["top_paths"][:12], 1):
            lines.append(
                f"| {i} | {row['path']} | {row['mean_signed_projection']:.4f} | "
                f"{row['std']:.4f} | {row['n']} |"
            )
        lines.append("")
    out_dir.joinpath("PHASE59_TEMPORAL_ORDER_TOKEN_PATH_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for model in MODELS:
        models[model] = summarize(load_json(input_dir / f"{model}_phase59_temporal_order_token_path.json"))
    out = {"models": models}
    out_dir.joinpath("phase59_temporal_order_token_path_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    write_markdown(out_dir, out)
    print(f"wrote {out_dir / 'phase59_temporal_order_token_path_summary.json'}")
    print(f"wrote {out_dir / 'PHASE59_TEMPORAL_ORDER_TOKEN_PATH_SUMMARY.md'}")


if __name__ == "__main__":
    main()
