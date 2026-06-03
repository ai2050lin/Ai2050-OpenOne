from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def top_pairs(matrix: dict[str, dict[str, float]], reverse: bool = True, n: int = 12) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    seen = set()
    for a, bs in matrix.items():
        for b, val in bs.items():
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            rows.append((a, b, fnum(val)))
    return sorted(rows, key=lambda x: x[2], reverse=reverse)[:n]


def summarize_model(data: dict[str, Any]) -> dict[str, Any]:
    relations = data.get("relations", {})
    ranked = sorted(
        [
            {
                "relation": rel,
                "net_gross": fnum(r.get("net_gross_mean")),
                "balance": fnum(r.get("balance_mean")),
                "interaction_frac": fnum(r.get("interaction_frac")),
                "n": int(fnum(r.get("n_observations"))),
            }
            for rel, r in relations.items()
        ],
        key=lambda x: x["net_gross"],
        reverse=True,
    )
    matrix = data.get("relation_similarity_matrix", {})
    return {
        "model": data.get("model"),
        "attn_implementations": data.get("attn_implementations"),
        "target_layers": data.get("target_layers"),
        "ranked_relations": ranked,
        "top_similarity_pairs": top_pairs(matrix, True, 12),
        "bottom_similarity_pairs": top_pairs(matrix, False, 12),
        "mean_interaction_frac": mean([x["interaction_frac"] for x in ranked]) if ranked else 0.0,
        "mean_net_gross": mean([x["net_gross"] for x in ranked]) if ranked else 0.0,
        "elapsed_sec": fnum(data.get("elapsed_sec")),
    }


def write_markdown(output_dir: Path, summaries: dict[str, Any]) -> None:
    lines = ["# Phase57 Global Relation Path Matrix Summary", ""]
    for model, s in summaries["models"].items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"attn_implementations: `{s['attn_implementations']}`")
        lines.append("")
        lines.append("| rank | relation | net/gross | balance | interaction | n |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, row in enumerate(s["ranked_relations"], 1):
            lines.append(
                f"| {i} | {row['relation']} | {row['net_gross']:.4f} | "
                f"{row['balance']:.4f} | {row['interaction_frac']:.4f} | {row['n']} |"
            )
        lines.append("")
        lines.append("### Top Similarity Pairs")
        lines.append("")
        lines.append("| relation_a | relation_b | cosine |")
        lines.append("|---|---|---:|")
        for a, b, val in s["top_similarity_pairs"][:8]:
            lines.append(f"| {a} | {b} | {val:.4f} |")
        lines.append("")
        lines.append("### Bottom Similarity Pairs")
        lines.append("")
        lines.append("| relation_a | relation_b | cosine |")
        lines.append("|---|---|---:|")
        for a, b, val in s["bottom_similarity_pairs"][:8]:
            lines.append(f"| {a} | {b} | {val:.4f} |")
        lines.append("")
    output_dir.joinpath("PHASE57_GLOBAL_RELATION_PATH_MATRIX_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model in MODELS:
        path = input_dir / f"{model}_phase57_global_relation_path_matrix.json"
        summaries[model] = summarize_model(load_json(path))
    out = {"models": summaries}
    output_dir.joinpath("phase57_global_relation_path_matrix_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir, out)
    print(f"wrote {output_dir / 'phase57_global_relation_path_matrix_summary.json'}")
    print(f"wrote {output_dir / 'PHASE57_GLOBAL_RELATION_PATH_MATRIX_SUMMARY.md'}")


if __name__ == "__main__":
    main()
