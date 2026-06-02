from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def pair_rows(matrix: dict[str, dict[str, float]], reverse: bool, n: int) -> list[tuple[str, str, float]]:
    out = []
    seen = set()
    for a, bs in matrix.items():
        for b, v in bs.items():
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, fnum(v)))
    return sorted(out, key=lambda x: x[2], reverse=reverse)[:n]


def summarize_model(data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for key, item in data.get("subtypes", {}).items():
        real = item.get("real", {})
        rows.append({
            "key": key,
            "relation": item.get("relation"),
            "subtype": item.get("subtype"),
            "net_gross": fnum(real.get("net_gross_mean")),
            "random_mean": fnum(real.get("random_mean_net_gross")),
            "random_advantage": fnum(real.get("random_advantage")),
            "balance": fnum(real.get("balance_mean")),
            "interaction": fnum(real.get("interaction_frac")),
            "n": int(fnum(real.get("n"))),
        })
    rows_by_adv = sorted(rows, key=lambda x: x["random_advantage"], reverse=True)
    rows_by_net = sorted(rows, key=lambda x: x["net_gross"], reverse=True)
    matrix = data.get("subtype_similarity_matrix", {})
    return {
        "model": data.get("model"),
        "attn_implementations": data.get("attn_implementations"),
        "ranked_by_random_advantage": rows_by_adv,
        "ranked_by_net_gross": rows_by_net,
        "top_similarity_pairs": pair_rows(matrix, True, 12),
        "bottom_similarity_pairs": pair_rows(matrix, False, 12),
        "elapsed_sec": fnum(data.get("elapsed_sec")),
    }


def write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = ["# Phase58 Relation Subtype Random Controls Summary", ""]
    for model, s in summary["models"].items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"attn_implementations: `{s['attn_implementations']}`")
        lines.append("")
        lines.append("### Ranked By Random Advantage")
        lines.append("")
        lines.append("| rank | subtype | net/gross | random | advantage | interaction | n |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for i, row in enumerate(s["ranked_by_random_advantage"], 1):
            lines.append(
                f"| {i} | {row['key']} | {row['net_gross']:.4f} | {row['random_mean']:.4f} | "
                f"{row['random_advantage']:.4f} | {row['interaction']:.4f} | {row['n']} |"
            )
        lines.append("")
        lines.append("### Top Similarity Pairs")
        lines.append("")
        lines.append("| subtype_a | subtype_b | cosine |")
        lines.append("|---|---|---:|")
        for a, b, val in s["top_similarity_pairs"][:8]:
            lines.append(f"| {a} | {b} | {val:.4f} |")
        lines.append("")
        lines.append("### Bottom Similarity Pairs")
        lines.append("")
        lines.append("| subtype_a | subtype_b | cosine |")
        lines.append("|---|---|---:|")
        for a, b, val in s["bottom_similarity_pairs"][:8]:
            lines.append(f"| {a} | {b} | {val:.4f} |")
        lines.append("")
    out_dir.joinpath("PHASE58_RELATION_SUBTYPE_RANDOM_CONTROLS_SUMMARY.md").write_text(
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
    models = {}
    for model in MODELS:
        models[model] = summarize_model(load_json(input_dir / f"{model}_phase58_relation_subtype_random_controls.json"))
    out = {"models": models}
    output_dir.joinpath("phase58_relation_subtype_random_controls_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir, out)
    print(f"wrote {output_dir / 'phase58_relation_subtype_random_controls_summary.json'}")
    print(f"wrote {output_dir / 'PHASE58_RELATION_SUBTYPE_RANDOM_CONTROLS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
