#!/usr/bin/env python3
"""Cross-model summary for Phase529 robust positive controls."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase529_robust_positive_controls")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase529_{model}_robust_positive_controls.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def family_rank(d: dict[str, Any], family: str) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for cand, adm in d["admission"].items():
        if adm["family"] == family:
            rows.append((cand, adm))
    return sorted(rows, key=lambda x: float(x[1]["best_own_delta"]), reverse=True)


def add_family(lines: list[str], d: dict[str, Any], family: str) -> None:
    lines.append(f"### {family}")
    lines.append("")
    lines.append(
        "| candidate | own task | best alpha | own Δ | same-family max abs Δ | "
        "off-family max abs Δ | ratio | pass | readout % | semantic % |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cand, adm in family_rank(d, family):
        read = d["readout_alignment"][cand]
        lines.append(
            f"| {cand} | {adm['own_task']} | {float(adm['best_alpha']):.1f} | "
            f"{float(adm['best_own_delta']):+.4f} | "
            f"{float(adm['best_max_same_family_other_abs']):.4f} | "
            f"{float(adm['best_max_off_family_abs']):.4f} | "
            f"{float(adm['best_selectivity_ratio']):.4f} | "
            f"{'yes' if adm['passes_basic_gate'] else 'no'} | "
            f"{float(read['readout_norm_pct']):.2f} | {float(read['semantic_norm_pct']):.2f} |"
        )
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase529 Robust Positive Controls Summary", ""]

    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        for family in ["category", "color", "object"]:
            add_family(lines, d, family)

        passed = [c for c, a in d["admission"].items() if a["passes_basic_gate"]]
        best_color = family_rank(d, "color")[0]
        best_object = family_rank(d, "object")[0]
        compact.append({
            "model": model,
            "passed": ",".join(passed) if passed else "none",
            "best_color": best_color[0],
            "best_color_delta": float(best_color[1]["best_own_delta"]),
            "best_color_ratio": float(best_color[1]["best_selectivity_ratio"]),
            "best_object": best_object[0],
            "best_object_delta": float(best_object[1]["best_own_delta"]),
            "best_object_ratio": float(best_object[1]["best_selectivity_ratio"]),
        })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append(
            "| model | passed candidates | best color | color own Δ | color ratio | "
            "best object | object own Δ | object ratio |"
        )
        lines.append("|---|---|---|---:|---:|---|---:|---:|")
        for r in compact:
            lines.append(
                f"| {r['model']} | {r['passed']} | {r['best_color']} | "
                f"{r['best_color_delta']:+.4f} | {r['best_color_ratio']:.4f} | "
                f"{r['best_object']} | {r['best_object_delta']:+.4f} | {r['best_object_ratio']:.4f} |"
            )
        lines.append("")

    out = root / "phase529_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
