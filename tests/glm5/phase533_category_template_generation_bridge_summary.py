#!/usr/bin/env python3
"""Summary for Phase533 category template robustness and generation bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase533_category_template_generation_bridge")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase533_{model}_category_template_generation_bridge.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cell(adm: dict[str, Any]) -> str:
    return f"{float(adm['best_own_delta']):+.3f}/{float(adm['best_selectivity_ratio']):.2f}/{'Y' if adm['passes_strict_gate'] else 'n'}"


def bridge_cell(row: dict[str, Any]) -> str:
    steps = ",".join(f"{float(x):.2f}" for x in row.get("step_hit_rates", []))
    return f"{float(row['hit_rate']):.2f} [{steps}]"


def candidate_verdict(d: dict[str, Any], cand: str) -> str:
    adm = d["admission"][cand]
    rsum = d["random_perp_summary"][cand]
    perp_delta = float(adm["perp"]["best_own_delta"])
    random_max = float(rsum["max_own_delta"])
    if adm["perp"]["passes_strict_gate"] and perp_delta > random_max and int(rsum["strict_pass_count"]) == 0:
        return "clean_nonrandom_perp"
    if adm["perp"]["passes_strict_gate"] and perp_delta > random_max:
        return "perp_above_random_but_random_passes"
    if adm["perp"]["passes_strict_gate"]:
        return "perp_not_above_random_max"
    if adm["parallel"]["passes_strict_gate"] and adm["random_readout"]["passes_strict_gate"]:
        return "readout_interface"
    if adm["full"]["passes_strict_gate"]:
        return "full_only"
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase533 Category Template and Generation Bridge Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"bridge_n={d['generation_bridge'][next(iter(d['generation_bridge']))]['baseline']['n'] if d.get('generation_bridge') else 0}, "
            f"max_new_tokens={d.get('max_new_tokens', 'unknown')}, alphas={d['alphas']}, "
            f"seeds={d['random_seeds']}, min_abs_delta={d['min_abs_delta']}, "
            f"attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Cell format: best own delta / selectivity ratio / strict gate.")
        lines.append("")
        lines.append("| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        clean_nonrandom = []
        above_random_with_random_passes = []
        for cand, meta in d["candidate_meta"].items():
            adm = d["admission"][cand]
            stats = d["component_stats"][cand]
            rsum = d["random_perp_summary"][cand]
            verdict = candidate_verdict(d, cand)
            if verdict == "clean_nonrandom_perp":
                clean_nonrandom.append(cand)
            elif verdict == "perp_above_random_but_random_passes":
                above_random_with_random_passes.append(cand)
            lines.append(
                f"| {cand} | {meta['family']} | {meta['own_task']} | "
                f"{float(stats['parallel_norm_pct']):.2f} | {cell(adm['full'])} | "
                f"{cell(adm['parallel'])} | {cell(adm['perp'])} | "
                f"{float(rsum['max_own_delta']):+.3f} | {int(rsum['strict_pass_count'])} | "
                f"{cell(adm['random_readout'])} | {verdict} |"
            )
        lines.append("")

        if d.get("template_cosines"):
            lines.append("### Category Template Cosines")
            lines.append("")
            names = sorted(d["template_cosines"])
            lines.append("| dir | " + " | ".join(names) + " |")
            lines.append("|---|" + "|".join(["---:"] * len(names)) + "|")
            for a in names:
                vals = [float(d["template_cosines"][a][b]) for b in names]
                lines.append(f"| {a} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")
            lines.append("")

        if d.get("generation_bridge"):
            lines.append("### Category Greedy Generation Bridge")
            lines.append("")
            lines.append("Cell format: any target-token hit rate [per-step hit rates].")
            lines.append("")
            lines.append("| candidate | baseline | perp | random_readout | perp-baseline | random_readout-baseline |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for cand, row in d["generation_bridge"].items():
                b = float(row["baseline"]["hit_rate"])
                p = float(row["perp"]["hit_rate"])
                rr = float(row["random_readout"]["hit_rate"])
                lines.append(
                    f"| {cand} | {bridge_cell(row['baseline'])} | {bridge_cell(row['perp'])} | "
                    f"{bridge_cell(row['random_readout'])} | {p - b:+.2f} | {rr - b:+.2f} |"
                )
            lines.append("")

        compact.append({
            "model": model,
            "clean_nonrandom_perp": ",".join(clean_nonrandom) if clean_nonrandom else "none",
            "above_random_with_random_passes": (
                ",".join(above_random_with_random_passes) if above_random_with_random_passes else "none"
            ),
        })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | clean nonrandom learned perp | learned perp above random max but random also passes |")
        lines.append("|---|---|---|")
        for row in compact:
            lines.append(
                f"| {row['model']} | {row['clean_nonrandom_perp']} | "
                f"{row['above_random_with_random_passes']} |"
            )
        lines.append("")

    out = root / "phase533_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
