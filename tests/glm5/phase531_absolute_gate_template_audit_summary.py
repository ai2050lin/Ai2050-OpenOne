#!/usr/bin/env python3
"""Summary for Phase531 absolute gate and template path audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase531_absolute_gate_template_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
COMPONENTS = ["full", "parallel", "perp", "random_perp", "random_readout"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase531_{model}_absolute_gate_template_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cell(adm: dict[str, Any]) -> str:
    strict = "Y" if adm["passes_strict_gate"] else "n"
    ratio = "R" if adm["passes_ratio_gate"] else "r"
    abs_gate = "A" if adm["passes_absolute_gate"] else "a"
    return f"{float(adm['best_own_delta']):+.3f}/{float(adm['best_selectivity_ratio']):.2f}/{ratio}{abs_gate}{strict}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase531 Absolute Gate and Template Audit Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, min_abs_delta={d['min_abs_delta']}, "
            f"attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")

        lines.append("Cell format: own Δ / ratio / RatioGate AbsoluteGate StrictGate")
        lines.append("")
        lines.append("| candidate | family | own task | par % | full | parallel | perp | random_perp | random_readout |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        passed, random_passed = [], []
        for cand, meta in d["candidate_meta"].items():
            stats = d["component_stats"][cand]
            adm = d["admission"][cand]
            for comp in COMPONENTS:
                if adm[comp]["passes_strict_gate"]:
                    target = random_passed if comp.startswith("random") else passed
                    target.append(f"{cand}:{comp}")
            lines.append(
                f"| {cand} | {meta['family']} | {meta['own_task']} | "
                f"{float(stats['parallel_norm_pct']):.2f} | "
                f"{cell(adm['full'])} | {cell(adm['parallel'])} | {cell(adm['perp'])} | "
                f"{cell(adm['random_perp'])} | {cell(adm['random_readout'])} |"
            )
        lines.append("")

        lines.append("### Template Direction Cosine")
        lines.append("")
        for pair, audit in d["template_audit"].items():
            lines.append(f"#### {pair}")
            names = list(audit["norms"])
            lines.append("")
            lines.append("| dir | norm | " + " | ".join(names) + " |")
            lines.append("|---|---:|" + "|".join(["---:"] * len(names)) + "|")
            for a in names:
                vals = [float(audit["cosines"][a][b]) for b in names]
                lines.append(
                    f"| {a} | {float(audit['norms'][a]):.4f} | "
                    + " | ".join(f"{v:+.4f}" for v in vals) + " |"
                )
            lines.append("")

        compact.append({
            "model": model,
            "passed": ",".join(passed) if passed else "none",
            "random_passed": ",".join(random_passed) if random_passed else "none",
        })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | strict passed learned components | strict passed random controls |")
        lines.append("|---|---|---|")
        for r in compact:
            lines.append(f"| {r['model']} | {r['passed']} | {r['random_passed']} |")
        lines.append("")

    out = root / "phase531_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
