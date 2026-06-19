#!/usr/bin/env python3
"""Summary for Phase530 state-pair decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase530_state_pair_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
COMPONENTS = ["full", "parallel", "perp"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase530_{model}_state_pair_decomposition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def component_cell(adm: dict[str, Any]) -> str:
    flag = "Y" if adm["passes_basic_gate"] else "n"
    return f"{float(adm['best_own_delta']):+.4f}/{float(adm['best_selectivity_ratio']):.2f}/{flag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase530 State-Pair Decomposition Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Cell format: best own Δ / selectivity ratio / pass")
        lines.append("")
        lines.append(
            "| candidate | family | own task | parallel norm % | perp norm % | "
            "full | parallel | perp |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        passed = []
        for cand, meta in d["candidate_meta"].items():
            stats = d["component_stats"][cand]
            adm = d["admission"][cand]
            for comp in COMPONENTS:
                if adm[comp]["passes_basic_gate"]:
                    passed.append(f"{cand}:{comp}")
            lines.append(
                f"| {cand} | {meta['family']} | {meta['own_task']} | "
                f"{float(stats['parallel_norm_pct']):.2f} | {float(stats['perp_norm_pct']):.2f} | "
                f"{component_cell(adm['full'])} | {component_cell(adm['parallel'])} | {component_cell(adm['perp'])} |"
            )
        lines.append("")

        def best_for(prefix: str, comp: str) -> tuple[str, float, float, bool]:
            rows = []
            for cand, adm in d["admission"].items():
                if cand.startswith(prefix):
                    rows.append((
                        float(adm[comp]["best_own_delta"]),
                        cand,
                        float(adm[comp]["best_selectivity_ratio"]),
                        bool(adm[comp]["passes_basic_gate"]),
                    ))
            if not rows:
                return ("none", 0.0, 0.0, False)
            val, cand, ratio, passed_flag = sorted(rows, reverse=True)[0]
            return cand, val, ratio, passed_flag

        compact.append({
            "model": model,
            "passed": ",".join(passed) if passed else "none",
            "bw_full": best_for("color_black_white", "full"),
            "rb_full": best_for("color_red_blue", "full"),
            "obj_full": best_for("object", "full"),
            "obj_perp": best_for("object", "perp"),
        })

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append(
            "| model | passed components | best BW full | best RB full | best object full | best object perp |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in compact:
            def fmt(t: tuple[str, float, float, bool]) -> str:
                return f"{t[0]} {t[1]:+.3f}/{t[2]:.2f}/{'Y' if t[3] else 'n'}"
            lines.append(
                f"| {r['model']} | {r['passed']} | {fmt(r['bw_full'])} | "
                f"{fmt(r['rb_full'])} | {fmt(r['obj_full'])} | {fmt(r['obj_perp'])} |"
            )
        lines.append("")

    out = root / "phase530_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
