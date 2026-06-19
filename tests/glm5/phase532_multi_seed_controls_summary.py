#!/usr/bin/env python3
"""Summary for Phase532 multi-seed controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase532_multi_seed_controls")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase532_{model}_multi_seed_controls.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def c(adm: dict[str, Any]) -> str:
    return f"{float(adm['best_own_delta']):+.3f}/{float(adm['best_selectivity_ratio']):.2f}/{'Y' if adm['passes_strict_gate'] else 'n'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines: list[str] = ["# Phase532 Multi-Seed Controls Summary", ""]
    compact = []

    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layer=L{d['layer']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"alphas={d['alphas']}, min_abs_delta={d['min_abs_delta']}, "
            f"seeds={d['random_seeds']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| candidate | family | own task | par % | full | parallel | perp | random_perp max | random_perp passes | random_readout | verdict |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        passed = []
        for cand, meta in d["candidate_meta"].items():
            adm = d["admission"][cand]
            stats = d["component_stats"][cand]
            rsum = d["random_perp_summary"][cand]
            learned_perp = adm["perp"]["passes_strict_gate"]
            random_any = int(rsum["strict_pass_count"]) > 0
            readout = adm["parallel"]["passes_strict_gate"] and adm["random_readout"]["passes_strict_gate"]
            learned_perp_delta = float(adm["perp"]["best_own_delta"])
            random_max = float(rsum["max_own_delta"])
            if learned_perp and learned_perp_delta > random_max:
                verdict = "nonrandom_perp"
                passed.append(f"{cand}:perp")
            elif learned_perp:
                verdict = "perp_not_above_random_max"
            elif readout and not adm["full"]["passes_strict_gate"] and not learned_perp:
                verdict = "readout_only"
            elif adm["full"]["passes_strict_gate"]:
                verdict = "full_pass"
            else:
                verdict = "fail"
            lines.append(
                f"| {cand} | {meta['family']} | {meta['own_task']} | "
                f"{float(stats['parallel_norm_pct']):.2f} | {c(adm['full'])} | {c(adm['parallel'])} | "
                f"{c(adm['perp'])} | {float(rsum['max_own_delta']):+.3f} | "
                f"{int(rsum['strict_pass_count'])} | {c(adm['random_readout'])} | {verdict} |"
            )
        lines.append("")
        compact.append({"model": model, "nonrandom_perp": ",".join(passed) if passed else "none"})

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | nonrandom learned perp components |")
        lines.append("|---|---|")
        for row in compact:
            lines.append(f"| {row['model']} | {row['nonrandom_perp']} |")
        lines.append("")

    out = root / "phase532_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
