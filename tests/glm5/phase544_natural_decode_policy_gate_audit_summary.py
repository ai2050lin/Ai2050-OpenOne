#!/usr/bin/env python3
"""Summary for Phase544 natural answer and decode-mode policy gate audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase544_natural_decode_policy_gate_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["baseline", "residual_perp", "residual_parallel", "residual_full"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase544_{model}_natural_decode_policy_gate_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_window(d: dict[str, Any], source: str, scaffold: str, mode: str, condition: str) -> tuple[str, dict[str, Any]]:
    rows = {
        win: d["audit"][win]["sources"][source][scaffold][mode][condition]
        for win in d["audit"]
    }
    final_k = str(d["max_new_tokens"])
    win = max(rows, key=lambda w: rows[w]["hit_at_k"][final_k]["family_target"])
    return win, rows[win]


def gain(base: dict[str, Any], row: dict[str, Any], k: int, metric: str = "family_target") -> float:
    key = str(k)
    return float(row["hit_at_k"][key][metric] - base["hit_at_k"][key][metric])


def classify(base: dict[str, Any], row: dict[str, Any], k: int) -> str:
    fam_gain = gain(base, row, k, "family_target")
    exact_gain = gain(base, row, k, "exact_target")
    rank_improve = float(base["mean_first_target_rank"] - row["mean_first_target_rank"])
    comp = float(row["hit_at_k"][str(k)]["family_competitor"])
    if fam_gain >= 0.25 and comp <= base["hit_at_k"][str(k)]["family_competitor"] + 0.15:
        return "natural_gate_open"
    if fam_gain >= 0.10:
        return "weak_natural_gate"
    if exact_gain >= 0.10 and fam_gain < 0.10:
        return "label_only_gain"
    if rank_improve > 50 and fam_gain < 0.05:
        return "rank_only"
    return "no_gate"


def curve(row: dict[str, Any], checkpoints: list[int], metric: str) -> str:
    vals = []
    for k in checkpoints:
        vals.append(f"k{k}={row['hit_at_k'][str(k)][metric]:.2f}")
    return ", ".join(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase544 Natural Answer and Decode-Mode Policy Gate Summary", ""]
    compact = []
    for model, d in data.items():
        final_k = int(d["max_new_tokens"])
        checkpoints = [int(x) for x in d["checkpoints"]]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"core={d['core_sources']}, scaffolds={d['scaffolds']}, modes={d['decode_modes']}, "
            f"windows={d['windows']}, train_n={d['train_n']}, test_n={d['test_n']}, "
            f"max_new_tokens={final_k}, alpha={d['alpha']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("| source | scaffold | mode | condition | win | family target curve | exact target curve | rankT | class |")
        lines.append("|---|---|---|---|---|---|---|---:|---|")
        for source in d["core_sources"]:
            for scaffold in d["scaffolds"]:
                for mode in d["decode_modes"]:
                    base_win, base = best_window(d, source, scaffold, mode, "baseline")
                    for condition in CONDITIONS:
                        win, row = best_window(d, source, scaffold, mode, condition)
                        cls = "baseline" if condition == "baseline" else classify(base, row, final_k)
                        if condition != "baseline":
                            compact.append({
                                "model": model,
                                "source": source,
                                "scaffold": scaffold,
                                "mode": mode,
                                "condition": condition,
                                "win": win,
                                "base_family": base["hit_at_k"][str(final_k)]["family_target"],
                                "family": row["hit_at_k"][str(final_k)]["family_target"],
                                "family_gain": gain(base, row, final_k, "family_target"),
                                "base_exact": base["hit_at_k"][str(final_k)]["exact_target"],
                                "exact": row["hit_at_k"][str(final_k)]["exact_target"],
                                "exact_gain": gain(base, row, final_k, "exact_target"),
                                "rank_improve": base["mean_first_target_rank"] - row["mean_first_target_rank"],
                                "class": cls,
                            })
                        lines.append(
                            f"| {source} | {scaffold} | {mode} | {condition} | {win} | "
                            f"{curve(row, checkpoints, 'family_target')} | {curve(row, checkpoints, 'exact_target')} | "
                            f"{float(row['mean_first_target_rank']):.1f} | {cls} |"
                        )
        lines.append("")

    if compact:
        lines.append("## Best Family-Hit Positive Rows")
        lines.append("")
        lines.append("| model | source | scaffold | mode | condition | win | base family | family | gain | exact gain | rank improve | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in sorted(compact, key=lambda x: (x["family_gain"], x["exact_gain"]), reverse=True)[:48]:
            lines.append(
                f"| {r['model']} | {r['source']} | {r['scaffold']} | {r['mode']} | {r['condition']} | {r['win']} | "
                f"{float(r['base_family']):.2f} | {float(r['family']):.2f} | {float(r['family_gain']):+.2f} | "
                f"{float(r['exact_gain']):+.2f} | {float(r['rank_improve']):.1f} | {r['class']} |"
            )
        lines.append("")

        lines.append("## Decode-Mode Max Gain")
        lines.append("")
        lines.append("| model | mode | max family gain | row |")
        lines.append("|---|---|---:|---|")
        for model in sorted(set(r["model"] for r in compact)):
            for mode in sorted(set(r["mode"] for r in compact if r["model"] == model)):
                rows = [r for r in compact if r["model"] == model and r["mode"] == mode]
                best = max(rows, key=lambda x: x["family_gain"])
                desc = f"{best['source']} {best['scaffold']} {best['condition']} {best['win']}"
                lines.append(f"| {model} | {mode} | {float(best['family_gain']):+.2f} | {desc} |")
        lines.append("")

    out = root / "phase544_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
