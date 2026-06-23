#!/usr/bin/env python3
"""Summarize Phase 586 distributed value path patch results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase586_distributed_value_path_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load(model: str):
    path = ROOT / f"phase586_{model}_distributed_value_path_patch_confirm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def best_items(data, mode: str, limit: int = 3):
    items = [
        v for v in data["summary"]["by_key"].values()
        if v["mode"] == mode
    ]
    items.sort(
        key=lambda x: (
            x["target_n"],
            x["target_patch_accuracy"],
            x["patch_accuracy"],
            x["mean_logprob_gain"],
        ),
        reverse=True,
    )
    return items[:limit]


def main() -> None:
    lines = [
        "# Phase 586 Distributed Value Path Patch Summary",
        "",
        "Confirm setting: 24 value cases per model, 5 positions, 4 layers, add/replace repair plus wrong-relation and random controls.",
        "",
        "| model | target cases | best repair position | layer | mode | patch acc | target patch | mean correct-logprob gain | best wrong-rel gain |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = load(model)
        best = best_items(data, "add_repair", 1)[0]
        best_replace = best_items(data, "replace_repair", 1)[0]
        if (best_replace["target_patch_accuracy"], best_replace["patch_accuracy"], best_replace["mean_logprob_gain"]) > (
            best["target_patch_accuracy"], best["patch_accuracy"], best["mean_logprob_gain"]
        ):
            best = best_replace
        wrong = best_items(data, "add_wrong_relation", 1)[0]
        lines.append(
            f"| {model} | {best['target_n']} | {best['position']} | L{best['layer']} | {best['mode']} | "
            f"{pct(best['patch_accuracy'])} | {best['target_patch_correct']}/{best['target_n']} "
            f"({pct(best['target_patch_accuracy'])}) | {best['mean_logprob_gain']:.3f} | "
            f"{wrong['mean_logprob_gain']:.3f} |"
        )

    lines += [
        "",
        "## Objective Facts",
        "",
        "- Qwen3 and GLM4 have too few failed target cases in this setup, so they are weak controls for value-gate localization.",
        "- DS7B is the main diagnostic model: repair prompt fixes the task, but distributed single-position patch still repairs 0/9 target cases.",
        "- DS7B prompt_last late-layer repair greatly increases correct-value logprob (L26 gain about +5.38), yet does not flip the final candidate winner.",
        "- Wrong-relation control also increases DS7B correct-value logprob less strongly (about +2.10), so the gain is not purely direction-specific enough to close the gate.",
        "- Phase586 therefore does not find the value gate location; it shows the value gate is not solved by single-position residual patch, even at rule/query/prompt positions.",
        "",
    ]
    out = ROOT / "phase586_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
