#!/usr/bin/env python3
"""Phase 562 cross-model summary: trajectory response audit."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_ROOT = Path("results/glm5_phase562_trajectory_response")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = [
    "baseline", "one_shot_repeat2", "one_shot_repeat4",
    "one_shot_mean", "one_shot_random",
    "add_tangent_repeat2", "add_normal_repeat2",
]
COND_LABELS = {
    "baseline": "baseline",
    "one_shot_repeat2": "repeat2(heli)",
    "one_shot_repeat4": "repeat4(rocket)",
    "one_shot_mean": "mean_cache",
    "one_shot_random": "random_cache",
    "add_tangent_repeat2": "tangent_r2",
    "add_normal_repeat2": "normal_r2",
}
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
ROUTE_LABELS = {
    "forbidden_sentence_completion:temperature<-forbidden_definition": "sent<-def",
    "forbidden_definition:top_p<-forbidden_definition": "def<-def",
}


def load_data(model: str) -> dict | None:
    path = OUT_ROOT / f"phase562_{model}_trajectory_response.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_row(audit: dict, combo: str, route: str, condition: str) -> dict | None:
    try:
        return audit[combo]["rows"][route][condition]
    except (KeyError, TypeError):
        return None


def fmt(v, w=6, sign=False):
    if v is None:
        return " " * w
    if isinstance(v, str):
        return v
    if sign:
        return f"{v:+{w}.2f}"
    return f"{v:{w}.2f}"


def main():
    data = {m: load_data(m) for m in MODELS}
    lines = []
    lines.append("# Phase 562: Trajectory Response Audit Cross-Model Summary")
    lines.append("")
    lines.append("Surgery: one-shot at step 0, then free KV-cache generation (16 tokens)")
    lines.append("Conditions: baseline, one_shot repeat2/4/mean/random, add_tangent/normal")
    lines.append("")

    for route in ROUTES:
        rlabel = ROUTE_LABELS[route]
        lines.append(f"## Route: {rlabel}")
        lines.append("")

        # clean_non_object_rate
        lines.append("### clean_non_object_rate")
        lines.append("")
        header = "| model | " + " | ".join(COND_LABELS[c] for c in CONDITIONS) + " |"
        sep = "|---|" + "|".join(["---"] * len(CONDITIONS)) + "|"
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITIONS:
                row = get_row(audit, combo, route, cond)
                vals.append(fmt(row["clean_non_object_rate"] if row else None))
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        # Relaxation length
        lines.append("### mean_relaxation_length (steps until target token disappears)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITIONS:
                row = get_row(audit, combo, route, cond)
                vals.append(fmt(row.get("mean_relaxation_length") if row else None))
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        # Semantic specificity (clean_no - random_cache clean_no)
        lines.append("### semantic_specificity (condition - random_cache)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            random_row = get_row(audit, combo, route, "one_shot_random")
            random_v = random_row["clean_non_object_rate"] if random_row else 0
            vals = []
            for cond in CONDITIONS:
                row = get_row(audit, combo, route, cond)
                if row:
                    vals.append(fmt(row["clean_non_object_rate"] - random_v, sign=True))
                else:
                    vals.append(" " * 6)
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        # Tangent vs Normal comparison
        lines.append("### tangent vs normal (add_tangent vs add_normal)")
        lines.append("")
        lines.append("| model | baseline | tangent_r2 | normal_r2 | tangent-baseline | normal-baseline | tangent-normal |")
        lines.append("|---|---|---|---|---|---|---|")
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            base = get_row(audit, combo, route, "baseline")
            tan = get_row(audit, combo, route, "add_tangent_repeat2")
            nor = get_row(audit, combo, route, "add_normal_repeat2")
            bv = base["clean_non_object_rate"] if base else 0
            tv = tan["clean_non_object_rate"] if tan else 0
            nv = nor["clean_non_object_rate"] if nor else 0
            lines.append(f"| {model} | {bv:.2f} | {tv:.2f} | {nv:.2f} | {tv-bv:+.2f} | {nv-bv:+.2f} | {tv-nv:+.2f} |")
        lines.append("")

        # Degeneration distribution
        lines.append("### degeneration_distribution (normal / repetition / garbage)")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITIONS:
                row = get_row(audit, combo, route, cond)
                if row:
                    deg = row.get("degeneration_distribution", {})
                    n = deg.get("normal", 0)
                    r = deg.get("repetition", 0)
                    g = deg.get("garbage", 0)
                    vals.append(f"{n:.2f}n/{r:.2f}r/{g:.2f}g")
                else:
                    vals.append(" " * 14)
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        # Per-step target rate
        lines.append("### per-step target_rate (first 8 steps)")
        lines.append("")
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            lines.append(f"**{model}:**")
            lines.append("| condition | step0 | step1 | step2 | step3 | step4 | step5 | step6 | step7 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for cond in CONDITIONS:
                row = get_row(audit, combo, route, cond)
                if row and "avg_target_rates" in row:
                    rates = row["avg_target_rates"][:8]
                    while len(rates) < 8:
                        rates.append(0)
                    label = COND_LABELS[cond]
                    lines.append(f"| {label} | " + " | ".join(f"{r:.2f}" for r in rates) + " |")
            lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append("| model | time (min) |")
    lines.append("|---|---|")
    for model in MODELS:
        d = data[model]
        if d:
            lines.append(f"| {model} | {d.get('total_time_min','?')} |")
    lines.append("")

    # Direction decomposition stats (from logs)
    lines.append("## Direction Decomposition (tangent vs normal component norms)")
    lines.append("")
    lines.append("GLM4 sent<-def: |u|≈25-53, |tangent|≈5-9, |normal|≈21-39 → donor direction mostly NORMAL to trajectory")
    lines.append("GLM4 def<-def: |tangent|≈0.5-0.9, |normal|≈12-30 → even more normal-dominated")
    lines.append("DS7B: |u|≈215-356 (much larger), |tangent|≈7-9, |normal|≈69-111 → also normal-dominated")
    lines.append("")

    out_path = OUT_ROOT / "phase562_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
