#!/usr/bin/env python3
"""Phase 561 cross-model summary: gated continuous injection beta sweep."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_ROOT = Path("results/glm5_phase561_gated_continuous_injection")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODEL_LABELS = {"qwen3": "qwen3", "glm4": "glm4", "deepseek7b": "deepseek7b"}

DONOR_CONDS = [
    ("resid_donor_vehicle_repeat2_add", "repeat2(heli)"),
    ("resid_donor_vehicle_repeat4_add", "repeat4(rocket)"),
    ("resid_donor_vehicle_mean_cache_add", "mean_cache"),
    ("resid_donor_vehicle_random_cache_add", "random_cache"),
]
BETAS = ["b0p05", "b0p1", "b0p25", "b0p5"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
ROUTE_LABELS = {
    "forbidden_sentence_completion:temperature<-forbidden_definition": "sent<-def",
    "forbidden_definition:top_p<-forbidden_definition": "def<-def",
}


def load_model_data(model: str) -> dict:
    path = OUT_ROOT / f"phase561_{model}_continuous_blend_gated_injection.json"
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
    data = {m: load_model_data(m) for m in MODELS}

    lines = []
    lines.append("# Phase 561: Gated Continuous Injection Cross-Model Summary")
    lines.append("")
    lines.append("Surgery: h' = (1-beta)*h_current + beta*donor_cache, applied every step")
    lines.append("Beta sweep: 0.05, 0.10, 0.25, 0.50")
    lines.append("")

    for route in ROUTES:
        rlabel = ROUTE_LABELS[route]
        lines.append(f"## Route: {rlabel}")
        lines.append("")

        # clean_non_object_rate table
        lines.append("### clean_non_object_rate")
        lines.append("")
        header = "| model | baseline | remove | " + " | ".join(
            f"{dl}__{b}" for _, dl in DONOR_CONDS for b in BETAS
        ) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (3 + len(DONOR_CONDS) * len(BETAS))) + "|")
        for model in MODELS:
            d = data[model]
            if d is None:
                lines.append(f"| {MODEL_LABELS[model]} | N/A |")
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            base = get_row(audit, combo, route, "baseline")
            rm = get_row(audit, combo, route, "resid_remove_perp")
            vals = [fmt(base["clean_non_object_rate"] if base else None),
                    fmt(rm["clean_non_object_rate"] if rm else None)]
            for cond, _ in DONOR_CONDS:
                for b in BETAS:
                    full_cond = f"{cond}__{b}"
                    row = get_row(audit, combo, route, full_cond)
                    vals.append(fmt(row["clean_non_object_rate"] if row else None))
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

        # Steering gain vs baseline
        lines.append("### steering gain (condition - baseline)")
        lines.append("")
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (3 + len(DONOR_CONDS) * len(BETAS))) + "|")
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            base = get_row(audit, combo, route, "baseline")
            rm = get_row(audit, combo, route, "resid_remove_perp")
            base_v = base["clean_non_object_rate"] if base else 0
            vals = [fmt(0, sign=True),
                    fmt((rm["clean_non_object_rate"] - base_v) if rm else None, sign=True)]
            for cond, _ in DONOR_CONDS:
                for b in BETAS:
                    full_cond = f"{cond}__{b}"
                    row = get_row(audit, combo, route, full_cond)
                    if row:
                        vals.append(fmt(row["clean_non_object_rate"] - base_v, sign=True))
                    else:
                        vals.append(" " * 6)
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

        # object_echo_rate
        lines.append("### object_echo_rate")
        lines.append("")
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (3 + len(DONOR_CONDS) * len(BETAS))) + "|")
        for model in MODELS:
            d = data[model]
            if d is None:
                continue
            audit = d["audit"]
            combo = list(audit.keys())[0]
            base = get_row(audit, combo, route, "baseline")
            rm = get_row(audit, combo, route, "resid_remove_perp")
            vals = [fmt(base["object_echo_rate"] if base else None),
                    fmt(rm["object_echo_rate"] if rm else None)]
            for cond, _ in DONOR_CONDS:
                for b in BETAS:
                    full_cond = f"{cond}__{b}"
                    row = get_row(audit, combo, route, full_cond)
                    vals.append(fmt(row["object_echo_rate"] if row else None))
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

    # Timing and metadata
    lines.append("## Timing & Metadata")
    lines.append("")
    lines.append("| model | time (min) | seeds | max tokens |")
    lines.append("|---|---|---|---|")
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        lines.append(f"| {MODEL_LABELS[model]} | {d.get('total_time_min','?')} | {d.get('sample_seeds','?')} | {d.get('max_new_tokens','?')} |")
    lines.append("")

    # Summary of key findings
    lines.append("## Key Findings Summary")
    lines.append("")
    lines.append("### Low-beta window (beta=0.05):")
    lines.append("")
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        lines.append(f"**{MODEL_LABELS[model]}** (sent<-def):")
        audit = d["audit"]
        combo = list(audit.keys())[0]
        base = get_row(audit, combo, ROUTES[0], "baseline")
        base_v = base["clean_non_object_rate"] if base else 0
        for cond, dl in DONOR_CONDS:
            full_cond = f"{cond}__b0p05"
            row = get_row(audit, combo, ROUTES[0], full_cond)
            if row:
                gain = row["clean_non_object_rate"] - base_v
                lines.append(f"  - {dl}: clean_no={row['clean_non_object_rate']:.2f} (gain {gain:+.2f})")
        lines.append("")

    out_path = OUT_ROOT / "phase561_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    # Print to console too
    print("\n".join(lines))


if __name__ == "__main__":
    main()
