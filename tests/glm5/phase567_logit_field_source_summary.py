#!/usr/bin/env python3
"""Phase 567 cross-model summary: step0 logit-field source decomposition."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_ROOT = Path("results/glm5_phase567_logit_field_source")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = [
    "baseline", "repeat4_all", "repeat4_attn_only", "repeat4_mlp_only",
    "repeat4_attn_mlp", "random_all", "random_attn_only", "random_mlp_only",
]
COND_LABELS = {
    "baseline": "baseline",
    "repeat4_all": "r4_all",
    "repeat4_attn_only": "r4_attn",
    "repeat4_mlp_only": "r4_mlp",
    "repeat4_attn_mlp": "r4_a+m",
    "random_all": "rand_all",
    "random_attn_only": "rand_attn",
    "random_mlp_only": "rand_mlp",
}


def load_data(model: str) -> dict | None:
    path = OUT_ROOT / f"phase567_{model}_logit_field_source.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_free_agg(audit: dict, condition: str) -> dict | None:
    try:
        combo = list(audit.keys())[0]
        route = list(audit[combo]["rows"].keys())[0]
        return audit[combo]["rows"][route][condition]["free"]
    except (KeyError, TypeError, IndexError):
        return None


def get_bfi_agg(audit: dict, condition: str) -> dict | None:
    try:
        combo = list(audit.keys())[0]
        route = list(audit[combo]["rows"].keys())[0]
        return audit[combo]["rows"][route][condition].get("bfi_prefix2")
    except (KeyError, TypeError, IndexError):
        return None


def fmt(v, w=7, sign=False):
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
    lines.append("# Phase 567: Step0 Logit-Field Source Decomposition Cross-Model Summary")
    lines.append("")
    lines.append("Surgery: module-level donor injection at step 0 (layer/attn/mlp), then free generation")
    lines.append("")

    # free clean_non_object_rate
    lines.append("## free clean_non_object_rate")
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
        vals = []
        for cond in CONDITIONS:
            agg = get_free_agg(audit, cond)
            vals.append(fmt(agg["clean_non_object_rate"] if agg else None))
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    lines.append("")

    # step0 margin
    lines.append("## step0 target-competitor margin")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        audit = d["audit"]
        vals = []
        for cond in CONDITIONS:
            agg = get_free_agg(audit, cond)
            vals.append(fmt(agg.get("step0_margin") if agg else None, sign=True))
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    lines.append("")

    # step0 target rank
    lines.append("## step0 target rank")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        audit = d["audit"]
        vals = []
        for cond in CONDITIONS:
            agg = get_free_agg(audit, cond)
            v = agg.get("step0_target_rank") if agg else None
            vals.append(f"{v:7.0f}" if v else " " * 7)
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    lines.append("")

    # bfi_prefix2 clean
    lines.append("## bfi_prefix2 clean_non_object_rate (baseline forced to intervention's first 2 tokens)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        audit = d["audit"]
        vals = []
        for cond in CONDITIONS:
            agg = get_bfi_agg(audit, cond)
            vals.append(fmt(agg["clean_non_object_rate"] if agg else None))
        lines.append(f"| {model} | " + " | ".join(vals) + " |")
    lines.append("")

    # Semantic specificity: repeat4_all - random_all
    lines.append("## Semantic specificity (repeat4_all - random_all)")
    lines.append("")
    lines.append("| model | free_clean | s0_margin | bfi_p2 |")
    lines.append("|---|---|---|---|")
    for model in MODELS:
        d = data[model]
        if d is None:
            continue
        audit = d["audit"]
        r4 = get_free_agg(audit, "repeat4_all")
        ra = get_free_agg(audit, "random_all")
        r4_bfi = get_bfi_agg(audit, "repeat4_all")
        ra_bfi = get_bfi_agg(audit, "random_all")
        fc = (r4["clean_non_object_rate"] - ra["clean_non_object_rate"]) if (r4 and ra) else None
        sm = ((r4.get("step0_margin",0) - ra.get("step0_margin",0))) if (r4 and ra) else None
        bp = ((r4_bfi["clean_non_object_rate"] - ra_bfi["clean_non_object_rate"]) if (r4_bfi and ra_bfi) else None)
        lines.append(f"| {model} | {fmt(fc, sign=True)} | {fmt(sm, sign=True)} | {fmt(bp, sign=True)} |")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append("| model | time (min) | test_n | seeds |")
    lines.append("|---|---|---|---|")
    for model in MODELS:
        d = data[model]
        if d:
            lines.append(f"| {model} | {d.get('total_time_min','?')} | {d.get('test_n','?')} | {len(d.get('sample_seeds',[]))} |")
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append("### GLM4 (core model, test_n=24, 8 seeds):")
    lines.append("")
    lines.append("| condition | free_clean | s0_margin | bfi_p2 |")
    lines.append("|---|---|---|---|")
    audit_glm4 = data["glm4"]["audit"] if data.get("glm4") else None
    if audit_glm4:
        for cond in CONDITIONS:
            agg = get_free_agg(audit_glm4, cond)
            bfi = get_bfi_agg(audit_glm4, cond)
            fc = f"{agg['clean_non_object_rate']:.2f}" if agg else "N/A"
            sm = f"{agg.get('step0_margin',0):+.2f}" if agg else "N/A"
            bp = f"{bfi['clean_non_object_rate']:.2f}" if bfi else "N/A"
            lines.append(f"| {COND_LABELS[cond]} | {fc} | {sm} | {bp} |")
    lines.append("")
    lines.append("**Critical:** Only `repeat4_all` (full layer restore) produces step0 margin flip (+0.40).")
    lines.append("Individual module injections (attn_only, mlp_only, attn_mlp) do NOT flip the margin.")
    lines.append("This suggests the logit-field flip requires the complete residual state, not just module deltas.")

    out_path = OUT_ROOT / "phase567_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
