#!/usr/bin/env python3
"""Phase 559 cross-model summary: prototype generation closure comparison."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT_ROOT = Path("results/glm5_phase559_prototype_generation_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODEL_LABELS = {"qwen3": "qwen3", "glm4": "glm4", "deepseek7b": "deepseek7b"}

CONDITION_ORDER = [
    "baseline",
    "add_perp",
    "resid_remove_perp",
    "resid_donor_vehicle_same_add",
    "resid_donor_vehicle_shuffle_add",
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_pca1_cache_add",
    "resid_donor_vehicle_random_cache_add",
    "resid_donor_tool_same_add",
]
COND_LABELS = {
    "baseline": "baseline",
    "add_perp": "add_perp",
    "resid_remove_perp": "remove_perp",
    "resid_donor_vehicle_same_add": "same",
    "resid_donor_vehicle_shuffle_add": "shuffle",
    "resid_donor_vehicle_repeat2_add": "repeat2(heli)",
    "resid_donor_vehicle_repeat4_add": "repeat4(rocket)",
    "resid_donor_vehicle_mean_cache_add": "mean_cache",
    "resid_donor_vehicle_pca1_cache_add": "pca1_cache",
    "resid_donor_vehicle_random_cache_add": "random_cache",
    "resid_donor_tool_same_add": "tool_same",
}
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
ROUTE_LABELS = {
    "forbidden_sentence_completion:temperature<-forbidden_definition": "sent_comp<-def",
    "forbidden_definition:top_p<-forbidden_definition": "def<-def",
}


def load_model_data(model: str) -> dict:
    path = OUT_ROOT / f"phase559_{model}_prototype_generation_closure.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_row(audit: dict, combo: str, route: str, condition: str) -> dict | None:
    try:
        return audit[combo]["rows"][route][condition]
    except (KeyError, TypeError):
        return None


def fmt(v: float | None, w: int = 6, sign: bool = False) -> str:
    if v is None:
        return " " * w
    if sign:
        return f"{v:+{w}.2f}"
    return f"{v:{w}.2f}"


def main() -> None:
    data = {m: load_model_data(m) for m in MODELS}

    lines: list[str] = []
    lines.append("# Phase 559: Prototype Generation Closure Cross-Model Summary")
    lines.append("")
    lines.append("## Object Audit (test_n=12, vehicle category)")
    lines.append("")
    lines.append("| repeat_idx | object | tokens | chars |")
    lines.append("|---|---|---|---|")
    for o in data["glm4"]["object_audit"]:
        lines.append(f"| repeat{o['repeat_index']} | {o['object']} | {o['token_length']} | {o['char_length']} |")
    lines.append("")
    lines.append(f"**Key:** repeat2 = helicopter (2 tokens), repeat4 = rocket (1 token)")
    lines.append("")

    for route in ROUTES:
        rlabel = ROUTE_LABELS[route]
        lines.append(f"## Route: {rlabel}")
        lines.append("")
        lines.append(f"Surgery mode: {data['glm4']['surgery_mode']}")
        lines.append("")

        # Main table: clean_non_object_rate
        lines.append("### clean_non_object_rate")
        lines.append("")
        header = "| model | " + " | ".join(COND_LABELS[c] for c in CONDITION_ORDER) + " |"
        sep = "|---|" + "|".join(["---"] * len(CONDITION_ORDER)) + "|"
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            audit = data[model]["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITION_ORDER:
                row = get_row(audit, combo, route, cond)
                vals.append(fmt(row["clean_non_object_rate"] if row else None))
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

        # Necessity drop and restore gain
        lines.append("### Necessity Drop (remove_perp - baseline) and Restore Gain (donor - remove_perp)")
        lines.append("")
        header2 = "| model | nec_drop | " + " | ".join(
            COND_LABELS[c] + "_rgain" if c.startswith("resid_donor") else COND_LABELS[c]
            for c in CONDITION_ORDER if c != "baseline" and c != "add_perp"
        ) + " |"
        lines.append(header2)
        lines.append("|---|" + "|".join(["---"] * (len(CONDITION_ORDER) - 2)) + "|")
        for model in MODELS:
            audit = data[model]["audit"]
            combo = list(audit.keys())[0]
            base = get_row(audit, combo, route, "baseline")
            rm = get_row(audit, combo, route, "resid_remove_perp")
            nec_drop = (rm["clean_non_object_rate"] - base["clean_non_object_rate"]) if (base and rm) else None
            vals = [fmt(nec_drop, sign=True)]
            for cond in CONDITION_ORDER:
                if cond in ("baseline", "add_perp"):
                    continue
                row = get_row(audit, combo, route, cond)
                if row and rm:
                    rg = row["clean_non_object_rate"] - rm["clean_non_object_rate"]
                    vals.append(fmt(rg, sign=True))
                else:
                    vals.append(" " * 6)
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

        # Object echo rate
        lines.append("### object_echo_rate")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            audit = data[model]["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITION_ORDER:
                row = get_row(audit, combo, route, cond)
                vals.append(fmt(row["object_echo_rate"] if row else None))
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

        # clean_non_object_score
        lines.append("### clean_non_object_score")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for model in MODELS:
            audit = data[model]["audit"]
            combo = list(audit.keys())[0]
            vals = []
            for cond in CONDITION_ORDER:
                row = get_row(audit, combo, route, cond)
                vals.append(fmt(row["clean_non_object_score"] if row else None))
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(vals) + " |")
        lines.append("")

    # Phase 558 vs Phase 559 comparison for GLM4
    lines.append("## Phase 558 (next-token margin) vs Phase 559 (generation closure) — GLM4 all L24/26/28")
    lines.append("")
    lines.append("| condition | P558 margin (sent) | P559 clean_no (sent) | P558 margin (def) | P559 clean_no (def) |")
    lines.append("|---|---|---|---|---|")
    p558_data = json.loads(Path("results/glm5_phase558_prototype_object_binding_fast/phase558_glm4_prototype_object_binding_fast.json").read_text(encoding="utf-8"))
    p558_audit = p558_data["audit"]
    p558_combo = "all"
    for cond in CONDITION_ORDER:
        p558_sent = get_row(p558_audit, p558_combo, "forbidden_sentence_completion:temperature<-forbidden_definition", cond)
        p558_def = get_row(p558_audit, p558_combo, "forbidden_definition:top_p<-forbidden_definition", cond)
        p559_audit = data["glm4"]["audit"]
        p559_sent = get_row(p559_audit, "all", "forbidden_sentence_completion:temperature<-forbidden_definition", cond)
        p559_def = get_row(p559_audit, "all", "forbidden_definition:top_p<-forbidden_definition", cond)
        p558_s = fmt(p558_sent["target_margin_mean"] if p558_sent else None, sign=True)
        p559_s = fmt(p559_sent["clean_non_object_rate"] if p559_sent else None)
        p558_d = fmt(p558_def["target_margin_mean"] if p558_def else None, sign=True)
        p559_d = fmt(p559_def["clean_non_object_rate"] if p559_def else None)
        lines.append(f"| {COND_LABELS[cond]} | {p558_s} | {p559_s} | {p558_d} | {p559_d} |")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append("| model | time (min) | attn |")
    lines.append("|---|---|---|")
    for model in MODELS:
        lines.append(f"| {MODEL_LABELS[model]} | {data[model]['total_time_min']} | {data[model]['attn_implementation']} |")
    lines.append("")

    out_path = OUT_ROOT / "phase559_cross_model_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
