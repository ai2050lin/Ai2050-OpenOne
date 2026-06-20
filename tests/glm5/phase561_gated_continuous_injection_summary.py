#!/usr/bin/env python3
"""Summary for Phase561 gated continuous injection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase561_gated_continuous_injection")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
DONORS = [
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_random_cache_add",
]
DONOR_LABELS = {
    "resid_donor_vehicle_repeat2_add": "repeat2(helicopter)",
    "resid_donor_vehicle_repeat4_add": "repeat4(rocket)",
    "resid_donor_vehicle_mean_cache_add": "mean",
    "resid_donor_vehicle_random_cache_add": "random",
}


def load(root: Path, model: str, mode: str) -> dict[str, Any] | None:
    path = root / f"phase561_{model}_{mode}_gated_injection.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def combo_name(data: dict[str, Any]) -> str:
    return list(data["audit"].keys())[0]


def row(data: dict[str, Any], route: str, key: str) -> dict[str, Any] | None:
    return data["audit"].get(combo_name(data), {}).get("rows", {}).get(route, {}).get(key)


def val(data: dict[str, Any], route: str, key: str, field: str) -> float | None:
    r = row(data, route, key)
    return None if r is None else float(r.get(field, 0.0))


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}"


def key_for(donor: str, beta: float) -> str:
    return f"{donor}__b{str(beta).replace('.', 'p')}"


def best_donor(data: dict[str, Any], route: str, donor: str, betas: list[float]) -> tuple[float | None, float | None, float | None]:
    best_beta: float | None = None
    best_clean: float | None = None
    best_format: float | None = None
    for beta in betas:
        key = key_for(donor, beta)
        clean = val(data, route, key, "clean_non_object_rate")
        if clean is None:
            continue
        if best_clean is None or clean > best_clean:
            best_clean = clean
            best_beta = beta
            best_format = val(data, route, key, "format_break_rate")
    return best_beta, best_clean, best_format


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--mode", default="continuous_blend")
    args = parser.parse_args()

    root = Path(args.root)
    data = {m: load(root, m, args.mode) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    betas = list(next(iter(data.values())).get("blend_betas", [0.05, 0.10, 0.25, 0.50])) if data else []

    lines = [f"# Phase561 Gated Continuous Injection Summary ({args.mode})", ""]

    if data:
        first = next(iter(data.values()))
        lines.append("## Object Audit")
        lines.append("")
        lines.append("| repeat | object | tokens | chars |")
        lines.append("|---|---|---:|---:|")
        for o in first.get("object_audit", []):
            if o["repeat_index"] in {2, 4}:
                lines.append(f"| repeat{o['repeat_index']} | {o['object']} | {o['token_length']} | {o['char_length']} |")
        lines.append("")

    for route in ROUTES:
        lines.append(f"## Route: {route}")
        lines.append("")
        lines.append("### Baseline And Remove")
        lines.append("")
        lines.append("| model | baseline clean | remove clean | remove delta | baseline fmt | remove fmt |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = val(data[model], route, "baseline", "clean_non_object_rate")
            rm = val(data[model], route, "resid_remove_perp", "clean_non_object_rate")
            base_fmt = val(data[model], route, "baseline", "format_break_rate")
            rm_fmt = val(data[model], route, "resid_remove_perp", "format_break_rate")
            delta = None if base is None or rm is None else rm - base
            lines.append(f"| {model} | {fmt(base)} | {fmt(rm)} | {fmt(delta)} | {fmt(base_fmt)} | {fmt(rm_fmt)} |")
        lines.append("")

        lines.append("### Best Beta Per Donor")
        lines.append("")
        lines.append("| model | donor | best beta | best clean | gain vs base | format break |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for model in MODELS:
            if model not in data:
                continue
            base = val(data[model], route, "baseline", "clean_non_object_rate")
            for donor in DONORS:
                beta, clean, fmt_rate = best_donor(data[model], route, donor, betas)
                gain = None if base is None or clean is None else clean - base
                lines.append(f"| {model} | {DONOR_LABELS[donor]} | {'' if beta is None else beta} | {fmt(clean)} | {fmt(gain)} | {fmt(fmt_rate)} |")
        lines.append("")

        for donor in DONORS:
            lines.append(f"### {DONOR_LABELS[donor]} Clean By Beta")
            lines.append("")
            lines.append("| model | " + " | ".join(str(b) for b in betas) + " |")
            lines.append("|---|" + "|".join(["---:"] * len(betas)) + "|")
            for model in MODELS:
                if model not in data:
                    continue
                vals = [fmt(val(data[model], route, key_for(donor, b), "clean_non_object_rate")) for b in betas]
                lines.append(f"| {model} | " + " | ".join(vals) + " |")
            lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("| model | minutes | seeds | max tokens |")
    lines.append("|---|---:|---|---:|")
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        lines.append(f"| {model} | {d.get('total_time_min')} | {d.get('sample_seeds')} | {d.get('max_new_tokens')} |")
    lines.append("")

    out = root / f"phase561_{args.mode}_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
