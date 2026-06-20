#!/usr/bin/env python3
"""Summary for Phase560 continuous prototype surgery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/glm5_phase560_continuous_prototype_surgery")
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]
CONDITIONS = [
    "baseline",
    "resid_remove_perp",
    "resid_donor_vehicle_repeat0_add",
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_repeat10_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_pca1_cache_add",
    "resid_donor_vehicle_random_cache_add",
]
LABELS = {
    "baseline": "base",
    "resid_remove_perp": "remove",
    "resid_donor_vehicle_repeat0_add": "repeat0(tram)",
    "resid_donor_vehicle_repeat2_add": "repeat2(heli)",
    "resid_donor_vehicle_repeat4_add": "repeat4(rocket)",
    "resid_donor_vehicle_repeat10_add": "repeat10(sled)",
    "resid_donor_vehicle_mean_cache_add": "mean",
    "resid_donor_vehicle_pca1_cache_add": "pca1",
    "resid_donor_vehicle_random_cache_add": "random",
}


def load(root: Path, model: str, mode: str) -> dict[str, Any] | None:
    path = root / f"phase560_{model}_{mode}_prototype_surgery.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row(data: dict[str, Any], route: str, cond: str) -> dict[str, Any] | None:
    combo = list(data["audit"].keys())[0]
    return data["audit"].get(combo, {}).get("rows", {}).get(route, {}).get(cond)


def val(data: dict[str, Any], route: str, cond: str, field: str = "clean_non_object_rate") -> float | None:
    r = row(data, route, cond)
    return None if r is None else float(r[field])


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--mode", default="continuous_static")
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m, args.mode) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = [f"# Phase560 Continuous Prototype Surgery Summary ({args.mode})", ""]
    if "glm4" in data:
        lines.append("## Object Audit")
        lines.append("")
        lines.append("| repeat | object | tokens | chars |")
        lines.append("|---|---|---:|---:|")
        for o in data["glm4"].get("object_audit", []):
            lines.append(f"| repeat{o['repeat_index']} | {o['object']} | {o['token_length']} | {o['char_length']} |")
        lines.append("")

    for route in ROUTES:
        lines.append(f"## Route: {route}")
        lines.append("")
        lines.append("### clean_non_object_rate")
        lines.append("")
        lines.append("| model | " + " | ".join(LABELS[c] for c in CONDITIONS) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(CONDITIONS)) + "|")
        for model in MODELS:
            if model not in data:
                continue
            vals = [fmt(val(data[model], route, c)) for c in CONDITIONS]
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        lines.append("### steering gain versus baseline")
        lines.append("")
        lines.append("| model | " + " | ".join(LABELS[c] for c in CONDITIONS[1:]) + " |")
        lines.append("|---|" + "|".join(["---:"] * (len(CONDITIONS) - 1)) + "|")
        for model in MODELS:
            if model not in data:
                continue
            base = val(data[model], route, "baseline")
            vals = []
            for c in CONDITIONS[1:]:
                x = val(data[model], route, c)
                vals.append("" if x is None or base is None else f"{x - base:+.2f}")
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")

        lines.append("### object_echo_rate")
        lines.append("")
        lines.append("| model | " + " | ".join(LABELS[c] for c in CONDITIONS) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(CONDITIONS)) + "|")
        for model in MODELS:
            if model not in data:
                continue
            vals = [fmt(val(data[model], route, c, "object_echo_rate")) for c in CONDITIONS]
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

    out = root / f"phase560_{args.mode}_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
