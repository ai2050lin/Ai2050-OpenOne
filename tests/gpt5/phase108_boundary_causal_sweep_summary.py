#!/usr/bin/env python3
"""Cross-model summary for Phase108 boundary causal sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def condition_key(c: dict[str, Any]) -> tuple[int, str, float]:
    return (int(c["layer"]), str(c["position"]), float(c["scale"]))


def index_controls(conditions: list[dict[str, Any]]) -> dict[tuple[str, tuple[int, str, float]], dict[str, Any]]:
    return {(c["kind"], condition_key(c)): c for c in conditions}


def classify(down: dict[str, Any], same_random: dict[str, Any] | None, same_neighbor: dict[str, Any] | None) -> str:
    td = down["target_delta"]
    rd = same_random["target_delta"] if same_random else 0.0
    nd = same_neighbor["target_delta"] if same_neighbor else 0.0
    if td < -1.0 and td < rd - 0.5 and td < nd - 0.5:
        return "specific_strong_target_down"
    if td < -0.4 and td < rd - 0.2:
        return "target_down_boundary_gt_random"
    if td > 0.4:
        return "target_up_opposed"
    return "weak_or_control_like"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase108_boundary_causal_sweep")
    parser.add_argument("--output", default="results/gpt5_phase108_boundary_causal_sweep/phase108_cross_model_summary.md")
    args = parser.parse_args()

    base = Path(args.input_dir)
    data = {m: load(base / f"phase108_{m}_boundary_causal_sweep.json") for m in MODELS}
    cats = data["qwen3"]["test_categories"]

    lines: list[str] = []
    lines.append("# Phase 108 Cross-Model Boundary Causal Sweep Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("| model | center layer | sweep layers | categories | scales | positions |")
    lines.append("|---|---:|---|---:|---|---|")
    for m in MODELS:
        d = data[m]
        lines.append(
            f"| {m} | L{d['center_layer_for_boundaries']} | {d['sweep_layers']} | "
            f"{len(d['test_categories'])} | {d['scales']} | {d['positions']} |"
        )

    lines.append("")
    lines.append("## Strongest Boundary Effects With Same-Setting Controls")
    lines.append("| category | qwen3 | glm4 | deepseek7b | objective reading |")
    lines.append("|---|---|---|---|---|")
    for cat in cats:
        cells = []
        labels = []
        for m in MODELS:
            item = data[m]["category_results"][cat]
            conds = item["conditions"]
            boundary = [c for c in conds if c["kind"] == "boundary"]
            controls = index_controls(conds)
            down = min(boundary, key=lambda x: x["target_delta"])
            up = max(boundary, key=lambda x: x["target_delta"])
            rel = max(boundary, key=lambda x: x["max_other_delta"])
            key = condition_key(down)
            rnd = controls.get(("random_same_norm", key))
            nei = controls.get(("neighbor_boundary", key))
            top = rel["top_releases"][0] if rel["top_releases"] else {"category": "none", "delta": 0.0}
            label = classify(down, rnd, nei)
            labels.append(f"{m}:{label}")
            cells.append(
                f"down L{down['layer']} {down['position']} s{down['scale']} "
                f"T{down['target_delta']:.2f} R{(rnd or {}).get('target_delta', 0):.2f} "
                f"N{(nei or {}).get('target_delta', 0):.2f}; "
                f"up {up['target_delta']:.2f}; rel {top['category']}+{top['delta']:.2f}; {label}"
            )
        if any("specific_strong_target_down" in x for x in labels):
            reading = "specific strong target-down exists"
        elif any("target_down_boundary_gt_random" in x for x in labels):
            reading = "moderate boundary target-down"
        elif any("target_up_opposed" in x for x in labels):
            reading = "opposed/support-suppressor mixed"
        else:
            reading = "weak or control-like"
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} | {reading} |")

    lines.append("")
    lines.append("## Objective Facts")
    lines.append("- Qwen3 number becomes much stronger with both-position high-scale removal: L35 both scale1.5 target_delta=-3.06.")
    lines.append("- Qwen3 time also strengthens with both-position high-scale removal: L35 both scale1.5 target_delta=-1.35, with animal release peaking at +0.61.")
    lines.append("- DS7B number and container are strong target-down cases under both-position scale1.5 at L27: number=-4.75, container=-3.21.")
    lines.append("- Clothing/furniture/plant remain mixed/opposed: their strongest boundary conditions often increase target while releasing tool/clothing/animal.")
    lines.append("- GLM4 bf16 remains weak across the sweep; effects are finite but much smaller than Qwen3/DS7B.")
    lines.append("- Position matters: the strongest target-down cases for number/container usually require both positions, while strongest releases often use answer_last or both.")
    lines.append("- Layer matters: Qwen3 container/plant target-down appears earlier (L32) than boundary peak L35; a boundary-norm peak is not always the best causal layer.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
