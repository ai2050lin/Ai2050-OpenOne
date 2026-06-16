#!/usr/bin/env python3
"""Cross-model summary for Phase 507 orthogonal semantic field results."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("results/glm5")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def avg(vals: list[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def load(model: str) -> dict[str, Any]:
    path = ROOT / f"phase507_{model}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def final_layer_key(data: dict[str, Any]) -> str:
    return str(data["L"])


def best_remove_delta(data: dict[str, Any], cat: str) -> tuple[str, float]:
    rows = data["exp2_midlayer_intervention"][cat]
    best_l = ""
    best_v = 0.0
    for layer, item in rows.items():
        v = float(item.get("remove_perp_delta_D", 0.0))
        if not best_l or v < best_v:
            best_l, best_v = layer, v
    return best_l, best_v


def strongest_positive_delta(data: dict[str, Any], cat: str) -> tuple[str, float]:
    rows = data["exp2_midlayer_intervention"][cat]
    best_l = ""
    best_v = 0.0
    for layer, item in rows.items():
        v = float(item.get("remove_perp_delta_D", 0.0))
        if not best_l or v > best_v:
            best_l, best_v = layer, v
    return best_l, best_v


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Phase507 Orthogonal Semantic Field Cross-model Summary", ""]
    all_data = {m: load(m) for m in MODELS if (ROOT / f"phase507_{m}.json").exists()}

    for model, data in all_data.items():
        cats = data["categories"]
        fl = final_layer_key(data)
        exp1 = data["exp1_orthogonal_decomposition"]
        exp3 = data["exp3_functional_probes"]
        exp6 = data["exp6_token_level_output"]

        final_ratios = [float(exp1[c][fl]["perp_para_ratio_mean"]) for c in cats]
        final_cos = [abs(float(exp1[c][fl]["cos_phi_qc_mean"])) for c in cats]
        final_n90 = [float(exp1[c][fl]["pca_n_90"]) for c in cats]
        final_perp_norm = [float(exp1[c][fl]["phi_perp_norm_mean"]) for c in cats]
        probe_last = exp3[str(max(int(k) for k in exp3.keys()))]
        token_rates = [float(exp6[c]["rich"]["cat_argmax_rate"]) for c in cats]

        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"L={data['L']}, d={data['d_model']}, categories={len(cats)}, "
            f"objects/category={data['n_objects_per_cat']}"
        )
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---:|")
        lines.append(f"| final mean perp/para ratio | {avg(final_ratios):.4f} |")
        lines.append(f"| final mean abs cos(phi,qc) | {avg(final_cos):.6f} |")
        lines.append(f"| final mean pca_n90 | {avg(final_n90):.4f} |")
        lines.append(f"| final mean phi_perp_norm | {avg(final_perp_norm):.4f} |")
        lines.append(f"| last-probe category acc para | {probe_last['category_accuracy']['para']:.4f} |")
        lines.append(f"| last-probe category acc perp | {probe_last['category_accuracy']['perp']:.4f} |")
        lines.append(f"| last-probe tc-mode acc para | {probe_last['tc_mode_accuracy']['para']:.4f} |")
        lines.append(f"| last-probe tc-mode acc perp | {probe_last['tc_mode_accuracy']['perp']:.4f} |")
        lines.append(f"| mean rich category argmax | {avg(token_rates):.4f} |")
        lines.append("")

        lines.append("### Category Details")
        lines.append("")
        lines.append("| category | final ratio | final cos | n90 | best rm_perp ΔD | best layer | strongest positive ΔD | pos layer | rich argmax |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cat in cats:
            best_l, best_v = best_remove_delta(data, cat)
            pos_l, pos_v = strongest_positive_delta(data, cat)
            item = exp1[cat][fl]
            argmax = float(exp6[cat]["rich"]["cat_argmax_rate"])
            lines.append(
                f"| {cat} | {float(item['perp_para_ratio_mean']):.4f} | "
                f"{float(item['cos_phi_qc_mean']):.6f} | {float(item['pca_n_90']):.0f} | "
                f"{best_v:.4f} | L{best_l} | {pos_v:.4f} | L{pos_l} | {argmax:.4f} |"
            )
        lines.append("")

    if all_data:
        lines.append("## Cross-model Takeaways")
        lines.append("")
        lines.append("| model | mean ratio | mean abs cos | probe perp cat acc | probe perp tc acc | mean rich argmax |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model, data in all_data.items():
            cats = data["categories"]
            fl = final_layer_key(data)
            exp1 = data["exp1_orthogonal_decomposition"]
            exp3 = data["exp3_functional_probes"]
            exp6 = data["exp6_token_level_output"]
            probe_last = exp3[str(max(int(k) for k in exp3.keys()))]
            lines.append(
                f"| {model} | "
                f"{avg([float(exp1[c][fl]['perp_para_ratio_mean']) for c in cats]):.4f} | "
                f"{avg([abs(float(exp1[c][fl]['cos_phi_qc_mean'])) for c in cats]):.6f} | "
                f"{probe_last['category_accuracy']['perp']:.4f} | "
                f"{probe_last['tc_mode_accuracy']['perp']:.4f} | "
                f"{avg([float(exp6[c]['rich']['cat_argmax_rate']) for c in cats]):.4f} |"
            )
        lines.append("")

    out = ROOT / "phase507_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
