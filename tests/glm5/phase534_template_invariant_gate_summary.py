#!/usr/bin/env python3
"""Summary for Phase534 template-invariant gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase534_template_invariant_gate")
MODELS = ["qwen3", "glm4", "deepseek7b"]
KEY_COMPONENTS = [
    "category_common_perp",
    "category_direct_perp",
    "category_belongs_perp",
    "category_kind_perp",
    "category_direct_residual",
    "category_belongs_residual",
    "category_kind_residual",
    "color_red_blue_perp",
    "object_car_truck_perp",
]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase534_{model}_template_invariant_gate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def own_cell(adm: dict[str, Any]) -> str:
    own = adm["own"]
    return f"{float(own['best_own_delta']):+.3f}/{float(own['best_selectivity_ratio']):.2f}/{'Y' if own['passes_strict_gate'] else 'n'}"


def transfer_cell(adm: dict[str, Any] | None) -> str:
    if adm is None or adm.get("transfer") is None:
        return "-"
    tr = adm["transfer"]
    return (
        f"{float(tr['best_category_min_delta']):+.3f}/"
        f"{float(tr['best_category_mean_delta']):+.3f}/"
        f"{float(tr['best_transfer_ratio']):.2f}/"
        f"{'Y' if tr['passes_transfer_gate'] else 'n'}"
    )


def bridge_cell(row: dict[str, Any]) -> str:
    return (
        f"hit={float(row['target_hit_rate']):.2f}, "
        f"path={float(row['semantic_path_hit_rate']):.2f}, "
        f"rank={float(row['mean_best_target_rank']):.1f}, "
        f"m1={float(row['mean_first_step_margin']):+.3f}"
    )


def verdict(d: dict[str, Any]) -> str:
    common = d["admission"]["category_common_perp"]["transfer"]
    direct = d["admission"]["category_direct_perp"]["own"]
    random = d["random_common_summary"]
    cum = d["cumulative_admission"]
    if common["passes_transfer_gate"] and int(random["strict_transfer_pass_count"]) == 0:
        return "common_transfer_candidate"
    if cum["passes_transfer_gate"] and not common["passes_transfer_gate"]:
        return "multi_layer_common_only"
    if direct["passes_strict_gate"] and not common["passes_transfer_gate"]:
        return "direct_template_path"
    return "no_category_common"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines: list[str] = ["# Phase534 Template-Invariant Gate Summary", ""]
    compact = []
    for model, d in data.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"layers={d['layers']}, primary=L{d['primary_layer']}, train_n={d['train_n']}, "
            f"test_n={d['test_n']}, bridge_n={d['bridge_n']}, max_new_tokens={d['max_new_tokens']}, "
            f"alphas={d['alphas']}, cumulative_alphas={d['cumulative_alphas']}, "
            f"seeds={d['random_seeds']}, attn={d.get('attn_implementation', 'unknown')}"
        )
        lines.append("")
        lines.append("Own format: own delta / selectivity ratio / strict. Transfer format: min / mean / ratio / pass.")
        lines.append("")
        lines.append("| component | own | category transfer |")
        lines.append("|---|---:|---:|")
        for comp in KEY_COMPONENTS:
            if comp in d["admission"]:
                lines.append(f"| {comp} | {own_cell(d['admission'][comp])} | {transfer_cell(d['admission'][comp])} |")
        lines.append("")

        lines.append("### Common Random And Cumulative")
        lines.append("")
        r = d["random_common_summary"]
        c = d["cumulative_admission"]
        lines.append(
            f"random_common max transfer-min={float(r['max_transfer_min_delta']):+.3f}, "
            f"strict transfer pass count={int(r['strict_transfer_pass_count'])}"
        )
        lines.append("")
        lines.append(
            f"cumulative common transfer-min={float(c['best_category_min_delta']):+.3f}, "
            f"mean={float(c['best_category_mean_delta']):+.3f}, "
            f"ratio={float(c['best_transfer_ratio']):.2f}, "
            f"pass={'Y' if c['passes_transfer_gate'] else 'n'}, alpha={c['best_alpha']}"
        )
        lines.append("")

        lines.append("### Category Template Cosines At Primary Layer")
        lines.append("")
        stats = d["layer_stats"][str(d["primary_layer"])]
        names = sorted(stats["category_template_cosines"])
        lines.append("| dir | " + " | ".join(names) + " | cos_to_common | residual_norm_pct |")
        lines.append("|---|" + "|".join(["---:"] * len(names)) + "|---:|---:|")
        for a in names:
            vals = [float(stats["category_template_cosines"][a][b]) for b in names]
            lines.append(
                f"| {a} | " + " | ".join(f"{v:+.4f}" for v in vals)
                + f" | {float(stats[a + '_cos_to_common']):+.4f} | "
                + f"{float(stats[a + '_residual_norm_pct']):.2f} |"
            )
        lines.append("")

        lines.append("### Generation Bridge")
        lines.append("")
        lines.append("| condition | trace |")
        lines.append("|---|---|")
        for name, row in d["generation_bridge"].items():
            lines.append(f"| {name} | {bridge_cell(row)} |")
        lines.append("")

        compact.append({"model": model, "verdict": verdict(d)})

    if compact:
        lines.append("## Cross-model Compact")
        lines.append("")
        lines.append("| model | verdict |")
        lines.append("|---|---|")
        for row in compact:
            lines.append(f"| {row['model']} | {row['verdict']} |")
        lines.append("")

    out = root / "phase534_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
