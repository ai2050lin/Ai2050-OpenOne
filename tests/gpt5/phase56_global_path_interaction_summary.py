from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]
RELATIONS = ["binding", "negation", "antonym", "role", "tense", "same_class"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def summarize_relation(rel_data: dict[str, Any]) -> dict[str, float]:
    if "balance_mean" in rel_data and "net_gross_mean" in rel_data:
        return {
            "balance_mean": fnum(rel_data.get("balance_mean")),
            "balance_std": fnum(rel_data.get("balance_std")),
            "net_gross_mean": fnum(rel_data.get("net_gross_mean")),
            "net_gross_std": fnum(rel_data.get("net_gross_std")),
            "n": int(fnum(rel_data.get("n_observations"))),
        }

    # Fallback for older per-layer-only outputs. Weight layer means by n.
    balances: list[float] = []
    net_gross: list[float] = []
    for layer_data in rel_data.get("per_layer", rel_data).values():
        if isinstance(layer_data, dict):
            n = int(fnum(layer_data.get("n"), 1.0)) or 1
            if "balance_mean" in layer_data:
                balances.extend([fnum(layer_data.get("balance_mean"))] * n)
            if "net_gross_mean" in layer_data:
                net_gross.extend([fnum(layer_data.get("net_gross_mean"))] * n)
    return {
        "balance_mean": mean(balances) if balances else 0.0,
        "balance_std": pstdev(balances) if len(balances) > 1 else 0.0,
        "net_gross_mean": mean(net_gross) if net_gross else 0.0,
        "net_gross_std": pstdev(net_gross) if len(net_gross) > 1 else 0.0,
        "n": len(net_gross),
    }


def summarize_model(model: str, phase344: dict[str, Any], phase346: dict[str, Any]) -> dict[str, Any]:
    relation_summary = {}
    for rel in RELATIONS:
        if rel in phase344.get("relation_results", {}):
            relation_summary[rel] = summarize_relation(phase344["relation_results"][rel])

    controls = {}
    for name, rows in phase344.get("control_summary", {}).items():
        if isinstance(rows, dict):
            controls[name] = {
                "net_gross_mean": fnum(rows.get("net_gross")),
                "balance_mean": fnum(rows.get("balance")),
                "n": int(fnum(rows.get("n"))),
            }
        else:
            ng = [fnum(row.get("net_gross_mean")) for row in rows] if isinstance(rows, list) else []
            bal = [fnum(row.get("balance_mean")) for row in rows] if isinstance(rows, list) else []
            controls[name] = {
                "net_gross_mean": mean(ng) if ng else 0.0,
                "balance_mean": mean(bal) if bal else 0.0,
                "n": len(ng),
            }

    inter344 = phase344.get("interaction_summary", {})
    inter346 = phase346.get("interaction_summary", {})
    closure = phase346.get("closure_summary", {})

    binding_ng = relation_summary.get("binding", {}).get("net_gross_mean", 0.0)
    other_ng = [
        relation_summary[rel]["net_gross_mean"]
        for rel in RELATIONS
        if rel != "binding" and rel in relation_summary
    ]
    binding_relative_rank = 1 + sum(1 for val in other_ng if val > binding_ng)

    return {
        "model": model,
        "relations": relation_summary,
        "controls": controls,
        "binding_net_gross_rank_among_relations": binding_relative_rank,
        "phase344_interaction": {
            "gate_main_pct_mean": fnum(inter344.get("gate_main_pct_mean")),
            "up_main_pct_mean": fnum(inter344.get("up_main_pct_mean")),
            "interaction_pct_mean": fnum(inter344.get("interaction_pct_mean")),
        },
        "phase346_interaction": {
            "gate_main_frac": fnum(inter346.get("gate_main_frac")),
            "up_main_frac": fnum(inter346.get("up_main_frac")),
            "interaction_frac": fnum(inter346.get("interaction_frac")),
            "total_effect_mean": fnum(inter346.get("total_effect_mean")),
        },
        "closure": {
            "mean_closure_ratio": fnum(closure.get("mean_closure_ratio")),
            "mean_final_binding": fnum(closure.get("mean_final_binding")),
            "mean_mlp_net_sum": fnum(closure.get("mean_mlp_net_sum")),
        },
    }


def write_markdown(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# Phase56 Global Path + MLP Interaction Summary", ""]
    lines.append("## Relation Path Comparison")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s['model']}")
        lines.append("")
        lines.append("| relation | balance | net/gross | n |")
        lines.append("|---|---:|---:|---:|")
        for rel in RELATIONS:
            if rel not in s["relations"]:
                continue
            r = s["relations"][rel]
            lines.append(
                f"| {rel} | {r['balance_mean']:.4f} | {r['net_gross_mean']:.4f} | {r['n']} |"
            )
        lines.append("")
        lines.append(f"binding_net_gross_rank_among_relations={s['binding_net_gross_rank_among_relations']}")
        lines.append("")
        lines.append("## MLP Gate/Up/Interaction")
        lines.append("")
        p = s["phase346_interaction"]
        lines.append(
            f"{s['model']}: gate={p['gate_main_frac']:.4f}, "
            f"up={p['up_main_frac']:.4f}, interaction={p['interaction_frac']:.4f}, "
            f"total_effect={p['total_effect_mean']:.4f}"
        )
        c = s["closure"]
        lines.append(
            f"closure: ratio={c['mean_closure_ratio']:.4f}, "
            f"final={c['mean_final_binding']:.4f}, mlp_sum={c['mean_mlp_net_sum']:.4f}"
        )
        lines.append("")

    output_dir.joinpath("PHASE56_GLOBAL_PATH_INTERACTION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase344-dir", required=True)
    parser.add_argument("--phase346-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    phase344_dir = Path(args.phase344_dir)
    phase346_dir = Path(args.phase346_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in MODELS:
        phase344 = load_json(phase344_dir / f"{model}_phase344_345.json")
        phase346 = load_json(phase346_dir / f"{model}_phase346.json")
        summaries.append(summarize_model(model, phase344, phase346))

    out = {"models": {s["model"]: s for s in summaries}}
    output_dir.joinpath("phase56_global_path_interaction_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir, summaries)
    print(f"wrote {output_dir / 'phase56_global_path_interaction_summary.json'}")
    print(f"wrote {output_dir / 'PHASE56_GLOBAL_PATH_INTERACTION_SUMMARY.md'}")


if __name__ == "__main__":
    main()
