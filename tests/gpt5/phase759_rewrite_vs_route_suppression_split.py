#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

MODELS = ("qwen3", "glm4", "deepseek7b")
IN_ROOT = Path("results/glm5_phase758_late_carrier_rewrite_relabel")
OUT_ROOT = Path("results/glm5_phase759_rewrite_vs_route_suppression_split")


def safe_mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def target_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    erase = float(row.get("erase_target_logit_drop") or 0.0)
    recovered = float(row.get("target_logit_recovered_by_restore") or 0.0)
    frac = row.get("target_recovery_fraction")
    frac = float(frac) if frac is not None else 0.0
    return erase >= args.min_erase_drop and recovered >= args.min_target_recovery and frac >= args.min_target_fraction


def route_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    erase_release = float(row.get("erase_total_positive_route_release") or 0.0)
    reduced = float(row.get("route_release_reduced_by_restore") or 0.0)
    restored_release = float(row.get("restored_total_positive_route_release") or 0.0)
    return (
        erase_release >= args.min_erase_route_release
        and reduced >= args.min_route_reduced
        and restored_release <= max(0.0, erase_release - args.min_route_reduced)
    )


def split_role(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    if not rows:
        return "empty"
    target_rate = sum(target_success(r, args) for r in rows) / len(rows)
    route_rate = sum(route_success(r, args) for r in rows) / len(rows)
    mean_recovered = safe_mean([r.get("target_logit_recovered_by_restore") for r in rows]) or 0.0
    mean_reduced = safe_mean([r.get("route_release_reduced_by_restore") for r in rows]) or 0.0
    if target_rate >= args.strong_rate and route_rate >= args.route_rate:
        return "joint_target_and_route_candidate"
    if target_rate >= args.strong_rate and mean_recovered >= args.min_target_recovery:
        return "target_rewrite_only_candidate"
    if route_rate >= args.route_rate and mean_reduced >= args.min_route_reduced:
        return "route_suppression_only_candidate"
    if target_rate >= args.weak_rate or route_rate >= args.weak_rate:
        return "weak_split_signal"
    if mean_recovered < -args.min_target_recovery:
        return "anti_target_restore"
    return "weak_or_unclear"


def summarize_group(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "target_success_rate": sum(target_success(r, args) for r in rows) / n if n else 0.0,
        "route_success_rate": sum(route_success(r, args) for r in rows) / n if n else 0.0,
        "mean_erase_target_logit_drop": safe_mean([r.get("erase_target_logit_drop") for r in rows]),
        "mean_target_recovered": safe_mean([r.get("target_logit_recovered_by_restore") for r in rows]),
        "mean_target_recovery_fraction": safe_mean([r.get("target_recovery_fraction") for r in rows]),
        "mean_erase_route_release": safe_mean([r.get("erase_total_positive_route_release") for r in rows]),
        "mean_route_release_reduced": safe_mean([r.get("route_release_reduced_by_restore") for r in rows]),
        "mean_restored_route_release": safe_mean([r.get("restored_total_positive_route_release") for r in rows]),
        "split_role_guess": split_role(rows, args),
    }


def analyze_model(model: str, round_name: str, args: argparse.Namespace) -> dict[str, Any]:
    path = IN_ROOT / round_name / f"phase758_{model}_rows.jsonl"
    rows = [r for r in read_jsonl(path) if r.get("row_kind") == "multisite_restore"]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_combo: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["combo_kind"]].append(row)
        by_combo[(row["site"], int(row["head"]), row["source_group"], row["combo_name"], row["combo_kind"])].append(row)

    combo_kind_summary = {
        kind: {**summarize_group(vals, args), "role_counts": dict(Counter(split_role([v], args) for v in vals))}
        for kind, vals in sorted(by_kind.items())
    }

    combo_summary = []
    for (site, head, source, combo_name, combo_kind), vals in by_combo.items():
        first = vals[0]
        combo_summary.append(
            {
                "site": site,
                "head": head,
                "subunit_id": f"{site}:H{head}",
                "source_group": source,
                "combo_name": combo_name,
                "combo_kind": combo_kind,
                "combo_sites": first.get("combo_sites"),
                "candidate_kind": first.get("candidate_kind"),
                "domains": sorted({v.get("domain") for v in vals}),
                "relations": sorted({v.get("relation") for v in vals}),
                **summarize_group(vals, args),
            }
        )
    combo_summary.sort(
        key=lambda r: (
            r["target_success_rate"],
            r["route_success_rate"],
            r.get("mean_target_recovered") or 0.0,
            r.get("mean_route_release_reduced") or 0.0,
        ),
        reverse=True,
    )
    return {
        "model": model,
        "round": round_name,
        "n_rows": len(rows),
        "combo_kind_summary": combo_kind_summary,
        "top_split_candidates": combo_summary[:80],
    }


def write_markdown(payload: dict[str, Any], out_dir: Path) -> None:
    lines = [
        f"# Phase 759 Rewrite vs Route Suppression Split ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Source: Phase 758 restore rows; no model inference in this phase.",
        "",
        "## Combo Kind Split",
        "",
        "| model | combo kind | n | target rate | route rate | target recovered | route reduced | role |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        for kind, row in summary["combo_kind_summary"].items():
            lines.append(
                f"| {model} | `{kind}` | {row['n']} | "
                f"{row['target_success_rate']:.3f} | {row['route_success_rate']:.3f} | "
                f"{(row.get('mean_target_recovered') or 0):.3f} | "
                f"{(row.get('mean_route_release_reduced') or 0):.3f} | "
                f"`{row['split_role_guess']}` |"
            )
    lines.extend(
        [
            "",
            "## Top Split Candidates",
            "",
            "| model | writer | source | combo | kind | n | target rate | route rate | recovered | route reduced | role |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model, summary in payload["by_model"].items():
        for row in summary["top_split_candidates"][:20]:
            lines.append(
                f"| {model} | {row['subunit_id']} | {row['source_group']} | `{row['combo_name']}` | `{row['combo_kind']}` | "
                f"{row['n']} | {row['target_success_rate']:.3f} | {row['route_success_rate']:.3f} | "
                f"{(row.get('mean_target_recovered') or 0):.3f} | "
                f"{(row.get('mean_route_release_reduced') or 0):.3f} | `{row['split_role_guess']}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Target recovery and route suppression are scored separately.",
            "- A target-rewrite candidate is not a suppressor unless route success also rises.",
            "- This is an offline split of Phase 758 rows, not new causal intervention.",
            "",
        ]
    )
    (out_dir / "phase759_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    parser.add_argument("--min-erase-drop", type=float, default=0.20)
    parser.add_argument("--min-target-recovery", type=float, default=0.10)
    parser.add_argument("--min-target-fraction", type=float, default=0.25)
    parser.add_argument("--min-erase-route-release", type=float, default=0.10)
    parser.add_argument("--min-route-reduced", type=float, default=0.05)
    parser.add_argument("--strong-rate", type=float, default=0.35)
    parser.add_argument("--route-rate", type=float, default=0.25)
    parser.add_argument("--weak-rate", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model = {}
    for model in MODELS:
        by_model[model] = analyze_model(model, args.round_name, args)
    payload = {
        "phase": 759,
        "title": "Rewrite vs Route Suppression Split",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "models": list(MODELS),
        "thresholds": vars(args),
        "by_model": by_model,
    }
    write_json(out_dir / "phase759_cross_model_summary.json", payload)
    write_markdown(payload, out_dir)
    print(json.dumps({"status": payload["status"], "models": payload["models"], "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
