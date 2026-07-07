#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


PHASE = 231
SOURCE_PHASE = 230
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase230_readout_threshold_barrier_analysis")
RESULT_ROOT = Path("tests/result/phase231_competitor_pressure_oracle_suppression")
DEFAULT_BUDGETS = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def analyze_model(model: str, input_round: str, out_dir: Path, budgets: list[float]) -> dict[str, Any]:
    rows = iter_jsonl(INPUT_ROOT / input_round / f"phase230_{model}_barrier_rows.jsonl")
    suppression_rows: list[dict[str, Any]] = []
    for row in rows:
        margin = finite_float(row.get("target_margin_vs_winner"))
        gap = finite_float(row.get("remaining_margin_gap"), -margin)
        for budget in budgets:
            post_margin = margin + budget
            suppression_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase231_oracle_competitor_suppression_row",
                    "model": model,
                    "spec_id": row.get("spec_id"),
                    "source_group": row.get("source_group"),
                    "intervention_type": row.get("intervention_type"),
                    "variant": row.get("variant"),
                    "component": row.get("component"),
                    "channel_scope": row.get("channel_scope"),
                    "alpha": row.get("alpha"),
                    "step": row.get("step"),
                    "winning_regime": row.get("winning_regime"),
                    "top_token": row.get("top_token"),
                    "target_logit_delta": finite_float(row.get("target_logit_delta")),
                    "rank_improve": finite_float(row.get("rank_improve")),
                    "target_margin_vs_winner": margin,
                    "remaining_margin_gap": gap,
                    "suppression_budget": budget,
                    "oracle_post_margin": post_margin,
                    "oracle_closes_threshold": post_margin >= 0.0,
                }
            )
    distribution_rows = summarize_distribution(rows)
    budget_rows = summarize_by_budget(suppression_rows)
    winner_budget_rows = summarize_by_winner_budget(suppression_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Competitor pressure oracle suppression",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "input_barrier_rows": len(rows),
        "suppression_rows": len(suppression_rows),
        "budgets": budgets,
        "gap_distribution": distribution_rows,
        "budget_summary": budget_rows,
        "winner_budget_summary": winner_budget_rows,
    }
    write_json(out_dir / f"phase231_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase231_{model}_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / f"phase231_{model}_gap_distribution_rows.jsonl", distribution_rows)
    write_jsonl(out_dir / f"phase231_{model}_budget_summary_rows.jsonl", budget_rows)
    write_jsonl(out_dir / f"phase231_{model}_winner_budget_summary_rows.jsonl", winner_budget_rows)
    return payload


def summarize_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("model"), row.get("spec_id"), row.get("source_group"), row.get("winning_regime"))].append(
            finite_float(row.get("remaining_margin_gap"))
        )
    out = []
    for key, gaps in buckets.items():
        model, spec_id, source_group, winner = key
        out.append(
            {
                "model": model,
                "spec_id": spec_id,
                "source_group": source_group,
                "winning_regime": winner,
                "rows": len(gaps),
                "mean_remaining_gap": mean(gaps),
                "median_remaining_gap": median(gaps),
                "p75_remaining_gap": quantile(gaps, 0.75),
                "p90_remaining_gap": quantile(gaps, 0.90),
                "max_remaining_gap": max(gaps),
            }
        )
    out.sort(key=lambda row: (int(row.get("rows") or 0), float(row.get("median_remaining_gap") or 0.0)), reverse=True)
    return out


def summarize_by_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("model"), row.get("suppression_budget"))].append(row)
    out = []
    for key, items in buckets.items():
        model, budget = key
        closed = [x for x in items if bool(x.get("oracle_closes_threshold"))]
        out.append(
            {
                "model": model,
                "suppression_budget": budget,
                "rows": len(items),
                "closed_rows": len(closed),
                "closure_rate": len(closed) / len(items) if items else 0.0,
                "mean_post_margin": mean(finite_float(x.get("oracle_post_margin")) for x in items),
            }
        )
    out.sort(key=lambda row: (str(row.get("model")), float(row.get("suppression_budget") or 0.0)))
    return out


def summarize_by_winner_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("model"),
                row.get("spec_id"),
                row.get("source_group"),
                row.get("winning_regime"),
                row.get("suppression_budget"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        model, spec_id, source_group, winner, budget = key
        closed = [x for x in items if bool(x.get("oracle_closes_threshold"))]
        out.append(
            {
                "model": model,
                "spec_id": spec_id,
                "source_group": source_group,
                "winning_regime": winner,
                "suppression_budget": budget,
                "rows": len(items),
                "closed_rows": len(closed),
                "closure_rate": len(closed) / len(items) if items else 0.0,
                "mean_remaining_gap": mean(finite_float(x.get("remaining_margin_gap")) for x in items),
            }
        )
    out.sort(
        key=lambda row: (
            str(row.get("model")),
            str(row.get("spec_id")),
            str(row.get("source_group")),
            str(row.get("winning_regime")),
            float(row.get("suppression_budget") or 0.0),
        )
    )
    return out


def summarize_round(input_round: str, round_name: str, budgets: list[float]) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [analyze_model(model, input_round, out_dir, budgets) for model in MODELS]
    all_distribution_rows: list[dict[str, Any]] = []
    all_budget_rows: list[dict[str, Any]] = []
    all_winner_budget_rows: list[dict[str, Any]] = []
    for model in MODELS:
        all_distribution_rows.extend(iter_jsonl(out_dir / f"phase231_{model}_gap_distribution_rows.jsonl"))
        all_budget_rows.extend(iter_jsonl(out_dir / f"phase231_{model}_budget_summary_rows.jsonl"))
        all_winner_budget_rows.extend(iter_jsonl(out_dir / f"phase231_{model}_winner_budget_summary_rows.jsonl"))

    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model competitor pressure oracle suppression",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "input_barrier_rows": sum(int(x.get("input_barrier_rows") or 0) for x in summaries),
        "suppression_rows": sum(int(x.get("suppression_rows") or 0) for x in summaries),
        "budgets": budgets,
        "gap_distribution": sorted(
            all_distribution_rows,
            key=lambda row: (int(row.get("rows") or 0), float(row.get("median_remaining_gap") or 0.0)),
            reverse=True,
        ),
        "budget_summary": all_budget_rows,
        "winner_budget_summary": all_winner_budget_rows,
    }
    write_json(out_dir / "phase231_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase231_cross_model_summary.md", payload)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "input_barrier_rows": payload["input_barrier_rows"],
                "suppression_rows": payload["suppression_rows"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 231 competitor pressure oracle suppression", ""]
    for key in ["input_barrier_rows", "suppression_rows"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "## Gap Distribution",
            "",
            "| model | spec | group | winner | rows | mean gap | median gap | p75 gap | p90 gap | max gap |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["gap_distribution"][:80]:
        lines.append(
            f"| {row.get('model')} | {row.get('spec_id')} | {row.get('source_group')} | {row.get('winning_regime')} | {row.get('rows')} | "
            f"{finite_float(row.get('mean_remaining_gap')):.4f} | {finite_float(row.get('median_remaining_gap')):.4f} | "
            f"{finite_float(row.get('p75_remaining_gap')):.4f} | {finite_float(row.get('p90_remaining_gap')):.4f} | {finite_float(row.get('max_remaining_gap')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Budget Summary",
            "",
            "| model | budget | rows | closed | closure rate | mean post margin |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["budget_summary"]:
        lines.append(
            f"| {row.get('model')} | {finite_float(row.get('suppression_budget')):.1f} | {row.get('rows')} | {row.get('closed_rows')} | "
            f"{finite_float(row.get('closure_rate')):.4f} | {finite_float(row.get('mean_post_margin')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Winner Budget Summary",
            "",
            "| model | spec | group | winner | budget | rows | closed | closure rate | mean gap |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    important = [r for r in payload["winner_budget_summary"] if finite_float(r.get("suppression_budget")) in {8.0, 16.0, 24.0, 32.0}]
    for row in important[:160]:
        lines.append(
            f"| {row.get('model')} | {row.get('spec_id')} | {row.get('source_group')} | {row.get('winning_regime')} | "
            f"{finite_float(row.get('suppression_budget')):.1f} | {row.get('rows')} | {row.get('closed_rows')} | "
            f"{finite_float(row.get('closure_rate')):.4f} | {finite_float(row.get('mean_remaining_gap')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase231 competitor pressure oracle suppression")
    parser.add_argument("--input-round", default="readout_threshold_barrier_analysis")
    parser.add_argument("--round-name", default="competitor_pressure_oracle_suppression")
    parser.add_argument("--budgets", default=",".join(str(x) for x in DEFAULT_BUDGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    summarize_round(args.input_round, args.round_name, budgets)


if __name__ == "__main__":
    main()
