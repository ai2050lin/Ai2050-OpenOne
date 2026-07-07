#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 230
SOURCE_PHASE = 229
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase229_readout_regime_selection_atlas")
RESULT_ROOT = Path("tests/result/phase230_readout_threshold_barrier_analysis")


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


def analyze_model(model: str, input_round: str, out_dir: Path) -> dict[str, Any]:
    rows = iter_jsonl(INPUT_ROOT / input_round / f"phase229_{model}_regime_rows.jsonl")
    barrier_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for row in rows:
        margin = finite_float(row.get("target_margin_vs_winner"))
        target_delta = finite_float(row.get("target_logit_delta"))
        rank_improve = finite_float(row.get("rank_improve"))
        margin_delta = finite_float(row.get("margin_delta_vs_winner"))
        base_margin = finite_float(row.get("base_target_margin_vs_winner"))
        if target_delta > 0 and margin < 0:
            barrier_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase230_threshold_barrier_row",
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
                    "target_logit_delta": target_delta,
                    "rank_improve": rank_improve,
                    "base_target_margin_vs_winner": base_margin,
                    "target_margin_vs_winner": margin,
                    "margin_delta_vs_winner": margin_delta,
                    "remaining_margin_gap": -margin,
                    "pressure_efficiency": margin_delta / target_delta if abs(target_delta) > 1e-9 else 0.0,
                }
            )
        if target_delta > 0 and margin > 0:
            closure_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase230_threshold_closure_candidate_row",
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
                    "target_logit_delta": target_delta,
                    "rank_improve": rank_improve,
                    "target_margin_vs_winner": margin,
                }
            )
    summary_rows = summarize_barriers(barrier_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Readout threshold barrier analysis",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "input_rows": len(rows),
        "barrier_rows": len(barrier_rows),
        "closure_candidate_rows": len(closure_rows),
        "top_barrier_summary": summary_rows[:80],
        "top_closure_candidates": sorted(closure_rows, key=lambda r: finite_float(r.get("target_margin_vs_winner")), reverse=True)[:40],
    }
    write_json(out_dir / f"phase230_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase230_{model}_barrier_rows.jsonl", barrier_rows)
    write_jsonl(out_dir / f"phase230_{model}_closure_candidate_rows.jsonl", closure_rows)
    write_jsonl(out_dir / f"phase230_{model}_barrier_summary_rows.jsonl", summary_rows)
    return payload


def summarize_barriers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("spec_id"),
                row.get("source_group"),
                row.get("intervention_type"),
                row.get("variant"),
                row.get("component"),
                row.get("channel_scope"),
                row.get("step"),
                row.get("winning_regime"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, group, intervention_type, variant, component, scope, step, winner = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": group,
                "intervention_type": intervention_type,
                "variant": variant,
                "component": component,
                "channel_scope": scope,
                "step": int(step),
                "winning_regime": winner,
                "rows": len(items),
                "mean_target_logit_delta": sum(finite_float(x.get("target_logit_delta")) for x in items) / len(items),
                "mean_rank_improve": sum(finite_float(x.get("rank_improve")) for x in items) / len(items),
                "mean_remaining_margin_gap": sum(finite_float(x.get("remaining_margin_gap")) for x in items) / len(items),
                "mean_margin_delta_vs_winner": sum(finite_float(x.get("margin_delta_vs_winner")) for x in items) / len(items),
                "mean_pressure_efficiency": sum(finite_float(x.get("pressure_efficiency")) for x in items) / len(items),
                "top_tokens": dict(Counter(str(x.get("top_token")) for x in items).most_common(8)),
            }
        )
    out.sort(key=lambda row: (float(row.get("mean_remaining_margin_gap") or 0.0), float(row.get("mean_target_logit_delta") or 0.0)), reverse=True)
    return out


def summarize_round(input_round: str, round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [analyze_model(model, input_round, out_dir) for model in MODELS]
    all_summary_rows: list[dict[str, Any]] = []
    all_closure_rows: list[dict[str, Any]] = []
    for model in MODELS:
        all_summary_rows.extend(iter_jsonl(out_dir / f"phase230_{model}_barrier_summary_rows.jsonl"))
        all_closure_rows.extend(iter_jsonl(out_dir / f"phase230_{model}_closure_candidate_rows.jsonl"))
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model readout threshold barrier analysis",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "input_rows": sum(int(x.get("input_rows") or 0) for x in summaries),
        "barrier_rows": sum(int(x.get("barrier_rows") or 0) for x in summaries),
        "closure_candidate_rows": sum(int(x.get("closure_candidate_rows") or 0) for x in summaries),
        "top_barrier_summary": sorted(
            all_summary_rows,
            key=lambda row: (float(row.get("mean_remaining_margin_gap") or 0.0), float(row.get("mean_target_logit_delta") or 0.0)),
            reverse=True,
        )[:120],
        "top_closure_candidates": sorted(all_closure_rows, key=lambda row: finite_float(row.get("target_margin_vs_winner")), reverse=True)[:60],
    }
    write_json(out_dir / "phase230_cross_model_summary.json", payload)
    lines = ["# Phase 230 readout threshold barrier analysis", ""]
    for key in ["input_rows", "barrier_rows", "closure_candidate_rows"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "## Barrier Summary",
            "",
            "| spec | group | type | variant | step | winner | rows | target delta | rank improve | remaining gap | margin delta | efficiency | top tokens |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["top_barrier_summary"][:80]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('intervention_type')} | {row.get('variant')} | {row.get('step')} | {row.get('winning_regime')} | {row.get('rows')} | "
            f"{finite_float(row.get('mean_target_logit_delta')):.4f} | {finite_float(row.get('mean_rank_improve')):.4f} | "
            f"{finite_float(row.get('mean_remaining_margin_gap')):.4f} | {finite_float(row.get('mean_margin_delta_vs_winner')):.4f} | "
            f"{finite_float(row.get('mean_pressure_efficiency')):.4f} | {row.get('top_tokens')} |"
        )
    lines.extend(["", "## Closure Candidates", "", "| model | spec | group | type | variant | step | winner | margin | target delta | rank improve | top token |", "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |"])
    for row in payload["top_closure_candidates"][:40]:
        lines.append(
            f"| {row.get('model')} | {row.get('spec_id')} | {row.get('source_group')} | {row.get('intervention_type')} | {row.get('variant')} | {row.get('step')} | {row.get('winning_regime')} | "
            f"{finite_float(row.get('target_margin_vs_winner')):.4f} | {finite_float(row.get('target_logit_delta')):.4f} | {finite_float(row.get('rank_improve')):.4f} | {row.get('top_token')} |"
        )
    (out_dir / "phase230_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "barrier_rows": payload["barrier_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase230 readout threshold barrier analysis")
    parser.add_argument("--input-round", default="readout_regime_selection_atlas")
    parser.add_argument("--round-name", default="readout_threshold_barrier_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_round(args.input_round, args.round_name)


if __name__ == "__main__":
    main()
