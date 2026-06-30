#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE799_ROOT = Path("tests/result/phase799_blocker_field_causal_suppressor_localization")
OUT_ROOT = Path("tests/result/phase800_target_booster_suppressor_disentanglement")
MODELS = ("qwen3", "glm4", "deepseek7b")
ROUNDS = ("smoke", "main", "confirm")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def mean(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def stdev(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None


def corr(xs: list[Any], ys: list[Any]) -> float | None:
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def fmt(value: Any) -> str:
    val = safe_float(value)
    if val is None:
        return ""
    return f"{val:.3f}"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_gain = [r.get("target_logit_gain") for r in rows]
    blocker_supp = [r.get("baseline_blocker_mean_suppression") for r in rows]
    count_delta = [r.get("full_blocker_count_delta") for r in rows]
    baseline_count = [r.get("baseline_full_blocker_count") for r in rows]
    after_count = [r.get("after_full_blocker_count") for r in rows]
    resolved = [r.get("resolved_baseline_blocker_rate") for r in rows]
    new_rate = [r.get("new_blocker_rate") for r in rows]
    anchor = [r.get("identity_anchor_gap_improvement") for r in rows]
    pressure = [r.get("closure_fiber_pressure_score") for r in rows]
    token_closure = [bool(r.get("token_closure_gain")) for r in rows]
    tg = mean(target_gain) or 0.0
    bs = mean(blocker_supp) or 0.0
    rr = mean(resolved) or 0.0
    nr = mean(new_rate) or 0.0
    bc = mean(baseline_count) or 0.0
    cd = mean(count_delta) or 0.0
    blocker_count_reduction_rate = (-cd / bc) if bc > 0 else None
    true_suppressor_score = max(tg, 0.0) * max(bs, 0.0) * max(rr, 0.0) * max(1.0 - nr, 0.0)
    threshold_shift_score = max(tg, 0.0) * max(-bs, 0.0) * max(rr, 0.0) * (1.0 + max(nr, 0.0))
    target_neutral_suppressor_score = max(bs, 0.0) * max(rr, 0.0) * max(1.0 - abs(tg) / 2.0, 0.0) * max(1.0 - nr, 0.0)
    if tg > 1.0 and bs <= 0.0 and rr > 0.5:
        label = "target_booster_or_threshold_shift"
    elif tg > 0.5 and bs > 0.25 and rr > 0.5 and nr < 0.1:
        label = "true_suppressor_like"
    elif bs > 0.25 and rr > 0.4:
        label = "partial_suppressor_like"
    elif nr >= 0.1:
        label = "unstable_new_blocker"
    else:
        label = "weak_or_mixed"
    return {
        "n": len(rows),
        "case_n": len({r.get("case_id") for r in rows}),
        "mean_target_gain": mean(target_gain),
        "std_target_gain": stdev(target_gain),
        "mean_blocker_suppression": mean(blocker_supp),
        "std_blocker_suppression": stdev(blocker_supp),
        "mean_full_blocker_count_delta": mean(count_delta),
        "mean_baseline_full_blocker_count": mean(baseline_count),
        "mean_after_full_blocker_count": mean(after_count),
        "mean_blocker_count_reduction_rate": blocker_count_reduction_rate,
        "mean_resolved_rate": mean(resolved),
        "mean_new_blocker_rate": mean(new_rate),
        "mean_anchor_gap_improvement": mean(anchor),
        "mean_pressure_score": mean(pressure),
        "token_closure_gain_rate": sum(1 for v in token_closure if v) / len(token_closure) if token_closure else None,
        "corr_target_gain_vs_blocker_suppression": corr(target_gain, blocker_supp),
        "corr_target_gain_vs_count_delta": corr(target_gain, count_delta),
        "corr_blocker_suppression_vs_count_delta": corr(blocker_supp, count_delta),
        "true_suppressor_score": true_suppressor_score,
        "threshold_shift_score": threshold_shift_score,
        "target_neutral_suppressor_score": target_neutral_suppressor_score,
        "disentangled_label": label,
    }


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model"),
        row.get("source_component_label"),
        row.get("source_selection_kind"),
        row.get("subspace_mode"),
        row.get("budget_label"),
        row.get("source_set_size"),
        row.get("ladder_id"),
        row.get("source_group"),
    )


def bin_by_target_gain(rows: list[dict[str, Any]], bins: int = 4) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: safe_float(r.get("target_logit_gain")) or 0.0)
    if not sorted_rows:
        return []
    out = []
    for idx in range(bins):
        start = idx * len(sorted_rows) // bins
        end = (idx + 1) * len(sorted_rows) // bins
        chunk = sorted_rows[start:end]
        if chunk:
            summary = summarize_rows(chunk)
            summary["bin"] = idx + 1
            summary["target_gain_min"] = safe_float(chunk[0].get("target_logit_gain"))
            summary["target_gain_max"] = safe_float(chunk[-1].get("target_logit_gain"))
            out.append(summary)
    return out


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    rows_by_round_model: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    all_rows: list[dict[str, Any]] = []
    for round_name in ROUNDS:
        for model in MODELS:
            path = PHASE799_ROOT / round_name / f"phase799_{model}_rows.jsonl"
            rows = read_jsonl(path)
            rows_by_round_model[round_name][model] = rows
            all_rows.extend(rows)

    by_round_model: dict[str, dict[str, Any]] = {}
    for round_name, by_model in rows_by_round_model.items():
        by_round_model[round_name] = {}
        for model, rows in by_model.items():
            by_round_model[round_name][model] = summarize_rows(rows)

    confirm_by_model = rows_by_round_model["confirm"]
    target_bins = {model: bin_by_target_gain(rows, args.bins) for model, rows in confirm_by_model.items()}

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in confirm_by_model.get("qwen3", []) + confirm_by_model.get("glm4", []) + confirm_by_model.get("deepseek7b", []):
        grouped[candidate_key(row)].append(row)

    candidates = []
    for key, rows in grouped.items():
        if len(rows) < args.min_group_rows:
            continue
        summary = summarize_rows(rows)
        (
            model,
            component,
            selection,
            subspace_mode,
            budget_label,
            source_set_size,
            ladder_id,
            source_group,
        ) = key
        summary.update(
            {
                "model": model,
                "source_component_label": component,
                "source_selection_kind": selection,
                "subspace_mode": subspace_mode,
                "budget_label": budget_label,
                "source_set_size": source_set_size,
                "ladder_id": ladder_id,
                "source_group": source_group,
            }
        )
        candidates.append(summary)
    candidates.sort(
        key=lambda r: (
            r.get("disentangled_label") != "true_suppressor_like",
            -(r.get("true_suppressor_score") or 0.0),
            -(r.get("threshold_shift_score") or 0.0),
        )
    )

    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in candidates:
        label_counts[str(row.get("model"))][str(row.get("disentangled_label"))] += 1

    return {
        "phase": 800,
        "task": "target_booster_vs_true_suppressor_disentanglement",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase": 799,
        "source_root": str(PHASE799_ROOT),
        "method": (
            "Offline analysis over Phase 799 rows. It does not run models. "
            "It separates target-boost, true blocker suppression, and threshold-shift behavior."
        ),
        "by_round_model": by_round_model,
        "target_gain_bins_confirm": target_bins,
        "candidate_label_counts_confirm": {m: dict(v) for m, v in label_counts.items()},
        "top_true_suppressor_like_candidates_confirm": [
            r for r in candidates if r.get("disentangled_label") == "true_suppressor_like"
        ][:40],
        "top_threshold_shift_candidates_confirm": [
            r for r in sorted(candidates, key=lambda x: -(x.get("threshold_shift_score") or 0.0))
            if r.get("disentangled_label") == "target_booster_or_threshold_shift"
        ][:40],
        "top_target_neutral_suppressor_candidates_confirm": sorted(
            candidates, key=lambda x: -(x.get("target_neutral_suppressor_score") or 0.0)
        )[:40],
        "all_candidate_groups_confirm": candidates,
        "boundary": (
            "Phase 800 is a data-level disentanglement. It improves interpretation of Phase 799, "
            "but does not replace direct target-neutral intervention tests."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 800 Target Booster vs True Suppressor Disentanglement",
        "",
        f"- Source phase: `{payload['source_phase']}`",
        f"- Source root: `{payload['source_root']}`",
        "- Model runs: none; this is offline analysis over Phase 799 rows.",
        "- Boundary: separates target boost, true blocker suppression, and threshold shift behavior.",
        "",
        "## Cross-Round Model Summary",
        "",
        "| round | model | rows | target gain | blocker suppression | count reduction | resolved | new rate | anchor gap | label hint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for round_name in ROUNDS:
        for model in MODELS:
            row = payload["by_round_model"].get(round_name, {}).get(model, {})
            lines.append(
                f"| {round_name} | {model} | {row.get('n')} | {fmt(row.get('mean_target_gain'))} | "
                f"{fmt(row.get('mean_blocker_suppression'))} | {fmt(row.get('mean_blocker_count_reduction_rate'))} | "
                f"{fmt(row.get('mean_resolved_rate'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
                f"{fmt(row.get('mean_anchor_gap_improvement'))} | {row.get('disentangled_label')} |"
            )
    lines += [
        "",
        "## Confirm Label Counts",
        "",
        "```json",
        json.dumps(payload["candidate_label_counts_confirm"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Top True Suppressor-Like Candidates",
        "",
        "| model | component | source group | ladder | target gain | blocker suppression | resolved | new rate | true score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["top_true_suppressor_like_candidates_confirm"][:24]:
        lines.append(
            f"| {row.get('model')} | `{row.get('source_component_label')}` | `{row.get('source_group')}` | "
            f"`{row.get('ladder_id')}` | {fmt(row.get('mean_target_gain'))} | {fmt(row.get('mean_blocker_suppression'))} | "
            f"{fmt(row.get('mean_resolved_rate'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
            f"{fmt(row.get('true_suppressor_score'))} |"
        )
    lines += [
        "",
        "## Top Threshold-Shift Candidates",
        "",
        "| model | component | source group | ladder | target gain | blocker suppression | resolved | new rate | threshold score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["top_threshold_shift_candidates_confirm"][:24]:
        lines.append(
            f"| {row.get('model')} | `{row.get('source_component_label')}` | `{row.get('source_group')}` | "
            f"`{row.get('ladder_id')}` | {fmt(row.get('mean_target_gain'))} | {fmt(row.get('mean_blocker_suppression'))} | "
            f"{fmt(row.get('mean_resolved_rate'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
            f"{fmt(row.get('threshold_shift_score'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-group-rows", type=int, default=2)
    parser.add_argument("--bins", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_payload(args)
    write_json(OUT_ROOT / "phase800_disentanglement_summary.json", payload)
    write_markdown(OUT_ROOT / "phase800_disentanglement_summary.md", payload)
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "status": "complete",
                "out_root": str(OUT_ROOT),
                "candidate_label_counts_confirm": payload["candidate_label_counts_confirm"],
                "top_true_suppressor_like_candidates_confirm": payload["top_true_suppressor_like_candidates_confirm"][:5],
                "top_threshold_shift_candidates_confirm": payload["top_threshold_shift_candidates_confirm"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
