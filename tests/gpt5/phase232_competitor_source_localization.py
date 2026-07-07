#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 232
SOURCE_PHASE = 229
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase229_readout_regime_selection_atlas")
RESULT_ROOT = Path("tests/result/phase232_competitor_source_localization")


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


def regimes_in_rows(rows: list[dict[str, Any]]) -> list[str]:
    regimes = set()
    for row in rows:
        regimes.update((row.get("regime_delta") or {}).keys())
        if row.get("winning_regime"):
            regimes.add(str(row["winning_regime"]))
        if row.get("base_winning_regime"):
            regimes.add(str(row["base_winning_regime"]))
    regimes.discard("target")
    regimes.discard("none")
    return sorted(regimes)


def analyze_model(model: str, input_round: str, out_dir: Path) -> dict[str, Any]:
    rows = iter_jsonl(INPUT_ROOT / input_round / f"phase229_{model}_regime_rows.jsonl")
    regimes = regimes_in_rows(rows)
    pressure_rows = summarize_pressure_sources(model, rows, regimes)
    switch_rows = summarize_switch_sources(model, rows)
    coupling_rows = summarize_patch_coupling(model, rows, regimes)
    priority_rows = select_priority_rows(pressure_rows, switch_rows, coupling_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Competitor source localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "input_rows": len(rows),
        "pressure_source_rows": len(pressure_rows),
        "switch_source_rows": len(switch_rows),
        "coupling_rows": len(coupling_rows),
        "priority_rows": priority_rows[:80],
    }
    write_json(out_dir / f"phase232_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase232_{model}_pressure_source_rows.jsonl", pressure_rows)
    write_jsonl(out_dir / f"phase232_{model}_switch_source_rows.jsonl", switch_rows)
    write_jsonl(out_dir / f"phase232_{model}_coupling_rows.jsonl", coupling_rows)
    write_jsonl(out_dir / f"phase232_{model}_priority_rows.jsonl", priority_rows)
    return payload


def summarize_pressure_sources(model: str, rows: list[dict[str, Any]], regimes: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for regime in regimes:
            buckets[
                (
                    model,
                    row.get("spec_id"),
                    row.get("source_group"),
                    row.get("intervention_type"),
                    row.get("variant"),
                    row.get("component"),
                    row.get("channel_scope"),
                    row.get("alpha"),
                    row.get("step"),
                    regime,
                )
            ].append(row)

    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        model_name, spec_id, source_group, intervention_type, variant, component, scope, alpha, step, regime = key
        deltas = [finite_float((x.get("regime_delta") or {}).get(regime)) for x in items]
        target_deltas = [finite_float(x.get("target_logit_delta")) for x in items]
        pressure_advantage = [d - t for d, t in zip(deltas, target_deltas)]
        winner_count = sum(1 for x in items if x.get("winning_regime") == regime)
        base_winner_count = sum(1 for x in items if x.get("base_winning_regime") == regime)
        changed_to_count = sum(
            1
            for x in items
            if bool(x.get("winning_regime_changed"))
            and x.get("winning_regime") == regime
            and x.get("base_winning_regime") != regime
        )
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase232_pressure_source_row",
                "model": model_name,
                "spec_id": spec_id,
                "source_group": source_group,
                "intervention_type": intervention_type,
                "variant": variant,
                "component": component,
                "channel_scope": scope,
                "alpha": alpha,
                "step": int(step),
                "regime": regime,
                "rows": len(items),
                "winner_count": winner_count,
                "winner_rate": winner_count / len(items) if items else 0.0,
                "base_winner_count": base_winner_count,
                "changed_to_count": changed_to_count,
                "changed_to_rate": changed_to_count / len(items) if items else 0.0,
                "mean_regime_delta": mean(deltas),
                "mean_target_delta": mean(target_deltas),
                "mean_competitor_minus_target_delta": mean(pressure_advantage),
                "positive_regime_delta_count": sum(1 for x in deltas if x > 0.0),
                "positive_regime_delta_rate": sum(1 for x in deltas if x > 0.0) / len(deltas) if deltas else 0.0,
                "top_tokens": dict(Counter(str(x.get("top_token")) for x in items).most_common(6)),
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("changed_to_rate") or 0.0),
            float(row.get("winner_rate") or 0.0),
            float(row.get("mean_competitor_minus_target_delta") or 0.0),
            float(row.get("mean_regime_delta") or 0.0),
        ),
        reverse=True,
    )
    return out


def summarize_switch_sources(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("winning_regime_changed")):
            continue
        buckets[
            (
                model,
                row.get("spec_id"),
                row.get("source_group"),
                row.get("intervention_type"),
                row.get("variant"),
                row.get("component"),
                row.get("channel_scope"),
                row.get("alpha"),
                row.get("step"),
                row.get("base_winning_regime"),
                row.get("winning_regime"),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        model_name, spec_id, source_group, intervention_type, variant, component, scope, alpha, step, source, target = key
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase232_switch_source_row",
                "model": model_name,
                "spec_id": spec_id,
                "source_group": source_group,
                "intervention_type": intervention_type,
                "variant": variant,
                "component": component,
                "channel_scope": scope,
                "alpha": alpha,
                "step": int(step),
                "base_winning_regime": source,
                "winning_regime": target,
                "rows": len(items),
                "mean_target_logit_delta": mean(finite_float(x.get("target_logit_delta")) for x in items),
                "mean_margin_delta_vs_winner": mean(finite_float(x.get("margin_delta_vs_winner")) for x in items),
                "mean_target_margin_vs_winner": mean(finite_float(x.get("target_margin_vs_winner")) for x in items),
                "top_tokens": dict(Counter(str(x.get("top_token")) for x in items).most_common(6)),
            }
        )
    out.sort(key=lambda row: (int(row.get("rows") or 0), abs(float(row.get("mean_margin_delta_vs_winner") or 0.0))), reverse=True)
    return out


def summarize_patch_coupling(model: str, rows: list[dict[str, Any]], regimes: list[str]) -> list[dict[str, Any]]:
    patch_rows = [x for x in rows if x.get("intervention_type") == "patch"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        for regime in regimes:
            buckets[
                (
                    model,
                    row.get("spec_id"),
                    row.get("source_group"),
                    row.get("variant"),
                    row.get("component"),
                    row.get("channel_scope"),
                    row.get("alpha"),
                    row.get("step"),
                    regime,
                )
            ].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        model_name, spec_id, source_group, variant, component, scope, alpha, step, regime = key
        target_deltas = [finite_float(x.get("target_logit_delta")) for x in items]
        regime_deltas = [finite_float((x.get("regime_delta") or {}).get(regime)) for x in items]
        same_rise = [1 for t, c in zip(target_deltas, regime_deltas) if t > 0.0 and c > 0.0]
        competitor_dominates = [1 for t, c in zip(target_deltas, regime_deltas) if c > t]
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase232_patch_coupling_row",
                "model": model_name,
                "spec_id": spec_id,
                "source_group": source_group,
                "variant": variant,
                "component": component,
                "channel_scope": scope,
                "alpha": alpha,
                "step": int(step),
                "regime": regime,
                "rows": len(items),
                "mean_target_delta": mean(target_deltas),
                "mean_regime_delta": mean(regime_deltas),
                "mean_competitor_minus_target_delta": mean(c - t for t, c in zip(target_deltas, regime_deltas)),
                "same_rise_rate": len(same_rise) / len(items) if items else 0.0,
                "competitor_dominates_rate": len(competitor_dominates) / len(items) if items else 0.0,
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("same_rise_rate") or 0.0),
            float(row.get("competitor_dominates_rate") or 0.0),
            float(row.get("mean_regime_delta") or 0.0),
        ),
        reverse=True,
    )
    return out


def select_priority_rows(
    pressure_rows: list[dict[str, Any]],
    switch_rows: list[dict[str, Any]],
    coupling_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    priority_regimes = {
        "qwen3": {"because_reason", "period_stop", "echo"},
        "glm4": {"for_continuation", "newline_boundary", "because_reason", "the_continuation", "comma_repeat"},
        "deepseek7b": {"be_continuation", "the_continuation", "prose", "newline_boundary", "echo"},
    }
    out: list[dict[str, Any]] = []
    for row in pressure_rows:
        if row.get("regime") in priority_regimes.get(str(row.get("model")), set()):
            score = (
                5.0 * finite_float(row.get("changed_to_rate"))
                + 3.0 * finite_float(row.get("winner_rate"))
                + finite_float(row.get("mean_competitor_minus_target_delta"))
                + 0.5 * finite_float(row.get("mean_regime_delta"))
            )
            item = dict(row)
            item["priority_kind"] = "pressure_source"
            item["priority_score"] = score
            out.append(item)
    for row in switch_rows:
        if row.get("winning_regime") in priority_regimes.get(str(row.get("model")), set()):
            item = dict(row)
            item["priority_kind"] = "switch_source"
            item["priority_score"] = int(row.get("rows") or 0) + abs(finite_float(row.get("mean_margin_delta_vs_winner")))
            out.append(item)
    for row in coupling_rows:
        if row.get("regime") in priority_regimes.get(str(row.get("model")), set()):
            score = (
                3.0 * finite_float(row.get("same_rise_rate"))
                + 2.0 * finite_float(row.get("competitor_dominates_rate"))
                + finite_float(row.get("mean_regime_delta"))
            )
            item = dict(row)
            item["priority_kind"] = "patch_coupling"
            item["priority_score"] = score
            out.append(item)
    out.sort(key=lambda row: float(row.get("priority_score") or 0.0), reverse=True)
    return out


def summarize_round(input_round: str, round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [analyze_model(model, input_round, out_dir) for model in MODELS]
    all_pressure: list[dict[str, Any]] = []
    all_switch: list[dict[str, Any]] = []
    all_coupling: list[dict[str, Any]] = []
    all_priority: list[dict[str, Any]] = []
    for model in MODELS:
        all_pressure.extend(iter_jsonl(out_dir / f"phase232_{model}_pressure_source_rows.jsonl"))
        all_switch.extend(iter_jsonl(out_dir / f"phase232_{model}_switch_source_rows.jsonl"))
        all_coupling.extend(iter_jsonl(out_dir / f"phase232_{model}_coupling_rows.jsonl"))
        all_priority.extend(iter_jsonl(out_dir / f"phase232_{model}_priority_rows.jsonl"))
    all_priority.sort(key=lambda row: float(row.get("priority_score") or 0.0), reverse=True)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model competitor source localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "input_rows": sum(int(x.get("input_rows") or 0) for x in summaries),
        "pressure_source_rows": len(all_pressure),
        "switch_source_rows": len(all_switch),
        "coupling_rows": len(all_coupling),
        "priority_rows": all_priority[:160],
    }
    write_json(out_dir / "phase232_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase232_cross_model_summary.md", payload)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "input_rows": payload["input_rows"],
                "priority_rows": len(payload["priority_rows"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 232 competitor source localization", ""]
    for key in ["input_rows", "pressure_source_rows", "switch_source_rows", "coupling_rows"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "## Priority Rows",
            "",
            "| kind | score | model | spec | group | type | variant | step | regime | rows | winner/switch rate | target delta | regime delta | comp-target | top tokens |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["priority_rows"][:120]:
        regime = row.get("regime") or row.get("winning_regime")
        rate = row.get("changed_to_rate", row.get("same_rise_rate", row.get("rows", 0)))
        lines.append(
            f"| {row.get('priority_kind')} | {finite_float(row.get('priority_score')):.4f} | {row.get('model')} | {row.get('spec_id')} | "
            f"{row.get('source_group')} | {row.get('intervention_type', 'patch')} | {row.get('variant')} | {row.get('step')} | {regime} | "
            f"{row.get('rows')} | {finite_float(rate):.4f} | {finite_float(row.get('mean_target_delta', row.get('mean_target_logit_delta'))):.4f} | "
            f"{finite_float(row.get('mean_regime_delta')):.4f} | {finite_float(row.get('mean_competitor_minus_target_delta')):.4f} | {row.get('top_tokens', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase232 competitor source localization")
    parser.add_argument("--input-round", default="readout_regime_selection_atlas")
    parser.add_argument("--round-name", default="competitor_source_localization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_round(args.input_round, args.round_name)


if __name__ == "__main__":
    main()
