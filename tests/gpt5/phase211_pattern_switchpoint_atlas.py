#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 211
SOURCE_PHASE = 210
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase211_pattern_switchpoint_atlas")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return None if not vals else float(sum(vals) / len(vals))


def add_stat(bucket: dict[str, list[Any]], row: dict[str, Any]) -> None:
    for key in [
        "residual_norm",
        "residual_mean",
        "residual_std",
        "target_rank",
        "stop_margin",
        "prose_margin",
        "echo_margin",
        "eos_rank",
        "period_rank",
    ]:
        bucket[key].append(row.get(key))


def finalize_stat(bucket: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "rows": len(bucket.get("residual_norm") or []),
        "residual_norm_mean": mean(bucket.get("residual_norm") or []),
        "residual_mean_mean": mean(bucket.get("residual_mean") or []),
        "residual_std_mean": mean(bucket.get("residual_std") or []),
        "target_rank_mean": mean(bucket.get("target_rank") or []),
        "stop_margin_mean": mean(bucket.get("stop_margin") or []),
        "prose_margin_mean": mean(bucket.get("prose_margin") or []),
        "echo_margin_mean": mean(bucket.get("echo_margin") or []),
        "eos_rank_mean": mean(bucket.get("eos_rank") or []),
        "period_rank_mean": mean(bucket.get("period_rank") or []),
    }


def trajectory_outcomes(input_dir: Path, model: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(input_dir / f"phase210_{model}_trajectory_rows.jsonl") or []:
        trajectory_id = str(row.get("trajectory_id"))
        if not trajectory_id:
            continue
        success = bool(row.get("pattern_match"))
        failure_mode = str(row.get("failure_mode") or "unknown")
        output_pattern = str(row.get("output_pattern") or "unknown")
        out[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "model": row.get("model"),
            "pattern_id": row.get("pattern_id"),
            "success": success,
            "outcome_group": "success" if success else f"drift:{failure_mode}",
            "failure_mode": failure_mode,
            "output_pattern": output_pattern,
            "answer_present": row.get("answer_present"),
            "ended_with_eos": row.get("ended_with_eos"),
        }
    return out


def outcome_distribution(outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in outcomes.values():
        buckets[(item.get("model"), item.get("pattern_id"), item.get("outcome_group"))].append(item)
    rows = []
    for key, items in buckets.items():
        model, pattern, outcome_group = key
        rows.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase211_outcome_distribution_row",
                "model": model,
                "pattern_id": pattern,
                "outcome_group": outcome_group,
                "rows": len(items),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "ended_with_eos": sum(1 for item in items if item.get("ended_with_eos")),
            }
        )
    rows.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), str(row.get("outcome_group"))))
    return rows


def build_state_outcome_summary(input_dir: Path, model: str, outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for row in iter_jsonl(input_dir / f"phase210_{model}_state_rows.jsonl") or []:
        outcome = outcomes.get(str(row.get("trajectory_id")))
        if not outcome:
            continue
        key = (
            row.get("model"),
            row.get("pattern_id"),
            outcome.get("outcome_group"),
            row.get("step"),
            row.get("layer_idx"),
        )
        add_stat(buckets[key], row)
    rows = []
    for key, bucket in buckets.items():
        model_name, pattern_id, outcome_group, step, layer_idx = key
        rows.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase211_state_outcome_summary_row",
                "model": model_name,
                "pattern_id": pattern_id,
                "outcome_group": outcome_group,
                "step": step,
                "layer_idx": layer_idx,
                **finalize_stat(bucket),
            }
        )
    rows.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), str(row.get("outcome_group")), int(row.get("step")), int(row.get("layer_idx"))))
    return rows


def contrast_score(success: dict[str, Any], drift: dict[str, Any]) -> dict[str, Any]:
    residual_norm_delta = finite(drift.get("residual_norm_mean")) - finite(success.get("residual_norm_mean"))
    target_rank_delta = finite(drift.get("target_rank_mean")) - finite(success.get("target_rank_mean"))
    stop_margin_delta = finite(drift.get("stop_margin_mean")) - finite(success.get("stop_margin_mean"))
    prose_margin_delta = finite(drift.get("prose_margin_mean")) - finite(success.get("prose_margin_mean"))
    echo_margin_delta = finite(drift.get("echo_margin_mean")) - finite(success.get("echo_margin_mean"))
    eos_rank_delta = finite(drift.get("eos_rank_mean")) - finite(success.get("eos_rank_mean"))
    norm_scale = max(1.0, abs(finite(success.get("residual_norm_mean"))))
    rank_scale = 1000.0
    switchpoint_score = (
        abs(residual_norm_delta) / norm_scale
        + abs(prose_margin_delta)
        + abs(echo_margin_delta)
        + abs(stop_margin_delta)
        + abs(target_rank_delta) / rank_scale
        + abs(eos_rank_delta) / rank_scale
    )
    return {
        "residual_norm_delta_drift_minus_success": residual_norm_delta,
        "target_rank_delta_drift_minus_success": target_rank_delta,
        "stop_margin_delta_drift_minus_success": stop_margin_delta,
        "prose_margin_delta_drift_minus_success": prose_margin_delta,
        "echo_margin_delta_drift_minus_success": echo_margin_delta,
        "eos_rank_delta_drift_minus_success": eos_rank_delta,
        "switchpoint_score": float(switchpoint_score),
    }


def build_switchpoint_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        key = (row.get("model"), row.get("pattern_id"), row.get("step"), row.get("layer_idx"))
        by_key[key][str(row.get("outcome_group"))] = row
    rows = []
    for key, groups in by_key.items():
        success = groups.get("success")
        if not success:
            continue
        model, pattern_id, step, layer_idx = key
        for outcome_group, drift in groups.items():
            if outcome_group == "success":
                continue
            rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase211_switchpoint_candidate_row",
                    "model": model,
                    "pattern_id": pattern_id,
                    "drift_group": outcome_group,
                    "step": step,
                    "layer_idx": layer_idx,
                    "success_rows": success.get("rows"),
                    "drift_rows": drift.get("rows"),
                    **contrast_score(success, drift),
                }
            )
    rows.sort(key=lambda row: finite(row.get("switchpoint_score")), reverse=True)
    return rows


def summarize_switchpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("model"), row.get("pattern_id"), row.get("drift_group"))].append(row)
    out = []
    for key, items in buckets.items():
        model, pattern_id, drift_group = key
        best = max(items, key=lambda row: finite(row.get("switchpoint_score")))
        layer_counts = Counter(str(item.get("layer_idx")) for item in items[:20])
        step_counts = Counter(str(item.get("step")) for item in items[:20])
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase211_switchpoint_summary_row",
                "model": model,
                "pattern_id": pattern_id,
                "drift_group": drift_group,
                "candidate_rows": len(items),
                "best_step": best.get("step"),
                "best_layer_idx": best.get("layer_idx"),
                "best_switchpoint_score": best.get("switchpoint_score"),
                "best_residual_norm_delta": best.get("residual_norm_delta_drift_minus_success"),
                "best_prose_margin_delta": best.get("prose_margin_delta_drift_minus_success"),
                "best_echo_margin_delta": best.get("echo_margin_delta_drift_minus_success"),
                "top20_layer_counts": dict(layer_counts.most_common()),
                "top20_step_counts": dict(step_counts.most_common()),
            }
        )
    out.sort(key=lambda row: finite(row.get("best_switchpoint_score")), reverse=True)
    return out


def eval_round(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = INPUT_ROOT / args.phase210_round
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_outcome_rows: list[dict[str, Any]] = []
    all_state_summary_rows: list[dict[str, Any]] = []
    all_switchpoint_rows: list[dict[str, Any]] = []
    model_payloads = []
    for model in MODELS:
        outcomes = trajectory_outcomes(input_dir, model)
        outcome_rows = outcome_distribution(outcomes)
        state_summary_rows = build_state_outcome_summary(input_dir, model, outcomes)
        switchpoint_rows = build_switchpoint_rows(state_summary_rows)
        switchpoint_summary_rows = summarize_switchpoints(switchpoint_rows)
        model_payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "model": model,
            "trajectory_count": len(outcomes),
            "outcome_rows": len(outcome_rows),
            "state_summary_rows": len(state_summary_rows),
            "switchpoint_rows": len(switchpoint_rows),
            "switchpoint_summary_rows": switchpoint_summary_rows,
            "top_switchpoint_rows": switchpoint_rows[: int(args.top_rows_per_model)],
        }
        write_json(out_dir / f"phase211_{model}_summary.json", model_payload)
        write_jsonl(out_dir / f"phase211_{model}_outcome_rows.jsonl", outcome_rows)
        write_jsonl(out_dir / f"phase211_{model}_state_outcome_summary_rows.jsonl", state_summary_rows)
        write_jsonl(out_dir / f"phase211_{model}_switchpoint_rows.jsonl", switchpoint_rows)
        all_outcome_rows.extend(outcome_rows)
        all_state_summary_rows.extend(state_summary_rows)
        all_switchpoint_rows.extend(switchpoint_rows)
        model_payloads.append(model_payload)
    cross_summary_rows = summarize_switchpoints(all_switchpoint_rows)
    payload = {
        "schema_version": "phase211_pattern_switchpoint_atlas_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_phase210_round": args.phase210_round,
        "round": args.round_name,
        "models": MODELS,
        "outcome_rows": len(all_outcome_rows),
        "state_summary_rows": len(all_state_summary_rows),
        "switchpoint_rows": len(all_switchpoint_rows),
        "model_summaries": model_payloads,
        "switchpoint_summary_rows": cross_summary_rows,
        "top_switchpoint_rows": all_switchpoint_rows[: int(args.top_rows)],
        "boundary": "Offline analysis of Phase210 scalar state metrics. Switchpoint candidates are not causal proof and do not use hidden vectors.",
    }
    write_json(out_dir / "phase211_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase211_cross_model_outcome_rows.jsonl", all_outcome_rows)
    write_jsonl(out_dir / "phase211_cross_model_state_outcome_summary_rows.jsonl", all_state_summary_rows)
    write_jsonl(out_dir / "phase211_cross_model_switchpoint_rows.jsonl", all_switchpoint_rows)
    write_summary_md(out_dir / "phase211_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 211 pattern switchpoint atlas", ""]
    lines.append(f"Outcome rows: {payload.get('outcome_rows')}")
    lines.append(f"State summary rows: {payload.get('state_summary_rows')}")
    lines.append(f"Switchpoint rows: {payload.get('switchpoint_rows')}")
    lines.append("")
    lines.append("| model | pattern | drift group | best step | best layer | score | norm delta | prose delta | echo delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("switchpoint_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('drift_group')} | {row.get('best_step')} | "
            f"{row.get('best_layer_idx')} | {finite(row.get('best_switchpoint_score')):.4f} | "
            f"{finite(row.get('best_residual_norm_delta')):.4f} | {finite(row.get('best_prose_margin_delta')):.4f} | "
            f"{finite(row.get('best_echo_margin_delta')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="pattern_switchpoint_atlas")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--top-rows", type=int, default=80)
    parser.add_argument("--top-rows-per-model", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = eval_round(args)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": payload.get("status"),
                "models": payload.get("models"),
                "state_summary_rows": payload.get("state_summary_rows"),
                "switchpoint_rows": payload.get("switchpoint_rows"),
                "top_summary": (payload.get("switchpoint_summary_rows") or [])[:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
