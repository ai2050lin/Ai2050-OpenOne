#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402


PHASE = 225
SOURCE_PHASE = 224
MODELS = ["qwen3", "glm4", "deepseek7b"]
PHASE224_ROOT = Path("tests/result/phase224_multilayer_activation_propagation")
PHASE223_ROOT = Path("tests/result/phase223_channel_activation_gate_validation")
RESULT_ROOT = Path("tests/result/phase225_readout_competition_threshold")

SPEC_MAP_224_TO_223 = {
    "qwen3_explain_l29_to_l31_l33_propagation": "qwen3_explain_l29_l31_activation_gate",
    "glm4_repeat_l30_to_l31_l32_propagation": "glm4_repeat_l30_activation_gate",
    "deepseek7b_explain_l24_to_l25_l26_propagation": "deepseek7b_explain_l24_activation_gate",
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def rank_improved(row: dict[str, Any]) -> bool:
    if row.get("base_target_rank") is None or row.get("patch_target_rank") is None:
        return False
    return finite_float(row.get("patch_target_rank")) < finite_float(row.get("base_target_rank"))


def margin_delta(row: dict[str, Any], name: str) -> float:
    return finite_float(row.get(f"patch_{name}")) - finite_float(row.get(f"base_{name}"))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def summarize_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("spec_id"), row.get("source_group"), row.get("condition"), row.get("observe_layer"))].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, condition, observe_layer = key
        shifts = [finite_float(x.get("projection_shift")) for x in items]
        abs_shifts = [abs(x) for x in shifts]
        top_changed_rows = [x for x in items if x.get("base_top_token_id") != x.get("patch_top_token_id")]
        rank_improved_rows = [x for x in items if rank_improved(x)]
        prose_deltas = [margin_delta(x, "prose_margin") for x in items]
        echo_deltas = [margin_delta(x, "echo_margin") for x in items]
        stop_deltas = [margin_delta(x, "stop_margin") for x in items]
        rank_deltas = [
            finite_float(x.get("base_target_rank")) - finite_float(x.get("patch_target_rank"))
            for x in items
            if x.get("base_target_rank") is not None and x.get("patch_target_rank") is not None
        ]
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase225_readout_threshold_summary_row",
                "spec_id": spec_id,
                "source_group": source_group,
                "condition": condition,
                "observe_layer": int(observe_layer),
                "rows": len(items),
                "mean_projection_shift": sum(shifts) / len(shifts) if shifts else 0.0,
                "mean_abs_projection_shift": sum(abs_shifts) / len(abs_shifts) if abs_shifts else 0.0,
                "projection_shift_p50": quantile(shifts, 0.5),
                "projection_shift_p90_abs": quantile(abs_shifts, 0.9),
                "top_token_changed": len(top_changed_rows),
                "rank_improved": len(rank_improved_rows),
                "mean_target_rank_gain": sum(rank_deltas) / len(rank_deltas) if rank_deltas else 0.0,
                "mean_prose_margin_delta": sum(prose_deltas) / len(prose_deltas) if prose_deltas else 0.0,
                "mean_echo_margin_delta": sum(echo_deltas) / len(echo_deltas) if echo_deltas else 0.0,
                "mean_stop_margin_delta": sum(stop_deltas) / len(stop_deltas) if stop_deltas else 0.0,
                "min_abs_shift_for_top_change": min([abs(finite_float(x.get("projection_shift"))) for x in top_changed_rows], default=None),
                "min_abs_shift_for_rank_improve": min([abs(finite_float(x.get("projection_shift"))) for x in rank_improved_rows], default=None),
                "top_token_pairs": dict(
                    Counter(
                        f"{x.get('base_top_token')}->{x.get('patch_top_token')}"
                        for x in top_changed_rows
                    ).most_common(12)
                ),
            }
        )
    out.sort(
        key=lambda row: (int(row.get("top_token_changed") or 0), int(row.get("rank_improved") or 0), abs(float(row.get("mean_projection_shift") or 0))),
        reverse=True,
    )
    return out


def behavior_effect_rows(model: str, phase223_round: str) -> list[dict[str, Any]]:
    path = PHASE223_ROOT / phase223_round / f"phase223_{model}_effect_rows.jsonl"
    rows = []
    for row in p214.iter_jsonl(path) or []:
        spec224 = None
        for k224, k223 in SPEC_MAP_224_TO_223.items():
            if row.get("spec_id") == k223:
                spec224 = k224
                break
        if spec224 is None:
            continue
        new = dict(row)
        new["spec_id_224"] = spec224
        rows.append(new)
    return rows


def correlate_behavior(summary_rows: list[dict[str, Any]], behavior_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    behavior_by: dict[tuple[str, str], dict[str, Any]] = {}
    for row in behavior_rows:
        behavior_by[(str(row.get("spec_id_224")), str(row.get("condition")))] = row
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        buckets[(str(row.get("spec_id")), str(row.get("condition")))].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, condition = key
        behavior = behavior_by.get(key, {})
        mean_shift = sum(finite_float(x.get("mean_projection_shift")) for x in items) / len(items)
        total_top = sum(finite_int(x.get("top_token_changed")) for x in items)
        total_rank = sum(finite_int(x.get("rank_improved")) for x in items)
        out.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase225_behavior_readout_correlation_row",
                "spec_id": spec_id,
                "condition": condition,
                "summary_layers": len(items),
                "mean_layer_projection_shift": mean_shift,
                "total_top_token_changed": total_top,
                "total_rank_improved": total_rank,
                "success_damage_match_loss": finite_int(behavior.get("damage_match_loss")),
                "drift_repair_match_gain": finite_int(behavior.get("repair_match_gain")),
                "success_patch_outputs": behavior.get("success_patch_outputs", {}),
                "drift_patch_outputs": behavior.get("drift_patch_outputs", {}),
            }
        )
    out.sort(
        key=lambda row: abs(int(row.get("success_damage_match_loss") or 0)) + abs(int(row.get("drift_repair_match_gain") or 0)) + int(row.get("total_top_token_changed") or 0),
        reverse=True,
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    p224_dir = PHASE224_ROOT / args.phase224_round
    p223_round = args.phase223_round
    propagation_rows = list(p214.iter_jsonl(p224_dir / f"phase224_{args.model}_propagation_rows.jsonl") or [])
    threshold_rows = summarize_thresholds(propagation_rows)
    behavior_rows = behavior_effect_rows(args.model, p223_round)
    correlation_rows = correlate_behavior(threshold_rows, behavior_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Readout competition threshold",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "propagation_rows": len(propagation_rows),
        "threshold_rows": len(threshold_rows),
        "behavior_rows": len(behavior_rows),
        "correlation_rows": len(correlation_rows),
        "total_top_token_changed": sum(int(row.get("top_token_changed") or 0) for row in threshold_rows),
        "total_rank_improved": sum(int(row.get("rank_improved") or 0) for row in threshold_rows),
        "top_threshold_rows": threshold_rows[:80],
        "top_correlation_rows": correlation_rows[:80],
    }
    write_json(out_dir / f"phase225_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase225_{args.model}_threshold_rows.jsonl", threshold_rows)
    write_jsonl(out_dir / f"phase225_{args.model}_behavior_correlation_rows.jsonl", correlation_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "propagation_rows": len(propagation_rows),
                "threshold_rows": len(threshold_rows),
                "top_token_changed": payload["total_top_token_changed"],
                "rank_improved": payload["total_rank_improved"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase225_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    threshold_rows = []
    correlation_rows = []
    for model in MODELS:
        threshold_rows.extend(p214.iter_jsonl(out_dir / f"phase225_{model}_threshold_rows.jsonl") or [])
        correlation_rows.extend(p214.iter_jsonl(out_dir / f"phase225_{model}_behavior_correlation_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model readout competition threshold",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "propagation_rows": sum(int(x.get("propagation_rows") or 0) for x in summaries),
        "threshold_rows": len(threshold_rows),
        "behavior_correlation_rows": len(correlation_rows),
        "total_top_token_changed": sum(int(row.get("top_token_changed") or 0) for row in threshold_rows),
        "total_rank_improved": sum(int(row.get("rank_improved") or 0) for row in threshold_rows),
        "top_threshold_rows": sorted(
            threshold_rows,
            key=lambda row: (int(row.get("top_token_changed") or 0), int(row.get("rank_improved") or 0), abs(float(row.get("mean_projection_shift") or 0))),
            reverse=True,
        )[:100],
        "top_correlation_rows": sorted(
            correlation_rows,
            key=lambda row: abs(int(row.get("success_damage_match_loss") or 0)) + abs(int(row.get("drift_repair_match_gain") or 0)) + int(row.get("total_top_token_changed") or 0),
            reverse=True,
        )[:100],
    }
    write_json(out_dir / "phase225_cross_model_summary.json", payload)
    lines = ["# Phase 225 readout competition threshold", ""]
    for key in [
        "propagation_rows",
        "threshold_rows",
        "behavior_correlation_rows",
        "total_top_token_changed",
        "total_rank_improved",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "| spec | group | condition | layer | rows | shift | top changed | rank improved | prose d | echo d | token pairs |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["top_threshold_rows"][:60]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('condition')} | {row.get('observe_layer')} | "
            f"{row.get('rows')} | {float(row.get('mean_projection_shift') or 0.0):.4f} | "
            f"{row.get('top_token_changed')} | {row.get('rank_improved')} | "
            f"{float(row.get('mean_prose_margin_delta') or 0.0):.4f} | {float(row.get('mean_echo_margin_delta') or 0.0):.4f} | "
            f"{row.get('top_token_pairs')} |"
        )
    lines.extend(
        [
            "",
            "## Behavior correlation",
            "",
            "| spec | condition | shift | top changed | rank improved | damage | repair | success outputs | drift outputs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["top_correlation_rows"][:40]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('condition')} | {float(row.get('mean_layer_projection_shift') or 0.0):.4f} | "
            f"{row.get('total_top_token_changed')} | {row.get('total_rank_improved')} | "
            f"{row.get('success_damage_match_loss')} | {row.get('drift_repair_match_gain')} | "
            f"{row.get('success_patch_outputs')} | {row.get('drift_patch_outputs')} |"
        )
    (out_dir / "phase225_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "propagation_rows": payload["propagation_rows"],
                "threshold_rows": payload["threshold_rows"],
                "top_token_changed": payload["total_top_token_changed"],
                "rank_improved": payload["total_rank_improved"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase225 readout competition threshold")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="readout_competition_threshold")
    parser.add_argument("--phase224-round", default="multilayer_activation_propagation")
    parser.add_argument("--phase223-round", default="channel_activation_gate_validation")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
