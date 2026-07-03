#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 863
MODELS = p846.MODELS
SOURCE_ROOT = Path("tests/result/phase862_negative_blocker_sign_mechanism_audit")
RESULT_ROOT = Path("tests/result/phase863_dominant_auxiliary_channel_role_split_audit")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def channel_key_from_rows(rows: list[dict[str, Any]], domain: str, subset_name: str) -> str | None:
    for row in rows:
        if (
            str(row.get("domain")) == domain
            and str(row.get("condition_type")) == "single_channel"
            and str(row.get("subset_name")) == subset_name
        ):
            keys = row.get("gear_keys") or []
            if keys:
                return str(keys[0])
    return None


def effect_rows(summary: dict[str, Any], domain: str, condition_type: str, subset_name: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in summary.get("pair_effects") or []:
        if str(row.get("domain")) != domain:
            continue
        if str(row.get("condition_type")) != condition_type:
            continue
        if subset_name is not None and str(row.get("subset_name")) != subset_name:
            continue
        out.append(row)
    return out


def channel_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clear_gain = sum(int(row.get("clear_rollout_gain") or 0) for row in rows)
    clear_loss = sum(int(row.get("clear_rollout_loss") or 0) for row in rows)
    blocker_modes = [str(row.get("edit_mode")) for row in rows if row.get("blocker_weakening_supported")]
    answer_modes = [str(row.get("edit_mode")) for row in rows if row.get("answer_lift_supported")]
    clear_modes = [str(row.get("edit_mode")) for row in rows if int(row.get("clear_rollout_gain") or 0) > 0]
    return {
        "modes": [str(row.get("edit_mode")) for row in rows],
        "total_clear_gain": clear_gain,
        "total_clear_loss": clear_loss,
        "clear_modes": sorted(clear_modes),
        "blocker_weakening_modes": sorted(blocker_modes),
        "answer_lift_modes": sorted(answer_modes),
        "mean_blocker_reduction": mean([finite(row.get("mean_class_blocker_reduction")) for row in rows]),
        "mean_original_blocker_delta": mean([finite(row.get("mean_original_blocker_delta")) for row in rows]),
        "mean_answer_delta": mean([finite(row.get("mean_answer_delta")) for row in rows]),
        "mean_object_delta": mean([finite(row.get("mean_object_delta")) for row in rows]),
        "blocker_weakening_mode_count": len(blocker_modes),
        "answer_lift_mode_count": len(answer_modes),
    }


def classify_channel(stats: dict[str, Any], max_gain: int) -> str:
    gain = int(stats.get("total_clear_gain") or 0)
    loss = int(stats.get("total_clear_loss") or 0)
    blocker_count = int(stats.get("blocker_weakening_mode_count") or 0)
    answer_count = int(stats.get("answer_lift_mode_count") or 0)
    if gain == max_gain and gain > 0:
        if blocker_count >= 2 and answer_count >= 2:
            return "dominant_answer_and_blocker_channel"
        if answer_count >= 2:
            return "dominant_answer_lift_channel"
        if blocker_count >= 2:
            return "dominant_blocker_channel"
        return "dominant_effect_channel"
    if loss > 0 and gain == 0:
        return "risky_or_antagonistic_auxiliary_channel"
    if gain > 0:
        if blocker_count > 0 and answer_count > 0:
            return "auxiliary_mixed_channel"
        if blocker_count > 0:
            return "auxiliary_blocker_channel"
        if answer_count > 0:
            return "auxiliary_answer_lift_channel"
    return "weak_or_unresolved_channel"


def analyze_model(model_name: str, round_name: str) -> dict[str, Any]:
    summary_path = SOURCE_ROOT / round_name / f"phase862_{model_name}_summary.json"
    rows_path = SOURCE_ROOT / round_name / f"phase862_{model_name}_rows.jsonl"
    if not summary_path.exists():
        return {"model": model_name, "status": "missing", "domains": [], "domain_results": []}
    summary = read_json(summary_path)
    rows = read_jsonl(rows_path) if rows_path.exists() else []
    if summary.get("status") != "complete":
        return {
            "model": model_name,
            "status": summary.get("status"),
            "domains": summary.get("domains") or [],
            "domain_results": [],
        }

    domain_results = []
    for domain in summary.get("domains") or []:
        full_rows = effect_rows(summary, str(domain), "full_set")
        subset_names = sorted({str(row.get("subset_name")) for row in summary.get("pair_effects") or [] if str(row.get("domain")) == str(domain) and str(row.get("condition_type")) == "single_channel"})
        channel_results = []
        for subset_name in subset_names:
            single_rows = effect_rows(summary, str(domain), "single_channel", subset_name)
            stats = channel_stats(single_rows)
            stats["subset_name"] = subset_name
            stats["gear_key"] = channel_key_from_rows(rows, str(domain), subset_name)
            channel_results.append(stats)
        max_gain = max((int(row.get("total_clear_gain") or 0) for row in channel_results), default=0)
        for row in channel_results:
            row["role_class"] = classify_channel(row, max_gain)
        channel_results.sort(key=lambda row: (int(row.get("total_clear_gain") or 0), -int(row.get("total_clear_loss") or 0)), reverse=True)
        full_stats = channel_stats(full_rows)
        dominant = channel_results[0] if channel_results else None
        auxiliary = channel_results[1:] if len(channel_results) > 1 else []
        domain_results.append(
            {
                "domain": domain,
                "full_set": full_stats,
                "channels": channel_results,
                "dominant_channel": dominant,
                "auxiliary_channels": auxiliary,
                "dominant_gain_share": (
                    int(dominant.get("total_clear_gain") or 0) / max(1, int(full_stats.get("total_clear_gain") or 0))
                    if dominant
                    else None
                ),
                "interpretation": domain_interpretation(full_stats, channel_results),
            }
        )
    return {
        "model": model_name,
        "status": "complete",
        "domains": summary.get("domains") or [],
        "domain_results": domain_results,
    }


def domain_interpretation(full_stats: dict[str, Any], channels: list[dict[str, Any]]) -> str:
    if not channels:
        return "no_single_channel_data"
    dominant = max(channels, key=lambda row: int(row.get("total_clear_gain") or 0))
    if int(dominant.get("total_clear_gain") or 0) >= int(full_stats.get("total_clear_gain") or 0):
        return "dominant_channel_explains_full_or_exceeds_full"
    if int(dominant.get("total_clear_gain") or 0) >= max(1, int(full_stats.get("total_clear_gain") or 0) // 2):
        return "dominant_channel_with_auxiliary_support"
    return "distributed_or_unresolved_channel_roles"


def summarize(models: list[dict[str, Any]], round_name: str) -> dict[str, Any]:
    domain_results = [domain for model in models for domain in model.get("domain_results") or []]
    return {
        "phase": PHASE,
        "title": "Dominant-Channel and Auxiliary-Channel Role Split Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(models) == len(MODELS) else "partial",
        "models": [row.get("model") for row in models],
        "model_summaries": {str(row.get("model")): row for row in models},
        "domain_count": len(domain_results),
        "domain_interpretations": {
            f"{model.get('model')}:{domain.get('domain')}": domain.get("interpretation")
            for model in models
            for domain in model.get("domain_results") or []
        },
        "boundary": "offline role split using Phase 862 single-channel data; no new model intervention",
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Phase 863 Dominant/Auxiliary Channel Role Split ({payload['round']})",
        "",
        "- Source: Phase 862 main single-channel effects.",
        "- Boundary: offline role split, not a new model intervention and not closure.",
        "",
        "## Domain Results",
        "",
        "| model | domain | full gain/loss | dominant gear | dominant role | dominant gain/loss | gain share | interpretation |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for model_name, model in payload.get("model_summaries", {}).items():
        for domain in model.get("domain_results") or []:
            full = domain.get("full_set") or {}
            dom = domain.get("dominant_channel") or {}
            lines.append(
                f"| {model_name} | {domain.get('domain')} | "
                f"{full.get('total_clear_gain', 0)}/{full.get('total_clear_loss', 0)} | "
                f"`{dom.get('gear_key')}` | `{dom.get('role_class')}` | "
                f"{dom.get('total_clear_gain', 0)}/{dom.get('total_clear_loss', 0)} | "
                f"{dom.get('total_clear_gain', 0) / max(1, int(full.get('total_clear_gain') or 0)):.3f} | "
                f"`{domain.get('interpretation')}` |"
            )
    lines += [
        "",
        "## Channel Details",
        "",
        "| model | domain | gear | subset | role | clear gain/loss | blocker modes | answer modes | mean blocker reduction | mean answer delta |",
        "|---|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for model_name, model in payload.get("model_summaries", {}).items():
        for domain in model.get("domain_results") or []:
            for channel in domain.get("channels") or []:
                lines.append(
                    f"| {model_name} | {domain.get('domain')} | `{channel.get('gear_key')}` | `{channel.get('subset_name')}` | "
                    f"`{channel.get('role_class')}` | {channel.get('total_clear_gain', 0)}/{channel.get('total_clear_loss', 0)} | "
                    f"`{channel.get('blocker_weakening_modes')}` | `{channel.get('answer_lift_modes')}` | "
                    f"{finite(channel.get('mean_blocker_reduction')):.4f} | {finite(channel.get('mean_answer_delta')):.4f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="main")
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()

    models = [analyze_model(model, args.source_round) for model in MODELS]
    payload = summarize(models, args.source_round)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase863_summary.json", payload)
    write_jsonl(out_dir / "phase863_domain_results.jsonl", [domain for model in models for domain in model.get("domain_results") or []])
    (out_dir / "phase863_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"phase": PHASE, "status": payload["status"], "domain_interpretations": payload["domain_interpretations"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
