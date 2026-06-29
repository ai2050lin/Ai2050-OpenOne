#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]
IN_ROOT = Path("results/glm5_phase765_commonsense_context_identity_closure_test")
OUT_ROOT = Path("results/glm5_phase766_prediction_sufficient_state_reliability_audit")


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize_observations(obs: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in obs:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(rows),
                "target_top1_rate": safe_mean([1.0 if r.get("target_top1") else 0.0 for r in rows]),
                "mean_target_rank": safe_mean([r.get("target_rank") for r in rows]),
                "mean_contrast_rank": safe_mean([r.get("contrast_rank") for r in rows]),
            }
        )
        out.append(payload)
    return out


def effect_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct = [r.get("source_direct_score") or {} for r in rows]
    return {
        "n": len(rows),
        "mean_target_logit_drop": safe_mean([r.get("target_logit_drop") for r in rows]),
        "mean_attention_mass": safe_mean([r.get("attention_mass_to_source") for r in rows]),
        "mean_direct_target_boost": safe_mean([d.get("direct_target_boost") for d in direct]),
        "mean_direct_total_route_suppression": safe_mean([d.get("direct_total_route_suppression") for d in direct]),
        "mean_direct_mean_margin_gain": safe_mean([d.get("direct_mean_margin_gain") for d in direct]),
        "mean_source_positions_n": safe_mean([r.get("source_positions_n") for r in rows]),
    }


def summarize_effects(effects: list[dict[str, Any]], observations: dict[str, dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"success": [], "failure": []})
    for row in effects:
        obs = observations.get(row["case_id"])
        if not obs:
            continue
        joined = {**row, **{f"obs_{k}": v for k, v in obs.items()}}
        status = "success" if obs.get("target_top1") else "failure"
        key = []
        for field in key_fields:
            if field.startswith("obs_"):
                key.append(joined.get(field))
            else:
                key.append(row.get(field))
        groups[tuple(key)][status].append(joined)
    out = []
    metric_names = [
        "mean_target_logit_drop",
        "mean_attention_mass",
        "mean_direct_target_boost",
        "mean_direct_total_route_suppression",
        "mean_direct_mean_margin_gain",
        "mean_source_positions_n",
    ]
    for key, buckets in sorted(groups.items()):
        success = effect_metrics(buckets["success"])
        failure = effect_metrics(buckets["failure"])
        payload = {field: value for field, value in zip(key_fields, key)}
        payload["success"] = success
        payload["failure"] = failure
        payload["gaps_success_minus_failure"] = {}
        for name in metric_names:
            sv = success.get(name)
            fv = failure.get(name)
            payload["gaps_success_minus_failure"][name] = None if sv is None or fv is None else float(sv) - float(fv)
        out.append(payload)
    return out


def top_discriminators(effect_rows: list[dict[str, Any]], metric: str, limit: int = 24) -> list[dict[str, Any]]:
    scored = []
    for row in effect_rows:
        gap = row.get("gaps_success_minus_failure", {}).get(metric)
        if gap is None:
            continue
        if row["success"]["n"] < 12 or row["failure"]["n"] < 12:
            continue
        scored.append(
            {
                "key": {k: v for k, v in row.items() if k not in {"success", "failure", "gaps_success_minus_failure"}},
                "success_n": row["success"]["n"],
                "failure_n": row["failure"]["n"],
                "metric": metric,
                "gap": gap,
                "success_value": row["success"].get(metric),
                "failure_value": row["failure"].get(metric),
            }
        )
    scored.sort(key=lambda r: abs(r["gap"]), reverse=True)
    return scored[:limit]


def audit_model(model: str, round_name: str) -> dict[str, Any]:
    row_path = IN_ROOT / round_name / f"phase765_{model}_rows.jsonl"
    rows = load_rows(row_path)
    observations = [r for r in rows if r.get("row_kind") == "commonsense_task_observation"]
    effects = [r for r in rows if r.get("row_kind") == "commonsense_fiber_effect"]
    obs_by_case = {r["case_id"]: r for r in observations}
    effect_by_relation_source = summarize_effects(effects, obs_by_case, ["obs_relation", "source_group"])
    effect_by_context_relation = summarize_effects(effects, obs_by_case, ["obs_context_format", "obs_relation"])
    effect_by_source = summarize_effects(effects, obs_by_case, ["source_group"])
    return {
        "model": model,
        "row_path": str(row_path),
        "n_rows": len(rows),
        "n_observations": len(observations),
        "n_effects": len(effects),
        "observation_by_relation": summarize_observations(observations, ["relation"]),
        "observation_by_context": summarize_observations(observations, ["context_format"]),
        "observation_by_domain": summarize_observations(observations, ["domain"]),
        "effect_by_relation_source": effect_by_relation_source,
        "effect_by_context_relation": effect_by_context_relation,
        "effect_by_source": effect_by_source,
        "top_attention_mass_gaps": top_discriminators(effect_by_relation_source, "mean_attention_mass"),
        "top_direct_target_boost_gaps": top_discriminators(effect_by_relation_source, "mean_direct_target_boost"),
        "top_target_drop_gaps": top_discriminators(effect_by_relation_source, "mean_target_logit_drop"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 766 Prediction-Sufficient State Reliability Audit ({payload['round']})",
        "",
        "- Status: `complete`",
        "- Input: Phase 765 commonsense confirm rows; no model was loaded.",
        "- Purpose: compare target-top1 success vs failure states.",
        "",
        "## Base Reliability By Relation",
        "",
        "| model | relation | n | top1 | target rank | contrast rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for row in payload["by_model"].get(model, {}).get("observation_by_relation", []):
            lines.append(
                f"| {model} | `{row['relation']}` | {row['n']} | {fmt(row['target_top1_rate'])} | "
                f"{fmt(row['mean_target_rank'])} | {fmt(row['mean_contrast_rank'])} |"
            )
    lines += [
        "",
        "## Success Minus Failure By Source Group",
        "",
        "| model | source | success n | failure n | target drop gap | attention gap | direct boost gap | route suppression gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for row in payload["by_model"].get(model, {}).get("effect_by_source", []):
            gap = row["gaps_success_minus_failure"]
            lines.append(
                f"| {model} | `{row['source_group']}` | {row['success']['n']} | {row['failure']['n']} | "
                f"{fmt(gap['mean_target_logit_drop'])} | {fmt(gap['mean_attention_mass'])} | "
                f"{fmt(gap['mean_direct_target_boost'])} | {fmt(gap['mean_direct_total_route_suppression'])} |"
            )
    lines += [
        "",
        "## Top Attention-Mass Gaps",
        "",
        "| model | key | success n | failure n | gap | success | failure |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for row in payload["by_model"].get(model, {}).get("top_attention_mass_gaps", [])[:8]:
            lines.append(
                f"| {model} | `{row['key']}` | {row['success_n']} | {row['failure_n']} | "
                f"{fmt(row['gap'])} | {fmt(row['success_value'])} | {fmt(row['failure_value'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- This is an observational audit over Phase 765 rows, not a new intervention.",
        "- If failures have lower attention/direct gaps on object or relation sources, the bottleneck is likely state formation.",
        "- If failures have similar source effects but low target top1, the bottleneck is more likely readout threshold or candidate competition.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    args = parser.parse_args()
    by_model = {}
    for model in MODELS:
        path = IN_ROOT / args.round_name / f"phase765_{model}_rows.jsonl"
        if path.exists():
            by_model[model] = audit_model(model, args.round_name)
    payload = {
        "phase": 766,
        "title": "Prediction-Sufficient State Reliability Audit",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
        "strict_interpretation": "Offline success/failure audit of commonsense prediction-sufficient states.",
    }
    out_dir = OUT_ROOT / args.round_name
    write_json(out_dir / "phase766_reliability_audit_summary.json", payload)
    write_markdown(out_dir / "phase766_reliability_audit_summary.md", payload)
    print(json.dumps({"status": "complete", "models": sorted(by_model), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
