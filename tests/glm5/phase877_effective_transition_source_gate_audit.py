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
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 877
RESULT_ROOT = Path("tests/result/phase877_effective_transition_source_gate_audit")
DEFAULT_ROWS = Path("tests/result/phase875_nonclean_output_transition_route_audit/combined_phase876/phase875_route_rows.jsonl")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def transition_label(row: dict[str, Any]) -> str:
    return f"{row.get('base_rollout_label')}->{row.get('intervened_rollout_label')}"


def parse_candidate_key(key: Any) -> tuple[list[str], str]:
    text = str(key or "")
    if ":" in text:
        gear_part, mode = text.rsplit(":", 1)
    else:
        gear_part, mode = text, str("")
    gears = [part for part in gear_part.split("+") if part and part != "original"]
    return gears, mode


def prompt_gate(prompt_variant: Any) -> str:
    variant = str(prompt_variant or "")
    if "semantic" in variant:
        return "semantic_prompt_gate"
    if "echo" in variant:
        return "echo_prompt_gate"
    if "format" in variant:
        return "format_prompt_gate"
    if "direct" in variant:
        return "direct_prompt_gate"
    return "other_prompt_gate"


def field_gate_labels(row: dict[str, Any]) -> list[str]:
    tags = set(str(tag) for tag in (row.get("field_tags") or []))
    labels: list[str] = []
    if "semantic_other_pressure" in tags:
        labels.append("semantic_field_gate")
    if "format_dominates" in tags:
        labels.append("format_field_gate")
    if "object_dominates_class" in tags or "object_echo_pressure" in tags:
        labels.append("object_echo_field_gate")
    if "protocol_pressure" in tags:
        labels.append("protocol_field_gate")
    if "too_many_blockers" in tags:
        labels.append("high_blocker_field_gate")
    if "field_low_pressure" in tags:
        labels.append("low_pressure_field_gate")
    return labels or ["unlabeled_field_gate"]


def route_source_gate(row: dict[str, Any]) -> str:
    route = str(row.get("primary_route"))
    if route == "clean_causal_transition":
        return "clean_blocker_weakening_gate"
    if route == "semantic_pressure_transition":
        return "semantic_answer_lift_gate"
    if route == "format_recovery":
        return "format_recovery_gate"
    if route == "object_echo_recovery":
        return "object_echo_recovery_gate"
    if route == "protocol_pressure_transition":
        return "protocol_pressure_gate"
    return "other_effective_transition_gate"


def dominant_pressure(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        "semantic": finite(row.get("field_semantic_other_pressure")),
        "format": finite(row.get("field_format_pressure")),
        "object_echo": finite(row.get("field_object_echo_pressure")),
        "protocol": finite(row.get("field_protocol_pressure")),
        "blocker_count": finite(row.get("field_class_blocker_count")),
    }
    route_values = {key: value for key, value in values.items() if key != "blocker_count"}
    max_value = max(route_values.values()) if route_values else 0.0
    winners = [key for key, value in route_values.items() if value == max_value and value > 0]
    return {
        "pressure_values": values,
        "dominant_pressure": "+".join(winners) if winners else "none",
        "dominant_pressure_value": max_value,
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    gears, mode_from_key = parse_candidate_key(row.get("candidate_key"))
    source_gate = route_source_gate(row)
    field_gates = field_gate_labels(row)
    pressure = dominant_pressure(row)
    return {
        "phase": PHASE,
        "phase872_round": row.get("phase872_round"),
        "transition_class": row.get("transition_class"),
        "primary_route": row.get("primary_route"),
        "route_tags": row.get("route_tags") or [],
        "source_gate": source_gate,
        "field_gates": field_gates,
        "prompt_gate": prompt_gate(row.get("prompt_variant")),
        "candidate_key": row.get("candidate_key"),
        "gear_keys": gears,
        "edit_mode": row.get("edit_mode") or mode_from_key,
        "source_purity_class": row.get("source_purity_class"),
        "source_predict_clean_mixed": row.get("source_predict_clean_mixed"),
        "model": row.get("model"),
        "domain": row.get("domain"),
        "case_id": row.get("case_id"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "transition_label": transition_label(row),
        "base_generated_clean": row.get("base_generated_clean"),
        "intervened_generated_clean": row.get("intervened_generated_clean"),
        "base_top1_role": row.get("base_top1_role"),
        "base_top1_token": row.get("base_top1_token"),
        "intervened_top1_role": row.get("intervened_top1_role"),
        "intervened_top1_token": row.get("intervened_top1_token"),
        "base_best_non_target_role": row.get("base_best_non_target_role"),
        "base_best_non_target_token": row.get("base_best_non_target_token"),
        "intervened_best_non_target_role": row.get("intervened_best_non_target_role"),
        "intervened_best_non_target_token": row.get("intervened_best_non_target_token"),
        "field_tags": row.get("field_tags") or [],
        "field_top10_role_counts": row.get("field_top10_role_counts") or {},
        **pressure,
        "answer_delta": finite(row.get("answer_delta")),
        "blocker_reduction": finite(row.get("blocker_reduction")),
        "clear_blocker_reduction": finite(row.get("clear_blocker_reduction")),
        "original_blocker_delta": finite(row.get("original_blocker_delta")),
        "object_delta": finite(row.get("object_delta")),
        "target_margin": finite(row.get("intervened_clear_margin_vs_non_target")),
        "base_clear_rank": finite(row.get("base_clear_rank")),
        "intervened_clear_rank": finite(row.get("intervened_clear_rank")),
        "field_class_blocker_count": finite(row.get("field_class_blocker_count")),
        "field_clear_class_blocker_count": finite(row.get("field_clear_class_blocker_count")),
        "reducible_original_blockers": bool(row.get("reducible_original_blockers")),
        "original_blocker_reduced": bool(row.get("original_blocker_reduced")),
        "nonclean_transition_reasons": row.get("nonclean_transition_reasons") or [],
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    out: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        out[name] = {
            "n": len(items),
            "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in items)),
            "route_counts": dict(Counter(str(row.get("primary_route")) for row in items)),
            "source_gate_counts": dict(Counter(str(row.get("source_gate")) for row in items)),
            "prompt_gate_counts": dict(Counter(str(row.get("prompt_gate")) for row in items)),
            "transition_label_counts": dict(Counter(str(row.get("transition_label")) for row in items)),
            "edit_mode_counts": dict(Counter(str(row.get("edit_mode")) for row in items)),
            "objects": sorted({str(row.get("object")) for row in items}),
            "prompts": sorted({str(row.get("prompt_variant")) for row in items}),
            "mean_answer_delta": mean([finite(row.get("answer_delta")) for row in items]),
            "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in items]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in items]),
            "mean_target_margin": mean([finite(row.get("target_margin")) for row in items]),
            "mean_field_class_blocker_count": mean([finite(row.get("field_class_blocker_count")) for row in items]),
        }
    return out


def object_prompt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('object')}::{row.get('prompt_variant')}"
        groups[key].append(row)
    out: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        out[name] = {
            "n": len(items),
            "route_counts": dict(Counter(str(row.get("primary_route")) for row in items)),
            "candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in items)),
            "edit_modes": sorted({str(row.get("edit_mode")) for row in items}),
            "transition_labels": dict(Counter(str(row.get("transition_label")) for row in items)),
            "source_gates": dict(Counter(str(row.get("source_gate")) for row in items)),
            "field_gates": dict(Counter(gate for row in items for gate in row.get("field_gates") or [])),
            "mean_answer_delta": mean([finite(row.get("answer_delta")) for row in items]),
            "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in items]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in items]),
            "mean_target_margin": mean([finite(row.get("target_margin")) for row in items]),
        }
    return out


def original_blocker_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonclean = [row for row in rows if row.get("transition_class") == "nonclean_output_transition"]
    return {
        "n_nonclean": len(nonclean),
        "original_blocker_not_reduced": sum(1 for row in nonclean if finite(row.get("original_blocker_delta")) >= 0),
        "original_blocker_reduced": sum(1 for row in nonclean if finite(row.get("original_blocker_delta")) < 0),
        "reducible_original_blockers": sum(1 for row in nonclean if row.get("reducible_original_blockers")),
        "reason_counts": dict(Counter(reason for row in nonclean for reason in row.get("nonclean_transition_reasons") or [])),
        "by_route": grouped_summary(nonclean, "primary_route"),
    }


def summarize(rows: list[dict[str, Any]], target_round: str) -> dict[str, Any]:
    effective = [
        compact_row(row)
        for row in rows
        if row.get("transition_class") in {"clean_causal_transition", "nonclean_output_transition"}
    ]
    target = [row for row in effective if str(row.get("phase872_round")) == target_round]
    target_nonclean = [row for row in target if row.get("transition_class") == "nonclean_output_transition"]
    return {
        "n_effective_all_rounds": len(effective),
        "n_effective_target_round": len(target),
        "target_round": target_round,
        "target_transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in target)),
        "target_route_counts": dict(Counter(str(row.get("primary_route")) for row in target)),
        "target_model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in target)),
        "target_candidate_counts": dict(Counter(str(row.get("candidate_key")) for row in target)),
        "target_source_gate_counts": dict(Counter(str(row.get("source_gate")) for row in target)),
        "target_prompt_gate_counts": dict(Counter(str(row.get("prompt_gate")) for row in target)),
        "target_field_gate_counts": dict(Counter(gate for row in target for gate in row.get("field_gates") or [])),
        "target_transition_label_counts": dict(Counter(str(row.get("transition_label")) for row in target)),
        "target_by_candidate": grouped_summary(target, "candidate_key"),
        "target_by_source_gate": grouped_summary(target, "source_gate"),
        "target_by_object_prompt": object_prompt_summary(target),
        "target_nonclean_original_blocker_diagnostic": original_blocker_diagnostic(target),
        "same_source_multi_route_candidates": {
            key: info
            for key, info in grouped_summary(target, "candidate_key").items()
            if len(info.get("route_counts", {})) > 1
        },
        "target_rows": target,
        "target_nonclean_rows": target_nonclean,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Phase 877 Effective Transition Source-Gate Audit",
        "",
        "- Boundary: offline audit over Phase875/876 transition rows; no new model run.",
        "- Goal: identify source gear, prompt gate, field gate, and blocker diagnostics for Phase876 effective transitions.",
        "",
        "## Summary",
        "",
        f"- Target round: `{s['target_round']}`",
        f"- Effective transitions in target round: `{s['n_effective_target_round']}`",
        f"- Transition classes: `{s['target_transition_class_counts']}`",
        f"- Routes: `{s['target_route_counts']}`",
        f"- Model/domains: `{s['target_model_domain_counts']}`",
        f"- Source gates: `{s['target_source_gate_counts']}`",
        f"- Prompt gates: `{s['target_prompt_gate_counts']}`",
        f"- Field gates: `{s['target_field_gate_counts']}`",
        f"- Same-source multi-route candidates: `{list(s['same_source_multi_route_candidates'].keys())}`",
        "",
        "## Source Candidates",
        "",
        "| candidate | n | classes | routes | source gates | prompts | labels | mean ans | mean blocker red. | mean orig blocker | mean margin |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for key, info in s["target_by_candidate"].items():
        lines.append(
            f"| `{key}` | {info['n']} | `{info['transition_class_counts']}` | `{info['route_counts']}` | "
            f"`{info['source_gate_counts']}` | `{info['prompt_gate_counts']}` | `{info['transition_label_counts']}` | "
            f"{finite(info['mean_answer_delta']):.3f} | {finite(info['mean_blocker_reduction']):.3f} | "
            f"{finite(info['mean_original_blocker_delta']):.3f} | {finite(info['mean_target_margin']):.3f} |"
        )
    lines += [
        "",
        "## Object Prompt Entrances",
        "",
        "| object::prompt | n | routes | candidates | modes | labels | field gates | mean ans | mean blocker red. | mean orig blocker | mean margin |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for key, info in s["target_by_object_prompt"].items():
        lines.append(
            f"| `{key}` | {info['n']} | `{info['route_counts']}` | `{info['candidate_counts']}` | "
            f"`{info['edit_modes']}` | `{info['transition_labels']}` | `{info['field_gates']}` | "
            f"{finite(info['mean_answer_delta']):.3f} | {finite(info['mean_blocker_reduction']):.3f} | "
            f"{finite(info['mean_original_blocker_delta']):.3f} | {finite(info['mean_target_margin']):.3f} |"
        )
    lines += [
        "",
        "## Target Rows",
        "",
        "| class | route | model | domain | object | prompt | candidate | mode | label | top1 | best blocker | gates | ans | blocker red. | orig blocker | margin |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in s.get("target_rows") or []:
        lines.append(
            f"| `{row.get('transition_class')}` | `{row.get('primary_route')}` | {row.get('model')} | {row.get('domain')} | "
            f"{row.get('object')} | `{row.get('prompt_variant')}` | `{row.get('candidate_key')}` | `{row.get('edit_mode')}` | "
            f"`{row.get('transition_label')}` | `{row.get('base_top1_role')}->{row.get('intervened_top1_role')}` | "
            f"`{row.get('base_best_non_target_role')}->{row.get('intervened_best_non_target_role')}` | "
            f"`{[row.get('source_gate'), row.get('prompt_gate'), row.get('field_gates')]}` | "
            f"{finite(row.get('answer_delta')):.3f} | {finite(row.get('blocker_reduction')):.3f} | "
            f"{finite(row.get('original_blocker_delta')):.3f} | {finite(row.get('target_margin')):.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase875-rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--target-round", default="validation_phase876")
    parser.add_argument("--output-round", default="source_gate_phase876")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.phase875_rows))
    payload = {
        "phase": PHASE,
        "title": "Effective Transition Source-Gate Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase875_rows": str(args.phase875_rows),
        "target_round": str(args.target_round),
        "summary": summarize(rows, str(args.target_round)),
        "boundary": "Offline source/gate audit; no new model run and no closure claim.",
    }
    out_dir = RESULT_ROOT / str(args.output_round)
    p846.write_json(out_dir / "phase877_summary.json", payload)
    p846.write_jsonl(out_dir / "phase877_source_gate_rows.jsonl", payload["summary"]["target_rows"])
    write_markdown(out_dir / "phase877_summary.md", payload)
    printable = dict(payload["summary"])
    printable.pop("target_rows", None)
    printable.pop("target_nonclean_rows", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
