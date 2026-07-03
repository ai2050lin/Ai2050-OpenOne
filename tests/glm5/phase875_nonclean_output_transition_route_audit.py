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


PHASE = 875
RESULT_ROOT = Path("tests/result/phase875_nonclean_output_transition_route_audit")
DEFAULT_PHASE874_ROWS = Path("tests/result/phase874_state_transition_decomposition/combined/phase874_transition_rows.jsonl")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def has_tag(row: dict[str, Any], tag: str) -> bool:
    return tag in (row.get("field_tags") or [])


def route_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    base_label = str(row.get("base_rollout_label"))
    best_non_role = str(row.get("intervened_best_non_target_role"))
    if has_tag(row, "semantic_other_pressure"):
        tags.append("semantic_pressure")
    if has_tag(row, "object_dominates_class") or has_tag(row, "object_echo_pressure") or base_label == "object_echo" or best_non_role == "object_echo":
        tags.append("object_echo_recovery")
    if has_tag(row, "format_dominates") or base_label == "format_or_empty" or best_non_role.startswith("format_"):
        tags.append("format_recovery")
    if has_tag(row, "protocol_pressure"):
        tags.append("protocol_pressure")
    if finite(row.get("answer_delta")) > 0:
        tags.append("answer_lift")
    if finite(row.get("blocker_reduction")) > 0:
        tags.append("blocker_count_reduced")
    if finite(row.get("original_blocker_delta")) >= 0:
        tags.append("original_blocker_not_reduced")
    if not bool(row.get("phase866_pair_rule")):
        tags.append("not_clean_pair_rule")
    return tags or ["unclassified"]


def primary_route(row: dict[str, Any]) -> str:
    tags = set(route_tags(row))
    base_label = str(row.get("base_rollout_label"))
    if base_label == "object_echo" or "object_echo_recovery" in tags:
        return "object_echo_recovery"
    if base_label == "format_or_empty" or "format_recovery" in tags:
        return "format_recovery"
    if "semantic_pressure" in tags:
        return "semantic_pressure_transition"
    if "protocol_pressure" in tags:
        return "protocol_pressure_transition"
    return "other_nonclean_transition"


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("transition_class") == "nonclean_output_transition":
        out["route_tags"] = route_tags(out)
        out["primary_route"] = primary_route(out)
    elif out.get("transition_class") == "clean_causal_transition":
        out["route_tags"] = ["clean_causal_transition"]
        out["primary_route"] = "clean_causal_transition"
    else:
        out["route_tags"] = []
        out["primary_route"] = str(out.get("transition_class"))
    out["answer_delta_positive"] = finite(out.get("answer_delta")) > 0
    out["blocker_reduction_positive"] = finite(out.get("blocker_reduction")) > 0
    out["original_blocker_reduced"] = finite(out.get("original_blocker_delta")) < 0
    out["target_margin_positive"] = finite(out.get("intervened_clear_margin_vs_non_target")) > 0
    return out


def grouped_numeric(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(row)
    out: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        out[key] = {
            "n": len(items),
            "round_counts": dict(Counter(str(row.get("phase872_round")) for row in items)),
            "model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in items)),
            "base_to_intervened_counts": dict(Counter(f"{row.get('base_rollout_label')}->{row.get('intervened_rollout_label')}" for row in items)),
            "edit_mode_counts": dict(Counter(str(row.get("edit_mode")) for row in items)),
            "mean_answer_delta": mean([finite(row.get("answer_delta")) for row in items]),
            "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in items]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in items]),
            "mean_object_delta": mean([finite(row.get("object_delta")) for row in items]),
            "mean_target_margin_vs_non_target": mean([finite(row.get("intervened_clear_margin_vs_non_target")) for row in items]),
            "objects": sorted({str(row.get("object")) for row in items}),
        }
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transition_rows = [
        row for row in rows if row.get("transition_class") in {"nonclean_output_transition", "clean_causal_transition"}
    ]
    nonclean = [row for row in rows if row.get("transition_class") == "nonclean_output_transition"]
    clean = [row for row in rows if row.get("transition_class") == "clean_causal_transition"]
    tag_counts = Counter(tag for row in nonclean for tag in row.get("route_tags") or [])
    return {
        "n_rows": len(rows),
        "n_output_transitions": len(transition_rows),
        "n_clean_causal_transition": len(clean),
        "n_nonclean_output_transition": len(nonclean),
        "transition_class_counts": dict(Counter(str(row.get("transition_class")) for row in rows)),
        "nonclean_primary_route_counts": dict(Counter(str(row.get("primary_route")) for row in nonclean)),
        "nonclean_route_tag_counts": dict(tag_counts),
        "nonclean_round_counts": dict(Counter(str(row.get("phase872_round")) for row in nonclean)),
        "nonclean_model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in nonclean)),
        "nonclean_base_to_intervened_counts": dict(Counter(f"{row.get('base_rollout_label')}->{row.get('intervened_rollout_label')}" for row in nonclean)),
        "nonclean_reason_counts": dict(Counter(reason for row in nonclean for reason in row.get("nonclean_transition_reasons") or [])),
        "nonclean_by_primary_route": grouped_numeric(nonclean, "primary_route"),
        "transition_by_primary_route": grouped_numeric(transition_rows, "primary_route"),
        "clean_reference": {
            "n": len(clean),
            "round_counts": dict(Counter(str(row.get("phase872_round")) for row in clean)),
            "model_domain_counts": dict(Counter(f"{row.get('model')}:{row.get('domain')}" for row in clean)),
            "mean_answer_delta": mean([finite(row.get("answer_delta")) for row in clean]),
            "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in clean]),
            "mean_original_blocker_delta": mean([finite(row.get("original_blocker_delta")) for row in clean]),
            "mean_target_margin_vs_non_target": mean([finite(row.get("intervened_clear_margin_vs_non_target")) for row in clean]),
            "objects": sorted({str(row.get("object")) for row in clean}),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Phase 875 Nonclean Output Transition Route Audit",
        "",
        "- Boundary: offline route audit over Phase 874 transition rows; no new model run.",
        "- Goal: split nonclean output transitions into route families instead of discarding them.",
        "",
        "## Summary",
        "",
        f"- Output transitions: `{summary['n_output_transitions']}`",
        f"- Clean causal transitions: `{summary['n_clean_causal_transition']}`",
        f"- Nonclean output transitions: `{summary['n_nonclean_output_transition']}`",
        f"- Nonclean primary route counts: `{summary['nonclean_primary_route_counts']}`",
        f"- Nonclean route tag counts: `{summary['nonclean_route_tag_counts']}`",
        f"- Nonclean reason counts: `{summary['nonclean_reason_counts']}`",
        "",
        "## Primary Routes",
        "",
        "| route | n | rounds | model/domains | base->intervened | edit modes | mean ans | mean blocker red. | mean orig blocker | mean margin | objects |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for route, info in summary.get("nonclean_by_primary_route", {}).items():
        lines.append(
            f"| `{route}` | {info['n']} | `{info['round_counts']}` | `{info['model_domain_counts']}` | "
            f"`{info['base_to_intervened_counts']}` | `{info['edit_mode_counts']}` | "
            f"{finite(info['mean_answer_delta']):.3f} | {finite(info['mean_blocker_reduction']):.3f} | "
            f"{finite(info['mean_original_blocker_delta']):.3f} | {finite(info['mean_target_margin_vs_non_target']):.3f} | "
            f"`{info['objects']}` |"
        )
    lines += [
        "",
        "## Transition Rows",
        "",
        "| class | route | round | model | domain | object | prompt | mode | labels | tags | reasons | ans | block red. | orig block | margin |",
        "|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows") or []:
        if row.get("transition_class") not in {"nonclean_output_transition", "clean_causal_transition"}:
            continue
        lines.append(
            f"| `{row.get('transition_class')}` | `{row.get('primary_route')}` | `{row.get('phase872_round')}` | "
            f"{row.get('model')} | {row.get('domain')} | {row.get('object')} | `{row.get('prompt_variant')}` | "
            f"`{row.get('edit_mode')}` | `{row.get('base_rollout_label')}->{row.get('intervened_rollout_label')}` | "
            f"`{row.get('route_tags')}` | `{row.get('nonclean_transition_reasons')}` | "
            f"{finite(row.get('answer_delta')):.3f} | {finite(row.get('blocker_reduction')):.3f} | "
            f"{finite(row.get('original_blocker_delta')):.3f} | {finite(row.get('intervened_clear_margin_vs_non_target')):.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase874-rows", default=str(DEFAULT_PHASE874_ROWS))
    parser.add_argument("--output-round", default="combined")
    args = parser.parse_args()

    rows = [enrich(row) for row in read_jsonl(Path(args.phase874_rows))]
    payload = {
        "phase": PHASE,
        "title": "Nonclean Output Transition Route Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase874_rows": str(args.phase874_rows),
        "summary": summarize(rows),
        "rows": rows,
        "boundary": "Offline route-family audit; no new model run and no closure claim.",
    }
    out_dir = RESULT_ROOT / str(args.output_round)
    p846.write_json(out_dir / "phase875_summary.json", payload)
    p846.write_jsonl(out_dir / "phase875_route_rows.jsonl", rows)
    write_markdown(out_dir / "phase875_summary.md", payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
