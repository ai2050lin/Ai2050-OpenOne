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


PHASE = 864
MODELS = p846.MODELS
PHASE862_ROOT = Path("tests/result/phase862_negative_blocker_sign_mechanism_audit")
PHASE863_SUMMARY = Path("tests/result/phase863_dominant_auxiliary_channel_role_split_audit/phase863_summary.json")
RESULT_ROOT = Path("tests/result/phase864_answer_lift_vs_blocker_weakening_route_separation")


FORMAT_ROLES = {"format_space", "format_punct", "protocol_word"}
OBJECT_ROLES = {"object_echo", "identity_class_overlap"}
ANSWER_ROLES = {"answer_class", "strict_target", "identity_class_overlap"}
BLOCKER_ROLES = {"other", "other_blocker", "number", "protocol_word", "format_space", "format_punct", "object_echo"}


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


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("domain")), str(row.get("case_id")), str(row.get("prompt_variant")))


def intervention_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("domain")),
        str(row.get("condition_type")),
        str(row.get("subset_name")),
        str(row.get("edit_mode")),
    )


def role_count(tokens: list[dict[str, Any]], roles: set[str]) -> int:
    return sum(1 for token in tokens or [] if str(token.get("role")) in roles)


def top1_role(tokens: list[dict[str, Any]]) -> str:
    if not tokens:
        return "none"
    return str(tokens[0].get("role") or "unknown")


def pair_delta(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    base_top = base.get("top_tokens") or []
    row_top = row.get("top_tokens") or []
    base_blocker_top = base.get("blocker_top_tokens") or []
    row_blocker_top = row.get("blocker_top_tokens") or []
    class_blocker_reduction = (
        finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
        if base.get("class_blocker_count") is not None and row.get("class_blocker_count") is not None
        else None
    )
    clear_blocker_reduction = (
        finite(base.get("clear_class_blocker_count")) - finite(row.get("clear_class_blocker_count"))
        if base.get("clear_class_blocker_count") is not None and row.get("clear_class_blocker_count") is not None
        else None
    )
    margin_gain = (
        finite(row.get("class_minus_object_logit")) - finite(base.get("class_minus_object_logit"))
        if base.get("class_minus_object_logit") is not None and row.get("class_minus_object_logit") is not None
        else None
    )
    return {
        "answer_delta": finite(row.get("class_answer_delta")),
        "strict_delta": finite(row.get("strict_delta")),
        "object_delta": finite(row.get("object_delta")),
        "margin_gain": margin_gain,
        "class_blocker_reduction": class_blocker_reduction,
        "clear_blocker_reduction": clear_blocker_reduction,
        "original_blocker_delta_mean": finite(row.get("original_blocker_delta_mean")),
        "rollout_clear_gain": bool(not base.get("rollout_clear_answer_class") and row.get("rollout_clear_answer_class")),
        "rollout_clear_loss": bool(base.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class")),
        "rollout_gain": bool(not base.get("rollout_answer_class") and row.get("rollout_answer_class")),
        "rollout_loss": bool(base.get("rollout_answer_class") and not row.get("rollout_answer_class")),
        "object_echo_reduced": bool(base.get("rollout_object_echo") and not row.get("rollout_object_echo")),
        "object_echo_induced": bool(not base.get("rollout_object_echo") and row.get("rollout_object_echo")),
        "format_or_other_reduced": bool(base.get("rollout_other_or_format") and not row.get("rollout_other_or_format")),
        "format_or_other_induced": bool(not base.get("rollout_other_or_format") and row.get("rollout_other_or_format")),
        "top_format_count_delta": role_count(row_top, FORMAT_ROLES) - role_count(base_top, FORMAT_ROLES),
        "top_object_count_delta": role_count(row_top, OBJECT_ROLES) - role_count(base_top, OBJECT_ROLES),
        "top_answer_count_delta": role_count(row_top, ANSWER_ROLES) - role_count(base_top, ANSWER_ROLES),
        "blocker_top_format_count_delta": role_count(row_blocker_top, FORMAT_ROLES) - role_count(base_blocker_top, FORMAT_ROLES),
        "blocker_top_object_count_delta": role_count(row_blocker_top, OBJECT_ROLES) - role_count(base_blocker_top, OBJECT_ROLES),
        "top1_role_before": top1_role(base_top),
        "top1_role_after": top1_role(row_top),
        "label_before": base.get("rollout_label"),
        "label_after": row.get("rollout_label"),
    }


def phase863_roles() -> dict[tuple[str, str, str], str]:
    if not PHASE863_SUMMARY.exists():
        return {}
    payload = read_json(PHASE863_SUMMARY)
    out: dict[tuple[str, str, str], str] = {}
    for model_name, model in (payload.get("model_summaries") or {}).items():
        for domain in model.get("domain_results") or []:
            for channel in domain.get("channels") or []:
                out[(str(model_name), str(domain.get("domain")), str(channel.get("gear_key")))] = str(channel.get("role_class"))
    return out


def aggregate_pairs(model_name: str, rows: list[dict[str, Any]], role_map: dict[tuple[str, str, str], str]) -> list[dict[str, Any]]:
    originals = {row_key(row): row for row in rows if row.get("condition_type") == "original"}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition_type") == "original":
            continue
        base = originals.get(row_key(row))
        if base is None:
            continue
        key = intervention_key(row)
        grouped[key].append(pair_delta(base, row))
        gear_keys = row.get("gear_keys") or []
        role_classes = [
            role_map.get((model_name, str(row.get("domain")), str(gear_key)), "unknown")
            for gear_key in gear_keys
        ]
        meta[key] = {
            "model": model_name,
            "domain": row.get("domain"),
            "condition_type": row.get("condition_type"),
            "subset_name": row.get("subset_name"),
            "edit_mode": row.get("edit_mode"),
            "gear_keys": gear_keys,
            "channel_role_classes": role_classes,
            "source_candidate_role": row.get("source_candidate_role"),
            "source_best_mode": row.get("source_best_mode"),
        }
    out = []
    for key, deltas in grouped.items():
        row = dict(meta[key])
        row.update(summarize_deltas(deltas))
        row["route_class"] = classify_route(row)
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("domain")), str(row.get("condition_type")), str(row.get("subset_name")), str(row.get("edit_mode"))))
    return out


def summarize_deltas(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_pairs": len(deltas),
        "clear_gain": sum(1 for row in deltas if row["rollout_clear_gain"]),
        "clear_loss": sum(1 for row in deltas if row["rollout_clear_loss"]),
        "rollout_gain": sum(1 for row in deltas if row["rollout_gain"]),
        "rollout_loss": sum(1 for row in deltas if row["rollout_loss"]),
        "object_echo_reduced": sum(1 for row in deltas if row["object_echo_reduced"]),
        "object_echo_induced": sum(1 for row in deltas if row["object_echo_induced"]),
        "format_or_other_reduced": sum(1 for row in deltas if row["format_or_other_reduced"]),
        "format_or_other_induced": sum(1 for row in deltas if row["format_or_other_induced"]),
        "mean_answer_delta": mean([row["answer_delta"] for row in deltas]),
        "mean_strict_delta": mean([row["strict_delta"] for row in deltas]),
        "mean_object_delta": mean([row["object_delta"] for row in deltas]),
        "mean_margin_gain": mean([finite(row["margin_gain"]) for row in deltas if row["margin_gain"] is not None]),
        "mean_class_blocker_reduction": mean([finite(row["class_blocker_reduction"]) for row in deltas if row["class_blocker_reduction"] is not None]),
        "mean_clear_blocker_reduction": mean([finite(row["clear_blocker_reduction"]) for row in deltas if row["clear_blocker_reduction"] is not None]),
        "mean_original_blocker_delta": mean([row["original_blocker_delta_mean"] for row in deltas]),
        "mean_top_format_count_delta": mean([finite(row["top_format_count_delta"]) for row in deltas]),
        "mean_top_object_count_delta": mean([finite(row["top_object_count_delta"]) for row in deltas]),
        "mean_top_answer_count_delta": mean([finite(row["top_answer_count_delta"]) for row in deltas]),
        "mean_blocker_top_format_count_delta": mean([finite(row["blocker_top_format_count_delta"]) for row in deltas]),
        "mean_blocker_top_object_count_delta": mean([finite(row["blocker_top_object_count_delta"]) for row in deltas]),
        "top1_transitions": dict(Counter(f"{row['top1_role_before']}->{row['top1_role_after']}" for row in deltas)),
        "label_transitions": dict(Counter(f"{row['label_before']}->{row['label_after']}" for row in deltas)),
    }


def classify_route(row: dict[str, Any]) -> str:
    clear_gain = int(row.get("clear_gain") or 0)
    clear_loss = int(row.get("clear_loss") or 0)
    answer = finite(row.get("mean_answer_delta"))
    blocker_reduction = finite(row.get("mean_class_blocker_reduction"))
    blocker_delta = finite(row.get("mean_original_blocker_delta"))
    object_delta = finite(row.get("mean_object_delta"))
    format_induced = int(row.get("format_or_other_induced") or 0)
    object_induced = int(row.get("object_echo_induced") or 0)
    answer_lift = clear_gain > 0 and answer > 0
    blocker_weakening = clear_gain > 0 and blocker_reduction > 0 and blocker_delta < 0
    harmful = clear_loss > 0 and clear_gain == 0
    object_risk = object_induced > 0 or object_delta > 0.25
    format_risk = format_induced > 0 or finite(row.get("mean_top_format_count_delta")) > 0.5
    if harmful:
        return "harmful_or_blocker_amplifying"
    if answer_lift and blocker_weakening and object_risk:
        return "mixed_answer_blocker_with_object_side_effect"
    if answer_lift and blocker_weakening:
        return "mixed_answer_lift_and_blocker_weakening"
    if answer_lift and object_risk:
        return "answer_lift_with_object_echo_side_effect"
    if answer_lift:
        return "answer_lift_dominant"
    if blocker_weakening:
        return "blocker_weakening_dominant"
    if format_risk:
        return "format_side_effect_or_unresolved"
    return "weak_or_unresolved"


def summarize_model(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
    return {
        "model": model_name,
        "status": "complete" if rows else "no_route_rows",
        "n_route_rows": len(rows),
        "domains": sorted(by_domain.keys()),
        "route_class_counts": dict(Counter(str(row.get("route_class")) for row in rows)),
        "domain_route_class_counts": {
            domain: dict(Counter(str(row.get("route_class")) for row in group))
            for domain, group in sorted(by_domain.items())
        },
        "full_set_rows": [row for row in rows if row.get("condition_type") == "full_set"],
        "dominant_channel_rows": [
            row
            for row in rows
            if row.get("condition_type") == "single_channel"
            and any(str(role).startswith("dominant") for role in row.get("channel_role_classes") or [])
        ],
    }


def run(source_round: str) -> dict[str, Any]:
    role_map = phase863_roles()
    model_summaries: dict[str, Any] = {}
    route_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        rows_path = PHASE862_ROOT / source_round / f"phase862_{model_name}_rows.jsonl"
        if not rows_path.exists():
            model_summaries[model_name] = {"model": model_name, "status": "missing", "n_route_rows": 0}
            continue
        source_rows = read_jsonl(rows_path)
        rows = aggregate_pairs(model_name, source_rows, role_map)
        route_rows.extend(rows)
        model_summaries[model_name] = summarize_model(model_name, rows)
    return {
        "phase": PHASE,
        "title": "Answer-Lift vs Blocker-Weakening Route Separation",
        "source_round": source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "model_summaries": model_summaries,
        "route_rows": route_rows,
        "route_class_counts": dict(Counter(str(row.get("route_class")) for row in route_rows)),
        "boundary": "offline route separation from Phase 862/863 data; no new model intervention",
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 864 Answer-Lift vs Blocker-Weakening Route Separation",
        "",
        "- Source: Phase 862 main rows + Phase 863 channel roles.",
        "- Boundary: offline route separation, not new model intervention and not closure.",
        "",
        "## Summary",
        "",
        f"- route_class_counts: `{payload.get('route_class_counts')}`",
        "",
        "## Full-Set Routes",
        "",
        "| model | domain | mode | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta | object echo +/- | format/other +/- |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("route_rows") or []:
        if row.get("condition_type") != "full_set":
            continue
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | `{row.get('edit_mode')}` | `{row.get('route_class')}` | "
            f"{row.get('clear_gain', 0)}/{row.get('clear_loss', 0)} | "
            f"{finite(row.get('mean_answer_delta')):.4f} | {finite(row.get('mean_class_blocker_reduction')):.4f} | "
            f"{finite(row.get('mean_original_blocker_delta')):.4f} | {finite(row.get('mean_object_delta')):.4f} | "
            f"{row.get('object_echo_reduced', 0)}/{row.get('object_echo_induced', 0)} | "
            f"{row.get('format_or_other_reduced', 0)}/{row.get('format_or_other_induced', 0)} |"
        )
    lines += [
        "",
        "## Dominant Channel Routes",
        "",
        "| model | domain | gear | mode | channel role | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("route_rows") or []:
        if row.get("condition_type") != "single_channel":
            continue
        roles = row.get("channel_role_classes") or []
        if not any(str(role).startswith("dominant") for role in roles):
            continue
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | `{'+'.join(row.get('gear_keys') or [])}` | `{row.get('edit_mode')}` | "
            f"`{roles}` | `{row.get('route_class')}` | {row.get('clear_gain', 0)}/{row.get('clear_loss', 0)} | "
            f"{finite(row.get('mean_answer_delta')):.4f} | {finite(row.get('mean_class_blocker_reduction')):.4f} | "
            f"{finite(row.get('mean_original_blocker_delta')):.4f} | {finite(row.get('mean_object_delta')):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="main")
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()
    payload = run(args.source_round)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase864_summary.json", payload)
    write_jsonl(out_dir / "phase864_route_rows.jsonl", payload.get("route_rows") or [])
    (out_dir / "phase864_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"phase": PHASE, "status": payload["status"], "route_class_counts": payload["route_class_counts"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
