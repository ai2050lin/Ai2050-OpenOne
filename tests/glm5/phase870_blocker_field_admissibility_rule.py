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


PHASE = 870
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase870_blocker_field_admissibility_rule")
PHASE867_ROOT = Path("tests/result/phase867_clean_route_holdout_prediction")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def row_context_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model")),
        str(row.get("domain")),
        str(row.get("case_id")),
        str(row.get("prompt_variant")),
    )


def role_counts(blockers: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for blocker in blockers or []:
        counts[str(blocker.get("role") or "none")] += 1
    return counts


def context_field(base: dict[str, Any], blocker_threshold: float, format_threshold: float) -> dict[str, Any]:
    counts = role_counts(base.get("blocker_class_top_blockers") or [])
    format_pressure = counts.get("format_space", 0) + counts.get("format_punct", 0)
    object_echo_pressure = counts.get("object_echo", 0)
    semantic_other_pressure = counts.get("other_blocker", 0)
    protocol_pressure = counts.get("protocol_word", 0)
    class_blocker_count = finite(base.get("class_blocker_count"))
    margin = finite(base.get("class_minus_object_logit"))
    tags: list[str] = []
    if class_blocker_count >= float(blocker_threshold):
        tags.append("too_many_blockers")
    if margin < 0:
        tags.append("object_dominates_class")
    if format_pressure >= float(format_threshold):
        tags.append("format_dominates")
    if object_echo_pressure > 0:
        tags.append("object_echo_pressure")
    if semantic_other_pressure > 0:
        tags.append("semantic_other_pressure")
    if protocol_pressure > 0:
        tags.append("protocol_pressure")
    if not tags:
        tags.append("field_low_pressure")
    base_field_admissible = not (
        class_blocker_count >= float(blocker_threshold) or margin < 0 or format_pressure >= float(format_threshold)
    )
    return {
        "field_class_blocker_count": class_blocker_count,
        "field_clear_class_blocker_count": finite(base.get("clear_class_blocker_count")),
        "field_class_minus_object_logit": margin,
        "field_object_rank": finite(base.get("object_rank")),
        "field_top1_role": str(base.get("blocker_class_top_blocker_role") or "none"),
        "field_format_pressure": format_pressure,
        "field_object_echo_pressure": object_echo_pressure,
        "field_semantic_other_pressure": semantic_other_pressure,
        "field_protocol_pressure": protocol_pressure,
        "field_top10_role_counts": dict(counts),
        "field_tags": tags,
        "field_base_admissible": base_field_admissible,
    }


def pair_row(base: dict[str, Any], row: dict[str, Any], blocker_threshold: float, format_threshold: float, object_delta_threshold: float) -> dict[str, Any]:
    field = context_field(base, blocker_threshold, format_threshold)
    clear_gain = bool((not base.get("rollout_clear_answer_class")) and row.get("rollout_clear_answer_class"))
    clear_loss = bool(base.get("rollout_clear_answer_class") and (not row.get("rollout_clear_answer_class")))
    object_echo_induced = bool((not base.get("rollout_object_echo")) and row.get("rollout_object_echo"))
    format_or_other_induced = bool((not base.get("rollout_other_or_format")) and row.get("rollout_other_or_format"))
    blocker_reduction = finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
    clear_blocker_reduction = finite(base.get("clear_class_blocker_count")) - finite(row.get("clear_class_blocker_count"))
    answer_delta = finite(row.get("class_answer_delta"))
    original_blocker_delta = finite(row.get("original_blocker_delta_mean"))
    object_delta = finite(row.get("object_delta"))
    gear_effect_basic = bool(answer_delta > 0 and blocker_reduction > 0)
    reducible_original_blockers = bool(original_blocker_delta < 0)
    no_side_effect = bool(
        object_delta <= float(object_delta_threshold)
        and not object_echo_induced
        and not format_or_other_induced
    )
    phase866_pair_rule = bool(gear_effect_basic and reducible_original_blockers and no_side_effect)
    field_plus_effect_rule = bool(field["field_base_admissible"] and phase866_pair_rule)
    field_strict_admissible = bool(field["field_base_admissible"] and field["field_semantic_other_pressure"] == 0)
    field_strict_plus_effect_rule = bool(field_strict_admissible and phase866_pair_rule)
    target_clean_transition = bool(clear_gain and not clear_loss and phase866_pair_rule)
    target_output_clean_transition = bool(clear_gain and not clear_loss)
    source_clean = bool(row.get("source_predict_clean_mixed"))
    transfer_status = (
        "stable_clean"
        if source_clean and target_clean_transition
        else "source_clean_failed"
        if source_clean and not target_clean_transition
        else "emergent_clean"
        if (not source_clean) and target_clean_transition
        else "stable_nonclean"
    )
    return {
        "model": row.get("model"),
        "domain": row.get("domain"),
        "case_id": row.get("case_id"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "candidate_key": row.get("candidate_key"),
        "edit_mode": row.get("edit_mode"),
        "source_purity_class": row.get("source_purity_class"),
        "source_predict_clean_mixed": source_clean,
        "base_rollout_label": base.get("rollout_label"),
        "intervened_rollout_label": row.get("rollout_label"),
        "clear_gain": clear_gain,
        "clear_loss": clear_loss,
        "target_output_clean_transition": target_output_clean_transition,
        "target_clean_transition": target_clean_transition,
        "transfer_status": transfer_status,
        "answer_delta": answer_delta,
        "blocker_reduction": blocker_reduction,
        "clear_blocker_reduction": clear_blocker_reduction,
        "original_blocker_delta": original_blocker_delta,
        "object_delta": object_delta,
        "object_echo_induced": object_echo_induced,
        "format_or_other_induced": format_or_other_induced,
        "gear_effect_basic": gear_effect_basic,
        "reducible_original_blockers": reducible_original_blockers,
        "no_side_effect": no_side_effect,
        "phase866_pair_rule": phase866_pair_rule,
        "field_plus_effect_rule": field_plus_effect_rule,
        "field_strict_admissible": field_strict_admissible,
        "field_strict_plus_effect_rule": field_strict_plus_effect_rule,
        **field,
    }


def binary_stats(rows: list[dict[str, Any]], pred_key: str, target_key: str) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get(pred_key) and row.get(target_key))
    fp = sum(1 for row in rows if row.get(pred_key) and not row.get(target_key))
    fn = sum(1 for row in rows if not row.get(pred_key) and row.get(target_key))
    tn = sum(1 for row in rows if not row.get(pred_key) and not row.get(target_key))
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / n if n else 0.0,
    }


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(row.get("transfer_status")) for row in rows)
    by_field_tags = Counter(tag for row in rows for tag in row.get("field_tags") or [])
    by_model_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_domain_field: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = f"{row.get('model')}:{row.get('domain')}"
        by_model_domain[key][str(row.get("transfer_status"))] += 1
        for tag in row.get("field_tags") or []:
            by_model_domain_field[key][tag] += 1
    return {
        "n_rows": len(rows),
        "transfer_status_counts": dict(by_status),
        "field_tag_counts": dict(by_field_tags),
        "model_domain_transfer_status_counts": {key: dict(value) for key, value in sorted(by_model_domain.items())},
        "model_domain_field_tag_counts": {key: dict(value) for key, value in sorted(by_model_domain_field.items())},
        "mean_field_class_blocker_count_by_status": {
            status: mean([finite(row.get("field_class_blocker_count")) for row in rows if row.get("transfer_status") == status])
            for status in sorted(by_status)
        },
        "mean_field_margin_by_status": {
            status: mean([finite(row.get("field_class_minus_object_logit")) for row in rows if row.get("transfer_status") == status])
            for status in sorted(by_status)
        },
    }


def load_pair_rows(
    source_round: str,
    source_root: Path,
    file_prefix: str,
    blocker_threshold: float,
    format_threshold: float,
    object_delta_threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_name in MODELS:
        rows = read_jsonl(source_root / source_round / f"{file_prefix}_{model_name}_rows.jsonl")
        originals = {row_context_key(row): row for row in rows if row.get("condition_type") == "original"}
        for row in rows:
            if row.get("condition_type") != "full_set":
                continue
            base = originals.get(row_context_key(row))
            if base is None:
                continue
            out.append(pair_row(base, row, blocker_threshold, format_threshold, object_delta_threshold))
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 870 Blocker Field Admissibility Rule ({payload['source_round']})",
        "",
        "- Source: Phase 867 paired original/intervened rows.",
        "- Boundary: single-context rule audit, not model training and not closure.",
        "",
        "## Rule Results",
        "",
        "| rule | target | n | TP | FP | FN | TN | precision | recall | accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("rule_results") or []:
        lines.append(
            f"| `{row['rule']}` | `{row['target']}` | {row['n']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['accuracy']:.3f} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Transfer status counts: `{payload['summary']['transfer_status_counts']}`",
        f"- Field tag counts: `{payload['summary']['field_tag_counts']}`",
        "",
        "## Pair Rows",
        "",
        "| model | domain | object | prompt | mode | status | field tags | field ok | phase866 | field+effect | target clean | clear gain/loss | ans | block red. | orig block |",
        "|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | {row.get('object')} | `{row.get('prompt_variant')}` | "
            f"`{row.get('edit_mode')}` | `{row.get('transfer_status')}` | `{row.get('field_tags')}` | "
            f"{row.get('field_base_admissible')} | {row.get('phase866_pair_rule')} | {row.get('field_plus_effect_rule')} | "
            f"{row.get('target_clean_transition')} | {int(bool(row.get('clear_gain')))}/{int(bool(row.get('clear_loss')))} | "
            f"{finite(row.get('answer_delta')):.3f} | {finite(row.get('blocker_reduction')):.3f} | "
            f"{finite(row.get('original_blocker_delta')):.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="holdout")
    parser.add_argument("--source-root", default=str(PHASE867_ROOT))
    parser.add_argument("--file-prefix", default="phase867")
    parser.add_argument("--output-round")
    parser.add_argument("--blocker-threshold", type=float, default=20.0)
    parser.add_argument("--format-threshold", type=float, default=2.0)
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_pair_rows(
        args.source_round,
        Path(args.source_root),
        str(args.file_prefix),
        float(args.blocker_threshold),
        float(args.format_threshold),
        float(args.object_delta_threshold),
    )
    rule_results = []
    for rule in (
        "source_predict_clean_mixed",
        "phase866_pair_rule",
        "field_base_admissible",
        "field_plus_effect_rule",
        "field_strict_admissible",
        "field_strict_plus_effect_rule",
    ):
        rule_results.append({"rule": rule, "target": "target_clean_transition", **binary_stats(rows, rule, "target_clean_transition")})
        rule_results.append({"rule": rule, "target": "target_output_clean_transition", **binary_stats(rows, rule, "target_output_clean_transition")})
    payload = {
        "phase": PHASE,
        "title": "Blocker Field Admissibility Rule",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_round": args.source_round,
        "source": str(Path(args.source_root) / args.source_round),
        "file_prefix": str(args.file_prefix),
        "blocker_threshold": float(args.blocker_threshold),
        "format_threshold": float(args.format_threshold),
        "object_delta_threshold": float(args.object_delta_threshold),
        "summary": grouped_summary(rows),
        "rule_results": rule_results,
        "rows": rows,
        "boundary": "Single-context field admissibility rule audit; no new model run and no closure claim.",
    }
    out_dir = RESULT_ROOT / (args.output_round or args.source_round)
    p846.write_json(out_dir / "phase870_summary.json", payload)
    p846.write_jsonl(out_dir / "phase870_pair_rows.jsonl", rows)
    write_markdown(out_dir / "phase870_summary.md", payload)
    print(json.dumps({"summary": payload["summary"], "rule_results": rule_results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
