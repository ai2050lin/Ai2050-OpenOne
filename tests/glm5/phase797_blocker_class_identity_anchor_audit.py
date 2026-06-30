#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


SOURCE_ROOT = Path("tests/result/phase796_global_competitor_token_identity_audit")
OUT_ROOT = Path("results/glm5_phase797_blocker_class_identity_anchor_audit")
RESULT_ROOT = Path("tests/result/phase797_blocker_class_identity_anchor_audit")

DEFAULT_CLASSES = [
    "designated_contrast",
    "candidate_list_or_case_value",
    "echo_token",
    "whitespace_or_newline",
    "punctuation",
    "high_frequency_or_format",
    "semantic_or_lexical_competitor",
]


def norm(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


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


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rows_above_target(topk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in topk or []:
        if row.get("is_target"):
            continue
        gap = row.get("gap_above_target")
        if gap is not None and float(gap) > 0:
            out.append(row)
    return out


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("class")) for r in rows))


def surface_variants_above_target(row: dict[str, Any], blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer = norm(row.get("target_answer"))
    target_id = int(row.get("target_token_id"))
    if not answer:
        return []
    variants = []
    for item in blockers:
        if int(item.get("token_id")) == target_id:
            continue
        if norm(item.get("token_text")) == answer:
            variants.append(item)
    return variants


def token_ids_for_classes(blockers: list[dict[str, Any]], selected_classes: set[str]) -> set[int]:
    return {int(r["token_id"]) for r in blockers if str(r.get("class")) in selected_classes}


def observed_closure_after_suppressing(blockers: list[dict[str, Any]], selected_classes: set[str]) -> bool:
    for item in blockers:
        if str(item.get("class")) not in selected_classes:
            return False
    return True


def remaining_after_suppression(blockers: list[dict[str, Any]], selected_classes: set[str]) -> dict[str, Any] | None:
    for item in blockers:
        if str(item.get("class")) not in selected_classes:
            return item
    return None


def minimal_class_set(blockers: list[dict[str, Any]], max_subset_size: int) -> tuple[list[str], bool]:
    classes = sorted({str(r.get("class")) for r in blockers})
    if not classes:
        return [], True
    limit = min(max_subset_size, len(classes))
    for size in range(1, limit + 1):
        for subset in itertools.combinations(classes, size):
            selected = set(subset)
            if observed_closure_after_suppressing(blockers, selected):
                return list(subset), True
    return [], False


def class_required_bias(blockers: list[dict[str, Any]], cls: str) -> float | None:
    vals = [float(r["gap_above_target"]) for r in blockers if str(r.get("class")) == cls and r.get("gap_above_target") is not None]
    return max(vals) if vals else None


def audit_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    after_topk = row.get("after_topk") or []
    blockers = rows_above_target(after_topk)
    target_rank = row.get("after_target_rank")
    top_k = len(after_topk)
    exact_topk_covers_all_above = bool(target_rank is not None and int(target_rank) <= top_k + 1)
    variants = surface_variants_above_target(row, blockers)
    min_classes, min_found = minimal_class_set(blockers, args.max_class_subset_size)
    top_blocker = blockers[0] if blockers else None
    class_rows = []
    for cls in args.blocker_classes:
        selected = {cls}
        remaining = remaining_after_suppression(blockers, selected)
        closes_observed_topk = observed_closure_after_suppressing(blockers, selected)
        class_rows.append(
            {
                "row_kind": "phase797_blocker_class_single_suppression",
                "model": row.get("model"),
                "round": row.get("round"),
                "case_id": row.get("case_id"),
                "domain": row.get("domain"),
                "relation": row.get("relation"),
                "object": row.get("object"),
                "target_answer": row.get("target_answer"),
                "ladder_id": row.get("ladder_id"),
                "source_group": row.get("source_group"),
                "source_selection_kind": row.get("source_selection_kind"),
                "subspace_mode": row.get("subspace_mode"),
                "budget_label": row.get("budget_label"),
                "source_set_size": row.get("source_set_size"),
                "blocker_class": cls,
                "blocker_count_above_target": sum(1 for b in blockers if str(b.get("class")) == cls),
                "required_bias_to_clear_class": class_required_bias(blockers, cls),
                "observed_topk_closes_if_suppressed": closes_observed_topk,
                "exact_closes_if_suppressed": bool(closes_observed_topk and exact_topk_covers_all_above),
                "remaining_top_blocker_class": remaining.get("class") if remaining else None,
                "remaining_top_blocker_gap": remaining.get("gap_above_target") if remaining else None,
                "exact_topk_covers_all_above": exact_topk_covers_all_above,
                "after_target_rank": target_rank,
                "after_global_margin": row.get("after_global_margin"),
                "target_logit_gain_vs_recipient": row.get("target_logit_gain_vs_recipient"),
                "delta_global_margin_vs_recipient": row.get("delta_global_margin_vs_recipient"),
                "recipient_top_competitor_suppressed": row.get("recipient_top_competitor_suppressed"),
                "token_closure_gain": row.get("token_closure_gain"),
            }
        )
    summary = {
        "row_kind": "phase797_blocker_identity_anchor_summary",
        "model": row.get("model"),
        "round": row.get("round"),
        "case_id": row.get("case_id"),
        "domain": row.get("domain"),
        "relation": row.get("relation"),
        "object": row.get("object"),
        "target_answer": row.get("target_answer"),
        "contrast_answer": row.get("contrast_answer"),
        "ladder_id": row.get("ladder_id"),
        "source_group": row.get("source_group"),
        "source_selection_kind": row.get("source_selection_kind"),
        "subspace_mode": row.get("subspace_mode"),
        "budget_label": row.get("budget_label"),
        "source_set_size": row.get("source_set_size"),
        "after_target_rank": target_rank,
        "after_global_margin": row.get("after_global_margin"),
        "delta_global_margin_vs_recipient": row.get("delta_global_margin_vs_recipient"),
        "target_logit_gain_vs_recipient": row.get("target_logit_gain_vs_recipient"),
        "token_closure_gain": row.get("token_closure_gain"),
        "exact_topk_covers_all_above": exact_topk_covers_all_above,
        "observed_blocker_count_above_target": len(blockers),
        "observed_blocker_class_counts": class_counts(blockers),
        "top_blocker_class": top_blocker.get("class") if top_blocker else None,
        "top_blocker_gap": top_blocker.get("gap_above_target") if top_blocker else None,
        "top_blocker_token_text": top_blocker.get("token_text") if top_blocker else None,
        "surface_target_variant_count_above": len(variants),
        "surface_target_variant_max_gap": safe_mean([max(float(v.get("gap_above_target")) for v in variants)]) if variants else None,
        "surface_target_variant_token_texts": [v.get("token_text") for v in variants[:8]],
        "identity_anchor_fragmented": bool(variants),
        "minimal_observed_class_set_to_clear_topk": min_classes,
        "minimal_observed_class_set_found": min_found,
        "exact_closure_by_minimal_class_set": bool(min_found and exact_topk_covers_all_above),
        "unobserved_blocker_risk": not exact_topk_covers_all_above,
        "interpretation_boundary": (
            "This is an oracle audit over Phase 796 top-k blockers, not a new neural intervention. "
            "Exact closure is only claimed when after_target_rank is within the saved top-k window."
        ),
    }
    return {"summary": summary, "class_rows": class_rows}


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_after_target_rank": safe_mean([v.get("after_target_rank") for v in vals]),
                "mean_after_global_margin": safe_mean([v.get("after_global_margin") for v in vals]),
                "mean_delta_global_margin_vs_recipient": safe_mean([v.get("delta_global_margin_vs_recipient") for v in vals]),
                "mean_target_logit_gain_vs_recipient": safe_mean([v.get("target_logit_gain_vs_recipient") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "exact_topk_cover_rate": safe_rate([v.get("exact_topk_covers_all_above") for v in vals]),
                "identity_anchor_fragmented_rate": safe_rate([v.get("identity_anchor_fragmented") for v in vals]),
                "exact_closure_by_minimal_class_set_rate": safe_rate([v.get("exact_closure_by_minimal_class_set") for v in vals]),
                "unobserved_blocker_risk_rate": safe_rate([v.get("unobserved_blocker_risk") for v in vals]),
                "mean_surface_target_variant_count_above": safe_mean([v.get("surface_target_variant_count_above") for v in vals]),
                "top_blocker_class_counts": dict(Counter(v.get("top_blocker_class") for v in vals)),
                "minimal_class_set_counts": dict(Counter("+".join(v.get("minimal_observed_class_set_to_clear_topk") or []) or "not_found" for v in vals)),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("identity_anchor_fragmented_rate") or 0.0,
            r.get("mean_delta_global_margin_vs_recipient") or 0.0,
        ),
        reverse=True,
    )
    return out


def group_class_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_blocker_count_above_target": safe_mean([v.get("blocker_count_above_target") for v in vals]),
                "mean_required_bias_to_clear_class": safe_mean([v.get("required_bias_to_clear_class") for v in vals]),
                "observed_topk_closure_rate_if_suppressed": safe_rate([v.get("observed_topk_closes_if_suppressed") for v in vals]),
                "exact_closure_rate_if_suppressed": safe_rate([v.get("exact_closes_if_suppressed") for v in vals]),
                "exact_topk_cover_rate": safe_rate([v.get("exact_topk_covers_all_above") for v in vals]),
                "mean_delta_global_margin_vs_recipient": safe_mean([v.get("delta_global_margin_vs_recipient") for v in vals]),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("observed_topk_closure_rate_if_suppressed") or 0.0,
            r.get("mean_blocker_count_above_target") or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize(summary_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": 797,
        "title": "Blocker-Class Targeted Suppression and Identity-Anchor Separation",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase": 796,
        "source_root": str(SOURCE_ROOT / args.round_name),
        "blocker_classes": args.blocker_classes,
        "max_class_subset_size": args.max_class_subset_size,
        "n_summary_rows": len(summary_rows),
        "n_class_rows": len(class_rows),
        "models": sorted({r.get("model") for r in summary_rows}),
        "by_model": group_rows(summary_rows, ["model"]),
        "by_model_ladder": group_rows(summary_rows, ["model", "ladder_id", "source_group"]),
        "by_model_top_blocker_class": group_rows(summary_rows, ["model", "top_blocker_class"]),
        "by_model_blocker_class_suppression": group_class_rows(class_rows, ["model", "blocker_class"]),
        "by_model_ladder_blocker_class_suppression": group_class_rows(class_rows, ["model", "ladder_id", "source_group", "blocker_class"]),
        "top_identity_fragmentation": group_rows(summary_rows, ["model", "ladder_id", "source_group"])[:60],
        "strict_boundary": (
            "This phase does not perform a new model forward pass. It audits whether Phase 796 top-k blockers are explainable by class-level suppression "
            "and whether target surface-form variants above the target token indicate identity-anchor fragmentation."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 797 Blocker-Class Targeted Suppression and Identity-Anchor Separation ({payload['round']})",
        "",
        "- Source: Phase 796 saved top-k rows.",
        "- Boundary: this is an oracle audit, not a new neural intervention.",
        "",
        "## By Model",
        "",
        "| model | rows | cases | target rank | global delta | exact top-k cover | identity fragmented | exact closure by class set | unobserved risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_model"]:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['case_n']} | {fmt(row['mean_after_target_rank'])} | "
            f"{fmt(row['mean_delta_global_margin_vs_recipient'])} | {fmt(row['exact_topk_cover_rate'])} | "
            f"{fmt(row['identity_anchor_fragmented_rate'])} | {fmt(row['exact_closure_by_minimal_class_set_rate'])} | "
            f"{fmt(row['unobserved_blocker_risk_rate'])} |"
        )
    lines += [
        "",
        "## Single Class Suppression Audit",
        "",
        "| model | blocker class | rows | mean count | required bias | observed top-k close | exact close |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_model_blocker_class_suppression"]:
        lines.append(
            f"| {row['model']} | `{row['blocker_class']}` | {row['n']} | "
            f"{fmt(row['mean_blocker_count_above_target'])} | {fmt(row['mean_required_bias_to_clear_class'])} | "
            f"{fmt(row['observed_topk_closure_rate_if_suppressed'])} | {fmt(row['exact_closure_rate_if_suppressed'])} |"
        )
    lines += [
        "",
        "## Identity Fragmentation Hotspots",
        "",
        "| model | ladder | source group | rows | fragmented | surface variants | global delta | minimal sets |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["top_identity_fragmentation"][:36]:
        lines.append(
            f"| {row['model']} | `{row['ladder_id']}` | `{row['source_group']}` | {row['n']} | "
            f"{fmt(row['identity_anchor_fragmented_rate'])} | {fmt(row['mean_surface_target_variant_count_above'])} | "
            f"{fmt(row['mean_delta_global_margin_vs_recipient'])} | `{row['minimal_class_set_counts']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return str(value)
    return f"{val:.4f}"


def run_round(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = SOURCE_ROOT / args.round_name
    if not source_dir.exists():
        raise SystemExit(f"missing source phase796 result dir: {source_dir}")
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for model in MODELS:
        path = source_dir / f"phase796_{model}_rows.jsonl"
        rows = load_jsonl(path)
        if args.max_rows_per_model > 0:
            rows = rows[: args.max_rows_per_model]
        for row in rows:
            audited = audit_row(row, args)
            summary_rows.append(audited["summary"])
            class_rows.extend(audited["class_rows"])
    payload = summarize(summary_rows, class_rows, args)
    for root in (OUT_ROOT / args.round_name, RESULT_ROOT / args.round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_jsonl(root / "phase797_summary_rows.jsonl", summary_rows)
        write_jsonl(root / "phase797_class_rows.jsonl", class_rows)
        write_json(root / "phase797_cross_model_summary.json", payload)
        write_markdown(root / "phase797_cross_model_summary.md", payload)
    print(
        json.dumps(
            {
                "round": args.round_name,
                "models": payload["models"],
                "n_summary_rows": payload["n_summary_rows"],
                "by_model": payload["by_model"],
                "by_model_blocker_class_suppression": payload["by_model_blocker_class_suppression"][:18],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    parser.add_argument("--blocker-classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--max-class-subset-size", type=int, default=4)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.blocker_classes = [x.strip() for x in str(args.blocker_classes).split(",") if x.strip()]
    run_round(args)


if __name__ == "__main__":
    main()
