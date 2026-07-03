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


PHASE = 869
RESULT_ROOT = Path("tests/result/phase869_blocker_field_conditional_gate_audit")
PHASE867_ROOT = Path("tests/result/phase867_clean_route_holdout_prediction")
PHASE868_ROOT = Path("tests/result/phase868_conditional_route_transfer_failure_taxonomy")
MODELS = p846.MODELS


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def summarize_original_field(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_top1 = Counter()
    role_top10 = Counter()
    top_tokens = Counter()
    for row in rows:
        role_top1[str(row.get("blocker_class_top_blocker_role") or "none")] += 1
        for blocker in row.get("blocker_class_top_blockers") or []:
            role_top10[str(blocker.get("role") or "none")] += 1
            top_tokens[str(blocker.get("token") or "")] += 1
    n = len(rows)
    role_top10_per_context = {
        role: count / n if n else 0.0 for role, count in sorted(role_top10.items(), key=lambda item: (-item[1], item[0]))
    }
    return {
        "n_contexts": n,
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows]),
        "mean_clear_class_blocker_count": mean([finite(row.get("clear_class_blocker_count")) for row in rows]),
        "mean_class_minus_object_logit": mean([finite(row.get("class_minus_object_logit")) for row in rows]),
        "mean_object_rank": mean([finite(row.get("object_rank")) for row in rows if row.get("object_rank") is not None]),
        "top1_role_counts": dict(role_top1),
        "top10_role_counts": dict(role_top10),
        "top10_role_per_context": role_top10_per_context,
        "top_tokens": dict(top_tokens.most_common(8)),
    }


def field_profile(summary: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    mean_blockers = finite(summary.get("mean_class_blocker_count"))
    margin = finite(summary.get("mean_class_minus_object_logit"))
    roles = summary.get("top10_role_per_context") or {}
    if mean_blockers >= 20:
        tags.append("high_blocker_count")
    if mean_blockers >= 60:
        tags.append("very_high_blocker_count")
    if margin < 0:
        tags.append("object_above_class")
    if finite(roles.get("format_space")) + finite(roles.get("format_punct")) >= 2.0:
        tags.append("format_pressure")
    if finite(roles.get("object_echo")) >= 1.0:
        tags.append("object_echo_pressure")
    if finite(roles.get("other_blocker")) >= 1.0:
        tags.append("semantic_other_pressure")
    if not tags:
        tags.append("low_or_mixed_pressure")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="holdout")
    args = parser.parse_args()

    phase868_path = PHASE868_ROOT / args.source_round / "phase868_rows.jsonl"
    transfer_rows = read_jsonl(phase868_path)
    transfer_by_model_domain: dict[str, Counter[str]] = defaultdict(Counter)
    reasons_by_model_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for row in transfer_rows:
        key = f"{row.get('model')}:{row.get('domain')}"
        transfer_by_model_domain[key][str(row.get("transfer_status"))] += 1
        for reason in row.get("failure_reasons") or []:
            reasons_by_model_domain[key][reason] += 1

    model_domain_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_name in MODELS:
        rows_path = PHASE867_ROOT / args.source_round / f"phase867_{model_name}_rows.jsonl"
        for row in read_jsonl(rows_path):
            if row.get("condition_type") != "original":
                continue
            key = f"{model_name}:{row.get('domain')}"
            model_domain_rows[key].append(row)

    summaries = []
    for key, rows in sorted(model_domain_rows.items()):
        base = summarize_original_field(rows)
        summaries.append(
            {
                "model_domain": key,
                **base,
                "field_profile": field_profile(base),
                "transfer_status_counts": dict(transfer_by_model_domain.get(key, Counter())),
                "transfer_failure_reasons": dict(reasons_by_model_domain.get(key, Counter())),
            }
        )

    payload = {
        "phase": PHASE,
        "title": "Blocker Field Conditional Gate Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_round": args.source_round,
        "source_phase867": str(PHASE867_ROOT / args.source_round),
        "source_phase868": str(phase868_path),
        "summaries": summaries,
        "boundary": "Offline blocker field description; no new intervention and no predictive closure.",
    }
    out_dir = RESULT_ROOT / args.source_round
    p846.write_json(out_dir / "phase869_summary.json", payload)
    p846.write_jsonl(out_dir / "phase869_model_domain_rows.jsonl", summaries)
    write_markdown(out_dir / "phase869_summary.md", payload)
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2), flush=True)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 869 Blocker Field Conditional Gate Audit ({payload['source_round']})",
        "",
        "- Source: Phase 867 original rows + Phase 868 transfer taxonomy.",
        "- Boundary: blocker field description, not closure.",
        "",
        "| model_domain | field profile | transfer status | failure reasons | mean blockers | margin class-object | object rank | top10 roles/context |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload.get("summaries") or []:
        lines.append(
            f"| `{row.get('model_domain')}` | `{row.get('field_profile')}` | "
            f"`{row.get('transfer_status_counts')}` | `{row.get('transfer_failure_reasons')}` | "
            f"{fmt(row.get('mean_class_blocker_count'))} | "
            f"{fmt(row.get('mean_class_minus_object_logit'))} | "
            f"{fmt(row.get('mean_object_rank'))} | "
            f"`{row.get('top10_role_per_context')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
