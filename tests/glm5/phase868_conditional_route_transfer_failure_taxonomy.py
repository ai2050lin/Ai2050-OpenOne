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


PHASE = 868
RESULT_ROOT = Path("tests/result/phase868_conditional_route_transfer_failure_taxonomy")
PHASE867_ROOT = Path("tests/result/phase867_clean_route_holdout_prediction")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(row: dict[str, Any], object_delta_threshold: float) -> dict[str, Any]:
    source_clean = bool(row.get("source_predict_clean_mixed"))
    holdout_clean = bool(row.get("holdout_clean_mixed"))
    if source_clean and holdout_clean:
        status = "stable_clean"
    elif source_clean and not holdout_clean:
        status = "source_clean_failed"
    elif (not source_clean) and holdout_clean:
        status = "emergent_clean"
    else:
        status = "stable_nonclean"

    reasons: list[str] = []
    if int(row.get("clear_rollout_gain") or 0) <= 0:
        reasons.append("no_clear_gain")
    if int(row.get("clear_rollout_loss") or 0) > 0:
        reasons.append("clear_loss")
    if finite(row.get("mean_answer_delta")) <= 0:
        reasons.append("answer_not_lifted")
    if finite(row.get("mean_class_blocker_reduction")) <= 0:
        reasons.append("blocker_not_reduced")
    if finite(row.get("mean_original_blocker_delta")) >= 0:
        reasons.append("original_blocker_not_negative")
    if finite(row.get("mean_object_delta")) > float(object_delta_threshold) or int(row.get("object_echo_induced") or 0) > 0:
        reasons.append("object_side_effect")
    if int(row.get("format_or_other_induced") or 0) > 0:
        reasons.append("format_or_other_side_effect")
    if not reasons and not holdout_clean:
        reasons.append("unresolved_no_formula_failure")
    return {**row, "transfer_status": status, "failure_reasons": reasons}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(row.get("transfer_status")) for row in rows)
    by_reason = Counter(reason for row in rows for reason in row.get("failure_reasons") or [])
    by_model_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_model_domain_reason: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = f"{row.get('model')}:{row.get('domain')}"
        by_model_domain[key][str(row.get("transfer_status"))] += 1
        for reason in row.get("failure_reasons") or []:
            by_model_domain_reason[key][reason] += 1
    return {
        "n_rows": len(rows),
        "status_counts": dict(by_status),
        "failure_reason_counts": dict(by_reason),
        "model_domain_status_counts": {key: dict(value) for key, value in sorted(by_model_domain.items())},
        "model_domain_failure_reason_counts": {key: dict(value) for key, value in sorted(by_model_domain_reason.items())},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 868 Conditional Route Transfer Failure Taxonomy ({payload['source_round']})",
        "",
        "- Source: Phase 867 holdout effects.",
        "- Boundary: offline taxonomy only, no new model run and no closure claim.",
        "",
        "## Summary",
        "",
        f"- Status counts: `{payload['summary']['status_counts']}`",
        f"- Failure reason counts: `{payload['summary']['failure_reason_counts']}`",
        "",
        "## Rows",
        "",
        "| model | domain | mode | source purity | transfer status | clear gain/loss | ans delta | blocker red. | orig blocker delta | object delta | reasons |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | `{row.get('edit_mode')}` | "
            f"`{row.get('source_purity_class')}` | `{row.get('transfer_status')}` | "
            f"{row.get('clear_rollout_gain', 0)}/{row.get('clear_rollout_loss', 0)} | "
            f"{finite(row.get('mean_answer_delta')):.4f} | "
            f"{finite(row.get('mean_class_blocker_reduction')):.4f} | "
            f"{finite(row.get('mean_original_blocker_delta')):.4f} | "
            f"{finite(row.get('mean_object_delta')):.4f} | "
            f"`{row.get('failure_reasons')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", default="holdout")
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    args = parser.parse_args()

    source = PHASE867_ROOT / args.source_round / "phase867_cross_model_summary.json"
    if not source.exists():
        raise FileNotFoundError(f"missing Phase 867 summary: {source}")
    phase867 = read_json(source)
    rows = []
    for summary in (phase867.get("model_summaries") or {}).values():
        for row in summary.get("holdout_effects") or []:
            rows.append(classify(row, float(args.object_delta_threshold)))
    payload = {
        "phase": PHASE,
        "title": "Conditional Route Transfer Failure Taxonomy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "source_round": args.source_round,
        "object_delta_threshold": float(args.object_delta_threshold),
        "summary": summarize(rows),
        "rows": rows,
        "boundary": "This phase taxonomizes holdout transfer failures; it does not run models.",
    }
    out_dir = RESULT_ROOT / args.source_round
    p846.write_json(out_dir / "phase868_summary.json", payload)
    p846.write_jsonl(out_dir / "phase868_rows.jsonl", rows)
    write_markdown(out_dir / "phase868_summary.md", payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
