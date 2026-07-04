#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 905
RESULT_ROOT = Path("tests/result/phase905_stop_action_boundary_audit")
PHASE904_ROOT = Path("tests/result/phase904_termination_control_candidate_search")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cat_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def token_contains_special(row: dict[str, Any]) -> bool:
    text = f"{row.get('first_suffix_token') or ''} {row.get('second_suffix_token') or ''} {row.get('suffix_text') or ''}"
    lowered = text.lower()
    return any(marker in lowered for marker in ["<|endoftext|>", "<eos>", "</s>", "<|eot_id|>"])


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop_top1 = [row for row in rows if row.get("stop_top1")]
    period_top1 = [row for row in stop_top1 if row.get("stop_best_category") == "period"]
    eos_top1 = [row for row in stop_top1 if row.get("stop_best_category") == "eos"]
    period_first = [row for row in stop_top1 if row.get("first_suffix_category") == "period"]
    eos_first = [row for row in stop_top1 if row.get("first_suffix_category") == "eos"]
    period_then_cont = [
        row
        for row in stop_top1
        if row.get("first_suffix_category") == "period" and bool(row.get("second_suffix_token"))
    ]
    return {
        "rows": len(rows),
        "answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "strict_clean_answer_no_protocol": sum(1 for row in rows if row.get("strict_clean_answer_no_protocol")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "strict_protocol_drift": sum(1 for row in rows if row.get("strict_protocol_drift")),
        "stop_top1": len(stop_top1),
        "stop_top1_strict_clean": sum(1 for row in stop_top1 if row.get("strict_clean_answer_no_protocol")),
        "stop_top1_protocol_drift": sum(1 for row in stop_top1 if row.get("protocol_drift")),
        "stop_top1_strict_protocol_drift": sum(1 for row in stop_top1 if row.get("strict_protocol_drift")),
        "stop_top1_period_best": len(period_top1),
        "stop_top1_eos_best": len(eos_top1),
        "stop_top1_period_first_suffix": len(period_first),
        "stop_top1_eos_first_suffix": len(eos_first),
        "stop_top1_period_then_continuation": len(period_then_cont),
        "stop_top1_decoded_special_marker": sum(1 for row in stop_top1 if token_contains_special(row)),
        "stop_best_categories": cat_counter(stop_top1, "stop_best_category"),
        "first_suffix_categories_when_stop_top1": cat_counter(stop_top1, "first_suffix_category"),
        "second_suffix_categories_when_stop_top1": cat_counter(stop_top1, "second_suffix_category"),
        "next_top_categories_when_stop_top1": cat_counter(stop_top1, "next_top_category"),
    }


def sample_rows(rows: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        if not row.get("stop_top1"):
            continue
        samples.append(
            {
                "model": row.get("model"),
                "control_label": row.get("control_label"),
                "case_id": row.get("case_id"),
                "object": row.get("object"),
                "stop_best_category": row.get("stop_best_category"),
                "first_suffix_token": row.get("first_suffix_token"),
                "first_suffix_category": row.get("first_suffix_category"),
                "second_suffix_token": row.get("second_suffix_token"),
                "second_suffix_category": row.get("second_suffix_category"),
                "suffix_text": row.get("suffix_text"),
                "combined_text": row.get("combined_text"),
                "protocol_drift": row.get("protocol_drift"),
                "strict_protocol_drift": row.get("strict_protocol_drift"),
                "strict_clean_answer_no_protocol": row.get("strict_clean_answer_no_protocol"),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def evidence_label(summary: dict[str, Any]) -> str:
    non_base = summary["non_baseline"]
    if non_base["stop_top1"] == 0:
        return "no_stop_top1_boundary_to_audit"
    if non_base["stop_top1_eos_best"] > 0 or non_base["stop_top1_eos_first_suffix"] > 0:
        if non_base["stop_top1_strict_clean"] > 0:
            return "eos_stop_boundary_partially_clean"
        return "eos_or_special_seen_but_not_clean"
    if non_base["stop_top1_period_best"] == non_base["stop_top1"]:
        return "stop_top1_is_period_not_termination_action"
    return "mixed_stop_boundary_without_clean_action"


def build_summary(phase904_round: str, round_name: str) -> dict[str, Any]:
    in_dir = PHASE904_ROOT / phase904_round
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(in_dir.glob("phase904_*_rows.jsonl")):
        if "summary" in path.name:
            continue
        for row in read_jsonl(path):
            model_rows[str(row.get("model"))].append(row)

    model_summaries = []
    all_rows: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    for model_name, rows in sorted(model_rows.items()):
        all_rows.extend(rows)
        baseline_rows = [row for row in rows if row.get("control_type") == "baseline"]
        non_base_rows = [row for row in rows if row.get("control_type") != "baseline"]
        summary = {
            "model": model_name,
            "baseline": summarize_rows(baseline_rows),
            "non_baseline": summarize_rows(non_base_rows),
            "stop_top1_samples": sample_rows(non_base_rows),
        }
        summary["evidence_label"] = evidence_label(summary)
        model_summaries.append(summary)
        all_samples.extend(summary["stop_top1_samples"])

    baseline_all = [row for row in all_rows if row.get("control_type") == "baseline"]
    non_base_all = [row for row in all_rows if row.get("control_type") != "baseline"]
    payload = {
        "phase": PHASE,
        "title": "Stop Action vs Stop Token Boundary Audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase904_round": phase904_round,
        "round_name": round_name,
        "models": [row["model"] for row in model_summaries],
        "overall": {
            "baseline": summarize_rows(baseline_all),
            "non_baseline": summarize_rows(non_base_all),
        },
        "model_summaries": model_summaries,
        "stop_top1_samples": all_samples[:24],
    }
    payload["evidence_label"] = evidence_label({"non_baseline": payload["overall"]["non_baseline"]})
    write_json(out_dir / "phase905_stop_action_boundary_summary.json", payload)
    write_markdown(out_dir / "phase905_stop_action_boundary_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 905 stop action boundary audit",
        "",
        "## Overall",
        "",
        f"- phase904_round: {payload.get('phase904_round')}",
        f"- evidence: {payload.get('evidence_label')}",
    ]
    overall = payload["overall"]["non_baseline"]
    for key in [
        "rows",
        "strict_clean_answer_no_protocol",
        "protocol_drift",
        "strict_protocol_drift",
        "stop_top1",
        "stop_top1_strict_clean",
        "stop_top1_protocol_drift",
        "stop_top1_period_best",
        "stop_top1_eos_best",
        "stop_top1_period_first_suffix",
        "stop_top1_eos_first_suffix",
        "stop_top1_period_then_continuation",
        "stop_top1_decoded_special_marker",
    ]:
        lines.append(f"- {key}: {overall.get(key)}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | stop_top1 | strict clean | drift | period best | eos best | period first | eos first | period then continuation | special marker | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        nb = summary["non_baseline"]
        lines.append(
            "| {model} | {stop_top1} | {strict_clean} | {drift} | {period_best} | {eos_best} | {period_first} | {eos_first} | {period_cont} | {special} | {evidence} |".format(
                model=summary["model"],
                stop_top1=nb["stop_top1"],
                strict_clean=nb["stop_top1_strict_clean"],
                drift=nb["stop_top1_protocol_drift"],
                period_best=nb["stop_top1_period_best"],
                eos_best=nb["stop_top1_eos_best"],
                period_first=nb["stop_top1_period_first_suffix"],
                eos_first=nb["stop_top1_eos_first_suffix"],
                period_cont=nb["stop_top1_period_then_continuation"],
                special=nb["stop_top1_decoded_special_marker"],
                evidence=summary["evidence_label"],
            )
        )
    lines.extend(["", "## Stop Top1 Samples", ""])
    lines.append("| model | control | case | first | second | suffix |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload.get("stop_top1_samples") or []:
        suffix = str(row.get("suffix_text") or "").replace("\n", "\\n")
        first = str(row.get("first_suffix_token") or "").replace("\n", "\\n")
        second = str(row.get("second_suffix_token") or "").replace("\n", "\\n")
        lines.append(
            "| {model} | {control} | {case_id} | `{first}` | `{second}` | `{suffix}` |".format(
                model=row.get("model"),
                control=row.get("control_label"),
                case_id=row.get("case_id"),
                first=first,
                second=second,
                suffix=suffix,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase904-round", default="termination_control_candidate_search")
    parser.add_argument("--round-name", default="stop_action_boundary_audit")
    args = parser.parse_args()
    payload = build_summary(args.phase904_round, args.round_name)
    print(json.dumps({"phase": PHASE, "status": "complete", "overall": payload["overall"]["non_baseline"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
