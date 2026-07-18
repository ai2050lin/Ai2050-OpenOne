#!/usr/bin/env python3
"""Phase486 read-only audit of Phase485 other outputs.

No model run. Classifies strict-other generations into recoverable and
non-recoverable event types before any prompt rewrite or geometry collection.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "tests" / "gpt5" / "result" / "phase485_core_surface_behavior_gate" / "phase485_core_surface_behavior_generations.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase486_readonly_other_event_audit"
ROWS_PATH = OUT_DIR / "phase486_readonly_other_event_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase486_readonly_other_event_summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def expected_truth_word(row: dict[str, Any]) -> str:
    return "true" if row["truth_value"] else "false"


def label_from_truth_word(word: str, mapping: str) -> str:
    truth = word.lower() == "true"
    if mapping == "mu_ab":
        return "A" if truth else "B"
    if mapping == "mu_ba":
        return "B" if truth else "A"
    raise ValueError(mapping)


def classify_event(row: dict[str, Any]) -> dict[str, Any]:
    text = row["generated_text"].strip()
    lowered = text.lower()
    labels = re.findall(r"\b[AB]\b", text.upper())
    truth_words = re.findall(r"\b(true|false)\b", lowered)
    yes_no = re.findall(r"\b(yes|no)\b", lowered)
    recovered = None
    event_type = "unrelated_or_content_continuation"
    reason = "no unique recoverable label"

    if len(set(labels)) == 1:
        recovered = labels[0]
        event_type = "single_label_with_wrapper"
        reason = "unique A/B label present"
    elif len(set(labels)) > 1:
        event_type = "multi_label_ambiguous"
        reason = "both A and B or multiple conflicting labels present"
    elif len(set(truth_words)) == 1:
        recovered = label_from_truth_word(truth_words[0], row["label_mapping"])
        event_type = "truth_word"
        reason = "unique true/false word present"
    elif len(set(truth_words)) > 1:
        event_type = "truth_word_ambiguous"
        reason = "both true and false present"
    elif len(set(yes_no)) == 1:
        # In these claim-validity prompts, yes maps to true, no maps to false.
        recovered = label_from_truth_word("true" if yes_no[0] == "yes" else "false", row["label_mapping"])
        event_type = "yes_no_word"
        reason = "unique yes/no word present"
    elif lowered.startswith(("to determine", "the claim", "step")):
        event_type = "explanation_prefix_truncated"
        reason = "generation starts an explanation but max_new_tokens ends early"
    elif re.match(r"^[a-z]\d{5}", lowered):
        event_type = "content_continuation"
        reason = "generation continues entity/content text"

    recoverable = recovered in {"A", "B"}
    semantic_recovered = recoverable and recovered == row["expected_label"]
    return {
        "event_type": event_type,
        "reason": reason,
        "recoverable_label": recovered,
        "recoverable": recoverable,
        "semantic_recovered": semantic_recovered,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for fields in (
            ("strict_classification",),
            ("variant_class",),
            ("variant_track",),
            ("variant_class", "event_type"),
            ("variant_track", "event_type"),
            ("truth_value", "event_type"),
            ("label_mapping", "event_type"),
        ):
            groups[(fields, tuple(row[field] for field in fields))].append(row)
    reports = []
    for (fields, key), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        report = {field: value for field, value in zip(fields, key, strict=True)}
        n = len(items)
        report.update({
            "n": n,
            "strict_semantic": sum(1 for row in items if row["strict_classification"] == "semantic"),
            "strict_wrong": sum(1 for row in items if row["strict_classification"] == "wrong"),
            "strict_other": sum(1 for row in items if row["strict_classification"] == "other"),
            "recoverable": sum(1 for row in items if row["recoverable"]),
            "semantic_recovered": sum(1 for row in items if row["semantic_recovered"]),
        })
        report["semantic_recovered_rate"] = report["semantic_recovered"] / n if n else 0.0
        reports.append(report)

    core = [row for row in rows if row["variant_class"] in {"identity", "core_surface"}]
    strict_core_sem = sum(1 for row in core if row["strict_classification"] == "semantic")
    recovered_core_sem = sum(1 for row in core if row["strict_classification"] == "semantic" or (row["strict_classification"] == "other" and row["semantic_recovered"]))
    core_wrong = sum(1 for row in core if row["strict_classification"] == "wrong")
    core_unrecoverable = len(core) - recovered_core_sem - core_wrong
    return {
        "schema_version": "phase486_readonly_other_event_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "readonly_other_event_audit_complete",
        "input": str(IN_PATH.relative_to(ROOT)),
        "total_rows": len(rows),
        "strict_counts": dict(Counter(row["strict_classification"] for row in rows)),
        "event_type_counts_for_strict_other": dict(Counter(row["event_type"] for row in rows if row["strict_classification"] == "other")),
        "group_reports": reports,
        "core_semantic_recovery": {
            "n": len(core),
            "strict_semantic": strict_core_sem,
            "strict_accuracy": strict_core_sem / len(core) if core else 0.0,
            "strict_wrong": core_wrong,
            "semantic_after_recovery": recovered_core_sem,
            "semantic_after_recovery_rate": recovered_core_sem / len(core) if core else 0.0,
            "unrecoverable_or_ambiguous": core_unrecoverable,
            "unrecoverable_or_ambiguous_rate": core_unrecoverable / len(core) if core else 0.0,
        },
        "interpretation": {
            "allowed_claim": "Phase485 failure can be decomposed into strict wrong labels versus recoverable/nonrecoverable output events.",
            "forbidden_claim": "No prompt rewrite, physical geometry collection, or semantic mechanism claim is authorized by this audit alone.",
            "next_step": "If recoverable wrappers dominate, freeze a dual strict/recovery scorer before a new behavior gate.",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in load_jsonl(IN_PATH):
        item = dict(row)
        item["strict_classification"] = item.pop("classification")
        item.update(classify_event(row))
        rows.append(item)
    write_jsonl(ROWS_PATH, rows)
    SUMMARY_PATH.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(ROWS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
