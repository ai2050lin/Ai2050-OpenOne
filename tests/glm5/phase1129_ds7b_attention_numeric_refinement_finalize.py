#!/usr/bin/env python3
"""Finalize the one-shot Phase1129 numerical root classification."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1129_ds7b_attention_numeric_refinement_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    scan = protocol.read_json(protocol.OUT_ROOT / "scan" / protocol.MODEL / "summary.json")
    cases = protocol.read_jsonl(protocol.OUT_ROOT / "scan" / protocol.MODEL / "cases.jsonl")
    events = protocol.read_jsonl(protocol.OUT_ROOT / "scan" / protocol.MODEL / "events.jsonl")
    if protocol.digest(cases) != scan["case_detail_digest"] or protocol.digest(events) != scan["event_digest"]:
        raise RuntimeError("Phase1129 scan digest mismatch")

    source_bad = [row for row in cases if not row["source_total_finite"]]
    source_good = [row for row in cases if row["source_total_finite"]]
    root_bad = Counter(row["root_event_class"] for row in source_bad if row["root_event_class"])
    root_good = Counter(row["root_event_class"] for row in source_good if row["root_event_class"])
    event_metrics: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["event_name"]].append(event)
    source_bad_ids = {int(row["case_index"]) for row in source_bad}
    source_good_ids = {int(row["case_index"]) for row in source_good}
    for event in protocol.EVENT_REGISTRY:
        rows = grouped[event["name"]]
        bad_rows = [row for row in rows if int(row["case_index"]) in source_bad_ids]
        good_rows = [row for row in rows if int(row["case_index"]) in source_good_ids]
        event_metrics[event["name"]] = {
            "event_class": event["event_class"],
            "source_bad_invalid_cases": sum(int(row["root_invalid_count"]) > 0 for row in bad_rows),
            "source_good_invalid_cases": sum(int(row["root_invalid_count"]) > 0 for row in good_rows),
            "source_bad_nan_cases": sum(int(row["nan_count"]) > 0 for row in bad_rows),
            "source_bad_posinf_cases": sum(int(row["posinf_count"]) > 0 for row in bad_rows),
            "source_bad_neginf_cases": sum(int(row["neginf_count"]) > 0 for row in bad_rows),
        }

    exact_parity = all(
        row["candidate_finite_parity"] and row["suffix_finite_parity"] and row["total_finite_parity"]
        for row in cases
    )
    root_complete = all(row["root_event_class"] is not None for row in source_bad)
    predictions = {
        "P1_identity_implementation_and_events": protocol_audit["passed"],
        "P2_exact_phase1128_finite_parity": exact_parity,
        "P3_all_source_nonfinite_cases_classified": root_complete,
        "P4_root_was_not_assumed": True,
        "P5_one_shot_refinement_closed": True,
    }
    final_core = {
        "schema_version": "phase1129_ds7b_attention_numeric_refinement_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "scan_summary_digest": scan["summary_digest"],
        "source_nonfinite_count": len(source_bad),
        "source_finite_count": len(source_good),
        "root_counts_source_nonfinite": dict(root_bad.most_common()),
        "root_counts_source_finite": dict(root_good.most_common()),
        "event_metrics": event_metrics,
        "predictions": predictions,
        "auto_continue": {
            "value": False,
            "reason": "The Phase1128 authorization allowed exactly one numerical refinement; it is now closed regardless of result.",
        },
        "interpretation_boundary": (
            "A root class identifies the first recorded FP16 numerical failure in this exact implementation. It does "
            "not identify language content, semantic computation, behavioral necessity, or a generally preferred precision."
        ),
    }
    final = dict(final_core)
    final["final_digest"] = protocol.digest(final_core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
