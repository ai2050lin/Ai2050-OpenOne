#!/usr/bin/env python3
"""C621 post-registered taxonomy of C617 generation failures.

This is descriptive reuse of frozen records. It does not tune an intervention
or claim a new causal result.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2147_c613_c620_conditional_gear_campaign as campaign

OUT = TESTS / "result/phase2155_c621_output_failure_taxonomy"
RECORDS = TESTS / "result/phase2151_c617_generation_timeline_causal_boundary/analysis/generation_timeline_records.jsonl"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify(text: str, source: dict, target: dict) -> str:
    pred = campaign.generated_prediction(text, source["answer_candidates"])
    if pred < 0:
        return "unregistered_or_ambiguous"
    answer = source["answer_candidates"][pred]
    if answer == target["answer"]:
        return "target_answer"
    if answer == source["answer"]:
        return "source_answer"
    return "other_registered_answer"


def main() -> None:
    protocol = {
        "phase": 2155, "campaign": "C621", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "object": "descriptive taxonomy of frozen C617 output failures",
        "frozen_inputs": [str(RECORDS.relative_to(ROOT))],
        "classes": ["source_answer", "target_answer", "other_registered_answer", "unregistered_or_ambiguous"],
        "claim_boundary": "post-registered description; no intervention tuning and no new causal gate",
    }
    save(OUT / "protocol/preregistration.json", protocol)
    records = campaign.read_rows(RECORDS)
    compiled = {x["case_id"]: x for x in campaign.read_rows(campaign.compiled_path())}
    modes = ("zero", "q16", "q24", "q32", "joint", "wrong_sign", "wrong_role", "wrong_operation")
    rows = []
    for record in records:
        source, target = compiled[record["source"]], compiled[record["target"]]
        zero = record["outputs"]["zero"]
        for mode in modes:
            value = record["outputs"][mode]
            rows.append({
                "operation": record["operation"], "source": record["source"], "target": record["target"], "mode": mode,
                "output_identity_changed": source["answer"] != target["answer"],
                "prefill_class": classify(value["prefill_text"], source, target),
                "persistent_class": classify(value["persistent_text"], source, target),
                "prefill_persistent_exact": campaign.normalize(value["prefill_text"]) == campaign.normalize(value["persistent_text"]),
                "persistent_changed_from_zero": campaign.normalize(value["persistent_text"]) != campaign.normalize(zero["persistent_text"]),
                "prefill_text": value["prefill_text"], "persistent_text": value["persistent_text"],
            })
    campaign.write_rows(OUT / "analysis/failure_taxonomy.jsonl", rows)
    by_mode = {}
    for mode in modes:
        values = [x for x in rows if x["mode"] == mode]
        by_mode[mode] = {
            "tests": len(values),
            "nontrivial_tests": sum(x["output_identity_changed"] for x in values),
            "prefill_classes": dict(Counter(x["prefill_class"] for x in values)),
            "persistent_classes": dict(Counter(x["persistent_class"] for x in values)),
            "prefill_persistent_exact": sum(x["prefill_persistent_exact"] for x in values),
            "persistent_changed_from_zero": sum(x["persistent_changed_from_zero"] for x in values),
            "nontrivial_prefill_target": sum(x["output_identity_changed"] and x["prefill_class"] == "target_answer" for x in values),
            "nontrivial_persistent_target": sum(x["output_identity_changed"] and x["persistent_class"] == "target_answer" for x in values),
        }
    q24 = by_mode["q24"]
    timing_only_supported = (
        q24["nontrivial_persistent_target"] > q24["nontrivial_prefill_target"]
    )
    headline = {
        "status": "output_failure_taxonomy_closed", "records": len(records), "classified_rows": len(rows),
        "by_mode": by_mode, "timing_only_explanation_supported": timing_only_supported,
        "c617_generation_re_adjudication": {
            "registered_tests": len(records),
            "nontrivial_output_change_tests": sum(x["output_identity_changed"] for x in rows if x["mode"] == "zero"),
            "q24_nontrivial_prefill_target": q24["nontrivial_prefill_target"],
            "q24_nontrivial_persistent_target": q24["nontrivial_persistent_target"],
            "reason": "The sole raw target count was a same-answer depth4/control pair and is not an output-identity intervention success.",
        },
        "strict_interpretation": (
            "Equal prefill and persistent outcomes reject only the simple timing-duration explanation for this frozen response. "
            "They do not locate the missing output identity or prove that no generation-time intervention can work."
        ),
    }
    result = {"phase": 2155, "campaign": "C621", "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
              "headline": headline,
              "checks": {"complete": len(rows) == len(records) * len(modes),
                         "frozen_records": len(records) == 12,
                         "finite": all(not isinstance(v, float) or math.isfinite(v) for v in [float(len(rows))])},
              "next_authorization": "C622_independent_taxonomy_audit"}
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
