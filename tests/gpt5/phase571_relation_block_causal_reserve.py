#!/usr/bin/env python3
"""Freeze a larger baseline-only causal reserve after DS7B batch drift."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase571_relation_block_match_analysis as matching  # noqa: E402
import phase571_relation_block_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
RESERVE_CAP = 256
AMENDMENT_PATH = OUT_DIR / "phase571_causal_reserve_amendment.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_causal_reserve_summary.json"


def reserve_model(model: str) -> dict[str, Any]:
    rows = list(matching.iter_jsonl(matching.rows_path(model)))
    by_case_repeat = {
        (row["case_id"], row["execution_repeat"]): row for row in rows
    }
    base_cases = {
        row["case_id"]: row for row in rows if row["execution_repeat"] == "noop1"
    }
    eligible = {phenotype: [] for phenotype in protocol.PHENOTYPES}
    for phenotype in protocol.PHENOTYPES:
        for case in base_cases.values():
            if case["pool"] != "block_causal":
                continue
            first = by_case_repeat[(case["case_id"], "noop1")]
            second = by_case_repeat[(case["case_id"], "noop2")]
            if (
                matching.phenotype_matches(first, phenotype)
                and matching.phenotype_matches(second, phenotype)
                and first["semantic_event"] == second["semantic_event"]
            ):
                eligible[phenotype].append(first)
    correct, confusion = matching.matched_balanced(
        eligible["stable_correct"], eligible["stable_relation_confusion"], RESERVE_CAP
    )
    usable = (min(len(correct), len(confusion)) // 4) * 4
    correct = correct[:usable]
    confusion = confusion[:usable]
    if usable < 128:
        raise RuntimeError(f"Phase571 expanded causal reserve still too small: {model}/{usable}")
    left_distribution = matching.distribution(correct)
    right_distribution = matching.distribution(confusion)
    if left_distribution != right_distribution:
        raise RuntimeError(f"Phase571 expanded causal reserve lost matching: {model}")
    report = {
        "schema_version": "phase571_causal_reserve_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "reserve_cap": RESERVE_CAP,
        "usable_matched_pairs_per_phenotype": usable,
        "match_key": ["source_factorial_cell", "target", "other_relation_target"],
        "selected_case_ids_by_phenotype": {
            "stable_correct": [row["case_id"] for row in correct],
            "stable_relation_confusion": [row["case_id"] for row in confusion],
        },
        "matched_stratum_count": len(left_distribution),
        "target_other_pair_count": len({
            (row["target"], row["other_relation_target"]) for row in correct
        }),
        "matched_stratum_distributions_exactly_equal": True,
        "selection_uses_only_noop_behavior": True,
        "intervention_results_read": False,
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    write_json(summary_path(model), report)
    return report


def freeze() -> dict[str, Any]:
    reports = [reserve_model(model) for model in protocol.MODELS]
    amendment = {
        "schema_version": "phase571_causal_reserve_amendment.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "reason": (
            "DS7B retained 124/160 factor-and-value-pair-matched cases under the "
            "new fixed causal batch shape, below the preregistered final denominator of 128"
        ),
        "change": "expand baseline-only reserve from at most 160 to at most 256 pairs",
        "unchanged_final_paired_cases_per_phenotype": 128,
        "unchanged_block_candidates": True,
        "unchanged_interventions": True,
        "unchanged_thresholds": True,
        "intervention_results_read_to_define_change": False,
        "base_protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "model_reserves": [
            {
                "model": report["model"],
                "usable_pairs": report["usable_matched_pairs_per_phenotype"],
                "summary_sha256": sha256_file(summary_path(report["model"])),
            }
            for report in reports
        ],
        "sealed_split_read": False,
    }
    write_json(AMENDMENT_PATH, amendment)
    print(json.dumps({
        "reserve_cap": RESERVE_CAP,
        "final_denominator": 128,
        "models": amendment["model_reserves"],
    }, ensure_ascii=False, indent=2))
    return amendment


if __name__ == "__main__":
    freeze()
