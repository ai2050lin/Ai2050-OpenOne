#!/usr/bin/env python3
"""Create exactly factor/value-pair matched Phase571 phenotype denominators."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase571_relation_block_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
MODELS = protocol.MODELS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_behavior_rows.jsonl.gz"


def source_summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_behavior_summary.json"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_matched_behavior_summary.json"


def phenotype_matches(row: dict[str, Any], phenotype: str) -> bool:
    if phenotype == "stable_correct":
        return bool(row["semantic_correct"])
    if phenotype == "stable_relation_confusion":
        return bool(row["relation_confusion"])
    raise ValueError(phenotype)


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row["source_factorial_cell"], row["target"], row["other_relation_target"]
    )


def matched_balanced(
    correct: list[dict[str, Any]],
    confusion: list[dict[str, Any]],
    cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    banks: dict[str, dict[tuple[str, str, str], deque[dict[str, Any]]]] = {
        "stable_correct": defaultdict(deque),
        "stable_relation_confusion": defaultdict(deque),
    }
    for row in sorted(correct, key=lambda item: item["case_id"]):
        banks["stable_correct"][stratum(row)].append(row)
    for row in sorted(confusion, key=lambda item: item["case_id"]):
        banks["stable_relation_confusion"][stratum(row)].append(row)
    keys = sorted(set(banks["stable_correct"]) & set(banks["stable_relation_confusion"]))
    selected_correct: list[dict[str, Any]] = []
    selected_confusion: list[dict[str, Any]] = []
    while keys and len(selected_correct) < cap:
        remaining = []
        for key in keys:
            if len(selected_correct) >= cap:
                break
            left = banks["stable_correct"][key]
            right = banks["stable_relation_confusion"][key]
            if left and right:
                selected_correct.append(left.popleft())
                selected_confusion.append(right.popleft())
            if left and right:
                remaining.append(key)
        keys = remaining
    return selected_correct, selected_confusion


def distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts["|".join(stratum(row))] += 1
    return dict(sorted(counts.items()))


def analyze_model(model: str) -> dict[str, Any]:
    source_summary = read_json(source_summary_path(model))
    rows = list(iter_jsonl(rows_path(model)))
    if source_summary["rows_sha256"] != sha256_file(rows_path(model)):
        raise RuntimeError(f"Phase571 {model} behavior row hash drift")
    by_case_repeat = {
        (row["case_id"], row["execution_repeat"]): row for row in rows
    }
    base_cases = {
        row["case_id"]: row for row in rows
        if row["execution_repeat"] in ("baseline", "noop1")
    }
    selected_ids: dict[str, dict[str, list[str]]] = {}
    eligible_counts: dict[str, dict[str, int]] = {}
    matched_counts: dict[str, int] = {}
    pair_counts: dict[str, dict[str, int]] = {}
    stratum_counts: dict[str, int] = {}
    distributions_equal: dict[str, bool] = {}
    for pool in protocol.OPEN_POOLS:
        eligible_by_phenotype: dict[str, list[dict[str, Any]]] = {}
        eligible_counts[pool] = {}
        for phenotype in protocol.PHENOTYPES:
            eligible = []
            for case in base_cases.values():
                if case["pool"] != pool:
                    continue
                if pool == "block_causal":
                    first = by_case_repeat[(case["case_id"], "noop1")]
                    second = by_case_repeat[(case["case_id"], "noop2")]
                    valid = (
                        phenotype_matches(first, phenotype)
                        and phenotype_matches(second, phenotype)
                        and first["semantic_event"] == second["semantic_event"]
                    )
                else:
                    first = by_case_repeat[(case["case_id"], "baseline")]
                    valid = phenotype_matches(first, phenotype)
                if valid:
                    eligible.append(first)
            eligible_by_phenotype[phenotype] = eligible
            eligible_counts[pool][phenotype] = len(eligible)
        cap = (
            protocol.CAUSAL_SELECTION_PER_PHENOTYPE
            if pool == "block_causal"
            else protocol.TRACE_SELECTION_PER_PHENOTYPE
        )
        correct, confusion = matched_balanced(
            eligible_by_phenotype["stable_correct"],
            eligible_by_phenotype["stable_relation_confusion"],
            cap,
        )
        selected = {
            "stable_correct": correct,
            "stable_relation_confusion": confusion,
        }
        selected_ids[pool] = {
            phenotype: [row["case_id"] for row in selected[phenotype]]
            for phenotype in protocol.PHENOTYPES
        }
        pair_counts[pool] = {
            phenotype: len({
                (row["target"], row["other_relation_target"])
                for row in selected[phenotype]
            })
            for phenotype in protocol.PHENOTYPES
        }
        left_distribution = distribution(correct)
        right_distribution = distribution(confusion)
        matched_counts[pool] = len(correct)
        stratum_counts[pool] = len(left_distribution)
        distributions_equal[pool] = left_distribution == right_distribution
    trace_qualified = all(
        matched_counts[pool] >= protocol.MINIMUM_CASES_PER_PHENOTYPE
        and pair_counts[pool][phenotype] >= 8
        and distributions_equal[pool]
        for pool in ("block_discovery", "block_confirmation")
        for phenotype in protocol.PHENOTYPES
    )
    causal_qualified = (
        matched_counts["block_causal"] >= protocol.MINIMUM_CASES_PER_PHENOTYPE
        and all(
            pair_counts["block_causal"][phenotype] >= 8
            for phenotype in protocol.PHENOTYPES
        )
        and distributions_equal["block_causal"]
    )
    summary = {
        "schema_version": "phase571_matched_behavior_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model": model,
        "source_behavior_summary_sha256": sha256_file(source_summary_path(model)),
        "source_behavior_rows_sha256": sha256_file(rows_path(model)),
        "match_key": ["source_factorial_cell", "target", "other_relation_target"],
        "eligible_counts_by_pool_phenotype": eligible_counts,
        "matched_case_count_per_phenotype_by_pool": matched_counts,
        "matched_stratum_count_by_pool": stratum_counts,
        "matched_stratum_distributions_exactly_equal": distributions_equal,
        "selected_case_ids_by_pool_phenotype": selected_ids,
        "selected_target_other_pair_counts": pair_counts,
        "qualified_for_signed_write_trace": trace_qualified,
        "qualified_for_coarse_block_causal": causal_qualified,
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    write_json(summary_path(model), summary)
    return summary


def analyze() -> dict[str, Any]:
    reports = [analyze_model(model) for model in MODELS]
    output = {
        "schema_version": "phase571_matched_behavior_analysis.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete",
        "model_reports": reports,
        "all_models_trace_qualified": all(
            report["qualified_for_signed_write_trace"] for report in reports
        ),
        "all_models_causal_qualified": all(
            report["qualified_for_coarse_block_causal"] for report in reports
        ),
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    path = OUT_DIR / "phase571_matched_behavior_analysis.json"
    write_json(path, output)
    print(json.dumps({
        "all_models_trace_qualified": output["all_models_trace_qualified"],
        "all_models_causal_qualified": output["all_models_causal_qualified"],
        "models": [
            {
                "model": report["model"],
                "matched_counts": report["matched_case_count_per_phenotype_by_pool"],
                "strata": report["matched_stratum_count_by_pool"],
                "pairs": report["selected_target_other_pair_counts"],
                "equal": report["matched_stratum_distributions_exactly_equal"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    analyze()
