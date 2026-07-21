#!/usr/bin/env python3
"""Apply the frozen Phase586 cross-model consensus gate."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterator

import phase585_object_swap_behavior as source_behavior
import phase585_object_swap_protocol as source
import phase586_cross_semantic_audit as audit
import phase586_cross_semantic_audit_protocol as protocol


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Any) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    judgments: dict[tuple[str, str, str], dict[str, str | None]] = defaultdict(dict)
    judge_summaries = {}
    for judge_model in protocol.MODELS:
        paths = audit.paths(judge_model)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary["rows_sha256"] != protocol.sha256_file(paths["rows"]):
            raise RuntimeError(f"Phase586 {judge_model} rows drift")
        if summary["sealed_split_read"] or not summary["judge_quality_gate_passes"]:
            raise RuntimeError(f"Phase586 invalid judge {judge_model}")
        judge_summaries[judge_model] = summary
        for row in iter_jsonl(paths["rows"]):
            judgments[(row["source_model"], row["case_id"], judge_model)][
                row["judge_repeat"]
            ] = row["judgment"]

    source_cases: dict[tuple[str, str], dict[str, Any]] = {}
    for source_model in source.MODELS:
        for row in iter_jsonl(source_behavior.paths(source_model)["rows"]):
            if row["execution_repeat"] == source.NOOP_REPEATS[0]:
                source_cases[(source_model, row["case_id"])] = row

    consensus: dict[tuple[str, str], dict[str, Any]] = {}
    for source_key, row in source_cases.items():
        votes = []
        unstable = []
        for judge_model in protocol.MODELS:
            repeats = judgments[(*source_key, judge_model)]
            stable = bool(
                set(repeats) == set(protocol.JUDGE_REPEATS)
                and repeats[protocol.JUDGE_REPEATS[0]] is not None
                and repeats[protocol.JUDGE_REPEATS[0]]
                == repeats[protocol.JUDGE_REPEATS[1]]
            )
            if stable:
                votes.append(repeats[protocol.JUDGE_REPEATS[0]])
            else:
                unstable.append(judge_model)
        counts = Counter(votes)
        accepted = bool(
            not unstable
            and counts.get("YES", 0) >= protocol.MIN_YES_VOTES
            and counts.get("NO", 0) <= protocol.MAX_NO_VOTES
        )
        consensus[source_key] = {
            "accepted": accepted,
            "vote_counts": dict(counts),
            "unstable_judges": unstable,
            "row": row,
        }

    model_results = {}
    sealed_authorized: dict[str, list[str]] = {}
    for source_model in source.MODELS:
        unit_metrics: dict[str, dict[str, Any]] = {}
        passing_relations = []
        for relation in source.RELATIONS:
            relation_pass = True
            for split in source.OPEN_SPLITS:
                cases = [
                    item
                    for (model, _), item in consensus.items()
                    if model == source_model
                    and item["row"]["relation"] == relation
                    and item["row"]["split"] == split
                ]
                by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for item in cases:
                    by_object[item["row"]["object_id"]].append(item)
                qualified = [
                    object_id
                    for object_id, object_cases in sorted(by_object.items())
                    if sum(item["accepted"] for item in object_cases)
                    >= source.MIN_STABLE_SURFACES_PER_OBJECT
                ]
                group_by_object = {
                    object_id: object_cases[0]["row"]["semantic_group"]
                    for object_id, object_cases in by_object.items()
                }
                qualified_by_group = dict(
                    Counter(group_by_object[object_id] for object_id in qualified)
                )
                accuracy = sum(item["accepted"] for item in cases) / len(cases)
                minimums = source.MIN_QUALIFIED_BY_SPLIT_GROUP[split]
                passes = bool(
                    accuracy >= source.MIN_SEMANTIC_ACCURACY
                    and all(
                        qualified_by_group.get(group, 0) >= minimum
                        for group, minimum in minimums.items()
                    )
                )
                unit_metrics.setdefault(split, {})[relation] = {
                    "case_count": len(cases),
                    "consensus_semantic_accuracy": accuracy,
                    "qualified_object_count": len(qualified),
                    "qualified_object_count_by_group": qualified_by_group,
                    "minimum_qualified_object_count_by_group": minimums,
                    "consensus_vote_pattern_counts": dict(
                        Counter(
                            ",".join(
                                f"{label}:{count}"
                                for label, count in sorted(item["vote_counts"].items())
                            )
                            for item in cases
                        )
                    ),
                    "pass": passes,
                }
                relation_pass = relation_pass and passes
            if relation_pass:
                passing_relations.append(relation)
        if passing_relations:
            sealed_authorized[source_model] = passing_relations
        model_results[source_model] = {
            "unit_metrics": unit_metrics,
            "sealed_behavior_authorized_relations": passing_relations,
        }

    payload = {
        "schema_version": "phase586_cross_semantic_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_retrospective_open_semantic_audit",
        "judge_summaries": judge_summaries,
        "model_results": model_results,
        "sealed_behavior_authorized_model_relations": sealed_authorized,
        "internal_trace_authorized_model_relations": {},
        "causal_intervention_authorized": False,
        "sealed_split_read": False,
        "evidence_classification": {
            "retrospective_observer_calibration": True,
            "independent_behavior_confirmation": False,
            "internal_structure_evidence": False,
        },
    }
    protocol.write_json(protocol.DECISION_PATH, payload)
    print(
        json.dumps(
            {
                "sealed_behavior_authorized_model_relations": sealed_authorized,
                "internal_trace_authorized_model_relations": {},
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
