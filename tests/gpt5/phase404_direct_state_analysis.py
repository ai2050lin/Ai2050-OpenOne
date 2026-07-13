#!/usr/bin/env python3
"""Audit Phase404 direct finite-state response graphs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase404_direct_state_protocol import (  # noqa: E402
    FAMILIES,
    MODELS,
    OUT,
    QUERIES,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    SURFACE_REPLICAS,
    expected_answer,
)


REQUIRED_CASES = {
    "knowledge_binding": 42,
    "rule_reasoning": 21,
    "grammar_constraint": 42,
}
REQUIRED_GROUPS = {
    "discovery": 6,
    "calibration": 3,
    "behavioral_holdout": 3,
}
BASELINE_MARGIN_MIN = 0.20


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def majority(values: list[str | None]) -> str | None:
    counts = Counter(values)
    if not counts:
        return None
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    winner = top[0][0]
    return winner if isinstance(winner, str) else None


def state_consensus(rows: list[dict[str, Any]], family: str) -> tuple[str | None, ...]:
    return tuple(
        majority(
            [
                row["predicted_candidate_private"]
                for row in rows
                if row["future_query_private"] == query
            ]
        )
        for query in QUERIES[family]
    )


def expected_fingerprint(family: str, state_id: str) -> tuple[str, ...]:
    return tuple(
        expected_answer(family, state_id, query) for query in QUERIES[family]
    )


def group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    expected_cases = (
        len(STATE_IDS[family]) * len(SURFACE_REPLICAS) * len(QUERIES[family])
    )
    correct_count = sum(row["finite_candidate_correct"] for row in rows)
    positive_margin_count = sum(
        row["target_minus_best_distractor_logit"] is not None
        and row["target_minus_best_distractor_logit"] > 0
        for row in rows
    )
    state_rows = []
    truth_units_pass = True
    observed_units_pass = True
    observed_fingerprints: list[tuple[str | None, ...]] = []
    expected_fingerprints: list[tuple[str, ...]] = []
    for state_id in STATE_IDS[family]:
        selected = [row for row in rows if row["state_id_private"] == state_id]
        surface_truth_pass = 0
        surface_winner_fingerprints: list[tuple[str, ...]] = []
        for surface in SURFACE_REPLICAS:
            surface_selected = [
                row
                for row in selected
                if row["surface_id_private"] == surface["surface_id"]
            ]
            predicted = tuple(
                next(
                    row["predicted_candidate_private"]
                    for row in surface_selected
                    if row["future_query_private"] == query
                )
                for query in QUERIES[family]
            )
            surface_winner_fingerprints.append(predicted)
            surface_truth_pass += int(
                len(surface_selected) == len(QUERIES[family])
                and all(row["finite_candidate_correct"] for row in surface_selected)
            )
        fingerprint_counts = Counter(surface_winner_fingerprints)
        observed_surface_consistency = max(fingerprint_counts.values())
        observed_consensus = state_consensus(selected, family)
        expected = expected_fingerprint(family, state_id)
        state_truth_pass = surface_truth_pass >= 3 and observed_consensus == expected
        state_observed_pass = observed_surface_consistency >= 3
        truth_units_pass = truth_units_pass and state_truth_pass
        observed_units_pass = observed_units_pass and state_observed_pass
        observed_fingerprints.append(observed_consensus)
        expected_fingerprints.append(expected)
        state_rows.append(
            {
                "state_id": state_id,
                "surface_truth_fingerprint_pass_count": surface_truth_pass,
                "surface_observed_fingerprint_consistency_count": observed_surface_consistency,
                "observed_consensus_fingerprint": list(observed_consensus),
                "expected_fingerprint": list(expected),
                "state_truth_pass": state_truth_pass,
                "state_observed_consistency_pass": state_observed_pass,
            }
        )
    observed_distinct = len(set(observed_fingerprints)) == len(
        observed_fingerprints
    )
    expected_distinct = len(set(expected_fingerprints)) == len(
        expected_fingerprints
    )
    return {
        "case_count": len(rows),
        "expected_case_count": expected_cases,
        "finite_candidate_correct_count": correct_count,
        "required_finite_candidate_correct_count": REQUIRED_CASES[family],
        "positive_target_margin_count": positive_margin_count,
        "all_state_truth_units_pass": truth_units_pass,
        "all_state_observed_consistency_units_pass": observed_units_pass,
        "observed_state_fingerprints_pairwise_distinct": observed_distinct,
        "expected_state_fingerprints_pairwise_distinct": expected_distinct,
        "states": state_rows,
        "observed_structure_group_pass": len(rows) == expected_cases
        and observed_units_pass
        and observed_distinct,
        "truth_predictive_group_pass": len(rows) == expected_cases
        and correct_count >= REQUIRED_CASES[family]
        and truth_units_pass
        and observed_distinct
        and expected_distinct,
    }


def matched_state_blind_baseline(rows: list[dict[str, Any]]) -> float:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row["anonymous_parallel_group_id"],
                row["surface_id_private"],
                row["future_query_private"],
            )
        ].append(row)
    correct = 0
    total = 0
    for selected in buckets.values():
        counts = Counter(row["target_private"] for row in selected)
        correct += max(counts.values())
        total += len(selected)
    return correct / total if total else 0.0


def model_family_audit(rows: list[dict[str, Any]], family: str, split: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["anonymous_parallel_group_id"]].append(row)
    groups = []
    for group_id, selected in sorted(by_group.items()):
        groups.append(
            {
                "anonymous_parallel_group_id": group_id,
                **group_audit(selected, family),
            }
        )
    correct = sum(row["finite_candidate_correct"] for row in rows)
    accuracy = correct / len(rows) if rows else 0.0
    baseline = matched_state_blind_baseline(rows)
    truth_group_pass_count = sum(
        row["truth_predictive_group_pass"] for row in groups
    )
    observed_group_pass_count = sum(
        row["observed_structure_group_pass"] for row in groups
    )
    return {
        "family_id": family,
        "split": split,
        "case_count": len(rows),
        "finite_candidate_correct_count": correct,
        "finite_candidate_accuracy": accuracy,
        "matched_state_blind_baseline_rate": baseline,
        "accuracy_minus_state_blind_baseline": accuracy - baseline,
        "baseline_margin_pass": accuracy - baseline >= BASELINE_MARGIN_MIN,
        "group_count": len(groups),
        "observed_structure_group_pass_count": observed_group_pass_count,
        "truth_predictive_group_pass_count": truth_group_pass_count,
        "required_truth_group_pass_count": REQUIRED_GROUPS[split],
        "model_family_pass": len(groups) == SPLIT_GROUP_COUNTS[split]
        and truth_group_pass_count >= REQUIRED_GROUPS[split]
        and accuracy - baseline >= BASELINE_MARGIN_MIN,
        "groups": groups,
    }


def authorized_families(stage: str) -> tuple[str, ...]:
    if stage == "discovery":
        return FAMILIES
    if stage == "calibration":
        return tuple(
            read_json(OUT / "phase404_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    return tuple(
        read_json(OUT / "phase404_calibration_analysis.json")[
            "crossmodel_candidate_families"
        ]
    )


def main(stage: str) -> None:
    families = authorized_families(stage)
    model_family_rows = []
    all_rows = []
    for model in MODELS:
        complete = read_json(
            OUT / "collection" / stage / model / "complete.json"
        )
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase404 collection: {model}/{stage}")
        path = OUT / "collection" / stage / "private" / model / "rows.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        all_rows.extend(rows)
        for family in families:
            selected = [row for row in rows if row["family_id"] == family]
            model_family_rows.append(
                {
                    "model": model,
                    **model_family_audit(selected, family, stage),
                }
            )
    crossmodel_candidates = []
    for family in families:
        selected = [
            row for row in model_family_rows if row["family_id"] == family
        ]
        if len(selected) == len(MODELS) and all(
            row["model_family_pass"] for row in selected
        ):
            crossmodel_candidates.append(family)
    payload = {
        "schema_version": "78.2.0",
        "phase_id": "Phase404-DirectStateAnalysis",
        "created_at": now(),
        "stage": stage,
        "authorized_families": list(families),
        "models": list(MODELS),
        "case_count": len(all_rows),
        "finite_candidate_correct_count": sum(
            row["finite_candidate_correct"] for row in all_rows
        ),
        "global_top_is_target_count": sum(
            row["global_top_is_target_token"] for row in all_rows
        ),
        "model_family_rows": model_family_rows,
        "crossmodel_candidate_families": crossmodel_candidates,
        "authorization": {
            "run_calibration": stage == "discovery"
            and bool(crossmodel_candidates),
            "run_behavioral_holdout": stage == "calibration"
            and bool(crossmodel_candidates),
            "run_physical_holdout_mapping": stage == "behavioral_holdout"
            and bool(crossmodel_candidates),
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "candidate_name": "direct_finite_predictive_state_candidate",
            "candidate_is_causal_state": False,
            "candidate_response_is_full_vocabulary_behavior": False,
            "semantic_transition_graph_is_internal_operator": False,
        },
    }
    write_json(OUT / f"phase404_{stage}_analysis.json", payload)
    write_jsonl(
        OUT / "analysis" / f"phase404_{stage}_model_family_rows.jsonl",
        model_family_rows,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "calibration", "behavioral_holdout"),
        required=True,
    )
    args = parser.parse_args()
    main(args.stage)
