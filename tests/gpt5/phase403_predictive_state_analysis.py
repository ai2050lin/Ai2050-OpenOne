#!/usr/bin/env python3
"""Audit Phase403 finite predictive states with frozen integer gates."""

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

from phase403_predictive_state_protocol import (  # noqa: E402
    CONTEXTS,
    FAMILIES,
    MODELS,
    OUT,
    QUERIES,
    SPLIT_GROUP_COUNTS,
    STATE_VARIANTS,
    SURFACE_REPLICAS,
    abstract_state,
)


BASE_SINGLE_REQUIRED_CASES = 63
DISCOVERY_REQUIRED_GROUPS = 6
CALIBRATION_REQUIRED_GROUPS = 3
HOLDOUT_REQUIRED_GROUPS = 3
SURFACE_REQUIRED = 3
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
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def fingerprint_surface_pass(rows: list[dict[str, Any]], family: str) -> bool:
    if len(rows) != len(QUERIES[family]):
        return False
    return all(
        row["semantic_correct"]
        and row["predicted_canonical_private"]
        == row["expected_canonical_private"]
        for row in rows
    )


def consensus_fingerprint(rows: list[dict[str, Any]], family: str) -> tuple[str | None, ...]:
    return tuple(
        majority(
            [
                row["predicted_canonical_private"]
                for row in rows
                if row["future_query_private"] == query
            ]
        )
        for query in QUERIES[family]
    )


def expected_fingerprint(rows: list[dict[str, Any]], family: str) -> tuple[str, ...]:
    return tuple(
        next(
            row["expected_canonical_private"]
            for row in rows
            if row["future_query_private"] == query
        )
        for query in QUERIES[family]
    )


def state_separation_pass(rows: list[dict[str, Any]], family: str, contexts: list[str]) -> bool:
    for context in contexts:
        per_state: dict[int, tuple[tuple[str | None, ...], tuple[str, ...]]] = {}
        for state_variant in STATE_VARIANTS:
            selected = [
                row
                for row in rows
                if row["state_variant_private"] == state_variant
                and row["operation_context_private"] == context
            ]
            per_state[state_variant] = (
                consensus_fingerprint(selected, family),
                expected_fingerprint(selected, family),
            )
        expected_diff = sum(
            left != right
            for left, right in zip(
                per_state[0][1], per_state[1][1], strict=True
            )
        )
        if expected_diff < 2:
            return False
        if per_state[0][0] != per_state[0][1] or per_state[1][0] != per_state[1][1]:
            return False
    return True


def base_single_group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    contexts = [name for name, kind in CONTEXTS[family] if kind in {"base", "single"}]
    expected_cases = len(STATE_VARIANTS) * len(SURFACE_REPLICAS) * len(contexts) * len(QUERIES[family])
    semantic_count = sum(row["semantic_correct"] for row in rows)
    unit_rows: list[dict[str, Any]] = []
    all_units_pass = True
    for state_variant in STATE_VARIANTS:
        for context in contexts:
            surface_passes = 0
            anchor_surface_passes = 0
            unseen_surface_passes = 0
            for surface in SURFACE_REPLICAS:
                selected = [
                    row
                    for row in rows
                    if row["state_variant_private"] == state_variant
                    and row["operation_context_private"] == context
                    and row["surface_id_private"] == surface["surface_id"]
                ]
                passed = fingerprint_surface_pass(selected, family)
                surface_passes += int(passed)
                anchor_surface_passes += int(
                    len(selected) == len(QUERIES[family])
                    and all(
                        row["semantic_correct"]
                        for row in selected
                        if row["future_query_role_private"] == "anchor"
                    )
                )
                unseen_surface_passes += int(
                    len(selected) == len(QUERIES[family])
                    and all(
                        row["semantic_correct"]
                        for row in selected
                        if row["future_query_role_private"]
                        == "pre_registered_unseen"
                    )
                )
            unit_pass = surface_passes >= SURFACE_REQUIRED
            all_units_pass = all_units_pass and unit_pass
            unit_rows.append(
                {
                    "state_variant": state_variant,
                    "operation_context": context,
                    "surface_fingerprint_pass_count": surface_passes,
                    "anchor_surface_pass_count": anchor_surface_passes,
                    "unseen_query_surface_pass_count": unseen_surface_passes,
                    "unit_pass": unit_pass,
                }
            )
    separation = state_separation_pass(rows, family, ["base"])
    return {
        "case_count": len(rows),
        "expected_case_count": expected_cases,
        "semantic_correct_count": semantic_count,
        "semantic_case_gate": semantic_count >= BASE_SINGLE_REQUIRED_CASES,
        "surface_units": unit_rows,
        "all_state_context_units_pass": all_units_pass,
        "base_state_separation_pass": separation,
        "group_pass": len(rows) == expected_cases
        and semantic_count >= BASE_SINGLE_REQUIRED_CASES
        and all_units_pass
        and separation,
    }


def matched_state_blind_baseline(rows: list[dict[str, Any]]) -> float:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["anonymous_parallel_group_id"],
            row["operation_context_private"],
            row["surface_id_private"],
            row["future_query_private"],
        )
        buckets[key].append(row)
    correct = 0
    total = 0
    for selected in buckets.values():
        target_counts = Counter(row["expected_canonical_private"] for row in selected)
        correct += max(target_counts.values())
        total += len(selected)
    return correct / total if total else 0.0


def base_single_model_family_audit(rows: list[dict[str, Any]], family: str, split: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["anonymous_parallel_group_id"]].append(row)
    group_rows = []
    for group_id, selected in sorted(by_group.items()):
        audit = base_single_group_audit(selected, family)
        group_rows.append({"anonymous_parallel_group_id": group_id, **audit})
    semantic_correct = sum(row["semantic_correct"] for row in rows)
    semantic_rate = semantic_correct / len(rows) if rows else 0.0
    baseline = matched_state_blind_baseline(rows)
    required_groups = DISCOVERY_REQUIRED_GROUPS if split == "discovery" else CALIBRATION_REQUIRED_GROUPS
    expected_groups = SPLIT_GROUP_COUNTS[split]
    group_pass_count = sum(row["group_pass"] for row in group_rows)
    return {
        "family_id": family,
        "split": split,
        "case_count": len(rows),
        "semantic_correct_count": semantic_correct,
        "semantic_correct_rate": semantic_rate,
        "matched_state_blind_baseline_rate": baseline,
        "semantic_minus_baseline": semantic_rate - baseline,
        "baseline_margin_pass": semantic_rate - baseline >= BASELINE_MARGIN_MIN,
        "group_count": len(group_rows),
        "group_pass_count": group_pass_count,
        "required_group_pass_count": required_groups,
        "model_family_pass": len(group_rows) == expected_groups
        and group_pass_count >= required_groups
        and semantic_rate - baseline >= BASELINE_MARGIN_MIN,
        "groups": group_rows,
    }


def composition_group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    contexts = [name for name, kind in CONTEXTS[family] if kind == "composition"]
    expected_cases = len(STATE_VARIANTS) * len(SURFACE_REPLICAS) * len(contexts) * len(QUERIES[family])
    required_cases = 21 if family == "grammar_constraint" else 42
    semantic_count = sum(row["semantic_correct"] for row in rows)
    units = []
    all_units_pass = True
    for state_variant in STATE_VARIANTS:
        for context in contexts:
            surface_pass_count = 0
            for surface in SURFACE_REPLICAS:
                selected = [
                    row
                    for row in rows
                    if row["state_variant_private"] == state_variant
                    and row["operation_context_private"] == context
                    and row["surface_id_private"] == surface["surface_id"]
                ]
                surface_pass_count += int(fingerprint_surface_pass(selected, family))
            unit_pass = surface_pass_count >= SURFACE_REQUIRED
            all_units_pass = all_units_pass and unit_pass
            units.append(
                {
                    "state_variant": state_variant,
                    "operation_context": context,
                    "surface_fingerprint_pass_count": surface_pass_count,
                    "unit_pass": unit_pass,
                }
            )
    order_distinction_pass = True
    if len(contexts) == 2:
        for state_variant in STATE_VARIANTS:
            fingerprints = []
            expected = []
            for context in contexts:
                selected = [
                    row
                    for row in rows
                    if row["state_variant_private"] == state_variant
                    and row["operation_context_private"] == context
                ]
                fingerprints.append(consensus_fingerprint(selected, family))
                expected.append(expected_fingerprint(selected, family))
            expected_diff = sum(
                left != right for left, right in zip(expected[0], expected[1], strict=True)
            )
            order_distinction_pass = order_distinction_pass and expected_diff >= 2
            order_distinction_pass = order_distinction_pass and fingerprints == expected
    return {
        "case_count": len(rows),
        "expected_case_count": expected_cases,
        "semantic_correct_count": semantic_count,
        "required_semantic_correct_count": required_cases,
        "all_composition_units_pass": all_units_pass,
        "composition_order_distinction_pass": order_distinction_pass,
        "units": units,
        "group_pass": len(rows) == expected_cases
        and semantic_count >= required_cases
        and all_units_pass
        and order_distinction_pass,
    }


def composition_model_family_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["anonymous_parallel_group_id"]].append(row)
    groups = []
    for group_id, selected in sorted(by_group.items()):
        groups.append(
            {
                "anonymous_parallel_group_id": group_id,
                **composition_group_audit(selected, family),
            }
        )
    semantic_correct = sum(row["semantic_correct"] for row in rows)
    semantic_rate = semantic_correct / len(rows) if rows else 0.0
    baseline = matched_state_blind_baseline(rows)
    group_pass_count = sum(row["group_pass"] for row in groups)
    return {
        "family_id": family,
        "split": "behavioral_holdout",
        "case_count": len(rows),
        "semantic_correct_count": semantic_correct,
        "semantic_correct_rate": semantic_rate,
        "matched_state_blind_baseline_rate": baseline,
        "semantic_minus_baseline": semantic_rate - baseline,
        "baseline_margin_pass": semantic_rate - baseline >= BASELINE_MARGIN_MIN,
        "group_count": len(groups),
        "group_pass_count": group_pass_count,
        "required_group_pass_count": HOLDOUT_REQUIRED_GROUPS,
        "model_family_pass": len(groups)
        == SPLIT_GROUP_COUNTS["behavioral_holdout"]
        and group_pass_count >= HOLDOUT_REQUIRED_GROUPS
        and semantic_rate - baseline >= BASELINE_MARGIN_MIN,
        "groups": groups,
    }


def expected_transition_table(family: str) -> list[dict[str, Any]]:
    rows = []
    for state_variant in STATE_VARIANTS:
        for context, kind in CONTEXTS[family]:
            rows.append(
                {
                    "source_state_variant": state_variant,
                    "operation_context": context,
                    "context_kind": kind,
                    "target_abstract_state": list(
                        abstract_state(family, state_variant, context)
                    ),
                }
            )
    return rows


def prior_authorized_families(stage: str) -> tuple[str, ...]:
    if stage == "discovery":
        return FAMILIES
    if stage == "calibration":
        return tuple(
            read_json(OUT / "phase403_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    return tuple(
        read_json(OUT / "phase403_calibration_analysis.json")[
            "crossmodel_candidate_families"
        ]
    )


def main(stage: str) -> None:
    families = prior_authorized_families(stage)
    model_rows: list[dict[str, Any]] = []
    all_raw_rows: list[dict[str, Any]] = []
    for model in MODELS:
        complete_path = OUT / "behavior" / stage / model / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(complete_path)
        complete = read_json(complete_path)
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase403 collection: {model}/{stage}")
        path = OUT / "behavior" / stage / "private" / model / "rows.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        all_raw_rows.extend(rows)
        for family in families:
            selected = [row for row in rows if row["family_id"] == family]
            audit = (
                composition_model_family_audit(selected, family)
                if stage == "behavioral_holdout"
                else base_single_model_family_audit(selected, family, stage)
            )
            model_rows.append({"model": model, **audit})

    crossmodel_candidates = []
    for family in families:
        selected = [row for row in model_rows if row["family_id"] == family]
        if len(selected) == len(MODELS) and all(
            row["model_family_pass"] for row in selected
        ):
            crossmodel_candidates.append(family)

    output_name = f"phase403_{stage}_analysis.json"
    payload = {
        "schema_version": "77.2.0",
        "phase_id": "Phase403-PredictiveStateAnalysis",
        "created_at": now(),
        "stage": stage,
        "authorized_families": list(families),
        "models": list(MODELS),
        "case_count": len(all_raw_rows),
        "semantic_correct_count": sum(
            row["semantic_correct"] for row in all_raw_rows
        ),
        "model_family_rows": model_rows,
        "crossmodel_candidate_families": crossmodel_candidates,
        "transition_tables": {
            family: expected_transition_table(family) for family in families
        },
        "exact_abstract_table_mapping_has_learned_parameters": False,
        "authorization": {
            "run_calibration": stage == "discovery"
            and bool(crossmodel_candidates),
            "run_behavioral_holdout_composition": stage == "calibration"
            and bool(crossmodel_candidates),
            "run_physical_mapping": stage == "behavioral_holdout"
            and bool(crossmodel_candidates),
            "run_limited_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "candidate_name": "finite_predictive_state_candidate",
            "candidate_is_interventional_causal_state": False,
            "observed_natural_transition_is_internal_operator": False,
            "exact_crossmodel_table_is_brain_isomorphism": False,
        },
    }
    write_json(OUT / output_name, payload)
    write_jsonl(OUT / "analysis" / f"phase403_{stage}_model_family_rows.jsonl", model_rows)
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
