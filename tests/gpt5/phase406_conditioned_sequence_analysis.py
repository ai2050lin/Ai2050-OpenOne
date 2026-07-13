#!/usr/bin/env python3
"""Parse and gate Phase406 exact condition-response sequence tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase406_conditioned_sequence_protocol import (  # noqa: E402
    FAMILIES,
    INTERFACES,
    MODELS,
    OUT,
    QUERIES,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    SURFACE_REPLICAS,
)


GROUP_CORRECT_MIN = {
    "knowledge_binding": 84,
    "rule_reasoning": 42,
    "grammar_constraint": 84,
}
GROUP_REQUIRED = {"discovery": 6, "calibration": 3, "behavioral_holdout": 3}


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


def _alternation(candidates: list[str]) -> str:
    return "|".join(
        re.escape(candidate) for candidate in sorted(candidates, key=len, reverse=True)
    )


def extract_semantic_label(
    text: str,
    candidates: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Conservatively extract one frozen semantic label from a short response."""

    aliases = aliases or {candidate: [candidate] for candidate in candidates}
    alias_to_canonical = {
        alias.lower(): canonical
        for canonical in candidates
        for alias in aliases.get(canonical, [canonical])
    }
    all_aliases = list(alias_to_canonical)
    alternatives = _alternation(all_aliases)
    boundary = rf"(?<![A-Za-z0-9])({alternatives})(?![A-Za-z0-9])"
    methods = [
        (
            "response_initial",
            re.compile(
                rf"^\s*[\[\(\{{\"'`*_#:\-]*\s*{boundary}", re.IGNORECASE
            ),
        ),
        (
            "quoted_label",
            re.compile(
                rf"[\"'`]\s*{boundary}\s*[\"'`]", re.IGNORECASE
            ),
        ),
        (
            "answer_slot",
            re.compile(
                rf"(?:answer|form|word|auxiliary|demonstrative|color|label)"
                rf"(?:\s+word)?\s*(?:is|would\s+be|should\s+be|:)\s*"
                rf"[\"'`]*\s*{boundary}",
                re.IGNORECASE,
            ),
        ),
        (
            "modal_slot",
            re.compile(
                rf"(?:should|would|must)\s+be\s*[\"'`]*\s*{boundary}",
                re.IGNORECASE,
            ),
        ),
    ]
    if set(candidates) == {"A", "B"}:
        methods.append(
            (
                "person_label",
                re.compile(r"\bperson\s+([AB])\b", re.IGNORECASE),
            )
        )

    for method, pattern in methods:
        match = pattern.search(text)
        if match:
            raw = match.group(match.lastindex or 1)
            label = alias_to_canonical.get(raw.lower())
            if label is not None:
                return {
                    "semantic_label_private": label,
                    "semantic_parse_method": method,
                    "semantic_span_start_private": match.start(match.lastindex or 1),
                    "semantic_span_end_private": match.end(match.lastindex or 1),
                    "semantic_parse_ambiguous": False,
                }

    # General word search is useful for colors and yes/no, but is deliberately
    # disabled for auxiliaries and single-letter labels because copulas and
    # articles otherwise create false semantic spans.
    risky = {"is", "are", "was", "were", "has", "have", "had", "a", "b"}
    safe_aliases = [alias for alias in all_aliases if alias.lower() not in risky]
    if safe_aliases:
        safe_boundary = (
            rf"(?<![A-Za-z0-9])({_alternation(safe_aliases)})(?![A-Za-z0-9])"
        )
        matches = list(re.finditer(safe_boundary, text, re.IGNORECASE))
        labels = [alias_to_canonical[match.group(1).lower()] for match in matches]
        unique = list(dict.fromkeys(labels))
        if len(unique) == 1:
            match = matches[0]
            return {
                "semantic_label_private": unique[0],
                "semantic_parse_method": "unique_general_label",
                "semantic_span_start_private": match.start(1),
                "semantic_span_end_private": match.end(1),
                "semantic_parse_ambiguous": False,
            }
        if len(unique) > 1:
            return {
                "semantic_label_private": None,
                "semantic_parse_method": "multiple_general_labels",
                "semantic_span_start_private": None,
                "semantic_span_end_private": None,
                "semantic_parse_ambiguous": True,
            }

    return {
        "semantic_label_private": None,
        "semantic_parse_method": "no_conservative_label",
        "semantic_span_start_private": None,
        "semantic_span_end_private": None,
        "semantic_parse_ambiguous": False,
    }


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = extract_semantic_label(
        row["generated_text_clean_private"],
        row["semantic_candidate_labels_private"],
        row.get("semantic_aliases_private"),
    )
    semantic_end = parsed["semantic_span_end_private"]
    suffix = (
        row["generated_text_clean_private"][semantic_end:]
        if semantic_end is not None
        else ""
    )
    sentence_boundary_after_semantic = bool(re.search(r"[.!?\n]", suffix))
    semantic_correct = (
        parsed["semantic_label_private"]
        == row["target_semantic_label_private"]
        and row["all_generated_step_logits_valid"]
    )
    return {
        **row,
        **parsed,
        "short_sequence_semantic_correct": semantic_correct,
        "semantic_answer_complete": semantic_correct
        and not parsed["semantic_parse_ambiguous"],
        "sentence_boundary_after_semantic": sentence_boundary_after_semantic,
        "sequence_stop_or_boundary_after_semantic": semantic_correct
        and (row["eos_observed"] or sentence_boundary_after_semantic),
    }


def strict_majority(labels: list[str | None]) -> tuple[str | None, int]:
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None, 0
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None, ordered[0][1]
    return ordered[0][0], ordered[0][1]


def leave_one_interface_folds(
    rows: list[dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    result = []
    for state_id in STATE_IDS[family]:
        for query in QUERIES[family]:
            for heldout_interface in INTERFACES:
                source_interface = next(
                    interface
                    for interface in INTERFACES
                    if interface != heldout_interface
                )
                source = [
                    row
                    for row in rows
                    if row["state_id_private"] == state_id
                    and row["future_query_private"] == query
                    and row["interface_private"] == source_interface
                ]
                heldout = [
                    row
                    for row in rows
                    if row["state_id_private"] == state_id
                    and row["future_query_private"] == query
                    and row["interface_private"] == heldout_interface
                ]
                predicted, source_count = strict_majority(
                    [row["semantic_label_private"] for row in source]
                )
                target = heldout[0]["target_semantic_label_private"] if heldout else None
                heldout_match_count = sum(
                    row["semantic_label_private"] == predicted for row in heldout
                )
                fold_pass = (
                    len(source) == len(SURFACE_REPLICAS)
                    and len(heldout) == len(SURFACE_REPLICAS)
                    and source_count >= 3
                    and predicted == target
                    and heldout_match_count >= 3
                )
                result.append(
                    {
                        "state_id": state_id,
                        "query": query,
                        "source_interface": source_interface,
                        "heldout_interface": heldout_interface,
                        "predicted_semantic_label_private": predicted,
                        "target_semantic_label_private": target,
                        "source_consensus_count": source_count,
                        "heldout_match_count": heldout_match_count,
                        "fold_pass": fold_pass,
                    }
                )
    return result


def group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    expected_cases = (
        len(STATE_IDS[family])
        * len(SURFACE_REPLICAS)
        * len(QUERIES[family])
        * len(INTERFACES)
    )
    units = []
    signature: dict[str, list[str | None]] = defaultdict(list)
    all_units_pass = True
    for state_id in STATE_IDS[family]:
        for query in QUERIES[family]:
            for interface in INTERFACES:
                selected = [
                    row
                    for row in rows
                    if row["state_id_private"] == state_id
                    and row["future_query_private"] == query
                    and row["interface_private"] == interface
                ]
                labels = [row["semantic_label_private"] for row in selected]
                majority, majority_count = strict_majority(labels)
                correct_count = sum(
                    row["short_sequence_semantic_correct"] for row in selected
                )
                unit_pass = (
                    len(selected) == len(SURFACE_REPLICAS)
                    and correct_count >= 3
                    and majority
                    == (selected[0]["target_semantic_label_private"] if selected else None)
                )
                all_units_pass = all_units_pass and unit_pass
                signature[state_id].append(majority)
                units.append(
                    {
                        "state_id": state_id,
                        "query": query,
                        "interface": interface,
                        "case_count": len(selected),
                        "semantic_correct_count": correct_count,
                        "majority_semantic_label_private": majority,
                        "majority_count": majority_count,
                        "unit_pass": unit_pass,
                    }
                )

    distinct_state_pairs = 0
    required_state_pairs = 0
    state_ids = list(STATE_IDS[family])
    for left_index, left in enumerate(state_ids):
        for right in state_ids[left_index + 1 :]:
            required_state_pairs += 1
            distinct_state_pairs += int(signature[left] != signature[right])

    folds = leave_one_interface_folds(rows, family)
    semantic_correct_count = sum(
        row["short_sequence_semantic_correct"] for row in rows
    )
    semantic_parse_count = sum(
        row["semantic_label_private"] is not None for row in rows
    )
    sequence_boundary_count = sum(
        row["sequence_stop_or_boundary_after_semantic"] for row in rows
    )
    group_pass = (
        len(rows) == expected_cases
        and semantic_correct_count >= GROUP_CORRECT_MIN[family]
        and all_units_pass
        and all(fold["fold_pass"] for fold in folds)
        and distinct_state_pairs == required_state_pairs
    )
    return {
        "case_count": len(rows),
        "expected_case_count": expected_cases,
        "first_step_candidate_correct_count": sum(
            row["first_step_candidate_correct"] for row in rows
        ),
        "first_step_global_top_is_target_count": sum(
            row["first_step_global_top_is_target"] for row in rows
        ),
        "semantic_parse_count": semantic_parse_count,
        "short_sequence_semantic_correct_count": semantic_correct_count,
        "required_short_sequence_semantic_correct_count": GROUP_CORRECT_MIN[family],
        "sequence_stop_or_boundary_after_semantic_count": sequence_boundary_count,
        "all_state_condition_units_pass": all_units_pass,
        "leave_one_interface_fold_pass_count": sum(
            fold["fold_pass"] for fold in folds
        ),
        "leave_one_interface_fold_count": len(folds),
        "all_leave_one_interface_folds_pass": all(
            fold["fold_pass"] for fold in folds
        ),
        "distinct_state_signature_pair_count": distinct_state_pairs,
        "required_distinct_state_signature_pair_count": required_state_pairs,
        "group_pass": group_pass,
        "state_condition_units": units,
        "leave_one_interface_folds": folds,
    }


def model_family_audit(
    rows: list[dict[str, Any]], family: str, split: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    groups = [
        {
            "anonymous_parallel_group_id": group_id,
            **group_audit(selected, family),
        }
        for group_id, selected in sorted(grouped.items())
    ]
    pass_count = sum(group["group_pass"] for group in groups)
    return {
        "family_id": family,
        "case_count": len(rows),
        "first_step_candidate_correct_count": sum(
            row["first_step_candidate_correct"] for row in rows
        ),
        "first_step_global_top_is_target_count": sum(
            row["first_step_global_top_is_target"] for row in rows
        ),
        "semantic_parse_count": sum(
            row["semantic_label_private"] is not None for row in rows
        ),
        "short_sequence_semantic_correct_count": sum(
            row["short_sequence_semantic_correct"] for row in rows
        ),
        "sequence_stop_or_boundary_after_semantic_count": sum(
            row["sequence_stop_or_boundary_after_semantic"] for row in rows
        ),
        "group_pass_count": pass_count,
        "required_group_pass_count": GROUP_REQUIRED[split],
        "model_family_pass": len(groups) == SPLIT_GROUP_COUNTS[split]
        and pass_count >= GROUP_REQUIRED[split],
        "groups": groups,
    }


def authorized_families(stage: str) -> tuple[str, ...]:
    if stage == "discovery":
        return FAMILIES
    if stage == "calibration":
        return tuple(
            read_json(OUT / "phase406_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    return tuple(
        read_json(OUT / "phase406_calibration_analysis.json")[
            "crossmodel_candidate_families"
        ]
    )


def main(stage: str) -> None:
    families = authorized_families(stage)
    summaries = []
    group_details = []
    all_enriched = []
    for model in MODELS:
        complete = read_json(OUT / "collection" / stage / model / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase406 collection: {model}/{stage}")
        path = OUT / "collection" / stage / "private" / model / "rows.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        enriched = [enrich_row(row) for row in rows]
        all_enriched.extend(enriched)
        write_jsonl(
            OUT / "analysis" / stage / "private" / model / "semantic_rows.jsonl",
            enriched,
        )
        for family in families:
            selected = [row for row in enriched if row["family_id"] == family]
            audit = model_family_audit(selected, family, stage)
            groups = audit.pop("groups")
            group_details.extend(
                {
                    "model": model,
                    "family_id": family,
                    "split": stage,
                    **group,
                }
                for group in groups
            )
            summaries.append({"model": model, **audit})

    crossmodel_candidates = []
    for family in families:
        selected = [row for row in summaries if row["family_id"] == family]
        if len(selected) == len(MODELS) and all(
            row["model_family_pass"] for row in selected
        ):
            crossmodel_candidates.append(family)

    payload = {
        "schema_version": "80.2.0",
        "phase_id": "Phase406-ConditionedSequenceAnalysis",
        "created_at": now(),
        "stage": stage,
        "authorized_families": list(families),
        "models": list(MODELS),
        "case_count": len(all_enriched),
        "first_step_candidate_correct_count": sum(
            row["first_step_candidate_correct"] for row in all_enriched
        ),
        "first_step_global_top_is_target_count": sum(
            row["first_step_global_top_is_target"] for row in all_enriched
        ),
        "semantic_parse_count": sum(
            row["semantic_label_private"] is not None for row in all_enriched
        ),
        "short_sequence_semantic_correct_count": sum(
            row["short_sequence_semantic_correct"] for row in all_enriched
        ),
        "sequence_stop_or_boundary_after_semantic_count": sum(
            row["sequence_stop_or_boundary_after_semantic"] for row in all_enriched
        ),
        "model_family_rows": summaries,
        "crossmodel_candidate_families": crossmodel_candidates,
        "authorization": {
            "run_calibration": stage == "discovery" and bool(crossmodel_candidates),
            "run_behavioral_holdout": stage == "calibration"
            and bool(crossmodel_candidates),
            "run_direct_operator": stage == "behavioral_holdout"
            and bool(crossmodel_candidates),
            "run_physical_mapping": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "candidate_name": "finite_conditioned_sequence_state_candidate",
            "leave_one_test_is_interface_pair_transfer_not_all_future_prediction": True,
            "six_condition_panel_is_exhaustive": False,
            "history_generalization_tested": False,
            "conditioned_sequence_state_is_causal_state": False,
            "semantic_transition_graph_is_internal_operator": False,
        },
    }
    write_json(OUT / f"phase406_{stage}_analysis.json", payload)
    write_jsonl(
        OUT / "analysis" / f"phase406_{stage}_group_details.jsonl",
        group_details,
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
