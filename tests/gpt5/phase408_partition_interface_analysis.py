#!/usr/bin/env python3
"""Analyze Phase408 response partitions without treating them as internal states."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_collection import (  # noqa: E402
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase408_partition_interface_protocol import (  # noqa: E402
    FAMILIES,
    INTERFACES,
    MODELS,
    OUT,
    STATE_IDS,
    interface_coordinate_map,
    token_words,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def contains_words(text_words: tuple[str, ...], alias_words: tuple[str, ...]) -> bool:
    width = len(alias_words)
    if width == 0 or width > len(text_words):
        return False
    return any(
        text_words[index : index + width] == alias_words
        for index in range(len(text_words) - width + 1)
    )


def parse_response(text: str, row: dict[str, Any]) -> dict[str, Any]:
    words = token_words(text)
    ambiguous_hits = [
        alias
        for alias in row["ambiguous_aliases_private"]
        if contains_words(words, token_words(alias))
    ]
    matched_aliases: dict[str, list[str]] = {}
    for raw_class, aliases in row["raw_response_aliases_private"].items():
        hits = [
            alias
            for alias in aliases
            if contains_words(words, token_words(alias))
        ]
        if hits:
            matched_aliases[raw_class] = hits
    matched_classes = sorted(matched_aliases)
    if ambiguous_hits or len(matched_classes) > 1:
        return {
            "semantic_class": "ambiguous",
            "raw_response_class": None,
            "decoded_semantic_state": None,
            "matched_raw_classes": matched_classes,
            "matched_aliases_private": matched_aliases,
            "ambiguous_alias_hits_private": ambiguous_hits,
        }
    if not matched_classes:
        return {
            "semantic_class": "unparsed",
            "raw_response_class": None,
            "decoded_semantic_state": None,
            "matched_raw_classes": [],
            "matched_aliases_private": {},
            "ambiguous_alias_hits_private": [],
        }
    raw_class = matched_classes[0]
    decoded = row["raw_class_to_semantic_state_private"][raw_class]
    semantic_class = (
        "allowed"
        if decoded == row["target_semantic_state_private"]
        else "rejected"
    )
    return {
        "semantic_class": semantic_class,
        "raw_response_class": raw_class,
        "decoded_semantic_state": decoded,
        "matched_raw_classes": matched_classes,
        "matched_aliases_private": matched_aliases,
        "ambiguous_alias_hits_private": [],
    }


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    final_parse = parse_response(row["generated_text_clean_private"], row)
    first_registered: int | None = None
    first_allowed: int | None = None
    first_boundary: int | None = None
    for step in row["step_ledger_private"]:
        prefix = step["decoded_prefix_private"]
        parsed = parse_response(prefix, row)
        if first_registered is None and parsed["raw_response_class"] is not None:
            first_registered = int(step["step"])
        if first_allowed is None and parsed["semantic_class"] == "allowed":
            first_allowed = int(step["step"])
        if (
            first_registered is not None
            and first_boundary is None
            and re.search(r"[.!?](?:\s|$)", prefix)
        ):
            first_boundary = int(step["step"])
    numeric_valid = (
        row["all_generated_raw_logits_valid"]
        and row["all_generated_processed_scores_valid"]
    )
    return {
        **row,
        **final_parse,
        "runtime_numeric_status": "valid" if numeric_valid else "invalid",
        "tau_registered_response_private": first_registered,
        "tau_allowed_response_private": first_allowed,
        "tau_boundary_private": first_boundary,
        "registered_response_observed": first_registered is not None,
        "allowed_response_observed": first_allowed is not None,
        "boundary_observed": first_boundary is not None,
        "stop_observed": row["eos_observed"],
        "registered_response_right_censored": first_registered is None,
        "allowed_response_right_censored": first_allowed is None,
        "boundary_right_censored": first_boundary is None,
        "stop_right_censored": not row["eos_observed"],
    }


def response_map(
    rows: list[dict[str, Any]], family: str
) -> tuple[dict[str, str] | None, bool, bool]:
    states = STATE_IDS[family]
    by_state = {row["state_id_private"]: row for row in rows}
    if set(by_state) != set(states) or len(by_state) != len(rows):
        return None, False, False
    mapping: dict[str, str] = {}
    label_aligned = True
    for state in states:
        row = by_state[state]
        if row["runtime_numeric_status"] != "valid":
            return None, False, False
        raw_class = row["raw_response_class"]
        if raw_class is None:
            return None, False, False
        mapping[state] = raw_class
        label_aligned = label_aligned and row["semantic_class"] == "allowed"
    separates = len(set(mapping.values())) == len(states)
    return mapping, separates, label_aligned


def derive_interface_map(
    source_state_map: dict[str, str], target_state_map: dict[str, str]
) -> dict[str, str] | None:
    if set(source_state_map) != set(target_state_map):
        return None
    result = {
        source_state_map[state]: target_state_map[state]
        for state in source_state_map
    }
    if len(result) != len(source_state_map):
        return None
    return result


def response_signature(maps: dict[str, dict[str, str]]) -> str:
    return json.dumps(maps, sort_keys=True, separators=(",", ":"))


def group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Phase408 empty group")
    condition_maps: dict[str, dict[str, str] | None] = {}
    condition_separation: dict[str, bool] = {}
    condition_label_alignment: dict[str, bool] = {}
    for lexical in (0, 1):
        for surface in ("r000", "r001", "r002", "r003"):
            for interface in INTERFACES[family]:
                key = f"lex{lexical}__{surface}__{interface}"
                selected = [
                    row
                    for row in rows
                    if row["lexical_replica_private"] == lexical
                    and row["surface_id_private"] == surface
                    and row["interface_private"] == interface
                ]
                mapping, separates, aligned = response_map(selected, family)
                condition_maps[key] = mapping
                condition_separation[key] = separates
                condition_label_alignment[key] = aligned

    stable_maps: dict[str, dict[str, str]] = {}
    interface_stability: dict[str, bool] = {}
    for interface in INTERFACES[family]:
        candidates = [
            condition_maps[f"lex{lexical}__{surface}__{interface}"]
            for lexical in (0, 1)
            for surface in ("r000", "r001", "r002", "r003")
        ]
        complete = all(item is not None for item in candidates)
        stable = complete and all(item == candidates[0] for item in candidates[1:])
        interface_stability[interface] = bool(stable)
        if stable:
            stable_maps[interface] = candidates[0]  # type: ignore[assignment]

    observed_pair_maps: dict[str, dict[str, str]] = {}
    covariance = len(stable_maps) == len(INTERFACES[family])
    if covariance:
        for source in INTERFACES[family]:
            for target in INTERFACES[family]:
                key = f"{source}__to__{target}"
                observed = derive_interface_map(stable_maps[source], stable_maps[target])
                if observed is None:
                    covariance = False
                    continue
                observed_pair_maps[key] = observed
                covariance = covariance and observed == interface_coordinate_map(
                    family, source, target
                )

    cycle_consistent = covariance
    if cycle_consistent:
        for source in INTERFACES[family]:
            for middle in INTERFACES[family]:
                for target in INTERFACES[family]:
                    first = observed_pair_maps[f"{source}__to__{middle}"]
                    second = observed_pair_maps[f"{middle}__to__{target}"]
                    direct = observed_pair_maps[f"{source}__to__{target}"]
                    composed = {key: second[value] for key, value in first.items()}
                    if composed != direct:
                        cycle_consistent = False

    all_numeric_valid = all(
        row["runtime_numeric_status"] == "valid" for row in rows
    )
    separation_count = sum(condition_separation.values())
    label_alignment_count = sum(condition_label_alignment.values())
    partition_pass = (
        separation_count == len(condition_separation)
        and len(stable_maps) == len(INTERFACES[family])
    )
    functional_pass = (
        all_numeric_valid and partition_pass and covariance and cycle_consistent
    )
    return {
        "model": rows[0]["model"],
        "family_id": family,
        "split": rows[0]["split"],
        "anonymous_parallel_group_id": rows[0]["anonymous_parallel_group_id"],
        "group_priority": rows[0]["group_priority"],
        "case_count": len(rows),
        "semantic_class_counts": dict(
            Counter(row["semantic_class"] for row in rows)
        ),
        "runtime_numeric_status_counts": dict(
            Counter(row["runtime_numeric_status"] for row in rows)
        ),
        "condition_count": len(condition_maps),
        "condition_separation_count": separation_count,
        "condition_label_alignment_count": label_alignment_count,
        "stable_interface_response_map_count": len(stable_maps),
        "interface_stability": interface_stability,
        "stable_state_maps_by_interface": stable_maps,
        "observed_pairwise_interface_maps": observed_pair_maps,
        "all_numeric_valid": all_numeric_valid,
        "partition_pass": partition_pass,
        "task_coordinate_covariance_pass": covariance,
        "observed_cycle_consistency_pass": cycle_consistent,
        "functional_partition_interface_pass": functional_pass,
        "response_signature": response_signature(stable_maps) if functional_pass else None,
        "claim_boundary": {
            "condition_separation_is_internal_information_measure": False,
            "cycle_consistency_is_independent_when_maps_share_state_rows": False,
            "task_coordinate_covariance_is_internal_operator": False,
        },
    }


def prior_signature(
    stage: str, model: str, family: str
) -> dict[str, dict[str, str]] | None:
    prior_name = {
        "calibration": "phase408_discovery_analysis.json",
        "behavioral_holdout": "phase408_calibration_analysis.json",
    }.get(stage)
    if prior_name is None:
        return None
    path = OUT / prior_name
    if not path.is_file():
        return None
    payload = read_json(path)
    for row in payload.get("model_family_audits", []):
        if row["model"] == model and row["family_id"] == family:
            return row.get("candidate_state_maps_by_interface")
    return None


def model_family_audit(
    groups: list[dict[str, Any]],
    model: str,
    family: str,
    stage: str,
) -> dict[str, Any]:
    selected = [
        row for row in groups if row["model"] == model and row["family_id"] == family
    ]
    expected_groups = 12 if stage == "discovery" else 6
    required_groups = 9 if stage == "discovery" else 5
    signatures = Counter(
        row["response_signature"]
        for row in selected
        if row["functional_partition_interface_pass"]
        and row["response_signature"] is not None
    )
    prior = prior_signature(stage, model, family)
    prior_serialized = response_signature(prior) if prior is not None else None
    if stage == "discovery":
        candidate_serialized, candidate_count = (
            signatures.most_common(1)[0] if signatures else (None, 0)
        )
    else:
        candidate_serialized = prior_serialized
        candidate_count = signatures.get(prior_serialized, 0)
    candidate_maps = (
        json.loads(candidate_serialized) if candidate_serialized is not None else None
    )
    gate = (
        len(selected) == expected_groups
        and candidate_count >= required_groups
        and candidate_maps is not None
    )
    return {
        "model": model,
        "family_id": family,
        "stage": stage,
        "group_count": len(selected),
        "functional_group_pass_count": sum(
            row["functional_partition_interface_pass"] for row in selected
        ),
        "label_aligned_group_count": sum(
            row["condition_label_alignment_count"] == row["condition_count"]
            for row in selected
        ),
        "candidate_signature_group_count": candidate_count,
        "required_candidate_signature_group_count": required_groups,
        "candidate_state_maps_by_interface": candidate_maps,
        "model_family_partition_candidate": gate,
    }


def main(stage: str) -> None:
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        complete_path = OUT / "collection" / stage / model / "complete.json"
        if not complete_path.is_file():
            raise RuntimeError(f"Phase408 missing collection marker: {model}/{stage}")
        complete = read_json(complete_path)
        if not complete.get("valid"):
            raise RuntimeError(f"Phase408 invalid collection: {model}/{stage}")
        all_rows.extend(
            read_jsonl(
                OUT / "collection" / stage / "private" / model / "rows.jsonl"
            )
        )
    enriched = [enrich_row(row) for row in all_rows]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in enriched:
        key = (
            row["model"],
            row["family_id"],
            row["anonymous_parallel_group_id"],
        )
        grouped.setdefault(key, []).append(row)
    group_rows = [
        group_audit(rows, key[1]) for key, rows in sorted(grouped.items())
    ]
    model_family_rows = [
        model_family_audit(group_rows, model, family, stage)
        for model in MODELS
        for family in FAMILIES
        if any(
            row["model"] == model and row["family_id"] == family
            for row in group_rows
        )
    ]
    strict_candidates = [
        family
        for family in FAMILIES
        if all(
            any(
                row["model"] == model
                and row["family_id"] == family
                and row["model_family_partition_candidate"]
                for row in model_family_rows
            )
            for model in MODELS
        )
    ]
    glm_pair_candidates = [
        family
        for family in FAMILIES
        if any(
            row["model"] == "glm4"
            and row["family_id"] == family
            and row["model_family_partition_candidate"]
            for row in model_family_rows
        )
        and sum(
            row["family_id"] == family
            and row["model_family_partition_candidate"]
            for row in model_family_rows
        )
        >= 2
    ]
    semantic_counts = Counter(row["semantic_class"] for row in enriched)
    runtime_counts = Counter(row["runtime_numeric_status"] for row in enriched)
    payload = {
        "schema_version": "82.3.0",
        "phase_id": f"Phase408-{stage.title().replace('_', '')}Analysis",
        "created_at": now(),
        "stage": stage,
        "case_count": len(enriched),
        "group_count": len(group_rows),
        "semantic_class_counts": dict(semantic_counts),
        "runtime_numeric_status_counts": dict(runtime_counts),
        "registered_response_observed_count": sum(
            row["registered_response_observed"] for row in enriched
        ),
        "allowed_response_observed_count": sum(
            row["allowed_response_observed"] for row in enriched
        ),
        "boundary_observed_count": sum(row["boundary_observed"] for row in enriched),
        "stop_observed_count": sum(row["stop_observed"] for row in enriched),
        "H48_right_edge_count": sum(row["H48_right_edge_reached"] for row in enriched),
        "condition_separation_pass_count": sum(
            row["condition_separation_count"] == row["condition_count"]
            for row in group_rows
        ),
        "surface_lexical_stability_pass_count": sum(
            row["stable_interface_response_map_count"] == 3 for row in group_rows
        ),
        "task_coordinate_covariance_pass_count": sum(
            row["task_coordinate_covariance_pass"] for row in group_rows
        ),
        "functional_group_pass_count": sum(
            row["functional_partition_interface_pass"] for row in group_rows
        ),
        "strict_crossmodel_partition_candidate_families": strict_candidates,
        "glm_inclusive_pair_candidate_families": glm_pair_candidates,
        "model_family_audits": model_family_rows,
        "claim_boundary": {
            "functional_response_separation_is_internal_state_information": False,
            "task_coordinate_covariance_is_discovered_interface_operator": False,
            "cycle_consistency_is_independent_causal_evidence": False,
            "crossmodel_candidate_is_physical_invariant": False,
        },
    }
    write_jsonl(OUT / "analysis" / stage / "phase408_group_audits.jsonl", group_rows)
    write_json(OUT / f"phase408_{stage}_analysis.json", payload)
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
