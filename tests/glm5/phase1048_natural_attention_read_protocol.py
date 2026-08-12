#!/usr/bin/env python3
"""Freeze a natural query-conditioned Attention read atlas."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1044_natural_recompute_trajectory_protocol as trajectory
import phase1045_receiver_mediation_protocol as mediation
import phase1047_concept_pair_confirmation_protocol as source


PHASE = 1048
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
MATERIAL_ROOT = material.OUT_ROOT
SOURCE_ROOT = source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1048_natural_attention_read_atlas"
)

DESTINATIONS = ("query_nonce", "pre_output")
SOURCES = ("selected_concept", "unselected_concept")
NORMALIZED_READ_SLOTS = 6
MAX_SOURCE_SPAN = 2
MAX_DESTINATION_SPAN = 2


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    if site == "selected_concept":
        return str(target["selected_role"])
    if site == "unselected_concept":
        return str(target["unselected_role"])
    if site in DESTINATIONS:
        return site
    raise ValueError(site)


def contiguous_bands(
    first_depth: int,
    last_depth: int,
    count: int,
) -> dict[str, list[int]]:
    size = last_depth - first_depth + 1
    if size < count:
        raise ValueError("fewer post-source layers than read slots")
    result: dict[str, list[int]] = {}
    for slot in range(1, count + 1):
        begin = first_depth + ((slot - 1) * size) // count
        end = first_depth + (slot * size) // count - 1
        result[str(slot)] = list(range(begin, end + 1))
    flat = [depth for values in result.values() for depth in values]
    if flat != list(range(first_depth, last_depth + 1)):
        raise RuntimeError("depth-band partition drift")
    return result


def load_used_target_indices() -> set[int]:
    paths = (
        trajectory.OUT_ROOT / "protocol" / "targets.jsonl",
        mediation.OUT_ROOT / "protocol" / "targets.jsonl",
        source.OUT_ROOT / "protocol" / "targets.jsonl",
    )
    return {
        int(row["target_index"])
        for path in paths
        for row in read_jsonl(path)
    }


def split_untouched_units(
    targets: list[dict[str, Any]],
    units: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[int(row["unit_index"])].append(row)
    if len(grouped) != 120 or any(len(rows) != 2 for rows in grouped.values()):
        raise RuntimeError("untouched query-pair structure drift")

    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for unit_index in grouped:
        unit = units[unit_index]
        buckets[
            (
                int(unit["template_index"]),
                str(unit["surface_stratum"]),
            )
        ].append(unit_index)

    discovery_units: set[int] = set()
    confirmation_units: set[int] = set()
    for key in sorted(buckets):
        values = sorted(buckets[key])
        if len(values) != 30:
            raise RuntimeError(f"split bucket {key} has {len(values)} units")
        discovery_units.update(values[::2])
        confirmation_units.update(values[1::2])
    if len(discovery_units) != 60 or len(confirmation_units) != 60:
        raise RuntimeError("discovery/confirmation unit count drift")

    def build(selected: set[int], partition: str) -> list[dict[str, Any]]:
        result = []
        pair_index = {
            unit_index: index
            for index, unit_index in enumerate(sorted(selected))
        }
        for row in targets:
            unit_index = int(row["unit_index"])
            if unit_index not in selected:
                continue
            current = dict(row)
            current["partition"] = partition
            current["query_pair_index"] = pair_index[unit_index]
            current["atlas_index"] = len(result)
            result.append(current)
        result.sort(
            key=lambda row: (
                int(row["query_pair_index"]),
                int(row["query"]),
            )
        )
        for index, row in enumerate(result):
            row["atlas_index"] = index
        return result

    return (
        build(discovery_units, "discovery"),
        build(confirmation_units, "confirmation"),
    )


def model_audit(
    model_name: str,
    discovery: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = {int(row["case_index"]): row for row in cases}
    failures = []
    for partition, rows in (
        ("discovery", discovery),
        ("confirmation", confirmation),
    ):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for target in rows:
            grouped[int(target["query_pair_index"])].append(target)
            case = lookup[int(target["target_case_index"])]
            for source_site in SOURCES:
                role = semantic_role(source_site, target)
                start, end = (
                    int(value) for value in case["anchor_spans"][role]
                )
                length = end - start + 1
                if length < 1 or length > MAX_SOURCE_SPAN:
                    failures.append({
                        "partition": partition,
                        "target_index": int(target["target_index"]),
                        "reason": f"{role} length {length}",
                    })
            for destination in DESTINATIONS:
                start, end = (
                    int(value)
                    for value in case["anchor_spans"][destination]
                )
                length = end - start + 1
                if length < 1 or length > MAX_DESTINATION_SPAN:
                    failures.append({
                        "partition": partition,
                        "target_index": int(target["target_index"]),
                        "reason": f"{destination} length {length}",
                    })
        for pair_index, pair in grouped.items():
            if sorted(int(row["query"]) for row in pair) != [0, 1]:
                failures.append({
                    "partition": partition,
                    "pair_index": pair_index,
                    "reason": "query pair is incomplete",
                })
                continue
            pair_cases = [
                lookup[int(row["target_case_index"])]
                for row in sorted(pair, key=lambda row: int(row["query"]))
            ]
            if len(pair_cases[0]["input_ids"]) != len(pair_cases[1]["input_ids"]):
                failures.append({
                    "partition": partition,
                    "pair_index": pair_index,
                    "reason": "query pair token lengths differ",
                })
    checks = {
        "discovery_targets_120": len(discovery) == 120,
        "confirmation_targets_120": len(confirmation) == 120,
        "discovery_pairs_60": len({
            int(row["query_pair_index"]) for row in discovery
        }) == 60,
        "confirmation_pairs_60": len({
            int(row["query_pair_index"]) for row in confirmation
        }) == 60,
        "all_spans_valid": not failures,
        "all_cases_present": bool(cases),
        "candidate_ids_constant": len({
            tuple(row["candidate_token_ids"]) for row in cases
        }) == 1,
    }
    return {
        "model": model_name,
        "case_count": len(cases),
        "failures": failures,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    all_targets = read_jsonl(
        MATERIAL_ROOT / "protocol" / "targets.jsonl"
    )
    units_list = read_jsonl(
        MATERIAL_ROOT / "protocol" / "units.jsonl"
    )
    units = {int(row["unit_index"]): row for row in units_list}
    used = load_used_target_indices()
    untouched = [
        row for row in all_targets
        if int(row["target_index"]) not in used
    ]
    if len(untouched) != 240:
        raise RuntimeError(f"expected 240 untouched targets, got {len(untouched)}")
    if any(int(units[int(row["unit_index"])]["surface_index"]) != 1 for row in untouched):
        raise RuntimeError("untouched material is not surface-1")
    discovery, confirmation = split_untouched_units(untouched, units)

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "discovery_targets.jsonl", discovery)
    write_jsonl(
        protocol_dir / "reserved_confirmation_targets.jsonl",
        confirmation,
    )

    needed_case_indices = {
        int(row[key])
        for row in discovery + confirmation
        for key in ("target_case_index", "cross_family_case_index")
    }
    model_audits = {}
    model_info = {}
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    for model_name in MODELS:
        all_cases = read_jsonl(
            MATERIAL_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        cases = [
            row for row in all_cases
            if int(row["case_index"]) in needed_case_indices
        ]
        cases.sort(key=lambda row: int(row["case_index"]))
        if {int(row["case_index"]) for row in cases} != needed_case_indices:
            raise RuntimeError(f"{model_name} case subset drift")
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", cases
        )
        model_audits[model_name] = model_audit(
            model_name, discovery, confirmation, cases
        )
        source_summary = read_json(
            SOURCE_ROOT / "atlas" / model_name / "summary.json"
        )
        n_layers = int(source_summary["model_info"]["n_layers"])
        source_depth = int(
            source_prereg["model_depths"][model_name]["source_depth"]
        )
        model_info[model_name] = {
            "n_layers": n_layers,
            "source_depth": source_depth,
            "read_depth_bands": contiguous_bands(
                source_depth + 1,
                n_layers,
                NORMALIZED_READ_SLOTS,
            ),
        }

    payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": source.PHASE,
        "source_protocol_digest": read_json(
            SOURCE_ROOT / "aggregate.json"
        )["protocol_digest"],
        "models": MODELS,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": MODELS,
        "destinations": DESTINATIONS,
        "sources": SOURCES,
        "normalized_read_slots": NORMALIZED_READ_SLOTS,
        "model_info": model_info,
        "discovery_target_indices": [
            int(row["target_index"]) for row in discovery
        ],
        "confirmation_target_indices": [
            int(row["target_index"]) for row in confirmation
        ],
    }
    prereg = {
        "schema_version": "phase1048_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "research_question": (
            "When the natural query switches between two facts, which "
            "Attention heads follow the currently answer-relevant concept "
            "source at both query directions, and do their A-times-V "
            "contributions exceed the nonselected-source control?"
        ),
        "sample_plan": {
            "untouched_surface_1_targets": len(untouched),
            "discovery_query_pairs": 60,
            "discovery_targets": len(discovery),
            "reserved_confirmation_query_pairs": 60,
            "reserved_confirmation_targets": len(confirmation),
        },
        "measurement_definitions": {
            "attention_mass": (
                "Sum of a head's causal Attention probabilities from one "
                "destination anchor to the full selected or unselected "
                "concept span. For a multi-token query anchor, the final "
                "query token is the destination."
            ),
            "av_contribution_norm": (
                "L2 norm of the pre-o_proj head vector sum_s A[d,s]V[s] "
                "over the concept span. It is a descriptive physical "
                "contribution, not yet a causal transport edge."
            ),
            "query_pair_symmetry": (
                "For each fact pair, the selected source changes when the "
                "query changes. A cell is scored by the weaker of its two "
                "selected-minus-unselected responses."
            ),
        },
        "descriptive_head_gate": {
            "pair_min_attention_advantage_median_min": 0.0,
            "pair_min_attention_advantage_positive_rate_min": 0.65,
            "pair_min_av_log_ratio_median_min": 0.048790164,
            "pair_min_av_ratio_positive_rate_min": 0.65,
            "minimum_finite_pair_rate": 0.95,
        },
        "cross_model_band_gate": {
            "minimum_passing_heads_per_model": 1,
            "minimum_models": 2,
            "maximum_frozen_bands": 2,
        },
        "automatic_followup": {
            "if_repeated_band": (
                "Use only the reserved query pairs for a causal K, V, KV, "
                "and destination-Q reset/replay test over the frozen "
                "functional depth band. Include selected/unselected and "
                "all-post-source controls."
            ),
            "if_no_repeated_band": (
                "Preserve the natural read atlas and stop. Do not infer a "
                "read path from Attention weights alone."
            ),
        },
        "claim_limits": [
            "Natural Attention and A-times-V measurements are descriptive.",
            "Selected is a retrospective label defined by the later query; "
            "the earlier concept token cannot see that future query.",
            "A passing head does not establish necessity, sufficiency, or "
            "a cross-model neuron correspondence.",
            "The material is a controlled two-fact family lookup task.",
            "No biological optimality or new mathematical law is inferred.",
        ],
        "model_audits": model_audits,
        "untouched_distribution": dict(Counter(
            (
                f"t{units[int(row['unit_index'])]['template_index']}/"
                f"{row['surface_stratum']}/q{row['query']}"
            )
            for row in untouched
        )),
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1048 protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1048_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": True,
    })
    print(
        f"Phase{PHASE} protocol frozen: {prereg['protocol_digest']}"
    )


if __name__ == "__main__":
    main()
