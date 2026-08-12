#!/usr/bin/env python3
"""Freeze the Phase1041 multi-position current-write alliance protocol.

The protocol reuses the independently expanded Phase1040 material, but uses a
disjoint nonce surface for discovery. It does not search arbitrary neurons or
fit a mechanism equation. It asks whether preregistered position coalitions
carry more specific causal effect than their singleton components.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as source


PHASE = 1041
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1041_position_write_alliance"
)
DISCOVERY_SURFACE = 0
CONFIRMATION_SURFACE = 1
HOLDOUT_SURFACE = 2
MAX_ROLE_SPAN = 3
MAX_PATCH_TOKENS = 14

ROLE_ORDER = (
    "definition_nonce_a",
    "concept_a",
    "definition_nonce_b",
    "concept_b",
    "query_nonce",
    "pre_output",
)
CACHE_CHANNELS = (
    "attention_write",
    "mlp_write",
    "layer_output",
)
PATCH_MODES = (
    "attention_write",
    "mlp_write",
    "current_write",
    "full_state",
)
CONDITIONS_BY_MODE = {
    "attention_write": ("cross_matched",),
    "mlp_write": (
        "cross_matched",
        "same_lexical",
        "cross_shuffled",
    ),
    "current_write": (
        "cross_matched",
        "same_lexical",
        "cross_shuffled",
    ),
    "full_state": ("cross_matched",),
}
INTERVENTIONS = tuple(
    (mode, condition)
    for mode in PATCH_MODES
    for condition in CONDITIONS_BY_MODE[mode]
)

# Sites are semantic roles. Their physical a/b slot is resolved per target.
POSITION_MASKS = {
    "selected_concept": ("selected_concept",),
    "selected_nonce": ("selected_nonce",),
    "selected_fact": ("selected_nonce", "selected_concept"),
    "unselected_fact": ("unselected_nonce", "unselected_concept"),
    "all_facts": (
        "selected_nonce",
        "selected_concept",
        "unselected_nonce",
        "unselected_concept",
    ),
    "query_nonce": ("query_nonce",),
    "pre_output": ("pre_output",),
    "selected_fact_query": (
        "selected_nonce",
        "selected_concept",
        "query_nonce",
    ),
    "selected_fact_boundary": (
        "selected_nonce",
        "selected_concept",
        "pre_output",
    ),
    "query_boundary": ("query_nonce", "pre_output"),
    "selected_fact_query_boundary": (
        "selected_nonce",
        "selected_concept",
        "query_nonce",
        "pre_output",
    ),
    "all_facts_query_boundary": (
        "selected_nonce",
        "selected_concept",
        "unselected_nonce",
        "unselected_concept",
        "query_nonce",
        "pre_output",
    ),
}
CONSTITUENTS = {
    "selected_fact": ("selected_nonce", "selected_concept"),
    "all_facts": ("selected_fact", "unselected_fact"),
    "selected_fact_query": ("selected_fact", "query_nonce"),
    "selected_fact_boundary": ("selected_fact", "pre_output"),
    "query_boundary": ("query_nonce", "pre_output"),
    "selected_fact_query_boundary": (
        "selected_fact",
        "query_nonce",
        "pre_output",
    ),
    "all_facts_query_boundary": (
        "selected_fact",
        "unselected_fact",
        "query_nonce",
        "pre_output",
    ),
}
CANDIDATE_MASKS = tuple(CONSTITUENTS)


write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def family_index(label: str) -> int:
    return source.FAMILIES.index(label)


def selected_query(target: dict[str, Any]) -> int:
    """Choose one balanced query per pair/template/stratum for discovery."""
    return (
        int(target["template_index"])
        + source.SURFACE_STRATA.index(str(target["surface_stratum"]))
        + family_index(str(target["target_family"]))
        + family_index(str(target["cross_family"]))
    ) % 2


def enrich_targets(
    targets: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for target in targets:
        row = dict(target)
        case = cases[int(row["target_case_index"])]
        row["surface_index"] = int(case["surface_index"])
        row["query_nonce"] = str(case["query_nonce"])
        row["selected_slot"] = str(row["selected_role"]).removeprefix(
            "concept_"
        )
        row["unselected_slot"] = str(
            row["unselected_role"]
        ).removeprefix("concept_")
        enriched.append(row)
    return enriched


def split_targets(
    targets: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    discovery = [
        row for row in targets
        if int(row["surface_index"]) == DISCOVERY_SURFACE
        and int(row["query"]) == selected_query(row)
    ]
    confirmation = [
        row for row in targets
        if int(row["surface_index"]) == CONFIRMATION_SURFACE
    ]
    holdout = [
        row for row in targets
        if int(row["surface_index"]) == HOLDOUT_SURFACE
        or (
            int(row["surface_index"]) == DISCOVERY_SURFACE
            and int(row["query"]) != selected_query(row)
        )
    ]
    return discovery, confirmation, holdout


def add_shuffled_donors(
    discovery: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for target in discovery:
        candidates = [
            row for row in discovery
            if int(row["template_index"])
            == int(target["template_index"])
            and row["surface_stratum"] == target["surface_stratum"]
            and int(row["surface_index"]) == int(target["surface_index"])
            and int(row["query"]) == int(target["query"])
            and row["selected_role"] == target["selected_role"]
            and row["query_nonce"] == target["query_nonce"]
            and row["ordered_pair"] != target["ordered_pair"]
            and row["target_family"] not in {
                target["target_family"],
                target["cross_family"],
            }
            and row["cross_family"] not in {
                target["target_family"],
                target["cross_family"],
            }
        ]
        if not candidates:
            raise RuntimeError(
                f"no shuffled donor for {target['target_index']}"
            )
        candidates.sort(key=lambda row: int(row["target_index"]))
        offset = int(target["target_index"]) % len(candidates)
        shuffled = candidates[offset]
        current = dict(target)
        current.update({
            "shuffled_target_index": int(shuffled["target_index"]),
            "shuffled_cross_case_index": int(
                shuffled["cross_family_case_index"]
            ),
            "shuffled_selected_role": str(
                shuffled["selected_role"]
            ),
            "shuffled_unselected_role": str(
                shuffled["unselected_role"]
            ),
            "shuffled_ordered_pair": str(shuffled["ordered_pair"]),
        })
        result.append(current)
    return result


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    prompt = str(row["rendered_prompt"])
    nonce_a, nonce_b = source.NONCE_PAIRS[
        int(row["template_index"])
    ][int(row["surface_index"])]
    fragments = {
        "definition_nonce_a": source.fragment(
            prompt, nonce_a, occurrence="first"
        ),
        "definition_nonce_b": source.fragment(
            prompt, nonce_b, occurrence="first"
        ),
    }
    located = offset_token_spans(
        tokenizer, prompt, prompt, fragments
    )
    anchors = {
        role: [int(value) for value in row["anchor_spans"][role]]
        for role in ("concept_a", "concept_b", "query_nonce", "pre_output")
    }
    anchors.update({
        role: [int(start), int(end)]
        for role, (start, end) in located.items()
    })
    for role in ROLE_ORDER:
        start, end = anchors[role]
        length = end - start + 1
        if length < 1 or length > MAX_ROLE_SPAN:
            raise RuntimeError(
                f"{model_name} {row['case_key']} {role} length={length}"
            )
    result = dict(row)
    result["schema_version"] = "phase1041_model_case.v1"
    result["phase"] = PHASE
    result["source_phase"] = source.PHASE
    result["anchor_spans"] = anchors
    return result


def used_case_indices(targets: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for row in targets:
        result.update({
            int(row["target_case_index"]),
            int(row["same_family_case_index"]),
            int(row["cross_family_case_index"]),
            int(row["shuffled_cross_case_index"]),
        })
    return result


def group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"template_{template}/{stratum}": sum(
            int(row["template_index"]) == template
            and row["surface_stratum"] == stratum
            for row in rows
        )
        for template in (0, 1)
        for stratum in source.SURFACE_STRATA
    }


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_cases = {
        int(row["case_index"]): row
        for row in read_jsonl(
            SOURCE_ROOT / "protocol" / "cases.common.jsonl"
        )
    }
    base_targets = enrich_targets(
        read_jsonl(SOURCE_ROOT / "protocol" / "targets.jsonl"),
        source_cases,
    )
    discovery, confirmation, holdout = split_targets(base_targets)
    discovery = add_shuffled_donors(discovery)

    if (
        len(discovery) != 120
        or len(confirmation) != 240
        or len(holdout) != 360
    ):
        raise RuntimeError("target split size drift")
    if set(row["target_index"] for row in discovery).intersection(
        row["target_index"] for row in confirmation
    ):
        raise RuntimeError("discovery/confirmation overlap")
    if set(row["target_index"] for row in discovery).intersection(
        row["target_index"] for row in holdout
    ):
        raise RuntimeError("discovery/holdout overlap")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "discovery_targets.jsonl", discovery)
    write_jsonl(
        protocol_dir / "reserved_confirmation_targets.jsonl",
        confirmation,
    )
    write_jsonl(
        protocol_dir / "reserved_holdout_targets.jsonl",
        holdout,
    )
    used = used_case_indices(discovery)

    model_audits = {}
    model_case_counts = {}
    max_span_by_model: dict[str, dict[str, int]] = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        source_model_cases = read_jsonl(
            SOURCE_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        subset = [
            model_case(tokenizer, model_name, row)
            for row in source_model_cases
            if int(row["case_index"]) in used
        ]
        subset.sort(key=lambda row: int(row["case_index"]))
        if {int(row["case_index"]) for row in subset} != used:
            raise RuntimeError(f"{model_name} case subset drift")
        write_jsonl(
            protocol_dir / f"discovery_cases.{model_name}.jsonl",
            subset,
        )
        max_spans = {
            role: max(
                int(row["anchor_spans"][role][1])
                - int(row["anchor_spans"][role][0])
                + 1
                for row in subset
            )
            for role in ROLE_ORDER
        }
        max_span_by_model[model_name] = max_spans
        model_case_counts[model_name] = len(subset)
        model_audits[model_name] = {
            "case_count": len(subset),
            "max_span_by_role": max_spans,
            "all_fp16": PRECISION == "fp16",
            "no_quantization": QUANTIZATION == "none",
            "candidate_ids_stable": len({
                tuple(row["candidate_token_ids"]) for row in subset
            }) == 1,
            "special_tokens_absent": all(
                not row["special_token_ids_present"] for row in subset
            ),
        }
        del tokenizer

    intervention_count = sum(
        len(values) for values in CONDITIONS_BY_MODE.values()
    )
    protocol_payload = {
        "phase": PHASE,
        "revision": PROTOCOL_REVISION,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "discovery_surface": DISCOVERY_SURFACE,
        "discovery_query_rule": (
            "(template + stratum + target_family + donor_family) mod 2"
        ),
        "role_order": ROLE_ORDER,
        "cache_channels": CACHE_CHANNELS,
        "position_masks": POSITION_MASKS,
        "constituents": CONSTITUENTS,
        "interventions": INTERVENTIONS,
        "model_physical_depth": source_prereg[
            "model_physical_depth"
        ],
        "discovery_target_keys": [
            int(row["target_index"]) for row in discovery
        ],
        "shuffled_pairs": [
            (
                int(row["target_index"]),
                int(row["shuffled_target_index"]),
            )
            for row in discovery
        ],
    }
    preregistration = {
        "schema_version": "phase1041_preregistration.v1",
        **protocol_payload,
        "protocol_digest": digest(protocol_payload),
        "sample_plan": {
            "discovery_targets": len(discovery),
            "reserved_confirmation_targets": len(confirmation),
            "reserved_holdout_targets": len(holdout),
            "position_masks": len(POSITION_MASKS),
            "interventions_per_mask": intervention_count,
            "patched_rows_per_model": (
                len(discovery)
                * len(POSITION_MASKS)
                * intervention_count
            ),
        },
        "discovery_candidate_rule": {
            "candidate_masks": CANDIDATE_MASKS,
            "minimum_models": 2,
            "cross_positive_rate_min": 0.65,
            "all_template_stratum_medians_positive": True,
            "purity_gain_over_selected_concept_min": 0.0,
            "matched_to_shuffled_ratio_min": 1.5,
            "cross_median_gain_over_selected_concept_min": 0.0,
            "best_constituent_gain_median_min": 0.0,
            "full_state_retention_min": 0.1,
            "claim_limit": (
                "Eligibility only freezes an alliance for independent "
                "confirmation. It is not a minimum causal subgraph."
            ),
        },
        "identity_constraints": {
            "full_state_is_reference_only": True,
            "upstream_plus_attention_plus_mlp_not_ranked": True,
            "cross_depth_coalitions_deferred": True,
            "kv_cache_deferred_to_separate_intervention_semantics": True,
        },
    }
    write_json(protocol_dir / "preregistration.json", preregistration)

    audit = {
        "schema_version": "phase1041_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "checks": {
            "source_phase1040_digest_present": bool(
                source_prereg["protocol_digest"]
            ),
            "target_partition_120_240_360": (
                len(discovery) == 120
                and len(confirmation) == 240
                and len(holdout) == 360
            ),
            "discovery_balanced_30_each_group": all(
                value == 30
                for value in group_counts(discovery).values()
            ),
            "all_30_ordered_pairs_each_group": all(
                len({
                    row["ordered_pair"]
                    for row in discovery
                    if int(row["template_index"]) == template
                    and row["surface_stratum"] == stratum
                }) == 30
                for template in (0, 1)
                for stratum in source.SURFACE_STRATA
            ),
            "shuffled_pairs_disjoint": all(
                row["shuffled_ordered_pair"] != row["ordered_pair"]
                and row["shuffled_cross_case_index"]
                != row["cross_family_case_index"]
                for row in discovery
            ),
            "all_model_cases_present": set(model_case_counts)
            == set(MODELS),
            "all_spans_within_frozen_max": all(
                value <= MAX_ROLE_SPAN
                for rows in max_span_by_model.values()
                for value in rows.values()
            ),
            "fp16_no_quantization": (
                PRECISION == "fp16" and QUANTIZATION == "none"
            ),
            "no_same_layer_identity_ranked_as_discovery": True,
        },
        "group_counts": group_counts(discovery),
        "model_case_counts": model_case_counts,
        "max_span_by_model": max_span_by_model,
        "model_audits": model_audits,
        "split_surface_counts": {
            "discovery": dict(Counter(
                int(row["surface_index"]) for row in discovery
            )),
            "confirmation": dict(Counter(
                int(row["surface_index"]) for row in confirmation
            )),
            "holdout": dict(Counter(
                int(row["surface_index"]) for row in holdout
            )),
        },
    }
    write_json(protocol_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(preregistration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
