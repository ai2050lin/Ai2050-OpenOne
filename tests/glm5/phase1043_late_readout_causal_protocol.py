#!/usr/bin/env python3
"""Freeze Phase1043 late query-write causal confirmation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1040_expanded_mlp_replication_protocol as source
import phase1041_position_write_alliance_protocol as split_source
import phase1042_role_depth_write_atlas_protocol as atlas_source


PHASE = 1043
REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
MAX_SPAN = 3
SITE = "query_nonce"
CHANNELS = ("attention_write", "mlp_write", "layer_output")
COMPONENT_CONDITIONS = (
    "cross_matched",
    "same_lexical",
    "cross_shuffled",
    "self_zero",
)
INTERVENTIONS = (
    ("candidate", "cross_matched"),
    ("candidate", "same_lexical"),
    ("candidate", "cross_shuffled"),
    ("candidate", "self_zero"),
    ("full_state", "cross_matched"),
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1043_late_readout_causal_confirmation"
)

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def donor_case(
    target: dict[str, Any],
    condition: str,
) -> int:
    if condition == "cross_matched":
        return int(target["cross_family_case_index"])
    if condition == "same_lexical":
        return int(target["same_family_case_index"])
    if condition == "cross_shuffled":
        return int(target["shuffled_cross_case_index"])
    if condition == "self_zero":
        return int(target["target_case_index"])
    raise ValueError(condition)


def main() -> None:
    atlas_aggregate = read_json(atlas_source.OUT_ROOT / "aggregate.json")
    atlas_prereg = read_json(
        atlas_source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    frozen = atlas_aggregate["frozen_causal_candidates"]
    if len(frozen) != 3:
        raise RuntimeError("Phase1042 frozen-candidate drift")
    if any(row["site"] != SITE for row in frozen):
        raise RuntimeError("Phase1042 site drift")

    common_cases = {
        int(row["case_index"]): row
        for row in read_jsonl(
            source.OUT_ROOT / "protocol" / "cases.common.jsonl"
        )
    }
    confirmation = split_source.read_jsonl(
        split_source.OUT_ROOT
        / "protocol"
        / "reserved_confirmation_targets.jsonl"
    )
    confirmation = split_source.add_shuffled_donors(confirmation)
    targets = [
        {**row, "confirmation_index": index}
        for index, row in enumerate(confirmation)
    ]
    used = split_source.used_case_indices(targets)

    candidates = []
    for row in frozen:
        slot = int(row["normalized_depth_slot"])
        candidates.append({
            "candidate_index": len(candidates),
            "normalized_depth_slot": slot,
            "channel": row["channel"],
            "site": row["site"],
            "physical_depths": {
                model: int(
                    atlas_prereg["model_physical_depths"][model][slot - 1]
                )
                for model in MODELS
            },
            "atlas_models": list(row["models"]),
        })

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
    write_json(protocol_dir / "candidates.json", candidates)

    model_case_counts = {}
    max_spans = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        source_rows = read_jsonl(
            source.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        subset = [
            split_source.model_case(tokenizer, model, row)
            for row in source_rows
            if int(row["case_index"]) in used
        ]
        subset = [
            {
                **row,
                "schema_version": "phase1043_model_case.v1",
                "phase": PHASE,
            }
            for row in subset
        ]
        subset.sort(key=lambda row: int(row["case_index"]))
        if {int(row["case_index"]) for row in subset} != used:
            raise RuntimeError(f"{model} case subset drift")
        write_jsonl(protocol_dir / f"cases.{model}.jsonl", subset)
        model_case_counts[model] = len(subset)
        max_spans[model] = max(
            int(row["anchor_spans"][SITE][1])
            - int(row["anchor_spans"][SITE][0])
            + 1
            for row in subset
        )
        del tokenizer

    payload = {
        "phase": PHASE,
        "revision": REVISION,
        "source_phase1042_digest": atlas_aggregate["protocol_digest"],
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "confirmation_surface": 1,
        "site": SITE,
        "channels": CHANNELS,
        "candidates": candidates,
        "interventions": INTERVENTIONS,
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
        "shuffled_pairs": [
            [
                int(row["target_index"]),
                int(row["shuffled_target_index"]),
            ]
            for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1043_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "sample_plan": {
            "targets": len(targets),
            "cases_per_model": model_case_counts,
            "frozen_candidates": len(candidates),
            "interventions_per_candidate": len(INTERVENTIONS),
            "patched_rows_per_model": (
                len(targets) * len(candidates) * len(INTERVENTIONS)
            ),
        },
        "confirmation_gate": {
            "minimum_models": 2,
            "cross_positive_rate_min": 0.65,
            "all_template_stratum_medians_positive": True,
            "matched_to_same_absolute_ratio_min": 1.5,
            "matched_to_shuffled_absolute_ratio_min": 1.5,
            "full_state_retention_min": 0.05,
            "minimum_finite_rate": 0.90,
            "claim_limit": (
                "A pass establishes a repeated local causal contribution "
                "under additive component replacement. It does not prove "
                "sufficiency, minimality, or a complete language mechanism."
            ),
        },
        "intervention_semantics": {
            "attention_write": (
                "Add the cached donor-minus-target attention output at "
                "the query span before residual addition; downstream MLP "
                "recomputes naturally."
            ),
            "mlp_write": (
                "Add the cached donor-minus-target MLP output at the "
                "query span before residual addition."
            ),
            "full_state": (
                "Add cached donor-minus-target layer output at the query "
                "span as a contextual reference only."
            ),
            "self_zero": "Inject an exactly zero payload.",
        },
    }
    write_json(protocol_dir / "preregistration.json", prereg)

    group_counts = {
        f"template_{template}/{stratum}": sum(
            int(row["template_index"]) == template
            and row["surface_stratum"] == stratum
            for row in targets
        )
        for template in (0, 1)
        for stratum in source.SURFACE_STRATA
    }
    audit = {
        "schema_version": "phase1043_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "phase1042_decision_requires_confirmation": (
                atlas_aggregate["automatic_next_decision"][
                    "causal_confirmation_needed"
                ]
            ),
            "three_frozen_candidates": len(candidates) == 3,
            "confirmation_targets_240": len(targets) == 240,
            "surface_one_only": all(
                int(row["surface_index"]) == 1 for row in targets
            ),
            "balanced_60_each_group": all(
                value == 60 for value in group_counts.values()
            ),
            "strict_shuffled_disjoint": all(
                row["ordered_pair"] != row["shuffled_ordered_pair"]
                and set(str(row["ordered_pair"]).split("->")).isdisjoint(
                    str(row["shuffled_ordered_pair"]).split("->")
                )
                for row in targets
            ),
            "all_cases_present": all(
                count == len(used)
                for count in model_case_counts.values()
            ),
            "all_spans_within_three": all(
                value <= MAX_SPAN for value in max_spans.values()
            ),
            "fp16_no_quantization": (
                PRECISION == "fp16" and QUANTIZATION == "none"
            ),
        },
        "group_counts": group_counts,
        "surface_counts": dict(
            Counter(int(row["surface_index"]) for row in targets)
        ),
        "model_case_counts": model_case_counts,
        "max_query_span": max_spans,
    }
    write_json(protocol_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(prereg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
