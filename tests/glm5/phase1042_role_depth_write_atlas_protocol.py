#!/usr/bin/env python3
"""Freeze a role-by-depth actual-write atlas after Phase1041."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1038_source_channel_protocol as depth_source
import phase1040_expanded_mlp_replication_protocol as material
import phase1041_position_write_alliance_protocol as alliance


PHASE = 1042
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = alliance.OUT_ROOT
MATERIAL_ROOT = material.OUT_ROOT
DEPTH_ROOT = depth_source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1042_role_depth_write_atlas"
)

SEMANTIC_SITES = (
    "selected_concept",
    "selected_nonce",
    "unselected_concept",
    "unselected_nonce",
    "query_nonce",
    "pre_output",
)
SITE_SIGNS = {
    "selected_concept": 1.0,
    "selected_nonce": 1.0,
    "unselected_concept": -1.0,
    "unselected_nonce": -1.0,
    "query_nonce": 1.0,
    "pre_output": 1.0,
}
CHANNELS = (
    "upstream_residual",
    "attention_write",
    "mlp_write",
    "layer_output",
)
WORLD_ORDER = ("b0l0", "b1l0", "b0l1", "b1l1")
MAX_SPAN = alliance.MAX_ROLE_SPAN


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    selected = str(target["selected_slot"])
    unselected = str(target["unselected_slot"])
    return {
        "selected_concept": f"concept_{selected}",
        "selected_nonce": f"definition_nonce_{selected}",
        "unselected_concept": f"concept_{unselected}",
        "unselected_nonce": f"definition_nonce_{unselected}",
        "query_nonce": "query_nonce",
        "pre_output": "pre_output",
    }[site]


def common_lookup() -> dict[tuple[int, int, int, int], int]:
    return {
        (
            int(row["unit_index"]),
            int(row["binding"]),
            int(row["query"]),
            int(row["lexical"]),
        ): int(row["case_index"])
        for row in read_jsonl(
            MATERIAL_ROOT / "protocol" / "cases.common.jsonl"
        )
    }


def atlas_targets() -> list[dict[str, Any]]:
    targets = read_jsonl(
        SOURCE_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    lookup = common_lookup()
    result = []
    for atlas_index, target in enumerate(targets):
        unit = int(target["unit_index"])
        query = int(target["query"])
        current = dict(target)
        current["atlas_index"] = atlas_index
        current["world_case_indices"] = {
            "b0l0": lookup[(unit, 0, query, 0)],
            "b1l0": lookup[(unit, 1, query, 0)],
            "b0l1": lookup[(unit, 0, query, 1)],
            "b1l1": lookup[(unit, 1, query, 1)],
        }
        result.append(current)
    by_target = {
        int(row["target_index"]): row for row in result
    }
    for row in result:
        shuffled = by_target[int(row["shuffled_target_index"])]
        row["shuffled_atlas_index"] = int(
            shuffled["atlas_index"]
        )
    return result


def model_cases(
    model_name: str,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    used = {
        int(case_index)
        for target in targets
        for case_index in target["world_case_indices"].values()
    }
    source_rows = read_jsonl(
        MATERIAL_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    tokenizer = tokenizer_for(model_name)
    rows = [
        alliance.model_case(tokenizer, model_name, row)
        for row in source_rows
        if int(row["case_index"]) in used
    ]
    del tokenizer
    rows.sort(key=lambda row: int(row["case_index"]))
    if {int(row["case_index"]) for row in rows} != used:
        raise RuntimeError(f"{model_name} atlas case subset drift")
    return rows


def main() -> None:
    phase1041 = read_json(SOURCE_ROOT / "aggregate.json")
    if phase1041["automatic_next_decision"][
        "confirmation_needed"
    ]:
        raise RuntimeError(
            "Phase1041 has a confirmation candidate; depth atlas not next"
        )
    depth_prereg = read_json(
        DEPTH_ROOT / "protocol" / "preregistration.json"
    )
    material_prereg = read_json(
        MATERIAL_ROOT / "protocol" / "preregistration.json"
    )
    targets = atlas_targets()
    if len(targets) != 120:
        raise RuntimeError("atlas target count drift")
    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)

    model_case_counts = {}
    max_spans = {}
    for model_name in MODELS:
        rows = model_cases(model_name, targets)
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", rows
        )
        model_case_counts[model_name] = len(rows)
        max_spans[model_name] = {
            site: max(
                int(row["anchor_spans"][
                    semantic_role(site, target)
                ][1])
                - int(row["anchor_spans"][
                    semantic_role(site, target)
                ][0])
                + 1
                for target in targets
                for row in (
                    next(
                        current
                        for current in rows
                        if int(current["case_index"])
                        == int(target["target_case_index"])
                    ),
                )
            )
            for site in SEMANTIC_SITES
        }

    payload = {
        "phase": PHASE,
        "revision": PROTOCOL_REVISION,
        "phase1041_digest": phase1041["protocol_digest"],
        "material_digest": material_prereg["protocol_digest"],
        "depth_source_digest": depth_prereg["protocol_digest"],
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "semantic_sites": SEMANTIC_SITES,
        "site_signs": SITE_SIGNS,
        "channels": CHANNELS,
        "world_order": WORLD_ORDER,
        "model_physical_depths": depth_prereg[
            "model_physical_depths"
        ],
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
        "shuffled_indices": [
            (
                int(row["atlas_index"]),
                int(row["shuffled_atlas_index"]),
            )
            for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1042_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "sample_plan": {
            "targets": len(targets),
            "worlds_per_target": len(WORLD_ORDER),
            "cases_per_model": len(targets) * len(WORLD_ORDER),
            "depths_per_model": 7,
            "channels": len(CHANNELS),
            "semantic_sites": len(SEMANTIC_SITES),
        },
        "descriptive_gate": {
            "same_pair_cosine_median_min": 0.2,
            "matched_minus_shuffled_median_min": 0.1,
            "advantage_positive_rate_min": 0.65,
            "family_to_lexical_norm_ratio_min": 1.0,
            "minimum_models": 2,
            "actual_write_channels": (
                "attention_write",
                "mlp_write",
            ),
            "claim_limit": (
                "A passing cell is a repeated role-depth response, not a "
                "causal path or language mechanism."
            ),
        },
        "automatic_followup": {
            "require_late_query_or_boundary_actual_write": True,
            "late_depth_slots": (4, 5, 6, 7),
            "route_if_present": (
                "Freeze a small role-depth causal confirmation using "
                "separate interventions and no exhaustive layer search."
            ),
            "route_if_absent": (
                "Preserve the atlas and stop this controlled family route; "
                "do not invent a cross-depth coalition."
            ),
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
        for stratum in material.SURFACE_STRATA
    }
    audit = {
        "schema_version": "phase1042_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "phase1041_negative_route_confirmed": not phase1041[
                "automatic_next_decision"
            ]["confirmation_needed"],
            "target_count_120": len(targets) == 120,
            "four_worlds_each": all(
                set(row["world_case_indices"]) == set(WORLD_ORDER)
                for row in targets
            ),
            "all_cases_480": all(
                value == 480 for value in model_case_counts.values()
            ),
            "balanced_30_each_group": all(
                value == 30 for value in group_counts.values()
            ),
            "seven_uniform_depths": all(
                len(values) == 7
                for values in prereg[
                    "model_physical_depths"
                ].values()
            ),
            "all_spans_at_most_three": all(
                value <= MAX_SPAN
                for rows in max_spans.values()
                for value in rows.values()
            ),
            "fp16_no_quantization": (
                PRECISION == "fp16" and QUANTIZATION == "none"
            ),
        },
        "model_case_counts": model_case_counts,
        "group_counts": group_counts,
        "max_spans": max_spans,
    }
    write_json(protocol_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(prereg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
