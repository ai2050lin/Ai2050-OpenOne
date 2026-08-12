#!/usr/bin/env python3
"""Freeze lexical holdout KV-head localization and natural validation."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1048_natural_attention_read_protocol as natural_atlas
import phase1049_qkv_read_path_protocol as causal


PHASE = 1050
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
MATERIAL_ROOT = material.OUT_ROOT
CAUSAL_ROOT = causal.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1050_head_group_natural_validation"
)
MAX_SOURCE_SPAN = 2
SOURCE_SITES = ("selected_concept", "unselected_concept")
ROLLOUT_STEPS = 3
ROLLOUT_PAIR_LIMIT = 12
GATES = {
    "source_positive_rate_min": 0.8,
    "causal_blocked_positive_rate_min": 0.6,
    "causal_mediation_median_min": 0.05,
    "causal_replay_positive_rate_min": 0.6,
    "causal_replay_median_min": 0.05,
    "selected_minus_unselected_mediation_min": 0.05,
    "natural_behavior_gated_pair_count_min": 20,
    "natural_directional_shift_positive_rate_min": 0.6,
    "natural_directional_shift_median_min": 0.0,
    "minimum_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    return natural_atlas.semantic_role(site, target)


def lexical_holdout_targets() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units = read_jsonl(MATERIAL_ROOT / "protocol" / "units.jsonl")
    cases = read_jsonl(
        MATERIAL_ROOT / "protocol" / "cases.common.jsonl"
    )
    lookup = {
        (
            int(row["unit_index"]),
            int(row["binding"]),
            int(row["query"]),
            int(row["lexical"]),
        ): row
        for row in cases
    }
    by_cell: dict[tuple[int, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for unit in units:
        by_cell[
            (
                int(unit["template_index"]),
                int(unit["unit_key"].split(".")[1][1:]),
                str(unit["surface_stratum"]),
            )
        ].append(unit)

    selected_units = []
    for cell, rows in sorted(by_cell.items()):
        by_family = {
            int(row["target_index"]): [] for row in rows
        }
        for row in rows:
            by_family[int(row["target_index"])].append(row)
        template_index, surface_index, stratum = cell
        stratum_slot = material.SURFACE_STRATA.index(stratum)
        for family_index in range(len(material.FAMILIES)):
            options = sorted(
                by_family[family_index],
                key=lambda row: int(row["donor_offset"]),
            )
            option_slot = (
                family_index
                + template_index
                + surface_index
                + stratum_slot
            ) % len(options)
            unit = dict(options[option_slot])
            unit["surface_index"] = surface_index
            unit["partition"] = (
                "discovery"
                if (
                    family_index
                    + template_index
                    + surface_index
                    + stratum_slot
                )
                % 2
                == 0
                else "confirmation"
            )
            selected_units.append(unit)

    targets = []
    needed_cases = set()
    for unit in selected_units:
        q0_slot = str(unit["q0_slot"])
        unit_index = int(unit["unit_index"])
        for query in (0, 1):
            selected_slot = (
                q0_slot
                if query == 0
                else ("b" if q0_slot == "a" else "a")
            )
            unselected_slot = "b" if selected_slot == "a" else "a"
            target_case = lookup[(unit_index, 0, query, 1)]
            cross_case = lookup[(unit_index, 1, query, 1)]
            needed_cases.update(
                (
                    int(target_case["case_index"]),
                    int(cross_case["case_index"]),
                )
            )
            targets.append({
                "schema_version": "phase1050_target.v1",
                "phase": PHASE,
                "target_index": len(targets),
                "unit_index": unit_index,
                "template_index": int(unit["template_index"]),
                "surface_index": int(unit["surface_index"]),
                "surface_stratum": str(unit["surface_stratum"]),
                "ordered_pair": (
                    f"{unit['target_family']}->{unit['donor_family']}"
                ),
                "donor_offset": int(unit["donor_offset"]),
                "query": query,
                "lexical": 1,
                "selected_role": f"concept_{selected_slot}",
                "unselected_role": f"concept_{unselected_slot}",
                "target_case_index": int(target_case["case_index"]),
                "cross_family_case_index": int(
                    cross_case["case_index"]
                ),
                "target_family_index": int(
                    target_case["expected_index"]
                ),
                "target_family": str(target_case["expected_label"]),
                "cross_family_index": int(
                    cross_case["expected_index"]
                ),
                "cross_family": str(cross_case["expected_label"]),
                "partition": str(unit["partition"]),
            })
    for index, row in enumerate(targets):
        row["atlas_index"] = index
        row["confirmation_index"] = index
    filtered_cases = [
        row for row in cases
        if int(row["case_index"]) in needed_cases
    ]
    filtered_cases.sort(key=lambda row: int(row["case_index"]))
    return targets, filtered_cases


def main() -> None:
    aggregate = read_json(CAUSAL_ROOT / "aggregate.json")
    next_decision = aggregate["automatic_next_decision"]
    if next_decision["route"] != "head_group_and_natural_rollout":
        raise RuntimeError(
            f"Phase1049 did not authorize Phase1050: {next_decision}"
        )
    causal_prereg = read_json(
        CAUSAL_ROOT / "protocol" / "preregistration.json"
    )
    targets, common_cases = lexical_holdout_targets()
    discovery = [
        row for row in targets if row["partition"] == "discovery"
    ]
    confirmation = [
        row for row in targets if row["partition"] == "confirmation"
    ]
    if len(discovery) != 120 or len(confirmation) != 120:
        raise RuntimeError(
            f"partition drift: {len(discovery)}/{len(confirmation)}"
        )

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
    write_jsonl(protocol_dir / "discovery_targets.jsonl", discovery)
    write_jsonl(
        protocol_dir / "confirmation_targets.jsonl", confirmation
    )
    write_jsonl(protocol_dir / "cases.common.jsonl", common_cases)

    model_info = {}
    model_audits = {}
    needed = {int(row["case_index"]) for row in common_cases}
    for model_name in MODELS:
        source_cases = read_jsonl(
            MATERIAL_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        cases = [
            row for row in source_cases
            if int(row["case_index"]) in needed
        ]
        cases.sort(key=lambda row: int(row["case_index"]))
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", cases
        )
        atlas_summary = read_json(
            natural_atlas.OUT_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        info = atlas_summary["model_info"]
        n_kv_heads = int(info["n_kv_heads"])
        depth_info = causal_prereg["model_info"][model_name]
        model_info[model_name] = {
            "n_layers": int(info["n_layers"]),
            "d_model": int(info["d_model"]),
            "n_heads": int(info["n_heads"]),
            "n_kv_heads": n_kv_heads,
            "source_depth": int(depth_info["source_depth"]),
            "slot_depths": depth_info["frozen_slot_depths"],
            "frozen_union_depths": depth_info[
                "frozen_union_depths"
            ],
            "candidate_kv_groups": list(range(n_kv_heads)),
            "maximum_frozen_groups": min(2, n_kv_heads),
        }
        case_lookup = {
            int(row["case_index"]): row for row in cases
        }
        failures = []
        for target in targets:
            target_case = case_lookup[
                int(target["target_case_index"])
            ]
            donor_case = case_lookup[
                int(target["cross_family_case_index"])
            ]
            for site in SOURCE_SITES:
                role = semantic_role(site, target)
                target_span = target_case["anchor_spans"][role]
                donor_span = donor_case["anchor_spans"][role]
                target_length = (
                    int(target_span[1]) - int(target_span[0]) + 1
                )
                donor_length = (
                    int(donor_span[1]) - int(donor_span[0]) + 1
                )
                if (
                    target_length != donor_length
                    or target_length > MAX_SOURCE_SPAN
                ):
                    failures.append({
                        "target_index": int(target["target_index"]),
                        "site": site,
                        "lengths": [target_length, donor_length],
                    })
        checks = {
            "all_cases_present": len(cases) == len(needed),
            "all_spans_aligned": not failures,
            "candidate_ids_constant": len({
                tuple(row["candidate_token_ids"]) for row in cases
            }) == 1,
            "candidate_count_10": all(
                len(row["candidate_token_ids"])
                == len(material.FAMILIES)
                for row in cases
            ),
        }
        model_audits[model_name] = {
            "case_count": len(cases),
            "failures": failures,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }

    balance = {
        partition: {
            "template": dict(Counter(
                int(row["template_index"])
                for row in targets
                if row["partition"] == partition
            )),
            "surface": dict(Counter(
                int(row["surface_index"])
                for row in targets
                if row["partition"] == partition
            )),
            "stratum": dict(Counter(
                str(row["surface_stratum"])
                for row in targets
                if row["partition"] == partition
            )),
            "family": dict(Counter(
                str(row["target_family"])
                for row in targets
                if row["partition"] == partition
            )),
        }
        for partition in ("discovery", "confirmation")
    }
    payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": causal.PHASE,
        "source_protocol_digest": causal_prereg["protocol_digest"],
        "models": MODELS,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": MODELS,
        "target_count": len(targets),
        "discovery_target_indices": [
            int(row["target_index"]) for row in discovery
        ],
        "confirmation_target_indices": [
            int(row["target_index"]) for row in confirmation
        ],
        "model_info": model_info,
        "confirmation_conditions": (
            "selected_top1",
            "unselected_top1",
            "selected_top2",
            "unselected_top2",
            "selected_complement_top1",
        ),
        "natural_conditions": (
            "selected_top1",
            "unselected_top1",
            "selected_top2",
        ),
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "gates": GATES,
    }
    prereg = {
        "schema_version": "phase1050_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "research_question": (
            "Which physical KV head groups carry the frozen-band portion "
            "of the early selected-fact effect, and do those groups also "
            "transport a natural lexical counterfactual on held-out "
            "lexical realizations?"
        ),
        "material_design": {
            "lexical_value": 1,
            "discovery_targets": len(discovery),
            "confirmation_targets": len(confirmation),
            "common_case_count": len(common_cases),
            "partition_balance": balance,
            "independence_limit": (
                "The lexical-1 cases did not enter Phase1048/1049 read-path "
                "selection, but appeared as same-family controls in older "
                "Phase1040-derived work; this is a lexical holdout, not a "
                "new task-family holdout."
            ),
        },
        "group_discovery": {
            "scan": (
                "Reset/replay K and V slices for each physical KV head "
                "group across the frozen slot-2 plus slot-3 depth union."
            ),
            "rank_score": (
                "minimum of median mediation fraction and median replay "
                "recovery; nonpositive scores remain exploratory."
            ),
            "maximum_frozen_groups_per_model": 2,
        },
        "confirmation_conditions": payload[
            "confirmation_conditions"
        ],
        "natural_validation": {
            "operation": (
                "Run the clean lexical target and its clean binding-flipped "
                "counterfactual as a pair, then exchange only frozen-group "
                "K/V slices at the selected or control source span."
            ),
            "conditions": payload["natural_conditions"],
            "full_vocabulary_top1": True,
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
            "rollout_selection": (
                "First confirmation targets, in preregistered order, for "
                "which both clean candidate predictions are correct."
            ),
        },
        "gates": payload["gates"],
        "automatic_followup": {
            "if_causal_and_natural_repeat": (
                "Freeze model-specific KV groups and replicate on a new "
                "task family with new templates and natural free output."
            ),
            "if_only_causal_repeats": (
                "Keep the head groups as an artificial source-effect route; "
                "redesign the natural counterfactual before broader claims."
            ),
            "if_neither_repeats": (
                "Retain the band-level Phase1049 result and stop physical "
                "head refinement."
            ),
        },
        "claim_limits": [
            "KV head groups are physical GQA groups, not semantic neurons.",
            "Head IDs are model-specific and are never aligned by number "
            "across architectures.",
            "The task is controlled category lookup, not unrestricted "
            "translation, punctuation, contrast, or world knowledge.",
            "Natural activation exchange is still an intervention, not "
            "ordinary unmodified generation.",
            "No result establishes biological optimality, a complete "
            "language code, or a new mathematical theory.",
        ],
        "model_audits": model_audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1050 model audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1050_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "target_count": len(targets),
        "discovery_count": len(discovery),
        "confirmation_count": len(confirmation),
        "model_audits": model_audits,
        "all_checks_passed": True,
    })
    print(
        f"Phase{PHASE} protocol frozen: {prereg['protocol_digest']}"
    )


if __name__ == "__main__":
    main()
