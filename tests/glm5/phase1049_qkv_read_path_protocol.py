#!/usr/bin/env python3
"""Freeze held-out Q/K/V read-path reset and replay."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1047_concept_pair_confirmation_protocol as source
import phase1048_natural_attention_read_protocol as atlas


PHASE = 1049
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
ATLAS_ROOT = atlas.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1049_qkv_read_path_confirmation"
)

FROZEN_SLOTS = (2, 3)
SOURCE_SITES = ("selected_concept", "unselected_concept")
Q_SITES = ("query_nonce", "pre_output")
MAX_SOURCE_SPAN = 2
MAX_Q_SPAN = 2

CONDITIONS = {
    "selected_k_slot2": {
        "scope": "slot2",
        "channels": ("k",),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_v_slot2": {
        "scope": "slot2",
        "channels": ("v",),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_kv_slot2": {
        "scope": "slot2",
        "channels": ("k", "v"),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_kv_slot3": {
        "scope": "slot3",
        "channels": ("k", "v"),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_k_union": {
        "scope": "frozen_union",
        "channels": ("k",),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_v_union": {
        "scope": "frozen_union",
        "channels": ("v",),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "selected_kv_union": {
        "scope": "frozen_union",
        "channels": ("k", "v"),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "unselected_kv_union": {
        "scope": "frozen_union",
        "channels": ("k", "v"),
        "source_site": "unselected_concept",
        "q_sites": (),
    },
    "selected_kv_all_postsource": {
        "scope": "all_postsource",
        "channels": ("k", "v"),
        "source_site": "selected_concept",
        "q_sites": (),
    },
    "unselected_kv_all_postsource": {
        "scope": "all_postsource",
        "channels": ("k", "v"),
        "source_site": "unselected_concept",
        "q_sites": (),
    },
    "query_q_union": {
        "scope": "frozen_union",
        "channels": ("q",),
        "source_site": None,
        "q_sites": ("query_nonce",),
    },
    "preoutput_q_union": {
        "scope": "frozen_union",
        "channels": ("q",),
        "source_site": None,
        "q_sites": ("pre_output",),
    },
    "query_preoutput_q_union": {
        "scope": "frozen_union",
        "channels": ("q",),
        "source_site": None,
        "q_sites": ("query_nonce", "pre_output"),
    },
    "selected_kv_query_preoutput_q_union": {
        "scope": "frozen_union",
        "channels": ("q", "k", "v"),
        "source_site": "selected_concept",
        "q_sites": ("query_nonce", "pre_output"),
    },
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    return atlas.semantic_role(site, target)


def condition_depths(
    model_name: str,
    spec: dict[str, Any],
    prereg: dict[str, Any],
) -> list[int]:
    info = prereg["model_info"][model_name]
    scope = spec["scope"]
    if scope == "slot2":
        return [int(value) for value in info["frozen_slot_depths"]["2"]]
    if scope == "slot3":
        return [int(value) for value in info["frozen_slot_depths"]["3"]]
    if scope == "frozen_union":
        return [int(value) for value in info["frozen_union_depths"]]
    if scope == "all_postsource":
        return [int(value) for value in info["all_postsource_depths"]]
    raise ValueError(scope)


def main() -> None:
    atlas_prereg = read_json(
        ATLAS_ROOT / "protocol" / "preregistration.json"
    )
    atlas_aggregate = read_json(ATLAS_ROOT / "aggregate.json")
    decision = atlas_aggregate["automatic_next_decision"]
    if not decision["causal_confirmation_needed"]:
        raise RuntimeError("Phase1048 did not authorize causal confirmation")
    frozen = decision["frozen_bands"]
    observed = tuple(
        int(row["normalized_read_slot"]) for row in frozen
    )
    if observed != FROZEN_SLOTS or any(
        row["destination"] != "query_nonce" for row in frozen
    ):
        raise RuntimeError(f"Phase1048 frozen candidate drift: {observed}")

    targets = read_jsonl(
        ATLAS_ROOT
        / "protocol"
        / "reserved_confirmation_targets.jsonl"
    )
    for index, row in enumerate(targets):
        row["confirmation_index"] = index
    if len(targets) != 120:
        raise RuntimeError("Phase1049 target count drift")
    discovery_indices = {
        int(row["target_index"])
        for row in read_jsonl(
            ATLAS_ROOT / "protocol" / "discovery_targets.jsonl"
        )
    }
    if discovery_indices.intersection(
        int(row["target_index"]) for row in targets
    ):
        raise RuntimeError("Phase1048/1049 target overlap")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
    needed = {
        int(row[key])
        for row in targets
        for key in ("target_case_index", "cross_family_case_index")
    }
    model_audits = {}
    model_info = {}
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    for model_name in MODELS:
        source_cases = read_jsonl(
            ATLAS_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        cases = [
            row for row in source_cases
            if int(row["case_index"]) in needed
        ]
        cases.sort(key=lambda row: int(row["case_index"]))
        if {int(row["case_index"]) for row in cases} != needed:
            raise RuntimeError(f"{model_name} confirmation cases drift")
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", cases
        )
        lookup = {int(row["case_index"]): row for row in cases}
        failures = []
        for target in targets:
            target_case = lookup[int(target["target_case_index"])]
            donor_case = lookup[int(target["cross_family_case_index"])]
            for site in SOURCE_SITES:
                role = semantic_role(site, target)
                target_span = target_case["anchor_spans"][role]
                donor_span = donor_case["anchor_spans"][role]
                lengths = (
                    int(target_span[1]) - int(target_span[0]) + 1,
                    int(donor_span[1]) - int(donor_span[0]) + 1,
                )
                if (
                    lengths[0] != lengths[1]
                    or lengths[0] > MAX_SOURCE_SPAN
                ):
                    failures.append({
                        "target_index": int(target["target_index"]),
                        "site": site,
                        "lengths": lengths,
                    })
            for site in Q_SITES:
                start, end = (
                    int(value)
                    for value in target_case["anchor_spans"][site]
                )
                if end - start + 1 > MAX_Q_SPAN:
                    failures.append({
                        "target_index": int(target["target_index"]),
                        "site": site,
                        "length": end - start + 1,
                    })

        n_layers = int(
            atlas_prereg["model_info"][model_name]["n_layers"]
        )
        source_depth = int(
            source_prereg["model_depths"][model_name]["source_depth"]
        )
        frozen_slot_depths = {
            str(slot): [
                int(value)
                for value in atlas_prereg["model_info"][model_name][
                    "read_depth_bands"
                ][str(slot)]
            ]
            for slot in FROZEN_SLOTS
        }
        frozen_union = sorted({
            depth
            for values in frozen_slot_depths.values()
            for depth in values
        })
        all_postsource = list(range(source_depth + 1, n_layers + 1))
        model_info[model_name] = {
            "n_layers": n_layers,
            "source_depth": source_depth,
            "frozen_slot_depths": frozen_slot_depths,
            "frozen_union_depths": frozen_union,
            "all_postsource_depths": all_postsource,
        }
        checks = {
            "target_count_120": len(targets) == 120,
            "query_pairs_60": len({
                int(row["query_pair_index"]) for row in targets
            }) == 60,
            "all_cases_present": bool(cases),
            "all_spans_valid": not failures,
            "all_frozen_depths_postsource": all(
                depth > source_depth for depth in frozen_union
            ),
            "candidate_ids_constant": len({
                tuple(row["candidate_token_ids"]) for row in cases
            }) == 1,
        }
        model_audits[model_name] = {
            "model": model_name,
            "case_count": len(cases),
            "failures": failures,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }

    payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": source.PHASE,
        "atlas_phase": atlas.PHASE,
        "atlas_protocol_digest": atlas_aggregate["protocol_digest"],
        "models": MODELS,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": MODELS,
        "frozen_slots": FROZEN_SLOTS,
        "conditions": CONDITIONS,
        "model_info": model_info,
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1049_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "research_question": (
            "Do source-position K and V projections in the naturally "
            "selected read bands causally mediate the early fact-state "
            "effect, and is that effect specific to the retrospectively "
            "selected fact rather than the other fact?"
        ),
        "sample_plan": {
            "heldout_query_pairs": 60,
            "targets": len(targets),
            "conditions": len(CONDITIONS),
            "baseline_paired_rows_per_model": len(targets) * 2,
            "intervention_paired_rows_per_model": (
                len(targets) * len(CONDITIONS) * 2
            ),
        },
        "intervention_semantics": {
            "source_edit": (
                "At the frozen early layer output, replace the full aligned "
                "selected-concept state with the cross-family donor state "
                "in row 0; row 1 receives exactly zero payload."
            ),
            "projection_cache": (
                "First cache the unmodified source-edit and zero trajectories "
                "at real q_proj, k_proj, and v_proj outputs."
            ),
            "reset_replay": (
                "During an intervention run, row 0 receives the cached zero "
                "projection at only the preregistered positions/layers and "
                "row 1 receives the cached source-edit projection. Every "
                "later operation naturally recomputes."
            ),
            "k_v_meaning": (
                "K tests source addressing changes; V tests source content "
                "changes; KV tests their joint projected path."
            ),
        },
        "causal_route_gate": {
            "source_shift_median_min": 0.0,
            "source_positive_rate_min": 0.8,
            "blocked_positive_rate_min": 0.65,
            "mediation_fraction_median_min": 0.1,
            "replay_positive_rate_min": 0.65,
            "replay_recovery_median_min": 0.1,
            "selected_minus_unselected_mediation_min": 0.1,
            "minimum_models": 2,
        },
        "automatic_followup": {
            "if_frozen_union_passes": (
                "Localize contributing head groups on new material and test "
                "full-vocabulary natural next-token and short rollout."
            ),
            "if_only_all_postsource_passes": (
                "Run a cumulative depth-boundary localization before any "
                "head claim."
            ),
            "if_neither_passes": (
                "Stop this Attention read-path block. Preserve the natural "
                "atlas as descriptive evidence only."
            ),
        },
        "claim_limits": [
            "Projection reset/replay uses an artificial early full-state "
            "source edit in a controlled lookup task.",
            "Band-level mediation does not identify a single head or neuron.",
            "K and V effects can interact nonlinearly and need not add.",
            "Candidate logits and full-vocabulary top-1 are not a multi-token "
            "natural rollout.",
            "No result proves biological optimality, a universal language "
            "mechanism, or a new mathematical law.",
        ],
        "model_audits": model_audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1049 protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1049_protocol_audit.v1",
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
