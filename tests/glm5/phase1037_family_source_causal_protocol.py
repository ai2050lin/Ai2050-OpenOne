#!/usr/bin/env python3
"""Freeze the Phase1037 queried-source semantic-family intervention."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1035_native_family_routing_protocol as source
import phase1036_family_contrast_protocol as evidence


PHASE = 1037
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = source.PRECISION
QUANTIZATION = source.QUANTIZATION
SOURCE_ROOT = source.OUT_ROOT
EVIDENCE_ROOT = evidence.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1037_family_source_causal"
)
DEPTH_SLOTS = (1, 4, 7)
CONDITIONS = (
    "self_selected",
    "same_family_selected",
    "cross_family_selected",
    "cross_family_unselected",
    "cross_family_wrong_target",
)


write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest


def build_targets(
    cases: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["unit_index"]),
            int(row["binding"]),
            int(row["query"]),
            int(row["lexical"]),
        ): int(row["case_index"])
        for row in cases
    }
    targets = []
    for unit in units:
        if unit["split"] != "confirmation":
            continue
        q0_slot = str(unit["q0_slot"])
        for query in (0, 1):
            selected_slot = (
                q0_slot
                if query == 0
                else ("b" if q0_slot == "a" else "a")
            )
            unselected_slot = "b" if selected_slot == "a" else "a"
            unit_index = int(unit["unit_index"])
            target_case = lookup[(unit_index, 0, query, 0)]
            same_case = lookup[(unit_index, 0, query, 1)]
            cross_case = lookup[(unit_index, 1, query, 0)]
            target = cases[target_case]
            cross = cases[cross_case]
            targets.append({
                "schema_version": "phase1037_target.v1",
                "phase": PHASE,
                "target_index": len(targets),
                "unit_index": unit_index,
                "template_index": int(unit["template_index"]),
                "query": query,
                "selected_role": f"concept_{selected_slot}",
                "unselected_role": f"concept_{unselected_slot}",
                "target_case_index": target_case,
                "same_family_case_index": same_case,
                "cross_family_case_index": cross_case,
                "target_family_index": int(target["expected_index"]),
                "target_family": str(target["expected_label"]),
                "cross_family_index": int(cross["expected_index"]),
                "cross_family": str(cross["expected_label"]),
            })
    return targets


def target_audit(
    targets: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "target_count_256": len(targets) == 256,
        "templates_confirmation_only": {
            int(row["template_index"]) for row in targets
        } == {2, 3},
        "query_balance": {
            int(value): sum(int(row["query"]) == value for row in targets)
            for value in (0, 1)
        } == {0: 128, 1: 128},
        "physical_role_balance": {
            role: sum(row["selected_role"] == role for row in targets)
            for role in ("concept_a", "concept_b")
        } == {"concept_a": 128, "concept_b": 128},
        "target_and_cross_families_differ": all(
            int(row["target_family_index"])
            != int(row["cross_family_index"])
            for row in targets
        ),
        "same_family_changes_only_lexical_member": all(
            cases[int(row["same_family_case_index"])]["lexical"] == 1
            and cases[int(row["same_family_case_index"])]["binding"] == 0
            and cases[int(row["same_family_case_index"])]["query"]
            == row["query"]
            for row in targets
        ),
        "cross_family_changes_binding_only": all(
            cases[int(row["cross_family_case_index"])]["binding"] == 1
            and cases[int(row["cross_family_case_index"])]["lexical"] == 0
            and cases[int(row["cross_family_case_index"])]["query"]
            == row["query"]
            for row in targets
        ),
    }
    return {
        "schema_version": "phase1037_target_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    evidence_prereg = read_json(
        EVIDENCE_ROOT / "protocol" / "preregistration.json"
    )
    evidence_aggregate = read_json(EVIDENCE_ROOT / "aggregate.json")
    if not evidence_aggregate["automatic_next_decision"][
        "causal_followup_needed"
    ]:
        raise RuntimeError("Phase1036 did not preregister a causal follow-up")

    cases = read_jsonl(
        SOURCE_ROOT / "protocol" / "cases.common.jsonl"
    )
    units = read_jsonl(SOURCE_ROOT / "protocol" / "units.jsonl")
    targets = build_targets(cases, units)
    audit = target_audit(targets, cases)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"target audit failed: {audit}")
    write_jsonl(OUT_ROOT / "protocol" / "targets.jsonl", targets)
    write_json(OUT_ROOT / "protocol" / "target_audit.json", audit)

    model_depths = {}
    for model in MODELS:
        selected = evidence_aggregate["model_summaries"][model][
            "selected_depths"
        ]
        model_depths[model] = [
            int(selected[index]) for index in DEPTH_SLOTS
        ]
    prereg_core: dict[str, Any] = {
        "schema_version": "phase1037_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Queried-source same-family versus cross-family causal transport"
        ),
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "evidence_phase": evidence.PHASE,
        "evidence_protocol_digest": evidence_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "target_count": len(targets),
        "confirmation_only": True,
        "depth_slot_rule": (
            "Use the earliest, middle, and latest nonfinal normalized depth "
            "slots that were conserved in Phase1036; do not select a causal "
            "peak."
        ),
        "normalized_depth_slots": list(DEPTH_SLOTS),
        "model_physical_depths": model_depths,
        "conditions": list(CONDITIONS),
        "patch_definition": (
            "At a frozen layer output, replace the complete one-token concept "
            "span with the clean donor state. No vector scaling, direction "
            "fitting, neuron ranking, or posthoc layer selection."
        ),
        "controls": {
            "self_selected": "instrument identity control",
            "same_family_selected": (
                "different lexical member of the same family at the queried "
                "physical source role"
            ),
            "cross_family_selected": (
                "opposite family at the queried physical source role"
            ),
            "cross_family_unselected": (
                "the cross-world unqueried role into the unqueried target role"
            ),
            "cross_family_wrong_target": (
                "the same queried cross-family donor state inserted into the "
                "unqueried target role"
            ),
        },
        "readouts": {
            "candidate_margin": (
                "cross-family candidate logit minus target-family candidate "
                "logit, reported relative to the unpatched clean target"
            ),
            "candidate_top1": "top family among the eight frozen candidates",
            "internal_prototype": (
                "discovery-only family prototypes at the penultimate layer; "
                "reported separately from logits"
            ),
            "strata": (
                "all rows plus clean candidate-correct and candidate-error "
                "strata; no row is deleted"
            ),
        },
        "causal_evidence_gate": {
            "cross_selected_margin_shift_median_min": 0.20,
            "selected_minus_unselected_median_min": 0.10,
            "selected_minus_wrong_target_median_min": 0.10,
            "self_absolute_margin_shift_median_max": 0.05,
            "same_family_absolute_margin_shift_median_max": 0.20,
            "both_confirmation_templates_required": True,
            "minimum_depths_per_model": 1,
            "minimum_models": 2,
        },
        "claim_limits": [
            (
                "A margin shift establishes local causal influence for this "
                "artificial definition-and-category task, not a complete "
                "knowledge graph."
            ),
            (
                "Failure of a whole-vector source patch does not prove the "
                "family representation is noncausal; the intervention may "
                "still be too coarse or incomplete."
            ),
            (
                "Success does not establish biological optimality or a "
                "universal language equation."
            ),
        ],
        "target_audit": audit,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "target_count": len(targets),
        "model_physical_depths": model_depths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
