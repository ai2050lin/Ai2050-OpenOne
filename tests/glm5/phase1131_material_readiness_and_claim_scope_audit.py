#!/usr/bin/env python3
"""Phase 1131: material-readiness and claim-scope audit.

This phase does not run a model. It checks whether a new independently
annotated material package exists, records the intake contract such a package
must satisfy, and keeps cross-model claims separate from any future
model-specific causal evidence tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "tests/glm5/result/phase1131_material_readiness_and_claim_scope"
DEFAULT_PACKAGE = RESULT_ROOT / "material/candidate_package.jsonl"

PHASE1130_FINAL = (
    REPO_ROOT
    / "tests/glm5/result/phase1130_postnumeric_route_audit/analysis/final_summary.json"
)
PHASE1130_AUDIT = (
    REPO_ROOT
    / "tests/glm5/result/phase1130_postnumeric_route_audit/audit/result_audit.json"
)

EXPECTED_PHASE1130_FINAL_DIGEST = (
    "a3a78cb20d667fedb1ee036460927f40a40642110834fd8617f2ef303a34d995"
)
EXPECTED_PHASE1130_AUDIT_DIGEST = (
    "701efb727252c1ca7ae7bffb071e7e21924db920cafdec84c6fc3f57562151e1"
)

EXTERNAL_ANALYSES = [
    {
        "source_id": "analysis_1",
        "sha256": "996519a768405a65ba8f10c9812193ae004b8365edbefbf18b2fe44b64a108c9",
        "bytes": 10224,
        "judgment": "mostly_correct_but_primitive_examples_are_not_a_material_package",
    },
    {
        "source_id": "analysis_2",
        "sha256": "82203155fd1bb4b19946b3183816b9d799539b16ae5f392040ff23baddc01f6c",
        "bytes": 14371,
        "judgment": "most_rigorous_and_correctly_identifies_material_supply_and_audit_self_reference",
    },
    {
        "source_id": "analysis_3",
        "sha256": "c580825c57699728a9158d6f35fbf8d1f9e6471c39bd360bc32167d2ced7737c",
        "bytes": 11787,
        "judgment": "correct_summary_with_unsupported_scale_bf16_adversarial_and_patching_claims",
    },
]

REUSED_MATERIAL_ROOTS = {
    "wordnet_quadrant": REPO_ROOT
    / "tests/glm5/result/phase1113_wordnet_semantic_quadrant",
    "wordnet_hypernym": REPO_ROOT
    / "tests/glm5/result/phase1114_wordnet_contextual_hypernym",
    "wordnet_confirmation": REPO_ROOT
    / "tests/glm5/result/phase1115_wordnet_context_modulation_confirmation",
    "wordnet_adjective": REPO_ROOT
    / "tests/glm5/result/phase1121_wordnet_adjective_double_orthogonal",
    "semeval_lexsub": REPO_ROOT
    / "tests/glm5/result/phase1126_semeval_lexsub_natural_cloze",
    "semeval_qwen14": REPO_ROOT
    / "tests/glm5/result/phase1127_semeval_score_anatomy_qwen14b",
}

REQUIRED_ITEM_FIELDS = [
    "item_id",
    "source_corpus",
    "source_license",
    "source_document_id",
    "primitive_family",
    "split",
    "context",
    "query",
    "active_candidate",
    "matched_null_candidate",
    "gold_answer",
    "annotator_ids",
    "annotation_blinded_to_model_outputs",
    "candidate_uniqueness_confirmed",
    "matched_null_globally_false_confirmed",
    "matched_null_locally_plausible_confirmed",
    "null_frozen_before_model_scoring",
    "same_part_of_speech",
    "surface_length_matched",
    "generation_provenance",
]

ALLOWED_SPLITS = {"discovery", "confirmation", "natural_use"}
MIN_ITEMS_PER_SPLIT = 128


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def canonical_digest(payload: dict[str, Any], digest_key: str) -> str:
    body = dict(payload)
    body.pop(digest_key, None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def all_true(rows: Iterable[dict[str, Any]], field: str) -> bool:
    values = [row.get(field) is True for row in rows]
    return bool(values) and all(values)


def audit_package(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "absent",
            "path": display_path(path),
            "row_count": 0,
            "schema_complete": False,
            "material_ready": False,
            "blockers": [
                "no_candidate_package",
                "no_independent_annotation_manifest",
                "no_prefrozen_matched_null_instances",
                "no_discovery_confirmation_natural_use_splits",
            ],
        }

    rows = read_jsonl(path)
    missing_fields = sorted(
        {
            field
            for row in rows
            for field in REQUIRED_ITEM_FIELDS
            if field not in row
        }
    )
    split_counts = Counter(str(row.get("split")) for row in rows)
    item_ids = [row.get("item_id") for row in rows]
    source_by_split: dict[str, set[str]] = {
        split: {
            str(row.get("source_document_id"))
            for row in rows
            if row.get("split") == split
        }
        for split in ALLOWED_SPLITS
    }
    source_disjoint = all(
        source_by_split[left].isdisjoint(source_by_split[right])
        for left in ALLOWED_SPLITS
        for right in ALLOWED_SPLITS
        if left < right
    )
    two_blind_annotators = bool(rows) and all(
        isinstance(row.get("annotator_ids"), list)
        and len(set(row["annotator_ids"])) >= 2
        and row.get("annotation_blinded_to_model_outputs") is True
        for row in rows
    )
    split_volume = all(split_counts[split] >= MIN_ITEMS_PER_SPLIT for split in ALLOWED_SPLITS)
    no_adaptive_test_model_generation = bool(rows) and all(
        row.get("generation_provenance") != "adaptive_to_test_model_output"
        for row in rows
    )

    checks = {
        "nonempty": bool(rows),
        "required_fields": not missing_fields,
        "unique_item_ids": len(item_ids) == len(set(item_ids)),
        "allowed_splits_only": set(split_counts).issubset(ALLOWED_SPLITS),
        "minimum_split_volume": split_volume,
        "source_documents_split_disjoint": source_disjoint,
        "two_blind_annotators": two_blind_annotators,
        "candidate_uniqueness": all_true(rows, "candidate_uniqueness_confirmed"),
        "matched_null_globally_false": all_true(
            rows, "matched_null_globally_false_confirmed"
        ),
        "matched_null_locally_plausible": all_true(
            rows, "matched_null_locally_plausible_confirmed"
        ),
        "null_prefrozen": all_true(rows, "null_frozen_before_model_scoring"),
        "same_part_of_speech": all_true(rows, "same_part_of_speech"),
        "surface_length_matched": all_true(rows, "surface_length_matched"),
        "no_adaptive_test_model_generation": no_adaptive_test_model_generation,
    }
    return {
        "status": "present",
        "path": display_path(path),
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_fields": missing_fields,
        "checks": checks,
        "schema_complete": checks["required_fields"],
        "material_ready": all(checks.values()),
        "blockers": sorted(name for name, value in checks.items() if not value),
    }


def build_protocol(phase1130: dict[str, Any]) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": "phase1131_material_readiness_and_claim_scope_protocol.v1",
        "phase": 1131,
        "analysis_type": "material_intake_and_prospective_claim_scope_audit",
        "model_execution": False,
        "phase1130_final_digest": phase1130["final_digest"],
        "external_analysis_manifest": EXTERNAL_ANALYSES,
        "material_contract": {
            "required_item_fields": REQUIRED_ITEM_FIELDS,
            "allowed_splits": sorted(ALLOWED_SPLITS),
            "minimum_independent_items_per_split": MIN_ITEMS_PER_SPLIT,
            "minimum_blind_annotators_per_item": 2,
            "model_outputs_hidden_from_annotators": True,
            "source_documents_disjoint_across_splits": True,
            "matched_null_frozen_before_model_scoring": True,
            "adaptive_generation_against_test_model_outputs": False,
            "semantic_validity_note": "Passing this machine audit checks provenance and structure only. Human semantic validity remains an external requirement.",
        },
        "claim_scope": {
            "cross_model_hidden_gate": "M_new AND A_independent AND N_prefrozen AND Q_FP16_ge_2 AND B_discovery_ge_2 AND B_confirmation_ge_2 AND S_specificity AND U_natural",
            "model_specific_causal_tier": {
                "status": "prospective_option_not_activated",
                "requirements": "M_new AND A_independent AND N_prefrozen AND Q_FP16_m AND B_discovery_m AND B_confirmation_m AND S_m AND U_m AND R_m AND N_hidden_m AND I_selective_m AND G_independent_material_m",
                "maximum_claim": "model-specific causal mechanism evidence only",
                "prohibitions": [
                    "no_cross_model_generalization_claim",
                    "no_language_universal_claim",
                    "no_reopening_wordnet_or_semeval",
                    "no_retrospective_application",
                ],
            },
        },
        "evidence_policy": {
            "new_k_item_allowed": False,
            "theory_update_allowed": False,
            "reason": "No new model or natural-language object is observed in this readiness phase.",
        },
    }
    protocol["protocol_digest"] = canonical_digest(protocol, "protocol_digest")
    return protocol


def build_final(
    protocol: dict[str, Any],
    phase1130: dict[str, Any],
    package_audit: dict[str, Any],
) -> dict[str, Any]:
    reused_roots = {
        name: {
            "exists": path.exists(),
            "classification": "previously_used_not_new_material",
            "path": str(path.relative_to(REPO_ROOT)),
        }
        for name, path in REUSED_MATERIAL_ROOTS.items()
    }
    external_judgment = {
        "shared_correct_core": [
            "Phase1130 is a zero-model route audit and adds no K item or language mechanism.",
            "Numerical disqualification removes semantic judgment authority but does not prove semantic absence.",
            "The next mainline bottleneck is independent material, annotation, and a prefrozen matched null.",
            "Existing mathematics is adequate for the current evidence and authorization logic.",
        ],
        "required_corrections": [
            "DS7B is semantically unqualified in Phase1126, not proven to have a purely numerical rather than semantic failure.",
            "K69 localizes the first recorded failure to the QK-score path; it does not isolate an unmasked all-negative-infinity row or raw matmul overflow.",
            "The 25-of-25 Phase1130 audit validates internal consistency and frozen-artifact linkage, not the scientific truth of its hard-coded route judgments.",
            "A digest detects artifact changes; it does not make governance immutable or prevent a new versioned protocol.",
            "The project has training-formation observations, but lacks a qualified natural-semantic formation and causal chain.",
            "The 4B-versus-14B knowledge and null-plausibility story is unobserved speculation.",
            "BF16 cannot be mandated under the current FP16 mainline and would not isolate architecture or scale effects.",
            "An LLM-generated adversarial null is one conditional supply route, not the only route and not proof of deep semantics.",
            "Illustrative comparison, classification, binding, transformation, and inference prompts are task ideas, not an independently annotated material package.",
        ],
    }
    final: dict[str, Any] = {
        "schema_version": "phase1131_material_readiness_and_claim_scope_final.v1",
        "phase": 1131,
        "protocol_digest": protocol["protocol_digest"],
        "phase1130_state": {
            "final_digest": phase1130["final_digest"],
            "auto_continue": phase1130["restart_decision"]["auto_continue"],
        },
        "external_analysis_judgment": external_judgment,
        "local_material_inventory": reused_roots,
        "candidate_package_audit": package_audit,
        "gate_scope_decision": {
            "cross_model_mainline_gate_unchanged": True,
            "single_model_causal_tier_activated": False,
            "reason": "The narrower tier is a reasonable prospective publication category, but activating it now would alter governance without a new material package and conflicts with the current cross-model mainline.",
        },
        "restart_decision": {
            "auto_continue": False,
            "reason": "No new material package exists, so M_new, A_independent, and N_prefrozen are false before any model is scored.",
            "model_test_authorized": False,
            "next_legal_input": "A populated candidate_package.jsonl that passes the material contract and external human semantic review.",
        },
        "evidence_update": {
            "new_k_item": None,
            "theory_update_number": None,
            "canonical_k_range": "K1-K69",
        },
        "interpretation_boundary": "This phase audits material availability and claim scope. It does not validate a material's semantics, run a model, or produce behavior, hidden-state, component, scale, precision-preference, or causal evidence.",
    }
    final["final_digest"] = canonical_digest(final, "final_digest")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-package",
        type=Path,
        default=DEFAULT_PACKAGE,
        help="JSONL package to audit; absence produces a blocked readiness result.",
    )
    args = parser.parse_args()
    package_path = args.candidate_package
    if not package_path.is_absolute():
        package_path = REPO_ROOT / package_path

    phase1130 = read_json(PHASE1130_FINAL)
    phase1130_audit = read_json(PHASE1130_AUDIT)
    package_audit = audit_package(package_path)
    protocol = build_protocol(phase1130)
    final = build_final(protocol, phase1130, package_audit)

    checks = {
        "phase1130_final_digest": phase1130["final_digest"]
        == EXPECTED_PHASE1130_FINAL_DIGEST,
        "phase1130_audit_digest": phase1130_audit["audit_digest"]
        == EXPECTED_PHASE1130_AUDIT_DIGEST,
        "phase1130_audit_passed": phase1130_audit["passed"] is True,
        "phase1130_auto_continue_false": phase1130["restart_decision"]["auto_continue"]
        is False,
        "reused_material_roots_exist": all(path.exists() for path in REUSED_MATERIAL_ROOTS.values()),
        "reused_material_not_classified_new": all(
            item["classification"] == "previously_used_not_new_material"
            for item in final["local_material_inventory"].values()
        ),
        "candidate_material_not_ready": package_audit["material_ready"] is False,
        "no_model_execution": protocol["model_execution"] is False,
        "cross_model_gate_unchanged": final["gate_scope_decision"][
            "cross_model_mainline_gate_unchanged"
        ]
        is True,
        "single_model_tier_not_activated": final["gate_scope_decision"][
            "single_model_causal_tier_activated"
        ]
        is False,
        "model_test_not_authorized": final["restart_decision"]["model_test_authorized"]
        is False,
        "auto_continue_false": final["restart_decision"]["auto_continue"] is False,
        "no_new_k_item": final["evidence_update"]["new_k_item"] is None,
        "no_theory_update": final["evidence_update"]["theory_update_number"] is None,
        "protocol_digest_valid": canonical_digest(protocol, "protocol_digest")
        == protocol["protocol_digest"],
        "final_digest_valid": canonical_digest(final, "final_digest")
        == final["final_digest"],
    }
    audit: dict[str, Any] = {
        "schema_version": "phase1131_material_readiness_and_claim_scope_result_audit.v1",
        "phase": 1131,
        "checks": checks,
        "passed_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = canonical_digest(audit, "audit_digest")

    write_json(RESULT_ROOT / "protocol/material_contract.json", protocol)
    write_json(RESULT_ROOT / "analysis/readiness_summary.json", final)
    write_json(RESULT_ROOT / "audit/result_audit.json", audit)

    print(
        json.dumps(
            {
                "phase": 1131,
                "passed": audit["passed"],
                "passed_count": audit["passed_count"],
                "total_count": audit["total_count"],
                "material_status": package_audit["status"],
                "material_ready": package_audit["material_ready"],
                "model_test_authorized": final["restart_decision"][
                    "model_test_authorized"
                ],
                "auto_continue": final["restart_decision"]["auto_continue"],
                "protocol_digest": protocol["protocol_digest"],
                "final_digest": final["final_digest"],
                "audit_digest": audit["audit_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
