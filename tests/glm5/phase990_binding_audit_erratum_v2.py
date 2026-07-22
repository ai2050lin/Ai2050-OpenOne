#!/usr/bin/env python3
"""Independent Phase990 audit erratum for a canonical-JSON order false negative.

The sealed v1 audit compared JSON object key iteration order with the required
execution order.  Canonical JSON sorts object keys, so that comparison is not
an execution-order witness.  This erratum leaves every sealed dependency,
threshold, dataset record, and tokenizer result unchanged.  It checks registry
membership as a set and checks the recorded execution order as a list.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import phase990_binding_audit as audit_v1
import phase990_binding_core as core
import phase990_binding_dataset as dataset
import phase990_protocol_freeze as protocol


ERRATUM_PATH = core.OUT / "protocol_audit_erratum_v2.json"
ERRATUM_SCRIPT_PATHS = {
    **core.SCRIPT_PATHS,
    "audit_erratum_v2": "tests/glm5/phase990_binding_audit_erratum_v2.py",
}


def _artifact_seal(path: Path) -> dict[str, Any]:
    core.require(path.is_file(), f"missing artifact: {path}")
    core.require(not path.is_symlink(), f"artifact cannot be a symlink: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(core.ROOT)).replace("\\", "/"),
        "bytes": len(payload),
        "sha256": core.sha256_bytes(payload),
    }


def _all_split_overlap_zero(report: Mapping[str, Any]) -> bool:
    return all(
        not any(overlap.values())
        for right_rows in report["split_overlap"].values()
        for overlap in right_rows.values()
    )


def audit_payload() -> dict[str, Any]:
    dataset_verification = dataset.verify_artifacts()
    protocol_verification = protocol.verify_artifact()

    definitions = core.load_json(core.DEFINITIONS_PATH, "definitions")
    corpus = core.load_json(core.DATASET_PATH, "dataset")
    corpus_audit = core.load_json(core.DATASET_AUDIT_PATH, "dataset audit")
    preregistration = core.load_json(core.PROTOCOL_PATH, "protocol")

    core.verify_self_hash(definitions, "definitions_sha256", "definitions")
    core.verify_self_hash(corpus, "dataset_sha256", "dataset")
    core.verify_self_hash(
        corpus_audit, "dataset_audit_sha256", "dataset audit"
    )
    core.verify_self_hash(preregistration, "protocol_sha256", "protocol")
    core.verify_file_seals(
        preregistration["source_script_seals"],
        core.SCRIPT_PATHS,
        "protocol independent source check",
    )

    tokenizer_audit = preregistration["tokenizer_audit"]
    tokenizer_models = tokenizer_audit["models"]
    canonical_key_order = list(tokenizer_models)
    required_execution_order = list(core.MODEL_ORDER)
    expected_comparisons = {
        "semantic_token_multiset": core.EXPECTED_WORLD_COUNT * 16,
        "relation_equal_length": core.EXPECTED_WORLD_COUNT * 8,
        "near_far_token_multiset": core.EXPECTED_WORLD_COUNT * 16,
        "order_token_multiset": core.EXPECTED_WORLD_COUNT * 16,
        "teacher_forced_canonical_boundary": (
            core.EXPECTED_ITEM_COUNT * len(core.VALUES)
        ),
    }

    checks = {
        "dataset_complete_reconstruction": dataset_verification["passed"],
        "protocol_complete_reconstruction": protocol_verification["passed"],
        "canonical_definitions_bytes": core.DEFINITIONS_PATH.read_bytes()
        == core.json_bytes(definitions),
        "canonical_dataset_bytes": core.DATASET_PATH.read_bytes()
        == core.json_bytes(corpus),
        "canonical_dataset_audit_bytes": core.DATASET_AUDIT_PATH.read_bytes()
        == core.json_bytes(corpus_audit),
        "canonical_protocol_bytes": core.PROTOCOL_PATH.read_bytes()
        == core.json_bytes(preregistration),
        "exact_world_count": corpus["counts"]["worlds"]
        == core.EXPECTED_WORLD_COUNT,
        "exact_record_count": corpus["counts"]["records"]
        == core.EXPECTED_ITEM_COUNT,
        "abstract_state_count_exact": corpus_audit[
            "abstract_semantic_state_count"
        ] == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS),
        "observable_state_count_exact": corpus_audit[
            "observable_semantic_state_count"
        ] == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS),
        "gold_path_deletion_redundancy_fully_measured": all(
            count == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
            for count in corpus_audit[
                "gold_path_deletion_recoverability_counts"
            ].values()
        ) and len(corpus_audit[
            "gold_path_deletion_recoverability_counts"
        ]) == 3,
        "all_split_overlaps_zero": _all_split_overlap_zero(corpus_audit),
        "all_lexical_grids_passed": all(
            condition["passed"]
            for split_rows in corpus_audit["lexical_balance"].values()
            for condition in split_rows.values()
        ),
        "selected_owner_shortcut_baseline_frozen": (
            all(
                condition[
                    "selected_owner_name_object_relation_lookup_baseline"
                ]["majority_correct"] * 2
                == condition[
                    "selected_owner_name_object_relation_lookup_baseline"
                ]["denominator"]
                for split_rows in corpus_audit["lexical_balance"].values()
                for condition in split_rows.values()
            )
            and all(
                baseline["majority_correct"] * 8
                == baseline["denominator"] * 3
                for baseline in corpus_audit[
                    "selected_owner_conjunction_baselines_pooled_across_transforms"
                ].values()
            )
        ),
        "all_shortcut_grids_passed": corpus_audit[
            "shortcut_balance_checks"
        ] == len(core.SPLIT_COUNTS) * len(core.VARIANTS),
        # Erratum: object-key order is not execution order.
        "tokenizer_model_registry_membership_exact": (
            set(tokenizer_models) == set(core.MODEL_ORDER)
            and len(tokenizer_models) == len(core.MODEL_ORDER)
        ),
        "tokenizer_execution_order_exact": tokenizer_audit["observed_order"]
        == required_execution_order,
        "v1_order_false_negative_reproduced": (
            canonical_key_order != required_execution_order
            and set(canonical_key_order) == set(required_execution_order)
        ),
        "tokenizer_record_counts_exact": all(
            report["record_count"] == core.EXPECTED_ITEM_COUNT
            for report in tokenizer_models.values()
        ),
        "tokenizer_comparison_counts_exact": all(
            report["comparison_counts"] == expected_comparisons
            and not report["comparison_failures"]
            for report in tokenizer_models.values()
        ),
        "prior_prompt_overlap_zero": preregistration[
            "prior_prompt_overlap_audit"
        ]["overlap_count"] == 0,
        "operational_weight_access_boundary_clean": all(
            report["weight_file_open_guard_attempts"] == []
            and report["trust_remote_code"] is False
            and report["model_loader_api_called"] is False
            and report["model_weights_loaded"] is False
            for report in tokenizer_models.values()
        ),
        "cuda_not_used": all(
            report["cuda_used"] is False
            for report in tokenizer_models.values()
        ),
        "gpu_generation_not_admitted": preregistration[
            "phase990_decision"
        ]["gpu_generation_admission"] == "not_tested",
        "protocol_rehash_attack_detectable_by_exact_rebuild": (
            audit_v1._exact_validator_rejects_rehash_attack(
                preregistration,
                "protocol_sha256",
                ("phase990_decision", "gpu_generation_admission"),
                "qualified",
            )
        ),
        "dataset_audit_rehash_attack_detectable_by_exact_rebuild": (
            audit_v1._exact_validator_rejects_rehash_attack(
                corpus_audit,
                "dataset_audit_sha256",
                ("cuda_used",),
                True,
            )
        ),
        "isolated_publication_negative_tests_passed": all(
            corpus_audit["publication_negative_tests"].values()
        ),
        "expanded_confirmation_generator_not_falsely_frozen": (
            preregistration["dataset_plan"]
            ["expanded_confirmation_generator_contract"]["status"]
            == "not_frozen_gpu_blocker"
            and preregistration["dataset_plan"]
            ["expanded_confirmation_generator_contract"]
            ["same_locked_generator_claim_authorized"] is False
        ),
        "v1_commit_marker_absent": not core.INDEPENDENT_AUDIT_PATH.exists(),
    }
    core.require(all(checks.values()), f"erratum audit failed: {checks}")

    return {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "independent_cpu_protocol_audit_erratum_v2_and_commit_marker",
        "passed": True,
        "checks": checks,
        "erratum": {
            "scope": "audit_implementation_only",
            "v1_failed_check": "tokenizer_model_registry_exact",
            "v1_failure_kind": "canonical_json_object_key_order_false_negative",
            "v1_expression": "list(tokenizer_models) == list(MODEL_ORDER)",
            "v2_membership_rule": "set(tokenizer_models) == set(MODEL_ORDER)",
            "v2_execution_order_rule": (
                "tokenizer_audit.observed_order == MODEL_ORDER"
            ),
            "canonical_model_key_order": canonical_key_order,
            "required_execution_order": required_execution_order,
            "dataset_or_threshold_changed": False,
            "tokenizer_result_changed": False,
            "model_behavior_observed": False,
        },
        "lineage": {
            "definitions_sha256": definitions["definitions_sha256"],
            "dataset_sha256": corpus["dataset_sha256"],
            "dataset_audit_sha256": corpus_audit["dataset_audit_sha256"],
            "protocol_sha256": preregistration["protocol_sha256"],
        },
        "artifact_file_seals": {
            "definitions": _artifact_seal(core.DEFINITIONS_PATH),
            "dataset": _artifact_seal(core.DATASET_PATH),
            "dataset_audit": _artifact_seal(core.DATASET_AUDIT_PATH),
            "protocol": _artifact_seal(core.PROTOCOL_PATH),
        },
        "source_script_seals": core.file_seals(ERRATUM_SCRIPT_PATHS),
        "scientific_limits": {
            "model_behavior_tested": False,
            "model_mechanism_tested": False,
            "model_weights_loaded": False,
            "cuda_used": False,
            "holdout_is_blind": False,
            "holdout_semantics": "preregistered_immutable_not_blind",
            "lexical_generalization_tested": False,
            "selected_owner_lexical_conjunction_exactly_balanced": False,
            "selected_owner_conjunction_limit": corpus_audit[
                "selected_owner_conjunction_limitation"
            ],
            "prompt_fact_deletion_is_valid_causal_test": False,
            "prompt_fact_deletion_limit": corpus_audit[
                "prompt_fact_deletion_limitation"
            ],
            "expanded_confirmation_generator_frozen": False,
            "same_locked_generator_extension_claim_authorized": False,
            "closure_reference_resolution_implemented": False,
            "graph_thread_reference_resolution_implemented": False,
        },
        "next_step_decision": {
            "automatically_run_gpu_models": False,
            "decision": "not_tested",
            "reason_codes": [
                "FORMAL_GENERATION_ADMISSION_FALSE",
                "USER_THEORY_DISCUSSION_GATE_REMAINS",
                "GPU_RUN_REQUIRES_SEPARATE_ADMISSION_ARTIFACT",
                "HOLDOUT_MODEL_ACCESS_GATE_NOT_SATISFIED",
                "EXTENSION_GENERATOR_NOT_FROZEN",
                "CLOSURE_REFERENCE_RESOLVER_NOT_IMPLEMENTED",
                "GRAPH_THREAD_REFERENCE_RESOLVER_NOT_IMPLEMENTED",
            ],
            "safe_automatic_next_step": (
                "append the CPU seal, audit erratum, limitations, and staged "
                "GPU admission requirements to the research memo"
            ),
        },
    }


def audit_document(created_at_utc: str | None = None) -> dict[str, Any]:
    return core.sealed_document(
        audit_payload(), "protocol_audit_erratum_v2_sha256", created_at_utc
    )


def _static_verify(document: Mapping[str, Any]) -> None:
    core.verify_self_hash(
        document,
        "protocol_audit_erratum_v2_sha256",
        "protocol audit erratum v2",
    )
    core.require(document.get("passed") is True, "erratum audit is not passed")
    core.require(
        ERRATUM_PATH.read_bytes() == core.json_bytes(document),
        "erratum audit bytes are not canonical",
    )


def write_artifact() -> dict[str, Any]:
    timestamp = None
    if ERRATUM_PATH.is_file():
        existing = core.load_json(ERRATUM_PATH, "protocol audit erratum v2")
        timestamp = core.validate_utc_timestamp(
            existing.get("created_at_utc"), "protocol audit erratum v2"
        )
    document = audit_document(timestamp)
    installed = core.install_exact(ERRATUM_PATH, core.json_bytes(document))
    _static_verify(document)
    return {
        "passed": True,
        "installed": installed,
        "protocol_audit_erratum_v2_sha256": document[
            "protocol_audit_erratum_v2_sha256"
        ],
        "protocol_audit_erratum_v2_file_sha256": core.sha256_file(
            ERRATUM_PATH
        ),
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def verify_artifact() -> dict[str, Any]:
    document = core.load_json(ERRATUM_PATH, "protocol audit erratum v2")
    _static_verify(document)
    expected = audit_document(str(document["created_at_utc"]))
    core.verify_exact_document(
        document,
        expected,
        "protocol_audit_erratum_v2_sha256",
        "protocol audit erratum v2",
    )
    return {
        "passed": True,
        "files_written": False,
        "protocol_audit_erratum_v2_sha256": document[
            "protocol_audit_erratum_v2_sha256"
        ],
        "protocol_audit_erratum_v2_file_sha256": core.sha256_file(
            ERRATUM_PATH
        ),
    }


def self_test() -> dict[str, Any]:
    payload = audit_payload()
    return {"passed": payload["passed"], "checks": payload["checks"]}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--self-test"]:
        result = self_test()
    elif arguments == ["--write"]:
        result = write_artifact()
    elif arguments == ["--verify"]:
        result = verify_artifact()
    else:
        raise SystemExit("use --self-test, --write, or --verify")
    print(core.canonical_json(result))


if __name__ == "__main__":
    main()
