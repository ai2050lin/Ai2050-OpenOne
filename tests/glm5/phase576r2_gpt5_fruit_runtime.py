#!/usr/bin/env python3
"""Immutable Phase576R2 cleanup-definition correction.

Phase576R1 reached the qwen3 engineering calculation but failed its formal
qualification with an allocator residue exactly reproduced by a synthetic
cuBLAS workspace diagnostic.  That is a runtime-cause candidate, not a proven
model attribution.  R2 preserves both earlier terminal histories, reuses the
byte-identical scientific denominator, and tests the candidate by completing
PyTorch's cleanup definition before the unchanged zero-allocation gate.

This source must be audited and source-sealed before the R2 freeze.  It must
not be edited after that freeze.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

R1_WRAPPER_PATH = GLM5 / "phase576r1_gpt5_fruit_runtime.py"
R1_WRAPPER_SHA256 = (
    "eee847e4c5262d5880ff11136a030730e46db951296962a71099c7c91bedbe09"
)


def _sha256_file_untrusted(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if (
    not R1_WRAPPER_PATH.is_file()
    or R1_WRAPPER_PATH.is_symlink()
    or R1_WRAPPER_PATH.stat().st_size != 47340
    or _sha256_file_untrusted(R1_WRAPPER_PATH) != R1_WRAPPER_SHA256
):
    raise RuntimeError("Phase576R2 refuses an unsealed Phase576R1 wrapper")

PROTOCOL_WAS_PRELOADED = "phase576_gpt5_fruit_protocol" in sys.modules
ENGINE_WAS_PRELOADED = "phase983_cross_model_engine" in sys.modules
R1_WRAPPER_WAS_PRELOADED = "phase576r1_gpt5_fruit_runtime" in sys.modules
PRELOADED_STAGES = tuple(
    name
    for name in (
        "phase576_gpt5_fruit_engineering_qualification",
        "phase576_gpt5_fruit_behavior",
        "phase576_gpt5_fruit_behavior_analysis",
        "phase576_gpt5_fruit_natural_trace",
    )
    if name in sys.modules
)

import phase576r1_gpt5_fruit_runtime as r1  # noqa: E402
import torch  # noqa: E402


protocol = r1.protocol
R2_OUT_DIR = ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure"
R1_OUT_DIR = ROOT / "tests/glm5/result/phase576r1_gpt5_fruit_structure"
WRAPPER_PATH = Path(__file__).resolve()
R1_SOURCE_KEY = "tests/glm5/phase576r1_gpt5_fruit_runtime.py"
R2_SOURCE_KEY = "tests/glm5/phase576r2_gpt5_fruit_runtime.py"
R2_CLEANUP_REPORT_PATH = R2_OUT_DIR / "phase576r2_cuda_cleanup_qualification.json"

R1_EVIDENCE: dict[str, dict[str, Any]] = {
    "frozen_protocol": {
        "path": R1_OUT_DIR / "phase576_frozen_protocol.json",
        "sha256": "22de92bb85a1129ec1cf026f383d82b60d526c85e0e82596aaf937f8379c4a75",
        "size_bytes": 58467,
    },
    "freeze_commit": {
        "path": R1_OUT_DIR / "phase576_freeze_commit.json",
        "sha256": "15e3612352cf61d757565995aaf02ae215b4061d400a9e99233ede89c9c14a18",
        "size_bytes": 1452,
    },
    "static_audit": {
        "path": R1_OUT_DIR / "phase576_static_audit.json",
        "sha256": "a83027cc4e9fb11783660631152886fc5ff994e9dc752bd29ebef250e5528d22",
        "size_bytes": 16385,
    },
    "sealed_commitment": {
        "path": R1_OUT_DIR / "phase576_sealed_commitment.json",
        "sha256": "f1261b1e4b63f3bbc9e696333def97ebe521508f816a29cec286fd1705cbc256",
        "size_bytes": 535,
    },
    "engineering_stage_start": {
        "path": R1_OUT_DIR / "engineering_qualification_execution/stage_start.json",
        "sha256": "060d754350a61c9e6eb256f6b7bd3974e58d055b749b2943f138706945e1ea9b",
        "size_bytes": 3258,
    },
    "qwen_running": {
        "path": R1_OUT_DIR / "engineering_qualification_execution/00_qwen3.running.json",
        "sha256": "96fbea3a48e06f7539ff04d97cb1e88ecc7d9bcd847207f54bcec360013ca9ae",
        "size_bytes": 6555,
    },
    "qwen_failed": {
        "path": R1_OUT_DIR / "engineering_qualification_execution/00_qwen3.failed.json",
        "sha256": "cfa7e172f29cfb6463472c926eeff1453de68bbc80c57746c981947ec8532397",
        "size_bytes": 1452,
    },
    "engineering_receipt": {
        "path": R1_OUT_DIR / "engineering_qualification_execution/execution_receipt.json",
        "sha256": "3845cc550be65af177c005b75157539fb4aae22210335dbccca5469ccb11feb5",
        "size_bytes": 13150,
    },
    "engineering_lease": {
        "path": R1_OUT_DIR / ".phase576_engineering_qualification.lease",
        "sha256": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "size_bytes": 1,
    },
    "r1_wrapper": {
        "path": R1_WRAPPER_PATH,
        "sha256": R1_WRAPPER_SHA256,
        "size_bytes": 47340,
        "root": ROOT,
    },
}

R1_FREEZE_REGISTRY_PATHS = {
    "phase576_confirmation_cases.jsonl",
    "phase576_discovery_cases.jsonl",
    "phase576_frozen_protocol.json",
    "phase576_heldout_recombination_cases.jsonl",
    "phase576_open_cases.jsonl",
    "phase576_sealed_commitment.json",
    "phase576_static_audit.json",
    "protocol/private/phase576_sealed_cases.jsonl",
}
R1_EXACT_DIRECTORY_INVENTORY = {
    "engineering_qualification_execution",
    "protocol",
    "protocol/private",
}
R1_EXACT_FILE_INVENTORY = {
    ".phase576_engineering_qualification.lease",
    "engineering_qualification_execution/00_qwen3.failed.json",
    "engineering_qualification_execution/00_qwen3.running.json",
    "engineering_qualification_execution/execution_receipt.json",
    "engineering_qualification_execution/stage_start.json",
    "phase576_confirmation_cases.jsonl",
    "phase576_discovery_cases.jsonl",
    "phase576_freeze_commit.json",
    "phase576_frozen_protocol.json",
    "phase576_heldout_recombination_cases.jsonl",
    "phase576_open_cases.jsonl",
    "phase576_sealed_commitment.json",
    "phase576_static_audit.json",
    "protocol/private/phase576_sealed_cases.jsonl",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_identity(path: Path) -> dict[str, Any]:
    identity = r1.strict_root_file_identity(path, ROOT)
    return {
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def verify_r1_failure_evidence() -> dict[str, Any]:
    """Verify the complete immutable R1 freeze and cleanup-only failure."""

    identities: dict[str, dict[str, Any]] = {}
    for name, expected in R1_EVIDENCE.items():
        evidence_root = expected.get("root", R1_OUT_DIR)
        identity = r1.strict_root_file_identity(expected["path"], evidence_root)
        if (
            identity["sha256"] != expected["sha256"]
            or identity["size_bytes"] != expected["size_bytes"]
        ):
            raise RuntimeError(f"Phase576R1 immutable evidence drift: {name}")
        identities[name] = identity

    frozen = r1.read_json(R1_EVIDENCE["frozen_protocol"]["path"])
    freeze_commit = r1.read_json(R1_EVIDENCE["freeze_commit"]["path"])
    stage_start = r1.read_json(R1_EVIDENCE["engineering_stage_start"]["path"])
    running = r1.read_json(R1_EVIDENCE["qwen_running"]["path"])
    failed = r1.read_json(R1_EVIDENCE["qwen_failed"]["path"])
    receipt = r1.read_json(R1_EVIDENCE["engineering_receipt"]["path"])

    registry = freeze_commit.get("artifact_sha256_by_path")
    if (
        not isinstance(registry, dict)
        or set(registry) != R1_FREEZE_REGISTRY_PATHS
        or freeze_commit.get("artifact_count") != 8
        or freeze_commit.get("complete") is not True
        or freeze_commit.get("overwrite_allowed") is not False
        or freeze_commit.get("atomic_directory_publish") is not True
    ):
        raise RuntimeError("Phase576R1 freeze registry is not the audited closure")
    freeze_identities: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in sorted(registry.items()):
        identity = r1.strict_root_file_identity(
            R1_OUT_DIR / Path(relative), R1_OUT_DIR
        )
        if identity["sha256"] != expected_hash:
            raise RuntimeError(f"Phase576R1 frozen artifact drift: {relative}")
        freeze_identities[relative] = identity

    source_seals = frozen.get("stage_source_seals")
    if not isinstance(source_seals, dict) or not source_seals:
        raise RuntimeError("Phase576R1 source registry is malformed")
    current_source_seals: dict[str, dict[str, Any]] = {}
    for key, expected in source_seals.items():
        observed = _source_identity(ROOT / Path(key))
        if observed != expected:
            raise RuntimeError(f"Phase576R1 frozen source drift: {key}")
        current_source_seals[key] = observed

    root_directories: set[str] = set()
    root_files: set[str] = set()
    for path in sorted(R1_OUT_DIR.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Phase576R1 root contains a symlink: {path}")
        relative = str(path.relative_to(R1_OUT_DIR)).replace("\\", "/")
        if path.is_dir():
            root_directories.add(relative)
        elif path.is_file():
            root_files.add(relative)
        else:
            raise RuntimeError(f"Phase576R1 root contains a special entry: {path}")

    execution_dir = R1_OUT_DIR / "engineering_qualification_execution"
    if execution_dir.is_symlink() or not execution_dir.is_dir():
        raise RuntimeError("Phase576R1 engineering directory is not canonical")
    inventory = []
    for path in sorted(execution_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Phase576R1 unexpected execution artifact: {path}")
        inventory.append(str(path.relative_to(execution_dir)).replace("\\", "/"))

    expected_attempt_id = "949b744d-b540-46f8-9f31-dab17f7bf06e:0:qwen3"
    r1_intended_runtime = frozen.get("runtime_erratum", {}).get(
        "intended_runtime"
    )
    current_runtime = r1.require_intended_cuda_runtime()
    checks = {
        "protocol_phase": frozen.get("phase_id") == "Phase576",
        "protocol_hash_bound": registry.get("phase576_frozen_protocol.json")
        == identities["frozen_protocol"]["sha256"],
        "source_registry_exact": current_source_seals == source_seals,
        "runtime_identity_continuity": r1_intended_runtime == current_runtime,
        "runtime_identity_hash_continuity": protocol.stable_hash(
            r1_intended_runtime
        ) == protocol.stable_hash(current_runtime),
        "r1_wrapper_in_registry": source_seals.get(R1_SOURCE_KEY)
        == {"size_bytes": 47340, "sha256": R1_WRAPPER_SHA256},
        "root_directory_inventory_exact": root_directories
        == R1_EXACT_DIRECTORY_INVENTORY,
        "root_file_inventory_exact": root_files == R1_EXACT_FILE_INVENTORY,
        "lease_identity_exact": identities["engineering_lease"]["sha256"]
        == "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "execution_inventory_exact": inventory
        == [
            "00_qwen3.failed.json",
            "00_qwen3.running.json",
            "execution_receipt.json",
            "stage_start.json",
        ],
        "same_run": receipt.get("run_id") == stage_start.get("run_id")
        == running.get("run_id") == failed.get("run_id")
        == "949b744d-b540-46f8-9f31-dab17f7bf06e",
        "same_attempt": running.get("attempt_id") == failed.get("attempt_id")
        == expected_attempt_id,
        "receipt_stage_hash": receipt.get("stage_start_sha256")
        == identities["engineering_stage_start"]["sha256"],
        "failed_running_chain": failed.get("running_status_sha256")
        == identities["qwen_running"]["sha256"],
        "terminal_failure": receipt.get("terminal_status") == "failed"
        and receipt.get("execution_passed") is False,
        "attempted_only_qwen": receipt.get("attempted_models_in_order")
        == ["qwen3"]
        and receipt.get("failed_models") == ["qwen3"]
        and receipt.get("completed_models") == []
        and receipt.get("not_attempted_models") == ["glm4", "deepseek7b"],
        "single_cleanup_component": failed.get("failure_components")
        == [
            {
                "error": "nonzero CUDA allocation after release: 8519680",
                "error_type": "CudaReleaseError",
                "stage": "cuda_release_gate",
            }
        ],
        "allocated_failure_exact": failed.get("cuda_memory_after_release", {}).get(
            "total_allocated_bytes"
        ) == 8519680
        and failed.get("cuda_memory_after_release", {}).get(
            "total_reserved_bytes"
        ) == 4410310656,
        "cleanup_not_complete": failed.get("cleanup_completed") is False
        and receipt.get("final_cuda_cleanup_pass") is False,
        "contract_unchanged": receipt.get("execution_contract_sha256")
        == failed.get("execution_contract_sha256")
        == "8b2f26867efb9912cc810a8f1eaee9cd3a6e63a5f0a1dc3960d9edaca8d470b7",
        "no_formal_case_access": receipt.get("formal_case_access") is False
        and receipt.get("formal_case_content_parsed") is False,
        "no_sealed_access": receipt.get("sealed_split_read") is False
        and receipt.get("sealed_case_payload_read") is False,
        "qualification_absent": not (
            R1_OUT_DIR / "phase576_engineering_qualification.json"
        ).exists(),
        "qwen_complete_absent": not (
            execution_dir / "00_qwen3.complete.json"
        ).exists(),
        "freeze_closure_exact": len(freeze_identities) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R1 failure closure mismatch: {checks}")
    return {
        "artifact_identities": identities,
        "freeze_artifact_identities": freeze_identities,
        "source_seals": source_seals,
        "root_directory_inventory": sorted(root_directories),
        "root_file_inventory": sorted(root_files),
        "r1_intended_runtime_identity_sha256": protocol.stable_hash(
            r1_intended_runtime
        ),
        "current_runtime_identity_sha256": protocol.stable_hash(current_runtime),
        "execution_inventory": inventory,
        "checks": checks,
        "run_id": receipt["run_id"],
        "attempted_models_in_order": receipt["attempted_models_in_order"],
        "completed_models": receipt["completed_models"],
        "failed_models": receipt["failed_models"],
        "not_attempted_models": receipt["not_attempted_models"],
        "failure_component": failed["failure_components"][0],
        "pytorch_cuda_allocated_after_release": 8519680,
        "pytorch_cuda_reserved_after_release": 4410310656,
        "formal_qualification_published": False,
        "precleanup_report_persisted": False,
        "precleanup_calculation_not_reusable_as_qualification": True,
    }


def r2_runtime_erratum_payload() -> dict[str, Any]:
    return {
        "schema_version": "phase576r2_cleanup_definition_erratum.v1",
        "retry_id": "Phase576R2",
        "reason": (
            "Phase576R1 failed its unchanged zero-allocation qualification gate "
            "after qwen3 with a residue consistent with a reproducible cuBLAS "
            "workspace allocation; the prior cleanup definition did not explicitly "
            "clear that runtime state"
        ),
        "correction_scope": "post_model_cuda_cleanup_definition_only",
        "original_phase576_failure": r1.verify_original_failure_evidence(),
        "phase576r1_failure": verify_r1_failure_evidence(),
        "intended_runtime": r1.require_intended_cuda_runtime(),
        "r2_result_root": r1.relative_path(R2_OUT_DIR),
        "cleanup_operation": {
            "api": "torch._C._cuda_clearCublasWorkspaces",
            "api_required_callable": True,
            "applied_after_original_release": True,
            "applied_before_existing_allocator_measurement": True,
            "applies_to": [
                "engineering_qualification",
                "open_behavior",
                "natural_trace",
            ],
            "same_process_zero_allocation_gate_retained": True,
            "allocation_tolerance_bytes": 0,
            "process_isolation_used": False,
            "process_isolation_is_only_a_future_fail_closed_fallback": True,
        },
        "scientific_denominator_changed": False,
        "case_prompts_targets_splits_changed": False,
        "behavior_thresholds_changed": False,
        "trace_scope_changed": False,
        "model_order_changed": False,
        "model_artifacts_changed": False,
        "original_phase576_result_overwritten": False,
        "phase576r1_result_overwritten": False,
        "r1_qwen_formal_qualification_reused": False,
        "all_models_must_be_rerun": True,
        "root_cause_label": "runtime_cleanup_definition_candidate",
        "root_cause_not_a_scientific_mechanism_claim": True,
    }


def r2_stage_source_seals() -> dict[str, dict[str, Any]]:
    r1.revalidate_intended_cuda_runtime()
    original = r1.read_json(r1.ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    seals = r1.verified_original_stage_source_seals(original)
    r1_identity = _source_identity(R1_WRAPPER_PATH)
    r2_identity = _source_identity(WRAPPER_PATH)
    if r1_identity != {"size_bytes": 47340, "sha256": R1_WRAPPER_SHA256}:
        raise RuntimeError("Phase576R1 wrapper drift before R2 source seal")
    seals[R1_SOURCE_KEY] = r1_identity
    seals[R2_SOURCE_KEY] = r2_identity
    return seals


def _staged_policy_equivalent(
    original: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    original_policy = copy.deepcopy(original.get("staged_analysis_seal_policy"))
    candidate_policy = copy.deepcopy(candidate.get("staged_analysis_seal_policy"))
    if not isinstance(original_policy, dict) or not isinstance(candidate_policy, dict):
        return False
    original_sources = original_policy.pop("initial_stage_sources", None)
    candidate_sources = candidate_policy.pop("initial_stage_sources", None)
    return all(
        (
            original_sources == original.get("stage_source_seals"),
            candidate_sources == candidate.get("stage_source_seals"),
            candidate_sources == r2_stage_source_seals(),
            original_policy == candidate_policy,
        )
    )


def _normalized_commitment(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    value.pop("created_at_utc", None)
    return value


def _normalized_static_audit(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    for key in (
        "schema_version",
        "phase_id",
        "created_at_utc",
        "open_cases_sha256",
        "open_case_sha256_by_split",
        "sealed_cases_sha256",
        "sealed_commitment_sha256",
        "protocol_sha256",
    ):
        value.pop(key, None)
    return value


def validate_unpublished_r2_protocol(
    candidate: dict[str, Any],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    """Run inside protocol serialization, before staging or atomic publish."""

    original_failure = r1.verify_original_failure_evidence()
    r1_failure = verify_r1_failure_evidence()
    original = r1.read_json(r1.ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    original_registry = r1.read_json(
        r1.ORIGINAL_EVIDENCE["freeze_commit"]["path"]
    )["artifact_sha256_by_path"]
    expected_seals = r2_stage_source_seals()
    split_hashes = {
        split: protocol.sha256_bytes(
            protocol.jsonl_bytes(
                [row for row in open_rows if row["split"] == split]
            )
        )
        for split in protocol.OPEN_SPLITS
    }
    original_commitment = r1.read_json(
        r1.ORIGINAL_OUT_DIR / "phase576_sealed_commitment.json"
    )
    regenerated_commitment = {
        "schema_version": "phase576_sealed_commitment.v2",
        "phase_id": protocol.PHASE,
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": protocol.sha256_bytes(
            protocol.jsonl_bytes(sealed_rows)
        ),
        "holdout_is_blind": False,
        "sealed_definition_is_public_in_source": True,
        "sealed_definition_cpu_read_during_freeze": True,
        "sealed_model_opened": False,
        "sealed_model_access_count": 0,
        "sealed_result_analysis_access_count": 0,
        "prior_sealed_files_read": False,
    }
    checks = {
        "phase_and_schema_identical": candidate.get("phase_id")
        == original.get("phase_id") == "Phase576"
        and candidate.get("schema_version") == original.get("schema_version"),
        "source_script_identical": candidate.get("source_script")
        == original.get("source_script")
        and candidate.get("source_script_sha256")
        == original.get("source_script_sha256"),
        "source_registry_original_plus_r1_plus_r2": candidate.get(
            "stage_source_seals"
        ) == expected_seals,
        "model_artifacts_identical": candidate.get("model_artifact_identities")
        == original.get("model_artifact_identities"),
        "scientific_and_atomic_fields_identical": all(
            candidate.get(key) == original.get(key)
            for key in r1.PREFREEZE_EQUIVALENCE_KEYS
        ),
        "staged_policy_normalized_identical": _staged_policy_equivalent(
            original, candidate
        ),
        "runtime_erratum_exact": candidate.get("runtime_erratum")
        == r2_runtime_erratum_payload(),
        "raw_static_audit_exact": audit
        == _normalized_static_audit(
            r1.read_json(r1.ORIGINAL_OUT_DIR / "phase576_static_audit.json")
        ),
        "sealed_commitment_normalized_exact": _normalized_commitment(
            regenerated_commitment
        ) == _normalized_commitment(original_commitment),
        "open_bytes_identical": protocol.sha256_bytes(
            protocol.jsonl_bytes(open_rows)
        ) == original_registry["phase576_open_cases.jsonl"],
        "split_bytes_identical": split_hashes
        == {
            "discovery": original_registry["phase576_discovery_cases.jsonl"],
            "confirmation": original_registry[
                "phase576_confirmation_cases.jsonl"
            ],
            "heldout_recombination": original_registry[
                "phase576_heldout_recombination_cases.jsonl"
            ],
        },
        "sealed_bytes_identical": protocol.sha256_bytes(
            protocol.jsonl_bytes(sealed_rows)
        ) == original_registry["protocol/private/phase576_sealed_cases.jsonl"],
        "original_freeze_closed": original_failure["freeze_closure"][
            "artifact_count"
        ] == 8,
        "r1_freeze_and_failure_closed": r1_failure["checks"][
            "freeze_closure_exact"
        ] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R2 unpublished equivalence failed: {checks}")


def r2_protocol_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = r1.ORIGINAL_PROTOCOL_PAYLOAD(*args, **kwargs)
    payload["runtime_erratum"] = r2_runtime_erratum_payload()
    open_rows = args[0] if args else kwargs["open_rows"]
    sealed_rows = args[1] if len(args) > 1 else kwargs["sealed_rows"]
    audit = args[2] if len(args) > 2 else kwargs["audit"]
    validate_unpublished_r2_protocol(payload, open_rows, sealed_rows, audit)
    return payload


def preflight_r2_freeze() -> dict[str, Any]:
    if R2_OUT_DIR.exists() or protocol.FREEZE_LOCK_PATH.exists():
        raise RuntimeError("Phase576R2 freeze requires an absent result root and lock")
    r1.verify_original_failure_evidence()
    verify_r1_failure_evidence()
    original = r1.read_json(r1.ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    registry = r1.read_json(r1.ORIGINAL_EVIDENCE["freeze_commit"]["path"])[
        "artifact_sha256_by_path"
    ]
    open_rows, sealed_rows, audit = protocol.build_all()
    if audit.get("valid") is not True or audit.get("failures") != []:
        raise RuntimeError("Phase576R2 regenerated denominator audit failed")
    model_artifacts = protocol.model_artifact_identity()
    source_seals = r2_stage_source_seals()
    candidate = r2_protocol_payload(
        open_rows,
        sealed_rows,
        audit,
        "preflight-only-not-published",
        model_artifacts,
        source_seals,
        r1.sha256_file(Path(protocol.__file__).resolve()),
    )
    byte_checks = {
        "open": protocol.sha256_bytes(protocol.jsonl_bytes(open_rows))
        == registry["phase576_open_cases.jsonl"],
        "sealed": protocol.sha256_bytes(protocol.jsonl_bytes(sealed_rows))
        == registry["protocol/private/phase576_sealed_cases.jsonl"],
        **{
            split: protocol.sha256_bytes(
                protocol.jsonl_bytes(
                    [row for row in open_rows if row["split"] == split]
                )
            ) == registry[f"phase576_{split}_cases.jsonl"]
            for split in protocol.OPEN_SPLITS
        },
    }
    checks = {
        "case_bytes_identical": all(byte_checks.values()),
        "model_artifacts_identical": model_artifacts
        == original.get("model_artifact_identities"),
        "source_registry_exact": candidate.get("stage_source_seals")
        == source_seals,
        "scientific_fields_identical": all(
            candidate.get(key) == original.get(key)
            for key in r1.PREFREEZE_EQUIVALENCE_KEYS
        ),
        "runtime_erratum_exact": candidate.get("runtime_erratum")
        == r2_runtime_erratum_payload(),
        "staged_policy_normalized_identical": _staged_policy_equivalent(
            original, candidate
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R2 prefreeze equivalence failed: {checks}")
    return {
        "checks": checks,
        "case_byte_checks": byte_checks,
        "model_artifact_identities_sha256": protocol.stable_hash(model_artifacts),
        "source_seals_sha256": protocol.stable_hash(source_seals),
    }


def verify_r2_equivalence() -> dict[str, Any]:
    if not protocol.PROTOCOL_PATH.is_file():
        raise RuntimeError("Phase576R2 frozen protocol is missing")
    original_failure = r1.verify_original_failure_evidence()
    r1_failure = verify_r1_failure_evidence()
    original = r1.read_json(r1.ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    retry = r1.read_json(protocol.PROTOCOL_PATH)
    retry_commitment = r1.read_json(protocol.SEALED_COMMITMENT_PATH)
    original_commitment = r1.read_json(
        r1.ORIGINAL_OUT_DIR / "phase576_sealed_commitment.json"
    )
    retry_audit = r1.read_json(protocol.STATIC_AUDIT_PATH)
    original_audit = r1.read_json(
        r1.ORIGINAL_OUT_DIR / "phase576_static_audit.json"
    )
    checks = {
        "phase_and_schema_identical": retry.get("phase_id")
        == original.get("phase_id") == "Phase576"
        and retry.get("schema_version") == original.get("schema_version"),
        "runtime_erratum_exact": retry.get("runtime_erratum")
        == r2_runtime_erratum_payload(),
        "source_registry_original_plus_r1_plus_r2": retry.get(
            "stage_source_seals"
        ) == r2_stage_source_seals(),
        "all_scientific_fields_identical": all(
            retry.get(key) == original.get(key)
            for key in r1.SCIENTIFIC_EQUIVALENCE_KEYS
        ),
        "atomic_policy_identical": retry.get("atomic_freeze_policy")
        == original.get("atomic_freeze_policy"),
        "staged_policy_normalized_identical": _staged_policy_equivalent(
            original, retry
        ),
        "static_audit_normalized_identical": _normalized_static_audit(
            retry_audit
        ) == _normalized_static_audit(original_audit),
        "sealed_commitment_normalized_identical": _normalized_commitment(
            retry_commitment
        ) == _normalized_commitment(original_commitment),
        "open_case_bytes_identical": r1.sha256_file(protocol.OPEN_CASES_PATH)
        == original["open_cases_sha256"],
        "split_case_bytes_identical": {
            split: r1.sha256_file(path)
            for split, path in protocol.OPEN_SPLIT_CASE_PATHS.items()
        } == original["open_case_sha256_by_split"],
        "sealed_case_bytes_identical": r1.sha256_file(
            protocol.SEALED_CASES_PATH
        ) == retry_commitment.get("sealed_cases_sha256")
        == original_failure["freeze_closure"]["artifact_identities"][
            "protocol/private/phase576_sealed_cases.jsonl"
        ]["sha256"],
        "original_failure_closed": original_failure["freeze_closure"][
            "artifact_count"
        ] == 8,
        "r1_failure_closed": r1_failure["checks"]["freeze_closure_exact"]
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R2 frozen equivalence failed: {checks}")
    return checks


_ORIGINAL_RELEASE: Callable[[Any], None] | None = None
_STRICT_RELEASE_INSTALLED = False


def strict_release_model_adapter(adapter: Any) -> None:
    """Run the old release, then clear cuBLAS state before the old 0B gate."""

    if _ORIGINAL_RELEASE is None:
        raise RuntimeError("Phase576R2 strict cleanup was not installed")
    original_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        _ORIGINAL_RELEASE(adapter)
    except BaseException as exc:
        original_error = exc
        exc.__traceback__ = None
    try:
        gc.collect()
        if not torch.cuda.is_available():
            raise RuntimeError("Phase576R2 strict cleanup requires CUDA")
        torch.cuda.synchronize()
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if not callable(clear):
            raise RuntimeError(
                "required torch._C._cuda_clearCublasWorkspaces is unavailable"
            )
        clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except BaseException as exc:
        cleanup_error = exc
        exc.__traceback__ = None
    if original_error is not None:
        raise original_error
    if cleanup_error is not None:
        raise cleanup_error


def install_strict_cleanup() -> Any:
    global _ORIGINAL_RELEASE, _STRICT_RELEASE_INSTALLED
    if _STRICT_RELEASE_INSTALLED:
        engine = importlib.import_module("phase983_cross_model_engine")
        if engine.release_model_adapter is not strict_release_model_adapter:
            raise RuntimeError("Phase576R2 strict cleanup binding was replaced")
        return engine
    if ENGINE_WAS_PRELOADED or "phase983_cross_model_engine" in sys.modules:
        raise RuntimeError("Phase576R2 refuses a preloaded engine module")
    engine = importlib.import_module("phase983_cross_model_engine")
    original = getattr(engine, "release_model_adapter", None)
    if not callable(original):
        raise RuntimeError("Phase576R2 engine has no callable release_model_adapter")
    _ORIGINAL_RELEASE = original
    engine.release_model_adapter = strict_release_model_adapter
    if engine.release_model_adapter is not strict_release_model_adapter:
        raise RuntimeError("Phase576R2 failed to bind the strict cleanup")
    _STRICT_RELEASE_INSTALLED = True
    return engine


def _assert_stage_cleanup_binding(module: Any) -> None:
    observed = getattr(module, "release_model_adapter", None)
    if observed is not strict_release_model_adapter:
        raise RuntimeError(
            f"Phase576R2 stage did not bind strict cleanup: {module.__name__}"
        )


def import_stage(module_name: str) -> Any:
    if not _STRICT_RELEASE_INSTALLED:
        raise RuntimeError("Phase576R2 strict cleanup must be installed first")
    module = importlib.import_module(module_name)
    if module_name in {
        "phase576_gpt5_fruit_engineering_qualification",
        "phase576_gpt5_fruit_behavior",
        "phase576_gpt5_fruit_natural_trace",
    }:
        _assert_stage_cleanup_binding(module)
    if module_name == "phase576_gpt5_fruit_behavior_analysis":
        behavior = sys.modules.get("phase576_gpt5_fruit_behavior")
        if behavior is not None:
            _assert_stage_cleanup_binding(behavior)
    for loaded_name in (
        "phase576_gpt5_fruit_engineering_qualification",
        "phase576_gpt5_fruit_behavior",
        "phase576_gpt5_fruit_natural_trace",
    ):
        loaded = sys.modules.get(loaded_name)
        if loaded is not None:
            _assert_stage_cleanup_binding(loaded)
    r1.assert_loaded_stage_modules_bound_to_retry()
    return module


def _atomic_write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def strict_cleanup_self_test() -> dict[str, Any]:
    if not _STRICT_RELEASE_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Phase576R2 cleanup self-test requires installed CUDA cleanup")
    if R2_CLEANUP_REPORT_PATH.exists():
        if R2_CLEANUP_REPORT_PATH.is_symlink() or not R2_CLEANUP_REPORT_PATH.is_file():
            raise RuntimeError("Phase576R2 cleanup report is not a regular file")
        existing = r1.read_json(R2_CLEANUP_REPORT_PATH)
        expected_keys = {
            "schema_version",
            "phase_id",
            "created_at_utc",
            "passed",
            "synthetic_only",
            "formal_case_content_read",
            "sealed_case_payload_read",
            "matrix_shape",
            "matrix_dtype",
            "baseline_allocated",
            "baseline_reserved",
            "allocated_during_matmul",
            "reserved_during_matmul",
            "allocated_before_strict_cleanup",
            "reserved_before_strict_cleanup",
            "allocated_after_strict_cleanup",
            "reserved_after_strict_cleanup",
            "allocation_tolerance_bytes",
            "clear_cublas_api_required",
            "runtime_gate_role",
            "not_a_model_or_scientific_qualification",
            "wrapper_source_sha256",
            "frozen_protocol_sha256",
            "freeze_commit_sha256",
            "stage_source_seals_sha256",
            "r1_failure_receipt_sha256",
            "runtime_identity",
        }
        if (
            set(existing) != expected_keys
            or
            existing.get("schema_version")
            != "phase576r2_cuda_cleanup_qualification.v1"
            or existing.get("phase_id") != "Phase576"
            or existing.get("passed") is not True
            or existing.get("synthetic_only") is not True
            or existing.get("formal_case_content_read") is not False
            or existing.get("sealed_case_payload_read") is not False
            or existing.get("matrix_shape") != [1024, 1024]
            or existing.get("matrix_dtype") != "torch.bfloat16"
            or existing.get("baseline_allocated") != 0
            or existing.get("allocated_before_strict_cleanup", 0) <= 0
            or existing.get("wrapper_source_sha256") != r1.sha256_file(WRAPPER_PATH)
            or existing.get("allocated_after_strict_cleanup") != 0
            or existing.get("reserved_after_strict_cleanup") != 0
            or existing.get("allocation_tolerance_bytes") != 0
            or existing.get("clear_cublas_api_required") is not True
            or existing.get("runtime_gate_role")
            != "mandatory_wrapper_runtime_entry_gate"
            or existing.get("not_a_model_or_scientific_qualification") is not True
            or existing.get("frozen_protocol_sha256")
            != r1.sha256_file(protocol.PROTOCOL_PATH)
            or existing.get("freeze_commit_sha256")
            != r1.sha256_file(protocol.FREEZE_COMMIT_PATH)
            or existing.get("stage_source_seals_sha256")
            != protocol.stable_hash(r1.read_json(protocol.PROTOCOL_PATH)[
                "stage_source_seals"
            ])
            or existing.get("r1_failure_receipt_sha256")
            != R1_EVIDENCE["engineering_receipt"]["sha256"]
            or existing.get("runtime_identity") != r1.require_intended_cuda_runtime()
        ):
            raise RuntimeError("existing Phase576R2 cleanup report is not reusable")
        strict_release_model_adapter(None)
        if int(torch.cuda.memory_allocated()) != 0:
            raise RuntimeError("Phase576R2 current CUDA baseline is not clean")
        return existing

    strict_release_model_adapter(None)
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    if baseline_allocated != 0:
        raise RuntimeError(
            f"Phase576R2 dirty CUDA baseline before cleanup self-test: {baseline_allocated}"
        )
    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize()
    allocated_during_matmul = int(torch.cuda.memory_allocated())
    reserved_during_matmul = int(torch.cuda.memory_reserved())
    del product, right, left
    gc.collect()
    allocated_before_strict_cleanup = int(torch.cuda.memory_allocated())
    reserved_before_strict_cleanup = int(torch.cuda.memory_reserved())
    strict_release_model_adapter(None)
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    passed = (
        allocated_before_strict_cleanup > 0
        and allocated_after == 0
        and reserved_after == 0
    )
    payload = {
        "schema_version": "phase576r2_cuda_cleanup_qualification.v1",
        "phase_id": "Phase576",
        "created_at_utc": now(),
        "passed": passed,
        "synthetic_only": True,
        "formal_case_content_read": False,
        "sealed_case_payload_read": False,
        "matrix_shape": [1024, 1024],
        "matrix_dtype": "torch.bfloat16",
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "allocated_during_matmul": allocated_during_matmul,
        "reserved_during_matmul": reserved_during_matmul,
        "allocated_before_strict_cleanup": allocated_before_strict_cleanup,
        "reserved_before_strict_cleanup": reserved_before_strict_cleanup,
        "allocated_after_strict_cleanup": allocated_after,
        "reserved_after_strict_cleanup": reserved_after,
        "allocation_tolerance_bytes": 0,
        "clear_cublas_api_required": True,
        "runtime_gate_role": "mandatory_wrapper_runtime_entry_gate",
        "not_a_model_or_scientific_qualification": True,
        "wrapper_source_sha256": r1.sha256_file(WRAPPER_PATH),
        "frozen_protocol_sha256": r1.sha256_file(protocol.PROTOCOL_PATH),
        "freeze_commit_sha256": r1.sha256_file(protocol.FREEZE_COMMIT_PATH),
        "stage_source_seals_sha256": protocol.stable_hash(
            r1.read_json(protocol.PROTOCOL_PATH)["stage_source_seals"]
        ),
        "r1_failure_receipt_sha256": R1_EVIDENCE[
            "engineering_receipt"
        ]["sha256"],
        "runtime_identity": r1.require_intended_cuda_runtime(),
    }
    _atomic_write_json_exclusive(R2_CLEANUP_REPORT_PATH, payload)
    if not passed:
        raise RuntimeError(
            f"Phase576R2 strict cleanup self-test retained {allocated_after} bytes"
        )
    return payload


def _call_behavior_analysis(stage: str) -> None:
    analysis = import_stage("phase576_gpt5_fruit_behavior_analysis")
    previous = sys.argv
    try:
        sys.argv = [str(Path(analysis.__file__).resolve()), "--stage", stage]
        analysis.main()
    finally:
        sys.argv = previous


def configure_r2_namespace() -> None:
    if (
        PROTOCOL_WAS_PRELOADED
        or R1_WRAPPER_WAS_PRELOADED
        or r1.PROTOCOL_WAS_PRELOADED
    ):
        raise RuntimeError("Phase576R2 refuses a process that preloaded the protocol")
    if ENGINE_WAS_PRELOADED or PRELOADED_STAGES:
        raise RuntimeError(
            "Phase576R2 refuses preloaded engine/stage modules: "
            f"engine={ENGINE_WAS_PRELOADED}, stages={PRELOADED_STAGES}"
        )
    r1.RETRY_OUT_DIR = R2_OUT_DIR
    r1.retry_stage_source_seals = r2_stage_source_seals
    r1.runtime_erratum_payload = r2_runtime_erratum_payload
    r1.validate_unpublished_retry_protocol = validate_unpublished_r2_protocol
    r1.retry_protocol_payload = r2_protocol_payload
    r1.staged_analysis_policy_equivalent = _staged_policy_equivalent
    r1.preflight_retry_freeze = preflight_r2_freeze
    r1.verify_retry_equivalence = verify_r2_equivalence
    r1.configure_retry_namespace()


def main() -> None:
    configure_r2_namespace()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--cleanup-self-test", action="store_true")
    group.add_argument("--engineering", action="store_true")
    group.add_argument("--locator-self-test", action="store_true")
    group.add_argument("--behavior", choices=protocol.OPEN_SPLITS)
    group.add_argument("--behavior-analysis", choices=protocol.OPEN_SPLITS)
    group.add_argument("--trace", choices=protocol.OPEN_SPLITS)
    args = parser.parse_args()

    runtime = r1.require_intended_cuda_runtime()
    r1.verify_original_failure_evidence()
    verify_r1_failure_evidence()
    if args.self_test:
        result = protocol.self_test()
        result["runtime_erratum"] = r2_runtime_erratum_payload()
    elif args.write:
        preflight = preflight_r2_freeze()
        result = protocol.freeze()
        result["prepublish_equivalence"] = preflight
        result["runtime_equivalence_checks"] = verify_r2_equivalence()
    else:
        equivalence = verify_r2_equivalence()
        post_equivalence: dict[str, Any]
        try:
            if args.verify:
                result = protocol.verify()
            else:
                install_strict_cleanup()
                if args.cleanup_self_test:
                    result = strict_cleanup_self_test()
                elif args.engineering:
                    strict_cleanup_self_test()
                    engineering = import_stage(
                        "phase576_gpt5_fruit_engineering_qualification"
                    )
                    engineering.main()
                    result = {"passed": True, "stage": "engineering_qualification"}
                elif args.locator_self_test:
                    trace = import_stage("phase576_gpt5_fruit_natural_trace")
                    result = trace.locator_self_test()
                elif args.behavior is not None:
                    strict_cleanup_self_test()
                    behavior = import_stage("phase576_gpt5_fruit_behavior")
                    result = behavior.run_stage(args.behavior)
                elif args.behavior_analysis is not None:
                    _call_behavior_analysis(args.behavior_analysis)
                    result = {"passed": True, "stage": args.behavior_analysis}
                else:
                    strict_cleanup_self_test()
                    trace = import_stage("phase576_gpt5_fruit_natural_trace")
                    result = trace.run_stage(args.trace)
        finally:
            r1.verify_original_failure_evidence()
            verify_r1_failure_evidence()
            post_equivalence = verify_r2_equivalence()
        result["runtime_equivalence_checks_before_stage"] = equivalence
        result["runtime_equivalence_checks"] = post_equivalence
    result["phase576r2_runtime_identity"] = runtime
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
