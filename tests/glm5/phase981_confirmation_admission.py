#!/usr/bin/env python3
"""Independent CPU, fail-closed admission for Phase 981 Qwen generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as boundary  # noqa: E402
import phase981_confirmation_core as core  # noqa: E402
import phase981_fresh_dataset as fresh  # noqa: E402
import phase981_semantic_gate as semantic_gate  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_no_holdout_import() -> None:
    forbidden = [name for name in sys.modules if name == "phase977_holdout_dataset"
                 or name.endswith(".phase977_holdout_dataset")]
    require(not forbidden, f"old holdout module imported: {forbidden}")


def verify_self_hash(
    value: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    payload = boundary.without_fields(value, hash_field, time_field)
    require(value.get(hash_field) == boundary.sha256_json(payload),
            f"{label} self-hash invalid")


def identity_from_audit(audit: dict[str, Any]) -> dict[str, Any]:
    identity = audit.get("identity")
    if not isinstance(identity, dict) and hasattr(fresh, "dataset_identity"):
        identity = fresh.dataset_identity()
    require(isinstance(identity, dict), "fresh audit lacks identity")
    result = dict(identity)
    result["identity_sha256"] = (
        result.get("identity_sha256") or result.get("dataset_identity_sha256")
    )
    require(isinstance(result["identity_sha256"], str)
            and len(result["identity_sha256"]) == 64,
            "fresh identity hash invalid")
    return result


def authenticate_protocol() -> dict[str, Any]:
    protocol = boundary.load_json(core.PROTOCOL_PATH, "Phase981 protocol")
    seal_audit = core.verify_protocol_file_seals(protocol)
    verify_self_hash(protocol, "protocol_sha256", "created_at_utc", "Phase981 protocol")
    require(protocol.get("phase") == core.PHASE, "wrong protocol phase")
    core.verify_protocol_boundary_contract(protocol)
    require(protocol.get("arms") == core.ARMS
            and protocol.get("direction") == core.PRIMARY_DIRECTION,
            "formal A/B contrast changed")
    require(protocol.get("expected_rows") == core.EXPECTED_ROWS
            and protocol.get("streams") == list(core.STREAMS),
            "formal denominator/stream registry changed")
    primary = protocol.get("primary_decision", {})
    require(primary.get("route") == "semantic_only"
            and primary.get("direction") == "B_minus_A"
            and primary.get("censor_route_can_admit") is False
            and primary.get("mixed_route_can_admit") is False,
            "primary decision is not fail-closed semantic-only")
    require(seal_audit["verified_script_count"] == len(core.PHASE981_SCRIPT_PATHS)
            and seal_audit["verified_dependency_count"]
            == len(core.RUNTIME_DEPENDENCY_PATHS)
            and seal_audit["verified_phase979_script_count"]
            == len(core.PHASE979_SCRIPT_PATHS),
            "protocol seal denominator changed")
    core.verify_protocol_integrity_metadata(protocol)
    for source_name in ("phase979_source", "phase980_design_source"):
        source = protocol.get(source_name, {})
        for entry in _path_entries(source):
            path = ROOT / str(entry["path"])
            expected = entry.get("file_sha256") or entry.get("sha256")
            require(path.is_file() and boundary.sha256_file(path) == expected,
                    f"authenticated source changed: {source_name}/{entry['path']}")
    return protocol


def _path_entries(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and (
            isinstance(value.get("file_sha256"), str)
            or isinstance(value.get("sha256"), str)
        ):
            yield value
        for child in value.values():
            yield from _path_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_entries(child)


def revalidate_dependencies(protocol: dict[str, Any]) -> dict[str, Any]:
    seal_audit = core.verify_protocol_file_seals(protocol)
    items = fresh.build_items()
    audit = fresh.audit_items(items)
    require(audit.get("passed") is True or audit.get("ok") is True,
            "fresh dataset audit failed at admission")
    identity = identity_from_audit(audit)
    committed = protocol.get("dataset_identity", {})
    require(identity["identity_sha256"] == committed.get("identity_sha256"),
            "fresh dataset identity differs from protocol")
    require(boundary.sha256_json(audit) == protocol.get("dataset_audit_sha256"),
            "fresh dataset audit differs from protocol")
    artifact_seals = protocol.get("dataset_artifact_seals", {})
    dataset_document = boundary.load_json(
        core.DATASET_ARTIFACT_PATH, "sealed dataset artifact")
    audit_document = boundary.load_json(
        core.DATASET_AUDIT_PATH, "sealed dataset audit artifact")
    require(dataset_document.get("dataset_sha256") == boundary.sha256_json(
        boundary.without_fields(dataset_document, "dataset_sha256")),
        "dataset artifact self-hash invalid at admission")
    require(audit_document.get("audit_sha256") == boundary.sha256_json(
        boundary.without_fields(audit_document, "audit_sha256")),
        "dataset audit artifact self-hash invalid at admission")
    artifact_items = dataset_document.get("items")
    require(isinstance(artifact_items, list)
            and boundary.sha256_json(sorted(
                artifact_items, key=lambda item: str(item.get("id", ""))))
            == boundary.sha256_json(sorted(
                items, key=lambda item: str(item.get("id", ""))))
            and dataset_document.get("identity") == audit_document.get("identity"),
            "dataset artifacts differ from runtime source")
    require(dataset_document.get("identity", {}).get("items_sha256")
            == identity.get("items_sha256"),
            "dataset artifact items hash differs from runtime identity")
    require(audit_document.get("passed") is True
            and audit_document.get("holdout_accessed") is False,
            "dataset artifact audit failed at admission")
    freshness = audit_document.get("freshness_against_phase979_public", {})
    require(freshness.get("normalized_prompt_overlap_n") == 0
            and freshness.get("structural_payload_overlap_n") == 0,
            "dataset artifact overlap is nonzero")
    require(boundary.sha256_file(core.DATASET_ARTIFACT_PATH)
            == artifact_seals.get("dataset", {}).get("file_sha256")
            and dataset_document["dataset_sha256"]
            == artifact_seals.get("dataset", {}).get("dataset_sha256")
            and boundary.sha256_file(core.DATASET_AUDIT_PATH)
            == artifact_seals.get("audit", {}).get("file_sha256")
            and audit_document["audit_sha256"]
            == artifact_seals.get("audit", {}).get("audit_sha256"),
            "dataset artifact seal differs from protocol")
    gate_contract = getattr(semantic_gate, "GATE_CONTRACT", None)
    require(isinstance(gate_contract, dict), "semantic gate contract missing")
    require(boundary.sha256_json(gate_contract)
            == protocol.get("semantic_gate", {}).get("contract_sha256"),
            "semantic gate contract differs from protocol")
    gate_test = semantic_gate.self_test()
    require(gate_test.get("passed") is True, "semantic gate self-test failed")
    seed_values = [
        {
            "id": str(item["id"]), "stream": stream,
            "seed": core.stable_pair_seed(
                identity["identity_sha256"], str(item["id"]), stream),
        }
        for stream in core.STREAMS for item in items
    ]
    seed_audit = protocol.get("common_random_number_contract", {}).get(
        "seed_audit", {})
    require(len(seed_values) == 768
            and len({value["seed"] for value in seed_values}) == 768
            and boundary.sha256_json(seed_values)
            == seed_audit.get("seed_registry_sha256")
            and seed_audit.get("all_three_streams_distinct_per_item") is True,
            "pair seed registry differs from protocol")
    model_identity = protocol.get("local_model_artifact_identity", {})
    core.verify_model_artifact_identity(model_identity)
    token_identity = core.verify_protocol_token_identity(
        protocol.get("tokenizer_audit"), core.EXPECTED_TOKENIZER_EOS_ID,
        core.EXPECTED_THINK_OPEN_ID, core.EXPECTED_THINK_CLOSE_ID,
        core.EXPECTED_A_ID, core.EXPECTED_B_ID)
    return {
        "dataset_identity_sha256": identity["identity_sha256"],
        "dataset_audit_sha256": boundary.sha256_json(audit),
        "dataset_artifact_sha256": dataset_document["dataset_sha256"],
        "dataset_artifact_file_sha256": boundary.sha256_file(
            core.DATASET_ARTIFACT_PATH),
        "dataset_audit_artifact_sha256": audit_document["audit_sha256"],
        "dataset_audit_artifact_file_sha256": boundary.sha256_file(
            core.DATASET_AUDIT_PATH),
        "normalized_prompt_overlap_n": 0,
        "structural_payload_overlap_n": 0,
        "semantic_gate_contract_sha256": boundary.sha256_json(gate_contract),
        "semantic_gate_self_test_passed": True,
        "pair_seed_registry_sha256": boundary.sha256_json(seed_values),
        "unique_pair_seed_count": len({value["seed"] for value in seed_values}),
        "model_artifact_identity_sha256": model_identity.get("identity_sha256"),
        "effective_eos_token_ids": token_identity["effective_eos_token_ids"],
        **seal_audit,
    }


def assert_no_generation_output() -> None:
    existing = [path for path in (
        core.MANIFEST_PATH, core.ROWS_PATH, core.STATUS_PATH,
        core.AUDIT_PATH, core.RUN_LOCK_PATH,
    ) if path.exists()]
    require(not existing,
            f"generation/audit output exists before admission: {[str(x) for x in existing]}")


def build_admission() -> dict[str, Any]:
    assert_no_holdout_import()
    core.assert_contract()
    protocol = authenticate_protocol()
    dependency_audit = revalidate_dependencies(protocol)
    assert_no_generation_output()
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "role": "independent_generation_admission",
        "protocol_sha256": protocol["protocol_sha256"],
        "dependency_audit": dependency_audit,
        "decision": "ADMIT_QWEN_EXTERNAL_GENERATION",
        "admitted": True,
        "qwen_external_generation_authorized": True,
        "gpu_authorized": True,
        "authorization_scope": json.loads(json.dumps(core.AUTHORIZATION_SCOPE)),
        "model_weights_loaded": False,
        "generation_performed": False,
        "gpu_used": False,
        "holdout": False, "holdout_loaded": False, "holdout_authorized": False,
        "mechanism": False, "mechanism_authorized": False,
        "fail_closed_boundary": (
            "Any missing, changed, malformed, pre-existing, or non-PASS dependency must "
            "raise before an admission document is written. This admission authorizes only "
            "the sealed Qwen external generation rows."
        ),
    }
    core.verify_admission_boundary_contract(payload)
    assert_no_holdout_import()
    return {
        **payload,
        "admission_sha256": boundary.sha256_json(payload),
        "admitted_at_utc": boundary.utc_now(),
    }


def install_or_validate(document: dict[str, Any]) -> None:
    verify_self_hash(document, "admission_sha256", "admitted_at_utc",
                     "new Phase981 admission")
    if core.ADMISSION_PATH.exists():
        prior = boundary.load_json(core.ADMISSION_PATH, "existing Phase981 admission")
        verify_self_hash(prior, "admission_sha256", "admitted_at_utc",
                         "existing Phase981 admission")
        require(prior["admission_sha256"] == document["admission_sha256"],
                "existing admission differs")
        return
    core.OUT.mkdir(parents=True, exist_ok=True)
    boundary.atomic_write_json(core.ADMISSION_PATH, document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = build_admission()
    if args.write and not args.self_test:
        install_or_validate(document)
    print(json.dumps({
        "phase": core.PHASE,
        "admission_sha256": document["admission_sha256"],
        "admitted": document["admitted"],
        "written": bool(args.write and not args.self_test),
        "qwen_external_generation_authorized": True,
        "gpu_authorized": True,
        "holdout": False, "mechanism": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
