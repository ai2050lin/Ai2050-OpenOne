#!/usr/bin/env python3
"""CPU-only formal-generation admission for Phase 983."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402
import phase983_cross_model_protocol as protocol_builder  # noqa: E402
import phase983_cross_model_qualification as qualification_builder  # noqa: E402


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def authenticate_protocol() -> dict[str, Any]:
    document = core.load_json(core.PROTOCOL_PATH, "Phase983 protocol")
    core.verify_self_hash(document, "protocol_sha256", "created_at_utc",
                          "Phase983 protocol")
    stored = core.without_fields(document, "protocol_sha256", "created_at_utc")
    current = protocol_builder.build_payload()
    core.require(stored == current, "protocol no longer matches sealed sources/artifacts")
    protocol_builder.verify_payload(stored, current)
    return document


def build_seed_registry(protocol: dict[str, Any]) -> dict[str, Any]:
    dataset = core.load_json(core.DATASET_PATH, "Phase983 dataset for seed admission")
    core.require(
        core.sha256_file(core.DATASET_PATH)
        == protocol["dataset"]["dataset_file_sha256"]
        and core.sha256_json(dataset) == protocol["dataset"]["dataset_content_sha256"],
        "seed registry dataset differs from preregistration",
    )
    items = dataset.get("items")
    core.require(isinstance(items, list) and len(items) == core.ITEM_COUNT,
                 "seed registry dataset denominator changed")
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        core.require(isinstance(item, dict), "seed registry item is not an object")
        seed_key = item.get("seed_key")
        core.require(isinstance(seed_key, str) and bool(seed_key),
                     "seed registry item lacks seed_key")
        blocks[seed_key].append(item)
    core.require(len(blocks) == core.SEMANTIC_INSTANCE_COUNT,
                 "unique seed-key denominator changed")
    for seed_key, twins in blocks.items():
        core.require(
            len(twins) == 2
            and {str(item.get("swap_side")) for item in twins} == set(core.SWAP_SIDES)
            and len({str(item.get("semantic_id")) for item in twins}) == 1
            and len({str(item.get("id")) for item in twins}) == 2,
            f"seed block is not one option-swap twin: {seed_key}",
        )

    entries: list[dict[str, Any]] = []
    per_model_unique: dict[str, int] = {}
    per_model_collisions: dict[str, int] = {}
    for model_key in core.MODEL_ORDER:
        model_seeds: list[int] = []
        for stream in core.STREAMS:
            for seed_key in sorted(blocks):
                seed = core.stable_pair_seed(
                    protocol["protocol_sha256"], model_key, seed_key, stream)
                core.require(
                    seed == core.stable_pair_seed(
                        protocol["protocol_sha256"], model_key, seed_key, stream,
                        core.ARM_A)
                    == core.stable_pair_seed(
                        protocol["protocol_sha256"], model_key, seed_key, stream,
                        core.ARM_B),
                    "arm entered the frozen pair seed",
                )
                model_seeds.append(seed)
                entries.append({
                    "model_key": model_key,
                    "stream": stream,
                    "seed_key": seed_key,
                    "pair_seed": seed,
                })
        unique = len(set(model_seeds))
        collision_count = len(model_seeds) - unique
        per_model_unique[model_key] = unique
        per_model_collisions[model_key] = collision_count
        core.require(len(model_seeds) == 384 and unique == 384 and collision_count == 0,
                     f"unexpected formal seed collision: {model_key}")
    core.require(len(entries) == 1152, "all-model seed registry denominator changed")
    return {
        "algorithm_contract": deepcopy(protocol["seed_contract"]),
        "dataset_namespace": protocol["protocol_sha256"],
        "entry_count": len(entries),
        "unique_seed_key_count": len(blocks),
        "blocks_per_model": 384,
        "per_model_unique_seed_count": per_model_unique,
        "per_model_collision_count": per_model_collisions,
        "collision_free_within_each_model": True,
        "cross_model_seed_equality_is_not_a_pairing_claim": True,
        "seed_registry_sha256": core.sha256_json(entries),
    }


def expected_authorization_scope(seed_registry_sha256: str) -> dict[str, Any]:
    return {
        "models": list(core.MODEL_ORDER),
        "strict_order": list(core.MODEL_ORDER),
        "one_model_subprocess_at_a_time": True,
        "experiment": core.EXPERIMENT,
        "items": core.ITEM_COUNT,
        "arms": deepcopy(core.ARMS),
        "streams": list(core.STREAMS),
        "sampling": deepcopy(core.SAMPLING),
        "quantization": deepcopy(core.QUANTIZATION),
        "decision_checkpoint": core.DECISION_CHECKPOINT,
        "expected_rows_per_model": core.EXPECTED_ROWS_PER_MODEL,
        "expected_rows_all_models": core.EXPECTED_ROWS_ALL_MODELS,
        "seed_registry_sha256": seed_registry_sha256,
        "external_generation_only": True,
        "activation_collection": False,
        "internal_intervention": False,
    }


def expected_pre_generation_state() -> dict[str, bool]:
    return {
        "model_weights_loaded_by_admission": False,
        "formal_generation_performed": False,
        "formal_rows_exist": False,
        "gpu_used_by_admission": False,
    }


def expected_decision_boundary() -> dict[str, bool]:
    return {
        "independent_cpu_audit_required": True,
        "runner_cannot_compute_scientific_decision": True,
        "no_pooling": True,
        "secondary_cannot_override_primary": True,
        "pass_does_not_authorize_holdout_or_mechanism": True,
    }


def authenticate_qualification(protocol: dict[str, Any]) -> dict[str, Any]:
    document = core.load_json(core.QUALIFICATION_PATH, "engineering qualification")
    qualification_builder.verify_existing(document, protocol)
    core.require(document.get("qualification_passed") is True,
                 "engineering qualification did not pass all three models")
    core.require(document.get("formal_dataset_used") is False
                 and document.get("formal_generation_performed") is False,
                 "qualification crossed into formal data/generation")
    return document


def build_payload() -> dict[str, Any]:
    protocol = authenticate_protocol()
    qualification_builder.recover_stale_lock_if_present(
        protocol["protocol_sha256"])
    qualification = authenticate_qualification(protocol)
    seed_registry = build_seed_registry(protocol)
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "decision": "ADMIT_SEQUENTIAL_CROSS_MODEL_EXTERNAL_GENERATION",
        "admitted": True,
        "gpu_authorized": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
        "qualification_sha256": qualification["qualification_sha256"],
        "qualification_file_sha256": core.sha256_file(core.QUALIFICATION_PATH),
        "seed_registry": seed_registry,
        "authorization_scope": expected_authorization_scope(
            seed_registry["seed_registry_sha256"]),
        "pre_generation_state": expected_pre_generation_state(),
        "decision_boundary": expected_decision_boundary(),
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
        "cpu_only": True,
        "model_weights_loaded": False,
        "formal_generation_performed": False,
        "gpu_used": False,
    }
    return payload


def verify_payload(
    payload: Any, expected_payload: dict[str, Any] | None = None,
) -> None:
    core.require(isinstance(payload, dict), "admission payload missing")
    expected = build_payload() if expected_payload is None else expected_payload
    core.require(isinstance(expected, dict) and payload == expected,
                 "admission differs from complete deterministic reconstruction")
    expected_keys = {
        "schema_version", "phase", "experiment", "decision", "admitted",
        "gpu_authorized", "protocol_sha256", "protocol_file_sha256",
        "qualification_sha256", "qualification_file_sha256", "seed_registry",
        "authorization_scope", "pre_generation_state", "decision_boundary",
        "holdout", "holdout_loaded", "holdout_authorized", "mechanism",
        "mechanism_authorized", "cpu_only", "model_weights_loaded",
        "formal_generation_performed", "gpu_used",
    }
    core.require(set(payload) == expected_keys, "admission top-level schema changed")
    core.require(payload.get("schema_version") == core.SCHEMA_VERSION
                 and payload.get("phase") == core.PHASE
                 and payload.get("experiment") == core.EXPERIMENT
                 and payload.get("decision")
                 == "ADMIT_SEQUENTIAL_CROSS_MODEL_EXTERNAL_GENERATION"
                 and payload.get("admitted") is True
                 and payload.get("gpu_authorized") is True,
                 "formal admission decision changed")
    core.require(_is_sha256(payload.get("protocol_sha256"))
                 and _is_sha256(payload.get("protocol_file_sha256"))
                 and _is_sha256(payload.get("qualification_sha256"))
                 and _is_sha256(payload.get("qualification_file_sha256")),
                 "admission lineage hashes malformed")
    seed_registry = payload.get("seed_registry")
    core.require(isinstance(seed_registry, dict) and set(seed_registry) == {
        "algorithm_contract", "dataset_namespace", "entry_count",
        "unique_seed_key_count", "blocks_per_model", "per_model_unique_seed_count",
        "per_model_collision_count", "collision_free_within_each_model",
        "cross_model_seed_equality_is_not_a_pairing_claim", "seed_registry_sha256",
    }, "admission seed registry schema changed")
    core.require(
        seed_registry.get("algorithm_contract") == protocol_builder.seed_contract()
        and seed_registry.get("dataset_namespace") == payload["protocol_sha256"]
        and seed_registry.get("entry_count") == 1152
        and seed_registry.get("unique_seed_key_count") == core.SEMANTIC_INSTANCE_COUNT
        and seed_registry.get("blocks_per_model") == 384
        and seed_registry.get("per_model_unique_seed_count")
        == {model: 384 for model in core.MODEL_ORDER}
        and seed_registry.get("per_model_collision_count")
        == {model: 0 for model in core.MODEL_ORDER}
        and seed_registry.get("collision_free_within_each_model") is True
        and seed_registry.get("cross_model_seed_equality_is_not_a_pairing_claim") is True
        and _is_sha256(seed_registry.get("seed_registry_sha256")),
        "admission seed registry changed",
    )
    scope = payload.get("authorization_scope")
    core.require(scope == expected_authorization_scope(
        seed_registry["seed_registry_sha256"]),
                 "formal authorization scope changed")
    state = payload.get("pre_generation_state")
    core.require(state == expected_pre_generation_state(),
                 "admission claims formal generation/model/GPU use")
    boundary = payload.get("decision_boundary")
    core.require(boundary == expected_decision_boundary(),
                 "scientific decision boundary weakened")
    core.require(payload.get("holdout") is False
                 and payload.get("holdout_loaded") is False
                 and payload.get("holdout_authorized") is False
                 and payload.get("mechanism") is False
                 and payload.get("mechanism_authorized") is False
                 and payload.get("cpu_only") is True
                 and payload.get("model_weights_loaded") is False
                 and payload.get("formal_generation_performed") is False
                 and payload.get("gpu_used") is False,
                 "admission scope/runtime state widened")


def negative_tests(
    payload: dict[str, Any], expected_payload: dict[str, Any],
) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    mutations = {
        "order_change_rejected": lambda value: value["authorization_scope"].__setitem__(
            "strict_order", list(reversed(core.MODEL_ORDER))),
        "pooling_rejected": lambda value: value["decision_boundary"].__setitem__(
            "no_pooling", False),
        "mechanism_rejected": lambda value: value.__setitem__(
            "mechanism_authorized", True),
        "pregenerated_rows_rejected": lambda value: value["pre_generation_state"].__setitem__(
            "formal_rows_exist", True),
        "quantization_change_rejected": lambda value: value["authorization_scope"][
            "quantization"].__setitem__("load_in_8bit", False),
        "missing_protocol_lineage_rejected": lambda value: value.pop("protocol_sha256"),
        "item_denominator_change_rejected": lambda value: value[
            "authorization_scope"].__setitem__("items", core.ITEM_COUNT - 1),
        "stream_registry_change_rejected": lambda value: value[
            "authorization_scope"].__setitem__("streams", [0, 1]),
        "per_model_rows_change_rejected": lambda value: value[
            "authorization_scope"].__setitem__(
                "expected_rows_per_model", core.EXPECTED_ROWS_PER_MODEL - 1),
        "seed_registry_rehash_rejected": lambda value: value[
            "seed_registry"].__setitem__("seed_registry_sha256", "0" * 64),
        "state_schema_deletion_rejected": lambda value: value[
            "pre_generation_state"].pop("formal_rows_exist"),
        "decision_boundary_schema_deletion_rejected": lambda value: value[
            "decision_boundary"].pop("no_pooling"),
        "extra_authorization_key_rejected": lambda value: value[
            "authorization_scope"].__setitem__("extra", True),
    }
    for name, mutate in mutations.items():
        candidate = deepcopy(payload)
        mutate(candidate)
        try:
            verify_payload(candidate, expected_payload)
        except (RuntimeError, KeyError, TypeError):
            tests[name] = True
        else:
            tests[name] = False
    rehashed_payload = deepcopy(payload)
    rehashed_payload["authorization_scope"]["items"] = core.ITEM_COUNT - 1
    rehashed_document = {
        **rehashed_payload,
        "admission_sha256": core.sha256_json(rehashed_payload),
        "created_at_utc": "2000-01-01T00:00:00+00:00",
    }
    core.verify_self_hash(rehashed_document, "admission_sha256", "created_at_utc",
                          "synthetic rehashed admission")
    try:
        verify_payload(rehashed_payload, expected_payload)
    except (RuntimeError, KeyError, TypeError):
        tests["self_rehashed_payload_tamper_rejected"] = True
    else:
        tests["self_rehashed_payload_tamper_rejected"] = False
    core.require(all(tests.values()), "admission negative test failed")
    return tests


def install(
    payload: dict[str, Any], expected_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    verify_payload(payload, expected_payload)
    if core.ADMISSION_PATH.exists():
        existing = core.load_json(core.ADMISSION_PATH, "existing Phase983 admission")
        core.verify_self_hash(existing, "admission_sha256", "created_at_utc",
                              "Phase983 admission")
        core.require(core.without_fields(existing, "admission_sha256", "created_at_utc")
                     == payload, "existing admission differs from current derivation")
        return existing, False
    core.require(not any(core.manifest_path(model).exists()
                         or core.rows_path(model).exists()
                         or core.status_path(model).exists()
                         for model in core.MODEL_ORDER),
                 "formal output exists before admission")
    document = {
        **payload,
        "admission_sha256": core.sha256_json(payload),
        "created_at_utc": core.utc_now(),
    }
    core.atomic_write_json(core.ADMISSION_PATH, document)
    installed = core.load_json(core.ADMISSION_PATH, "installed Phase983 admission")
    core.verify_self_hash(installed, "admission_sha256", "created_at_utc",
                          "installed Phase983 admission")
    core.require(installed == document,
                 "installed admission differs after JSON serialization")
    return installed, True


def run(write: bool) -> dict[str, Any]:
    payload = build_payload()
    expected_payload = build_payload()
    verify_payload(payload, expected_payload)
    tests = negative_tests(payload, expected_payload)
    result = {
        "admission_payload_sha256": core.sha256_json(payload),
        "negative_tests": tests,
        "cpu_only": True,
        "model_weights_loaded": False,
        "gpu_used": False,
        "formal_generation_performed": False,
        "files_written": False,
    }
    if write:
        document, created = install(payload, expected_payload)
        result.update({
            "admission_sha256": document["admission_sha256"],
            "admission_file_sha256": core.sha256_file(core.ADMISSION_PATH),
            "files_written": created,
            "existing": not created,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.write), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
