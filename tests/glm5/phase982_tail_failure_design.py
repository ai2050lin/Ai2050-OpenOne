#!/usr/bin/env python3
"""CPU-only post-hoc description of Phase981 right-tail failures.

This script has no admission authority.  It authenticates the frozen Phase981
chain, recomputes descriptive checkpoint/tail tables from all 1536 sealed rows,
and preserves the Phase981 NO-GO without changing any threshold or state rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as boundary  # noqa: E402
import phase981_confirmation_core as p981  # noqa: E402


PHASE = 982
SCHEMA_VERSION = 1
EXPERIMENT = "phase981_tail_failure_posthoc_description"
PHASE981_OUT = GLM5 / "result" / "phase981_fresh256_confirmation"
OUT = GLM5 / "result" / "phase982_tail_failure_design"
REPORT_PATH = OUT / "report.json"
SCRIPT_PATH = GLM5 / "phase982_tail_failure_design.py"

SOURCE_PATHS = {
    "protocol": PHASE981_OUT / "protocol_preregistration.json",
    "admission": PHASE981_OUT / "generation_admission.json",
    "manifest": PHASE981_OUT / "manifest_confirmation.json",
    "status": PHASE981_OUT / "generator_status_confirmation.json",
    "audit": PHASE981_OUT / "confirmation_audit.json",
    "rows": PHASE981_OUT / "rows_confirmation.jsonl",
    "dataset": PHASE981_OUT / "dataset.json",
    "dataset_audit": PHASE981_OUT / "audit.json",
}

DOCUMENT_HASH_FIELDS = {
    "protocol": ("protocol_sha256", "created_at_utc"),
    "admission": ("admission_sha256", "admitted_at_utc"),
    "manifest": ("manifest_sha256", "created_at_utc"),
    "status": ("status_sha256", "updated_at_utc"),
    "audit": ("audit_sha256", "audited_at_utc"),
}

ADJACENT_CHECKPOINTS = tuple(zip(p981.CHECKPOINTS[:-1], p981.CHECKPOINTS[1:]))

EXPECTED_PHASE981_SELF_HASHES = {
    "protocol": "be459ea62a21537e029d54059a6bf5aca09a53ab04667532001dc5a05aeec4b2",
    "admission": "b8f6dea42aa258813e4e1b22e7bbce8bde30799918ec6d5b14bee155ee04eb05",
    "manifest": "53aff0a990951cf9cfff09e6793088ad6cc4c5ad0187824375a49c36ad9153d0",
    "status": "4777106303341ca5c77b50e21e6c12d29027679ddf760cf4370090d510b4833b",
    "audit": "100f2a568e9275a4ecfeac50c583cad1d54cadd4bb9730f4ee67cf7fa9cb5189",
}

EXPECTED_PHASE981_FILE_HASHES = {
    "protocol": "8557c658b74f3493618116775d5743e193af71d67bf9dc724e2b3f1c6057fb52",
    "admission": "83d72011b424cc5d9df82c4547a47aa0a85ced96382ac4dae2a23686919002a9",
    "manifest": "41e1293bbe5e0105a71e03c7c1589d2826527a9f07869be4d122aa8da029583c",
    "status": "672b09ba7e85d251a92a299d10f275949eb30f5ba7a9153aecf32b6a985a1563",
    "audit": "834c304734591bdfd6cb9d047664884f61a3a17ddf52adda312179686f0b629a",
    "rows": "baa0afaf6fa5bfe3080564b6c909fc04057fc42c983a9342f8665615f5671245",
    "dataset": "e146cbb6173cd7bfe8d8d4f148844d18a602ccd24598a90640650991cb66f49b",
    "dataset_audit": "ddcbaf9d389510864846c50c05f31c3cbf22e98405222cf177cb5e3a7a60580e",
}

METHOD_BOUNDARY = {
    "posthoc": True,
    "descriptive_only": True,
    "non_admission": True,
    "no_gate_or_threshold_design": True,
    "no_future_gpu_protocol_or_admission_created": True,
    "same_256_items_repeated_across_three_seed_streams": True,
    "stream_appearances_are_not_independent_items": True,
    "no_statistical_or_mechanism_claim": True,
    "no_4096_token_extrapolation": True,
}

EXPECTED_DECISION_PRESERVATION_FIELDS = {
    "source_audit_sha256": EXPECTED_PHASE981_SELF_HASHES["audit"],
    "source_fresh_confirmation_passed": False,
    "phase981_NO_GO_unchanged": True,
    "phase981_gate_reopened": False,
    "thresholds_changed": False,
    "terminal_state_rules_changed": False,
    "secondary_censor_can_change_primary": False,
}

REPORT_TAMPER_TEST_NAMES = (
    "rehashed_gate_reopened_true_rejected",
    "rehashed_terminal_rules_true_rejected",
    "rehashed_descriptive_only_false_rejected",
    "rehashed_no_statistical_claim_false_rejected",
    "rehashed_script_sha_forgery_rejected",
    "rehashed_tail_C_count_change_rejected",
    "rehashed_source_hash_tamper_rejected",
    "rehashed_source_count_tamper_rejected",
)

# The fail-open report produced by the immediately preceding Phase982 script is
# the only identity eligible for a one-time in-place upgrade.  Any other prior
# report remains fail-closed.
SUPERSEDED_REPORTS = {
    "c8b203aa169cb9546a082ce79a4da856d72724d19fb9d1630f8bea588ef8be95": {
        "file_sha256":
            "ac4118c666185fae1ef3955cce776f0a14fedab410c790ea701cb5e484ff716b",
        "script_sha256":
            "1a6f3ae636c6e7ce573f3bfe6d12c3ebc7ba86ec258095a1e03ee96c0690fbae",
    },
}

EXPECTED_B_C_BY_CHECKPOINT = {
    "stream_0": {"256": 241, "512": 159, "1024": 57, "1536": 28, "2048": 22},
    "stream_1": {"256": 242, "512": 161, "1024": 64, "1536": 26, "2048": 14},
    "stream_2": {"256": 244, "512": 157, "1024": 64, "1536": 25, "2048": 18},
}

EXPECTED_1536_TO_2048_C_DESTINATIONS = {
    "stream_0": {"V": 2, "C": 22, "I_mode": 0, "I_sem": 4},
    "stream_1": {"V": 5, "C": 14, "I_mode": 0, "I_sem": 7},
    "stream_2": {"V": 4, "C": 18, "I_mode": 0, "I_sem": 3},
}

EXPECTED_1536_TO_2048_NON_C_STABILITY = {
    "stream_0": {"V_to_V": 210, "I_sem_to_I_sem": 18, "non_C_to_C": 0},
    "stream_1": {"V_to_V": 211, "I_sem_to_I_sem": 19, "non_C_to_C": 0},
    "stream_2": {"V_to_V": 213, "I_sem_to_I_sem": 18, "non_C_to_C": 0},
}

EXPECTED_TAIL_TASK_COUNTS = {
    "stream_0": {
        "boolean_logic": 0, "constraint_order": 0, "modular_arithmetic": 3,
        "multistep_arithmetic": 0, "relation_path": 0, "sequence_rule": 6,
        "state_machine": 4, "string_transform": 9,
    },
    "stream_1": {
        "boolean_logic": 0, "constraint_order": 0, "modular_arithmetic": 1,
        "multistep_arithmetic": 0, "relation_path": 0, "sequence_rule": 4,
        "state_machine": 4, "string_transform": 5,
    },
    "stream_2": {
        "boolean_logic": 0, "constraint_order": 0, "modular_arithmetic": 2,
        "multistep_arithmetic": 0, "relation_path": 0, "sequence_rule": 5,
        "state_machine": 3, "string_transform": 8,
    },
}

EXPECTED_TAIL_DIFFICULTY_COUNTS = {
    "stream_0": {"easy": 6, "hard": 16},
    "stream_1": {"easy": 1, "hard": 13},
    "stream_2": {"easy": 6, "hard": 12},
}

EXPECTED_TAIL_OVERLAP = {
    "three_stream_intersection_n": 4,
    "three_stream_union_n": 32,
    "pairwise_intersection_counts": {
        "stream_0_intersection_stream_1": 8,
        "stream_0_intersection_stream_2": 10,
        "stream_1_intersection_stream_2": 8,
    },
    "membership_frequency_counts": {
        "in_exactly_1_streams": 14,
        "in_exactly_2_streams": 14,
        "in_exactly_3_streams": 4,
    },
}

EXPECTED_TAIL_ITEM_SET_HASHES = {
    "stream_0": "f67d32f0373e5abb9b07c88d3cdf34689552f12b08d496a85b2835910a3be353",
    "stream_1": "079f5e80793108dcd18c19d6e114a3a9b8638999dfad96cdb97077d221a58183",
    "stream_2": "c08e58cbfa20e8ad73e4413f6b8f162774613d0073d4074dcf26bc9817bd6ec4",
    "stream_0_intersection_stream_1": "5be454d9366afc4f5f1c4030b61d3687d703d21cd1830788094fd11546c0c37f",
    "stream_0_intersection_stream_2": "03e6e398e03006adcf7bec870cbcc9c10f130099fde8a91a9bfd2955bff19ec9",
    "stream_1_intersection_stream_2": "b15469d548c0327074d5205d7fa80b674390820b377b0ba9e7291429622bead7",
    "three_stream_intersection": "cc0ee19c6e77e36d1f3fc09f350e869ece586250269eea262942a4b6864f3f84",
    "three_stream_union": "2e40e63544b2a6b0f2f2e11903ea33af6420e3d8a60ff86d98fe7d28e6e3e49b",
}

EXPECTED_TAIL_ANSWER_COUNTS = {
    "stream_0": {"A": 19, "B": 3},
    "stream_1": {"A": 11, "B": 3},
    "stream_2": {"A": 15, "B": 3},
    "three_stream_union": {"A": 26, "B": 6},
    "three_stream_intersection": {"A": 4, "B": 0},
}

EXPECTED_VENN_PATTERN_COUNTS = {
    "000": 224, "100": 8, "010": 2, "001": 4,
    "110": 4, "101": 6, "011": 4, "111": 4,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def assert_no_holdout_import() -> None:
    forbidden = [
        name for name in sys.modules
        if name == "phase977_holdout_dataset"
        or name.endswith(".phase977_holdout_dataset")
    ]
    require(not forbidden, f"old holdout module imported: {forbidden}")


def assert_no_model_runtime_import() -> None:
    forbidden = [
        name for name in ("torch", "transformers", "model_utils")
        if name in sys.modules
    ]
    require(not forbidden, f"model runtime module imported: {forbidden}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite JSON constant: {value}")


def strict_json_loads(payload: str, label: str) -> Any:
    try:
        return json.loads(
            payload, object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {label}") from exc


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def verify_self_hash(
    document: dict[str, Any], hash_field: str, time_field: str | None,
    label: str,
) -> None:
    blocked = {hash_field}
    if time_field is not None:
        blocked.add(time_field)
    payload = {key: value for key, value in document.items() if key not in blocked}
    require(document.get(hash_field) == boundary.sha256_json(payload),
            f"{label} self-hash invalid")


def report_payload(document: dict[str, Any]) -> dict[str, Any]:
    """Return the timestamp-independent, self-hashable report identity."""
    return {
        key: value for key, value in document.items()
        if key not in {"report_sha256", "created_at_utc"}
    }


def reseal_report(document: dict[str, Any]) -> None:
    """Recompute a report self-hash after an intentional test mutation."""
    document["report_sha256"] = boundary.sha256_json(report_payload(document))


def validate_exact_key_registry(
    actual: list[tuple[str, str, int]],
    expected: list[tuple[str, str, int]],
) -> None:
    require(len(actual) == p981.EXPECTED_ROWS,
            f"row denominator changed: {len(actual)}")
    require(len(set(actual)) == len(actual), "duplicate row key")
    require(set(actual) == set(expected), "row key set differs from frozen grid")
    require(actual == expected, "row order differs from frozen canonical grid")


def validate_row_self_hash(row: dict[str, Any], line_number: int) -> None:
    require(row.get("row_sha256") == boundary.sha256_json(
        boundary.without_fields(row, "row_sha256")),
        f"row self-hash mismatch: line {line_number}")


def validate_checkpoint_structure(
    row: dict[str, Any], eos_ids: set[int], line_number: int,
) -> None:
    generated = row.get("generated_ids")
    require(isinstance(generated, list) and generated
            and all(isinstance(value, int) and not isinstance(value, bool)
                    for value in generated),
            f"invalid generated IDs: line {line_number}")
    eos_positions = [
        index for index, value in enumerate(generated) if int(value) in eos_ids
    ]
    require(
        (len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
        or (not eos_positions and len(generated) == p981.MAX_NEW_TOKENS),
        f"invalid first-EOS/cap termination: line {line_number}",
    )
    checkpoints = row.get("checkpoints")
    require(isinstance(checkpoints, dict)
            and set(checkpoints) == {str(value) for value in p981.CHECKPOINTS},
            f"checkpoint registry changed: line {line_number}")
    for budget in p981.CHECKPOINTS:
        snapshot = generated[:budget]
        checkpoint = checkpoints[str(budget)]
        require(isinstance(checkpoint, dict),
                f"checkpoint is not an object: line {line_number}/{budget}")
        snapshot_eos = [
            index for index, value in enumerate(snapshot) if int(value) in eos_ids
        ]
        require(checkpoint.get("budget") == budget
                and checkpoint.get("n_tokens") == len(snapshot)
                and checkpoint.get("eos_positions") == snapshot_eos
                and checkpoint.get("has_eos") is bool(snapshot_eos)
                and checkpoint.get("first_eos_position")
                == (snapshot_eos[0] if snapshot_eos else None)
                and checkpoint.get("t_eos_step")
                == (snapshot_eos[0] + 1 if snapshot_eos else None)
                and checkpoint.get("hit_budget")
                is (not snapshot_eos and len(snapshot) == budget)
                and checkpoint.get("terminal_state") in p981.SIX_STATES,
                f"checkpoint structural mismatch: line {line_number}/{budget}")


def read_and_validate_rows(
    protocol: dict[str, Any], admission: dict[str, Any],
    manifest: dict[str, Any], audit: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    path = SOURCE_PATHS["rows"]
    require(path.is_file(), "missing Phase981 rows")
    raw_payload = path.read_bytes()
    require(raw_payload and raw_payload.endswith(b"\n"),
            "rows JSONL is empty or lacks final newline")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_payload.splitlines(), 1):
        try:
            row = strict_json_loads(raw.decode("utf-8"), f"row {line_number}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed row: line {line_number}") from exc
        require(isinstance(row, dict), f"row is not an object: line {line_number}")
        validate_row_self_hash(row, line_number)
        rows.append(row)

    expected_grid = p981.canonical_grid(items)
    expected_keys = [
        (str(item["id"]), arm, stream)
        for item, arm, stream in expected_grid
    ]
    actual_keys = [p981.row_key(row) for row in rows]
    validate_exact_key_registry(actual_keys, expected_keys)

    item_by_id = {str(item["id"]): item for item in items}
    grid_positions = {key: index for index, key in enumerate(expected_keys)}
    eos_ids = {int(value) for value in manifest.get("eos_token_ids", [])}
    require(eos_ids == set(p981.EXPECTED_EOS_TOKEN_IDS),
            "Phase981 effective EOS registry changed")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    cell_counts: Counter[tuple[str, int]] = Counter()
    for line_number, row in enumerate(rows, 1):
        key = actual_keys[line_number - 1]
        item = item_by_id[key[0]]
        arm, stream = key[1], key[2]
        validate_checkpoint_structure(row, eos_ids, line_number)
        p981.verify_row_generation_contract(row)
        require(
            row.get("schema_version") == p981.SCHEMA_VERSION
            and row.get("phase") == p981.PHASE
            and row.get("experiment") == p981.EXPERIMENT
            and row.get("protocol_sha256") == protocol["protocol_sha256"]
            and row.get("admission_sha256") == admission["admission_sha256"]
            and row.get("manifest_sha256") == manifest["manifest_sha256"]
            and row.get("id") == item["id"]
            and row.get("task") == item["task"]
            and row.get("difficulty") == item["difficulty"]
            and row.get("prompt") == item["prompt"]
            and row.get("answer") == item["answer"]
            and row.get("arm_spec") == p981.ARMS[arm]
            and row.get("stream") == stream
            and row.get("pair_id") == p981.pair_id(item["id"], stream)
            and row.get("pair_seed") == p981.stable_pair_seed(
                manifest["dataset_identity"]["identity_sha256"],
                item["id"], stream)
            and row.get("sampling") == p981.SAMPLING
            and row.get("batch_index")
            == grid_positions[key] // p981.BATCH_SIZE + 1,
            f"row lineage/metadata differs: {key}",
        )
        records[key] = row
        cell_counts[(arm, stream)] += 1

    expected_cells = Counter({
        (arm, stream): p981.ITEM_COUNT
        for stream in p981.STREAMS for arm in p981.ARMS
    })
    require(cell_counts == expected_cells, "six arm/stream cells changed")
    for stream in p981.STREAMS:
        for item in items:
            left = records[(item["id"], "A", stream)]
            right = records[(item["id"], "B", stream)]
            require(left["pair_id"] == right["pair_id"]
                    and left["pair_seed"] == right["pair_seed"],
                    f"A/B pair coupling changed: {item['id']}/stream_{stream}")

    rows_file_sha = boundary.sha256_file(path)
    row_audit = audit.get("row_audit", {})
    require(rows_file_sha == row_audit.get("rows_file_sha256")
            and row_audit.get("row_count") == p981.EXPECTED_ROWS
            and row_audit.get("all_row_self_hashes_valid") is True
            and row_audit.get("all_prefixes_and_checkpoints_recomputed") is True
            and row_audit.get("all_first_eos_absorbing") is True
            and row_audit.get("all_A_B_pair_seeds_equal") is True,
            "Phase981 independent row audit contract changed")
    return records, {
        "rows_file_sha256": rows_file_sha,
        "row_count": len(records),
        "unique_row_key_count": len(records),
        "canonical_order": True,
        "cell_counts": {
            f"{arm}|stream_{stream}": cell_counts[(arm, stream)]
            for stream in p981.STREAMS for arm in p981.ARMS
        },
        "pair_count": p981.ITEM_COUNT * len(p981.STREAMS),
        "all_A_B_pair_seeds_equal": True,
        "first_eos_and_checkpoint_structure_verified": True,
        "phase981_independent_checkpoint_recomputation_authenticated": True,
    }


def authenticate_phase981() -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]], dict[str, Any],
]:
    assert_no_holdout_import()
    for name, expected in EXPECTED_PHASE981_FILE_HASHES.items():
        path = SOURCE_PATHS[name]
        require(path.is_file() and boundary.sha256_file(path) == expected,
                f"frozen Phase981 file changed: {name}")
    documents = {
        name: load_json(SOURCE_PATHS[name], f"Phase981 {name}")
        for name in DOCUMENT_HASH_FIELDS
    }
    for name, (hash_field, time_field) in DOCUMENT_HASH_FIELDS.items():
        verify_self_hash(documents[name], hash_field, time_field,
                         f"Phase981 {name}")
        require(documents[name].get(hash_field)
                == EXPECTED_PHASE981_SELF_HASHES[name],
                f"frozen Phase981 identity changed: {name}")
    protocol = documents["protocol"]
    admission = documents["admission"]
    manifest = documents["manifest"]
    status = documents["status"]
    audit = documents["audit"]

    seal_audit = p981.verify_protocol_file_seals(protocol)
    p981.verify_protocol_integrity_metadata(protocol)
    p981.verify_protocol_boundary_contract(protocol)
    p981.verify_admission_boundary_contract(admission)
    p981.verify_manifest_dependency_contract(manifest, protocol)
    p981.verify_complete_status_generation_contract(status)
    require(protocol.get("phase") == admission.get("phase")
            == manifest.get("phase") == status.get("phase")
            == audit.get("phase") == p981.PHASE,
            "Phase981 source phase changed")
    require(admission.get("protocol_sha256") == protocol["protocol_sha256"]
            and manifest.get("protocol_sha256") == protocol["protocol_sha256"]
            and manifest.get("admission_sha256") == admission["admission_sha256"]
            and status.get("protocol_sha256") == protocol["protocol_sha256"]
            and status.get("admission_sha256") == admission["admission_sha256"]
            and status.get("manifest_sha256") == manifest["manifest_sha256"]
            and audit.get("protocol_sha256") == protocol["protocol_sha256"]
            and audit.get("admission_sha256") == admission["admission_sha256"]
            and audit.get("manifest_sha256") == manifest["manifest_sha256"]
            and audit.get("status_sha256") == status["status_sha256"],
            "Phase981 artifact lineage mismatch")
    require(manifest.get("protocol_file_sha256")
            == boundary.sha256_file(SOURCE_PATHS["protocol"])
            and manifest.get("admission_file_sha256")
            == boundary.sha256_file(SOURCE_PATHS["admission"]),
            "Phase981 manifest source file hash mismatch")

    dataset = load_json(SOURCE_PATHS["dataset"], "Phase981 dataset")
    dataset_audit = load_json(
        SOURCE_PATHS["dataset_audit"], "Phase981 dataset audit")
    verify_self_hash(dataset, "dataset_sha256", None, "Phase981 dataset")
    verify_self_hash(dataset_audit, "audit_sha256", None,
                     "Phase981 dataset audit")
    dataset_seals = protocol.get("dataset_artifact_seals", {})
    require(boundary.sha256_file(SOURCE_PATHS["dataset"])
            == dataset_seals.get("dataset", {}).get("file_sha256")
            and dataset.get("dataset_sha256")
            == dataset_seals.get("dataset", {}).get("dataset_sha256")
            and boundary.sha256_file(SOURCE_PATHS["dataset_audit"])
            == dataset_seals.get("audit", {}).get("file_sha256")
            and dataset_audit.get("audit_sha256")
            == dataset_seals.get("audit", {}).get("audit_sha256"),
            "Phase981 dataset artifact seal mismatch")
    items = dataset.get("items")
    require(isinstance(items, list) and len(items) == p981.ITEM_COUNT,
            "Phase981 dataset item denominator changed")
    require(dataset.get("identity", {}).get("identity_sha256")
            == protocol.get("dataset_identity", {}).get("identity_sha256")
            == manifest.get("dataset_identity", {}).get("identity_sha256")
            == audit.get("dataset_identity", {}).get("identity_sha256"),
            "Phase981 dataset identity lineage mismatch")
    require(dataset_audit.get("passed") is True
            and dataset_audit.get("generation_performed") is False
            and dataset_audit.get("gpu_used") is False
            and dataset_audit.get("model_weights_loaded") is False
            and dataset_audit.get("holdout_accessed") is False,
            "Phase981 dataset audit boundary changed")

    gate_result = audit.get("semantic_gate_result", {})
    require(audit.get("semantic_gate_contract_sha256")
            == protocol.get("semantic_gate", {}).get("contract_sha256"),
            "Phase981 gate contract lineage mismatch")
    require(audit.get("fresh_confirmation_passed") is False
            and gate_result.get("primary_semantic_passed") is False
            and audit.get("fresh_confirmation_passed")
            is gate_result.get("primary_semantic_passed")
            and audit.get("secondary_censor_can_change_primary") is False
            and gate_result.get("secondary_censor_can_set_primary") is False
            and gate_result.get("no_route_or_in_primary") is True,
            "Phase981 final NO-GO/route boundary changed")
    require(protocol.get("primary_decision", {}).get("route") == "semantic_only"
            and protocol.get("primary_decision", {}).get(
                "censor_route_can_admit") is False
            and protocol.get("primary_decision", {}).get(
                "mixed_route_can_admit") is False,
            "Phase981 primary route boundary changed")
    for name, document in documents.items():
        require(document.get("holdout") is False
                and document.get("holdout_loaded") is False
                and document.get("mechanism") is False
                and document.get("mechanism_authorized") is False,
                f"Phase981 {name} crossed holdout/mechanism boundary")
    require(audit.get("model_weights_loaded_by_auditor") is False
            and audit.get("gpu_used_by_auditor") is False,
            "Phase981 independent auditor boundary changed")

    records, row_authentication = read_and_validate_rows(
        protocol, admission, manifest, audit, items)
    source_authentication = {
        "artifacts": {
            name: {
                "path": relative(path),
                "file_sha256": boundary.sha256_file(path),
                **({DOCUMENT_HASH_FIELDS[name][0]: documents[name][
                    DOCUMENT_HASH_FIELDS[name][0]]}
                   if name in DOCUMENT_HASH_FIELDS else {}),
            }
            for name, path in SOURCE_PATHS.items()
        },
        "code_seal_audit": seal_audit,
        "row_authentication": row_authentication,
        "all_document_self_hashes_valid": True,
        "all_lineage_and_key_file_hashes_valid": True,
        "phase981_final_audit_authenticated": True,
    }
    assert_no_holdout_import()
    return documents, items, records, source_authentication


def state_at(row: dict[str, Any], checkpoint: int) -> str:
    state = str(row["checkpoints"][str(checkpoint)]["terminal_state"])
    require(state in p981.SIX_STATES, f"unknown terminal state: {state}")
    return state


def count_registry(labels: Iterable[str], values: Iterable[str]) -> dict[str, int]:
    counter = Counter(values)
    require(set(counter).issubset(set(labels)), "count outside frozen registry")
    return {label: int(counter[label]) for label in labels}


def checkpoint_state_tables(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in p981.ARMS:
        arm_output: dict[str, Any] = {}
        for stream in p981.STREAMS:
            checkpoint_output: dict[str, Any] = {}
            for checkpoint in p981.CHECKPOINTS:
                states = [
                    state_at(records[(item["id"], arm, stream)], checkpoint)
                    for item in items
                ]
                require(len(states) == p981.ITEM_COUNT,
                        "checkpoint state denominator changed")
                checkpoint_output[str(checkpoint)] = {
                    "N": len(states),
                    "six_state_counts": count_registry(p981.SIX_STATES, states),
                    "four_channel_counts": count_registry(
                        p981.FOUR_CHANNELS,
                        (p981.four_channel(state) for state in states),
                    ),
                }
            arm_output[f"stream_{stream}"] = checkpoint_output
        output[arm] = arm_output
    return output


def transition_matrix(
    labels: tuple[str, ...], pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    matrix = {left: {right: 0 for right in labels} for left in labels}
    total = 0
    for left, right in pairs:
        require(left in matrix and right in matrix[left],
                "transition outside frozen registry")
        matrix[left][right] += 1
        total += 1
    require(total == p981.ITEM_COUNT, "transition denominator changed")
    return matrix


def candidate_adjacent_transitions(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    streams: dict[str, Any] = {}
    aggregate_1536_to_2048 = {label: 0 for label in p981.FOUR_CHANNELS}
    for stream in p981.STREAMS:
        transitions: dict[str, Any] = {}
        for left_checkpoint, right_checkpoint in ADJACENT_CHECKPOINTS:
            six_pairs: list[tuple[str, str]] = []
            four_pairs: list[tuple[str, str]] = []
            for item in items:
                row = records[(item["id"], "B", stream)]
                left = state_at(row, left_checkpoint)
                right = state_at(row, right_checkpoint)
                six_pairs.append((left, right))
                four_pairs.append((p981.four_channel(left), p981.four_channel(right)))
            matrix6 = transition_matrix(p981.SIX_STATES, six_pairs)
            matrix4 = transition_matrix(p981.FOUR_CHANNELS, four_pairs)
            source_c = {
                target: int(matrix4["C"][target]) for target in p981.FOUR_CHANNELS
            }
            label = f"{left_checkpoint}_to_{right_checkpoint}"
            transitions[label] = {
                "N": p981.ITEM_COUNT,
                "six_by_six": matrix6,
                "four_by_four": matrix4,
                "source_C_count": sum(source_c.values()),
                "source_C_destinations_4channel": source_c,
            }
            if (left_checkpoint, right_checkpoint) == (1536, 2048):
                for target, value in source_c.items():
                    aggregate_1536_to_2048[target] += value
        streams[f"stream_{stream}"] = transitions
    return {
        "candidate_B_by_stream": streams,
        "focus_1536_to_2048": {
            "by_stream": {
                label: values["1536_to_2048"][
                    "source_C_destinations_4channel"]
                for label, values in streams.items()
            },
            "aggregate_stream_appearances": aggregate_1536_to_2048,
            "aggregate_is_descriptive_repeated_stream_appearances_not_independent_items": True,
        },
    }


def tail_item_analysis(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    item_by_id = {str(item["id"]): item for item in items}
    by_stream: dict[str, Any] = {}
    sets: dict[int, set[str]] = {}
    aggregate_task: Counter[str] = Counter()
    aggregate_difficulty: Counter[str] = Counter()
    task_registry = sorted({str(item["task"]) for item in items})
    for stream in p981.STREAMS:
        ids = sorted(
            str(item["id"]) for item in items
            if p981.four_channel(state_at(
                records[(item["id"], "B", stream)], p981.DECISION_CHECKPOINT,
            )) == "C"
        )
        sets[stream] = set(ids)
        task_counts = Counter(str(item_by_id[item_id]["task"]) for item_id in ids)
        difficulties = Counter(
            str(item_by_id[item_id]["difficulty"]) for item_id in ids)
        task_difficulty = Counter(
            (str(item_by_id[item_id]["task"]),
             str(item_by_id[item_id]["difficulty"]))
            for item_id in ids
        )
        answers = Counter(str(item_by_id[item_id]["answer"]) for item_id in ids)
        terminal_states = Counter(state_at(
            records[(item_id, "B", stream)], p981.DECISION_CHECKPOINT,
        ) for item_id in ids)
        for item_id in ids:
            row = records[(item_id, "B", stream)]
            checkpoint = row["checkpoints"][str(p981.DECISION_CHECKPOINT)]
            require(state_at(row, p981.DECISION_CHECKPOINT)
                    == "CENSORED_BEFORE_VALID_CLOSE"
                    and len(row["generated_ids"]) == p981.MAX_NEW_TOKENS
                    and checkpoint.get("hit_budget") is True
                    and checkpoint.get("has_eos") is False
                    and checkpoint.get("generated_think_open_positions") == [0]
                    and checkpoint.get("generated_think_close_positions") == []
                    and checkpoint.get("close_observed") is False
                    and checkpoint.get("final_region_valid") is False
                    and checkpoint.get("answer_observed") is False,
                    f"final C tail anatomy changed: {item_id}/stream_{stream}")
        aggregate_task.update(task_counts)
        aggregate_difficulty.update(difficulties)
        by_stream[f"stream_{stream}"] = {
            "C_count": len(ids),
            "by_task": {
                task: int(task_counts[task])
                for task in task_registry
            },
            "by_difficulty": {
                difficulty: int(difficulties[difficulty])
                for difficulty in p981.DIFFICULTIES
            },
            "by_task_difficulty": {
                task: {
                    difficulty: int(task_difficulty[(task, difficulty)])
                    for difficulty in p981.DIFFICULTIES
                }
                for task in task_registry
            },
            "by_answer": {answer: int(answers[answer]) for answer in ("A", "B")},
            "terminal_state_counts": count_registry(
                p981.SIX_STATES, terminal_states.elements()),
            "item_ids": ids,
            "item_ids_sha256": boundary.sha256_json(ids),
        }
    intersection = set.intersection(*(sets[stream] for stream in p981.STREAMS))
    union = set.union(*(sets[stream] for stream in p981.STREAMS))
    frequency = Counter(
        sum(item_id in sets[stream] for stream in p981.STREAMS)
        for item_id in union
    )
    exact_frequency_ids = {
        f"in_exactly_{count}_streams": sorted(
            item_id for item_id in union
            if sum(item_id in sets[stream] for stream in p981.STREAMS) == count
        )
        for count in (1, 2, 3)
    }
    pairwise = {
        f"stream_{left}_intersection_stream_{right}": sorted(
            sets[left] & sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    baseline_c = {
        f"stream_{stream}": sum(
            p981.four_channel(state_at(
                records[(item["id"], "A", stream)], p981.DECISION_CHECKPOINT,
            )) == "C" for item in items
        )
        for stream in p981.STREAMS
    }
    all_item_ids = {str(item["id"]) for item in items}
    venn_patterns = Counter(
        "".join("1" if item_id in sets[stream] else "0"
                for stream in p981.STREAMS)
        for item_id in all_item_ids
    )
    union_task_difficulty = Counter(
        (str(item_by_id[item_id]["task"]),
         str(item_by_id[item_id]["difficulty"]))
        for item_id in union
    )
    intersection_task_difficulty = Counter(
        (str(item_by_id[item_id]["task"]),
         str(item_by_id[item_id]["difficulty"]))
        for item_id in intersection
    )
    union_answers = Counter(str(item_by_id[item_id]["answer"]) for item_id in union)
    intersection_answers = Counter(
        str(item_by_id[item_id]["answer"]) for item_id in intersection)
    pairwise_hashes = {
        key: boundary.sha256_json(value) for key, value in pairwise.items()
    }
    return {
        "checkpoint": p981.DECISION_CHECKPOINT,
        "candidate_B": by_stream,
        "baseline_A_C_counts": baseline_c,
        "aggregate_stream_appearances": {
            "C_count": sum(len(sets[stream]) for stream in p981.STREAMS),
            "by_task": {key: int(aggregate_task[key])
                        for key in sorted(aggregate_task)},
            "by_difficulty": {key: int(aggregate_difficulty[key])
                              for key in p981.DIFFICULTIES},
            "not_an_independent_item_denominator": True,
        },
        "item_set_overlap": {
            "three_stream_intersection": sorted(intersection),
            "three_stream_intersection_n": len(intersection),
            "three_stream_union": sorted(union),
            "three_stream_union_n": len(union),
            "pairwise_intersections": pairwise,
            "pairwise_intersection_counts": {
                key: len(value) for key, value in pairwise.items()
            },
            "pairwise_intersection_sha256": pairwise_hashes,
            "three_stream_intersection_sha256": boundary.sha256_json(
                sorted(intersection)),
            "three_stream_union_sha256": boundary.sha256_json(sorted(union)),
            "membership_frequency_counts": {
                f"in_exactly_{count}_streams": int(frequency[count])
                for count in (1, 2, 3)
            },
            "membership_frequency_item_ids": exact_frequency_ids,
            "venn_pattern_counts_across_all_256_items": {
                pattern: int(venn_patterns[pattern])
                for pattern in ("000", "100", "010", "001",
                                "110", "101", "011", "111")
            },
            "stream_appearance_identity": (
                sum(len(sets[stream]) for stream in p981.STREAMS)
                == sum(count * frequency[count] for count in (1, 2, 3))
            ),
            "all_256_items_partitioned": sum(venn_patterns.values()) == 256,
            "union_breakdown": {
                "by_task_difficulty": {
                    task: {
                        difficulty: int(union_task_difficulty[(task, difficulty)])
                        for difficulty in p981.DIFFICULTIES
                    } for task in task_registry
                },
                "by_answer": {
                    answer: int(union_answers[answer]) for answer in ("A", "B")
                },
            },
            "three_stream_intersection_breakdown": {
                "by_task_difficulty": {
                    task: {
                        difficulty: int(intersection_task_difficulty[
                            (task, difficulty)])
                        for difficulty in p981.DIFFICULTIES
                    } for task in task_registry
                },
                "by_answer": {
                    answer: int(intersection_answers[answer])
                    for answer in ("A", "B")
                },
            },
        },
    }


def validate_final_matrices(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]], audit: dict[str, Any],
) -> None:
    frozen = audit.get("transition_matrices_A_rows_B_columns", {})
    for stream in p981.STREAMS:
        pairs6 = []
        pairs4 = []
        for item in items:
            left = state_at(
                records[(item["id"], "A", stream)], p981.DECISION_CHECKPOINT)
            right = state_at(
                records[(item["id"], "B", stream)], p981.DECISION_CHECKPOINT)
            pairs6.append((left, right))
            pairs4.append((p981.four_channel(left), p981.four_channel(right)))
        label = f"stream_{stream}"
        require(transition_matrix(p981.SIX_STATES, pairs6)
                == frozen.get(label, {}).get("six_by_six")
                and transition_matrix(p981.FOUR_CHANNELS, pairs4)
                == frozen.get(label, {}).get("four_by_four"),
                f"Phase982/final Phase981 matrix mismatch: {label}")


def verify_expected_tail_regression(
    checkpoint_tables: dict[str, Any], adjacent: dict[str, Any],
    tails: dict[str, Any],
) -> dict[str, Any]:
    computed_c = {
        stream: {
            checkpoint: checkpoint_tables["B"][stream][checkpoint][
                "four_channel_counts"]["C"]
            for checkpoint in ("256", "512", "1024", "1536", "2048")
        }
        for stream in ("stream_0", "stream_1", "stream_2")
    }
    require(computed_c == EXPECTED_B_C_BY_CHECKPOINT,
            "frozen B checkpoint C counts changed")
    focus = adjacent["focus_1536_to_2048"]
    require(focus["by_stream"] == EXPECTED_1536_TO_2048_C_DESTINATIONS,
            "frozen 1536->2048 C destinations changed")
    non_c_stability: dict[str, dict[str, int]] = {}
    for stream in ("stream_0", "stream_1", "stream_2"):
        matrix = adjacent["candidate_B_by_stream"][stream][
            "1536_to_2048"]["four_by_four"]
        non_c_stability[stream] = {
            "V_to_V": int(matrix["V"]["V"]),
            "I_sem_to_I_sem": int(matrix["I_sem"]["I_sem"]),
            "non_C_to_C": sum(
                int(matrix[source]["C"])
                for source in p981.FOUR_CHANNELS if source != "C"
            ),
        }
    require(non_c_stability == EXPECTED_1536_TO_2048_NON_C_STABILITY,
            "frozen 1536->2048 non-C stability changed")

    candidate = tails["candidate_B"]
    require({stream: candidate[stream]["by_task"] for stream in candidate}
            == EXPECTED_TAIL_TASK_COUNTS,
            "frozen tail task counts changed")
    require({stream: candidate[stream]["by_difficulty"] for stream in candidate}
            == EXPECTED_TAIL_DIFFICULTY_COUNTS,
            "frozen tail difficulty counts changed")
    require({stream: candidate[stream]["by_answer"] for stream in candidate}
            == {key: EXPECTED_TAIL_ANSWER_COUNTS[key]
                for key in ("stream_0", "stream_1", "stream_2")},
            "frozen tail answer counts changed")
    require({stream: candidate[stream]["item_ids_sha256"] for stream in candidate}
            == {key: EXPECTED_TAIL_ITEM_SET_HASHES[key]
                for key in ("stream_0", "stream_1", "stream_2")},
            "frozen stream tail item sets changed")

    overlap = tails["item_set_overlap"]
    require(overlap["three_stream_intersection_n"]
            == EXPECTED_TAIL_OVERLAP["three_stream_intersection_n"]
            and overlap["three_stream_union_n"]
            == EXPECTED_TAIL_OVERLAP["three_stream_union_n"]
            and overlap["pairwise_intersection_counts"]
            == EXPECTED_TAIL_OVERLAP["pairwise_intersection_counts"]
            and overlap["membership_frequency_counts"]
            == EXPECTED_TAIL_OVERLAP["membership_frequency_counts"]
            and overlap["venn_pattern_counts_across_all_256_items"]
            == EXPECTED_VENN_PATTERN_COUNTS
            and overlap["stream_appearance_identity"] is True
            and overlap["all_256_items_partitioned"] is True,
            "frozen tail overlap/Venn identities changed")
    expected_pairwise_hashes = {
        key: EXPECTED_TAIL_ITEM_SET_HASHES[key]
        for key in (
            "stream_0_intersection_stream_1",
            "stream_0_intersection_stream_2",
            "stream_1_intersection_stream_2",
        )
    }
    require(overlap["pairwise_intersection_sha256"] == expected_pairwise_hashes
            and overlap["three_stream_intersection_sha256"]
            == EXPECTED_TAIL_ITEM_SET_HASHES["three_stream_intersection"]
            and overlap["three_stream_union_sha256"]
            == EXPECTED_TAIL_ITEM_SET_HASHES["three_stream_union"],
            "frozen tail overlap item hashes changed")
    require(overlap["union_breakdown"]["by_answer"]
            == EXPECTED_TAIL_ANSWER_COUNTS["three_stream_union"]
            and overlap["three_stream_intersection_breakdown"]["by_answer"]
            == EXPECTED_TAIL_ANSWER_COUNTS["three_stream_intersection"],
            "frozen overlap answer counts changed")
    require(tails["baseline_A_C_counts"]
            == {"stream_0": 0, "stream_1": 0, "stream_2": 0},
            "frozen baseline C counts changed")
    return {
        "static_phase981_file_hashes_matched": True,
        "static_phase981_self_hashes_matched": True,
        "static_B_C_checkpoint_counts_matched": True,
        "static_1536_to_2048_transitions_matched": True,
        "static_tail_task_difficulty_answer_counts_matched": True,
        "static_tail_item_set_hashes_and_Venn_matched": True,
        "all_54_final_C_rows_have_exact_frozen_tail_anatomy": True,
        "non_C_to_C_at_1536_to_2048": 0,
    }


def negative_self_tests(
    first_row: dict[str, Any], expected_keys: list[tuple[str, str, int]],
) -> dict[str, bool]:
    def rejected(callable_value) -> bool:
        try:
            callable_value()
        except RuntimeError:
            return True
        return False

    synthetic = {"phase": PHASE}
    synthetic["hash"] = boundary.sha256_json(synthetic)
    verify_self_hash(synthetic, "hash", None, "synthetic")
    tampered_document = dict(synthetic)
    tampered_document["phase"] = PHASE + 1
    tampered_row = deepcopy(first_row)
    tampered_row["phase"] = 0
    duplicate_keys = list(expected_keys)
    duplicate_keys[-1] = duplicate_keys[0]
    missing_keys = list(expected_keys[:-1])
    invalid_state = {"terminal_state": "NOT_A_STATE"}
    tests = {
        "missing_source_rejected": rejected(
            lambda: load_json(OUT / "definitely_missing_source.json", "missing source")),
        "tampered_document_self_hash_rejected": rejected(
            lambda: verify_self_hash(
                tampered_document, "hash", None, "tampered synthetic")),
        "tampered_row_self_hash_rejected": rejected(
            lambda: validate_row_self_hash(tampered_row, 1)),
        "duplicate_row_key_rejected": rejected(
            lambda: validate_exact_key_registry(duplicate_keys, expected_keys)),
        "missing_row_rejected": rejected(
            lambda: validate_exact_key_registry(missing_keys, expected_keys)),
        "invalid_terminal_state_rejected": rejected(
            lambda: require(
                invalid_state["terminal_state"] in p981.SIX_STATES,
                "invalid terminal state")),
        "duplicate_JSON_key_rejected": rejected(
            lambda: strict_json_loads('{"x":1,"x":2}', "duplicate JSON")),
        "non_finite_JSON_rejected": rejected(
            lambda: strict_json_loads('{"x":NaN}', "non-finite JSON")),
    }
    require(all(tests.values()), "Phase982 fail-closed self-test failed")
    return tests


def build_report() -> dict[str, Any]:
    assert_no_holdout_import()
    assert_no_model_runtime_import()
    documents, items, records, source_authentication = authenticate_phase981()
    protocol = documents["protocol"]
    audit = documents["audit"]
    validate_final_matrices(records, items, audit)
    checkpoint_tables = checkpoint_state_tables(records, items)
    adjacent = candidate_adjacent_transitions(records, items)
    tails = tail_item_analysis(records, items)
    regression_checks = verify_expected_tail_regression(
        checkpoint_tables, adjacent, tails)
    expected_keys = [
        (str(item["id"]), arm, stream)
        for item, arm, stream in p981.canonical_grid(items)
    ]
    self_tests = negative_self_tests(records[expected_keys[0]], expected_keys)
    primary_contract = protocol.get("semantic_gate", {}).get(
        "contract", {}).get("primary_semantic", {})
    source_gate = audit["semantic_gate_result"]
    stream_failures = {
        label: {
            "primary_semantic_stream_passed": value[
                "primary_semantic_stream_passed"],
            "delta_V": value["overall"]["delta_V"],
            "R_I_sem": value["overall"]["R_I_sem"],
            "delta_C": value["overall"]["delta_C"],
            "delta_C_max_frozen": primary_contract.get("delta_C_max"),
            "delta_C_gate_passed": value[
                "primary_semantic_checks"]["delta_C_at_most_12"],
        }
        for label, value in source_gate["stream_results"].items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "cpu_only_posthoc_tail_failure_descriptive_report",
        "script": relative(SCRIPT_PATH),
        "script_sha256": boundary.sha256_file(SCRIPT_PATH),
        "phase981_source_authentication": source_authentication,
        "checkpoint_state_counts_by_arm_stream": checkpoint_tables,
        "candidate_B_adjacent_checkpoint_transitions": adjacent,
        "tail_C_at_2048": tails,
        "frozen_expected_regression_checks": regression_checks,
        "phase981_decision_preservation": {
            "source_audit_sha256": audit["audit_sha256"],
            "source_fresh_confirmation_passed": False,
            "phase981_NO_GO_unchanged": True,
            "phase981_gate_reopened": False,
            "thresholds_changed": False,
            "terminal_state_rules_changed": False,
            "frozen_primary_semantic_contract": primary_contract,
            "stream_failure_summary": stream_failures,
            "secondary_censor_can_change_primary": False,
            "interpretation": (
                "Phase982 only describes where the frozen Phase981 trajectories "
                "remain in the right tail. It cannot admit Phase981, change its "
                "NO-GO, or reinterpret censored rows as valid stops."
            ),
        },
        "method_boundary": deepcopy(METHOD_BOUNDARY),
        "fail_closed_self_tests": self_tests,
        "report_fail_closed_tamper_tests": {
            name: True for name in REPORT_TAMPER_TEST_NAMES
        },
        "cpu_only": True,
        "gpu_authorized": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "model_runtime_modules_imported": False,
        "generation_performed": False,
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    assert_no_holdout_import()
    assert_no_model_runtime_import()
    report = {
        **payload,
        "report_sha256": boundary.sha256_json(payload),
        "created_at_utc": boundary.utc_now(),
    }
    observed_tamper_tests = dynamic_report_tamper_tests(report, payload)
    require(
        boundary.canonical_json(observed_tamper_tests)
        == boundary.canonical_json(payload["report_fail_closed_tamper_tests"]),
        "Phase982 dynamic report tamper-test registry mismatch",
    )
    return report


def exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""
    return boundary.canonical_json(left) == boundary.canonical_json(right)


def verify_report_against_expected(
    document: dict[str, Any], expected_payload: dict[str, Any],
) -> None:
    """Verify a report against a freshly reconstructed trusted identity."""
    require(isinstance(document, dict), "Phase982 report must be an object")
    verify_self_hash(document, "report_sha256", "created_at_utc",
                     "Phase982 report")
    require(document.get("script") == relative(SCRIPT_PATH),
            "Phase982 report script path changed")
    require(document.get("script_sha256") == boundary.sha256_file(SCRIPT_PATH),
            "Phase982 report script hash is not current")
    require(exact_json_equal(document.get("method_boundary"), METHOD_BOUNDARY),
            "Phase982 method boundary changed")

    decision = document.get("phase981_decision_preservation")
    require(isinstance(decision, dict),
            "Phase982 decision-preservation object missing")
    for key, expected in EXPECTED_DECISION_PRESERVATION_FIELDS.items():
        actual = decision.get(key)
        if isinstance(expected, bool):
            require(actual is expected,
                    f"Phase982 decision-preservation field changed: {key}")
        else:
            require(actual == expected,
                    f"Phase982 decision-preservation field changed: {key}")

    strict_boundary_flags = {
        "cpu_only": True,
        "gpu_authorized": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "model_runtime_modules_imported": False,
        "generation_performed": False,
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    require(document.get("phase") == PHASE,
            "Phase982 report phase changed")
    for key, expected in strict_boundary_flags.items():
        require(document.get(key) is expected,
                f"Phase982 runtime boundary changed: {key}")

    require(exact_json_equal(
        document.get("phase981_source_authentication"),
        expected_payload.get("phase981_source_authentication"),
    ), "Phase982 source authentication/counts changed")
    require(exact_json_equal(
        document.get("frozen_expected_regression_checks"),
        expected_payload.get("frozen_expected_regression_checks"),
    ), "Phase982 frozen regression claims changed")
    require(exact_json_equal(report_payload(document), expected_payload),
            "Phase982 report differs from deterministic expected identity")


def dynamic_report_tamper_tests(
    report: dict[str, Any], expected_payload: dict[str, Any],
) -> dict[str, bool]:
    """Ensure self-consistent, rehashed report forgeries remain rejected."""
    cases: tuple[tuple[str, tuple[str, ...], Any], ...] = (
        (
            "rehashed_gate_reopened_true_rejected",
            ("phase981_decision_preservation", "phase981_gate_reopened"),
            True,
        ),
        (
            "rehashed_terminal_rules_true_rejected",
            ("phase981_decision_preservation", "terminal_state_rules_changed"),
            True,
        ),
        (
            "rehashed_descriptive_only_false_rejected",
            ("method_boundary", "descriptive_only"),
            False,
        ),
        (
            "rehashed_no_statistical_claim_false_rejected",
            ("method_boundary", "no_statistical_or_mechanism_claim"),
            False,
        ),
        (
            "rehashed_script_sha_forgery_rejected",
            ("script_sha256",),
            "0" * 64,
        ),
        (
            "rehashed_tail_C_count_change_rejected",
            ("tail_C_at_2048", "candidate_B", "stream_0", "C_count"),
            21,
        ),
        (
            "rehashed_source_hash_tamper_rejected",
            (
                "phase981_source_authentication", "artifacts", "rows",
                "file_sha256",
            ),
            "0" * 64,
        ),
        (
            "rehashed_source_count_tamper_rejected",
            ("phase981_source_authentication", "row_authentication", "row_count"),
            1535,
        ),
    )
    require(tuple(name for name, _, _ in cases) == REPORT_TAMPER_TEST_NAMES,
            "Phase982 report tamper-test names changed")
    results: dict[str, bool] = {}
    for name, path, forged_value in cases:
        tampered = deepcopy(report)
        cursor: Any = tampered
        for key in path[:-1]:
            require(isinstance(cursor, dict) and key in cursor,
                    f"missing report tamper-test path: {name}/{key}")
            cursor = cursor[key]
        require(isinstance(cursor, dict) and path[-1] in cursor,
                f"missing report tamper-test target: {name}/{path[-1]}")
        cursor[path[-1]] = forged_value
        reseal_report(tampered)
        try:
            verify_report_against_expected(tampered, expected_payload)
        except RuntimeError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()),
            "Phase982 rehashed report tamper test failed open")
    return results


def verify_report(document: dict[str, Any]) -> None:
    """Public verifier: reconstruct expected identity from sealed Phase981."""
    expected_report = build_report()
    verify_report_against_expected(document, report_payload(expected_report))


def install_or_validate(
    document: dict[str, Any], expected_payload: dict[str, Any],
) -> None:
    verify_report_against_expected(document, expected_payload)
    if REPORT_PATH.exists():
        prior_file_sha256 = boundary.sha256_file(REPORT_PATH)
        prior = load_json(REPORT_PATH, "existing Phase982 report")
        verify_self_hash(prior, "report_sha256", "created_at_utc",
                         "existing Phase982 report")
        if prior.get("report_sha256") == document["report_sha256"]:
            verify_report_against_expected(prior, expected_payload)
            return
        superseded = SUPERSEDED_REPORTS.get(str(prior.get("report_sha256")))
        require(
            superseded is not None
            and prior_file_sha256 == superseded["file_sha256"]
            and prior.get("script_sha256") == superseded["script_sha256"]
            and prior.get("phase") == PHASE,
            "existing Phase982 report differs from the authorized predecessor",
        )
        boundary.atomic_write_json(REPORT_PATH, document)
        written = load_json(REPORT_PATH, "upgraded Phase982 report")
        verify_report_against_expected(written, expected_payload)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    boundary.atomic_write_json(REPORT_PATH, document)
    written = load_json(REPORT_PATH, "written Phase982 report")
    verify_report_against_expected(written, expected_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="authenticate and test without writing")
    parser.add_argument("--write", action="store_true",
                        help="write or idempotently validate report.json")
    args = parser.parse_args()
    report = build_report()
    expected_payload = report_payload(report)
    verify_report_against_expected(report, expected_payload)
    report_tamper_tests = dynamic_report_tamper_tests(
        report, expected_payload)
    # Changing only the excluded timestamp must not change the report identity.
    shifted = deepcopy(report)
    shifted["created_at_utc"] = "2099-01-01T00:00:00+00:00"
    verify_report_against_expected(shifted, expected_payload)
    if args.write and not args.self_test:
        install_or_validate(report, expected_payload)
    focus = report["candidate_B_adjacent_checkpoint_transitions"][
        "focus_1536_to_2048"]
    print(json.dumps({
        "phase": PHASE,
        "report_sha256": report["report_sha256"],
        "self_test": bool(args.self_test),
        "written": bool(args.write and not args.self_test),
        "report_exists": REPORT_PATH.exists(),
        "report_fail_closed_tamper_tests": report_tamper_tests,
        "phase981_NO_GO_unchanged": True,
        "focus_1536_to_2048": focus,
        "tail_C_at_2048_counts": {
            label: value["C_count"]
            for label, value in report["tail_C_at_2048"][
                "candidate_B"].items()
        },
        "cpu_only": True,
        "gpu_used": False,
        "model_weights_loaded": False,
        "model_runtime_modules_imported": False,
        "holdout": False,
        "mechanism": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
