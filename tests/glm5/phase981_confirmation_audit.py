#!/usr/bin/env python3
"""Independent CPU audit and semantic-only decision for Phase 981."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter, defaultdict
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
from model_utils import MODEL_CONFIGS  # noqa: E402


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


def dataset_identity(audit: dict[str, Any]) -> dict[str, Any]:
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
            "fresh identity invalid")
    return result


def authenticate_chain() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    list[dict[str, Any]], dict[str, Any]
]:
    protocol = boundary.load_json(core.PROTOCOL_PATH, "Phase981 protocol")
    admission = boundary.load_json(core.ADMISSION_PATH, "Phase981 admission")
    manifest = boundary.load_json(core.MANIFEST_PATH, "Phase981 manifest")
    status = boundary.load_json(core.STATUS_PATH, "Phase981 generator status")
    seal_audit = core.verify_protocol_file_seals(protocol)
    verify_self_hash(protocol, "protocol_sha256", "created_at_utc", "protocol")
    verify_self_hash(admission, "admission_sha256", "admitted_at_utc", "admission")
    verify_self_hash(manifest, "manifest_sha256", "created_at_utc", "manifest")
    verify_self_hash(status, "status_sha256", "updated_at_utc", "status")
    core.verify_protocol_boundary_contract(protocol)
    core.verify_admission_boundary_contract(admission)
    require(admission.get("protocol_sha256") == protocol["protocol_sha256"]
            and manifest.get("protocol_sha256") == protocol["protocol_sha256"]
            and manifest.get("admission_sha256") == admission["admission_sha256"]
            and status.get("protocol_sha256") == protocol["protocol_sha256"]
            and status.get("admission_sha256") == admission["admission_sha256"]
            and status.get("manifest_sha256") == manifest["manifest_sha256"],
            "Phase981 lineage mismatch")
    require(manifest.get("protocol_file_sha256")
            == boundary.sha256_file(core.PROTOCOL_PATH)
            and manifest.get("admission_file_sha256")
            == boundary.sha256_file(core.ADMISSION_PATH),
            "manifest source-file hashes mismatch")
    admission_dependency_audit = admission.get("dependency_audit", {})
    require(admission_dependency_audit.get("script_seals_sha256")
            == seal_audit["script_seals_sha256"]
            and admission_dependency_audit.get("dependency_seals_sha256")
            == seal_audit["dependency_seals_sha256"]
            and admission_dependency_audit.get("phase979_script_hashes_sha256")
            == seal_audit["phase979_script_hashes_sha256"],
            "admission code seal audit differs from frozen protocol")
    core.verify_manifest_dependency_contract(manifest, protocol)
    require(admission.get("admitted") is True
            and admission.get("qwen_external_generation_authorized") is True
            and admission.get("gpu_authorized") is True,
            "generation lacked independent admission")
    require(status.get("expected_rows") == core.EXPECTED_ROWS,
            "generation expected row denominator changed")
    core.verify_complete_status_generation_contract(status)
    require(manifest.get("expected_rows") == core.EXPECTED_ROWS
            and manifest.get("streams") == list(core.STREAMS)
            and manifest.get("sampling") == core.SAMPLING
            and manifest.get("checkpoints") == list(core.CHECKPOINTS)
            and manifest.get("max_new_tokens") == core.MAX_NEW_TOKENS
            and manifest.get("batch_size") == core.BATCH_SIZE
            and manifest.get("model_name") == core.MODEL_NAME
            and manifest.get("device_type") == "cuda",
            "manifest execution contract changed")
    expected_cell_counts = {
        f"{arm}|stream_{stream}": 256
        for stream in core.STREAMS for arm in core.ARMS
    }
    require(status.get("completed_by_arm_stream") == expected_cell_counts,
            "status arm/stream denominator changed")
    require(protocol.get("primary_decision", {}).get("route") == "semantic_only"
            and protocol.get("primary_decision", {}).get("censor_route_can_admit") is False,
            "primary route changed")
    require(protocol.get("arms") == core.ARMS
            and manifest.get("arms") == core.ARMS
            and protocol.get("direction") == "B_minus_A",
            "formal A/B configuration changed")
    for document in (protocol, admission, manifest, status):
        require(document.get("holdout") is False
                and document.get("holdout_loaded") is False
                and document.get("mechanism") is False
                and document.get("mechanism_authorized") is False,
                "authenticated chain crossed forbidden boundary")
    require(seal_audit["verified_script_count"] == len(core.PHASE981_SCRIPT_PATHS)
            and seal_audit["verified_dependency_count"]
            == len(core.RUNTIME_DEPENDENCY_PATHS)
            and seal_audit["verified_phase979_script_count"]
            == len(core.PHASE979_SCRIPT_PATHS),
            "sealed code denominator changed")
    core.verify_protocol_integrity_metadata(protocol)
    core.verify_model_artifact_identity(
        protocol.get("local_model_artifact_identity"))
    items = fresh.build_items()
    data_audit = fresh.audit_items(items)
    require(data_audit.get("passed") is True or data_audit.get("ok") is True,
            "fresh dataset audit failed")
    identity = dataset_identity(data_audit)
    require(identity["identity_sha256"]
            == protocol.get("dataset_identity", {}).get("identity_sha256")
            == manifest.get("dataset_identity", {}).get("identity_sha256"),
            "dataset identity mismatch")
    require(boundary.sha256_json(data_audit) == protocol.get("dataset_audit_sha256")
            == manifest.get("dataset_audit_sha256"),
            "dataset audit mismatch")
    artifact_seals = protocol.get("dataset_artifact_seals", {})
    require(boundary.sha256_file(core.DATASET_ARTIFACT_PATH)
            == artifact_seals.get("dataset", {}).get("file_sha256")
            and boundary.sha256_file(core.DATASET_AUDIT_PATH)
            == artifact_seals.get("audit", {}).get("file_sha256"),
            "dataset artifact file seal mismatch")
    require(boundary.sha256_json(semantic_gate.GATE_CONTRACT)
            == protocol.get("semantic_gate", {}).get("contract_sha256"),
            "semantic gate contract mismatch")
    assert_no_holdout_import()
    return protocol, admission, manifest, status, items, data_audit


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[core.MODEL_NAME]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def verify_runtime_token_contract(
    manifest: dict[str, Any], protocol: dict[str, Any], tok,
) -> dict[str, Any]:
    think_open = boundary.single_token_id(tok, "<think>")
    think_close = boundary.single_token_id(tok, "</think>")
    a_id = boundary.single_token_id(tok, "A")
    b_id = boundary.single_token_id(tok, "B")
    identity = core.verify_protocol_token_identity(
        protocol.get("tokenizer_audit"), tok.eos_token_id,
        think_open, think_close, a_id, b_id)
    require(manifest.get("eos_token_ids") == identity["effective_eos_token_ids"]
            and manifest.get("think_open_id") == think_open
            and manifest.get("think_close_id") == think_close
            and manifest.get("pad_token_id") == int(tok.pad_token_id)
            and manifest.get("tokenizer_class") == type(tok).__name__
            and manifest.get("tokenizer_length") == len(tok),
            "manifest token identity differs from independent audit")
    return identity


def read_rows(
    manifest: dict[str, Any], items: list[dict[str, Any]], tok,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    require(core.ROWS_PATH.is_file(), "missing Phase981 rows")
    payload = core.ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"), "rows JSONL lacks final newline")
    expected = core.expected_keys(items)
    item_by_id = {str(item["id"]): item for item in items}
    eos_ids = [int(value) for value in manifest["eos_token_ids"]]
    think_open = int(manifest["think_open_id"])
    think_close = int(manifest["think_close_id"])
    grid_positions = {
        (str(item["id"]), arm, stream): index
        for index, (item, arm, stream) in enumerate(core.canonical_grid(items))
    }
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed row {line_number}") from exc
        require(isinstance(row, dict), f"row {line_number} is not an object")
        require(row.get("row_sha256")
                == boundary.sha256_json(boundary.without_fields(row, "row_sha256")),
                f"row self-hash mismatch {line_number}")
        key = core.row_key(row)
        require(key in expected and key not in records, f"invalid/duplicate row key: {key}")
        item = item_by_id[key[0]]
        arm, stream = key[1], key[2]
        user, rendered, ids = core.render_prefix(tok, item, arm)
        generated = row.get("generated_ids")
        require(isinstance(generated, list) and generated
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in generated), f"invalid generated IDs: {key}")
        eos_positions = boundary.positions_of(generated, set(eos_ids))
        require((len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
                or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
                f"invalid termination: {key}")
        checkpoints = core.analyze_checkpoints(
            tok, item, arm, generated, eos_ids, think_open, think_close)
        identity_sha = manifest["dataset_identity"]["identity_sha256"]
        require(
            row.get("schema_version") == core.SCHEMA_VERSION
            and row.get("phase") == core.PHASE
            and row.get("experiment") == core.EXPERIMENT
            and row.get("protocol_sha256") == manifest["protocol_sha256"]
            and row.get("admission_sha256") == manifest["admission_sha256"]
            and row.get("manifest_sha256") == manifest["manifest_sha256"]
            and row.get("task") == item["task"]
            and row.get("difficulty") == item["difficulty"]
            and row.get("prompt") == item["prompt"]
            and row.get("answer") == item["answer"]
            and row.get("arm_spec") == core.ARMS[arm]
            and row.get("stream") == stream
            and row.get("pair_id") == core.pair_id(item["id"], stream)
            and row.get("pair_seed")
            == core.stable_pair_seed(identity_sha, item["id"], stream)
            and row.get("sampling") == core.SAMPLING
            and row.get("batch_index") == grid_positions[key] // core.BATCH_SIZE + 1
            and row.get("effective_user_prompt") == user
            and row.get("rendered_prefix_sha256") == boundary.sha256_json(rendered)
            and row.get("input_ids") == ids
            and row.get("prompt_len") == len(ids)
            and row.get("generated_raw")
            == tok.decode(generated, skip_special_tokens=False)
            and row.get("generated_plain")
            == tok.decode(generated, skip_special_tokens=True).strip()
            and row.get("checkpoints") == checkpoints
            and row.get("max_new_tokens") == core.MAX_NEW_TOKENS,
            f"row metadata/derivation mismatch: {key}",
        )
        core.verify_row_generation_contract(row)
        records[key] = row
    require(set(records) == expected and len(records) == core.EXPECTED_ROWS,
            "row key set/denominator mismatch")
    for stream in core.STREAMS:
        for item in items:
            a = records[(item["id"], "A", stream)]
            b = records[(item["id"], "B", stream)]
            require(a["pair_id"] == b["pair_id"] and a["pair_seed"] == b["pair_seed"],
                    f"A/B pair coupling mismatch: {item['id']}/stream{stream}")
    return records, {
        "rows_file_sha256": boundary.sha256_file(core.ROWS_PATH),
        "row_count": len(records),
        "all_row_self_hashes_valid": True,
        "all_prefixes_and_checkpoints_recomputed": True,
        "all_first_eos_absorbing": True,
        "all_A_B_pair_seeds_equal": True,
    }


def state_rows(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]], stream: int, arm: str,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for item in items:
        row = records[(item["id"], arm, stream)]
        terminal = row["checkpoints"][str(core.DECISION_CHECKPOINT)]["terminal_state"]
        output.append({
            "id": str(item["id"]), "task": str(item["task"]),
            "difficulty": str(item["difficulty"]),
            "state": core.four_channel(str(terminal)),
        })
    return output


def transition_matrices(
    records: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]], stream: int,
) -> dict[str, Any]:
    pairs6: list[tuple[str, str]] = []
    pairs4: list[tuple[str, str]] = []
    pairs3: list[tuple[str, str]] = []
    for item in items:
        a = records[(item["id"], "A", stream)]
        b = records[(item["id"], "B", stream)]
        state_a = a["checkpoints"][str(core.DECISION_CHECKPOINT)]["terminal_state"]
        state_b = b["checkpoints"][str(core.DECISION_CHECKPOINT)]["terminal_state"]
        pairs6.append((state_a, state_b))
        pairs4.append((core.four_channel(state_a), core.four_channel(state_b)))
        pairs3.append((core.three_channel(state_a), core.three_channel(state_b)))
    matrices = {
        "six_by_six": core.matrix(core.SIX_STATES, pairs6),
        "four_by_four": core.matrix(core.FOUR_CHANNELS, pairs4),
        "three_by_three": core.matrix(core.THREE_CHANNELS, pairs3),
    }
    for matrix in matrices.values():
        require(sum(sum(row.values()) for row in matrix.values()) == 256,
                "transition matrix denominator changed")
    return matrices


def matrix_margins(matrix: dict[str, dict[str, int]]) -> dict[str, Any]:
    labels = list(matrix)
    return {
        "A_rows": {left: sum(matrix[left].values()) for left in labels},
        "B_columns": {
            right: sum(matrix[left][right] for left in labels) for right in labels
        },
    }


def build_audit() -> dict[str, Any]:
    assert_no_holdout_import()
    protocol, admission, manifest, status, items, data_audit = authenticate_chain()
    tok = load_tokenizer()
    try:
        token_identity = verify_runtime_token_contract(manifest, protocol, tok)
        records, row_audit = read_rows(manifest, items, tok)
    finally:
        del tok
        gc.collect()
    gate_input = {
        f"stream_{stream}": (
            state_rows(records, items, stream, "A"),
            state_rows(records, items, stream, "B"),
        )
        for stream in core.STREAMS
    }
    gate_result = semantic_gate.evaluate_three_streams(gate_input)
    require(gate_result.get("secondary_censor_can_set_primary") is False,
            "secondary censor leaked into primary")
    matrices: dict[str, Any] = {}
    for stream in core.STREAMS:
        label = f"stream_{stream}"
        values = transition_matrices(records, items, stream)
        require(values["four_by_four"]
                == gate_result["stream_results"][label]["transition_matrix_4x4"],
                f"independent/gate 4x4 mismatch: {label}")
        matrices[label] = {
            **values,
            "six_by_six_margins": matrix_margins(values["six_by_six"]),
            "four_by_four_margins": matrix_margins(values["four_by_four"]),
            "three_by_three_margins": matrix_margins(values["three_by_three"]),
        }
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE, "experiment": core.EXPERIMENT,
        "role": "independent_cpu_semantic_confirmation_audit",
        "protocol_sha256": protocol["protocol_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "status_sha256": status["status_sha256"],
        "dataset_identity": dataset_identity(data_audit),
        "row_audit": row_audit,
        "token_identity_independently_verified": token_identity,
        "code_seal_audit": core.verify_protocol_file_seals(protocol),
        "decision_checkpoint": core.DECISION_CHECKPOINT,
        "terminal_partition": "V+C+I_mode+I_sem=N",
        "transition_matrices_A_rows_B_columns": matrices,
        "semantic_gate_contract_sha256": boundary.sha256_json(
            semantic_gate.GATE_CONTRACT),
        "semantic_gate_result": gate_result,
        "primary_route": "semantic_only",
        "primary_direction": "B_minus_A",
        "fresh_confirmation_passed": bool(
            gate_result["primary_semantic_passed"]),
        "direct_item_I_sem_to_V_evidence_passed": bool(
            gate_result["direct_item_I_sem_to_V_evidence_passed"]),
        "secondary_censor_descriptive_passed": bool(
            gate_result["secondary_censor_descriptive_passed"]),
        "secondary_censor_can_change_primary": False,
        "model_weights_loaded_by_auditor": False,
        "gpu_used_by_auditor": False,
        "holdout": False, "holdout_loaded": False, "holdout_authorized": False,
        "mechanism": False, "mechanism_authorized": False,
        "interpretation_boundary": (
            "The primary decision is only the precommitted semantic external bundle "
            "contrast B-A. Censor diagnostics cannot admit it. Paired transitions are "
            "under a frozen stochastic coupling and are not internal mechanism evidence."
        ),
    }
    assert_no_holdout_import()
    return {**payload, "audit_sha256": boundary.sha256_json(payload),
            "audited_at_utc": boundary.utc_now()}


def install_or_validate(document: dict[str, Any]) -> None:
    verify_self_hash(document, "audit_sha256", "audited_at_utc", "new audit")
    if core.AUDIT_PATH.exists():
        prior = boundary.load_json(core.AUDIT_PATH, "existing Phase981 audit")
        verify_self_hash(prior, "audit_sha256", "audited_at_utc", "existing audit")
        require(prior["audit_sha256"] == document["audit_sha256"],
                "existing Phase981 audit differs")
        return
    boundary.atomic_write_json(core.AUDIT_PATH, document)


def self_test() -> dict[str, Any]:
    gate_test = semantic_gate.self_test()
    require(gate_test.get("passed") is True, "semantic gate self-test failed")
    labels6 = core.SIX_STATES
    pairs6 = [(state, state) for state in labels6 for _ in range(2)]
    matrix6 = core.matrix(labels6, pairs6)
    require(sum(sum(row.values()) for row in matrix6.values()) == 12,
            "local matrix helper self-test failed")
    require(core.four_channel("EOS_INVALID_MODE") == "I_mode"
            and core.four_channel("EOS_INVALID_SEMANTIC") == "I_sem"
            and core.three_channel("EOS_INVALID_MODE") == "I",
            "local channel mapping self-test failed")
    valid_row = {
        **core.ROW_GENERATION_CONTRACT,
        "max_new_tokens": core.MAX_NEW_TOKENS,
    }
    core.verify_row_generation_contract(valid_row)
    tampered_row = dict(valid_row)
    tampered_row.pop("generation_performed")
    try:
        core.verify_row_generation_contract(tampered_row)
    except RuntimeError:
        missing_row_flag_rejected = True
    else:
        missing_row_flag_rejected = False
    require(missing_row_flag_rejected, "missing row generation flag passed")
    valid_status = {
        "complete": True, "completed_rows": core.EXPECTED_ROWS,
        "generation_performed": True, "model_weights_loaded": True,
        "decision_computed": False,
        "holdout": False, "holdout_loaded": False,
        "mechanism": False, "mechanism_authorized": False,
    }
    core.verify_complete_status_generation_contract(valid_status)
    tampered_status = dict(valid_status)
    tampered_status["model_weights_loaded"] = False
    try:
        core.verify_complete_status_generation_contract(tampered_status)
    except RuntimeError:
        false_status_model_flag_rejected = True
    else:
        false_status_model_flag_rejected = False
    require(false_status_model_flag_rejected,
            "false complete-status model flag passed")
    token_identity = core.token_identity_from_artifacts(
        core.EXPECTED_TOKENIZER_EOS_ID)
    require(token_identity["effective_eos_token_ids"]
            == list(core.EXPECTED_EOS_TOKEN_IDS),
            "EOS identity self-test failed")
    test_protocol = {
        "script_seals": core.build_file_seals(
            core.PHASE981_SCRIPT_PATHS, "Phase981 self-test script"),
        "dependency_seals": core.build_file_seals(
            core.RUNTIME_DEPENDENCY_PATHS, "Phase981 self-test dependency"),
        "phase979_source": {
            "phase979_script_hashes": core.build_file_seals(
                core.PHASE979_SCRIPT_PATHS, "Phase979 self-test script"),
        },
    }
    seal_audit = core.verify_protocol_file_seals(test_protocol)
    valid_manifest = {
        "script_seals": json.loads(json.dumps(test_protocol["script_seals"])),
        "dependency_seals": json.loads(json.dumps(
            test_protocol["dependency_seals"])),
        "phase979_script_hashes": json.loads(json.dumps(
            test_protocol["phase979_source"]["phase979_script_hashes"])),
        **seal_audit,
        "runner_sha256": test_protocol["script_seals"]["runner"]["sha256"],
        "boundary_core_sha256": test_protocol["dependency_seals"][
            "phase979_boundary_core"]["sha256"],
        "generation_contract": json.loads(json.dumps(core.GENERATION_CONTRACT)),
        "model_weights_loaded": True,
        "gpu_used": True,
        "generation_performed": False,
        "holdout": False,
        "holdout_loaded": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    # Counts returned by verify_protocol_file_seals are audit metadata, not
    # persisted manifest fields.
    for key in (
        "verified_script_count", "verified_dependency_count",
        "verified_phase979_script_count",
    ):
        valid_manifest.pop(key)
    core.verify_manifest_dependency_contract(valid_manifest, test_protocol)
    tampered_manifest = json.loads(json.dumps(valid_manifest))
    tampered_manifest["runner_sha256"] = "0" * 64
    try:
        core.verify_manifest_dependency_contract(tampered_manifest, test_protocol)
    except RuntimeError:
        changed_manifest_runner_rejected = True
    else:
        changed_manifest_runner_rejected = False
    require(changed_manifest_runner_rejected,
            "changed manifest runner seal passed")
    return {
        "passed": True,
        "semantic_gate_self_test_sha256": boundary.sha256_json(gate_test),
        "six_state_matrix_helper_passed": True,
        "channel_mapping_passed": True,
        "missing_row_generation_flag_rejected": True,
        "false_complete_status_model_flag_rejected": True,
        "effective_eos_token_ids": token_identity["effective_eos_token_ids"],
        "changed_manifest_runner_rejected": True,
        "gpu_used": False, "files_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    document = build_audit()
    if args.write:
        install_or_validate(document)
    print(json.dumps({
        "phase": core.PHASE,
        "audit_sha256": document["audit_sha256"],
        "fresh_confirmation_passed": document["fresh_confirmation_passed"],
        "direct_item_I_sem_to_V_evidence_passed": document[
            "direct_item_I_sem_to_V_evidence_passed"],
        "secondary_censor_can_change_primary": False,
        "written": bool(args.write),
        "holdout": False, "mechanism": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
