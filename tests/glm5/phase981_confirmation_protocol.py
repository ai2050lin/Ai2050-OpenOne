#!/usr/bin/env python3
"""CPU-only preregistration for the Phase 981 fresh256 confirmation."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as boundary  # noqa: E402
import phase979_diagnostic_dataset as phase979_data  # noqa: E402
import phase980_rescue_gate_feasibility as phase980_design  # noqa: E402
import phase981_confirmation_core as core  # noqa: E402
import phase981_fresh_dataset as fresh  # noqa: E402
import phase981_semantic_gate as semantic_gate  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE979_OUT = GLM5 / "result" / "phase979_three_boundary_factorial"
PHASE979_PROTOCOL = PHASE979_OUT / "protocol_preregistration.json"
PHASE979_AUDIT = PHASE979_OUT / "audit_natural.json"
PHASE979_STATUS = PHASE979_OUT / "generator_status_natural.json"
PHASE979_ROWS = PHASE979_OUT / "rows_natural.jsonl"
PHASE980_SCRIPT = GLM5 / "phase980_rescue_gate_feasibility.py"
PHASE980_REPORT = GLM5 / "result" / "phase980_rescue_gate_design" / "feasibility_report.json"

SCRIPT_PATHS = {
    label: ROOT / relative_path
    for label, relative_path in core.PHASE981_SCRIPT_PATHS.items()
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


def verify_self_hash(
    document: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    payload = boundary.without_fields(document, hash_field, time_field)
    require(document.get(hash_field) == boundary.sha256_json(payload),
            f"{label} self-hash invalid")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }


def authenticate_phase979() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = boundary.load_json(PHASE979_PROTOCOL, "Phase979 protocol")
    audit = boundary.load_json(PHASE979_AUDIT, "Phase979 natural audit")
    status = boundary.load_json(PHASE979_STATUS, "Phase979 natural status")
    verify_self_hash(protocol, "protocol_sha256", "created_at_utc", "Phase979 protocol")
    verify_self_hash(audit, "audit_sha256", "audited_at_utc", "Phase979 audit")
    verify_self_hash(status, "status_sha256", "updated_at_utc", "Phase979 status")
    require(protocol.get("phase") == 979 and audit.get("phase") == 979,
            "wrong Phase979 source phase")
    require(audit.get("protocol_sha256") == protocol["protocol_sha256"],
            "Phase979 protocol/audit mismatch")
    require(status.get("protocol_sha256") == protocol["protocol_sha256"],
            "Phase979 protocol/status mismatch")
    require(status.get("complete") is True and status.get("completed_rows") == 2048,
            "Phase979 natural source incomplete")
    rows_sha = boundary.sha256_file(PHASE979_ROWS)
    require(rows_sha == audit.get("rows_file_sha256")
            == audit.get("row_audit", {}).get("rows_file_sha256"),
            "Phase979 rows hash mismatch")
    require(audit.get("passed_candidate_screens") == []
            and audit.get("new_independent_confirmation_candidate_exists") is False,
            "Phase979 P is not empty")
    require(protocol.get("holdout_loaded") is False
            and protocol.get("mechanism_authorized") is False
            and audit.get("holdout_loaded") is False
            and audit.get("mechanism_authorized") is False
            and status.get("holdout_loaded") is False
            and status.get("mechanism_authorized") is False,
            "Phase979 crossed a forbidden boundary")
    phase979_script_hashes = protocol.get("phase979_script_hashes")
    core.verify_file_seals(
        phase979_script_hashes, core.PHASE979_SCRIPT_PATHS,
        "Phase979 authenticated script",
    )
    return {
        "protocol": {
            "path": relative(PHASE979_PROTOCOL),
            "file_sha256": boundary.sha256_file(PHASE979_PROTOCOL),
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "natural_audit": {
            "path": relative(PHASE979_AUDIT),
            "file_sha256": boundary.sha256_file(PHASE979_AUDIT),
            "audit_sha256": audit["audit_sha256"],
        },
        "natural_status": {
            "path": relative(PHASE979_STATUS),
            "file_sha256": boundary.sha256_file(PHASE979_STATUS),
            "status_sha256": status["status_sha256"],
        },
        "natural_rows": {
            "path": relative(PHASE979_ROWS), "file_sha256": rows_sha,
        },
        "phase979_script_hashes": json.loads(json.dumps(phase979_script_hashes)),
        "phase979_script_hashes_sha256": core.canonical_sha256(
            phase979_script_hashes),
        "P": [], "P_is_empty": True,
        "holdout": False, "mechanism": False,
    }, protocol


def authenticate_phase980() -> dict[str, Any]:
    report = boundary.load_json(PHASE980_REPORT, "Phase980 feasibility report")
    verify_self_hash(report, "report_sha256", "created_at_utc", "Phase980 report")
    require(report.get("phase") == 980 and report.get("design_only") is True,
            "Phase980 is not the design-only source")
    require(report.get("gpu_authorized") is False
            and report.get("holdout") is False
            and report.get("mechanism") is False,
            "Phase980 design boundary changed")
    require(report.get("source_decision", {}).get("P") == []
            and report.get("source_decision", {}).get("P_is_empty") is True,
            "Phase980 source P is not empty")
    design = report.get("future_confirmation_gate_design", {})
    require(design.get("population", {}).get("N_per_stream") == 256
            and design.get("population", {}).get("stream_count") == 3,
            "Phase980 future denominator changed")
    require(PHASE980_SCRIPT.is_file(), "missing Phase980 design script")
    rebuilt = phase980_design.build_report()
    require(rebuilt.get("report_sha256") == report["report_sha256"],
            "current Phase980 script does not reproduce the sealed design report")
    return {
        "script": {
            "path": relative(PHASE980_SCRIPT),
            "file_sha256": boundary.sha256_file(PHASE980_SCRIPT),
        },
        "report": {
            "path": relative(PHASE980_REPORT),
            "file_sha256": boundary.sha256_file(PHASE980_REPORT),
            "report_sha256": report["report_sha256"],
        },
        "authenticated_design_only": True,
        "current_script_reproduces_report": True,
        "gpu_authorized": False,
        "holdout": False,
        "mechanism": False,
        "phase981_narrowing": (
            "Phase981 preselects only the semantic reservoir route before generation; "
            "the Phase980 censor route is secondary descriptive and cannot admit Phase981."
        ),
    }


def dataset_audit() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    items = fresh.build_items()
    audit = fresh.audit_items(items)
    require(isinstance(audit, dict) and (audit.get("passed") is True
            or audit.get("ok") is True), "fresh dataset audit failed")
    require(len(items) == core.ITEM_COUNT, "fresh dataset is not 256 items")
    required = {"id", "task", "difficulty", "prompt", "answer"}
    for item in items:
        require(required <= set(item), f"fresh item lacks fields: {item.get('id')}")
        require(item["difficulty"] in core.DIFFICULTIES, "unknown difficulty")
        require(item["answer"] in {"A", "B"}, "answer is not A/B")
        require(str(item["difficulty"]).casefold()
                not in str(item["prompt"]).casefold(),
                "difficulty label leaks into visible prompt")
    tasks = tuple(getattr(fresh, "TASKS", sorted({str(x["task"]) for x in items})))
    require(len(tasks) == core.TASK_COUNT, "fresh task count changed")
    task_counts = Counter(str(item["task"]) for item in items)
    difficulty_counts = Counter(str(item["difficulty"]) for item in items)
    label_counts = Counter(str(item["answer"]) for item in items)
    require(all(task_counts[task] == core.ITEMS_PER_TASK for task in tasks),
            "fresh task balance changed")
    require(difficulty_counts == {"easy": 128, "hard": 128},
            "fresh difficulty balance changed")
    require(label_counts == {"A": 128, "B": 128}, "fresh label balance changed")
    for task in tasks:
        for difficulty in core.DIFFICULTIES:
            selected = [item for item in items if item["task"] == task
                        and item["difficulty"] == difficulty]
            require(len(selected) == 16, f"task/difficulty balance changed: {task}")
            require(Counter(str(item["answer"]) for item in selected)
                    == {"A": 8, "B": 8},
                    f"task/difficulty/label balance changed: {task}/{difficulty}")
    old_items = phase979_data.build_items()
    id_overlap = ({str(item["id"]) for item in items}
                  & {str(item["id"]) for item in old_items})
    require(not id_overlap,
            "fresh IDs overlap Phase979")
    prompt_overlap = (
        {" ".join(str(item["prompt"]).casefold().split()) for item in items}
        & {" ".join(str(item["prompt"]).casefold().split()) for item in old_items}
    )
    require(not prompt_overlap,
            "fresh prompts overlap Phase979")
    structural_overlap = (
        {boundary.sha256_json({
            "task": item.get("task"), "spec": item.get("spec"),
            "options": item.get("options"),
        }) for item in items}
        & {boundary.sha256_json({
            "task": item.get("task"), "spec": item.get("spec"),
            "options": item.get("options"),
        }) for item in old_items}
    )
    spec_overlap = (
        {boundary.sha256_json({
            "task": item.get("task"), "spec": item.get("spec"),
        }) for item in items}
        & {boundary.sha256_json({
            "task": item.get("task"), "spec": item.get("spec"),
        }) for item in old_items}
    )
    require(not structural_overlap and not spec_overlap,
            "fresh structural payload overlaps Phase979")
    identity = audit.get("identity")
    if not isinstance(identity, dict) and hasattr(fresh, "dataset_identity"):
        identity = fresh.dataset_identity()
    require(isinstance(identity, dict), "fresh audit lacks dataset identity")
    identity_sha = identity.get("identity_sha256") or identity.get("dataset_identity_sha256")
    require(isinstance(identity_sha, str) and len(identity_sha) == 64,
            "fresh identity hash invalid")
    return items, audit, {
        **identity,
        "identity_sha256": identity_sha,
        "task_registry": list(tasks),
        "freshness_against_phase979": {
            "normalized_prompt_overlap_n": len(prompt_overlap),
            "structural_payload_overlap_n": len(structural_overlap),
            "task_spec_overlap_n": len(spec_overlap),
            "id_overlap_n": len(id_overlap),
        },
    }


def gate_audit() -> dict[str, Any]:
    contract = getattr(semantic_gate, "GATE_CONTRACT", None)
    require(isinstance(contract, dict), "semantic gate lacks GATE_CONTRACT")
    result = semantic_gate.self_test()
    require(isinstance(result, dict) and result.get("passed") is True,
            "semantic gate self-test failed")
    require(contract.get("denominators", {}).get("per_stream") == 256,
            "semantic gate N changed")
    require(contract.get("arms", {}).get("contrast") == "B-A",
            "semantic direction changed")
    require(contract.get("states") == ["V", "C", "I_mode", "I_sem"],
            "semantic channel registry changed")
    require(contract.get("primary_semantic", {}).get(
        "censor_route_can_pass_primary") is False,
            "semantic-only primary changed")
    require(contract.get("secondary_censor_descriptive", {}).get(
        "can_change_primary") is False,
            "censor route became admissive")
    return {
        "contract": json.loads(json.dumps(contract)),
        "contract_sha256": boundary.sha256_json(contract),
        "self_test": result,
    }


def dataset_artifact_seals(
    items: list[dict[str, Any]], runtime_audit: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    dataset_document = boundary.load_json(
        core.DATASET_ARTIFACT_PATH, "Phase981 dataset artifact")
    audit_document = boundary.load_json(
        core.DATASET_AUDIT_PATH, "Phase981 dataset artifact audit")
    dataset_payload = boundary.without_fields(dataset_document, "dataset_sha256")
    require(dataset_document.get("dataset_sha256")
            == boundary.sha256_json(dataset_payload),
            "dataset artifact self-hash invalid")
    audit_payload = boundary.without_fields(audit_document, "audit_sha256")
    require(audit_document.get("audit_sha256") == boundary.sha256_json(audit_payload),
            "dataset audit artifact self-hash invalid")
    artifact_items = dataset_document.get("items")
    require(isinstance(artifact_items, list)
            and boundary.sha256_json(sorted(
                artifact_items, key=lambda item: str(item.get("id", ""))))
            == boundary.sha256_json(sorted(
                items, key=lambda item: str(item.get("id", "")))),
            "dataset artifact items differ from source builder")
    require(dataset_document.get("identity")
            == {key: value for key, value in identity.items()
                if key not in {"task_registry", "freshness_against_phase979"}},
            "dataset artifact identity differs")
    require(dataset_document.get("identity", {}).get("items_sha256")
            == identity.get("items_sha256"),
            "dataset artifact items hash differs from runtime identity")
    require(audit_document.get("passed") is True
            and audit_document.get("ok") is True
            and audit_document.get("holdout_accessed") is False
            and audit_document.get("model_weights_loaded") is False
            and audit_document.get("generation_performed") is False,
            "dataset audit artifact crossed design boundary")
    require(audit_document.get("identity") == dataset_document.get("identity"),
            "dataset/audit artifact identity mismatch")
    require(audit_document.get("dataset_document_sha256")
            == dataset_document["dataset_sha256"],
            "dataset audit references wrong dataset self-hash")
    dataset_file_sha = boundary.sha256_file(core.DATASET_ARTIFACT_PATH)
    require(audit_document.get("dataset_file_sha256") == dataset_file_sha,
            "dataset audit references wrong dataset file hash")
    freshness = audit_document.get("freshness_against_phase979_public", {})
    require(freshness.get("passed") is True
            and freshness.get("normalized_prompt_overlap_n") == 0
            and freshness.get("structural_payload_overlap_n") == 0,
            "dataset artifact freshness audit failed")
    require(audit_document.get("script_sha256")
            == boundary.sha256_file(SCRIPT_PATHS["dataset"]),
            "dataset audit script seal mismatch")
    require(runtime_audit.get("identity") == audit_document.get("identity"),
            "runtime/artifact dataset audit identity mismatch")
    return {
        "dataset": {
            "path": relative(core.DATASET_ARTIFACT_PATH),
            "file_sha256": dataset_file_sha,
            "dataset_sha256": dataset_document["dataset_sha256"],
        },
        "audit": {
            "path": relative(core.DATASET_AUDIT_PATH),
            "file_sha256": boundary.sha256_file(core.DATASET_AUDIT_PATH),
            "audit_sha256": audit_document["audit_sha256"],
        },
        "normalized_prompt_overlap_n": 0,
        "structural_payload_overlap_n": 0,
        "identity_sha256": identity["identity_sha256"],
    }


def model_identity(source_protocol: dict[str, Any]) -> dict[str, Any]:
    identity = source_protocol.get("local_model_artifact_identity")
    core.verify_model_artifact_identity(identity)
    model_root = ROOT / str(identity["path"])
    configured = Path(MODEL_CONFIGS[core.MODEL_NAME]["path"]).resolve()
    require(model_root.resolve() == configured and model_root.is_dir(),
            "Qwen model path changed")
    return json.loads(json.dumps(identity))


def tokenizer_audit(items: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[core.MODEL_NAME]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        a_id = boundary.single_token_id(tok, "A")
        b_id = boundary.single_token_id(tok, "B")
        think_open = boundary.single_token_id(tok, "<think>")
        think_close = boundary.single_token_id(tok, "</think>")
        core.verify_think_and_answer_token_ids(
            think_open, think_close, a_id, b_id)
        token_identity = core.token_identity_from_artifacts(tok.eos_token_id)
        probes: list[dict[str, Any]] = []
        for item in items:
            for arm in core.ARMS:
                user, rendered, ids = core.render_prefix(tok, item, arm)
                require(not boundary.positions_of(ids, {think_open, think_close}),
                        "soft arm unexpectedly prefills generated think tags")
                expected_suffix = (" /no_think" if arm == "A" else " /think")
                require(user == str(item["prompt"]) + expected_suffix,
                        "formal soft suffix changed")
                probes.append({
                    "id": item["id"], "arm": arm,
                    "user_sha256": boundary.sha256_json(user),
                    "rendered_sha256": boundary.sha256_json(rendered),
                    "input_ids_sha256": boundary.sha256_json(ids),
                    "prompt_len": len(ids),
                })
        require(len(probes) == 512, "tokenizer probe denominator changed")
        return {
            "tokenizer_class": type(tok).__name__,
            "tokenizer_length": len(tok),
            "chat_template_sha256": boundary.sha256_json(
                str(getattr(tok, "chat_template", ""))),
            "eos_token_id": int(tok.eos_token_id),
            "pad_token_id": int(tok.pad_token_id),
            "token_identity": token_identity,
            "think_open_id": think_open,
            "think_close_id": think_close,
            "A_id": a_id, "B_id": b_id,
            "prefix_count": len(probes),
            "prefixes_sha256": boundary.sha256_json(probes),
        }
    finally:
        del tok
        gc.collect()


def script_seals() -> dict[str, dict[str, str]]:
    return core.build_file_seals(core.PHASE981_SCRIPT_PATHS, "Phase981 script")


def dependency_seals() -> dict[str, dict[str, str]]:
    return core.build_file_seals(
        core.RUNTIME_DEPENDENCY_PATHS, "Phase981 runtime dependency")


def integrity_negative_self_test(protocol: dict[str, Any]) -> dict[str, bool]:
    def rejected(mutator, validator) -> bool:
        tampered = json.loads(json.dumps(protocol))
        mutator(tampered)
        try:
            validator(tampered)
        except RuntimeError:
            return True
        return False

    checks = {
        "empty_phase981_script_seals_rejected": rejected(
            lambda value: value.__setitem__("script_seals", {}),
            core.verify_protocol_file_seals,
        ),
        "changed_dependency_hash_rejected": rejected(
            lambda value: value["dependency_seals"][
                "phase979_boundary_core"].__setitem__("sha256", "0" * 64),
            core.verify_protocol_file_seals,
        ),
        "empty_phase979_script_hashes_rejected": rejected(
            lambda value: value["phase979_source"].__setitem__(
                "phase979_script_hashes", {}),
            core.verify_protocol_file_seals,
        ),
        "empty_model_file_registry_rejected": rejected(
            lambda value: value["local_model_artifact_identity"].__setitem__(
                "files", {}),
            lambda value: core.verify_model_artifact_identity(
                value["local_model_artifact_identity"]),
        ),
        "expanded_intervention_scope_rejected": rejected(
            lambda value: value["test_authorization_scope"].__setitem__(
                "layer_span_cross_time_interventions", True),
            lambda value: core.verify_authorization_scope(
                value["test_authorization_scope"]),
        ),
        "changed_eos_identity_rejected": rejected(
            lambda value: value["tokenizer_audit"]["token_identity"].__setitem__(
                "effective_eos_token_ids", [core.EXPECTED_TOKENIZER_EOS_ID]),
            lambda value: core.verify_protocol_token_identity(
                value["tokenizer_audit"], core.EXPECTED_TOKENIZER_EOS_ID,
                core.EXPECTED_THINK_OPEN_ID, core.EXPECTED_THINK_CLOSE_ID,
                core.EXPECTED_A_ID, core.EXPECTED_B_ID),
        ),
        "protocol_holdout_loaded_true_rejected": rejected(
            lambda value: value.__setitem__("holdout_loaded", True),
            core.verify_protocol_boundary_contract,
        ),
        "protocol_gpu_authorized_true_rejected": rejected(
            lambda value: value["execution_contract"].__setitem__(
                "gpu_authorized", True),
            core.verify_protocol_boundary_contract,
        ),
        "admission_holdout_loaded_true_rejected": rejected(
            lambda value: value["test_admission"].__setitem__(
                "holdout_loaded", True),
            lambda value: core.verify_admission_boundary_contract(
                value["test_admission"]),
        ),
    }
    require(all(checks.values()), "integrity negative self-test failed")
    return checks


def seed_audit(items: list[dict[str, Any]], identity_sha256: str) -> dict[str, Any]:
    values = [
        {
            "id": str(item["id"]), "stream": stream,
            "seed": core.stable_pair_seed(identity_sha256, str(item["id"]), stream),
        }
        for stream in core.STREAMS for item in items
    ]
    require(len(values) == 768 and len({value["seed"] for value in values}) == 768,
            "pair seed collision or denominator change")
    return {
        "pair_seed_count": len(values),
        "unique_pair_seed_count": len({value["seed"] for value in values}),
        "all_three_streams_distinct_per_item": all(
            len({core.stable_pair_seed(identity_sha256, str(item["id"]), stream)
                 for stream in core.STREAMS}) == 3 for item in items
        ),
        "seed_registry_sha256": boundary.sha256_json(values),
    }


def assert_clean_first_seal() -> None:
    if core.PROTOCOL_PATH.exists():
        return
    forbidden = [
        core.ADMISSION_PATH, core.MANIFEST_PATH, core.ROWS_PATH,
        core.STATUS_PATH, core.AUDIT_PATH, core.RUN_LOCK_PATH,
    ]
    existing = [relative(path) for path in forbidden if path.exists()]
    require(not existing, f"Phase981 output exists before protocol seal: {existing}")


def build_protocol() -> dict[str, Any]:
    assert_no_holdout_import()
    core.assert_contract()
    phase979, source_protocol = authenticate_phase979()
    phase980 = authenticate_phase980()
    items, data_audit, identity = dataset_audit()
    artifact_seals = dataset_artifact_seals(items, data_audit, identity)
    gate = gate_audit()
    seeds = seed_audit(items, identity["identity_sha256"])
    runtime = runtime_versions()
    require(runtime == source_protocol.get("runtime_versions"),
            "runtime differs from Phase979 formal runtime")
    seals = script_seals()
    dependencies = dependency_seals()
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "role": "fresh_precommitted_confirmation_protocol",
        "phase979_source": phase979,
        "phase980_design_source": phase980,
        "script_seals": seals,
        "dependency_seals": dependencies,
        "dataset_identity": identity,
        "dataset_audit_sha256": boundary.sha256_json(data_audit),
        "dataset_artifact_seals": artifact_seals,
        "dataset_contract": {
            "fresh_source_precommitted_not_analyst_blind_holdout": True,
            "n_items": 256, "tasks": identity["task_registry"],
            "items_per_task": 32,
            "difficulty_counts": {"easy": 128, "hard": 128},
            "per_task_difficulty_count": 16,
            "per_task_difficulty_label_count": 8,
            "difficulty_hidden_from_visible_prompt": True,
            "phase979_id_overlap_n": identity["freshness_against_phase979"][
                "id_overlap_n"],
            "phase979_normalized_prompt_overlap_n": identity[
                "freshness_against_phase979"]["normalized_prompt_overlap_n"],
            "phase979_structural_payload_overlap_n": identity[
                "freshness_against_phase979"]["structural_payload_overlap_n"],
            "phase979_task_spec_overlap_n": identity[
                "freshness_against_phase979"]["task_spec_overlap_n"],
            "old_holdout_imported": False,
            "old_holdout_parsed": False,
        },
        "model_name": core.MODEL_NAME,
        "local_model_artifact_identity": model_identity(source_protocol),
        "runtime_versions": runtime,
        "tokenizer_audit": tokenizer_audit(items),
        "arms": core.ARMS,
        "direction": core.PRIMARY_DIRECTION,
        "sampling": core.SAMPLING,
        "streams": list(core.STREAMS),
        "common_random_number_contract": {
            "same_item_stream_A_and_B_share_seed": True,
            "arm_excluded_from_seed": True,
            "seed_formula": (
                "sha256('phase981|fresh256|dataset_identity|item=<id>|stream=<r>')[:8] "
                "mod (2**31-1)"
            ),
            "same_256_items_reused_in_all_three_streams": True,
            "distinct_seed_namespace_per_stream": True,
            "streams_are_not_independent_datasets": True,
            "streams_are_not_used_for_variance_estimation": True,
            "three_stream_gate_is_a_preregistered_count_AND_not_an_independence_inference": True,
            "observations_per_stream": 256,
            "related_easy_hard_construction_pairs": 128,
            "observations_are_not_256_independent_samples": True,
            "coupling_is_precommitted_but_not_a_mechanism_claim": True,
            "seed_audit": seeds,
        },
        "expected_rows": core.EXPECTED_ROWS,
        "batch_size": core.BATCH_SIZE,
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "checkpoints": list(core.CHECKPOINTS),
        "decision_checkpoint": core.DECISION_CHECKPOINT,
        "trajectory_contract": {
            "single_rollout_per_row": True,
            "checkpoint_snapshots_are_prefixes_not_reruns": True,
            "first_eos_absorbing": True,
            "per_row_private_cuda_generator": True,
            "cap_categories_are_right_censored_not_failures": True,
            "six_terminal_states": list(core.SIX_STATES),
        },
        "semantic_gate": gate,
        "primary_decision": {
            "route": "semantic_only",
            "direction": "B_minus_A",
            "formula": "stream_0 AND stream_1 AND stream_2 semantic gate",
            "censor_route_can_admit": False,
            "mixed_route_can_admit": False,
            "decision_checkpoint": 2048,
        },
        "transition_matrices": {
            "mandatory_per_stream_4x4": list(core.FOUR_CHANNELS),
            "secondary_per_stream_3x3": list(core.THREE_CHANNELS),
            "secondary_per_stream_6x6": list(core.SIX_STATES),
            "rows_are_A_states_columns_are_B_states": True,
            "all_margins_must_recover_arm_counts": True,
            "paired_matrix_is_not_internal_mechanism_evidence": True,
        },
        "interpretation_boundary": (
            "A PASS confirms only the external soft-configuration bundle B over A on "
            "fresh items for the sealed Qwen3 model under the frozen common-random-number "
            "coupling. It does not isolate thinking, decoding, layers, spans, or an "
            "internal mechanism, and it cannot be generalized directly to GLM or "
            "DeepSeek models."
        ),
        "model_scope_contract": json.loads(json.dumps(core.MODEL_SCOPE_CONTRACT)),
        "execution_contract": json.loads(json.dumps(
            core.PROTOCOL_EXECUTION_CONTRACT)),
        "holdout": False, "holdout_loaded": False, "holdout_authorized": False,
        "mechanism": False, "mechanism_authorized": False,
    }
    core.verify_protocol_file_seals(payload)
    core.verify_protocol_boundary_contract(payload)
    test_admission = {
        "decision": "ADMIT_QWEN_EXTERNAL_GENERATION",
        "admitted": True,
        "qwen_external_generation_authorized": True,
        "gpu_authorized": True,
        "authorization_scope": json.loads(json.dumps(core.AUTHORIZATION_SCOPE)),
        "model_weights_loaded": False,
        "generation_performed": False,
        "gpu_used": False,
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    core.verify_admission_boundary_contract(test_admission)
    payload["integrity_contract"] = {
        "script_seals_exact": True,
        "runtime_dependency_seals_exact": True,
        "phase979_script_hashes_persisted_and_exact": True,
        "model_artifact_file_registry_exact_and_nonempty": True,
        "negative_tamper_tests": integrity_negative_self_test({
            **payload,
            "test_authorization_scope": json.loads(json.dumps(
                core.AUTHORIZATION_SCOPE)),
            "test_admission": test_admission,
        }),
    }
    core.verify_protocol_integrity_metadata(payload)
    assert_no_holdout_import()
    return {
        **payload,
        "protocol_sha256": boundary.sha256_json(payload),
        "created_at_utc": boundary.utc_now(),
    }


def install_or_validate(document: dict[str, Any], freeze: bool) -> None:
    verify_self_hash(document, "protocol_sha256", "created_at_utc", "Phase981 protocol")
    if core.PROTOCOL_PATH.exists():
        prior = boundary.load_json(core.PROTOCOL_PATH, "existing Phase981 protocol")
        verify_self_hash(prior, "protocol_sha256", "created_at_utc",
                         "existing Phase981 protocol")
        require(prior["protocol_sha256"] == document["protocol_sha256"],
                "existing Phase981 protocol differs")
        return
    require(freeze, "protocol is not sealed; rerun with --freeze")
    core.OUT.mkdir(parents=True, exist_ok=True)
    boundary.atomic_write_json(core.PROTOCOL_PATH, document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert_clean_first_seal()
    protocol = build_protocol()
    if not args.self_test:
        install_or_validate(protocol, bool(args.freeze))
    print(json.dumps({
        "phase": core.PHASE,
        "protocol_sha256": protocol["protocol_sha256"],
        "self_test": bool(args.self_test),
        "sealed": core.PROTOCOL_PATH.exists(),
        "expected_rows": core.EXPECTED_ROWS,
        "gpu_authorized": False,
        "holdout": False,
        "mechanism": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
