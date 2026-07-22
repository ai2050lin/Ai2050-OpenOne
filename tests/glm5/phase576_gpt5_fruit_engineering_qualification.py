#!/usr/bin/env python3
"""Run the Phase576 synthetic repeat-forward engineering qualification.

This is an engineering-only CUDA/load/determinism check.  It never parses a
Phase576 research case or reads the sealed-case payload.  Every state
transition is an immutable, atomically published JSON artifact.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import msvcrt
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
GPT5 = ROOT / "tests/gpt5"
GLM5 = ROOT / "tests/glm5"
for path in (GPT5, GLM5):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import phase576_gpt5_fruit_protocol as protocol  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase983_cross_model_engine import (  # noqa: E402
    MODEL_ORDER as ENGINE_MODEL_ORDER,
    load_model_adapter,
    release_model_adapter,
)


QUALIFICATION_PATH = protocol.ENGINEERING_QUALIFICATION_PATH
EXECUTION_DIR = protocol.OUT_DIR / "engineering_qualification_execution"
STAGE_START_PATH = EXECUTION_DIR / "stage_start.json"
EXECUTION_RECEIPT_PATH = EXECUTION_DIR / "execution_receipt.json"
EXECUTION_LEASE_PATH = (
    protocol.OUT_DIR / ".phase576_engineering_qualification.lease"
)
QUARANTINE_INTENT_NAME = "phase576_engineering_quarantine_intent.json"
QUARANTINE_RECEIPT_NAME = "phase576_engineering_quarantine_receipt.json"
QUARANTINE_DIRECTORY_PREFIX = ".phase576-engq-aborted-"
SOURCE_KEY = "tests/glm5/phase576_gpt5_fruit_engineering_qualification.py"
SYNTHETIC_PROMPT = (
    "ENGINEERING QUALIFICATION ONLY; this is not a research case. "
    "Reply with exactly the word amber."
)
EXPECTED_COMPATIBILITY_CHECKS = {
    "cuda_input",
    "cuda_only_no_offload",
    "int8_loaded",
    "sdpa_loaded",
    "hidden_state_count",
    "hidden_size",
    "finite_logits",
    "repeat_logits_exact",
    "repeat_final_state_exact",
    "batch_rows_identical",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def json_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def output_relative_path(path: Path) -> str:
    return str(path.relative_to(protocol.OUT_DIR)).replace("\\", "/")


def atomic_create_json(path: Path, payload: Any) -> str:
    """Atomically publish complete JSON while refusing an existing target.

    A fully flushed same-directory temporary file is hard-linked to the final
    name.  Creating that hard link is an atomic no-replace operation; unlike
    ``os.replace``, it cannot silently overwrite evidence from an earlier run.
    """

    data = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temp.read_bytes() != data:
            raise RuntimeError(f"atomic JSON read-back mismatch: {temp}")
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}") from exc
        if path.read_bytes() != data:
            raise RuntimeError(f"published JSON read-back mismatch: {path}")
        return sha256_bytes(data)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temp.exists():
            temp.unlink()


def acquire_execution_lease() -> Any:
    """Acquire the process-wide Windows lease for this one-shot stage."""

    EXECUTION_LEASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = EXECUTION_LEASE_PATH.open("a+b", buffering=0)
    if EXECUTION_LEASE_PATH.stat().st_size == 0:
        handle.write(b"\0")
        os.fsync(handle.fileno())
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "Phase576 engineering qualification is actively leased by "
            "another process"
        ) from exc
    return handle


def release_execution_lease(handle: Any) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def require_exact_keys(
    payload: dict[str, Any], expected: set[str], label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise RuntimeError(
            f"{label} key registry mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def is_atomic_temp_name(name: str) -> bool:
    prefix = ".tmp-"
    if not name.startswith(prefix):
        return False
    components = name[len(prefix):].split("-")
    return (
        len(components) == 2
        and components[0].isdigit()
        and len(components[1]) == 32
        and all(character in "0123456789abcdef" for character in components[1])
    )


def directory_hash_inventory(
    root: Path, *, ignored_relative_paths: set[str] | None = None
) -> list[dict[str, Any]]:
    """Hash every regular file and register every directory below ``root``.

    Links and special files are refused so an incomplete attempt cannot make
    the quarantine traversal escape the evidence directory.  Each file is
    stat'ed before and after hashing to reject a concurrently changing byte
    stream even if an unrelated program ignores the OS lease.
    """

    ignored = ignored_relative_paths or set()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid engineering evidence directory: {root}")
    inventory: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            path = Path(entry.path)
            relative = str(path.relative_to(root)).replace("\\", "/")
            if relative in ignored:
                continue
            if entry.is_symlink():
                raise RuntimeError(
                    "refusing engineering quarantine containing a link: "
                    f"{relative}"
                )
            if entry.is_dir(follow_symlinks=False):
                inventory.append({"path": relative, "kind": "directory"})
                visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise RuntimeError(
                    "refusing engineering quarantine containing a special "
                    f"filesystem entry: {relative}"
                )
            before = entry.stat(follow_symlinks=False)
            digest = sha256_file(path)
            after = entry.stat(follow_symlinks=False)
            stable_identity = (
                before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and getattr(before, "st_ino", None)
                == getattr(after, "st_ino", None)
            )
            if not stable_identity:
                raise RuntimeError(
                    f"engineering evidence changed while hashing: {relative}"
                )
            inventory.append({
                "path": relative,
                "kind": "file",
                "size_bytes": int(after.st_size),
                "sha256": digest,
            })

    visit(root)
    return sorted(inventory, key=lambda item: (item["path"], item["kind"]))


def validate_quarantine_intent(
    intent: dict[str, Any], intent_path: Path, destination: Path
) -> list[dict[str, Any]]:
    require_exact_keys(intent, {
        "schema_version", "phase_id", "created_at_utc", "reason",
        "original_execution_path", "quarantine_path",
        "terminal_execution_receipt_observed", "file_inventory",
        "file_inventory_sha256", "inventory_entry_count",
        "inventoried_file_count", "inventoried_byte_count",
        "atomic_same_parent_rename_required",
    }, "engineering quarantine intent")
    inventory = intent.get("file_inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("engineering quarantine intent inventory is invalid")
    file_rows = [row for row in inventory if row.get("kind") == "file"]
    checks = (
        intent.get("schema_version")
        == "phase576_engineering_quarantine_intent.v1",
        intent.get("phase_id") == protocol.PHASE,
        isinstance(intent.get("created_at_utc"), str),
        intent.get("reason")
        == "nonterminal_engineering_execution_atomically_quarantined",
        intent.get("original_execution_path") == relative_path(EXECUTION_DIR),
        intent.get("quarantine_path") == relative_path(destination),
        intent.get("terminal_execution_receipt_observed") is False,
        intent.get("file_inventory_sha256") == json_sha256(inventory),
        intent.get("inventory_entry_count") == len(inventory),
        intent.get("inventoried_file_count") == len(file_rows),
        intent.get("inventoried_byte_count")
        == sum(int(row.get("size_bytes", -1)) for row in file_rows),
        intent.get("atomic_same_parent_rename_required") is True,
        intent_path.is_file(),
        not intent_path.is_symlink(),
    )
    if not all(checks):
        raise RuntimeError("engineering quarantine intent failed validation")
    return inventory


def validate_committed_quarantine_inventory(
    root: Path, intent: dict[str, Any]
) -> list[str]:
    """Validate pre-intent bytes, allowing only proven hard-link residues."""

    intent_path = root / QUARANTINE_INTENT_NAME
    receipt_path = root / QUARANTINE_RECEIPT_NAME
    committed = intent["file_inventory"]
    committed_paths = {row["path"] for row in committed}
    observed = directory_hash_inventory(
        root,
        ignored_relative_paths={QUARANTINE_INTENT_NAME, QUARANTINE_RECEIPT_NAME},
    )
    extras = [row for row in observed if row["path"] not in committed_paths]
    allowed_finals = [intent_path]
    if receipt_path.is_file() and not receipt_path.is_symlink():
        allowed_finals.append(receipt_path)
    registered_temps: list[str] = []
    for row in extras:
        relative = row["path"]
        temporary = root / relative
        if not all((
            row.get("kind") == "file",
            "/" not in relative,
            is_atomic_temp_name(relative),
            temporary.is_file(),
            not temporary.is_symlink(),
            any(
                os.path.samefile(temporary, final)
                and sha256_file(temporary) == sha256_file(final)
                for final in allowed_finals
            ),
        )):
            raise RuntimeError(
                "quarantined engineering evidence contains an unregistered "
                f"post-intent artifact: {relative}"
            )
        registered_temps.append(relative)
    filtered = [row for row in observed if row["path"] not in registered_temps]
    if filtered != committed:
        raise RuntimeError(
            "quarantined engineering inventory differs from committed inventory"
        )
    return registered_temps


def publish_quarantine_receipt(
    destination: Path, intent: dict[str, Any], intent_sha256: str
) -> dict[str, Any]:
    intent_path = destination / QUARANTINE_INTENT_NAME
    inventory = validate_quarantine_intent(intent, intent_path, destination)
    validate_committed_quarantine_inventory(destination, intent)
    receipt_path = destination / QUARANTINE_RECEIPT_NAME
    receipt = {
        "schema_version": "phase576_engineering_quarantine_receipt.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "reason": intent["reason"],
        "original_execution_path": intent["original_execution_path"],
        "quarantine_path": intent["quarantine_path"],
        "quarantine_intent_path": relative_path(intent_path),
        "quarantine_intent_sha256": intent_sha256,
        "atomic_same_parent_rename_completed": True,
        "terminal_execution_receipt_observed": False,
        "file_inventory": inventory,
        "file_inventory_sha256": intent["file_inventory_sha256"],
        "inventory_entry_count": intent["inventory_entry_count"],
        "inventoried_file_count": intent["inventoried_file_count"],
        "inventoried_byte_count": intent["inventoried_byte_count"],
    }
    if receipt_path.exists():
        existing = read_json(receipt_path)
        comparable = dict(existing)
        comparable.pop("created_at_utc", None)
        expected = dict(receipt)
        expected.pop("created_at_utc", None)
        if comparable != expected:
            raise RuntimeError(
                "existing engineering quarantine receipt is inconsistent"
            )
        return existing
    atomic_create_json(receipt_path, receipt)
    return receipt


def repair_pending_quarantine_receipts() -> None:
    """Finish receipt publication after a crash following atomic rename."""

    parent = protocol.OUT_DIR
    pattern = f"{QUARANTINE_DIRECTORY_PREFIX}*"
    for destination in sorted(parent.glob(pattern)):
        if not destination.is_dir() or destination.is_symlink():
            raise RuntimeError(
                f"invalid engineering quarantine path: {destination}"
            )
        intent_path = destination / QUARANTINE_INTENT_NAME
        if not intent_path.is_file() or intent_path.is_symlink():
            raise RuntimeError(
                "engineering quarantine lacks its persistent intent: "
                f"{destination}"
            )
        intent = read_json(intent_path)
        validate_quarantine_intent(intent, intent_path, destination)
        publish_quarantine_receipt(
            destination, intent, sha256_file(intent_path)
        )


def quarantine_incomplete_execution() -> dict[str, Any] | None:
    """Atomically preserve a hard-crashed nonterminal execution and rerun."""

    source = EXECUTION_DIR
    if not source.exists():
        return None
    if EXECUTION_RECEIPT_PATH.exists():
        raise RuntimeError(
            "refusing to quarantine engineering evidence with an execution receipt"
        )
    parent = source.parent.resolve(strict=True)
    if source.is_symlink() or source.resolve(strict=True).parent != parent:
        raise RuntimeError("refusing to quarantine aliased engineering evidence")

    existing_intent_path = source / QUARANTINE_INTENT_NAME
    if existing_intent_path.exists():
        if not existing_intent_path.is_file() or existing_intent_path.is_symlink():
            raise RuntimeError("invalid existing engineering quarantine intent")
        intent = read_json(existing_intent_path)
        destination_relative = intent.get("quarantine_path")
        if not isinstance(destination_relative, str):
            raise RuntimeError("engineering quarantine destination is missing")
        destination = (ROOT / destination_relative).resolve()
        try:
            destination.relative_to(parent)
        except ValueError as exc:
            raise RuntimeError(
                "engineering quarantine destination escapes output directory"
            ) from exc
        if destination.parent != parent:
            raise RuntimeError("engineering quarantine must remain in one parent")
        inventory = validate_quarantine_intent(
            intent, existing_intent_path, destination
        )
        validate_committed_quarantine_inventory(source, intent)
    else:
        inventory = directory_hash_inventory(source)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = parent / (
            f"{QUARANTINE_DIRECTORY_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}"
        )
        file_rows = [row for row in inventory if row["kind"] == "file"]
        intent = {
            "schema_version": "phase576_engineering_quarantine_intent.v1",
            "phase_id": protocol.PHASE,
            "created_at_utc": now(),
            "reason": "nonterminal_engineering_execution_atomically_quarantined",
            "original_execution_path": relative_path(source),
            "quarantine_path": relative_path(destination),
            "terminal_execution_receipt_observed": False,
            "file_inventory": inventory,
            "file_inventory_sha256": json_sha256(inventory),
            "inventory_entry_count": len(inventory),
            "inventoried_file_count": len(file_rows),
            "inventoried_byte_count": sum(
                int(row["size_bytes"]) for row in file_rows
            ),
            "atomic_same_parent_rename_required": True,
        }
        atomic_create_json(existing_intent_path, intent)

    if destination.exists():
        raise RuntimeError("engineering quarantine destination already exists")
    os.rename(source, destination)
    intent_path = destination / QUARANTINE_INTENT_NAME
    receipt = publish_quarantine_receipt(
        destination, intent, sha256_file(intent_path)
    )
    return {
        "quarantine_path": relative_path(destination),
        "quarantine_receipt_path": relative_path(
            destination / QUARANTINE_RECEIPT_NAME
        ),
        "quarantine_receipt_sha256": sha256_file(
            destination / QUARANTINE_RECEIPT_NAME
        ),
        "file_inventory_sha256": receipt["file_inventory_sha256"],
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def cuda_memory_snapshot() -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append({
                "device_index": index,
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            })
    return {
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
        "total_allocated_bytes": sum(row["allocated_bytes"] for row in devices),
        "total_reserved_bytes": sum(row["reserved_bytes"] for row in devices),
    }


def execution_contract() -> dict[str, Any]:
    return {
        "qualification_kind": "repeat_forward_engineering_qualification",
        "synthetic_nonresearch_prompt_sha256": sha256_bytes(
            SYNTHETIC_PROMPT.encode("utf-8")
        ),
        "required_model_order": list(protocol.MODELS),
        "batch_size": 2,
        "repeat_forward_count": 2,
        "greedy_generation_capsule_repeat_count": 2,
        "greedy_generation_max_new_tokens": protocol.MAX_NEW_TOKENS,
        "greedy_generation_do_sample": False,
        "greedy_generation_return_dict_in_generate": True,
        "greedy_generation_output_logits": True,
        "greedy_generation_output_hidden_states": True,
        "greedy_generation_use_cache": True,
        "generation_step_zero_requires_full_prompt": True,
        "generation_steps_after_zero_require_single_token": True,
        "generation_every_step_every_hidden_state_finite": True,
        "generation_capsule_exact_repeat_required": True,
        "tokenizer_add_special_tokens": True,
        "tokenizer_padding": True,
        "tokenizer_truncation": False,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
        "cuda_required": True,
        "cuda_only_no_offload_required": True,
        "int8_required": True,
        "nonquantized_dtype_required": "torch.bfloat16",
        "attention_implementation_required": "sdpa",
        "model_eval_required": True,
        "finite_logits_required_for_every_forward": True,
        "finite_hidden_state_required_for_every_layer_and_forward": True,
        "hidden_state_bfloat16_required_for_every_layer_and_forward": True,
        "exact_repeat_logits_required": True,
        "exact_repeat_hidden_states_required_for_every_layer": True,
        "identical_batch_rows_required_for_logits_and_every_hidden_state": True,
        "per_model_cuda_allocated_bytes_after_release_required": 0,
        "continue_after_clean_model_failure": True,
        "stop_before_next_model_after_unclean_cuda_release": True,
        "final_qualification_requires_all_three_models": True,
        "formal_case_content_parsed": False,
        "sealed_case_payload_read": False,
        "activation_persisted": False,
        "causal_intervention": False,
    }


def verify_integrity_before_or_after_models() -> dict[str, Any]:
    required = (
        protocol.PROTOCOL_PATH,
        protocol.STATIC_AUDIT_PATH,
        protocol.SEALED_COMMITMENT_PATH,
        protocol.FREEZE_COMMIT_PATH,
        protocol.OPEN_CASES_PATH,
        *protocol.OPEN_SPLIT_CASE_PATHS.values(),
        protocol.SEALED_CASES_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing Phase576 frozen artifacts: {missing}")

    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.STATIC_AUDIT_PATH)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    freeze_commit = read_json(protocol.FREEZE_COMMIT_PATH)
    protocol.verify_frozen_source_seals(frozen)
    protocol.verify_frozen_model_artifacts(frozen)
    prior_open_identities, _ = protocol.prior_open_file_snapshots()

    protocol_hash = sha256_file(protocol.PROTOCOL_PATH)
    audit_hash = sha256_file(protocol.STATIC_AUDIT_PATH)
    commitment_hash = sha256_file(protocol.SEALED_COMMITMENT_PATH)
    freeze_commit_hash = sha256_file(protocol.FREEZE_COMMIT_PATH)
    open_hash = sha256_file(protocol.OPEN_CASES_PATH)
    split_hashes = {
        split: sha256_file(path)
        for split, path in protocol.OPEN_SPLIT_CASE_PATHS.items()
    }
    source_hash = sha256_file(Path(__file__).resolve())
    sealed_policy = frozen.get("sealed_policy", {})
    trace_policy = frozen.get("trace_policy", {})
    committed_artifacts = freeze_commit.get("artifact_sha256_by_path", {})
    expected_committed_paths = {
        output_relative_path(protocol.OPEN_CASES_PATH),
        *(output_relative_path(path) for path in protocol.OPEN_SPLIT_CASE_PATHS.values()),
        output_relative_path(protocol.SEALED_CASES_PATH),
        output_relative_path(protocol.SEALED_COMMITMENT_PATH),
        output_relative_path(protocol.PROTOCOL_PATH),
        output_relative_path(protocol.STATIC_AUDIT_PATH),
    }
    observed_nonsealed_artifact_hashes = {
        output_relative_path(protocol.OPEN_CASES_PATH): open_hash,
        **{
            output_relative_path(protocol.OPEN_SPLIT_CASE_PATHS[split]): value
            for split, value in split_hashes.items()
        },
        output_relative_path(protocol.SEALED_COMMITMENT_PATH): commitment_hash,
        output_relative_path(protocol.PROTOCOL_PATH): protocol_hash,
        output_relative_path(protocol.STATIC_AUDIT_PATH): audit_hash,
    }
    frozen_prior_open_identities = frozen.get("prior_open_file_identities")
    audit_prior_open_identities = audit.get("prior_open_file_identities")
    created_times = {
        frozen.get("created_at_utc"),
        audit.get("created_at_utc"),
        commitment.get("created_at_utc"),
        freeze_commit.get("created_at_utc"),
    }
    checks = {
        "frozen_schema": frozen.get("schema_version") == protocol.SCHEMA_VERSION,
        "frozen_phase": frozen.get("phase_id") == protocol.PHASE,
        "frozen_created": isinstance(frozen.get("created_at_utc"), str)
        and bool(frozen.get("created_at_utc")),
        "required_model_order": frozen.get("models_in_required_execution_order")
        == list(protocol.MODELS)
        and tuple(protocol.MODELS) == tuple(ENGINE_MODEL_ORDER),
        "model_artifact_registry": set(
            frozen.get("model_artifact_identities", {})
        ) == set(protocol.MODELS),
        "protocol_source_hash": frozen.get("source_script_sha256")
        == sha256_file(Path(protocol.__file__).resolve()),
        "qualification_source_hash": frozen.get("stage_source_seals", {})
        .get(SOURCE_KEY, {}).get("sha256") == source_hash,
        "audit_schema": audit.get("schema_version")
        == "phase576_static_audit.v2",
        "audit_phase": audit.get("phase_id") == protocol.PHASE,
        "audit_created": isinstance(audit.get("created_at_utc"), str)
        and bool(audit.get("created_at_utc")),
        "audit_valid": audit.get("valid") is True
        and audit.get("failures") == [],
        "audit_protocol_hash": audit.get("protocol_sha256") == protocol_hash,
        "commitment_schema": commitment.get("schema_version")
        == "phase576_sealed_commitment.v2",
        "commitment_phase": commitment.get("phase_id") == protocol.PHASE,
        "commitment_created": isinstance(commitment.get("created_at_utc"), str)
        and bool(commitment.get("created_at_utc")),
        "frozen_commitment_hash": frozen.get("sealed_commitment_sha256")
        == commitment_hash,
        "audit_commitment_hash": audit.get("sealed_commitment_sha256")
        == commitment_hash,
        "freeze_commit_schema": freeze_commit.get("schema_version")
        == "phase576_freeze_commit.v1",
        "freeze_commit_phase": freeze_commit.get("phase_id") == protocol.PHASE,
        "freeze_commit_complete": freeze_commit.get("complete") is True,
        "freeze_commit_no_overwrite": freeze_commit.get("overwrite_allowed")
        is False,
        "freeze_commit_atomic_publish": freeze_commit.get(
            "atomic_directory_publish"
        ) is True,
        "freeze_lock_absent": not protocol.FREEZE_LOCK_PATH.exists(),
        "freeze_created_time_chain": len(created_times) == 1
        and None not in created_times,
        "freeze_artifact_registry": isinstance(committed_artifacts, dict)
        and set(committed_artifacts) == expected_committed_paths
        and freeze_commit.get("artifact_count") == len(expected_committed_paths),
        "freeze_nonsealed_artifact_hashes": isinstance(committed_artifacts, dict)
        and all(
            committed_artifacts.get(path) == value
            for path, value in observed_nonsealed_artifact_hashes.items()
        ),
        "freeze_sealed_hash_bound_without_payload_read": committed_artifacts.get(
            output_relative_path(protocol.SEALED_CASES_PATH)
        ) == commitment.get("sealed_cases_sha256")
        == audit.get("sealed_cases_sha256"),
        "freeze_source_registry_hash": freeze_commit.get(
            "source_seals_sha256"
        ) == protocol.stable_hash(frozen.get("stage_source_seals")),
        "freeze_model_registry_hash": freeze_commit.get(
            "model_artifact_identities_sha256"
        ) == protocol.stable_hash(frozen.get("model_artifact_identities")),
        "freeze_prior_open_registry_hash": freeze_commit.get(
            "prior_open_file_identities_sha256"
        ) == protocol.stable_hash(frozen_prior_open_identities),
        "open_cases_hash_chain": frozen.get("open_cases_sha256")
        == audit.get("open_cases_sha256") == open_hash,
        "open_split_hash_chain": frozen.get("open_case_sha256_by_split")
        == audit.get("open_case_sha256_by_split") == split_hashes,
        "open_case_count_chain": frozen.get("open_case_count")
        == audit.get("open_case_count"),
        "split_case_count_chain": frozen.get("cases_per_split")
        == audit.get("cases_per_split"),
        "sealed_hash_commitment_chain_without_payload_read":
        audit.get("sealed_cases_sha256") == commitment.get("sealed_cases_sha256"),
        "sealed_case_count_chain": frozen.get("sealed_case_count")
        == audit.get("sealed_case_count") == commitment.get("sealed_case_count"),
        "holdout_not_blind": commitment.get("holdout_is_blind") is False
        and sealed_policy.get("holdout_is_blind") is False,
        "sealed_definition_public": commitment.get(
            "sealed_definition_is_public_in_source"
        ) is True and sealed_policy.get("sealed_definition_is_public_in_source") is True,
        "sealed_model_unopened": commitment.get("sealed_model_opened") is False,
        "sealed_model_access_zero": type(
            commitment.get("sealed_model_access_count")
        ) is int and commitment.get("sealed_model_access_count") == 0,
        "sealed_analysis_access_zero": type(
            commitment.get("sealed_result_analysis_access_count")
        ) is int and commitment.get("sealed_result_analysis_access_count") == 0,
        "prior_open_identity_chain": frozen_prior_open_identities
        == audit_prior_open_identities == prior_open_identities,
        "prior_open_read_registry": audit.get("prior_open_files_read")
        == [identity["path"] for identity in prior_open_identities],
        "audit_prior_sealed_unread": audit.get("prior_sealed_files_read") == [],
        "commitment_prior_sealed_unread": commitment.get(
            "prior_sealed_files_read"
        ) is False,
        "frozen_prior_sealed_unread": sealed_policy.get(
            "prior_sealed_files_read"
        ) is False,
        "prior_sealed_public_registry": audit.get(
            "prior_sealed_object_ids_registered_from_public_protocols"
        ) == sorted(protocol.PRIOR_SEALED_OBJECT_IDS),
        "prior_sealed_object_disjoint": audit.get(
            "prior_sealed_object_overlap"
        ) == [],
        "audit_did_not_load_models": audit.get("model_weights_loaded") is False,
        "audit_did_not_use_cuda": audit.get("cuda_used") is False,
        "audit_did_not_analyze_sealed": audit.get(
            "sealed_model_or_result_read_for_analysis"
        ) is False,
        "complete_case_grid": frozen.get("case_grid_contract")
        == audit.get("case_grid_audit")
        and isinstance(audit.get("case_grid_audit"), dict)
        and audit["case_grid_audit"].get("valid") is True,
        "cross_model_observational_comparison_rule": isinstance(
            audit.get("cross_model_observational_comparison_rule_audit"), dict
        ) and bool(audit["cross_model_observational_comparison_rule_audit"])
        and all(audit["cross_model_observational_comparison_rule_audit"].values()),
        "no_prefrozen_coordinates": trace_policy.get(
            "candidate_coordinates_before_trace"
        ) == [],
        "no_prefrozen_mechanism_formulas": trace_policy.get(
            "candidate_mechanism_formulas_before_trace"
        ) == [],
        "batch_absorbing_feedback_is_in_scope": trace_policy.get(
            "batch_absorbing_eos_and_pad_feedback_positions_included"
        ) is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576 engineering integrity check failed: {checks}")
    return {
        "frozen": frozen,
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "static_audit_sha256": audit_hash,
        "sealed_commitment_sha256": commitment_hash,
        "freeze_commit_sha256": freeze_commit_hash,
        "open_cases_sha256": open_hash,
        "open_case_sha256_by_split": split_hashes,
        "qualification_source_sha256": source_hash,
        "prior_open_file_identities": prior_open_identities,
        "formal_case_content_parsed": False,
        "open_case_bytes_hashed_for_integrity": True,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
    }


def tensor_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def inspect_generation_output(
    output: Any,
    adapter: Any,
    encoded_input_ids: torch.Tensor,
    layer_count: int,
    hidden_size: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    sequences = getattr(output, "sequences", None)
    hidden_steps = getattr(output, "hidden_states", None)
    generation_logits = getattr(output, "logits", None)
    if not isinstance(sequences, torch.Tensor):
        raise RuntimeError("generation output lacks tensor sequences")
    if not isinstance(hidden_steps, (tuple, list)) or not hidden_steps:
        raise RuntimeError("generation output lacks hidden-state steps")
    if not isinstance(generation_logits, (tuple, list)) or not generation_logits:
        raise RuntimeError("generation output lacks raw-logit steps")
    prompt_width = int(encoded_input_ids.shape[1])
    generated_width = int(sequences.shape[1]) - prompt_width
    if generated_width <= 0:
        raise RuntimeError("generation produced no token step")

    expected_hidden_count = layer_count + 1
    input_device = str(adapter.input_device)
    step_diagnostics: list[dict[str, Any]] = []
    for step_index, hidden_bank in enumerate(hidden_steps):
        if not isinstance(hidden_bank, (tuple, list)):
            raise RuntimeError(f"generation step {step_index} hidden bank invalid")
        expected_sequence_width = prompt_width if step_index == 0 else 1
        layers: list[dict[str, Any]] = []
        for layer_index, hidden in enumerate(hidden_bank):
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(
                    f"generation step {step_index}/layer {layer_index} is not a tensor"
                )
            layers.append({
                "layer_index": layer_index,
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": str(hidden.device),
                "finite": tensor_finite(hidden),
                "batch_rows_identical": hidden.ndim >= 1
                and hidden.shape[0] == 2
                and bool(torch.equal(hidden[0], hidden[1])),
            })
        step_logits = (
            generation_logits[step_index]
            if step_index < len(generation_logits) else None
        )
        logits_valid = isinstance(step_logits, torch.Tensor)
        step_diagnostics.append({
            "step_index": step_index,
            "expected_sequence_width": expected_sequence_width,
            "hidden_state_count": len(hidden_bank),
            "layers": layers,
            "logits_shape": list(step_logits.shape) if logits_valid else None,
            "logits_dtype": str(step_logits.dtype) if logits_valid else None,
            "logits_device": str(step_logits.device) if logits_valid else None,
            "logits_finite": tensor_finite(step_logits) if logits_valid else False,
            "logits_batch_rows_identical": logits_valid
            and step_logits.ndim >= 1
            and step_logits.shape[0] == 2
            and bool(torch.equal(step_logits[0], step_logits[1])),
            "logits_contract_valid": logits_valid
            and list(step_logits.shape) == [2, vocabulary_size]
            and str(step_logits.device) == input_device,
            "all_layers_valid": len(hidden_bank) == expected_hidden_count
            and all(
                row["shape"] == [2, expected_sequence_width, hidden_size]
                and row["dtype"] == "torch.bfloat16"
                and row["device"] == input_device
                and row["finite"]
                and row["batch_rows_identical"]
                for row in layers
            ),
        })

    eos_ids = set(int(value) for value in adapter.eos_identity[
        "effective_eos_token_ids"
    ])
    pad_id = int(adapter.pad_token_id)
    row_capsules: list[dict[str, Any]] = []
    for row_index, sequence in enumerate(sequences):
        token_ids = [int(value) for value in sequence.tolist()]
        suffix = token_ids[prompt_width:]
        first_eos_index = next(
            (index for index, value in enumerate(suffix) if value in eos_ids),
            None,
        )
        content_ids = (
            suffix if first_eos_index is None else suffix[:first_eos_index]
        )
        trailing_ids = (
            [] if first_eos_index is None else suffix[first_eos_index + 1:]
        )
        row_capsules.append({
            "row_index": row_index,
            "generated_token_ids": suffix,
            "generated_token_ids_before_eos": content_ids,
            "first_eos_index": first_eos_index,
            "first_eos_token_id": None
            if first_eos_index is None else suffix[first_eos_index],
            "eos_seen": first_eos_index is not None,
            "budget_terminated": first_eos_index is None
            and len(suffix) == protocol.MAX_NEW_TOKENS,
            "post_eos_tokens_are_absorbing": all(
                value == pad_id or value in eos_ids for value in trailing_ids
            ),
            "token_ids_in_vocabulary": all(
                0 <= value < vocabulary_size for value in token_ids
            ),
            "prompt_prefix_exact": token_ids[:prompt_width]
            == [int(value) for value in encoded_input_ids[row_index].tolist()],
        })
    checks = {
        "sequence_batch_shape": sequences.ndim == 2
        and int(sequences.shape[0]) == 2,
        "generated_width_within_contract": 1
        <= generated_width <= protocol.MAX_NEW_TOKENS,
        "hidden_step_count": len(hidden_steps) == generated_width,
        "logit_step_count": len(generation_logits) == generated_width,
        "every_step_all_layers_valid": all(
            row["all_layers_valid"] for row in step_diagnostics
        ),
        "every_step_logits_finite": all(
            row["logits_finite"] and row["logits_contract_valid"]
            for row in step_diagnostics
        ),
        "every_step_logits_batch_rows_identical": all(
            row["logits_batch_rows_identical"] for row in step_diagnostics
        ),
        "sequence_batch_rows_identical": bool(
            torch.equal(sequences[0], sequences[1])
        ),
        "token_capsules_valid": all(
            row["token_ids_in_vocabulary"]
            and row["prompt_prefix_exact"]
            and row["post_eos_tokens_are_absorbing"]
            and (row["eos_seen"] or row["budget_terminated"])
            for row in row_capsules
        ),
    }
    return {
        "checks": checks,
        "prompt_width": prompt_width,
        "generated_width": generated_width,
        "effective_eos_token_ids": sorted(eos_ids),
        "pad_token_id": pad_id,
        "rows": row_capsules,
        "steps": step_diagnostics,
    }


def collect_repeat_forward_report(adapter: Any, model: str) -> dict[str, Any]:
    adapter.tokenizer.padding_side = "left"
    rendered = render_chat(adapter.tokenizer, model, SYNTHETIC_PROMPT)
    encoded = adapter.tokenizer(
        [rendered, rendered],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    encoded = {
        key: value.to(adapter.input_device) for key, value in encoded.items()
    }
    encoded_rows_identical = all(
        value.ndim < 2 or bool(torch.equal(value[0], value[1]))
        for value in encoded.values()
    )
    with torch.inference_mode():
        first = adapter.model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        second = adapter.model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    if first.hidden_states is None or second.hidden_states is None:
        raise RuntimeError(f"{model}: hidden states unavailable")
    if not isinstance(first.logits, torch.Tensor) or not isinstance(
        second.logits, torch.Tensor
    ):
        raise RuntimeError(f"{model}: logits unavailable")

    layer_count = int(adapter.config.num_hidden_layers)
    hidden_size = int(adapter.config.hidden_size)
    configured_vocabulary_size = int(adapter.config.vocab_size)
    expected_hidden_count = layer_count + 1
    hidden_count_ok = (
        len(first.hidden_states) == expected_hidden_count
        and len(second.hidden_states) == expected_hidden_count
    )
    layer_diagnostics: list[dict[str, Any]] = []
    if hidden_count_ok:
        for layer_index, (left, right) in enumerate(
            zip(first.hidden_states, second.hidden_states)
        ):
            if not isinstance(left, torch.Tensor) or not isinstance(
                right, torch.Tensor
            ):
                raise RuntimeError(f"{model}: non-tensor hidden state at {layer_index}")
            layer_diagnostics.append({
                "layer_index": layer_index,
                "first_shape": list(left.shape),
                "second_shape": list(right.shape),
                "first_dtype": str(left.dtype),
                "second_dtype": str(right.dtype),
                "first_device": str(left.device),
                "second_device": str(right.device),
                "first_finite": tensor_finite(left),
                "second_finite": tensor_finite(right),
                "repeat_exact": bool(torch.equal(left, right)),
                "first_batch_rows_identical": left.ndim >= 1
                and left.shape[0] == 2
                and bool(torch.equal(left[0], left[1])),
                "second_batch_rows_identical": right.ndim >= 1
                and right.shape[0] == 2
                and bool(torch.equal(right[0], right[1])),
            })

    logits_first_finite = tensor_finite(first.logits)
    logits_second_finite = tensor_finite(second.logits)
    repeat_logits_exact = bool(torch.equal(first.logits, second.logits))
    logits_batch_rows_identical = (
        first.logits.ndim >= 1
        and second.logits.ndim >= 1
        and first.logits.shape[0] == second.logits.shape[0] == 2
        and bool(torch.equal(first.logits[0], first.logits[1]))
        and bool(torch.equal(second.logits[0], second.logits[1]))
    )
    expected_hidden_shape = [
        int(first.logits.shape[0]),
        int(first.logits.shape[1]),
        hidden_size,
    ]
    quantization = adapter.identity.get("loaded_quantization", {})
    detailed_checks = {
        "first_logits_finite": logits_first_finite,
        "second_logits_finite": logits_second_finite,
        "logits_shapes_equal": tuple(first.logits.shape)
        == tuple(second.logits.shape),
        "logits_shapes_match_model_contract": list(first.logits.shape)
        == [2, int(encoded["input_ids"].shape[1]), configured_vocabulary_size]
        and list(second.logits.shape)
        == [2, int(encoded["input_ids"].shape[1]), configured_vocabulary_size],
        "logits_on_input_cuda_device": str(first.logits.device)
        == str(second.logits.device) == str(adapter.input_device),
        "all_hidden_states_finite": hidden_count_ok and all(
            row["first_finite"] and row["second_finite"]
            for row in layer_diagnostics
        ),
        "all_hidden_states_bfloat16": hidden_count_ok and all(
            row["first_dtype"] == "torch.bfloat16"
            and row["second_dtype"] == "torch.bfloat16"
            for row in layer_diagnostics
        ),
        "all_hidden_states_on_input_cuda_device": hidden_count_ok and all(
            row["first_device"] == str(adapter.input_device)
            and row["second_device"] == str(adapter.input_device)
            for row in layer_diagnostics
        ),
        "all_hidden_state_shapes_exact": hidden_count_ok and all(
            row["first_shape"] == expected_hidden_shape
            and row["second_shape"] == expected_hidden_shape
            for row in layer_diagnostics
        ),
        "repeat_all_hidden_states_exact": hidden_count_ok and all(
            row["repeat_exact"] for row in layer_diagnostics
        ),
        "all_hidden_state_batch_rows_identical": hidden_count_ok and all(
            row["first_batch_rows_identical"]
            and row["second_batch_rows_identical"]
            for row in layer_diagnostics
        ),
        "encoded_batch_rows_identical": encoded_rows_identical,
        "model_eval": adapter.model.training is False,
        "nonquantized_parameters_bfloat16": quantization.get(
            "floating_parameter_dtypes"
        ) == ["torch.bfloat16"],
        "loaded_on_single_cuda_device": adapter.input_device.type == "cuda"
        and set(adapter.identity.get("hf_device_map", {}).values())
        == {str(adapter.input_device)},
    }
    checks = {
        "cuda_input": adapter.input_device.type == "cuda",
        "cuda_only_no_offload": adapter.identity.get(
            "cuda_only_no_cpu_or_disk_offload"
        ) is True,
        "int8_loaded": quantization.get("load_in_8bit") is True,
        "sdpa_loaded": adapter.identity.get("loaded_attn_implementation")
        == "sdpa",
        "hidden_state_count": hidden_count_ok,
        "hidden_size": hidden_count_ok and all(
            row["first_shape"][-1] == hidden_size
            and row["second_shape"][-1] == hidden_size
            for row in layer_diagnostics
        ),
        "finite_logits": logits_first_finite and logits_second_finite,
        "repeat_logits_exact": repeat_logits_exact,
        "repeat_final_state_exact": hidden_count_ok
        and bool(torch.equal(first.hidden_states[-1], second.hidden_states[-1])),
        "batch_rows_identical": logits_batch_rows_identical,
    }
    if set(checks) != EXPECTED_COMPATIBILITY_CHECKS:
        raise RuntimeError(f"{model}: compatibility check registry drift")
    if not all(checks.values()) or not all(detailed_checks.values()):
        raise RuntimeError(
            f"{model}: repeat-forward engineering checks failed: "
            f"compatibility={checks}, detailed={detailed_checks}"
        )

    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape[0] != 2:
        raise RuntimeError(f"{model}: invalid encoded input_ids")
    direct_sequence_length = int(first.logits.shape[1])
    direct_vocabulary_size = int(first.logits.shape[-1])
    direct_logits_diagnostics = {
        "first_shape": list(first.logits.shape),
        "second_shape": list(second.logits.shape),
        "first_dtype": str(first.logits.dtype),
        "second_dtype": str(second.logits.dtype),
        "first_device": str(first.logits.device),
        "second_device": str(second.logits.device),
        "first_finite": logits_first_finite,
        "second_finite": logits_second_finite,
        "repeat_exact": repeat_logits_exact,
        "batch_rows_identical": logits_batch_rows_identical,
    }
    del first, second

    generation_kwargs = {
        **encoded,
        "max_new_tokens": protocol.MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_hidden_states": True,
        "pad_token_id": adapter.pad_token_id,
        "eos_token_id": list(
            adapter.eos_identity["effective_eos_token_ids"]
        ),
    }
    with torch.inference_mode():
        first_generation = adapter.model.generate(**generation_kwargs)
        second_generation = adapter.model.generate(**generation_kwargs)
    first_generation_diagnostics = inspect_generation_output(
        first_generation,
        adapter,
        input_ids,
        layer_count,
        hidden_size,
        direct_vocabulary_size,
    )
    second_generation_diagnostics = inspect_generation_output(
        second_generation,
        adapter,
        input_ids,
        layer_count,
        hidden_size,
        direct_vocabulary_size,
    )
    first_generation_hidden = first_generation.hidden_states
    second_generation_hidden = second_generation.hidden_states
    first_generation_logits = first_generation.logits
    second_generation_logits = second_generation.logits
    generation_repeat_checks = {
        "sequence_capsule_exact": bool(torch.equal(
            first_generation.sequences, second_generation.sequences
        )),
        "diagnostic_capsule_exact": first_generation_diagnostics
        == second_generation_diagnostics,
        "hidden_step_registry_exact": len(first_generation_hidden)
        == len(second_generation_hidden)
        and all(
            len(left_bank) == len(right_bank)
            and all(torch.equal(left, right) for left, right in zip(
                left_bank, right_bank
            ))
            for left_bank, right_bank in zip(
                first_generation_hidden, second_generation_hidden
            )
        ),
        "logit_step_registry_exact": len(first_generation_logits)
        == len(second_generation_logits)
        and all(torch.equal(left, right) for left, right in zip(
            first_generation_logits, second_generation_logits
        )),
    }
    if (
        not all(first_generation_diagnostics["checks"].values())
        or not all(second_generation_diagnostics["checks"].values())
        or not all(generation_repeat_checks.values())
    ):
        raise RuntimeError(
            f"{model}: greedy generation hidden-state API qualification failed: "
            f"first={first_generation_diagnostics['checks']}, "
            f"second={second_generation_diagnostics['checks']}, "
            f"repeat={generation_repeat_checks}"
        )
    return {
        "schema_version": "phase576_repeat_forward_model_report.v1",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "model": model,
        "passed": True,
        "checks": checks,
        "detailed_checks": detailed_checks,
        "layer_diagnostics": layer_diagnostics,
        "logits_diagnostics": direct_logits_diagnostics,
        "layer_count": layer_count,
        "hidden_state_count": expected_hidden_count,
        "hidden_size": hidden_size,
        "sequence_length": direct_sequence_length,
        "vocabulary_size": direct_vocabulary_size,
        "greedy_generation_capsule": {
            "max_new_tokens": protocol.MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_hidden_states": True,
            "first": first_generation_diagnostics,
            "second": second_generation_diagnostics,
            "repeat_checks": generation_repeat_checks,
        },
        "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
        "input_token_ids_sha256": json_sha256(
            [int(value) for value in input_ids[0].tolist()]
        ),
        "input_token_count": int(input_ids.shape[1]),
        "loaded_model_identity": adapter.identity,
        "formal_case_access": False,
        "formal_case_content_parsed": False,
        "sealed_split_read": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
        "activation_persisted": False,
        "causal_intervention": False,
    }


def model_status_path(index: int, model: str, state: str) -> Path:
    return EXECUTION_DIR / f"{index:02d}_{model}.{state}.json"


def execute_model(
    run_id: str,
    index: int,
    model: str,
    frozen: dict[str, Any],
    contract_sha256: str,
    stage_start_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    attempt_id = f"{run_id}:{index}:{model}"
    running_path = model_status_path(index, model, "running")
    running_payload = {
        "schema_version": "phase576_repeat_forward_model_status.v1",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_index": index,
        "model_order_index": index,
        "model": model,
        "status": "running",
        "started_at_utc": now(),
        "execution_contract_sha256": contract_sha256,
        "stage_start_sha256": stage_start_sha256,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "qualification_source_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_model_artifact_identity": frozen[
            "model_artifact_identities"
        ][model],
        "cuda_memory_before_load": cuda_memory_snapshot(),
        "formal_case_content_parsed": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
    }
    running_sha256 = atomic_create_json(running_path, running_payload)
    started = time.time()
    report: dict[str, Any] | None = None
    failure_components: list[dict[str, str]] = []
    adapter = None
    try:
        if running_payload["cuda_memory_before_load"]["total_allocated_bytes"] != 0:
            raise RuntimeError(f"{model}: nonzero CUDA allocation before load")
        adapter = load_model_adapter(model)
        report = collect_repeat_forward_report(adapter, model)
    except Exception as exc:
        failure_components.append({
            "stage": "qualification",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
    finally:
        try:
            release_model_adapter(adapter)
        except Exception as exc:
            failure_components.append({
                "stage": "release_model_adapter",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        adapter = None
        gc.collect()
    memory_after = cuda_memory_snapshot()
    if memory_after["total_allocated_bytes"] != 0:
        failure_components.append({
            "stage": "cuda_release_gate",
            "error_type": "CudaReleaseError",
            "error": (
                "nonzero CUDA allocation after release: "
                f"{memory_after['total_allocated_bytes']}"
            ),
        })

    elapsed = time.time() - started
    if failure_components or report is None:
        error_type = (
            failure_components[0]["error_type"]
            if len(failure_components) == 1
            else "CompositeEngineeringQualificationError"
        )
        error = " | ".join(
            f"{item['stage']}:{item['error_type']}:{item['error']}"
            for item in failure_components
        ) or "qualification returned no report"
        failed_path = model_status_path(index, model, "failed")
        failed_payload = {
            "schema_version": "phase576_repeat_forward_model_status.v1",
            "phase_id": protocol.PHASE,
            "qualification_kind": "repeat_forward_engineering_qualification",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "attempt_index": index,
            "model_order_index": index,
            "model": model,
            "status": "failed",
            "started_at_utc": running_payload["started_at_utc"],
            "failed_at_utc": now(),
            "elapsed_seconds": elapsed,
            "running_status_sha256": running_sha256,
            "execution_contract_sha256": contract_sha256,
            "error_type": error_type,
            "error": error,
            "failure_components": failure_components,
            "cleanup_completed": memory_after["total_allocated_bytes"] == 0,
            "cuda_memory_after_release": memory_after,
            "formal_case_content_parsed": False,
            "sealed_case_payload_read": False,
            "prior_sealed_files_read": False,
        }
        terminal_sha256 = atomic_create_json(failed_path, failed_payload)
        return ({
            "attempt_id": attempt_id,
            "attempt_index": index,
            "model": model,
            "status": "failed",
            "running_status_path": relative_path(running_path),
            "running_status_sha256": running_sha256,
            "terminal_status_path": relative_path(failed_path),
            "terminal_status_sha256": terminal_sha256,
            "error_type": error_type,
            "cleanup_completed": failed_payload["cleanup_completed"],
            "pytorch_cuda_allocated_after_release": memory_after[
                "total_allocated_bytes"
            ],
        }, None)

    report.update({
        "frozen_model_artifact_identity": frozen[
            "model_artifact_identities"
        ][model],
        "elapsed_seconds": elapsed,
        "cleanup_completed": True,
        "cuda_memory_after_release": memory_after,
        "pytorch_cuda_allocated_after_release": memory_after[
            "total_allocated_bytes"
        ],
    })
    complete_path = model_status_path(index, model, "complete")
    complete_payload = {
        "schema_version": "phase576_repeat_forward_model_status.v1",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_index": index,
        "model_order_index": index,
        "model": model,
        "status": "complete",
        "started_at_utc": running_payload["started_at_utc"],
        "completed_at_utc": now(),
        "elapsed_seconds": elapsed,
        "running_status_sha256": running_sha256,
        "execution_contract_sha256": contract_sha256,
        "cleanup_completed": True,
        "cuda_memory_after_release": memory_after,
        "pytorch_cuda_allocated_after_release": 0,
        "report": report,
        "report_sha256": json_sha256(report),
        "formal_case_content_parsed": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
    }
    terminal_sha256 = atomic_create_json(complete_path, complete_payload)
    return ({
        "attempt_id": attempt_id,
        "attempt_index": index,
        "model": model,
        "status": "complete",
        "running_status_path": relative_path(running_path),
        "running_status_sha256": running_sha256,
        "terminal_status_path": relative_path(complete_path),
        "terminal_status_sha256": terminal_sha256,
        "report_sha256": complete_payload["report_sha256"],
        "cleanup_completed": True,
        "pytorch_cuda_allocated_after_release": 0,
    }, report)


def clean_cuda_memory(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    devices = payload.get("devices")
    return (
        payload.get("cuda_available") is True
        and isinstance(devices, list)
        and payload.get("total_allocated_bytes") == 0
        and all(
            isinstance(row, dict)
            and type(row.get("device_index")) is int
            and row.get("allocated_bytes") == 0
            and type(row.get("reserved_bytes")) is int
            and row["reserved_bytes"] >= 0
            for row in devices
        )
    )


def validate_success_report(
    report: dict[str, Any], model: str, frozen: dict[str, Any]
) -> None:
    require_exact_keys(report, {
        "schema_version", "phase_id", "qualification_kind", "model",
        "passed", "checks", "detailed_checks", "layer_diagnostics",
        "logits_diagnostics", "layer_count", "hidden_state_count",
        "hidden_size", "sequence_length", "vocabulary_size",
        "greedy_generation_capsule", "rendered_prompt_sha256",
        "input_token_ids_sha256", "input_token_count",
        "loaded_model_identity", "formal_case_access",
        "formal_case_content_parsed", "sealed_split_read",
        "sealed_case_payload_read", "prior_sealed_files_read",
        "activation_persisted", "causal_intervention",
        "frozen_model_artifact_identity", "elapsed_seconds",
        "cleanup_completed", "cuda_memory_after_release",
        "pytorch_cuda_allocated_after_release",
    }, f"{model} engineering report")
    checks = report.get("checks")
    detailed = report.get("detailed_checks")
    layers = report.get("layer_diagnostics")
    logits = report.get("logits_diagnostics")
    generation = report.get("greedy_generation_capsule")
    loaded = report.get("loaded_model_identity")
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{model}: loaded model identity is invalid")
    quantization = loaded.get("loaded_quantization")
    if not isinstance(quantization, dict):
        raise RuntimeError(f"{model}: loaded quantization identity is invalid")
    scalar_checks = (
        report.get("schema_version")
        == "phase576_repeat_forward_model_report.v1",
        report.get("phase_id") == protocol.PHASE,
        report.get("qualification_kind")
        == "repeat_forward_engineering_qualification",
        report.get("model") == model,
        report.get("passed") is True,
        isinstance(checks, dict),
        set(checks or {}) == EXPECTED_COMPATIBILITY_CHECKS,
        all(value is True for value in (checks or {}).values()),
        isinstance(detailed, dict) and bool(detailed),
        all(value is True for value in (detailed or {}).values()),
        type(report.get("layer_count")) is int
        and report["layer_count"] > 0,
        report.get("hidden_state_count") == report.get("layer_count", -1) + 1,
        type(report.get("hidden_size")) is int and report["hidden_size"] > 0,
        type(report.get("sequence_length")) is int
        and report["sequence_length"] > 0,
        type(report.get("vocabulary_size")) is int
        and report["vocabulary_size"] > 0,
        type(report.get("input_token_count")) is int
        and report["input_token_count"] > 0,
        report.get("frozen_model_artifact_identity")
        == frozen["model_artifact_identities"][model],
        report.get("formal_case_access") is False,
        report.get("formal_case_content_parsed") is False,
        report.get("sealed_split_read") is False,
        report.get("sealed_case_payload_read") is False,
        report.get("prior_sealed_files_read") is False,
        report.get("activation_persisted") is False,
        report.get("causal_intervention") is False,
        report.get("cleanup_completed") is True,
        report.get("pytorch_cuda_allocated_after_release") == 0,
        clean_cuda_memory(report.get("cuda_memory_after_release")),
        isinstance(report.get("elapsed_seconds"), (int, float))
        and report["elapsed_seconds"] >= 0,
        loaded.get("cuda_only_no_cpu_or_disk_offload") is True,
        loaded.get("loaded_attn_implementation") == "sdpa",
        quantization.get("load_in_8bit") is True,
        quantization.get("floating_parameter_dtypes") == ["torch.bfloat16"],
    )
    if not all(scalar_checks):
        raise RuntimeError(f"{model}: engineering report scalar validation failed")

    if not isinstance(layers, list) or len(layers) != report["hidden_state_count"]:
        raise RuntimeError(f"{model}: engineering layer registry is invalid")
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise RuntimeError(f"{model}: layer {index} diagnostic is invalid")
        if not all((
            layer.get("layer_index") == index,
            layer.get("first_shape") == layer.get("second_shape"),
            isinstance(layer.get("first_shape"), list),
            layer.get("first_shape", [None])[-1] == report["hidden_size"],
            layer.get("first_dtype") == "torch.bfloat16",
            layer.get("second_dtype") == "torch.bfloat16",
            isinstance(layer.get("first_device"), str)
            and layer["first_device"].startswith("cuda"),
            layer.get("second_device") == layer.get("first_device"),
            layer.get("first_finite") is True,
            layer.get("second_finite") is True,
            layer.get("repeat_exact") is True,
            layer.get("first_batch_rows_identical") is True,
            layer.get("second_batch_rows_identical") is True,
        )):
            raise RuntimeError(f"{model}: layer {index} diagnostic failed")

    if not isinstance(logits, dict) or not all((
        logits.get("first_shape") == logits.get("second_shape"),
        logits.get("first_dtype") == logits.get("second_dtype"),
        isinstance(logits.get("first_device"), str)
        and logits["first_device"].startswith("cuda"),
        logits.get("second_device") == logits.get("first_device"),
        logits.get("first_finite") is True,
        logits.get("second_finite") is True,
        logits.get("repeat_exact") is True,
        logits.get("batch_rows_identical") is True,
    )):
        raise RuntimeError(f"{model}: logits diagnostic failed validation")

    if not isinstance(generation, dict) or not all((
        generation.get("max_new_tokens") == protocol.MAX_NEW_TOKENS,
        generation.get("do_sample") is False,
        generation.get("use_cache") is True,
        generation.get("return_dict_in_generate") is True,
        generation.get("output_logits") is True,
        generation.get("output_hidden_states") is True,
        isinstance(generation.get("repeat_checks"), dict),
        bool(generation.get("repeat_checks")),
        all(value is True for value in generation.get("repeat_checks", {}).values()),
    )):
        raise RuntimeError(f"{model}: generation capsule is invalid")
    for side_name in ("first", "second"):
        side = generation.get(side_name)
        if not isinstance(side, dict):
            raise RuntimeError(f"{model}: {side_name} generation capsule missing")
        side_checks = side.get("checks")
        rows = side.get("rows")
        steps = side.get("steps")
        if not all((
            isinstance(side_checks, dict) and bool(side_checks),
            all(value is True for value in (side_checks or {}).values()),
            isinstance(rows, list) and len(rows) == 2,
            isinstance(steps, list) and bool(steps),
        )):
            raise RuntimeError(f"{model}: {side_name} generation registry invalid")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or not all((
                row.get("row_index") == row_index,
                row.get("token_ids_in_vocabulary") is True,
                row.get("prompt_prefix_exact") is True,
                row.get("post_eos_tokens_are_absorbing") is True,
                row.get("eos_seen") is True
                or row.get("budget_terminated") is True,
            )):
                raise RuntimeError(
                    f"{model}: {side_name} generation row {row_index} invalid"
                )
        for step_index, step in enumerate(steps):
            step_layers = step.get("layers") if isinstance(step, dict) else None
            if not isinstance(step, dict) or not all((
                step.get("step_index") == step_index,
                step.get("all_layers_valid") is True,
                step.get("logits_finite") is True,
                step.get("logits_batch_rows_identical") is True,
                step.get("logits_contract_valid") is True,
                isinstance(step_layers, list)
                and len(step_layers) == report["hidden_state_count"],
            )):
                raise RuntimeError(
                    f"{model}: {side_name} generation step {step_index} invalid"
                )
            if not all(
                layer.get("finite") is True
                and layer.get("dtype") == "torch.bfloat16"
                and isinstance(layer.get("device"), str)
                and layer["device"].startswith("cuda")
                and layer.get("batch_rows_identical") is True
                for layer in step_layers
            ):
                raise RuntimeError(
                    f"{model}: {side_name} generation layer registry invalid"
                )


def validate_successful_execution_receipt() -> dict[str, Any]:
    """Reconstruct only from a fully validated immutable success receipt."""

    if not EXECUTION_RECEIPT_PATH.is_file() or EXECUTION_RECEIPT_PATH.is_symlink():
        raise RuntimeError("successful engineering execution receipt is missing")
    receipt = read_json(EXECUTION_RECEIPT_PATH)
    require_exact_keys(receipt, {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "stage_start_sha256", "created_at_utc", "terminal_status",
        "execution_passed", "required_model_order", "attempts",
        "attempted_models_in_order", "completed_models", "failed_models",
        "not_attempted_models", "fatal_error", "execution_contract",
        "execution_contract_sha256", "preflight_integrity",
        "postflight_integrity", "runtime_identity", "final_cuda_memory",
        "final_cuda_cleanup_pass", "qualification_publish_authorized",
        "qualification_published_at_receipt_commit", "qualification_path",
        "formal_case_access", "formal_case_content_parsed",
        "open_case_bytes_hashed_for_integrity", "sealed_split_read",
        "sealed_case_payload_read", "prior_sealed_files_read",
        "activation_persisted", "causal_intervention",
    }, "engineering execution receipt")
    expected_order = list(protocol.MODELS)
    contract = execution_contract()
    contract_sha256 = json_sha256(contract)
    receipt_checks = (
        receipt.get("schema_version")
        == "phase576_repeat_forward_execution_receipt.v1",
        receipt.get("phase_id") == protocol.PHASE,
        receipt.get("qualification_kind")
        == "repeat_forward_engineering_qualification",
        isinstance(receipt.get("run_id"), str) and bool(receipt["run_id"]),
        isinstance(receipt.get("created_at_utc"), str),
        receipt.get("terminal_status")
        == "models_complete_qualification_pending_publish",
        receipt.get("execution_passed") is True,
        receipt.get("required_model_order") == expected_order,
        receipt.get("attempted_models_in_order") == expected_order,
        receipt.get("completed_models") == expected_order,
        receipt.get("failed_models") == [],
        receipt.get("not_attempted_models") == [],
        receipt.get("fatal_error") is None,
        receipt.get("execution_contract") == contract,
        receipt.get("execution_contract_sha256") == contract_sha256,
        receipt.get("runtime_identity") == runtime_identity(),
        clean_cuda_memory(receipt.get("final_cuda_memory")),
        receipt.get("final_cuda_cleanup_pass") is True,
        receipt.get("qualification_publish_authorized") is True,
        receipt.get("qualification_published_at_receipt_commit") is False,
        receipt.get("qualification_path") == relative_path(QUALIFICATION_PATH),
        receipt.get("formal_case_access") is False,
        receipt.get("formal_case_content_parsed") is False,
        receipt.get("open_case_bytes_hashed_for_integrity") is True,
        receipt.get("sealed_split_read") is False,
        receipt.get("sealed_case_payload_read") is False,
        receipt.get("prior_sealed_files_read") is False,
        receipt.get("activation_persisted") is False,
        receipt.get("causal_intervention") is False,
    )
    if not all(receipt_checks):
        raise RuntimeError("engineering success receipt failed scalar validation")

    if not STAGE_START_PATH.is_file() or STAGE_START_PATH.is_symlink():
        raise RuntimeError("engineering stage-start artifact is missing")
    stage_start = read_json(STAGE_START_PATH)
    require_exact_keys(stage_start, {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "started_at_utc", "required_model_order", "execution_contract",
        "execution_contract_sha256", "qualification_source_sha256",
        "protocol_sha256_observed_at_start",
        "static_audit_sha256_observed_at_start",
        "sealed_commitment_sha256_observed_at_start",
        "freeze_commit_sha256_observed_at_start",
        "formal_case_content_parsed", "sealed_case_payload_read",
        "prior_sealed_files_read", "activation_persisted",
        "causal_intervention",
    }, "engineering stage start")
    if not all((
        stage_start.get("schema_version")
        == "phase576_repeat_forward_stage_start.v1",
        stage_start.get("phase_id") == protocol.PHASE,
        stage_start.get("qualification_kind")
        == "repeat_forward_engineering_qualification",
        stage_start.get("run_id") == receipt["run_id"],
        isinstance(stage_start.get("started_at_utc"), str),
        stage_start.get("required_model_order") == expected_order,
        stage_start.get("execution_contract") == contract,
        stage_start.get("execution_contract_sha256") == contract_sha256,
        stage_start.get("qualification_source_sha256")
        == sha256_file(Path(__file__).resolve()),
        stage_start.get("formal_case_content_parsed") is False,
        stage_start.get("sealed_case_payload_read") is False,
        stage_start.get("prior_sealed_files_read") is False,
        stage_start.get("activation_persisted") is False,
        stage_start.get("causal_intervention") is False,
        receipt.get("stage_start_sha256") == sha256_file(STAGE_START_PATH),
    )):
        raise RuntimeError("engineering stage-start validation failed")

    current = verify_integrity_before_or_after_models()
    frozen = current["frozen"]
    current_without_frozen = {
        key: value for key, value in current.items() if key != "frozen"
    }
    preflight = receipt.get("preflight_integrity")
    postflight = receipt.get("postflight_integrity")
    if not isinstance(preflight, dict) or not isinstance(postflight, dict):
        raise RuntimeError("engineering success receipt lacks integrity snapshots")
    if preflight != current_without_frozen or postflight != current_without_frozen:
        raise RuntimeError(
            "engineering receipt integrity snapshots do not match current frozen state"
        )
    if not isinstance(preflight.get("checks"), dict) or not all(
        value is True for value in preflight["checks"].values()
    ):
        raise RuntimeError("engineering preflight integrity checks are invalid")
    observed_start_hashes = {
        "protocol_sha256_observed_at_start": preflight["protocol_sha256"],
        "static_audit_sha256_observed_at_start": preflight["static_audit_sha256"],
        "sealed_commitment_sha256_observed_at_start": preflight[
            "sealed_commitment_sha256"
        ],
        "freeze_commit_sha256_observed_at_start": preflight[
            "freeze_commit_sha256"
        ],
    }
    if any(stage_start.get(key) != value for key, value in observed_start_hashes.items()):
        raise RuntimeError("engineering stage-start frozen hash chain is invalid")

    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(expected_order):
        raise RuntimeError("engineering receipt attempt registry is invalid")
    reports: list[dict[str, Any]] = []
    expected_files = {
        "stage_start.json",
        "execution_receipt.json",
    }
    for index, model in enumerate(expected_order):
        attempt = attempts[index]
        if not isinstance(attempt, dict):
            raise RuntimeError(f"{model}: engineering attempt is invalid")
        require_exact_keys(attempt, {
            "attempt_id", "attempt_index", "model", "status",
            "running_status_path", "running_status_sha256",
            "terminal_status_path", "terminal_status_sha256",
            "report_sha256", "cleanup_completed",
            "pytorch_cuda_allocated_after_release",
        }, f"{model} engineering attempt")
        running_path = model_status_path(index, model, "running")
        complete_path = model_status_path(index, model, "complete")
        expected_files.update({running_path.name, complete_path.name})
        if not running_path.is_file() or running_path.is_symlink():
            raise RuntimeError(f"{model}: running status is missing")
        if not complete_path.is_file() or complete_path.is_symlink():
            raise RuntimeError(f"{model}: complete status is missing")
        attempt_id = f"{receipt['run_id']}:{index}:{model}"
        if not all((
            attempt.get("attempt_id") == attempt_id,
            attempt.get("attempt_index") == index,
            attempt.get("model") == model,
            attempt.get("status") == "complete",
            attempt.get("running_status_path") == relative_path(running_path),
            attempt.get("running_status_sha256") == sha256_file(running_path),
            attempt.get("terminal_status_path") == relative_path(complete_path),
            attempt.get("terminal_status_sha256") == sha256_file(complete_path),
            attempt.get("cleanup_completed") is True,
            attempt.get("pytorch_cuda_allocated_after_release") == 0,
        )):
            raise RuntimeError(f"{model}: attempt hash chain is invalid")

        running = read_json(running_path)
        require_exact_keys(running, {
            "schema_version", "phase_id", "qualification_kind", "run_id",
            "attempt_id", "attempt_index", "model_order_index", "model",
            "status", "started_at_utc", "execution_contract_sha256",
            "stage_start_sha256", "protocol_sha256",
            "qualification_source_sha256", "frozen_model_artifact_identity",
            "cuda_memory_before_load", "formal_case_content_parsed",
            "sealed_case_payload_read", "prior_sealed_files_read",
        }, f"{model} running status")
        if not all((
            running.get("schema_version")
            == "phase576_repeat_forward_model_status.v1",
            running.get("phase_id") == protocol.PHASE,
            running.get("qualification_kind")
            == "repeat_forward_engineering_qualification",
            running.get("run_id") == receipt["run_id"],
            running.get("attempt_id") == attempt_id,
            running.get("attempt_index") == index,
            running.get("model_order_index") == index,
            running.get("model") == model,
            running.get("status") == "running",
            isinstance(running.get("started_at_utc"), str),
            running.get("execution_contract_sha256") == contract_sha256,
            running.get("stage_start_sha256") == receipt["stage_start_sha256"],
            running.get("protocol_sha256") == preflight["protocol_sha256"],
            running.get("qualification_source_sha256")
            == preflight["qualification_source_sha256"],
            running.get("frozen_model_artifact_identity")
            == frozen["model_artifact_identities"][model],
            clean_cuda_memory(running.get("cuda_memory_before_load")),
            running.get("formal_case_content_parsed") is False,
            running.get("sealed_case_payload_read") is False,
            running.get("prior_sealed_files_read") is False,
        )):
            raise RuntimeError(f"{model}: running status validation failed")

        complete = read_json(complete_path)
        require_exact_keys(complete, {
            "schema_version", "phase_id", "qualification_kind", "run_id",
            "attempt_id", "attempt_index", "model_order_index", "model",
            "status", "started_at_utc", "completed_at_utc",
            "elapsed_seconds", "running_status_sha256",
            "execution_contract_sha256", "cleanup_completed",
            "cuda_memory_after_release",
            "pytorch_cuda_allocated_after_release", "report",
            "report_sha256", "formal_case_content_parsed",
            "sealed_case_payload_read", "prior_sealed_files_read",
        }, f"{model} complete status")
        report = complete.get("report")
        if not isinstance(report, dict):
            raise RuntimeError(f"{model}: complete status lacks report")
        report_sha256 = json_sha256(report)
        if not all((
            complete.get("schema_version")
            == "phase576_repeat_forward_model_status.v1",
            complete.get("phase_id") == protocol.PHASE,
            complete.get("qualification_kind")
            == "repeat_forward_engineering_qualification",
            complete.get("run_id") == receipt["run_id"],
            complete.get("attempt_id") == attempt_id,
            complete.get("attempt_index") == index,
            complete.get("model_order_index") == index,
            complete.get("model") == model,
            complete.get("status") == "complete",
            complete.get("started_at_utc") == running["started_at_utc"],
            isinstance(complete.get("completed_at_utc"), str),
            isinstance(complete.get("elapsed_seconds"), (int, float))
            and complete["elapsed_seconds"] >= 0,
            complete.get("running_status_sha256")
            == attempt["running_status_sha256"],
            complete.get("execution_contract_sha256") == contract_sha256,
            complete.get("cleanup_completed") is True,
            clean_cuda_memory(complete.get("cuda_memory_after_release")),
            complete.get("pytorch_cuda_allocated_after_release") == 0,
            complete.get("report_sha256") == report_sha256,
            attempt.get("report_sha256") == report_sha256,
            complete.get("formal_case_content_parsed") is False,
            complete.get("sealed_case_payload_read") is False,
            complete.get("prior_sealed_files_read") is False,
        )):
            raise RuntimeError(f"{model}: complete status validation failed")
        validate_success_report(report, model, frozen)
        if not all((
            report.get("elapsed_seconds") == complete["elapsed_seconds"],
            report.get("cuda_memory_after_release")
            == complete["cuda_memory_after_release"],
        )):
            raise RuntimeError(f"{model}: report/status binding failed")
        reports.append(report)

    inventory = directory_hash_inventory(EXECUTION_DIR)
    observed_files = {
        row["path"] for row in inventory if row["kind"] == "file"
    }
    observed_directories = [
        row["path"] for row in inventory if row["kind"] == "directory"
    ]
    missing_files = expected_files - observed_files
    extra_files = observed_files - expected_files
    registered_receipt_publication_temps: list[str] = []
    for relative in sorted(extra_files):
        temporary = EXECUTION_DIR / relative
        if not all((
            "/" not in relative,
            is_atomic_temp_name(relative),
            temporary.is_file(),
            not temporary.is_symlink(),
            os.path.samefile(temporary, EXECUTION_RECEIPT_PATH),
            sha256_file(temporary) == sha256_file(EXECUTION_RECEIPT_PATH),
        )):
            raise RuntimeError(
                "engineering success execution contains an unregistered "
                f"artifact: {relative}"
            )
        registered_receipt_publication_temps.append(relative)
    if missing_files or observed_directories:
        raise RuntimeError(
            "engineering success execution contains unregistered artifacts: "
            f"registered_receipt_publication_temps="
            f"{registered_receipt_publication_temps}, "
            f"missing={sorted(missing_files)}, "
            f"directories={observed_directories}"
        )
    return {
        "receipt": receipt,
        "receipt_sha256": sha256_file(EXECUTION_RECEIPT_PATH),
        "frozen": frozen,
        "reports": reports,
    }


def build_qualification_payload(
    validated: dict[str, Any]
) -> dict[str, Any]:
    """The sole canonical producer for fresh and recovered qualification."""

    receipt = validated["receipt"]
    frozen = validated["frozen"]
    reports = validated["reports"]
    preflight = receipt["preflight_integrity"]
    postflight = receipt["postflight_integrity"]
    return {
        "schema_version": "phase576_engineering_qualification.v2",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "run_id": receipt["run_id"],
        "created_at_utc": now(),
        "terminal_status": "complete",
        "passed": True,
        "models_in_execution_order": list(protocol.MODELS),
        "attempts": receipt["attempts"],
        "qualified_models": receipt["completed_models"],
        "failed_models": [],
        "blocked_models": [],
        "reports": reports,
        "execution_contract": receipt["execution_contract"],
        "execution_contract_sha256": receipt["execution_contract_sha256"],
        "execution_receipt_path": relative_path(EXECUTION_RECEIPT_PATH),
        "execution_receipt_sha256": validated["receipt_sha256"],
        "published_from_pending_execution_receipt": True,
        "protocol_sha256": preflight["protocol_sha256"],
        "static_audit_sha256": preflight["static_audit_sha256"],
        "sealed_commitment_sha256": preflight["sealed_commitment_sha256"],
        "freeze_commit_sha256": preflight["freeze_commit_sha256"],
        "open_cases_sha256": preflight["open_cases_sha256"],
        "open_case_sha256_by_split": preflight["open_case_sha256_by_split"],
        "qualification_source_sha256": preflight[
            "qualification_source_sha256"
        ],
        "stage_source_seals": frozen["stage_source_seals"],
        "model_artifact_identities": frozen["model_artifact_identities"],
        "preflight_integrity_checks": preflight["checks"],
        "postflight_integrity_checks": postflight["checks"],
        "runtime_identity": receipt["runtime_identity"],
        "final_cuda_memory": receipt["final_cuda_memory"],
        "formal_case_access": False,
        "formal_case_content_parsed": False,
        "open_case_bytes_hashed_for_integrity": True,
        "sealed_split_read": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
        "activation_persisted": False,
        "causal_intervention": False,
    }


def recover_qualification_from_receipt() -> dict[str, Any]:
    validated = validate_successful_execution_receipt()
    payload = build_qualification_payload(validated)
    atomic_create_json(QUALIFICATION_PATH, payload)
    return payload


def execute_fresh_qualification() -> None:
    if QUALIFICATION_PATH.exists():
        raise RuntimeError(
            "refusing to overwrite Phase576 engineering qualification"
        )
    if not EXECUTION_DIR.is_dir() or any(EXECUTION_DIR.iterdir()):
        raise RuntimeError(
            "fresh engineering execution requires a new empty evidence directory"
        )

    run_id = str(uuid.uuid4())
    contract = execution_contract()
    contract_sha256 = json_sha256(contract)
    stage_start = {
        "schema_version": "phase576_repeat_forward_stage_start.v1",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "run_id": run_id,
        "started_at_utc": now(),
        "required_model_order": list(protocol.MODELS),
        "execution_contract": contract,
        "execution_contract_sha256": contract_sha256,
        "qualification_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256_observed_at_start": sha256_if_file(
            protocol.PROTOCOL_PATH
        ),
        "static_audit_sha256_observed_at_start": sha256_if_file(
            protocol.STATIC_AUDIT_PATH
        ),
        "sealed_commitment_sha256_observed_at_start": sha256_if_file(
            protocol.SEALED_COMMITMENT_PATH
        ),
        "freeze_commit_sha256_observed_at_start": sha256_if_file(
            protocol.FREEZE_COMMIT_PATH
        ),
        "formal_case_content_parsed": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
        "activation_persisted": False,
        "causal_intervention": False,
    }
    stage_start_sha256 = atomic_create_json(STAGE_START_PATH, stage_start)

    attempts: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    postflight: dict[str, Any] | None = None
    fatal_error: dict[str, str] | None = None
    try:
        preflight = verify_integrity_before_or_after_models()
        baseline = cuda_memory_snapshot()
        if not baseline["cuda_available"]:
            raise RuntimeError("CUDA is required for engineering qualification")
        if baseline["total_allocated_bytes"] != 0:
            raise RuntimeError(
                "nonzero PyTorch CUDA allocation at qualification baseline"
            )
        frozen = preflight["frozen"]
        for index, model in enumerate(protocol.MODELS):
            if attempts and not attempts[-1]["cleanup_completed"]:
                break
            attempt, report = execute_model(
                run_id,
                index,
                model,
                frozen,
                contract_sha256,
                stage_start_sha256,
            )
            attempts.append(attempt)
            if report is not None:
                reports.append(report)
        postflight = verify_integrity_before_or_after_models()
        if preflight["protocol_sha256"] != postflight["protocol_sha256"]:
            raise RuntimeError("protocol changed during engineering qualification")
        if preflight["static_audit_sha256"] != postflight[
            "static_audit_sha256"
        ]:
            raise RuntimeError("static audit changed during engineering qualification")
        if preflight["sealed_commitment_sha256"] != postflight[
            "sealed_commitment_sha256"
        ]:
            raise RuntimeError("sealed commitment changed during qualification")
        if preflight["freeze_commit_sha256"] != postflight[
            "freeze_commit_sha256"
        ]:
            raise RuntimeError("freeze commit changed during qualification")
        if preflight["qualification_source_sha256"] != postflight[
            "qualification_source_sha256"
        ]:
            raise RuntimeError("qualification source changed during execution")
    except Exception as exc:
        fatal_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    final_memory = cuda_memory_snapshot()
    expected_order = list(protocol.MODELS)
    completed_models = [
        item["model"] for item in attempts if item["status"] == "complete"
    ]
    failed_models = [
        item["model"] for item in attempts if item["status"] == "failed"
    ]
    attempted_models = [item["model"] for item in attempts]
    not_attempted_models = [
        model for model in expected_order if model not in attempted_models
    ]
    all_models_passed = (
        fatal_error is None
        and attempted_models == expected_order
        and completed_models == expected_order
        and not failed_models
        and not not_attempted_models
        and len(reports) == len(expected_order)
        and final_memory["total_allocated_bytes"] == 0
        and preflight is not None
        and postflight is not None
    )
    receipt = {
        "schema_version": "phase576_repeat_forward_execution_receipt.v1",
        "phase_id": protocol.PHASE,
        "qualification_kind": "repeat_forward_engineering_qualification",
        "run_id": run_id,
        "stage_start_sha256": stage_start_sha256,
        "created_at_utc": now(),
        "terminal_status": (
            "models_complete_qualification_pending_publish"
            if all_models_passed else "failed"
        ),
        "execution_passed": all_models_passed,
        "required_model_order": expected_order,
        "attempts": attempts,
        "attempted_models_in_order": attempted_models,
        "completed_models": completed_models,
        "failed_models": failed_models,
        "not_attempted_models": not_attempted_models,
        "fatal_error": fatal_error,
        "execution_contract": contract,
        "execution_contract_sha256": contract_sha256,
        "preflight_integrity": None if preflight is None else {
            key: value for key, value in preflight.items() if key != "frozen"
        },
        "postflight_integrity": None if postflight is None else {
            key: value for key, value in postflight.items() if key != "frozen"
        },
        "runtime_identity": runtime_identity(),
        "final_cuda_memory": final_memory,
        "final_cuda_cleanup_pass": final_memory["total_allocated_bytes"] == 0,
        "qualification_publish_authorized": all_models_passed,
        "qualification_published_at_receipt_commit": False,
        "qualification_path": relative_path(QUALIFICATION_PATH),
        "formal_case_access": False,
        "formal_case_content_parsed": False,
        "open_case_bytes_hashed_for_integrity": preflight is not None,
        "sealed_split_read": False,
        "sealed_case_payload_read": False,
        "prior_sealed_files_read": False,
        "activation_persisted": False,
        "causal_intervention": False,
    }
    receipt_sha256 = atomic_create_json(EXECUTION_RECEIPT_PATH, receipt)

    if not all_models_passed:
        print(json.dumps({
            "passed": False,
            "run_id": run_id,
            "completed_models": completed_models,
            "failed_models": failed_models,
            "not_attempted_models": not_attempted_models,
            "execution_receipt": relative_path(EXECUTION_RECEIPT_PATH),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        raise RuntimeError(
            "Phase576 repeat-forward engineering qualification failed; "
            "immutable receipt was published and silent rerun is forbidden"
        )

    validated = validate_successful_execution_receipt()
    payload = build_qualification_payload(validated)
    atomic_create_json(QUALIFICATION_PATH, payload)
    reports = validated["reports"]
    print(json.dumps({
        "passed": True,
        "run_id": run_id,
        "models": [
            {
                "model": report["model"],
                "layer_count": report["layer_count"],
                "hidden_size": report["hidden_size"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for report in reports
        ],
        "execution_receipt": relative_path(EXECUTION_RECEIPT_PATH),
        "formal_case_content_parsed": False,
        "sealed_case_payload_read": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))


def run_with_execution_lease() -> None:
    if QUALIFICATION_PATH.exists():
        raise RuntimeError(
            "refusing to overwrite Phase576 engineering qualification"
        )

    repair_pending_quarantine_receipts()
    if EXECUTION_DIR.exists():
        if EXECUTION_DIR.is_symlink() or not EXECUTION_DIR.is_dir():
            raise RuntimeError("invalid Phase576 engineering execution path")
        if EXECUTION_RECEIPT_PATH.exists():
            if (
                not EXECUTION_RECEIPT_PATH.is_file()
                or EXECUTION_RECEIPT_PATH.is_symlink()
            ):
                raise RuntimeError(
                    "invalid Phase576 engineering execution receipt path"
                )
            receipt = read_json(EXECUTION_RECEIPT_PATH)
            terminal_status = receipt.get("terminal_status")
            if terminal_status == "failed":
                raise RuntimeError(
                    "prior Phase576 engineering execution has a terminal "
                    "failure receipt; rerun is forbidden"
                )
            if terminal_status != "models_complete_qualification_pending_publish":
                raise RuntimeError(
                    "prior Phase576 engineering execution receipt has an "
                    f"unsupported terminal state: {terminal_status!r}"
                )
            payload = recover_qualification_from_receipt()
            print(json.dumps({
                "passed": True,
                "run_id": payload["run_id"],
                "publication_recovered": True,
                "models": [
                    {
                        "model": report["model"],
                        "layer_count": report["layer_count"],
                        "hidden_size": report["hidden_size"],
                        "elapsed_seconds": report["elapsed_seconds"],
                    }
                    for report in payload["reports"]
                ],
                "execution_receipt": relative_path(EXECUTION_RECEIPT_PATH),
                "formal_case_content_parsed": False,
                "sealed_case_payload_read": False,
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return
        quarantine_incomplete_execution()

    EXECUTION_DIR.mkdir(parents=True, exist_ok=False)
    execute_fresh_qualification()


def main() -> None:
    lease = acquire_execution_lease()
    try:
        run_with_execution_lease()
    finally:
        release_execution_lease(lease)


if __name__ == "__main__":
    main()
