#!/usr/bin/env python3
"""Resumable one-model worker for sealed Phase 983 generation.

The worker records trajectories and externally parsed checkpoints only.  It
cannot import the scientific gate or compute a PASS/FAIL decision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any


GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402
import phase983_cross_model_engine as engine  # noqa: E402


ROW_KEYS = frozenset({
    "schema_version", "phase", "experiment", "protocol_sha256",
    "admission_sha256", "manifest_sha256", "model_key", "id",
    "semantic_id", "seed_key", "task", "difficulty", "gold_label",
    "swap_variant", "arm", "arm_spec", "stream", "pair_id", "pair_seed",
    "batch_index", "effective_user_prompt", "rendered_prefix_sha256",
    "input_ids", "prompt_len", "generated_ids", "generated_plain",
    "first_eos_token_id", "first_eos_absorbing", "checkpoints",
    "decision_terminal_state", "max_new_tokens", "sampling",
    "compact_active_rows", "private_generator_per_row",
    "same_pair_seed_across_arms", "same_pair_seed_across_option_swap_twins",
    "generation_performed", "decision_computed", "holdout", "holdout_loaded",
    "mechanism", "mechanism_authorized", "row_sha256",
})

MANIFEST_KEYS = frozenset({
    "schema_version", "phase", "experiment", "model_key", "model_order_index",
    "protocol_sha256", "protocol_file_sha256", "admission_sha256",
    "admission_file_sha256", "qualification_sha256", "dataset_file_sha256",
    "dataset_audit_content_sha256", "model_artifact_identity_sha256",
    "loaded_model_identity", "loaded_tokenizer_identity", "runtime_versions",
    "eos_token_ids", "pad_token_id", "arms", "streams", "sampling",
    "quantization", "batch_size", "checkpoints", "max_new_tokens",
    "expected_rows", "dataset_namespace", "engine", "script_seals",
    "dependency_seals", "creation_state", "holdout", "holdout_loaded",
    "mechanism", "mechanism_authorized", "manifest_sha256", "created_at_utc",
})

STATUS_KEYS = frozenset({
    "schema_version", "phase", "experiment", "model_key", "protocol_sha256",
    "admission_sha256", "manifest_sha256", "completed_rows", "expected_rows",
    "cell_counts", "complete", "elapsed_seconds_current_process",
    "rows_file_sha256", "rows_file_bytes", "rows_file_line_count",
    "rows_file_terminal_newline", "generation_performed",
    "model_weights_loaded", "gpu_used", "decision_computed", "holdout",
    "mechanism", "status_sha256", "updated_at_utc",
})


def raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows_file_identity(path: Path, require_terminal_newline: bool) -> dict[str, Any]:
    """Hash a JSONL artifact once while also authenticating its byte/line shape."""
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    final_byte = b""
    if path.exists():
        core.require(path.is_file(), f"rows artifact is not a file: {path}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
                line_count += block.count(b"\n")
                final_byte = block[-1:]
    terminal_newline = bool(byte_count and final_byte == b"\n")
    if require_terminal_newline:
        core.require(terminal_newline, f"completed rows file lacks terminal newline: {path}")
    return {
        "rows_file_sha256": digest.hexdigest(),
        "rows_file_bytes": byte_count,
        "rows_file_line_count": line_count,
        "rows_file_terminal_newline": terminal_newline,
    }


def build_rows_tracker(path: Path) -> dict[str, Any]:
    """Build one resumable-process in-memory SHA state from the current JSONL."""
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    final_byte = b""
    if path.exists():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
                line_count += block.count(b"\n")
                final_byte = block[-1:]
    return {
        "digest": digest,
        "bytes": byte_count,
        "line_count": line_count,
        "terminal_newline": bool(byte_count and final_byte == b"\n"),
    }


def tracker_identity(tracker: dict[str, Any]) -> dict[str, Any]:
    digest = tracker.get("digest")
    core.require(hasattr(digest, "copy"), "rows SHA tracker is invalid")
    return {
        "rows_file_sha256": digest.copy().hexdigest(),
        "rows_file_bytes": int(tracker["bytes"]),
        "rows_file_line_count": int(tracker["line_count"]),
        "rows_file_terminal_newline": bool(tracker["terminal_newline"]),
    }


def update_rows_tracker(tracker: dict[str, Any], row: dict[str, Any]) -> None:
    payload = (core.canonical_json(row) + "\n").encode("utf-8")
    tracker["digest"].update(payload)
    tracker["bytes"] += len(payload)
    tracker["line_count"] += 1
    tracker["terminal_newline"] = True


def _windows_process_probe(pid: int) -> tuple[str, str | None]:
    """Return (alive/dead/unknown, creation token) without os.kill on Windows."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return ("dead", None) if error == error_invalid_parameter else (
            "unknown", f"windows-error:{error}")
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_time),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            return "unknown", f"windows-error:{ctypes.get_last_error()}"
        value = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        return "alive", f"windows-filetime:{value}"
    finally:
        kernel32.CloseHandle(handle)


def process_probe(pid: int) -> tuple[str, str | None]:
    """Probe liveness and a PID-reuse-resistant process-start identity."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return "dead", None
    if os.name == "nt":
        return _windows_process_probe(pid)
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        text = proc_stat.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "dead", None
    except OSError as exc:
        return "unknown", f"proc-error:{type(exc).__name__}"
    try:
        fields = text[text.rfind(")") + 2:].split()
        return "alive", f"proc-start-ticks:{fields[19]}"
    except (IndexError, ValueError):
        return "unknown", "malformed-proc-stat"


def _lock_document(model_key: str, protocol_sha256: str) -> dict[str, Any]:
    state, token = process_probe(os.getpid())
    core.require(state == "alive" and token is not None,
                 "cannot authenticate current runner process")
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "kind": "phase983_model_runner",
        "pid": os.getpid(),
        "process_start_token": token,
        "model_key": model_key,
        "protocol_sha256": protocol_sha256,
    }
    return {**payload, "lock_sha256": core.sha256_json(payload)}


def _acquire_recovery_guard(path: Path) -> int:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError(f"lock recovery already active: {path}") from exc


def _release_recovery_guard(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def _recover_stale_lock(path: Path, model_key: str, protocol_sha256: str) -> Path:
    """Atomically archive a proved-stale lock under a per-lock recovery guard."""
    core.require(len(protocol_sha256) == 64,
                 "runner stale-lock protocol hash invalid")
    guard = path.with_name(f"{path.name}.recovery")
    guard_descriptor = _acquire_recovery_guard(guard)
    try:
        document = core.load_json(path, f"{model_key} runner lock")
        supplied = document.get("lock_sha256")
        core.require(supplied == core.sha256_json(
            core.without_fields(document, "lock_sha256")),
            f"{model_key} runner lock self-hash invalid")
        core.require(document.get("kind") == "phase983_model_runner"
                     and document.get("model_key") == model_key
                     and isinstance(document.get("protocol_sha256"), str)
                     and len(document["protocol_sha256"]) == 64
                     and isinstance(document.get("process_start_token"), str)
                     and bool(document["process_start_token"]),
                     f"{model_key} runner lock schema invalid")
        pid = document.get("pid")
        core.require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
                     f"{model_key} runner lock PID invalid")
        state, observed_token = process_probe(pid)
        core.require(state != "unknown",
                     f"cannot prove whether {model_key} runner PID {pid} is stale")
        if state == "alive" and observed_token == document.get("process_start_token"):
            raise RuntimeError(
                f"active {model_key} runner lock exists for PID {pid}")
        # A dead PID or a live PID with a different creation token proves that
        # the lock owner no longer exists.  Preserve the lock as audit evidence.
        stale = path.with_name(
            f"{path.name}.stale.{pid}.{os.getpid()}.{time.time_ns()}.json")
        os.replace(path, stale)
        return stale
    finally:
        _release_recovery_guard(guard_descriptor)


def authenticate(model_key: str) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    core.require(model_key in core.MODEL_ORDER, "model outside formal registry")
    protocol = core.load_json(core.PROTOCOL_PATH, "Phase983 protocol")
    core.verify_self_hash(protocol, "protocol_sha256", "created_at_utc",
                          "Phase983 protocol")
    core.require(protocol.get("phase") == core.PHASE
                 and protocol.get("experiment") == core.EXPERIMENT
                 and protocol.get("model_order") == list(core.MODEL_ORDER)
                 and protocol.get("arms") == core.ARMS
                 and protocol.get("sampling") == core.SAMPLING
                 and protocol.get("quantization") == core.QUANTIZATION
                 and protocol.get("unique_decision_checkpoint")
                 == core.DECISION_CHECKPOINT
                 and protocol.get("expected_rows_per_model")
                 == core.EXPECTED_ROWS_PER_MODEL,
                 "protocol runtime contract changed")
    core.verify_file_seals(protocol.get("script_seals"), core.SCRIPT_PATHS,
                           "Phase983 script")
    core.verify_file_seals(protocol.get("dependency_seals"), core.DEPENDENCY_PATHS,
                           "Phase983 dependency")

    admission = core.load_json(core.ADMISSION_PATH, "Phase983 admission")
    core.verify_self_hash(admission, "admission_sha256", "created_at_utc",
                          "Phase983 admission")
    scope = admission.get("authorization_scope")
    core.require(admission.get("admitted") is True
                 and admission.get("gpu_authorized") is True
                 and admission.get("protocol_sha256") == protocol["protocol_sha256"]
                 and isinstance(scope, dict)
                 and scope.get("models") == list(core.MODEL_ORDER)
                 and scope.get("strict_order") == list(core.MODEL_ORDER)
                 and scope.get("arms") == core.ARMS
                 and scope.get("sampling") == core.SAMPLING
                 and scope.get("quantization") == core.QUANTIZATION
                 and scope.get("expected_rows_per_model")
                 == core.EXPECTED_ROWS_PER_MODEL
                 and scope.get("external_generation_only") is True,
                 "formal admission scope changed")
    core.require(admission.get("holdout") is False
                 and admission.get("mechanism") is False,
                 "admission opened holdout/mechanism")

    qualification = core.load_json(
        core.QUALIFICATION_PATH, "engineering qualification")
    core.verify_self_hash(qualification, "qualification_sha256", "created_at_utc",
                          "engineering qualification")
    core.require(qualification.get("qualification_passed") is True
                 and qualification.get("protocol_sha256")
                 == protocol["protocol_sha256"]
                 and admission.get("qualification_sha256")
                 == qualification["qualification_sha256"],
                 "qualification lineage changed")

    dataset = core.load_json(core.DATASET_PATH, "Phase983 dataset")
    dataset_audit = core.load_json(core.DATASET_AUDIT_PATH, "Phase983 dataset audit")
    core.require(core.sha256_file(core.DATASET_PATH)
                 == protocol["dataset"]["dataset_file_sha256"]
                 and core.sha256_file(core.DATASET_AUDIT_PATH)
                 == protocol["dataset"]["dataset_audit_file_sha256"]
                 and core.sha256_json(dataset)
                 == protocol["dataset"]["dataset_content_sha256"]
                 and core.sha256_json(dataset_audit)
                 == protocol["dataset"]["dataset_audit_content_sha256"],
                 "fresh dataset artifacts changed")
    items = dataset.get("items")
    core.require(isinstance(items, list) and len(items) == core.ITEM_COUNT,
                 "dataset denominator changed")

    identity = protocol["model_artifact_identities"][model_key]
    root = core.ROOT / identity["relative_path"]
    core.require(root.resolve() == (core.ROOT / core.MODEL_PATHS[model_key]).resolve(),
                 "model root changed")
    for name, seal in identity["files"].items():
        path = root / name
        core.require(path.is_file() and path.stat().st_size == seal["bytes"]
                     and core.sha256_file(path) == seal["sha256"],
                     f"model artifact changed: {model_key}/{name}")
    identity_payload = core.without_fields(identity, "identity_sha256")
    core.require(identity["identity_sha256"] == core.sha256_json(identity_payload),
                 "model artifact identity hash invalid")

    previous = core.MODEL_ORDER[:core.MODEL_ORDER.index(model_key)]
    for prior in previous:
        verify_complete_model_output(prior, protocol, admission, items)
    later = core.MODEL_ORDER[core.MODEL_ORDER.index(model_key) + 1:]
    if any(core.manifest_path(value).exists()
           or core.rows_path(value).exists()
           or core.status_path(value).exists() for value in later):
        verify_complete_model_output(model_key, protocol, admission, items)
    return protocol, admission, dataset_audit, items


def runtime_versions() -> dict[str, Any]:
    import bitsandbytes
    import torch
    import transformers
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "platform": platform.platform(),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "cuda_device_index": device_index,
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [int(properties.major), int(properties.minor)],
        "cuda_total_memory_bytes": int(properties.total_memory),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_memory_efficient_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def assistant_prefill_for_user(tok: Any, user: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": user}]
    without_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    with_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    core.require(isinstance(without_generation, str)
                 and isinstance(with_generation, str)
                 and with_generation.startswith(without_generation),
                 "runtime native generation render is not a suffix extension")
    base_ids = list(tok(
        without_generation, add_special_tokens=False,
        return_attention_mask=False).input_ids)
    full_ids = list(tok(
        with_generation, add_special_tokens=False,
        return_attention_mask=False).input_ids)
    core.require(full_ids[:len(base_ids)] == base_ids
                 and len(full_ids) > len(base_ids),
                 "runtime native assistant prefill token suffix changed")
    text = with_generation[len(without_generation):]
    input_ids = [int(value) for value in full_ids[len(base_ids):]]
    core.require(text and input_ids, "runtime native assistant prefill is empty")
    return text, input_ids


def native_generation_prefill(tok: Any) -> dict[str, Any]:
    probe = "PHASE983_NATIVE_GENERATION_PREFILL_PROBE"
    messages = [{"role": "user", "content": probe}]
    without_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    with_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    core.require(isinstance(without_generation, str)
                 and isinstance(with_generation, str)
                 and with_generation.startswith(without_generation),
                 "runtime native generation render is not a suffix extension")
    base_ids = list(tok(
        without_generation, add_special_tokens=False,
        return_attention_mask=False).input_ids)
    full_ids = list(tok(
        with_generation, add_special_tokens=False,
        return_attention_mask=False).input_ids)
    core.require(full_ids[:len(base_ids)] == base_ids
                 and len(full_ids) > len(base_ids),
                 "runtime native generation prefill token suffix changed")
    text = with_generation[len(without_generation):]
    input_ids = [int(value) for value in full_ids[len(base_ids):]]
    isolated_ids = [int(value) for value in tok(
        text, add_special_tokens=False,
        return_attention_mask=False).input_ids]
    core.require(text and input_ids == isolated_ids and input_ids,
                 "runtime native generation prefill is not isolated")
    return {
        "probe_text": probe,
        "without_generation_prompt_sha256": raw_sha256(without_generation),
        "with_generation_prompt_sha256": raw_sha256(with_generation),
        "assistant_prefill_text": text,
        "assistant_prefill_text_sha256": raw_sha256(text),
        "assistant_prefill_token_ids": input_ids,
        "assistant_prefill_token_ids_sha256": core.sha256_json(input_ids),
    }


def runtime_tokenizer_identity(
    tok: Any, probe_item: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    all_special_ids = sorted({
        int(value) for value in getattr(tok, "all_special_ids", [])
        if isinstance(value, int) and not isinstance(value, bool)
    })
    eos = sorted(int(value) for value in frozen["effective_eos_token_ids"])
    template = str(getattr(tok, "chat_template", ""))
    user_a, rendered_a, ids_a = core.render_prefix(tok, probe_item, core.ARM_A)
    user_b, rendered_b, ids_b = core.render_prefix(tok, probe_item, core.ARM_B)
    native_prefill = native_generation_prefill(tok)
    prefill_a = assistant_prefill_for_user(tok, user_a)
    prefill_b = assistant_prefill_for_user(tok, user_b)
    core.require(prefill_a == prefill_b
                 and prefill_a[0] == native_prefill["assistant_prefill_text"]
                 and prefill_a[1] == native_prefill["assistant_prefill_token_ids"],
                 "runtime native assistant prefill differs across arms")
    identity = {
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "tokenizer_eos_token_id": int(tok.eos_token_id),
        "effective_pad_token_id": int(tok.pad_token_id),
        "effective_eos_token_ids": eos,
        "all_special_ids": all_special_ids,
        "unexpected_special_token_ids": sorted(set(all_special_ids) - set(eos)),
        "chat_template_sha256": raw_sha256(template),
        "native_generation_prefill": native_prefill,
        "probe": {
            "item_id": str(probe_item["id"]),
            "arm_A_prefix_sha256": raw_sha256(rendered_a),
            "arm_B_prefix_sha256": raw_sha256(rendered_b),
            "arm_A_input_ids_sha256": core.sha256_json(ids_a),
            "arm_B_input_ids_sha256": core.sha256_json(ids_b),
            "arm_A_prompt_tokens": len(ids_a),
            "arm_B_prompt_tokens": len(ids_b),
        },
        "native_thinking_switch_used": False,
    }
    frozen_subset = {key: frozen[key] for key in identity}
    core.require(identity == frozen_subset,
                 "runtime tokenizer/template/special-token identity changed")
    return identity


def make_manifest(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
    dataset_audit: dict[str, Any], probe_item: dict[str, Any],
    adapter: engine.ModelAdapter,
) -> dict[str, Any]:
    eos = adapter.eos_identity["effective_eos_token_ids"]
    frozen_tokenizer = protocol["tokenizer_adapters"][model_key]
    core.require(eos == frozen_tokenizer["effective_eos_token_ids"],
                 "loaded EOS union differs from protocol")
    core.require(adapter.pad_token_id == frozen_tokenizer["effective_pad_token_id"],
                 "loaded PAD differs from protocol")
    loaded_quant = adapter.identity.get("loaded_quantization", {})
    core.require(loaded_quant.get("load_in_8bit") is True
                 and loaded_quant.get("llm_int8_enable_fp32_cpu_offload") is False
                 and adapter.identity.get("loaded_attn_implementation") == "sdpa",
                 "loaded precision/attention policy changed")
    tokenizer_identity = runtime_tokenizer_identity(
        adapter.tokenizer, probe_item, frozen_tokenizer,
    )
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "model_key": model_key,
        "model_order_index": core.MODEL_ORDER.index(model_key),
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
        "admission_sha256": admission["admission_sha256"],
        "admission_file_sha256": core.sha256_file(core.ADMISSION_PATH),
        "qualification_sha256": admission["qualification_sha256"],
        "dataset_file_sha256": core.sha256_file(core.DATASET_PATH),
        "dataset_audit_content_sha256": core.sha256_json(dataset_audit),
        "model_artifact_identity_sha256": protocol[
            "model_artifact_identities"][model_key]["identity_sha256"],
        "loaded_model_identity": adapter.identity,
        "loaded_tokenizer_identity": tokenizer_identity,
        "runtime_versions": runtime_versions(),
        "eos_token_ids": eos,
        "pad_token_id": adapter.pad_token_id,
        "arms": core.ARMS,
        "streams": list(core.STREAMS),
        "sampling": core.SAMPLING,
        "quantization": core.QUANTIZATION,
        "batch_size": core.BATCH_SIZE,
        "checkpoints": list(core.CHECKPOINTS),
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
        "dataset_namespace": protocol["protocol_sha256"],
        "engine": {
            "compact_active_rows": True,
            "dynamic_cache_batch_select_indices": True,
            "dense_reference_forbidden_in_formal_run": True,
            "private_generator_per_row": True,
            "same_item_stream_seed_across_arms": True,
            "same_semantic_twin_seed_across_option_surfaces": True,
            "two_by_two_arm_option_crn_block": True,
            "arm_excluded_from_seed": True,
            "swap_side_excluded_from_seed": True,
            "model_namespace_in_seed": True,
            "first_eos_absorbing": True,
            "one_longest_rollout_for_checkpoints": True,
        },
        "script_seals": protocol["script_seals"],
        "dependency_seals": protocol["dependency_seals"],
        "creation_state": {
            "model_weights_loaded": True,
            "gpu_used": True,
            "generation_performed": False,
            "decision_computed": False,
        },
        "holdout": False,
        "holdout_loaded": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    document = {
        **payload,
        "manifest_sha256": core.sha256_json(payload),
        "created_at_utc": core.utc_now(),
    }
    core.require(set(document) == MANIFEST_KEYS, "manifest schema construction changed")
    return document


def install_manifest(document: dict[str, Any]) -> dict[str, Any]:
    model_key = str(document["model_key"])
    core.require(set(document) == MANIFEST_KEYS,
                 f"{model_key} manifest has unknown/missing keys")
    core.verify_self_hash(document, "manifest_sha256", "created_at_utc",
                          f"{model_key} manifest")
    path = core.manifest_path(model_key)
    if path.exists():
        existing = core.load_json(path, f"existing {model_key} manifest")
        core.require(set(existing) == MANIFEST_KEYS,
                     f"existing {model_key} manifest schema changed")
        core.verify_self_hash(existing, "manifest_sha256", "created_at_utc",
                              f"existing {model_key} manifest")
        core.require(existing["manifest_sha256"] == document["manifest_sha256"],
                     f"existing {model_key} manifest differs")
        return existing
    core.atomic_write_json(path, document)
    installed = core.load_json(path, f"installed {model_key} manifest")
    core.require(set(installed) == MANIFEST_KEYS,
                 f"installed {model_key} manifest schema changed")
    core.verify_self_hash(installed, "manifest_sha256", "created_at_utc",
                          f"installed {model_key} manifest")
    core.require(installed == document,
                 f"installed {model_key} manifest changed in serialization")
    return installed


def build_row(
    manifest: dict[str, Any], adapter: engine.ModelAdapter,
    item: dict[str, Any], arm: str, stream: int, batch_index: int,
    sampled: engine.SampledRow,
) -> dict[str, Any]:
    core.require(sampled.item_id == str(item["id"])
                 and sampled.arm == arm and sampled.stream == stream
                 and sampled.model_key == manifest["model_key"],
                 "sampled row identity changed")
    seed_key = str(item["seed_key"])
    core.require(sampled.seed_key == seed_key,
                 "sampled row semantic-twin seed key changed")
    user, rendered, expected_ids = core.render_prefix(adapter.tokenizer, item, arm)
    core.require(list(sampled.input_ids) == expected_ids
                 and sampled.rendered_prefix_sha256 == raw_sha256(rendered),
                 "engine/native prefix differs from frozen core rendering")
    expected_seed = engine.stable_pair_seed(
        manifest["dataset_namespace"], manifest["model_key"], seed_key,
        stream, arm,
    )
    core.require(sampled.pair_seed == expected_seed, "pair seed changed")
    generated = list(sampled.generated_ids)
    checkpoints = core.analyze_checkpoints(
        adapter.tokenizer, item, generated, manifest["eos_token_ids"],
    )
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "protocol_sha256": manifest["protocol_sha256"],
        "admission_sha256": manifest["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "model_key": manifest["model_key"],
        "id": str(item["id"]),
        "semantic_id": str(item["semantic_id"]),
        "seed_key": seed_key,
        "task": str(item["task"]),
        "difficulty": str(item["difficulty"]),
        "gold_label": str(item["answer"]),
        "swap_variant": str(item["swap_side"]),
        "arm": arm,
        "arm_spec": core.ARMS[arm],
        "stream": stream,
        "pair_id": core.pair_id(manifest["model_key"], str(item["id"]), stream),
        "pair_seed": sampled.pair_seed,
        "batch_index": batch_index,
        "effective_user_prompt": user,
        "rendered_prefix_sha256": sampled.rendered_prefix_sha256,
        "input_ids": list(sampled.input_ids),
        "prompt_len": len(sampled.input_ids),
        "generated_ids": generated,
        "generated_plain": adapter.tokenizer.decode(
            generated, skip_special_tokens=False).strip(),
        "first_eos_token_id": sampled.first_eos_token_id,
        "first_eos_absorbing": sampled.first_eos_absorbing,
        "checkpoints": checkpoints,
        "decision_terminal_state": checkpoints[str(core.DECISION_CHECKPOINT)][
            "terminal_state"],
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "sampling": core.SAMPLING,
        "compact_active_rows": True,
        "private_generator_per_row": True,
        "same_pair_seed_across_arms": True,
        "same_pair_seed_across_option_swap_twins": True,
        "generation_performed": True,
        "decision_computed": False,
        "holdout": False,
        "holdout_loaded": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    document = {**payload, "row_sha256": core.sha256_json(payload)}
    core.require(set(document) == ROW_KEYS, "row schema construction changed")
    return document


def validate_row(
    row: dict[str, Any], manifest: dict[str, Any], tokenizer: Any,
    item_by_id: dict[str, dict[str, Any]], positions: dict[tuple[str, str, int], int],
) -> None:
    core.require(set(row) == ROW_KEYS, "row has unknown/missing keys")
    core.require(row.get("row_sha256")
                 == core.sha256_json(core.without_fields(row, "row_sha256")),
                 "row self-hash invalid")
    key = core.row_key(row)
    core.require(key in positions, f"row outside canonical grid: {key}")
    item = item_by_id[key[0]]
    arm, stream = key[1], key[2]
    user, rendered, input_ids = core.render_prefix(tokenizer, item, arm)
    generated = row.get("generated_ids")
    core.require(isinstance(generated, list) and generated
                 and all(isinstance(value, int) and not isinstance(value, bool)
                         for value in generated), "row generated IDs invalid")
    expected_checkpoints = core.analyze_checkpoints(
        tokenizer, item, generated, manifest["eos_token_ids"],
    )
    expected_seed = engine.stable_pair_seed(
        manifest["dataset_namespace"], manifest["model_key"],
        str(item["seed_key"]), stream, arm,
    )
    final = expected_checkpoints[str(core.DECISION_CHECKPOINT)]
    core.require(
        row.get("phase") == core.PHASE
        and row.get("protocol_sha256") == manifest["protocol_sha256"]
        and row.get("admission_sha256") == manifest["admission_sha256"]
        and row.get("manifest_sha256") == manifest["manifest_sha256"]
        and row.get("model_key") == manifest["model_key"]
        and row.get("semantic_id") == str(item["semantic_id"])
        and row.get("seed_key") == str(item["seed_key"])
        and row.get("task") == str(item["task"])
        and row.get("difficulty") == str(item["difficulty"])
        and row.get("gold_label") == str(item["answer"])
        and row.get("swap_variant") == str(item["swap_side"])
        and row.get("arm_spec") == core.ARMS[arm]
        and row.get("pair_seed") == expected_seed
        and row.get("pair_id") == core.pair_id(manifest["model_key"], key[0], stream)
        and row.get("batch_index") == positions[key] // core.BATCH_SIZE + 1
        and row.get("effective_user_prompt") == user
        and row.get("rendered_prefix_sha256") == raw_sha256(rendered)
        and row.get("input_ids") == input_ids
        and row.get("generated_plain")
        == tokenizer.decode(generated, skip_special_tokens=False).strip()
        and row.get("checkpoints") == expected_checkpoints
        and row.get("decision_terminal_state") == final["terminal_state"]
        and row.get("sampling") == core.SAMPLING
        and row.get("max_new_tokens") == core.MAX_NEW_TOKENS
        and row.get("compact_active_rows") is True
        and row.get("private_generator_per_row") is True
        and row.get("same_pair_seed_across_arms") is True
        and row.get("same_pair_seed_across_option_swap_twins") is True
        and row.get("generation_performed") is True
        and row.get("decision_computed") is False
        and row.get("holdout") is False
        and row.get("mechanism") is False,
        f"row lineage/content changed: {key}",
    )
    eos_positions = [
        index for index, value in enumerate(generated)
        if value in set(manifest["eos_token_ids"])
    ]
    core.require((eos_positions == [len(generated) - 1])
                 or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
                 f"row termination invalid: {key}")


def repair_truncated_tail(path: Path) -> None:
    if not path.exists():
        return
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return
    boundary = payload.rfind(b"\n")
    prefix = payload[:boundary + 1] if boundary >= 0 else b""
    tail = payload[boundary + 1:]
    try:
        value = json.loads(
            tail.decode("utf-8"), object_pairs_hook=core._pairs_no_duplicates,
            parse_constant=core._reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        with path.open("r+b") as handle:
            handle.truncate(len(prefix))
            handle.flush()
            os.fsync(handle.fileno())
        return
    core.require(isinstance(value, dict)
                 and value.get("row_sha256")
                 == core.sha256_json(core.without_fields(value, "row_sha256")),
                 "complete no-newline tail lacks a valid row self-hash")
    core.require(tail == core.canonical_json(value).encode("utf-8"),
                 "complete no-newline tail is not canonical JSONL")
    # append_jsonl writes canonical JSON plus exactly one newline.  Therefore a
    # strict, canonical, self-hashed final object proves a one-byte torn tail;
    # adding that byte is lossless and the full row is validated immediately by
    # load_rows below.
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(
    manifest: dict[str, Any], tokenizer: Any, items: list[dict[str, Any]],
    positions: dict[tuple[str, str, int], int],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = core.rows_path(manifest["model_key"])
    repair_truncated_tail(path)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return output
    item_by_id = {str(item["id"]): item for item in items}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(
                    line, object_pairs_hook=core._pairs_no_duplicates,
                    parse_constant=core._reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid row JSON at line {line_number}") from exc
            core.require(isinstance(row, dict), "row is not an object")
            validate_row(row, manifest, tokenizer, item_by_id, positions)
            key = core.row_key(row)
            core.require(key not in output, f"duplicate row key: {key}")
            output[key] = row
    return output


def validate_crn_topology(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    items: list[dict[str, Any]], complete: bool,
) -> None:
    """Authenticate the intended A/B x original/swapped common-RNG blocks."""
    by_seed_key: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        seed_key = str(item["seed_key"])
        by_seed_key.setdefault(seed_key, []).append(item)
    core.require(len(by_seed_key) == core.SEMANTIC_INSTANCE_COUNT,
                 "semantic-twin seed-key denominator changed")
    for seed_key, twins in by_seed_key.items():
        core.require(len(twins) == 2
                     and len({str(item["semantic_id"]) for item in twins}) == 1
                     and {str(item["swap_side"]) for item in twins}
                     == set(core.SWAP_SIDES),
                     f"invalid semantic-twin seed block: {seed_key}")
        item_ids = [str(item["id"]) for item in twins]
        for stream in core.STREAMS:
            block = [
                rows[(item_id, arm, stream)]
                for item_id in item_ids for arm in core.ARMS
                if (item_id, arm, stream) in rows
            ]
            if complete:
                core.require(len(block) == 4,
                             f"incomplete 2x2 CRN block: {seed_key}/stream_{stream}")
            if block:
                core.require({row["seed_key"] for row in block} == {seed_key}
                             and len({row["pair_seed"] for row in block}) == 1,
                             f"2x2 CRN seed mismatch: {seed_key}/stream_{stream}")


def _load_manifest_for_lineage(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
) -> dict[str, Any]:
    manifest = core.load_json(core.manifest_path(model_key), f"{model_key} manifest")
    core.require(set(manifest) == MANIFEST_KEYS,
                 f"{model_key} manifest has unknown/missing keys")
    core.verify_self_hash(manifest, "manifest_sha256", "created_at_utc",
                          f"{model_key} manifest")
    core.require(
        manifest.get("model_key") == model_key
        and manifest.get("protocol_sha256") == protocol["protocol_sha256"]
        and manifest.get("admission_sha256") == admission["admission_sha256"]
        and manifest.get("expected_rows") == core.EXPECTED_ROWS_PER_MODEL
        and manifest.get("arms") == core.ARMS
        and manifest.get("streams") == list(core.STREAMS)
        and manifest.get("sampling") == core.SAMPLING
        and manifest.get("max_new_tokens") == core.MAX_NEW_TOKENS
        and manifest.get("creation_state", {}).get("decision_computed") is False,
        f"{model_key} manifest lineage changed",
    )
    return manifest


def _verify_complete_status_document(
    status: dict[str, Any], model_key: str, protocol_sha256: str,
    admission_sha256: str, manifest_sha256: str,
    file_identity: dict[str, Any],
) -> None:
    core.require(set(status) == STATUS_KEYS,
                 f"{model_key} status has unknown/missing keys")
    core.verify_self_hash(status, "status_sha256", "updated_at_utc",
                          f"{model_key} status")
    expected_cells = {
        arm: {str(stream): core.ITEM_COUNT for stream in core.STREAMS}
        for arm in core.ARMS
    }
    core.require(
        status.get("model_key") == model_key
        and status.get("protocol_sha256") == protocol_sha256
        and status.get("admission_sha256") == admission_sha256
        and status.get("manifest_sha256") == manifest_sha256
        and status.get("completed_rows") == core.EXPECTED_ROWS_PER_MODEL
        and status.get("expected_rows") == core.EXPECTED_ROWS_PER_MODEL
        and status.get("cell_counts") == expected_cells
        and status.get("complete") is True
        and status.get("rows_file_sha256") == file_identity["rows_file_sha256"]
        and status.get("rows_file_bytes") == file_identity["rows_file_bytes"]
        and status.get("rows_file_line_count")
        == file_identity["rows_file_line_count"] == core.EXPECTED_ROWS_PER_MODEL
        and status.get("rows_file_terminal_newline") is True
        and file_identity["rows_file_terminal_newline"] is True
        and status.get("generation_performed") is True
        and status.get("model_weights_loaded") is True
        and status.get("gpu_used") is True
        and status.get("decision_computed") is False
        and status.get("holdout") is False
        and status.get("mechanism") is False,
        f"{model_key} completed status/rows lineage changed",
    )


def verify_complete_model_output(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fully authenticate a completed model before skip/order authorization."""
    manifest = _load_manifest_for_lineage(model_key, protocol, admission)
    # CPU-only tokenizer/config loading is sufficient to validate every stored
    # row.  No model weights or CUDA objects are touched on this path.
    tokenizer = engine._load_inspection_bundle(model_key).tokenizer
    grid = core.canonical_grid(items)
    positions = {
        (str(item["id"]), arm, stream): index
        for index, (item, arm, stream) in enumerate(grid)
    }
    rows = load_rows(manifest, tokenizer, items, positions)
    core.require(len(rows) == core.EXPECTED_ROWS_PER_MODEL
                 and set(rows) == set(positions),
                 f"{model_key} completed row grid changed")
    validate_crn_topology(rows, items, True)
    identity = rows_file_identity(core.rows_path(model_key), True)
    status = core.load_json(core.status_path(model_key), f"{model_key} status")
    _verify_complete_status_document(
        status, model_key, protocol["protocol_sha256"],
        admission["admission_sha256"], manifest["manifest_sha256"], identity,
    )
    return status


def write_status(
    manifest: dict[str, Any], rows: dict[tuple[str, str, int], dict[str, Any]],
    rows_tracker: dict[str, Any], elapsed_seconds: float, complete: bool,
) -> dict[str, Any]:
    model_key = manifest["model_key"]
    if complete:
        core.require(len(rows) == core.EXPECTED_ROWS_PER_MODEL,
                     f"{model_key} cannot publish a premature complete status")
    counts = {arm: {str(stream): 0 for stream in core.STREAMS} for arm in core.ARMS}
    for _item, arm, stream in rows:
        counts[arm][str(stream)] += 1
    file_identity = tracker_identity(rows_tracker)
    core.require(file_identity["rows_file_line_count"] == len(rows),
                 f"{model_key} status row/file denominator mismatch")
    if rows:
        core.require(file_identity["rows_file_terminal_newline"] is True,
                     f"{model_key} rows tracker lacks terminal newline")
    if complete:
        disk_identity = rows_file_identity(core.rows_path(model_key), True)
        core.require(file_identity == disk_identity,
                     f"{model_key} final incremental rows SHA differs from disk")
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "model_key": model_key,
        "protocol_sha256": manifest["protocol_sha256"],
        "admission_sha256": manifest["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_rows": len(rows),
        "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
        "cell_counts": counts,
        "complete": bool(complete),
        "elapsed_seconds_current_process": elapsed_seconds,
        **file_identity,
        "generation_performed": bool(rows),
        "model_weights_loaded": True,
        "gpu_used": True,
        "decision_computed": False,
        "holdout": False,
        "mechanism": False,
    }
    document = {
        **payload,
        "status_sha256": core.sha256_json(payload),
        "updated_at_utc": core.utc_now(),
    }
    core.require(set(document) == STATUS_KEYS, "status schema construction changed")
    core.atomic_write_json(core.status_path(model_key), document)
    installed = core.load_json(
        core.status_path(model_key), f"installed {model_key} status")
    core.require(set(installed) == STATUS_KEYS,
                 f"installed {model_key} status schema changed")
    core.verify_self_hash(installed, "status_sha256", "updated_at_utc",
                          f"installed {model_key} status")
    core.require(installed == document,
                 f"installed {model_key} status changed in serialization")
    return installed


def acquire_lock(model_key: str, protocol_sha256: str) -> int:
    path = core.run_lock_path(model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    core.require(len(protocol_sha256) == 64, "runner lock protocol hash invalid")
    document = _lock_document(model_key, protocol_sha256)
    for attempt in range(2):
        try:
            return core.atomic_publish_lock(path, document)
        except FileExistsError:
            core.require(attempt == 0, f"{model_key} runner lock reappeared")
            _recover_stale_lock(path, model_key, protocol_sha256)
    raise RuntimeError(
        f"could not acquire {model_key} runner lock")  # pragma: no cover


def completed_preflight(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not core.status_path(model_key).exists():
        return None
    status = core.load_json(core.status_path(model_key), f"{model_key} status")
    core.require(set(status) == STATUS_KEYS,
                 f"{model_key} status has unknown/missing keys")
    core.verify_self_hash(status, "status_sha256", "updated_at_utc",
                          f"{model_key} status")
    if status.get("complete") is not True:
        return None
    return verify_complete_model_output(
        model_key, protocol, admission, items,
    )


def run(model_key: str) -> None:
    protocol, admission, dataset_audit, items = authenticate(model_key)
    current_lock = core.run_lock_path(model_key)
    if current_lock.exists():
        _recover_stale_lock(
            current_lock, model_key, protocol["protocol_sha256"])
    existing_complete = completed_preflight(
        model_key, protocol, admission, items,
    )
    if existing_complete is not None:
        print(json.dumps({
            "model_key": model_key, "already_complete": True,
            "completed_rows": existing_complete["completed_rows"],
            "decision_computed": False,
        }, ensure_ascii=False, sort_keys=True))
        return
    core.require(not core.COMBINED_AUDIT_PATH.exists(),
                 "independent audit exists; incomplete generation is frozen")
    import torch
    core.require(torch.cuda.is_available(), "formal generation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    current_runtime = runtime_versions()
    for prior in core.MODEL_ORDER[:core.MODEL_ORDER.index(model_key)]:
        prior_manifest = _load_manifest_for_lineage(prior, protocol, admission)
        core.require(prior_manifest.get("runtime_versions") == current_runtime,
                     f"runtime/hardware drift since completed {prior}")
    if core.manifest_path(model_key).exists():
        current_manifest = _load_manifest_for_lineage(
            model_key, protocol, admission)
        core.require(current_manifest.get("runtime_versions") == current_runtime,
                     f"runtime/hardware drift within resumed {model_key}")
    for value in core.MODEL_ORDER:
        path = core.run_lock_path(value)
        if path.exists():
            _recover_stale_lock(path, value, protocol["protocol_sha256"])
    descriptor = acquire_lock(model_key, protocol["protocol_sha256"])
    start = time.time()
    adapter: engine.ModelAdapter | None = None
    try:
        adapter = engine.load_model_adapter(model_key)
        manifest = install_manifest(make_manifest(
            model_key, protocol, admission, dataset_audit, items[0], adapter))
        grid = core.canonical_grid(items)
        positions = {
            (str(item["id"]), arm, stream): index
            for index, (item, arm, stream) in enumerate(grid)
        }
        rows = load_rows(manifest, adapter.tokenizer, items, positions)
        validate_crn_topology(rows, items, False)
        rows_tracker = build_rows_tracker(core.rows_path(model_key))
        core.require(tracker_identity(rows_tracker)["rows_file_line_count"] == len(rows),
                     f"{model_key} resumed rows tracker denominator changed")
        item_by_id = {str(item["id"]): item for item in items}
        write_status(manifest, rows, rows_tracker, time.time() - start,
                     len(rows) == core.EXPECTED_ROWS_PER_MODEL)
        total_batches = core.EXPECTED_ROWS_PER_MODEL // core.BATCH_SIZE
        for batch_index, jobs in enumerate(core.chunks(grid), 1):
            keys = [(str(item["id"]), arm, stream) for item, arm, stream in jobs]
            if all(key in rows for key in keys):
                continue
            requests = [engine.SamplingRequest(
                item_id=str(item["id"]), stream=stream, arm=arm,
                user_text=core.effective_user_prompt(item, arm),
                seed_key=str(item["seed_key"]),
            ) for item, arm, stream in jobs]
            sampled_rows = engine.sample_batch(
                adapter, requests, manifest["dataset_namespace"],
                core.MAX_NEW_TOKENS, core.BATCH_SIZE,
            )
            built = [build_row(
                manifest, adapter, item, arm, stream, batch_index,
                sampled_rows[index],
            ) for index, (item, arm, stream) in enumerate(jobs)]
            for row in built:
                validate_row(row, manifest, adapter.tokenizer, item_by_id, positions)
                key = core.row_key(row)
                if key in rows:
                    core.require(rows[key]["row_sha256"] == row["row_sha256"],
                                 f"partial batch replay changed row: {key}")
                else:
                    core.append_jsonl(core.rows_path(model_key), row)
                    update_rows_tracker(rows_tracker, row)
                    rows[key] = row
            complete = len(rows) == core.EXPECTED_ROWS_PER_MODEL
            if not complete:
                write_status(
                    manifest, rows, rows_tracker, time.time() - start, False)
            if batch_index % 8 == 0 or complete:
                print(
                    f"[{model_key}] batches={batch_index}/{total_batches}; "
                    f"rows={len(rows)}/{core.EXPECTED_ROWS_PER_MODEL}",
                    flush=True,
                )
        core.require(len(rows) == core.EXPECTED_ROWS_PER_MODEL
                     and set(rows) == set(positions),
                     f"{model_key} formal grid incomplete")
        validate_crn_topology(rows, items, True)
        final_status = write_status(
            manifest, rows, rows_tracker, time.time() - start, True)
        core.verify_self_hash(final_status, "status_sha256", "updated_at_utc",
                              f"final {model_key} status")
        print(
            f"[{model_key}] complete rows={len(rows)}; "
            f"elapsed={(time.time()-start)/3600:.2f}h",
            flush=True,
        )
    finally:
        engine.release_model_adapter(adapter)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.close(descriptor)
        core.run_lock_path(model_key).unlink(missing_ok=True)


def _expect_error(callable_value: Any) -> bool:
    try:
        callable_value()
    except (RuntimeError, OSError, ValueError):
        return True
    return False


def self_test() -> dict[str, Any]:
    """CPU-only recovery, lineage, and 2x2 seed-topology negative tests."""
    protocol_sha256 = "a" * 64
    admission_sha256 = "b" * 64
    manifest_sha256 = "c" * 64
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="phase983_runner_selftest_") as temporary:
        root = Path(temporary)

        atomic_lock = root / "atomic.lock"
        atomic_document = _lock_document("qwen3", protocol_sha256)
        atomic_descriptor = core.atomic_publish_lock(
            atomic_lock, atomic_document)
        try:
            installed_lock = core.load_json(
                atomic_lock, "self-test atomically published lock")
            checks["atomic_lock_published_only_after_complete_fsync"] = (
                installed_lock == atomic_document
                and atomic_lock.read_bytes()
                == (core.canonical_json(atomic_document) + "\n").encode("utf-8")
            )
            original_lock_bytes = atomic_lock.read_bytes()
            checks["atomic_lock_collision_preserves_existing"] = _expect_error(
                lambda: core.atomic_publish_lock(
                    atomic_lock, atomic_document)) and (
                        atomic_lock.read_bytes() == original_lock_bytes)
        finally:
            os.close(atomic_descriptor)
            atomic_lock.unlink(missing_ok=True)
        checks["atomic_lock_candidate_cleanup"] = not list(
            root.glob(".atomic.lock.*.candidate"))

        guard_path = root / "recovery.guard"
        guard_descriptor = _acquire_recovery_guard(guard_path)
        try:
            checks["concurrent_recovery_guard_rejected"] = _expect_error(
                lambda: _acquire_recovery_guard(guard_path))
        finally:
            _release_recovery_guard(guard_descriptor)
        replacement_guard = _acquire_recovery_guard(guard_path)
        _release_recovery_guard(replacement_guard)
        checks["crash_safe_recovery_guard_reacquired"] = True

        row_payload = {"schema_version": core.SCHEMA_VERSION, "id": "tail-row"}
        row = {**row_payload, "row_sha256": core.sha256_json(row_payload)}
        encoded = core.canonical_json(row).encode("utf-8")
        no_newline = root / "complete_no_newline.jsonl"
        no_newline.write_bytes(encoded)
        repair_truncated_tail(no_newline)
        checks["complete_canonical_self_hashed_tail_repaired"] = (
            no_newline.read_bytes() == encoded + b"\n"
        )

        partial = root / "partial_tail.jsonl"
        partial.write_bytes(encoded + b"\n" + b'{"incomplete":')
        repair_truncated_tail(partial)
        checks["partial_tail_truncated_to_last_fsync_row"] = (
            partial.read_bytes() == encoded + b"\n"
        )

        bad_row = {**row, "id": "tampered"}
        bad_tail = root / "bad_complete_tail.jsonl"
        bad_tail.write_bytes(core.canonical_json(bad_row).encode("utf-8"))
        checks["bad_complete_tail_rejected"] = _expect_error(
            lambda: repair_truncated_tail(bad_tail))

        lock = root / "runner.lock"
        active = _lock_document("qwen3", protocol_sha256)
        lock.write_text(core.canonical_json(active) + "\n", encoding="utf-8")
        checks["active_lock_rejected"] = _expect_error(
            lambda: _recover_stale_lock(
                lock, "qwen3", protocol_sha256)) and lock.exists()
        lock.unlink()

        dead_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "kind": "phase983_model_runner",
            "pid": 2_147_483_640,
            "process_start_token": "proved-old-process",
            "model_key": "qwen3",
            "protocol_sha256": protocol_sha256,
        }
        dead = {**dead_payload, "lock_sha256": core.sha256_json(dead_payload)}
        lock.write_text(core.canonical_json(dead) + "\n", encoding="utf-8")
        archived = _recover_stale_lock(lock, "qwen3", protocol_sha256)
        checks["proved_stale_lock_archived"] = archived.is_file() and not lock.exists()

        tampered = {**dead, "protocol_sha256": "d" * 64}
        lock.write_text(core.canonical_json(tampered) + "\n", encoding="utf-8")
        checks["tampered_lock_rejected"] = _expect_error(
            lambda: _recover_stale_lock(lock, "qwen3", protocol_sha256))
        lock.unlink(missing_ok=True)

        rows_path = root / "rows.jsonl"
        rows_path.write_bytes(b"{}\n" * core.EXPECTED_ROWS_PER_MODEL)
        identity = rows_file_identity(rows_path, True)
        cells = {
            arm: {str(stream): core.ITEM_COUNT for stream in core.STREAMS}
            for arm in core.ARMS
        }
        status_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "model_key": "qwen3",
            "protocol_sha256": protocol_sha256,
            "admission_sha256": admission_sha256,
            "manifest_sha256": manifest_sha256,
            "completed_rows": core.EXPECTED_ROWS_PER_MODEL,
            "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
            "cell_counts": cells,
            "complete": True,
            "elapsed_seconds_current_process": 1.0,
            **identity,
            "generation_performed": True,
            "model_weights_loaded": True,
            "gpu_used": True,
            "decision_computed": False,
            "holdout": False,
            "mechanism": False,
        }
        status = {
            **status_payload,
            "status_sha256": core.sha256_json(status_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        _verify_complete_status_document(
            status, "qwen3", protocol_sha256, admission_sha256,
            manifest_sha256, identity,
        )
        checks["complete_status_binds_lineage_and_rows"] = True
        wrong_payload = {**status_payload, "protocol_sha256": "e" * 64}
        wrong = {
            **wrong_payload,
            "status_sha256": core.sha256_json(wrong_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        checks["self_consistent_wrong_protocol_rejected"] = _expect_error(
            lambda: _verify_complete_status_document(
                wrong, "qwen3", protocol_sha256, admission_sha256,
                manifest_sha256, identity))
        extra_status_payload = {**status_payload, "unknown": "must-fail"}
        extra_status = {
            **extra_status_payload,
            "status_sha256": core.sha256_json(extra_status_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        checks["self_hashed_extra_status_key_rejected"] = _expect_error(
            lambda: _verify_complete_status_document(
                extra_status, "qwen3", protocol_sha256, admission_sha256,
                manifest_sha256, identity))
        checks["extra_row_key_schema_rejected"] = _expect_error(
            lambda: core.require(
                set(ROW_KEYS) | {"unknown"} == ROW_KEYS,
                "row has unknown/missing keys"))
        rows_path.write_bytes(b"X" + rows_path.read_bytes()[1:])
        changed_identity = rows_file_identity(rows_path, True)
        checks["changed_rows_file_rejected"] = _expect_error(
            lambda: _verify_complete_status_document(
                status, "qwen3", protocol_sha256, admission_sha256,
                manifest_sha256, changed_identity))

    seed_original_a = engine.stable_pair_seed(
        protocol_sha256, "qwen3", "shared-seed-key", 0, "A")
    seed_swapped_b = engine.stable_pair_seed(
        protocol_sha256, "qwen3", "shared-seed-key", 0, "B")
    seed_other = engine.stable_pair_seed(
        protocol_sha256, "qwen3", "different-seed-key", 0, "A")
    checks["two_by_two_crn_seed_shared"] = seed_original_a == seed_swapped_b
    checks["different_semantic_seed_key_separated"] = seed_original_a != seed_other
    core.require(all(checks.values()), f"runner self-test failed: {checks}")
    return {
        "schema_version": core.SCHEMA_VERSION,
        "tests": checks,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written_outside_temporary_directory": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--model", choices=core.MODEL_ORDER)
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        run(str(args.model))


if __name__ == "__main__":
    main()
