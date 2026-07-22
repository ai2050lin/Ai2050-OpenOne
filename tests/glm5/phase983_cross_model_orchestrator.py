#!/usr/bin/env python3
"""Strict sequential orchestrator: Qwen3 -> GLM4 -> DeepSeek7B."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402


RUNNER = GLM5 / "phase983_cross_model_runner.py"

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

ORCHESTRATOR_STATUS_KEYS = frozenset({
    "schema_version", "phase", "experiment", "protocol_sha256",
    "admission_sha256", "strict_model_order", "models",
    "completed_model_count", "complete",
    "elapsed_seconds_current_orchestration",
    "one_model_subprocess_at_a_time", "decision_computed", "holdout",
    "mechanism", "orchestrator_status_sha256", "updated_at_utc",
})

ORCHESTRATOR_MODEL_RECORD_KEYS = frozenset({
    "model_key", "returncode", "elapsed_seconds_this_invocation",
    "status_sha256", "rows_file_sha256", "rows_file_bytes",
    "rows_file_line_count", "gpu_before", "gpu_after_child_exit",
    "separate_subprocess", "complete",
})

GPU_SNAPSHOT_KEYS = frozenset({
    "index", "name", "memory_total_mib", "memory_used_mib",
    "memory_free_mib",
})


def rows_file_identity(path: Path, require_terminal_newline: bool) -> dict[str, Any]:
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
        core.require(terminal_newline, f"completed rows file lacks newline: {path}")
    return {
        "rows_file_sha256": digest.hexdigest(),
        "rows_file_bytes": byte_count,
        "rows_file_line_count": line_count,
        "rows_file_terminal_newline": terminal_newline,
    }


def _windows_process_probe(pid: int) -> tuple[str, str | None]:
    import ctypes
    from ctypes import wintypes

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
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return ("dead", None) if error == 87 else (
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
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return "dead", None
    if os.name == "nt":
        return _windows_process_probe(pid)
    path = Path(f"/proc/{pid}/stat")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "dead", None
    except OSError as exc:
        return "unknown", f"proc-error:{type(exc).__name__}"
    try:
        fields = text[text.rfind(")") + 2:].split()
        return "alive", f"proc-start-ticks:{fields[19]}"
    except (IndexError, ValueError):
        return "unknown", "malformed-proc-stat"


def _lock_document(protocol_sha256: str) -> dict[str, Any]:
    state, token = process_probe(os.getpid())
    core.require(state == "alive" and token is not None,
                 "cannot authenticate current orchestrator process")
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "kind": "phase983_orchestrator",
        "pid": os.getpid(),
        "process_start_token": token,
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
        raise RuntimeError("orchestrator lock recovery already active") from exc


def _release_recovery_guard(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def _recover_stale_lock(path: Path, protocol_sha256: str) -> Path:
    core.require(len(protocol_sha256) == 64,
                 "orchestrator stale-lock protocol hash invalid")
    guard = path.with_name(f"{path.name}.recovery")
    guard_descriptor = _acquire_recovery_guard(guard)
    try:
        document = core.load_json(path, "Phase983 orchestrator lock")
        core.require(document.get("lock_sha256") == core.sha256_json(
            core.without_fields(document, "lock_sha256")),
            "orchestrator lock self-hash invalid")
        core.require(document.get("kind") == "phase983_orchestrator"
                     and isinstance(document.get("protocol_sha256"), str)
                     and len(document["protocol_sha256"]) == 64
                     and isinstance(document.get("process_start_token"), str)
                     and bool(document["process_start_token"]),
                     "orchestrator lock schema invalid")
        pid = document.get("pid")
        core.require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
                     "orchestrator lock PID invalid")
        state, observed_token = process_probe(pid)
        core.require(state != "unknown",
                     f"cannot prove whether orchestrator PID {pid} is stale")
        if state == "alive" and observed_token == document.get("process_start_token"):
            raise RuntimeError(f"active Phase983 orchestrator PID {pid} exists")
        stale = path.with_name(
            f"{path.name}.stale.{pid}.{os.getpid()}.{time.time_ns()}.json")
        os.replace(path, stale)
        return stale
    finally:
        _release_recovery_guard(guard_descriptor)


def authenticate() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = core.load_json(core.PROTOCOL_PATH, "Phase983 protocol")
    admission = core.load_json(core.ADMISSION_PATH, "Phase983 admission")
    core.verify_self_hash(protocol, "protocol_sha256", "created_at_utc",
                          "Phase983 protocol")
    core.verify_self_hash(admission, "admission_sha256", "created_at_utc",
                          "Phase983 admission")
    core.require(admission.get("admitted") is True
                 and admission.get("gpu_authorized") is True
                 and admission.get("protocol_sha256") == protocol["protocol_sha256"]
                 and admission.get("authorization_scope", {}).get("strict_order")
                 == list(core.MODEL_ORDER), "orchestrator admission changed")
    core.verify_file_seals(protocol.get("script_seals"), core.SCRIPT_PATHS,
                           "Phase983 script")
    return protocol, admission


def gpu_snapshot() -> dict[str, Any]:
    completed = subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="strict",
       timeout=30, check=False)
    core.require(completed.returncode == 0, "nvidia-smi failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    core.require(len(lines) == 1, "formal run requires one visible GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    core.require(len(fields) == 5, "unexpected nvidia-smi schema")
    return {
        "index": int(fields[0]), "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
    }


def acquire_lock(protocol_sha256: str) -> int:
    core.ORCHESTRATOR_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    core.require(len(protocol_sha256) == 64,
                 "orchestrator lock protocol hash invalid")
    document = _lock_document(protocol_sha256)
    for attempt in range(2):
        try:
            return core.atomic_publish_lock(
                core.ORCHESTRATOR_LOCK_PATH, document)
        except FileExistsError:
            core.require(attempt == 0, "orchestrator lock reappeared")
            _recover_stale_lock(core.ORCHESTRATOR_LOCK_PATH, protocol_sha256)
    raise RuntimeError(
        "could not acquire Phase983 orchestrator lock")  # pragma: no cover


def _verify_model_status_documents(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
    manifest: dict[str, Any], status: dict[str, Any], identity: dict[str, Any],
) -> None:
    core.require(set(manifest) == MANIFEST_KEYS,
                 f"{model_key} manifest has unknown/missing keys")
    core.require(set(status) == STATUS_KEYS,
                 f"{model_key} status has unknown/missing keys")
    core.verify_self_hash(manifest, "manifest_sha256", "created_at_utc",
                          f"{model_key} manifest")
    core.verify_self_hash(status, "status_sha256", "updated_at_utc",
                          f"{model_key} status")
    expected_cells = {
        arm: {str(stream): core.ITEM_COUNT for stream in core.STREAMS}
        for arm in core.ARMS
    }
    core.require(manifest.get("schema_version") == core.SCHEMA_VERSION
                 and manifest.get("phase") == core.PHASE
                 and manifest.get("experiment") == core.EXPERIMENT
                 and manifest.get("model_key") == model_key
                 and manifest.get("model_order_index")
                 == core.MODEL_ORDER.index(model_key)
                 and manifest.get("protocol_sha256") == protocol["protocol_sha256"]
                 and manifest.get("admission_sha256") == admission["admission_sha256"]
                 and manifest.get("expected_rows") == core.EXPECTED_ROWS_PER_MODEL,
                 f"{model_key} manifest lineage changed")
    core.require(status.get("schema_version") == core.SCHEMA_VERSION
                 and status.get("phase") == core.PHASE
                 and status.get("experiment") == core.EXPERIMENT
                 and status.get("model_key") == model_key
                 and status.get("protocol_sha256") == protocol["protocol_sha256"]
                 and status.get("admission_sha256") == admission["admission_sha256"]
                 and status.get("manifest_sha256") == manifest["manifest_sha256"]
                 and status.get("complete") is True
                 and status.get("completed_rows") == core.EXPECTED_ROWS_PER_MODEL
                 and status.get("expected_rows") == core.EXPECTED_ROWS_PER_MODEL
                 and status.get("cell_counts") == expected_cells
                 and status.get("rows_file_sha256") == identity["rows_file_sha256"]
                 and status.get("rows_file_bytes") == identity["rows_file_bytes"]
                 and status.get("rows_file_line_count")
                 == identity["rows_file_line_count"] == core.EXPECTED_ROWS_PER_MODEL
                 and status.get("rows_file_terminal_newline") is True
                 and identity["rows_file_terminal_newline"] is True
                 and status.get("generation_performed") is True
                 and status.get("model_weights_loaded") is True
                 and status.get("gpu_used") is True
                 and _finite_nonnegative(
                     status.get("elapsed_seconds_current_process"))
                 and status.get("decision_computed") is False
                 and status.get("holdout") is False
                 and status.get("mechanism") is False,
                 f"{model_key} did not complete formal generation")


def verify_model_status(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
) -> dict[str, Any]:
    manifest = core.load_json(core.manifest_path(model_key), f"{model_key} manifest")
    status = core.load_json(core.status_path(model_key), f"{model_key} status")
    identity = rows_file_identity(core.rows_path(model_key), True)
    _verify_model_status_documents(
        model_key, protocol, admission, manifest, status, identity,
    )
    return status


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _verify_gpu_snapshot(snapshot: Any, label: str) -> None:
    core.require(isinstance(snapshot, dict) and set(snapshot) == GPU_SNAPSHOT_KEYS,
                 f"{label} GPU snapshot schema changed")
    core.require(
        isinstance(snapshot.get("index"), int)
        and not isinstance(snapshot.get("index"), bool)
        and snapshot["index"] >= 0
        and isinstance(snapshot.get("name"), str)
        and bool(snapshot["name"])
        and all(
            isinstance(snapshot.get(field), int)
            and not isinstance(snapshot.get(field), bool)
            and snapshot[field] >= 0
            for field in ("memory_total_mib", "memory_used_mib", "memory_free_mib")
        )
        and snapshot["memory_total_mib"] > 0
        and snapshot["memory_used_mib"] <= snapshot["memory_total_mib"]
        and snapshot["memory_free_mib"] <= snapshot["memory_total_mib"],
        f"{label} GPU snapshot values invalid",
    )


def _verify_complete_orchestrator_document(
    document: dict[str, Any], protocol: dict[str, Any],
    admission: dict[str, Any], current_statuses: dict[str, dict[str, Any]],
) -> None:
    core.require(set(document) == ORCHESTRATOR_STATUS_KEYS,
                 "complete orchestrator status has unknown/missing keys")
    core.verify_self_hash(
        document, "orchestrator_status_sha256", "updated_at_utc",
        "complete orchestrator status",
    )
    core.require(
        document.get("schema_version") == core.SCHEMA_VERSION
        and document.get("phase") == core.PHASE
        and document.get("experiment") == core.EXPERIMENT
        and document.get("protocol_sha256") == protocol["protocol_sha256"]
        and document.get("admission_sha256") == admission["admission_sha256"]
        and document.get("strict_model_order") == list(core.MODEL_ORDER)
        and document.get("completed_model_count") == len(core.MODEL_ORDER)
        and document.get("complete") is True
        and _finite_nonnegative(
            document.get("elapsed_seconds_current_orchestration"))
        and document.get("one_model_subprocess_at_a_time") is True
        and document.get("decision_computed") is False
        and document.get("holdout") is False
        and document.get("mechanism") is False,
        "complete orchestrator status lineage/state changed",
    )
    records = document.get("models")
    core.require(
        isinstance(records, list) and len(records) == len(core.MODEL_ORDER),
        "complete orchestrator model denominator changed",
    )
    core.require(set(current_statuses) == set(core.MODEL_ORDER),
                 "current completed model status registry changed")
    for model_key, record in zip(core.MODEL_ORDER, records, strict=True):
        core.require(
            isinstance(record, dict)
            and set(record) == ORCHESTRATOR_MODEL_RECORD_KEYS,
            f"{model_key} orchestrator record schema changed",
        )
        status = current_statuses[model_key]
        core.require(
            record.get("model_key") == model_key
            and record.get("returncode") == 0
            and not isinstance(record.get("returncode"), bool)
            and _finite_nonnegative(
                record.get("elapsed_seconds_this_invocation"))
            and record.get("status_sha256") == status["status_sha256"]
            and record.get("rows_file_sha256") == status["rows_file_sha256"]
            and record.get("rows_file_bytes") == status["rows_file_bytes"]
            and record.get("rows_file_line_count")
            == status["rows_file_line_count"] == core.EXPECTED_ROWS_PER_MODEL
            and record.get("separate_subprocess") is True
            and record.get("complete") is True,
            f"{model_key} orchestrator record changed",
        )
        _verify_gpu_snapshot(record.get("gpu_before"), f"{model_key} before")
        _verify_gpu_snapshot(
            record.get("gpu_after_child_exit"), f"{model_key} after")


def completed_preflight(
    protocol: dict[str, Any], admission: dict[str, Any],
) -> dict[str, Any] | None:
    if not core.ORCHESTRATOR_STATUS_PATH.exists():
        return None
    document = core.load_json(
        core.ORCHESTRATOR_STATUS_PATH, "Phase983 orchestrator status")
    core.require(set(document) == ORCHESTRATOR_STATUS_KEYS,
                 "orchestrator status has unknown/missing keys")
    core.verify_self_hash(
        document, "orchestrator_status_sha256", "updated_at_utc",
        "Phase983 orchestrator status",
    )
    if document.get("complete") is not True:
        core.require(document.get("complete") is False,
                     "orchestrator complete flag is not boolean")
        return None
    statuses = {
        model_key: verify_model_status(model_key, protocol, admission)
        for model_key in core.MODEL_ORDER
    }
    _verify_complete_orchestrator_document(
        document, protocol, admission, statuses)
    return document


def write_orchestrator_status(
    protocol: dict[str, Any], admission: dict[str, Any], records: list[dict[str, Any]],
    complete: bool, start: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "protocol_sha256": protocol["protocol_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "strict_model_order": list(core.MODEL_ORDER),
        "models": records,
        "completed_model_count": len(records),
        "complete": bool(complete),
        "elapsed_seconds_current_orchestration": time.time() - start,
        "one_model_subprocess_at_a_time": True,
        "decision_computed": False,
        "holdout": False,
        "mechanism": False,
    }
    document = {
        **payload,
        "orchestrator_status_sha256": core.sha256_json(payload),
        "updated_at_utc": core.utc_now(),
    }
    core.require(set(document) == ORCHESTRATOR_STATUS_KEYS,
                 "orchestrator status construction schema changed")
    core.atomic_write_json(core.ORCHESTRATOR_STATUS_PATH, document)
    installed = core.load_json(
        core.ORCHESTRATOR_STATUS_PATH, "installed orchestrator status")
    core.verify_self_hash(
        installed, "orchestrator_status_sha256", "updated_at_utc",
        "installed orchestrator status")
    core.require(installed == document,
                 "installed orchestrator status changed in serialization")
    return installed


def runner_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def run_child(model_key: str) -> int:
    environment = runner_environment()
    process = subprocess.Popen(
        [sys.executable, str(RUNNER), "--model", model_key],
        cwd=str(core.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="strict", bufsize=1,
        env=environment,
    )
    core.require(process.stdout is not None, "runner stdout pipe missing")
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return int(process.wait())


def run() -> None:
    protocol, admission = authenticate()
    if core.ORCHESTRATOR_LOCK_PATH.exists():
        _recover_stale_lock(
            core.ORCHESTRATOR_LOCK_PATH, protocol["protocol_sha256"])
    for model_key in core.MODEL_ORDER:
        model_lock = core.run_lock_path(model_key)
        if model_lock.exists():
            # A completed orchestration must not leave a live child or an
            # unauthenticated stale lock behind.  Proved-dead locks are
            # archived before the read-only complete fast path.
            import phase983_cross_model_runner as runner
            runner._recover_stale_lock(
                model_lock, model_key, protocol["protocol_sha256"])
    existing_complete = completed_preflight(protocol, admission)
    if existing_complete is not None:
        print(json.dumps({
            "already_complete": True,
            "completed_model_count": existing_complete["completed_model_count"],
            "decision_computed": False,
        }, ensure_ascii=False, sort_keys=True))
        return
    core.require(not core.COMBINED_AUDIT_PATH.exists(),
                 "independent audit exists; incomplete generation is frozen")
    descriptor = acquire_lock(protocol["protocol_sha256"])
    start = time.time()
    records: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    try:
        for model_key in core.MODEL_ORDER:
            before = gpu_snapshot()
            print(
                f"[orchestrator] starting {model_key}; "
                f"GPU free={before['memory_free_mib']} MiB",
                flush=True,
            )
            model_start = time.time()
            returncode = run_child(model_key)
            after = gpu_snapshot()
            core.require(returncode == 0, f"{model_key} runner exited {returncode}")
            status = verify_model_status(model_key, protocol, admission)
            record = {
                "model_key": model_key,
                "returncode": returncode,
                "elapsed_seconds_this_invocation": time.time() - model_start,
                "status_sha256": status["status_sha256"],
                "rows_file_sha256": status["rows_file_sha256"],
                "rows_file_bytes": status["rows_file_bytes"],
                "rows_file_line_count": status["rows_file_line_count"],
                "gpu_before": before,
                "gpu_after_child_exit": after,
                "separate_subprocess": True,
                "complete": True,
            }
            records.append(record)
            final = write_orchestrator_status(
                protocol, admission, records, len(records) == len(core.MODEL_ORDER), start)
            print(
                f"[orchestrator] completed {model_key}; "
                f"GPU free after exit={after['memory_free_mib']} MiB",
                flush=True,
            )
        core.require(final is not None
                     and final["complete"] is True
                     and final["completed_model_count"] == len(core.MODEL_ORDER),
                     "orchestrator final model denominator changed")
        print(
            f"[orchestrator] complete models={len(records)}; "
            f"elapsed={(time.time()-start)/3600:.2f}h",
            flush=True,
        )
    finally:
        os.close(descriptor)
        core.ORCHESTRATOR_LOCK_PATH.unlink(missing_ok=True)


def _expect_error(callable_value: Any) -> bool:
    try:
        callable_value()
    except (RuntimeError, OSError, ValueError):
        return True
    return False


def self_test() -> dict[str, Any]:
    protocol_sha256 = "a" * 64
    admission_sha256 = "b" * 64
    checks: dict[str, bool] = {}
    encoding_probe = subprocess.run(
        [sys.executable, "-c", "print('Phase983 UTF-8 probe: 进度 █')"],
        capture_output=True, text=False, env=runner_environment(),
        timeout=30, check=False,
    )
    checks["runner_child_stdout_forced_to_strict_utf8"] = (
        encoding_probe.returncode == 0
        and encoding_probe.stderr == b""
        and encoding_probe.stdout.decode("utf-8", errors="strict").strip()
        == "Phase983 UTF-8 probe: 进度 █"
    )
    with tempfile.TemporaryDirectory(
        prefix="phase983_orchestrator_selftest_",
    ) as temporary:
        root = Path(temporary)
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

        lock = root / "orchestrator.lock"
        active = _lock_document(protocol_sha256)
        lock.write_text(core.canonical_json(active) + "\n", encoding="utf-8")
        checks["active_lock_rejected"] = _expect_error(
            lambda: _recover_stale_lock(lock, protocol_sha256)) and lock.exists()
        lock.unlink()

        dead_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "kind": "phase983_orchestrator",
            "pid": 2_147_483_640,
            "process_start_token": "proved-old-process",
            "protocol_sha256": protocol_sha256,
        }
        dead = {**dead_payload, "lock_sha256": core.sha256_json(dead_payload)}
        lock.write_text(core.canonical_json(dead) + "\n", encoding="utf-8")
        archived = _recover_stale_lock(lock, protocol_sha256)
        checks["proved_stale_lock_archived"] = archived.is_file() and not lock.exists()

        tampered = {**dead, "pid": dead["pid"] - 1}
        lock.write_text(core.canonical_json(tampered) + "\n", encoding="utf-8")
        checks["tampered_lock_rejected"] = _expect_error(
            lambda: _recover_stale_lock(lock, protocol_sha256))
        lock.unlink(missing_ok=True)

        rows_path = root / "rows.jsonl"
        rows_path.write_bytes(b"{}\n" * core.EXPECTED_ROWS_PER_MODEL)
        identity = rows_file_identity(rows_path, True)
        manifest_payload = {
            key: None for key in MANIFEST_KEYS
            if key not in {"manifest_sha256", "created_at_utc"}
        }
        manifest_payload.update({
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "model_key": "qwen3",
            "model_order_index": 0,
            "protocol_sha256": protocol_sha256,
            "admission_sha256": admission_sha256,
            "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
        })
        manifest = {
            **manifest_payload,
            "manifest_sha256": core.sha256_json(manifest_payload),
            "created_at_utc": "2026-07-18T00:00:00+00:00",
        }
        cells = {
            arm: {str(stream): core.ITEM_COUNT for stream in core.STREAMS}
            for arm in core.ARMS
        }
        status_payload = {
            key: None for key in STATUS_KEYS
            if key not in {"status_sha256", "updated_at_utc"}
        }
        status_payload.update({
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "model_key": "qwen3",
            "protocol_sha256": protocol_sha256,
            "admission_sha256": admission_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "complete": True,
            "completed_rows": core.EXPECTED_ROWS_PER_MODEL,
            "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
            "cell_counts": cells,
            "elapsed_seconds_current_process": 1.0,
            **identity,
            "generation_performed": True,
            "model_weights_loaded": True,
            "gpu_used": True,
            "decision_computed": False,
            "holdout": False,
            "mechanism": False,
        })
        status = {
            **status_payload,
            "status_sha256": core.sha256_json(status_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        protocol = {"protocol_sha256": protocol_sha256}
        admission = {"admission_sha256": admission_sha256}
        _verify_model_status_documents(
            "qwen3", protocol, admission, manifest, status, identity,
        )
        checks["status_binds_protocol_manifest_and_rows"] = True

        wrong_status_payload = {
            **status_payload, "admission_sha256": "d" * 64,
        }
        wrong_status = {
            **wrong_status_payload,
            "status_sha256": core.sha256_json(wrong_status_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        checks["self_consistent_wrong_admission_rejected"] = _expect_error(
            lambda: _verify_model_status_documents(
                "qwen3", protocol, admission, manifest, wrong_status, identity))
        extra_manifest_payload = {
            **manifest_payload, "unknown": "must-fail",
        }
        extra_manifest = {
            **extra_manifest_payload,
            "manifest_sha256": core.sha256_json(extra_manifest_payload),
            "created_at_utc": "2026-07-18T00:00:00+00:00",
        }
        checks["self_hashed_extra_manifest_key_rejected"] = _expect_error(
            lambda: _verify_model_status_documents(
                "qwen3", protocol, admission, extra_manifest, status, identity))
        rows_path.write_bytes(rows_path.read_bytes() + b"{}\n")
        changed_identity = rows_file_identity(rows_path, True)
        checks["changed_rows_file_rejected"] = _expect_error(
            lambda: _verify_model_status_documents(
                "qwen3", protocol, admission, manifest, status, changed_identity))

        snapshot = {
            "index": 0,
            "name": "self-test GPU",
            "memory_total_mib": 16_000,
            "memory_used_mib": 1_000,
            "memory_free_mib": 15_000,
        }
        current_statuses = {
            model_key: {
                **status,
                "status_sha256": str(index + 1) * 64,
            }
            for index, model_key in enumerate(core.MODEL_ORDER)
        }
        records = [{
            "model_key": model_key,
            "returncode": 0,
            "elapsed_seconds_this_invocation": 1.0,
            "status_sha256": current_statuses[model_key]["status_sha256"],
            "rows_file_sha256": status["rows_file_sha256"],
            "rows_file_bytes": status["rows_file_bytes"],
            "rows_file_line_count": status["rows_file_line_count"],
            "gpu_before": snapshot,
            "gpu_after_child_exit": snapshot,
            "separate_subprocess": True,
            "complete": True,
        } for model_key in core.MODEL_ORDER]
        orchestrator_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "protocol_sha256": protocol_sha256,
            "admission_sha256": admission_sha256,
            "strict_model_order": list(core.MODEL_ORDER),
            "models": records,
            "completed_model_count": len(core.MODEL_ORDER),
            "complete": True,
            "elapsed_seconds_current_orchestration": 3.0,
            "one_model_subprocess_at_a_time": True,
            "decision_computed": False,
            "holdout": False,
            "mechanism": False,
        }
        orchestrator_document = {
            **orchestrator_payload,
            "orchestrator_status_sha256": core.sha256_json(orchestrator_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        _verify_complete_orchestrator_document(
            orchestrator_document, protocol, admission, current_statuses)
        checks["complete_orchestrator_status_is_read_only_verifiable"] = True

        changed_records = [dict(record) for record in records]
        changed_records[0]["rows_file_sha256"] = "f" * 64
        changed_payload = {**orchestrator_payload, "models": changed_records}
        changed_document = {
            **changed_payload,
            "orchestrator_status_sha256": core.sha256_json(changed_payload),
            "updated_at_utc": "2026-07-18T00:00:00+00:00",
        }
        checks["self_hashed_orchestrator_row_drift_rejected"] = _expect_error(
            lambda: _verify_complete_orchestrator_document(
                changed_document, protocol, admission, current_statuses))

    core.require(all(checks.values()), f"orchestrator self-test failed: {checks}")
    return {
        "schema_version": core.SCHEMA_VERSION,
        "tests": checks,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written_outside_temporary_directory": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
