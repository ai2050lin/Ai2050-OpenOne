#!/usr/bin/env python3
"""Run the sealed, dataset-free Phase 983 engineering qualification.

Each model is loaded in a fresh subprocess in the strict order Qwen3, GLM4,
DeepSeek7B.  The smoke compares compact and dense cache engines token-for-token
on fixed engineering prompts that are not members of the formal dataset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402
import phase983_cross_model_protocol as protocol_builder  # noqa: E402


ENGINE_PATH = GLM5 / "phase983_cross_model_engine.py"
PARSER_SELF_TEST_KEYS = (
    "log_prefix_valid_object_accepted",
    "duplicate_key_rejected",
    "nonfinite_constant_rejected",
    "trailing_nonwhitespace_rejected",
    "serialized_negative_map_order_independent",
)
SMOKE_NEGATIVE_TEST_KEYS = (
    "wrong_model_rejected",
    "formal_result_rejected",
    "dense_compact_mismatch_rejected",
    "non_int8_rejected",
    "cpu_offload_rejected",
    "chat_template_identity_rejected",
    "artifact_identity_rejected",
    "artifact_path_rejected",
    "extra_loaded_identity_field_rejected",
)
LOADED_IDENTITY_KEYS = {
    "schema_version", "model_key", "model_order_index", "artifact_identity",
    "architecture", "model_type", "model_class", "model_class_declares_sdpa",
    "model_forward_has_logits_to_keep", "tokenizer_class", "tokenizer_length",
    "chat_template_sha256", "all_special_ids", "native_generation_prefill",
    "native_single_user_probe", "eos_identity", "pad_token_id",
    "planned_quantization", "weights_loaded", "gpu_used", "loaded_model_class",
    "loaded_attn_implementation", "loaded_quantization", "input_device",
    "hf_device_map", "cuda_only_no_cpu_or_disk_offload",
}
ARTIFACT_IDENTITY_KEYS = {
    "logical_name", "repo_id", "local_dir", "small_files",
    "weight_file_registry", "weight_file_count", "weight_total_bytes",
    "weight_note", "engineering_identity_sha256",
}
QUANTIZATION_IDENTITY_KEYS = {
    "backend", "load_in_8bit", "llm_int8_enable_fp32_cpu_offload",
    "non_quantized_dtype", "device_map", "attn_implementation",
    "local_files_only", "model_reports_loaded_in_8bit",
    "quantizer_reports_load_in_8bit", "linear8bitlt_module_count",
    "floating_parameter_dtypes",
}
EXPECTED_REPO_IDS = {
    "qwen3": "Qwen/Qwen3-4B",
    "glm4": "zai-org/glm-4-9b-chat-hf",
    "deepseek7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_smoke_negative_test_map(value: Any, label: str) -> None:
    core.require(
        isinstance(value, dict)
        and set(value) == set(SMOKE_NEGATIVE_TEST_KEYS)
        and all(item is True for item in value.values()),
        f"{label} negative tests changed",
    )


def authenticate_protocol() -> dict[str, Any]:
    document = core.load_json(core.PROTOCOL_PATH, "Phase983 protocol")
    core.verify_self_hash(document, "protocol_sha256", "created_at_utc",
                          "Phase983 protocol")
    payload = core.without_fields(document, "protocol_sha256", "created_at_utc")
    protocol_builder.verify_payload(payload)
    execution = document.get("execution_contract", {})
    core.require(execution.get("engineering_qualification_authorized") is True
                 and execution.get("formal_generation_authorized") is False,
                 "protocol does not authorize only engineering qualification")
    core.require(document["script_seals"]["engine"]["sha256"]
                 == core.sha256_file(ENGINE_PATH), "sealed engine changed")
    return document


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30, check=False,
    )
    core.require(completed.returncode == 0, "nvidia-smi GPU snapshot failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    core.require(len(lines) == 1, "qualification requires exactly one visible GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    core.require(len(fields) == 5, "unexpected nvidia-smi snapshot schema")
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
    }


def _parse_engine_stdout(stdout: str) -> dict[str, Any]:
    positions = [index for index in range(len(stdout)) if stdout.startswith("{", index)]
    for index in reversed(positions):
        try:
            value = json.loads(
                stdout[index:], object_pairs_hook=core._pairs_no_duplicates,
                parse_constant=core._reject_constant,
            )
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("engine subprocess produced no parseable JSON object")


def parser_self_tests() -> dict[str, bool]:
    fixture = {"engineering_parser_fixture": True, "value": 1}
    accepted = _parse_engine_stdout(
        "library diagnostic before result\n" + core.canonical_json(fixture) + "\n"
    ) == fixture

    def rejected(text: str) -> bool:
        try:
            _parse_engine_stdout(text)
        except RuntimeError:
            return True
        return False

    serialized_negative = json.loads(json.dumps(
        {key: True for key in SMOKE_NEGATIVE_TEST_KEYS}, sort_keys=True,
    ))
    try:
        validate_smoke_negative_test_map(
            serialized_negative, "serialized round-trip")
    except RuntimeError:
        serialized_order_independent = False
    else:
        serialized_order_independent = True
    tests = {
        "log_prefix_valid_object_accepted": accepted,
        "duplicate_key_rejected": rejected('{"x":1,"x":2}'),
        "nonfinite_constant_rejected": rejected('{"x":NaN}'),
        "trailing_nonwhitespace_rejected": rejected('{"x":1}\ntrailing'),
        "serialized_negative_map_order_independent": serialized_order_independent,
    }
    core.require(tuple(tests) == PARSER_SELF_TEST_KEYS and all(tests.values()),
                 "qualification stdout parser self-test failed")
    return tests


def validate_smoke(
    result: Any, model_key: str, protocol: dict[str, Any],
) -> None:
    core.require(isinstance(result, dict), f"{model_key} smoke result missing")
    core.require(set(result) == {
        "schema_version", "model_key", "model_identity", "comparison",
        "engineering_smoke_only", "formal_result", "files_written", "gpu_used",
    } and result.get("schema_version") == core.SCHEMA_VERSION
                 and result.get("model_key") == model_key
                 and result.get("engineering_smoke_only") is True
                 and result.get("formal_result") is False
                 and result.get("files_written") is False
                 and result.get("gpu_used") is True,
                 f"{model_key} smoke scope changed")
    comparison = result.get("comparison")
    core.require(isinstance(comparison, dict) and set(comparison) == {
        "model_key", "exact_token_match", "row_count", "max_new_tokens",
        "sampling_contract", "rows", "engineering_smoke_only",
    }
                 and comparison.get("model_key") == model_key
                 and comparison.get("exact_token_match") is True
                 and comparison.get("row_count") == 8
                 and comparison.get("max_new_tokens") == 24
                 and comparison.get("sampling_contract") == core.SAMPLING
                 and comparison.get("engineering_smoke_only") is True,
                 f"{model_key} compact/dense qualification failed")
    rows = comparison.get("rows")
    expected_row_keys = {
        "item_id", "seed_key", "arm", "stream", "exact_token_match",
        "first_mismatch_index", "compact_token_count", "dense_token_count",
        "compact_tokens_sha256", "dense_tokens_sha256",
    }
    core.require(isinstance(rows, list) and len(rows) == 8,
                 f"{model_key} smoke row denominator changed")
    for index, row in enumerate(rows):
        core.require(
            isinstance(row, dict) and set(row) == expected_row_keys
            and row.get("item_id") == f"engineering_smoke_{index:02d}"
            and row.get("seed_key") == row.get("item_id")
            and row.get("arm") == "engineering_smoke"
            and row.get("stream") == index
            and row.get("exact_token_match") is True
            and row.get("first_mismatch_index") is None
            and isinstance(row.get("compact_token_count"), int)
            and not isinstance(row["compact_token_count"], bool)
            and 0 < row["compact_token_count"] <= 24
            and row.get("dense_token_count") == row["compact_token_count"]
            and _is_sha256(row.get("compact_tokens_sha256"))
            and row.get("dense_tokens_sha256") == row["compact_tokens_sha256"],
            f"{model_key} smoke row equivalence failed: {index}",
        )
    core.require(all(row.get("exact_token_match") is True for row in rows),
                 f"{model_key} smoke row equivalence failed")
    identity = result.get("model_identity")
    quantization = identity.get("loaded_quantization") if isinstance(identity, dict) else None
    planned_quantization = {
        "backend": "bitsandbytes",
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "non_quantized_dtype": "torch.bfloat16",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    core.require(isinstance(identity, dict) and set(identity) == LOADED_IDENTITY_KEYS
                 and identity.get("model_key") == model_key
                 and identity.get("weights_loaded") is True
                 and identity.get("gpu_used") is True
                 and identity.get("loaded_attn_implementation") == "sdpa"
                 and isinstance(quantization, dict)
                 and set(quantization) == QUANTIZATION_IDENTITY_KEYS
                 and identity.get("planned_quantization") == planned_quantization
                 and {key: quantization.get(key) for key in planned_quantization}
                 == planned_quantization
                 and isinstance(quantization.get("model_reports_loaded_in_8bit"), bool)
                 and isinstance(quantization.get("quantizer_reports_load_in_8bit"), bool)
                 and (quantization["model_reports_loaded_in_8bit"]
                      or quantization["quantizer_reports_load_in_8bit"])
                 and isinstance(quantization.get("linear8bitlt_module_count"), int)
                 and not isinstance(quantization["linear8bitlt_module_count"], bool)
                 and quantization["linear8bitlt_module_count"] > 0
                 and quantization.get("floating_parameter_dtypes")
                 == ["torch.bfloat16"],
                 f"{model_key} load/quantization identity changed")
    frozen_tokenizer = protocol.get("tokenizer_adapters", {}).get(model_key)
    core.require(
        isinstance(frozen_tokenizer, dict)
        and identity.get("tokenizer_length") == frozen_tokenizer.get("tokenizer_length")
        and identity.get("chat_template_sha256")
        == frozen_tokenizer.get("chat_template_sha256")
        and identity.get("all_special_ids") == frozen_tokenizer.get("all_special_ids")
        and identity.get("native_generation_prefill")
        == frozen_tokenizer.get("native_generation_prefill")
        and identity.get("pad_token_id") == frozen_tokenizer.get("effective_pad_token_id")
        and identity.get("eos_identity", {}).get("effective_eos_token_ids")
        == frozen_tokenizer.get("effective_eos_token_ids"),
        f"{model_key} loaded tokenizer/EOS identity differs from preregistration",
    )
    artifact = identity.get("artifact_identity")
    frozen_artifact = protocol.get("model_artifact_identities", {}).get(model_key)
    core.require(isinstance(artifact, dict) and set(artifact) == ARTIFACT_IDENTITY_KEYS
                 and isinstance(frozen_artifact, dict)
                 and artifact.get("logical_name") == model_key
                 and artifact.get("repo_id") == EXPECTED_REPO_IDS[model_key]
                 and artifact.get("local_dir")
                 == str((core.ROOT / core.MODEL_PATHS[model_key]).resolve()),
                 f"{model_key} loaded artifact identity missing")
    core.require(artifact.get("engineering_identity_sha256")
                 == core.sha256_json(core.without_fields(
                     artifact, "engineering_identity_sha256")),
                 f"{model_key} engineering artifact self-hash changed")
    expected_small = {
        name: {"size_bytes": seal["bytes"], "sha256": seal["sha256"]}
        for name, seal in frozen_artifact["files"].items()
        if name in {"config.json", "generation_config.json", "tokenizer_config.json",
                    "model.safetensors.index.json"}
    }
    expected_weights = [
        {"name": name, "size_bytes": seal["bytes"]}
        for name, seal in sorted(frozen_artifact["files"].items())
        if name.endswith(".safetensors")
    ]
    core.require(
        artifact.get("small_files") == expected_small
        and artifact.get("weight_file_registry") == expected_weights
        and artifact.get("weight_file_count") == len(expected_weights)
        and artifact.get("weight_total_bytes")
        == sum(value["size_bytes"] for value in expected_weights),
        f"{model_key} loaded artifact registry differs from preregistration",
    )
    device_map = identity.get("hf_device_map")
    core.require(isinstance(device_map, dict) and device_map,
                 f"{model_key} device map missing")
    core.require(all(isinstance(value, str)
                     and re.fullmatch(r"cuda:\d+", value) is not None
                     for value in device_map.values())
                 and len(set(device_map.values())) == 1
                 and identity.get("input_device") == next(iter(device_map.values()))
                 and identity.get("cuda_only_no_cpu_or_disk_offload") is True,
                 f"{model_key} used forbidden CPU/disk offload")


def run_model_subprocess(
    model_key: str, protocol: dict[str, Any],
) -> dict[str, Any]:
    before = gpu_snapshot()
    start = time.time()
    environment = os.environ.copy()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [sys.executable, str(ENGINE_PATH), "--gpu-smoke", model_key],
        cwd=str(core.ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="strict", env=environment, timeout=2 * 60 * 60, check=False,
    )
    elapsed = time.time() - start
    after = gpu_snapshot()
    stdout = completed.stdout
    stderr = completed.stderr
    record: dict[str, Any] = {
        "model_key": model_key,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "gpu_before": before,
        "gpu_after_subprocess_exit": after,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stderr_tail": stderr[-2000:],
        "separate_subprocess": True,
    }
    try:
        result = _parse_engine_stdout(stdout)
        validate_smoke(result, model_key, protocol)
        core.require(completed.returncode == 0, f"{model_key} smoke process failed")
        # A child process cannot retain a CUDA allocation after exit.  The
        # snapshot is still kept to expose any unrelated/external GPU use.
        record.update({"passed": True, "result": result, "error": None})
    except Exception as exc:
        record.update({
            "passed": False,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": stdout[-2000:],
        })
    return record


def negative_tests(
    smoke: dict[str, Any], model_key: str, protocol: dict[str, Any],
) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    mutations = {
        "wrong_model_rejected": lambda value: value.__setitem__("model_key", "other"),
        "formal_result_rejected": lambda value: value.__setitem__("formal_result", True),
        "dense_compact_mismatch_rejected": lambda value: value[
            "comparison"].__setitem__("exact_token_match", False),
        "non_int8_rejected": lambda value: value["model_identity"][
            "loaded_quantization"].__setitem__("load_in_8bit", False),
        "cpu_offload_rejected": lambda value: value["model_identity"][
            "hf_device_map"].__setitem__("bad", "cpu"),
        "chat_template_identity_rejected": lambda value: value["model_identity"].__setitem__(
            "chat_template_sha256", "0" * 64),
        "artifact_identity_rejected": lambda value: value["model_identity"][
            "artifact_identity"].__setitem__("logical_name", "other"),
        "artifact_path_rejected": lambda value: value["model_identity"][
            "artifact_identity"].__setitem__("local_dir", "D:/wrong"),
        "extra_loaded_identity_field_rejected": lambda value: value[
            "model_identity"].__setitem__("extra", True),
    }
    for name, mutate in mutations.items():
        candidate = deepcopy(smoke)
        mutate(candidate)
        try:
            validate_smoke(candidate, model_key, protocol)
        except (RuntimeError, KeyError, TypeError):
            tests[name] = True
        else:
            tests[name] = False
    core.require(tuple(tests) == SMOKE_NEGATIVE_TEST_KEYS and all(tests.values()),
                 "qualification negative test failed")
    return tests


def verify_existing(document: dict[str, Any], protocol: dict[str, Any]) -> None:
    core.verify_self_hash(document, "qualification_sha256", "created_at_utc",
                          "engineering qualification")
    expected_keys = {
        "schema_version", "phase", "experiment", "protocol_sha256",
        "protocol_file_sha256", "engine_script_sha256", "model_order", "models",
        "qualification_passed", "parser_self_tests", "lock_self_tests",
        "negative_tests",
        "formal_dataset_used", "formal_generation_performed",
        "engineering_smoke_only", "separate_model_subprocesses",
        "one_model_resident_at_a_time", "gpu_used", "holdout", "mechanism",
        "qualification_sha256", "created_at_utc",
    }
    core.require(set(document) == expected_keys,
                 "qualification top-level schema changed")
    core.require(document.get("schema_version") == core.SCHEMA_VERSION
                 and document.get("phase") == core.PHASE
                 and document.get("experiment") == core.EXPERIMENT
                 and document.get("protocol_sha256") == protocol["protocol_sha256"]
                 and document.get("protocol_file_sha256")
                 == core.sha256_file(core.PROTOCOL_PATH)
                 and document.get("engine_script_sha256")
                 == protocol["script_seals"]["engine"]["sha256"]
                 == core.sha256_file(ENGINE_PATH)
                 and document.get("model_order") == list(core.MODEL_ORDER)
                 and document.get("formal_dataset_used") is False
                 and document.get("formal_generation_performed") is False
                 and document.get("engineering_smoke_only") is True
                 and document.get("separate_model_subprocesses") is True
                 and document.get("one_model_resident_at_a_time") is True
                 and document.get("gpu_used") is True
                 and document.get("holdout") is False
                 and document.get("mechanism") is False,
                 "qualification lineage/scope changed")
    expected_parser_tests = parser_self_tests()
    core.require(document.get("parser_self_tests") == expected_parser_tests,
                 "qualification parser evidence changed")
    core.require(document.get("lock_self_tests") == lock_self_tests(),
                 "qualification stale-lock evidence changed")
    records = document.get("models")
    core.require(isinstance(records, list) and len(records) == len(core.MODEL_ORDER),
                 "qualification model denominator changed")
    core.require([record.get("model_key") for record in records]
                 == list(core.MODEL_ORDER), "qualification order changed")
    negative = document.get("negative_tests")
    core.require(isinstance(negative, dict), "qualification negative tests missing")

    def verify_snapshot(snapshot: Any, label: str) -> None:
        core.require(isinstance(snapshot, dict) and set(snapshot) == {
            "index", "name", "memory_total_mib", "memory_used_mib", "memory_free_mib",
        }, f"{label} GPU snapshot schema changed")
        core.require(isinstance(snapshot["index"], int)
                     and not isinstance(snapshot["index"], bool)
                     and snapshot["index"] >= 0
                     and isinstance(snapshot["name"], str) and bool(snapshot["name"])
                     and all(isinstance(snapshot[field], int)
                             and not isinstance(snapshot[field], bool)
                             and snapshot[field] >= 0 for field in (
                                 "memory_total_mib", "memory_used_mib", "memory_free_mib")),
                     f"{label} GPU snapshot values changed")

    for record in records:
        passed_record = record.get("passed") is True
        record_keys = {
            "model_key", "returncode", "elapsed_seconds", "gpu_before",
            "gpu_after_subprocess_exit", "stdout_sha256", "stderr_sha256",
            "stderr_tail", "separate_subprocess", "passed", "result", "error",
        }
        if not passed_record:
            record_keys.add("stdout_tail")
        core.require(isinstance(record, dict) and set(record) == record_keys,
                     f"qualification record schema changed: {record.get('model_key')}")
        core.require(isinstance(record.get("returncode"), int)
                     and not isinstance(record["returncode"], bool)
                     and core.finite_number(record.get("elapsed_seconds"), "elapsed") >= 0
                     and _is_sha256(record.get("stdout_sha256"))
                     and _is_sha256(record.get("stderr_sha256"))
                     and isinstance(record.get("stderr_tail"), str)
                     and record.get("separate_subprocess") is True,
                     f"qualification process evidence changed: {record.get('model_key')}")
        verify_snapshot(record.get("gpu_before"), f"{record['model_key']} before")
        verify_snapshot(record.get("gpu_after_subprocess_exit"),
                        f"{record['model_key']} after")
        if record.get("passed") is True:
            core.require(record.get("returncode") == 0 and record.get("error") is None,
                         f"{record['model_key']} passing process evidence inconsistent")
            validate_smoke(record.get("result"), str(record["model_key"]), protocol)
            model_tests = negative.get(record["model_key"])
            validate_smoke_negative_test_map(
                model_tests, str(record["model_key"]))
        else:
            core.require(record.get("result") is None
                         and isinstance(record.get("error"), str) and bool(record["error"])
                         and isinstance(record.get("stdout_tail"), str),
                         f"{record['model_key']} failure evidence changed")
    core.require(set(negative) == {
        str(record["model_key"]) for record in records if record.get("passed") is True
    }, "qualification negative-test model registry changed")
    expected_pass = all(record.get("passed") is True for record in records)
    core.require(document.get("qualification_passed") is expected_pass,
                 "qualification decision inconsistent with records")


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
        return ("dead", None) if error == 87 else ("unknown", f"windows-error:{error}")
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
                 "cannot authenticate current qualification process")
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "kind": "phase983_engineering_qualification",
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
        raise RuntimeError("qualification lock recovery already active") from exc


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
    core.require(_is_sha256(protocol_sha256),
                 "qualification stale-lock protocol hash invalid")
    guard = path.with_name(f"{path.name}.recovery")
    guard_descriptor = _acquire_recovery_guard(guard)
    try:
        document = core.load_json(path, "Phase983 qualification lock")
        core.require(document.get("lock_sha256") == core.sha256_json(
            core.without_fields(document, "lock_sha256")),
            "qualification lock self-hash invalid")
        core.require(set(document) == {
            "schema_version", "kind", "pid", "process_start_token",
            "protocol_sha256", "lock_sha256",
        } and document.get("schema_version") == core.SCHEMA_VERSION
                     and document.get("kind") == "phase983_engineering_qualification"
                     and document.get("protocol_sha256") == protocol_sha256
                     and isinstance(document.get("process_start_token"), str)
                     and bool(document["process_start_token"]),
                     "qualification lock schema/protocol changed")
        pid = document.get("pid")
        core.require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
                     "qualification lock PID invalid")
        state, observed_token = process_probe(pid)
        core.require(state != "unknown",
                     f"cannot prove whether qualification PID {pid} is stale")
        if state == "alive" and observed_token == document["process_start_token"]:
            raise RuntimeError(f"active qualification PID {pid} exists")
        stale = path.with_name(
            f"{path.name}.stale.{pid}.{os.getpid()}.{time.time_ns()}.json")
        os.replace(path, stale)
        return stale
    finally:
        _release_recovery_guard(guard_descriptor)


def acquire_lock(protocol_sha256: str) -> int:
    core.QUALIFICATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    core.require(_is_sha256(protocol_sha256),
                 "qualification lock protocol hash invalid")
    document = _lock_document(protocol_sha256)
    for attempt in range(2):
        try:
            return core.atomic_publish_lock(
                core.QUALIFICATION_LOCK_PATH, document)
        except FileExistsError:
            core.require(attempt == 0, "qualification lock reappeared")
            _recover_stale_lock(core.QUALIFICATION_LOCK_PATH, protocol_sha256)
    raise RuntimeError("could not acquire qualification lock")  # pragma: no cover


def recover_stale_lock_if_present(protocol_sha256: str) -> Path | None:
    """Reject a live qualification owner or archive a proved-dead lock."""
    if not core.QUALIFICATION_LOCK_PATH.exists():
        return None
    return _recover_stale_lock(
        core.QUALIFICATION_LOCK_PATH, protocol_sha256)


def lock_self_tests() -> dict[str, bool]:
    protocol_sha256 = "a" * 64

    def rejected(callable_value: Any) -> bool:
        try:
            callable_value()
        except (RuntimeError, OSError, ValueError):
            return True
        return False

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="phase983_qualification_lock_") as temporary:
        root = Path(temporary)
        guard_path = root / "qualification.lock.recovery"
        guard_descriptor = _acquire_recovery_guard(guard_path)
        try:
            checks["concurrent_recovery_guard_rejected"] = rejected(
                lambda: _acquire_recovery_guard(guard_path))
        finally:
            _release_recovery_guard(guard_descriptor)

        lock = root / "qualification.lock"
        active = _lock_document(protocol_sha256)
        lock.write_text(core.canonical_json(active) + "\n", encoding="utf-8")
        checks["active_pid_lock_rejected"] = (
            rejected(lambda: _recover_stale_lock(lock, protocol_sha256))
            and lock.exists()
        )
        lock.unlink()

        dead_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "kind": "phase983_engineering_qualification",
            "pid": 2_147_483_640,
            "process_start_token": "proved-old-process",
            "protocol_sha256": protocol_sha256,
        }
        dead = {**dead_payload, "lock_sha256": core.sha256_json(dead_payload)}
        lock.write_text(core.canonical_json(dead) + "\n", encoding="utf-8")
        archived = _recover_stale_lock(lock, protocol_sha256)
        checks["proved_dead_pid_lock_archived"] = archived.is_file() and not lock.exists()

        foreign_payload = {**dead_payload, "protocol_sha256": "b" * 64}
        foreign = {**foreign_payload, "lock_sha256": core.sha256_json(foreign_payload)}
        lock.write_text(core.canonical_json(foreign) + "\n", encoding="utf-8")
        checks["foreign_protocol_lock_rejected"] = rejected(
            lambda: _recover_stale_lock(lock, protocol_sha256)) and lock.exists()
        lock.unlink()

        tampered = {**dead, "pid": dead["pid"] - 1}
        lock.write_text(core.canonical_json(tampered) + "\n", encoding="utf-8")
        checks["tampered_lock_rejected"] = rejected(
            lambda: _recover_stale_lock(lock, protocol_sha256)) and lock.exists()
        lock.unlink()
    core.require(all(checks.values()), "qualification lock self-test failed")
    return checks


def run(write: bool) -> dict[str, Any]:
    protocol = authenticate_protocol()
    recover_stale_lock_if_present(protocol["protocol_sha256"])
    parser_tests = parser_self_tests()
    stale_lock_tests = lock_self_tests()
    if core.QUALIFICATION_PATH.exists():
        existing = core.load_json(core.QUALIFICATION_PATH, "engineering qualification")
        verify_existing(existing, protocol)
        return {
            "qualification_sha256": existing["qualification_sha256"],
            "qualification_passed": existing["qualification_passed"],
            "existing": True,
            "files_written": False,
        }
    core.require(not core.ADMISSION_PATH.exists(),
                 "formal admission already exists; qualification is frozen")
    core.require(not any(core.manifest_path(model).exists()
                         or core.rows_path(model).exists()
                         or core.status_path(model).exists()
                         for model in core.MODEL_ORDER),
                 "formal output exists before qualification")
    core.require(write, "qualification artifact absent; pass --write to run GPU smoke")
    descriptor = acquire_lock(protocol["protocol_sha256"])
    try:
        records = [run_model_subprocess(model, protocol) for model in core.MODEL_ORDER]
        passed = all(record["passed"] is True for record in records)
        tests = {}
        for record in records:
            if record["passed"] is True:
                tests[record["model_key"]] = negative_tests(
                    record["result"], record["model_key"], protocol)
        payload = {
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
            "engine_script_sha256": core.sha256_file(ENGINE_PATH),
            "model_order": list(core.MODEL_ORDER),
            "models": records,
            "qualification_passed": passed,
            "parser_self_tests": parser_tests,
            "lock_self_tests": stale_lock_tests,
            "negative_tests": tests,
            "formal_dataset_used": False,
            "formal_generation_performed": False,
            "engineering_smoke_only": True,
            "separate_model_subprocesses": True,
            "one_model_resident_at_a_time": True,
            "gpu_used": True,
            "holdout": False,
            "mechanism": False,
        }
        document = {
            **payload,
            "qualification_sha256": core.sha256_json(payload),
            "created_at_utc": core.utc_now(),
        }
        core.atomic_write_json(core.QUALIFICATION_PATH, document)
        installed = core.load_json(
            core.QUALIFICATION_PATH, "installed engineering qualification")
        verify_existing(installed, protocol)
        core.require(installed == document,
                     "installed qualification changed in JSON serialization")
        return {
            "qualification_sha256": installed["qualification_sha256"],
            "qualification_file_sha256": core.sha256_file(core.QUALIFICATION_PATH),
            "qualification_passed": passed,
            "existing": False,
            "files_written": True,
        }
    finally:
        os.close(descriptor)
        core.QUALIFICATION_LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.write), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
