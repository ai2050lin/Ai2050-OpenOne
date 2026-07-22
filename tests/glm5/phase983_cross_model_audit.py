#!/usr/bin/env python3
"""Independent CPU audit and scientific decision for Phase 983."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


# The audit may load tokenizers later, but CUDA/model weights remain hidden.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402


GATE_PATH = GLM5 / "phase983_cross_model_gate.py"
ENGINE_NAMESPACE = "phase983-cross-model-engine-v1"

ROW_KEYS = frozenset({
    "schema_version", "phase", "experiment", "protocol_sha256",
    "admission_sha256", "manifest_sha256", "model_key", "id", "seed_key",
    "semantic_id", "task", "difficulty", "gold_label", "swap_variant",
    "arm", "arm_spec", "stream", "pair_id", "pair_seed", "batch_index",
    "effective_user_prompt", "rendered_prefix_sha256", "input_ids",
    "prompt_len", "generated_ids", "generated_plain", "first_eos_token_id",
    "first_eos_absorbing", "checkpoints", "decision_terminal_state",
    "max_new_tokens", "sampling", "compact_active_rows",
    "private_generator_per_row", "same_pair_seed_across_arms",
    "same_pair_seed_across_option_swap_twins",
    "generation_performed", "decision_computed", "holdout",
    "holdout_loaded", "mechanism", "mechanism_authorized", "row_sha256",
})
MANIFEST_KEYS = frozenset({
    "schema_version", "phase", "experiment", "model_key",
    "model_order_index", "protocol_sha256", "protocol_file_sha256",
    "admission_sha256", "admission_file_sha256", "qualification_sha256",
    "dataset_file_sha256", "dataset_audit_content_sha256",
    "model_artifact_identity_sha256", "loaded_model_identity",
    "loaded_tokenizer_identity",
    "runtime_versions", "eos_token_ids", "pad_token_id", "arms", "streams",
    "sampling", "quantization", "batch_size", "checkpoints",
    "max_new_tokens", "expected_rows", "dataset_namespace", "engine",
    "script_seals", "dependency_seals", "creation_state", "holdout",
    "holdout_loaded", "mechanism", "mechanism_authorized", "manifest_sha256",
    "created_at_utc",
})
STATUS_KEYS = frozenset({
    "schema_version", "phase", "experiment", "model_key", "protocol_sha256",
    "admission_sha256", "manifest_sha256", "completed_rows", "expected_rows",
    "cell_counts", "complete", "elapsed_seconds_current_process",
    "rows_file_bytes", "rows_file_sha256", "rows_file_line_count",
    "rows_file_terminal_newline", "generation_performed", "model_weights_loaded",
    "gpu_used", "decision_computed", "holdout", "mechanism", "status_sha256",
    "updated_at_utc",
})
LOADED_IDENTITY_KEYS = frozenset({
    "schema_version", "model_key", "model_order_index", "artifact_identity",
    "architecture", "model_type", "model_class", "model_class_declares_sdpa",
    "model_forward_has_logits_to_keep", "tokenizer_class", "tokenizer_length",
    "chat_template_sha256", "all_special_ids", "native_generation_prefill",
    "native_single_user_probe", "eos_identity",
    "pad_token_id", "planned_quantization", "weights_loaded", "gpu_used",
    "loaded_model_class", "loaded_attn_implementation", "loaded_quantization",
    "input_device", "hf_device_map", "cuda_only_no_cpu_or_disk_offload",
})
RUNTIME_VERSION_KEYS = frozenset({
    "python", "torch", "transformers", "bitsandbytes", "platform",
    "torch_cuda", "cudnn", "cuda_device_index", "cuda_device_name",
    "cuda_compute_capability", "cuda_total_memory_bytes",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
    "deterministic_algorithms", "sdpa_flash_enabled",
    "sdpa_memory_efficient_enabled", "sdpa_math_enabled",
})
ENGINE_KEYS = frozenset({
    "compact_active_rows", "dynamic_cache_batch_select_indices",
    "dense_reference_forbidden_in_formal_run", "private_generator_per_row",
    "same_item_stream_seed_across_arms",
    "same_semantic_twin_seed_across_option_surfaces",
    "two_by_two_arm_option_crn_block", "arm_excluded_from_seed",
    "swap_side_excluded_from_seed", "model_namespace_in_seed", "first_eos_absorbing",
    "one_longest_rollout_for_checkpoints",
})
CREATION_STATE_KEYS = frozenset({
    "model_weights_loaded", "gpu_used", "generation_performed",
    "decision_computed",
})
_INDEPENDENT_FINAL_RE = re.compile(r"(?:\A|\n)FINAL: ([AB])\Z")
EXPECTED_REPO_IDS = {
    "qwen3": "Qwen/Qwen3-4B",
    "glm4": "zai-org/glm-4-9b-chat-hf",
    "deepseek7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}


def raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    core.require(isinstance(value, dict), f"{label} must be an object")
    observed = set(value)
    core.require(observed == set(expected), (
        f"{label} schema changed; missing={sorted(set(expected) - observed)}, "
        f"extra={sorted(observed - set(expected))}"
    ))


def strict_int_list(value: Any, label: str) -> list[int]:
    core.require(isinstance(value, list) and value, f"{label} must be a nonempty list")
    core.require(all(isinstance(item, int) and not isinstance(item, bool)
                     and item >= 0 for item in value),
                 f"{label} contains a non-integer/negative token ID")
    return [int(item) for item in value]


def normalize_optional_eos(value: Any, label: str) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    core.require(all(isinstance(item, int) and not isinstance(item, bool)
                     and item >= 0 for item in values),
                 f"{label} EOS registry is invalid")
    return sorted({int(item) for item in values})


def independent_parse_final_contract(text: str) -> dict[str, Any]:
    """Second implementation: do not call the generation-side core parser."""
    stripped = str(text).strip()
    marker_like_count = len(re.findall(
        r"FINAL\s*:", stripped, flags=re.IGNORECASE))
    exact = _INDEPENDENT_FINAL_RE.search(stripped)
    return {
        "plain_text": stripped,
        "marker_like_count": marker_like_count,
        "exact_terminal_marker": exact is not None,
        "parsed_label": exact.group(1) if exact is not None else None,
        "protocol_valid": exact is not None and marker_like_count == 1,
    }


def independent_analyze_ids(
    tokenizer: Any, item: dict[str, Any], ids_value: Any,
    eos_ids_value: Any, budget: int,
) -> dict[str, Any]:
    """Independently reconstruct one checkpoint and reject hidden specials."""
    ids = strict_int_list(ids_value, "checkpoint trajectory")
    core.require(isinstance(budget, int) and not isinstance(budget, bool)
                 and budget > 0 and len(ids) <= budget,
                 f"invalid independent checkpoint length {len(ids)}/{budget}")
    eos_ids = strict_int_list(list(eos_ids_value), "frozen EOS registry")
    eos_set = set(eos_ids)
    eos_positions = [index for index, token_id in enumerate(ids)
                     if token_id in eos_set]
    core.require(len(eos_positions) <= 1, "independent audit found multiple EOS tokens")
    has_eos = bool(eos_positions)
    if has_eos:
        core.require(eos_positions == [len(ids) - 1],
                     "independent audit found nonabsorbing EOS")
    content_ids = ids[:-1] if has_eos else ids

    raw_special_ids = getattr(tokenizer, "all_special_ids", None)
    core.require(isinstance(raw_special_ids, (list, tuple, set)),
                 "tokenizer special-token registry is invalid")
    special_ids: set[int] = set()
    for token_id in raw_special_ids:
        core.require(isinstance(token_id, int) and not isinstance(token_id, bool)
                     and token_id >= 0,
                     "tokenizer special-token registry contains an invalid ID")
        special_ids.add(int(token_id))
    unexpected_positions = [
        index for index, token_id in enumerate(content_ids)
        if token_id in special_ids and token_id not in eos_set
    ]
    unexpected_ids = sorted({content_ids[index] for index in unexpected_positions})
    decoded = tokenizer.decode(content_ids, skip_special_tokens=False)
    parsed = independent_parse_final_contract(decoded)
    if unexpected_positions:
        parsed["protocol_valid"] = False
    semantic_match = (
        bool(parsed["protocol_valid"])
        and parsed["parsed_label"] == str(item["answer"])
    )
    if not has_eos:
        state = "C"
    elif not bool(parsed["protocol_valid"]):
        state = "I_protocol"
    elif semantic_match:
        state = "V"
    else:
        state = "I_sem"
    if state != "I_protocol":
        protocol_subtype = None
    elif unexpected_positions:
        protocol_subtype = "EOS_WITH_UNEXPECTED_SPECIAL_TOKEN"
    else:
        protocol_subtype = "EOS_WITH_INVALID_FINAL_CONTRACT"
    if state != "C":
        censor_subtype = None
    elif unexpected_positions:
        censor_subtype = "CENSORED_WITH_UNEXPECTED_SPECIAL_TOKEN"
    elif bool(parsed["protocol_valid"]):
        censor_subtype = "CENSORED_AFTER_EXACT_FINAL"
    elif parsed["marker_like_count"]:
        censor_subtype = "CENSORED_WITH_MALFORMED_OR_NONTERMINAL_FINAL"
    else:
        censor_subtype = "CENSORED_BEFORE_FINAL"
    return {
        "budget": budget,
        "n_tokens": len(ids),
        "has_eos": has_eos,
        "eos_positions": eos_positions,
        "hit_budget": (not has_eos and len(ids) == budget),
        "unexpected_special_count": len(unexpected_positions),
        "unexpected_special_positions": unexpected_positions,
        "unexpected_special_token_ids": unexpected_ids,
        "terminal_state": state,
        "protocol_subtype": protocol_subtype,
        "valid_stop": state == "V",
        "semantic_match": bool(semantic_match),
        "censor_subtype": censor_subtype,
        **parsed,
    }


def independent_analyze_checkpoints(
    tokenizer: Any, item: dict[str, Any], ids_value: Any, eos_ids_value: Any,
) -> dict[str, Any]:
    ids = strict_int_list(ids_value, "full generated trajectory")
    eos_ids = strict_int_list(list(eos_ids_value), "frozen EOS registry")
    core.require(len(ids) <= core.MAX_NEW_TOKENS,
                 "full generated trajectory exceeds the frozen horizon")
    eos_set = set(eos_ids)
    first_eos = next((index for index, token_id in enumerate(ids)
                      if token_id in eos_set), None)
    if first_eos is not None:
        trimmed = ids[:first_eos + 1]
    else:
        trimmed = ids
    output: dict[str, Any] = {}
    for checkpoint in core.CHECKPOINTS:
        prefix = trimmed if len(trimmed) <= checkpoint else trimmed[:checkpoint]
        output[str(checkpoint)] = independent_analyze_ids(
            tokenizer, item, prefix, eos_ids, checkpoint)
    return output


def independent_pair_seed(
    protocol_sha256: str, model_key: str, seed_key: str, stream: int,
) -> int:
    core.require(isinstance(protocol_sha256, str) and len(protocol_sha256) == 64,
                 "invalid independent seed namespace")
    core.require(model_key in core.MODEL_ORDER, "invalid independent seed model")
    core.require(isinstance(seed_key, str) and seed_key.strip(),
                 "invalid independent seed key")
    core.require(isinstance(stream, int) and not isinstance(stream, bool)
                 and stream in core.STREAMS, "invalid independent seed stream")
    payload = {
        "dataset_namespace": protocol_sha256,
        "engine_namespace": ENGINE_NAMESPACE,
        "item_id": seed_key,
        "model_key": model_key,
        "stream": stream,
    }
    value = int.from_bytes(
        hashlib.sha256(core.canonical_json(payload).encode("utf-8")).digest()[:8],
        "big",
    )
    return int(value % (2**31 - 1))


def verify_model_files(protocol: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in core.MODEL_ORDER:
        identity = protocol["model_artifact_identities"][model_key]
        core.require(isinstance(identity, dict) and set(identity) == {
            "model_key", "relative_path", "files", "weight_file_count",
            "weight_bytes", "weights_loaded", "gpu_accessed", "identity_sha256",
        }, f"model artifact identity schema changed: {model_key}")
        root = core.ROOT / identity["relative_path"]
        expected_root = (core.ROOT / core.MODEL_PATHS[model_key]).resolve()
        core.require(identity.get("model_key") == model_key
                     and identity.get("relative_path") == core.MODEL_PATHS[model_key]
                     and root.resolve() == expected_root,
                     f"model artifact root changed: {model_key}")
        expected_names = {
            "config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json", "model.safetensors.index.json",
            *[path.name for path in root.glob("*.safetensors")],
        }
        expected_names.update(
            name for name in ("vocab.json", "merges.txt")
            if (root / name).is_file())
        core.require(set(identity.get("files", {})) == expected_names,
                     f"model artifact file registry changed: {model_key}")
        verified = 0
        total_bytes = 0
        for name, seal in identity["files"].items():
            path = root / name
            core.require(isinstance(seal, dict)
                         and set(seal) == {"bytes", "sha256"}
                         and isinstance(seal["bytes"], int)
                         and not isinstance(seal["bytes"], bool)
                         and isinstance(seal["sha256"], str)
                         and len(seal["sha256"]) == 64
                         and path.is_file() and path.stat().st_size == seal["bytes"]
                         and core.sha256_file(path) == seal["sha256"],
                         f"model file changed: {model_key}/{name}")
            verified += 1
            total_bytes += seal["bytes"]
        core.require(identity["identity_sha256"]
                      == core.sha256_json(core.without_fields(
                          identity, "identity_sha256")),
                      f"model identity self-hash invalid: {model_key}")
        weight_files = [name for name in identity["files"]
                        if name.endswith(".safetensors")]
        core.require(identity["weight_file_count"] == len(weight_files)
                     and identity["weight_bytes"] == sum(
                         identity["files"][name]["bytes"] for name in weight_files)
                     and identity["weights_loaded"] is False
                     and identity["gpu_accessed"] is False,
                     f"model artifact identity counters/flags changed: {model_key}")
        output[model_key] = {
            "identity_sha256": identity["identity_sha256"],
            "verified_file_count": verified,
            "verified_bytes": total_bytes,
        }
    return output


def _expected_engine_contract() -> dict[str, bool]:
    return {
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
    }


def _expected_planned_quantization() -> dict[str, Any]:
    return {
        "backend": "bitsandbytes",
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "non_quantized_dtype": "torch.bfloat16",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }


def validate_loaded_identity_static(
    loaded: Any, model_key: str, protocol: dict[str, Any],
) -> None:
    require_exact_keys(loaded, LOADED_IDENTITY_KEYS,
                       f"{model_key} loaded model identity")
    frozen_tok = protocol["tokenizer_adapters"][model_key]
    expected_eos = frozen_tok["effective_eos_token_ids"]
    config_path = core.ROOT / core.MODEL_PATHS[model_key] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_architecture = list(config.get("architectures", []) or [])
    core.require(
        loaded["schema_version"] == core.SCHEMA_VERSION
        and loaded["model_key"] == model_key
        and loaded["model_order_index"] == core.MODEL_ORDER.index(model_key)
        and loaded["architecture"] == expected_architecture
        and loaded["model_type"] == config.get("model_type")
        and loaded["model_class"] in expected_architecture
        and loaded["loaded_model_class"] == loaded["model_class"]
        and loaded["model_class_declares_sdpa"] is True
        and loaded["model_forward_has_logits_to_keep"] is True
        and loaded["tokenizer_class"] == frozen_tok["tokenizer_class"]
        and loaded["tokenizer_length"] == frozen_tok["tokenizer_length"]
        and loaded["pad_token_id"] == frozen_tok["effective_pad_token_id"]
        and loaded["planned_quantization"] == _expected_planned_quantization()
        and loaded["weights_loaded"] is True
        and loaded["gpu_used"] is True
        and loaded["loaded_attn_implementation"] == "sdpa"
        and loaded["cuda_only_no_cpu_or_disk_offload"] is True,
        f"{model_key} loaded identity static contract changed",
    )

    artifact = loaded["artifact_identity"]
    artifact_keys = frozenset({
        "logical_name", "repo_id", "local_dir", "small_files",
        "weight_file_registry", "weight_file_count", "weight_total_bytes",
        "weight_note", "engineering_identity_sha256",
    })
    require_exact_keys(artifact, artifact_keys,
                       f"{model_key} engineering artifact identity")
    identity_without_hash = {
        key: value for key, value in artifact.items()
        if key != "engineering_identity_sha256"
    }
    core.require(artifact["engineering_identity_sha256"]
                 == core.sha256_json(identity_without_hash),
                 f"{model_key} engineering identity self-hash invalid")
    formal_files = protocol["model_artifact_identities"][model_key]["files"]
    expected_small = {
        name: {"size_bytes": formal_files[name]["bytes"],
               "sha256": formal_files[name]["sha256"]}
        for name in (
            "config.json", "generation_config.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )
    }
    expected_weights = [
        {"name": name, "size_bytes": seal["bytes"]}
        for name, seal in sorted(formal_files.items())
        if name.endswith(".safetensors")
    ]
    expected_root = (core.ROOT / core.MODEL_PATHS[model_key]).resolve()
    core.require(
        artifact["logical_name"] == model_key
        and artifact["repo_id"] == EXPECTED_REPO_IDS[model_key]
        and Path(artifact["local_dir"]).resolve() == expected_root
        and artifact["small_files"] == expected_small
        and artifact["weight_file_registry"] == expected_weights
        and artifact["weight_file_count"] == len(expected_weights)
        and artifact["weight_total_bytes"]
        == sum(value["size_bytes"] for value in expected_weights),
        f"{model_key} engineering artifact lineage changed",
    )

    eos_identity = loaded["eos_identity"]
    core.require(isinstance(eos_identity, dict)
                 and set(eos_identity) == {
                     "sources", "effective_eos_token_ids", "multiple_effective_eos"
                 }
                 and eos_identity["effective_eos_token_ids"] == expected_eos
                 and eos_identity["multiple_effective_eos"]
                 is (len(expected_eos) > 1),
                 f"{model_key} loaded EOS identity changed")
    sources = eos_identity["sources"]
    core.require(isinstance(sources, dict)
                 and set(sources) == {
                     "tokenizer", "inspected_config",
                     "inspected_generation_config", "loaded_model_config",
                     "loaded_model_generation_config",
                 }
                 and sorted({token_id for values in sources.values()
                             for token_id in values}) == expected_eos,
                 f"{model_key} loaded EOS sources changed")

    quant = loaded["loaded_quantization"]
    expected_quant_keys = set(_expected_planned_quantization()) | {
        "model_reports_loaded_in_8bit", "quantizer_reports_load_in_8bit",
        "linear8bitlt_module_count", "floating_parameter_dtypes",
    }
    core.require(isinstance(quant, dict) and set(quant) == expected_quant_keys,
                 f"{model_key} loaded quantization schema changed")
    core.require(
        all(quant[key] == value
            for key, value in _expected_planned_quantization().items())
        and (quant["model_reports_loaded_in_8bit"] is True
             or quant["quantizer_reports_load_in_8bit"] is True)
        and isinstance(quant["linear8bitlt_module_count"], int)
        and not isinstance(quant["linear8bitlt_module_count"], bool)
        and quant["linear8bitlt_module_count"] > 0
        and quant["floating_parameter_dtypes"] == ["torch.bfloat16"],
        f"{model_key} loaded quantization evidence changed",
    )
    device_map = loaded["hf_device_map"]
    input_device = loaded["input_device"]
    core.require(isinstance(input_device, str)
                 and re.fullmatch(r"cuda:\d+", input_device) is not None
                 and isinstance(device_map, dict) and device_map
                 and all(isinstance(name, str) and str(device) == input_device
                         for name, device in device_map.items()),
                 f"{model_key} CUDA-only placement evidence changed")


def validate_manifest_contract(
    manifest: dict[str, Any], model_key: str, protocol: dict[str, Any],
    qualification: dict[str, Any], admission: dict[str, Any],
    dataset_audit: dict[str, Any],
) -> None:
    require_exact_keys(manifest, MANIFEST_KEYS, f"{model_key} manifest")
    loaded = manifest["loaded_model_identity"]
    validate_loaded_identity_static(loaded, model_key, protocol)
    loaded_tokenizer = manifest["loaded_tokenizer_identity"]
    core.require(isinstance(loaded_tokenizer, dict),
                 f"{model_key} loaded tokenizer identity missing")
    runtime = manifest["runtime_versions"]
    string_runtime_keys = {
        "python", "torch", "transformers", "bitsandbytes", "platform",
        "torch_cuda", "cudnn", "cuda_device_name",
    }
    bool_runtime_keys = {
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
        "deterministic_algorithms", "sdpa_flash_enabled",
        "sdpa_memory_efficient_enabled", "sdpa_math_enabled",
    }
    capability = runtime.get("cuda_compute_capability") if isinstance(runtime, dict) else None
    core.require(
        isinstance(runtime, dict) and set(runtime) == set(RUNTIME_VERSION_KEYS)
        and all(isinstance(runtime[key], str) and runtime[key]
                for key in string_runtime_keys)
        and isinstance(runtime["cuda_device_index"], int)
        and not isinstance(runtime["cuda_device_index"], bool)
        and runtime["cuda_device_index"] >= 0
        and isinstance(capability, list) and len(capability) == 2
        and all(isinstance(value, int) and not isinstance(value, bool)
                and value >= 0 for value in capability)
        and isinstance(runtime["cuda_total_memory_bytes"], int)
        and not isinstance(runtime["cuda_total_memory_bytes"], bool)
        and runtime["cuda_total_memory_bytes"] > 0
        and all(isinstance(runtime[key], bool) for key in bool_runtime_keys)
        and runtime["cuda_matmul_allow_tf32"] is False
        and runtime["cudnn_allow_tf32"] is False
        and runtime["cudnn_benchmark"] is False
        and any(runtime[key] for key in {
            "sdpa_flash_enabled", "sdpa_memory_efficient_enabled",
            "sdpa_math_enabled"}),
        f"{model_key} runtime-version/determinism schema changed",
    )
    expected_payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "model_key": model_key,
        "model_order_index": core.MODEL_ORDER.index(model_key),
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
        "admission_sha256": admission["admission_sha256"],
        "admission_file_sha256": core.sha256_file(core.ADMISSION_PATH),
        "qualification_sha256": qualification["qualification_sha256"],
        "dataset_file_sha256": core.sha256_file(core.DATASET_PATH),
        "dataset_audit_content_sha256": core.sha256_json(dataset_audit),
        "model_artifact_identity_sha256": protocol[
            "model_artifact_identities"][model_key]["identity_sha256"],
        "loaded_model_identity": loaded,
        "loaded_tokenizer_identity": loaded_tokenizer,
        "runtime_versions": runtime,
        "eos_token_ids": protocol["tokenizer_adapters"][model_key][
            "effective_eos_token_ids"],
        "pad_token_id": protocol["tokenizer_adapters"][model_key][
            "effective_pad_token_id"],
        "arms": core.ARMS,
        "streams": list(core.STREAMS),
        "sampling": core.SAMPLING,
        "quantization": core.QUANTIZATION,
        "batch_size": core.BATCH_SIZE,
        "checkpoints": list(core.CHECKPOINTS),
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
        "dataset_namespace": protocol["protocol_sha256"],
        "engine": _expected_engine_contract(),
        "script_seals": protocol["script_seals"],
        "dependency_seals": protocol["dependency_seals"],
        "creation_state": {
            "model_weights_loaded": True, "gpu_used": True,
            "generation_performed": False, "decision_computed": False,
        },
        "holdout": False,
        "holdout_loaded": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    core.require(core.without_fields(
        manifest, "manifest_sha256", "created_at_utc") == expected_payload,
        f"{model_key} manifest differs from full independent reconstruction")
    core.require(manifest["manifest_sha256"] == core.sha256_json(expected_payload)
                 and isinstance(manifest["created_at_utc"], str)
                 and manifest["created_at_utc"],
                 f"{model_key} manifest hash/timestamp invalid")


def authenticate_documents() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    protocol = core.load_json(core.PROTOCOL_PATH, "Phase983 protocol")
    qualification = core.load_json(core.QUALIFICATION_PATH, "engineering qualification")
    admission = core.load_json(core.ADMISSION_PATH, "Phase983 admission")
    orchestrator = core.load_json(
        core.ORCHESTRATOR_STATUS_PATH, "Phase983 orchestrator status")
    dataset = core.load_json(core.DATASET_PATH, "Phase983 dataset")
    dataset_audit = core.load_json(core.DATASET_AUDIT_PATH, "Phase983 dataset audit")
    core.verify_self_hash(protocol, "protocol_sha256", "created_at_utc",
                          "Phase983 protocol")
    core.verify_self_hash(qualification, "qualification_sha256", "created_at_utc",
                          "engineering qualification")
    core.verify_self_hash(admission, "admission_sha256", "created_at_utc",
                          "Phase983 admission")
    core.verify_self_hash(orchestrator, "orchestrator_status_sha256", "updated_at_utc",
                          "Phase983 orchestrator status")
    core.require(protocol.get("model_order") == list(core.MODEL_ORDER)
                 and protocol.get("arms") == core.ARMS
                 and protocol.get("sampling") == core.SAMPLING
                 and protocol.get("quantization") == core.QUANTIZATION
                 and protocol.get("expected_rows_all_models")
                 == core.EXPECTED_ROWS_ALL_MODELS,
                 "protocol scientific grid changed")
    core.require(qualification.get("qualification_passed") is True
                 and qualification.get("formal_dataset_used") is False
                 and qualification.get("formal_generation_performed") is False,
                 "engineering qualification boundary failed")
    core.require(admission.get("admitted") is True
                 and admission.get("protocol_sha256") == protocol["protocol_sha256"]
                 and admission.get("qualification_sha256")
                 == qualification["qualification_sha256"],
                 "admission lineage changed")
    core.require(orchestrator.get("complete") is True
                 and orchestrator.get("completed_model_count")
                 == len(core.MODEL_ORDER)
                 and orchestrator.get("strict_model_order") == list(core.MODEL_ORDER)
                 and orchestrator.get("decision_computed") is False,
                 "orchestrator did not complete without a decision")
    core.require(not core.ORCHESTRATOR_LOCK_PATH.exists()
                 and not core.QUALIFICATION_LOCK_PATH.exists()
                 and not any(core.run_lock_path(model).exists()
                             for model in core.MODEL_ORDER),
                 "a generation/qualification lock remains")
    core.verify_file_seals(protocol.get("script_seals"), core.SCRIPT_PATHS,
                           "Phase983 script")
    core.verify_file_seals(protocol.get("dependency_seals"), core.DEPENDENCY_PATHS,
                           "Phase983 dependency")
    core.require(core.sha256_file(core.DATASET_PATH)
                 == protocol["dataset"]["dataset_file_sha256"]
                 and core.sha256_file(core.DATASET_AUDIT_PATH)
                 == protocol["dataset"]["dataset_audit_file_sha256"]
                 and core.sha256_json(dataset)
                 == protocol["dataset"]["dataset_content_sha256"]
                 and core.sha256_json(dataset_audit)
                 == protocol["dataset"]["dataset_audit_content_sha256"],
                 "dataset lineage changed")
    items = dataset.get("items")
    core.require(isinstance(items, list) and len(items) == core.ITEM_COUNT,
                 "dataset denominator changed")
    model_file_audit = verify_model_files(protocol)
    return (protocol, qualification, admission, orchestrator, dataset,
            dataset_audit, items, model_file_audit)


def load_model_rows(
    model_key: str, protocol: dict[str, Any], admission: dict[str, Any],
    qualification: dict[str, Any], dataset_audit: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = core.load_json(core.manifest_path(model_key), f"{model_key} manifest")
    status = core.load_json(core.status_path(model_key), f"{model_key} status")
    core.verify_self_hash(manifest, "manifest_sha256", "created_at_utc",
                          f"{model_key} manifest")
    core.verify_self_hash(status, "status_sha256", "updated_at_utc",
                          f"{model_key} status")
    validate_manifest_contract(
        manifest, model_key, protocol, qualification, admission, dataset_audit)
    require_exact_keys(status, STATUS_KEYS, f"{model_key} status")
    core.require(status.get("model_key") == model_key
                 and status.get("schema_version") == core.SCHEMA_VERSION
                 and status.get("phase") == core.PHASE
                 and status.get("experiment") == core.EXPERIMENT
                 and status.get("protocol_sha256") == protocol["protocol_sha256"]
                 and status.get("admission_sha256") == admission["admission_sha256"]
                 and status.get("manifest_sha256") == manifest["manifest_sha256"]
                 and status.get("complete") is True
                 and status.get("completed_rows") == core.EXPECTED_ROWS_PER_MODEL
                 and status.get("expected_rows") == core.EXPECTED_ROWS_PER_MODEL
                 and status.get("decision_computed") is False
                 and status.get("holdout") is False
                 and status.get("mechanism") is False,
                 f"{model_key} status changed")

    path = core.rows_path(model_key)
    core.require(path.is_file() and path.stat().st_size > 0,
                 f"{model_key} rows file missing/empty")
    with path.open("rb") as binary_handle:
        binary_handle.seek(-1, os.SEEK_END)
        terminal_newline = binary_handle.read(1) == b"\n"
    core.require(terminal_newline, f"{model_key} rows file is truncated")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(
                    line, object_pairs_hook=core._pairs_no_duplicates,
                    parse_constant=core._reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid {model_key} JSONL line {line_number}") from exc
            core.require(isinstance(row, dict) and set(row) == set(ROW_KEYS)
                         and row.get("row_sha256")
                         == core.sha256_json(core.without_fields(row, "row_sha256")),
                         f"{model_key} row self-hash invalid at {line_number}")
            key = core.row_key(row)
            core.require(key not in seen, f"{model_key} duplicate row key: {key}")
            seen.add(key)
            rows.append(row)
    expected = {
        (str(item["id"]), arm, stream)
        for item in items for arm in core.ARMS for stream in core.STREAMS
    }
    core.require(len(rows) == core.EXPECTED_ROWS_PER_MODEL and seen == expected,
                 f"{model_key} row grid incomplete")
    counts = {
        arm: {str(stream): 0 for stream in core.STREAMS} for arm in core.ARMS
    }
    for row in rows:
        counts[str(row["arm"])][str(row["stream"])] += 1
    elapsed = status["elapsed_seconds_current_process"]
    core.require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                 and math.isfinite(float(elapsed)) and float(elapsed) >= 0.0,
                 f"{model_key} status elapsed time invalid")
    status_payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "model_key": model_key,
        "protocol_sha256": protocol["protocol_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_rows": core.EXPECTED_ROWS_PER_MODEL,
        "expected_rows": core.EXPECTED_ROWS_PER_MODEL,
        "cell_counts": counts,
        "complete": True,
        "elapsed_seconds_current_process": elapsed,
        "rows_file_bytes": path.stat().st_size,
        "rows_file_sha256": core.sha256_file(path),
        "rows_file_line_count": len(rows),
        "rows_file_terminal_newline": True,
        "generation_performed": True,
        "model_weights_loaded": True,
        "gpu_used": True,
        "decision_computed": False,
        "holdout": False,
        "mechanism": False,
    }
    core.require(core.without_fields(status, "status_sha256", "updated_at_utc")
                 == status_payload
                 and status["status_sha256"] == core.sha256_json(status_payload)
                 and isinstance(status["updated_at_utc"], str)
                 and status["updated_at_utc"],
                 f"{model_key} status differs from rows reconstruction")
    return manifest, status, rows


def verify_exact_row(row: dict[str, Any], expected_payload: dict[str, Any], label: str) -> None:
    expected = {
        **expected_payload,
        "row_sha256": core.sha256_json(expected_payload),
    }
    require_exact_keys(row, ROW_KEYS, label)
    core.require(row == expected, f"{label} differs from full reconstruction")


def rehashed_row_negative_tests(
    row: dict[str, Any], expected_payload: dict[str, Any],
) -> dict[str, bool]:
    def mutate_first_eos(value: dict[str, Any]) -> None:
        value["first_eos_token_id"] = (
            -1 if value["first_eos_token_id"] is None else None)

    def mutate_checkpoint(value: dict[str, Any]) -> None:
        checkpoint = value["checkpoints"][str(core.DECISION_CHECKPOINT)]
        checkpoint["terminal_state"] = (
            "C" if checkpoint["terminal_state"] != "C" else "V")

    mutations = {
        "phase_rehash_rejected": lambda value: value.__setitem__("phase", 982),
        "admission_rehash_rejected": lambda value: value.__setitem__(
            "admission_sha256", "0" * 64),
        "seed_key_rehash_rejected": lambda value: value.__setitem__(
            "seed_key", "p983_seed_tampered"),
        "first_eos_rehash_rejected": mutate_first_eos,
        "sampling_rehash_rejected": lambda value: value["sampling"].__setitem__(
            "temperature", 0.7),
        "checkpoint_rehash_rejected": mutate_checkpoint,
        "extra_field_rehash_rejected": lambda value: value.__setitem__(
            "unregistered", True),
    }
    tests: dict[str, bool] = {}
    for name, mutate in mutations.items():
        candidate = deepcopy(row)
        mutate(candidate)
        candidate["row_sha256"] = core.sha256_json(core.without_fields(
            candidate, "row_sha256"))
        try:
            verify_exact_row(candidate, expected_payload, "synthetic rehashed row")
        except RuntimeError:
            tests[name] = True
        else:
            tests[name] = False
    core.require(all(tests.values()), f"rehashed row negative tests failed: {tests}")
    return tests


def tokenizer_recompute(
    model_key: str, protocol: dict[str, Any], manifest: dict[str, Any],
    rows: list[dict[str, Any]], items: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from transformers import AutoTokenizer

    root = core.ROOT / core.MODEL_PATHS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    frozen = protocol["tokenizer_adapters"][model_key]
    template = getattr(tokenizer, "chat_template", None)
    core.require(type(tokenizer).__name__ == frozen["tokenizer_class"]
                 and len(tokenizer) == frozen["tokenizer_length"]
                 and int(tokenizer.pad_token_id) == frozen["effective_pad_token_id"]
                 and isinstance(template, str) and template
                 and raw_sha256(template) == frozen["chat_template_sha256"],
                 f"{model_key} tokenizer identity changed")
    loaded_identity = manifest["loaded_model_identity"]
    core.require(loaded_identity["chat_template_sha256"] == raw_sha256(template),
                 f"{model_key} loaded chat-template hash changed")
    loaded_tokenizer_keys = (
        "tokenizer_class", "tokenizer_length", "tokenizer_eos_token_id",
        "effective_pad_token_id", "effective_eos_token_ids", "all_special_ids",
        "unexpected_special_token_ids", "chat_template_sha256",
        "native_generation_prefill", "probe", "native_thinking_switch_used",
    )
    expected_loaded_tokenizer = {
        key: frozen[key] for key in loaded_tokenizer_keys
    }
    core.require(manifest["loaded_tokenizer_identity"]
                 == expected_loaded_tokenizer,
                 f"{model_key} runtime tokenizer identity differs from protocol")
    inspection_text = "ENGINEERING INSPECTION ONLY: return A."
    inspection_rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": inspection_text}],
        tokenize=False, add_generation_prompt=True,
    )
    inspection_ids = tokenizer(
        inspection_rendered, add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    core.require(loaded_identity["native_single_user_probe"] == {
        "rendered_sha256": raw_sha256(inspection_rendered),
        "input_token_count": len(inspection_ids),
    }, f"{model_key} native inspection rendering changed")
    frozen_eos = strict_int_list(
        frozen["effective_eos_token_ids"], f"{model_key} frozen EOS IDs")
    actual_special = sorted({
        int(value) for value in tokenizer.all_special_ids
        if isinstance(value, int) and not isinstance(value, bool)
    })
    core.require(
        int(tokenizer.eos_token_id) == frozen["tokenizer_eos_token_id"]
        and actual_special == frozen["all_special_ids"]
        and loaded_identity["all_special_ids"] == actual_special
        and loaded_identity["native_generation_prefill"]
        == frozen["native_generation_prefill"]
        and sorted(set(actual_special) - set(frozen_eos))
        == frozen["unexpected_special_token_ids"]
        and set(frozen_eos).issubset(actual_special),
        f"{model_key} tokenizer special/EOS registry changed",
    )
    config_json = json.loads((root / "config.json").read_text(encoding="utf-8"))
    generation_json = json.loads(
        (root / "generation_config.json").read_text(encoding="utf-8"))
    expected_loaded_eos_sources = {
        "tokenizer": [int(tokenizer.eos_token_id)],
        "inspected_config": normalize_optional_eos(
            config_json.get("eos_token_id"), f"{model_key} config"),
        "inspected_generation_config": normalize_optional_eos(
            generation_json.get("eos_token_id"), f"{model_key} generation config"),
        "loaded_model_config": normalize_optional_eos(
            config_json.get("eos_token_id"), f"{model_key} loaded config"),
        "loaded_model_generation_config": normalize_optional_eos(
            generation_json.get("eos_token_id"),
            f"{model_key} loaded generation config"),
    }
    core.require(loaded_identity["eos_identity"]["sources"]
                 == expected_loaded_eos_sources,
                 f"{model_key} loaded EOS source registries changed")
    core.require(manifest["eos_token_ids"] == frozen_eos,
                 f"{model_key} manifest EOS differs from frozen protocol")
    item_by_id = {str(item["id"]): item for item in items}
    positions = {
        (str(item["id"]), arm, stream): index
        for index, (item, arm, stream) in enumerate(core.canonical_grid(items))
    }
    gate_rows: list[dict[str, str]] = []
    termination = Counter()
    token_total = 0
    crn_cells: defaultdict[tuple[str, int], dict[tuple[str, str], int]] = defaultdict(dict)
    row_negative_tests: dict[str, bool] | None = None
    for row in rows:
        key = core.row_key(row)
        item = item_by_id[key[0]]
        arm, stream = key[1], key[2]
        user, rendered, input_ids = core.render_prefix(tokenizer, item, arm)
        generated = strict_int_list(row.get("generated_ids"),
                                    f"{model_key} generated IDs")
        independent = independent_analyze_checkpoints(
            tokenizer, item, generated, frozen_eos)
        core.require(all(
            not (checkpoint["unexpected_special_count"] > 0
                 and checkpoint["terminal_state"] == "V")
            for checkpoint in independent.values()
        ), f"{model_key} unexpected special token entered V: {key}")
        shared_core = core.analyze_checkpoints(
            tokenizer, item, generated, frozen_eos)
        core.require(shared_core == independent,
                     f"{model_key} shared/independent checkpoint disagreement: {key}")
        plain = tokenizer.decode(generated, skip_special_tokens=False).strip()
        seed = independent_pair_seed(
            protocol["protocol_sha256"], model_key, str(item["seed_key"]), stream)
        eos_positions = [index for index, token_id in enumerate(generated)
                         if token_id in set(frozen_eos)]
        core.require((eos_positions == [len(generated) - 1])
                     or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
                     f"{model_key} generated trajectory violates EOS/horizon: {key}")
        first_eos = generated[-1] if eos_positions else None
        final = independent[str(core.DECISION_CHECKPOINT)]
        state = str(final["terminal_state"])
        expected_payload = {
            "schema_version": core.SCHEMA_VERSION,
            "phase": core.PHASE,
            "experiment": core.EXPERIMENT,
            "protocol_sha256": protocol["protocol_sha256"],
            "admission_sha256": manifest["admission_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "model_key": model_key,
            "id": str(item["id"]),
            "seed_key": str(item["seed_key"]),
            "semantic_id": str(item["semantic_id"]),
            "task": str(item["task"]),
            "difficulty": str(item["difficulty"]),
            "gold_label": str(item["answer"]),
            "swap_variant": str(item["swap_side"]),
            "arm": arm,
            "arm_spec": core.ARMS[arm],
            "stream": stream,
            "pair_id": core.pair_id(model_key, str(item["id"]), stream),
            "pair_seed": seed,
            "batch_index": positions[key] // core.BATCH_SIZE + 1,
            "effective_user_prompt": user,
            "rendered_prefix_sha256": raw_sha256(rendered),
            "input_ids": input_ids,
            "prompt_len": len(input_ids),
            "generated_ids": generated,
            "generated_plain": plain,
            "first_eos_token_id": first_eos,
            "first_eos_absorbing": True,
            "checkpoints": independent,
            "decision_terminal_state": state,
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
        verify_exact_row(row, expected_payload, f"{model_key} row {key}")
        if row_negative_tests is None:
            row_negative_tests = rehashed_row_negative_tests(row, expected_payload)
        crn_cell = (str(item["swap_side"]), arm)
        crn_group = crn_cells[(str(item["semantic_id"]), stream)]
        core.require(crn_cell not in crn_group,
                     f"{model_key} duplicate 2x2 CRN cell: {key}")
        crn_group[crn_cell] = seed
        token_total += len(generated)
        termination[state] += 1
        gate_rows.append({
            "id": key[0],
            "semantic_id": str(item["semantic_id"]),
            "task": str(item["task"]),
            "difficulty": str(item["difficulty"]),
            "gold_label": str(item["answer"]),
            "swap_variant": str(item["swap_side"]),
            "state": state,
            "arm": arm,
            "stream": str(stream),
        })
    expected_crn_cells = {
        (swap, arm) for swap in core.SWAP_SIDES for arm in core.ARMS
    }
    for group_key, cells in crn_cells.items():
        core.require(set(cells) == expected_crn_cells and len(set(cells.values())) == 1,
                     f"{model_key} 2x2 option-swap/arm CRN group failed: {group_key}")
    core.require(len(crn_cells) == core.SEMANTIC_INSTANCE_COUNT * len(core.STREAMS),
                 f"{model_key} 2x2 CRN group denominator changed")
    core.require(row_negative_tests is not None, "row negative tests were not exercised")
    return gate_rows, {
        "tokenizer_class": type(tokenizer).__name__,
        "verified_rows": len(rows),
        "generated_token_total": token_total,
        "decision_state_counts": dict(sorted(termination.items())),
        "independent_parser_used": True,
        "shared_core_cross_checked_per_checkpoint": True,
        "unexpected_non_eos_special_tokens_cannot_be_V": True,
        "verified_2x2_crn_groups": len(crn_cells),
        "rehashed_row_negative_tests": row_negative_tests,
    }


def validate_gate_contract_alignment(
    frozen: Any, live: Any,
) -> None:
    core.require(isinstance(frozen, dict) and isinstance(live, dict),
                 "gate contract is missing")
    core.require(
        frozen == live
        and live["phase"] == core.PHASE
        and live["models"] == list(core.MODEL_ORDER)
        and live["streams"] == [f"stream_{value}" for value in core.STREAMS]
        and live["states"] == list(core.TERMINAL_STATES)
        and live["denominators"]["per_model_stream_arm"] == core.ITEM_COUNT
        and live["decision"]["pooling_allowed"] is False
        and live["decision"]["secondary_can_set_primary"] is False
        and live["operational_threshold_interpretation"][
            "caller_overrides_allowed"] is False,
        "live gate contract differs byte-for-byte from frozen protocol",
    )


def run_gate_subprocess(
    gate_inputs: dict[str, dict[str, list[list[dict[str, str]]]]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    self_test = subprocess.run(
        [sys.executable, str(GATE_PATH), "--self-test", "--contract"],
        cwd=str(core.ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120, check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    core.require(self_test.returncode == 0, "gate self-test subprocess failed")
    self_report = json.loads(self_test.stdout)
    core.require(all(self_report.get("tests", {}).values()), "gate synthetic test failed")
    current_gate_hash = core.sha256_file(GATE_PATH)
    live_contract = self_report.get("gate_contract")
    core.require(self_report.get("script_sha256") == current_gate_hash
                 and self_report.get("gate_contract_sha256")
                 == core.sha256_json(live_contract),
                 "gate subprocess hash/contract self-report changed")
    validate_gate_contract_alignment(protocol.get("gate_contract"), live_contract)
    code = (
        "import json,sys; import phase983_cross_model_gate as g; "
        "d=json.load(sys.stdin); "
        "print(json.dumps(g.evaluate_cross_models(d),ensure_ascii=False,sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(GLM5),
        input=json.dumps(gate_inputs, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    core.require(completed.returncode == 0,
                 f"gate evaluation failed: {completed.stderr[-1000:]}")
    decision = json.loads(completed.stdout)
    core.require(isinstance(decision, dict)
                 and decision.get("pooling_used") is False
                 and decision.get("secondary_can_set_primary") is False,
                 "gate returned an invalid decision boundary")
    return decision, {
        "script_sha256": current_gate_hash,
        "self_test_script_sha256": self_report["script_sha256"],
        "gate_contract_sha256": self_report["gate_contract_sha256"],
        "synthetic_case_count": self_report["synthetic_case_count"],
        "all_synthetic_tests_passed": True,
        "evaluation_subprocess_imported_transformers": False,
        "pooling_used": False,
        "live_contract_matches_protocol": True,
    }


def descriptive_summaries(
    rows_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key, rows in rows_by_model.items():
        model: dict[str, Any] = {}
        for stream in core.STREAMS:
            stream_output: dict[str, Any] = {}
            for arm in core.ARMS:
                selected = [row for row in rows
                            if row["stream"] == stream and row["arm"] == arm]
                checkpoints = {}
                for checkpoint in core.CHECKPOINTS:
                    counts = Counter(
                        row["checkpoints"][str(checkpoint)]["terminal_state"]
                        for row in selected
                    )
                    checkpoints[str(checkpoint)] = {
                        state: counts[state] for state in core.TERMINAL_STATES
                    }
                stream_output[arm] = checkpoints
            pairs_a = {row["id"]: row for row in rows
                       if row["stream"] == stream and row["arm"] == core.ARM_A}
            pairs_b = {row["id"]: row for row in rows
                       if row["stream"] == stream and row["arm"] == core.ARM_B}
            matrix = core.matrix((
                pairs_a[item_id]["decision_terminal_state"],
                pairs_b[item_id]["decision_terminal_state"],
            ) for item_id in sorted(pairs_a))
            stream_output["decision_transition_matrix"] = matrix
            model[f"stream_{stream}"] = stream_output
        output[model_key] = model
    return output


def build_payload() -> dict[str, Any]:
    (protocol, qualification, admission, orchestrator, _dataset, dataset_audit,
     items, model_file_audit) = authenticate_documents()
    manifests: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    tokenizer_audits: dict[str, Any] = {}
    gate_inputs: dict[str, dict[str, list[list[dict[str, str]]]]] = {}
    source_hashes: dict[str, Any] = {}
    for model_key in core.MODEL_ORDER:
        manifest, status, rows = load_model_rows(
            model_key, protocol, admission, qualification, dataset_audit, items)
        manifests[model_key] = manifest
        statuses[model_key] = status
        rows_by_model[model_key] = rows
        gate_rows, tokenizer_audit = tokenizer_recompute(
            model_key, protocol, manifest, rows, items)
        tokenizer_audits[model_key] = tokenizer_audit
        stream_map: dict[str, list[list[dict[str, str]]]] = {}
        for stream in core.STREAMS:
            baseline = [{key: value for key, value in row.items()
                         if key not in {"arm", "stream"}}
                        for row in gate_rows
                        if row["arm"] == core.ARM_A and row["stream"] == str(stream)]
            candidate = [{key: value for key, value in row.items()
                          if key not in {"arm", "stream"}}
                         for row in gate_rows
                         if row["arm"] == core.ARM_B and row["stream"] == str(stream)]
            stream_map[f"stream_{stream}"] = [baseline, candidate]
        gate_inputs[model_key] = stream_map
        source_hashes[model_key] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": core.sha256_file(core.manifest_path(model_key)),
            "status_sha256": status["status_sha256"],
            "status_file_sha256": core.sha256_file(core.status_path(model_key)),
            "rows_file_sha256": core.sha256_file(core.rows_path(model_key)),
            "row_count": len(rows),
        }
    gate_decision, gate_audit = run_gate_subprocess(gate_inputs, protocol)
    summaries = descriptive_summaries(rows_by_model)
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "integrity_decision": "GO",
        "scientific_decision": (
            "PASS" if gate_decision["cross_model_pass"] else "NO-GO"
        ),
        "cross_model_pass": bool(gate_decision["cross_model_pass"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "qualification_sha256": qualification["qualification_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "orchestrator_status_sha256": orchestrator[
            "orchestrator_status_sha256"],
        "source_hashes": source_hashes,
        "model_file_audit": model_file_audit,
        "tokenizer_recomputation": tokenizer_audits,
        "gate_audit": gate_audit,
        "gate_decision": gate_decision,
        "checkpoint_descriptive_summaries": summaries,
        "accounting_scope": {
            "terminal_states_are_external_bins": True,
            "V_plus_C_plus_I_protocol_plus_I_sem_is_accounting_identity": True,
            "streams_are_seed_robustness_not_independent_samples": True,
            "models_were_not_pooled": True,
            "option_swap_twins_are_paired_not_independent": True,
            "two_by_two_arm_option_crn_verified_per_model_stream_semantic_id": True,
        },
        "claim_boundary": {
            "external_instruction_bundle_only": True,
            "native_thinking_switch_equivalence": False,
            "individual_counterfactual_causality": False,
            "shared_internal_cross_model_mechanism": False,
            "holdout_authorized": False,
            "mechanism_authorized": False,
            "multimodal_generalization": False,
        },
        "runtime_boundary": {
            "audit_cpu_only": True,
            "model_weights_loaded_by_audit": False,
            "gpu_used_by_audit": False,
            "tokenizers_loaded": True,
            "gate_evaluated_in_clean_no_transformers_subprocess": True,
            "decision_computed_only_by_audit": True,
            "terminal_parser_independently_reimplemented": True,
            "checkpoints_independently_reconstructed": True,
            "shared_core_cross_checked_but_not_trusted_as_sole_oracle": True,
            "non_eos_special_tokens_excluded_from_V": True,
        },
        "expected_rows_all_models": core.EXPECTED_ROWS_ALL_MODELS,
        "verified_rows_all_models": sum(len(rows) for rows in rows_by_model.values()),
        "holdout": False,
        "mechanism": False,
    }
    core.require(payload["verified_rows_all_models"] == core.EXPECTED_ROWS_ALL_MODELS,
                 "combined row denominator changed")
    return payload


def verify_report(document: dict[str, Any], expected: dict[str, Any] | None = None) -> None:
    core.verify_self_hash(document, "audit_sha256", "created_at_utc",
                          "Phase983 combined audit")
    if expected is None:
        expected = build_payload()
    core.require(core.without_fields(document, "audit_sha256", "created_at_utc")
                 == expected, "combined audit differs from complete reconstruction")


def negative_report_tests(expected: dict[str, Any]) -> dict[str, bool]:
    base = {
        **expected,
        "audit_sha256": core.sha256_json(expected),
        "created_at_utc": "2000-01-01T00:00:00+00:00",
    }
    tests: dict[str, bool] = {}
    mutations = {
        "decision_flip_rejected": lambda value: value.__setitem__(
            "cross_model_pass", not value["cross_model_pass"]),
        "integrity_flip_rejected": lambda value: value.__setitem__(
            "integrity_decision", "NO-GO"),
        "pooling_claim_rejected": lambda value: value["accounting_scope"].__setitem__(
            "models_were_not_pooled", False),
        "mechanism_open_rejected": lambda value: value["claim_boundary"].__setitem__(
            "mechanism_authorized", True),
        "row_count_rejected": lambda value: value.__setitem__(
            "verified_rows_all_models", core.EXPECTED_ROWS_ALL_MODELS - 1),
        "source_hash_rejected": lambda value: value["source_hashes"][
            "qwen3"].__setitem__("rows_file_sha256", "0" * 64),
        "gate_result_rejected": lambda value: value["gate_decision"].__setitem__(
            "pooling_used", True),
        "gate_contract_alignment_rejected": lambda value: value["gate_audit"].__setitem__(
            "live_contract_matches_protocol", False),
        "independent_parser_claim_rejected": lambda value: value[
            "tokenizer_recomputation"]["qwen3"].__setitem__(
                "independent_parser_used", False),
        "crn_group_count_rejected": lambda value: value[
            "tokenizer_recomputation"]["qwen3"].__setitem__(
                "verified_2x2_crn_groups", 0),
        "gpu_audit_rejected": lambda value: value["runtime_boundary"].__setitem__(
            "gpu_used_by_audit", True),
    }
    for name, mutate in mutations.items():
        candidate = deepcopy(base)
        mutate(candidate)
        candidate["audit_sha256"] = core.sha256_json(core.without_fields(
            candidate, "audit_sha256", "created_at_utc"))
        try:
            verify_report(candidate, expected)
        except RuntimeError:
            tests[name] = True
        else:
            tests[name] = False
    core.require(all(tests.values()), "combined audit negative test failed")
    return tests


def static_self_test() -> dict[str, Any]:
    class FakeTokenizer:
        all_special_ids = [7, 99]

        @staticmethod
        def decode(ids: list[int], skip_special_tokens: bool = False) -> str:
            pieces = {
                1: "work\n", 2: "FINAL: A", 3: "more",
                7: "<EOS>", 99: "<|assistant|>",
            }
            return "".join(
                "" if skip_special_tokens and token_id in {7, 99}
                else pieces[token_id]
                for token_id in ids
            )

    item = {"answer": "A"}
    fake = FakeTokenizer()
    exact = independent_analyze_ids(fake, item, [1, 2, 7], [7], 256)
    hidden_special = independent_analyze_ids(fake, item, [2, 99, 7], [7], 256)
    censored_special = independent_analyze_ids(fake, item, [2, 99, 3], [7], 3)
    try:
        require_exact_keys({"a": 1}, frozenset({"a", "b"}), "missing-key probe")
    except RuntimeError:
        missing_key_rejected = True
    else:
        missing_key_rejected = False
    try:
        require_exact_keys({"a": 1, "b": 2}, frozenset({"a"}), "extra-key probe")
    except RuntimeError:
        extra_key_rejected = True
    else:
        extra_key_rejected = False
    cases = {
        "exact_A": independent_parse_final_contract(
            "work\nFINAL: A")["protocol_valid"],
        "trailing_text_invalid": not independent_parse_final_contract(
            "FINAL: A\nmore")["protocol_valid"],
        "duplicate_invalid": not independent_parse_final_contract(
            "FINAL: A\nFINAL: A")["protocol_valid"],
        "malformed_then_valid_invalid": not independent_parse_final_contract(
            "FINAL: X\nFINAL: A")["protocol_valid"],
        "wrong_case_invalid": not independent_parse_final_contract(
            "final: A")["protocol_valid"],
        "independent_exact_is_V": exact["terminal_state"] == "V",
        "non_eos_special_is_protocol_invalid": (
            hidden_special["terminal_state"] == "I_protocol"
            and hidden_special["protocol_subtype"]
            == "EOS_WITH_UNEXPECTED_SPECIAL_TOKEN"
            and hidden_special["unexpected_special_count"] == 1
        ),
        "censored_special_subtyped": (
            censored_special["terminal_state"] == "C"
            and censored_special["censor_subtype"]
            == "CENSORED_WITH_UNEXPECTED_SPECIAL_TOKEN"
        ),
        "independent_matches_core_exact": exact == core.analyze_ids(
            fake, item, [1, 2, 7], [7], 256),
        "independent_matches_core_special": hidden_special == core.analyze_ids(
            fake, item, [2, 99, 7], [7], 256),
        "missing_schema_key_rejected": missing_key_rejected,
        "extra_schema_key_rejected": extra_key_rejected,
        "seed_arm_absent": independent_pair_seed(
            "a" * 64, "qwen3", "x", 0) == independent_pair_seed(
                "a" * 64, "qwen3", "x", 0),
        "seed_option_swap_absent_via_shared_seed_key": independent_pair_seed(
            "a" * 64, "qwen3", "shared-semantic-seed", 0)
            == independent_pair_seed(
                "a" * 64, "qwen3", "shared-semantic-seed", 0),
        "seed_model_present": independent_pair_seed(
            "a" * 64, "qwen3", "x", 0) != independent_pair_seed(
                "a" * 64, "glm4", "x", 0),
    }
    core.require(all(cases.values()), "audit static self-test failed")
    return {
        "phase": core.PHASE,
        "tests": cases,
        "cpu_only": True,
        "model_weights_loaded": False,
        "gpu_used": False,
        "files_written": False,
    }


def run(write: bool, verify_only: bool) -> dict[str, Any]:
    if not write and not verify_only:
        return static_self_test()
    expected = build_payload()
    tests = negative_report_tests(expected)
    if core.COMBINED_AUDIT_PATH.exists():
        existing = core.load_json(core.COMBINED_AUDIT_PATH, "Phase983 combined audit")
        verify_report(existing, expected)
        return {
            "audit_sha256": existing["audit_sha256"],
            "audit_file_sha256": core.sha256_file(core.COMBINED_AUDIT_PATH),
            "scientific_decision": existing["scientific_decision"],
            "negative_tests": tests,
            "existing": True,
            "files_written": False,
        }
    core.require(write, "combined audit absent; --verify cannot create it")
    document = {
        **expected,
        "audit_sha256": core.sha256_json(expected),
        "created_at_utc": core.utc_now(),
    }
    core.atomic_write_json(core.COMBINED_AUDIT_PATH, document)
    installed = core.load_json(
        core.COMBINED_AUDIT_PATH, "installed Phase983 combined audit")
    verify_report(installed, expected)
    core.require(installed == document,
                 "installed combined audit changed in serialization")
    return {
        "audit_sha256": installed["audit_sha256"],
        "audit_file_sha256": core.sha256_file(core.COMBINED_AUDIT_PATH),
        "scientific_decision": installed["scientific_decision"],
        "cross_model_pass": installed["cross_model_pass"],
        "negative_tests": tests,
        "existing": False,
        "files_written": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        run(args.write, args.verify), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
