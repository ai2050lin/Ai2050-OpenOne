#!/usr/bin/env python3
"""Freeze the fail-closed Phase992 delayed-binding behavior activation.

This is a CPU-only protocol builder.  It seals the four execution/scoring
sources, this protocol source, the qualified Phase991 package, the Phase983
adapter engine, the formal Python executable, and the exact external-behavior
contracts.  It never imports torch/transformers, initializes CUDA, loads model
weights, opens scoring truth, or executes a model.

The resulting activation authorizes *external behavior only*.  Hidden states,
attentions, hooks, causal interventions, scoring before the broker release,
and mechanism/formula claims remain fail-closed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


PHASE = 992
SCHEMA_VERSION = 1
EXPERIMENT = "delayed_two_hop_gpu_behavior"
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
RESULT_ROOT = GLM5 / "result"
OUT = RESULT_ROOT / "phase992_delayed_binding_behavior_protocol"
EXECUTION_OUT = RESULT_ROOT / "phase992_delayed_binding_behavior_execution"

PREREGISTRATION = "protocol_preregistration.json"
AUDIT = "independent_source_audit.json"
FREEZE = "freeze_commit.json"
ACTIVATION = "activation.json"

PHASE991_OUT = RESULT_ROOT / "phase991_delayed_binding_gpu_admission"
PHASE991_FILES = {
    "freeze": "freeze_commit.json",
    "admission": "gpu_admission_preregistration.json",
    "independent_audit": "independent_execution_audit.json",
    "stage_commit": "stage_commit.json",
    "model_manifests": "model_artifact_manifests.json",
    "holdout_commitment": "holdout_access_commitment.json",
}
PHASE991_SELF_HASH_FIELDS = {
    "freeze": "freeze_commit_sha256",
    "admission": "gpu_admission_sha256",
    "independent_audit": "independent_audit_sha256",
    "stage_commit": "stage_commit_sha256",
    "model_manifests": "model_manifest_sha256",
    "holdout_commitment": "holdout_commitment_sha256",
}

SOURCE_PATHS = {
    "protocol": "tests/glm5/phase992_delayed_binding_protocol.py",
    "broker": "tests/glm5/phase992_holdout_broker.py",
    "runner": "tests/glm5/phase992_delayed_binding_runner.py",
    "scorer": "tests/glm5/phase992_delayed_binding_scorer.py",
    "audit": "tests/glm5/phase992_delayed_binding_audit.py",
}
ENGINE_PATH = "tests/glm5/phase983_cross_model_engine.py"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
PUBLIC_SPLITS = ("discovery", "confirmation", "adversarial")
PRIMARY_SCOPE_RECORDS = 8192
HOLDOUT_SCOPE_RECORDS = 2048
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
QUALIFIED_PHASE991_FREEZE_SELF_SHA256 = (
    "a334ad8d6aaa773a30a014c223176dfa07b6008c25e649f1e7c141c559f6d4c5"
)
QUALIFIED_PHASE991_FREEZE_FILE_SHA256 = (
    "33a6e3b667b77e21a49093ce0ee8476cc1682e91a72daac318f2a7c6e3a45b2a"
)

GENERATION_CONTRACT = {
    "scientific_object": "local frozen INT8 checkpoints under the frozen runtime",
    "input_mode": "raw_text_no_chat_template_add_special_tokens_false",
    "prompt_source": "exact Phase991 runtime prompt row; prompt_sha256 must match",
    "chat_template": False,
    "add_special_tokens": False,
    "padding_side": "left",
    "truncation": False,
    "batch_size": 8,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
    "max_new_tokens": 24,
    "output_scores": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict_in_generate": True,
    "quantization": "bitsandbytes_int8",
    "load_in_8bit": True,
    "nonquantized_dtype": "torch.bfloat16",
    "attention_implementation": "sdpa",
    "device_map": "cuda_only_no_cpu_or_disk_entries",
    "cpu_or_disk_offload": False,
    "automatic_fallback": False,
    "one_model_resident_at_a_time": True,
    "model_order": list(MODEL_ORDER),
    "effective_eos": "sorted union of tokenizer, config, and generation-config EOS IDs",
    "first_eos_absorbing": True,
    "decode_scope": "generated continuation through first EOS exclusive",
    "pad_is_not_eos_unless_identical_id": True,
    "phase983_engine_use": (
        "adapter load/release, EOS/pad/quantization/device identity only; "
        "native-chat rendering and stochastic sample_batch are forbidden"
    ),
}

TEACHER_FORCED_CONTRACT = {
    "role": "diagnostic_only_not_replacement_for_natural_generation",
    "same_raw_prompt_as_generation": True,
    "context": "prompt + '\\n' + 'The retrieved marker is'",
    "prefix": "The retrieved marker is",
    "joiner": "\n",
    "candidate_values_in_order": ["red", "blue", "green", "black"],
    "candidate_continuations": {
        "red": " red", "blue": " blue", "green": " green", "black": " black"
    },
    "candidate_count": 4,
    "all_candidates_single_token_each_frozen_tokenizer": True,
    "boundary_identity": "full_ids == context_ids + candidate_ids",
    "score_position": "logits at final context token for first continuation token",
    "stored_values": "four candidate logits plus gold-minus-max-foil margin",
    "margin": "gold candidate logit - max(three foil candidate logits)",
    "no_hidden_states_attentions_or_hooks": True,
}

PARSER_CONTRACT = {
    "normalization": "Unicode text is casefolded; ASCII word-boundary marker matches only",
    "marker_regex": r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])",
    "primary_prediction": "first complete marker match in generated text before EOS",
    "no_marker": "unparsed",
    "multiple_distinct_markers": "first marker scores semantics; ambiguity is separately reported",
    "strict_reference_sentence": "The retrieved marker is {value}.",
    "strict_format_is_secondary": True,
    "eos_and_budget_are_secondary": True,
}

THRESHOLD_CONTRACT = {
    "overall_each_major_condition_min_percent": 90,
    "each_preregistered_variant_min_percent": 85,
    "counterfactual_pair_sensitivity_min_percent": 85,
    "shortcut_baseline_must_be_reported": True,
    "behavior_above_shortcut_is_not_mechanism_evidence": True,
    "public_splits_must_pass_before_sealed_access": True,
    "all_three_primary_models_must_pass_before_extension_model_access": True,
}

SCORING_EXECUTION_CONTRACT = {
    "parser_search_text": "generated continuation before first effective EOS",
    "parser_may_not_use_prompt_or_truth": True,
    "independent_unit": "semantic_world_id",
    "32_variants_per_world_are_paired_not_independent": True,
    "natural_generation_is_primary": True,
    "teacher_forced_is_diagnostic_only": True,
    "no_threshold_tuning_after_any_model_output": True,
    "exact_full_denominators_required": True,
}

ARTIFACT_LAYOUT = {
    "protocol_root": "tests/glm5/result/phase992_delayed_binding_behavior_protocol",
    "execution_root": "tests/glm5/result/phase992_delayed_binding_behavior_execution",
    "activation_is_unique_and_not_copied_to_execution_root": True,
    "primary_raw": "raw/primary/{model}.jsonl.gz",
    "holdout_raw": "raw/holdout/{model}.jsonl.gz",
    "primary_receipt": "receipts/primary_{model}.json",
    "holdout_receipt": "receipts/holdout_{model}.json",
    "cleanup_receipt": "receipts/cleanup_{scope}_{model}.json",
    "public_score": "scores/public_score.json",
    "public_admission": "public_behavior_admission.json",
    "holdout_score": "scores/holdout_score.json",
    "lease": "execution.lease.json",
    "holdout_events": "holdout_access/events/{ordinal}_{model}_{action}.json",
    "holdout_grant_receipt": "holdout_access/grant_{model_index}_{model}.json",
    "holdout_seal_receipt": "holdout_access/seal_{model_index}_{model}.json",
    "holdout_abort_receipt": "holdout_access/abort_{model_index}_{model}.json",
    "holdout_grant_failure_receipt": "holdout_access/grant_failure_{model_index}_{model}.json",
    "holdout_final_chain_receipt": "holdout_access/final_chain_receipt.json",
    "holdout_temporary_copy": "temporary_holdout/{run_id}_{model}.jsonl",
    "forbidden_unsealed_plaintext_raw": True,
}

SERIAL_AND_LEASE_CONTRACT = {
    "exclusive_lease_required": True,
    "lease_created_atomically_no_overwrite": True,
    "lease_identity_fields": [
        "schema_version", "activation_sha256", "run_id", "pid",
        "process_start_token", "scope", "model", "created_at_utc", "lease_sha256",
    ],
    "pid_reuse_resistant_start_token_required": True,
    "unknown_liveness_is_not_dead": True,
    "stale_lease_not_auto_deleted": True,
    "manual_recovery_requires_separate_frozen_recovery_receipt": True,
    "model_order": list(MODEL_ORDER),
    "order_is_mandatory_within_each_scope": True,
    "one_model_subprocess_at_a_time": True,
    "one_model_resident_at_a_time": True,
    "no_next_model_before_cleanup_receipt": True,
    "no_unfrozen_resume": True,
    "oom_or_interruption": "inconclusive_and_fail_closed",
    "source_or_activation_or_artifact_drift": "global_stop_no_next_model",
}

CLEANUP_AND_RESOURCE_CONTRACT = {
    "minimum_free_disk_gib_before_each_model": 80,
    "planned_result_quota_gib": [40, 60],
    "gpu_baseline_before_each_model": True,
    "child_process_must_exit_after_each_model_scope": True,
    "torch_cuda_allocated_and_reserved_bytes_before_exit": 0,
    "strict_release_sequence": [
        "engine_release", "synchronize", "clear_cublas_workspaces",
        "garbage_collect", "empty_cache", "ipc_collect", "empty_cache", "synchronize",
    ],
    "nvidia_smi_post_cleanup_recovery_required": True,
    "post_cleanup_used_mib_max_over_preload_baseline": 512,
    "cleanup_receipt_requires": [
        "model_released", "child_exit_zero", "cuda_allocated_zero",
        "cuda_reserved_zero", "baseline_recovered", "cleanup_pass",
    ],
    "cleanup_failure": "global_stop_no_next_model",
    "cpu_or_disk_offload_forbidden": True,
    "disk_minimum_failure": "global_stop_no_next_model",
}

HOLDOUT_CONTRACT = {
    "holdout_semantics": "preregistered_immutable_not_blind",
    "runner_may_not_read_scoring_truth": True,
    "broker_is_access_mediator_not_OS_sandbox": True,
    "phase_1": (
        "seal all three public-scope raw outputs and cleanup receipts; then and only "
        "then scorer may open public truth"
    ),
    "public_scope_splits": list(PUBLIC_SPLITS),
    "public_scope_records_per_model": PRIMARY_SCOPE_RECORDS,
    "phase_2_release_condition": "all_three_models_public_gate_PASS",
    "phase_2": (
        "after public PASS, broker releases sealed_holdout in model order; seal all "
        "three holdout raw outputs and cleanup receipts before holdout truth/scoring"
    ),
    "holdout_records_per_model": HOLDOUT_SCOPE_RECORDS,
    "first_access_marker_create_before_open": True,
    "hash_chained_broker_log": True,
    "failure_after_grant": (
        "revoke any temporary copy, append abort_and_revoke, seal an inconclusive "
        "failure receipt, and prohibit later model access"
    ),
    "scoring_release_condition": {
        "public": "all_three_primary_raw_plus_cleanup_sealed",
        "holdout": (
            "sealed_public_score_all_three_PASS_and_all_three_holdout_raw_plus_cleanup_sealed"
        ),
    },
    "expanded_confirmation": (
        "not authorized by this activation; a later sealed release requires all three "
        "complete primary gates PASS"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def strict_load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    expected = value.get(field)
    require(isinstance(expected, str) and len(expected) == 64, f"missing {field}")
    payload = {key: item for key, item in value.items() if key != field}
    require(sha256_json(payload) == expected, f"self hash mismatch: {field}")


def seal_document(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    require(field not in value, f"reserved self-hash field: {field}")
    value[field] = sha256_json(value)
    return value


def file_seal(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased file: {path}")
    return {
        "path": str(path.relative_to(relative_to)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _ast_keyword_literals(tree: ast.AST, keyword: str) -> list[Any]:
    values: list[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for item in node.keywords:
                if item.arg == keyword and isinstance(item.value, ast.Constant):
                    values.append(item.value.value)
    return values


def _call_attribute_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def static_source_checks() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    seals: dict[str, dict[str, Any]] = {}
    parsed: dict[str, ast.AST] = {}
    texts: dict[str, str] = {}
    for role, relative in SOURCE_PATHS.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        require("\x00" not in text, f"NUL in source: {role}")
        parsed[role] = ast.parse(text, filename=str(path))
        texts[role] = text
        seals[role] = file_seal(path)

    protocol_imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(parsed["protocol"])
        if isinstance(node, ast.Import) and node.names
    }
    protocol_from_imports = {
        (node.module or "").split(".")[0]
        for node in ast.walk(parsed["protocol"])
        if isinstance(node, ast.ImportFrom)
    }
    require(not ({"torch", "transformers", "bitsandbytes"} & (protocol_imports | protocol_from_imports)),
            "protocol imports a model/GPU package")

    runner_tree = parsed["runner"]
    calls = _call_attribute_names(runner_tree)
    forbidden_calls = {
        "register_forward_hook", "register_forward_pre_hook",
        "register_full_backward_hook", "register_backward_hook",
        "named_modules", "backward", "retain_grad",
    }
    require(not (calls & forbidden_calls), f"runner contains internal-access call: {calls & forbidden_calls}")
    require(True not in _ast_keyword_literals(runner_tree, "output_hidden_states"),
            "runner enables hidden states")
    require(True not in _ast_keyword_literals(runner_tree, "output_attentions"),
            "runner enables attentions")
    require(True not in _ast_keyword_literals(runner_tree, "output_scores"),
            "runner enables generation scores")
    require("apply_chat_template" not in calls, "runner calls a chat template")
    require(False in _ast_keyword_literals(runner_tree, "add_special_tokens"),
            "runner lacks an explicit add_special_tokens=False call")

    for role in ("broker", "scorer", "audit"):
        imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(parsed[role])
            if isinstance(node, ast.Import) and node.names
        }
        from_imports = {
            (node.module or "").split(".")[0]
            for node in ast.walk(parsed[role])
            if isinstance(node, ast.ImportFrom)
        }
        require("transformers" not in (imports | from_imports),
                f"{role} imports transformers")
    require("phase992_gpu_behavior_activation.v1" in texts["runner"],
            "runner does not pin activation schema")
    require("internal_trace_authorized" in texts["runner"],
            "runner does not inspect internal-trace authorization")
    require("phase983_cross_model_engine" in texts["runner"],
            "runner does not use the frozen Phase983 engine")
    require("load_model_adapter" in texts["runner"] and "release_model_adapter" in texts["runner"],
            "runner lacks frozen adapter lifecycle calls")

    engine = ROOT / ENGINE_PATH
    engine_text = engine.read_text(encoding="utf-8")
    engine_tree = ast.parse(engine_text, filename=str(engine))
    engine_functions = {
        node.name for node in ast.walk(engine_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_engine = {
        "load_model_adapter", "release_model_adapter", "eos_identity_from_sources",
        "tokenizer_pad_id", "self_test",
    }
    require(required_engine <= engine_functions, "Phase983 engine API drift")
    engine_seal = file_seal(engine)
    report = {
        "all_five_phase992_sources_ast_parse": True,
        "protocol_model_and_gpu_imports_absent": True,
        "runner_hidden_attention_score_hooks_absent": True,
        "runner_raw_input_literal_present": True,
        "runner_activation_schema_and_internal_trace_guard_present": True,
        "runner_phase983_adapter_lifecycle_present": True,
        "nonrunner_transformers_imports_absent": True,
        "phase983_required_adapter_api_present": True,
        "phase983_engine": engine_seal,
    }
    return seals, report


def parse_natural_text(text: str) -> dict[str, Any]:
    require(isinstance(text, str), "parser input must be text")
    matches = re.findall(PARSER_CONTRACT["marker_regex"], text.casefold())
    distinct = list(dict.fromkeys(matches))
    return {
        "prediction": matches[0] if matches else None,
        "unparsed": not matches,
        "ambiguous": len(distinct) > 1,
        "matches": matches,
    }


def simulation_self_test() -> dict[str, Any]:
    parser_cases = {
        "strict": parse_natural_text("The retrieved marker is Red."),
        "first_complete": parse_natural_text("blue, then red"),
        "ascii_boundary": parse_natural_text("blueberry is not blue"),
        "none": parse_natural_text("unknown"),
        "ambiguous": parse_natural_text("green then black"),
    }
    parser_pass = (
        parser_cases["strict"]["prediction"] == "red"
        and parser_cases["first_complete"]["prediction"] == "blue"
        and parser_cases["ascii_boundary"]["matches"] == ["blue"]
        and parser_cases["none"]["unparsed"] is True
        and parser_cases["ambiguous"]["ambiguous"] is True
    )

    states = ["frozen", "engineering", "public_raw", "public_score", "holdout_raw", "holdout_score"]
    allowed = {
        "frozen": {"engineering"},
        "engineering": {"public_raw"},
        "public_raw": {"public_score"},
        "public_score": {"holdout_raw"},
        "holdout_raw": {"holdout_score"},
        "holdout_score": set(),
    }
    valid_path = all(states[index + 1] in allowed[states[index]] for index in range(len(states) - 1))
    invalid_early_holdout_rejected = "holdout_raw" not in allowed["public_raw"]
    invalid_early_scoring_rejected = "public_score" not in allowed["engineering"]
    order_rejected = list(reversed(MODEL_ORDER)) != list(MODEL_ORDER)
    authorization = {
        "gpu_behavior_execution_authorized": True,
        "internal_trace_authorized": False,
        "scoring_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_formula_authorized": False,
    }
    fail_closed = (
        authorization["gpu_behavior_execution_authorized"] is True
        and all(authorization[key] is False for key in authorization if key != "gpu_behavior_execution_authorized")
    )
    checks = {
        "parser_contract_examples": parser_pass,
        "valid_staged_path": valid_path,
        "early_holdout_rejected": invalid_early_holdout_rejected,
        "early_scoring_rejected": invalid_early_scoring_rejected,
        "wrong_model_order_rejected": order_rejected,
        "activation_authority_fail_closed": fail_closed,
        "generation_contract_raw": GENERATION_CONTRACT["chat_template"] is False,
        "generation_contract_greedy": GENERATION_CONTRACT["do_sample"] is False,
        "behavior_only": GENERATION_CONTRACT["output_hidden_states"] is False,
    }
    require(all(checks.values()), f"simulation self-test failed: {checks}")
    return {"passed": True, "checks": checks, "parser_cases": parser_cases}


def engineering_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = PHASE991_OUT / "runtime_prompts" / "public" / "discovery.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            require(isinstance(row, dict), "engineering source row is not object")
            rows.append(row)
    require(len(rows) == 3072, "discovery prompt count drift")
    chosen: list[dict[str, Any]] = []
    seen_worlds: set[str] = set()
    offsets = (0, 5, 10, 15, 20, 25, 30, 31)
    by_world: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        world = str(row["semantic_world_id"])
        if world not in by_world:
            by_world[world] = []
            order.append(world)
        by_world[world].append(row)
    for world, offset in zip(order[:8], offsets, strict=True):
        variants = by_world[world]
        require(len(variants) == 32, "engineering world lacks 32 paired variants")
        row = variants[offset]
        require(row["input_mode"] == "raw_text_no_chat_template_add_special_tokens_false",
                "engineering row input mode drift")
        require(sha256_bytes(row["prompt"].encode("utf-8")) == row["prompt_sha256"],
                "engineering prompt hash drift")
        require(world not in seen_worlds, "engineering world duplicate")
        seen_worlds.add(world)
        chosen.append({
            "ordinal": len(chosen),
            "record_id": row["record_id"],
            "semantic_world_id": world,
            "variant_id": row["variant_id"],
            "prompt_sha256": row["prompt_sha256"],
        })
    require(len(chosen) == 8 and len(seen_worlds) == 8, "engineering eight selection failed")
    contract = {
        "record_count_per_model": 8,
        "same_records_all_models": True,
        "source_split": "discovery",
        "selection": "first eight discovery worlds; frozen variant offsets 0,5,10,15,20,25,30,31",
        "engineering_records": chosen,
        "engineering_records_sha256": sha256_json(chosen),
        "must_precede_full_public_execution": True,
        "required_checks": [
            "exact_input_replay", "natural_generation_schema", "four_teacher_forced_logits",
            "single_token_candidate_identity", "effective_EOS_and_pad_identity",
            "loaded_INT8_BF16_SDPA_cuda_only_identity", "raw_receipt_self_hash",
            "child_exit_and_cleanup_pass",
        ],
        "engineering_is_not_scientific_accuracy_evidence": True,
        "engineering_failure": "global_stop_before_full_execution",
    }
    return chosen, contract


def resource_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / 1024**3
    require(free_gib >= CLEANUP_AND_RESOURCE_CONTRACT["minimum_free_disk_gib_before_each_model"],
            "free disk below frozen 80 GiB minimum")
    return {
        "disk_total_bytes": usage.total,
        "disk_free_bytes_at_activation_freeze": usage.free,
        "disk_free_gib_at_activation_freeze": free_gib,
        "minimum_free_disk_gib": 80,
        "runner_must_recheck_before_each_model": True,
        "nvidia_smi_or_torch_GPU_query_performed_by_protocol": False,
        "cuda_used": False,
    }


def phase991_anchors() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    anchors: dict[str, Any] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name, filename in PHASE991_FILES.items():
        path = PHASE991_OUT / filename
        document = strict_load_json(path)
        verify_self_hash(document, PHASE991_SELF_HASH_FIELDS[name])
        documents[name] = document
        seal = file_seal(path)
        anchors[name] = {
            **seal,
            PHASE991_SELF_HASH_FIELDS[name]: document[PHASE991_SELF_HASH_FIELDS[name]],
        }
    freeze = documents["freeze"]
    audit = documents["independent_audit"]
    admission = documents["admission"]
    require(anchors["freeze"]["sha256"] == QUALIFIED_PHASE991_FREEZE_FILE_SHA256,
            "Phase991 freeze file hash is not the qualified anchor")
    require(freeze["freeze_commit_sha256"] == QUALIFIED_PHASE991_FREEZE_SELF_SHA256,
            "Phase991 freeze self hash is not the qualified anchor")
    require(freeze["cpu_gpu_admission_package"] == "qualified", "Phase991 not qualified")
    require(freeze["gpu_runner_creation_authorized"] is True, "runner creation not authorized")
    require(freeze["formal_gpu_model_execution_authorized"] is False,
            "Phase991 unexpectedly executed GPU")
    require(freeze["internal_trace_authorized"] is False, "Phase991 internal trace drift")
    require(audit["passed"] is True and all(audit["checks"].values()),
            "Phase991 independent audit failed")
    require(admission["generation_contract_sha256"] == sha256_json(admission["generation_contract"]),
            "Phase991 generation contract hash drift")
    require(admission["teacher_forced_contract_sha256"] == sha256_json(admission["teacher_forced_contract"]),
            "Phase991 teacher-forced contract hash drift")
    require(admission["thresholds_sha256"] == sha256_json(THRESHOLD_CONTRACT),
            "Phase992 threshold contract no longer equals Phase991")
    require(admission["equivalence_rule_sha256"] == sha256_json(PARSER_CONTRACT),
            "Phase992 parser contract no longer equals Phase991")

    manifest = documents["model_manifests"]
    models = manifest["models_in_required_order"]
    require([entry["model"] for entry in models] == list(MODEL_ORDER), "model manifest order drift")
    identities = [{
        "model": entry["model"],
        "logical_path": entry["logical_path"],
        "resolved_root": entry["resolved_root"],
        "file_count": entry["file_count"],
        "weight_shard_count": entry["weight_shard_count"],
        "weight_bytes": entry["weight_bytes"],
        "files_manifest_sha256": entry["files_manifest_sha256"],
    } for entry in models]
    model_contract = {
        "phase991_model_manifest": anchors["model_manifests"],
        "models_in_required_order": identities,
        "all_model_files_independently_rehashed_by_phase991_audit": True,
        "phase992_protocol_does_not_rehash_42_GB_or_load_weights": True,
    }
    runtime = admission["runtime_and_precision_contract"]
    formal = Path(runtime["formal_python"]).resolve(strict=True)
    formal_identity = {
        "path": str(formal),
        "sha256": sha256_file(formal),
    }
    require(formal_identity["sha256"] == FORMAL_PYTHON_SHA256,
            "formal Python hash drift")
    require(formal_identity["sha256"] == runtime["formal_python_sha256"],
            "formal Python differs from Phase991")
    return anchors, model_contract, formal_identity


def independent_source_hash_audit(
    formal_python: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    request = {
        "schema_version": "phase992_independent_hash_request.v1",
        "root": str(ROOT),
        "expected": deepcopy(dict(expected)),
    }
    with tempfile.TemporaryDirectory(prefix="phase992-hash-audit-") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_text(canonical_json(request), encoding="utf-8")
        completed = subprocess.run(
            [formal_python["path"], "-B", str(Path(__file__).resolve()),
             "--independent-source-audit", str(request_path)],
            check=False, capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
    require(completed.returncode == 0,
            f"independent source audit failed: {completed.stderr[-1000:]}")
    report = json.loads(completed.stdout)
    require(report["passed"] is True and report["all_hashes_match"] is True,
            "independent source audit mismatch")
    return report


def component_self_tests(formal_python: Mapping[str, Any]) -> dict[str, Any]:
    commands = {
        "broker": ["--self-test"],
        "runner": ["--self-test"],
        "scorer": ["--self-test", "--no-write"],
        "audit": ["--self-test", "--no-write"],
    }
    reports: dict[str, Any] = {}
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for role, arguments in commands.items():
        source = ROOT / SOURCE_PATHS[role]
        completed = subprocess.run(
            [formal_python["path"], "-B", str(source), *arguments],
            check=False, capture_output=True, text=True, timeout=180,
            env=environment,
        )
        require(completed.returncode == 0,
                f"{role} self-test failed: {completed.stderr[-1000:]}")
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{role} self-test did not emit one JSON document") from error
        require(isinstance(report, dict) and report.get("passed") is True,
                f"{role} self-test did not pass")
        require(report.get("cuda_used", False) is False, f"{role} self-test used CUDA")
        require(report.get("truth_opened", False) is False,
                f"{role} self-test opened scoring truth")
        require(report.get("files_written", 0) in (0, False),
                f"{role} self-test wrote persistent files")
        reports[role] = {
            "command_arguments": arguments,
            "report": report,
            "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
            "returncode": completed.returncode,
        }
    return {
        "passed": True,
        "formal_python": deepcopy(dict(formal_python)),
        "cuda_visible_devices": "",
        "components": reports,
        "component_count": len(reports),
    }


def run_independent_source_audit(request_path: Path) -> dict[str, Any]:
    request = strict_load_json(request_path)
    root = Path(request["root"]).resolve(strict=True)
    require(root == ROOT.resolve(strict=True), "independent audit root mismatch")
    current_python = Path(sys.executable).resolve(strict=True)
    require(sha256_file(current_python) == FORMAL_PYTHON_SHA256,
            "independent audit did not use formal Python")
    reports: dict[str, Any] = {}
    for name, expected in request["expected"].items():
        path = root / expected["path"]
        require(path.is_file() and not path.is_symlink(), f"independent source missing: {name}")
        # Deliberately do not call the primary file_seal/sha256_file helpers:
        # this path uses a different block size and assembles the identity
        # independently in the fresh formal-Python process.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(64 * 1024)
                if not block:
                    break
                digest.update(block)
        actual = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": os.stat(path, follow_symlinks=False).st_size,
            "sha256": digest.hexdigest(),
        }
        match = actual == expected
        reports[name] = {"expected": expected, "actual": actual, "match": match}
        require(match, f"independent source drift: {name}")
    return {
        "schema_version": "phase992_independent_hash_audit.v1",
        "phase": PHASE,
        "passed": True,
        "all_hashes_match": all(entry["match"] for entry in reports.values()),
        "source_count": len(reports),
        "sources": reports,
        "formal_python": {"path": str(current_python), "sha256": sha256_file(current_python)},
        "cuda_used": False,
        "model_weights_loaded": False,
    }


def build_documents(created_at_utc: str) -> tuple[dict[str, Any], ...]:
    source_seals, static_report = static_source_checks()
    simulation = simulation_self_test()
    _, engineering = engineering_records()
    resources = resource_preflight()
    anchors, model_contract, formal_python = phase991_anchors()
    require(Path(sys.executable).resolve(strict=True) == Path(formal_python["path"]),
            "protocol write must run under formal Python")
    engine_seal = static_report["phase983_engine"]
    all_source_seals = {**source_seals, "phase983_engine": engine_seal}
    components = component_self_tests(formal_python)
    independent = independent_source_hash_audit(formal_python, all_source_seals)

    prereg_payload = {
        "schema_version": "phase992_behavior_protocol.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "pre_gpu_external_behavior_preregistration",
        "created_at_utc": created_at_utc,
        "source_seals": source_seals,
        "phase983_engine": engine_seal,
        "phase991_anchors": anchors,
        "formal_python": formal_python,
        "model_artifact_contract": model_contract,
        "generation_contract": deepcopy(GENERATION_CONTRACT),
        "generation_contract_sha256": sha256_json(GENERATION_CONTRACT),
        "teacher_forced_contract": deepcopy(TEACHER_FORCED_CONTRACT),
        "teacher_forced_contract_sha256": sha256_json(TEACHER_FORCED_CONTRACT),
        "parser_contract": deepcopy(PARSER_CONTRACT),
        "parser_contract_sha256": sha256_json(PARSER_CONTRACT),
        "threshold_contract": deepcopy(THRESHOLD_CONTRACT),
        "threshold_contract_sha256": sha256_json(THRESHOLD_CONTRACT),
        "scoring_execution_contract": deepcopy(SCORING_EXECUTION_CONTRACT),
        "scoring_execution_contract_sha256": sha256_json(SCORING_EXECUTION_CONTRACT),
        "engineering_contract": engineering,
        "artifact_layout": deepcopy(ARTIFACT_LAYOUT),
        "serial_and_lease_contract": deepcopy(SERIAL_AND_LEASE_CONTRACT),
        "cleanup_and_resource_contract": deepcopy(CLEANUP_AND_RESOURCE_CONTRACT),
        "resource_preflight": resources,
        "holdout_contract": deepcopy(HOLDOUT_CONTRACT),
        "failure_policy": {
            "engineering_failure": "global_stop_before_full_execution",
            "integrity_lease_order_cleanup_resource_failure": "global_stop_no_next_model",
            "scientific_behavior_failure": "after cleanup continue next model but block internal trace",
            "oom_or_interruption": "inconclusive; no unfrozen resume",
        },
        "scientific_limits": {
            "internal_structure_discovered": False,
            "internal_trace_authorized": False,
            "causal_intervention_authorized": False,
            "mechanism_formula_authorized": False,
            "task_truth_graph_is_model_internal_graph": False,
            "behavior_pass_proves_two_hop_internal_mechanism": False,
            "INT8_result_generalizes_to_full_precision": False,
            "Python_broker_is_OS_sandbox": False,
        },
    }
    prereg = seal_document(prereg_payload, "protocol_sha256")

    audit_payload = {
        "schema_version": "phase992_protocol_audit.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "cpu_only_static_simulation_and_independent_source_audit",
        "created_at_utc": created_at_utc,
        "protocol_sha256": prereg["protocol_sha256"],
        "static_checks": static_report,
        "simulation_self_test": simulation,
        "component_self_tests": components,
        "independent_source_hash_audit": independent,
        "checks": {
            "phase991_qualified": anchors["freeze"]["freeze_commit_sha256"] == QUALIFIED_PHASE991_FREEZE_SELF_SHA256,
            "formal_python_qualified": formal_python["sha256"] == FORMAL_PYTHON_SHA256,
            "all_static_checks_pass": all(
                value is True for key, value in static_report.items()
                if key != "phase983_engine"
            ),
            "simulation_pass": simulation["passed"],
            "all_component_self_tests_pass": components["passed"],
            "independent_source_hashes_pass": independent["all_hashes_match"],
            "engineering_eight_frozen": engineering["record_count_per_model"] == 8,
            "disk_preflight_pass": resources["disk_free_gib_at_activation_freeze"] >= 80,
            "raw_greedy_int8_bf16_sdpa_frozen": (
                GENERATION_CONTRACT["input_mode"] == "raw_text_no_chat_template_add_special_tokens_false"
                and GENERATION_CONTRACT["do_sample"] is False
                and GENERATION_CONTRACT["quantization"] == "bitsandbytes_int8"
                and GENERATION_CONTRACT["nonquantized_dtype"] == "torch.bfloat16"
                and GENERATION_CONTRACT["attention_implementation"] == "sdpa"
            ),
            "internal_access_still_forbidden": GENERATION_CONTRACT["output_hidden_states"] is False,
            "cuda_not_used": True,
            "model_weights_not_loaded": True,
            "scoring_truth_not_opened": True,
        },
        "cuda_used": False,
        "model_weights_loaded": False,
        "scoring_truth_opened": False,
    }
    require(all(audit_payload["checks"].values()), "Phase992 protocol audit failed")
    audit = seal_document(audit_payload, "audit_sha256")

    freeze_payload = {
        "schema_version": "phase992_behavior_freeze.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "qualified_behavior_only_activation_freeze",
        "created_at_utc": created_at_utc,
        "protocol_sha256": prereg["protocol_sha256"],
        "audit_sha256": audit["audit_sha256"],
        "qualified_phase991_freeze": anchors["freeze"],
        "source_seals": source_seals,
        "phase983_engine": engine_seal,
        "formal_python": formal_python,
        "model_order": list(MODEL_ORDER),
        "decision": {
            "protocol_and_sources": "qualified",
            "activation_publication_authorized": True,
            "gpu_behavior_execution_authorized": True,
            "internal_trace_authorized": False,
            "hidden_states_authorized": False,
            "attentions_authorized": False,
            "scoring_authorized_at_activation": False,
            "causal_intervention_authorized": False,
            "mechanism_formula_authorized": False,
        },
        "gpu_used": False,
        "model_weights_loaded": False,
    }
    freeze = seal_document(freeze_payload, "freeze_sha256")

    activation_payload = {
        "schema_version": "phase992_gpu_behavior_activation.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "fail_closed_external_behavior_execution_activation",
        "created_at_utc": created_at_utc,
        "gpu_behavior_execution_authorized": True,
        "behavior_only_authorized": True,
        "internal_trace_authorized": False,
        "hidden_states_authorized": False,
        "attentions_authorized": False,
        "scoring_authorized": False,
        "scoring_release_condition": HOLDOUT_CONTRACT["scoring_release_condition"],
        "causal_intervention_authorized": False,
        "mechanism_formula_authorized": False,
        "expanded_confirmation_authorized": False,
        "model_order": list(MODEL_ORDER),
        "formal_python": formal_python,
        "qualified_phase991_freeze": anchors["freeze"],
        "phase991_anchors": anchors,
        "protocol": {"path": PREREGISTRATION, "protocol_sha256": prereg["protocol_sha256"]},
        "audit": {"path": AUDIT, "audit_sha256": audit["audit_sha256"]},
        "freeze": {"path": FREEZE, "freeze_sha256": freeze["freeze_sha256"]},
        "source_seals": source_seals,
        "phase983_engine": engine_seal,
        "model_artifact_contract": model_contract,
        "generation_contract": deepcopy(GENERATION_CONTRACT),
        "generation_contract_sha256": sha256_json(GENERATION_CONTRACT),
        "teacher_forced_contract": deepcopy(TEACHER_FORCED_CONTRACT),
        "teacher_forced_contract_sha256": sha256_json(TEACHER_FORCED_CONTRACT),
        "parser_contract": deepcopy(PARSER_CONTRACT),
        "parser_contract_sha256": sha256_json(PARSER_CONTRACT),
        "threshold_contract": deepcopy(THRESHOLD_CONTRACT),
        "threshold_contract_sha256": sha256_json(THRESHOLD_CONTRACT),
        "scoring_execution_contract": deepcopy(SCORING_EXECUTION_CONTRACT),
        "scoring_execution_contract_sha256": sha256_json(SCORING_EXECUTION_CONTRACT),
        "engineering_contract": engineering,
        "artifact_layout": deepcopy(ARTIFACT_LAYOUT),
        "execution_root": str(EXECUTION_OUT.relative_to(ROOT)).replace("\\", "/"),
        "serial_and_lease_contract": deepcopy(SERIAL_AND_LEASE_CONTRACT),
        "cleanup_and_resource_contract": deepcopy(CLEANUP_AND_RESOURCE_CONTRACT),
        "resource_preflight": resources,
        "holdout_contract": deepcopy(HOLDOUT_CONTRACT),
        "runner_fail_closed_unless_all_identity_and_authority_checks_pass": True,
    }
    activation = seal_document(activation_payload, "activation_sha256")
    return prereg, audit, freeze, activation


def write_package() -> dict[str, Any]:
    require(not OUT.exists(), f"refusing to overwrite Phase992 package: {OUT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    created = utc_now()
    pending = Path(tempfile.mkdtemp(prefix=".phase992-protocol-", dir=RESULT_ROOT))
    package = pending / "package"
    package.mkdir()
    try:
        prereg, audit, freeze, activation = build_documents(created)
        write_new_json(package / PREREGISTRATION, prereg)
        write_new_json(package / AUDIT, audit)
        write_new_json(package / FREEZE, freeze)
        write_new_json(package / ACTIVATION, activation)
        os.replace(package, OUT)
        pending.rmdir()
        return {
            "passed": True,
            "protocol_sha256": prereg["protocol_sha256"],
            "audit_sha256": audit["audit_sha256"],
            "freeze_sha256": freeze["freeze_sha256"],
            "activation_sha256": activation["activation_sha256"],
            "gpu_behavior_execution_authorized": True,
            "internal_trace_authorized": False,
            "gpu_used": False,
            "model_weights_loaded": False,
        }
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def verify_package() -> dict[str, Any]:
    require(OUT.is_dir() and not OUT.is_symlink(), "Phase992 package missing/aliased")
    prereg = strict_load_json(OUT / PREREGISTRATION)
    audit = strict_load_json(OUT / AUDIT)
    freeze = strict_load_json(OUT / FREEZE)
    activation = strict_load_json(OUT / ACTIVATION)
    for value, field in (
        (prereg, "protocol_sha256"), (audit, "audit_sha256"),
        (freeze, "freeze_sha256"), (activation, "activation_sha256"),
    ):
        verify_self_hash(value, field)
    current_sources, static_report = static_source_checks()
    current_anchors, current_models, current_formal = phase991_anchors()
    require(prereg["source_seals"] == current_sources, "Phase992 source drift")
    require(activation["source_seals"] == current_sources, "activation source drift")
    require(prereg["phase983_engine"] == static_report["phase983_engine"], "engine drift")
    require(prereg["phase991_anchors"] == current_anchors, "Phase991 anchor drift")
    require(activation["phase991_anchors"] == current_anchors, "activation Phase991 anchor drift")
    require(prereg["model_artifact_contract"] == current_models, "model manifest drift")
    require(prereg["formal_python"] == current_formal, "formal Python drift")
    require(audit["protocol_sha256"] == prereg["protocol_sha256"], "audit/protocol mismatch")
    require(freeze["audit_sha256"] == audit["audit_sha256"], "freeze/audit mismatch")
    require(activation["freeze"]["freeze_sha256"] == freeze["freeze_sha256"],
            "activation/freeze mismatch")
    require(activation["gpu_behavior_execution_authorized"] is True,
            "GPU behavior is not authorized")
    for field, contract in (
        ("generation_contract_sha256", GENERATION_CONTRACT),
        ("teacher_forced_contract_sha256", TEACHER_FORCED_CONTRACT),
        ("parser_contract_sha256", PARSER_CONTRACT),
        ("threshold_contract_sha256", THRESHOLD_CONTRACT),
        ("scoring_execution_contract_sha256", SCORING_EXECUTION_CONTRACT),
    ):
        require(activation[field] == sha256_json(contract), f"activation contract drift: {field}")
    for field in (
        "internal_trace_authorized", "scoring_authorized",
        "hidden_states_authorized", "attentions_authorized",
        "causal_intervention_authorized", "mechanism_formula_authorized",
        "expanded_confirmation_authorized",
    ):
        require(activation[field] is False, f"unsafe activation field: {field}")
    return {
        "passed": True,
        "protocol_sha256": prereg["protocol_sha256"],
        "audit_sha256": audit["audit_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "source_count": len(current_sources),
        "gpu_behavior_execution_authorized": True,
        "internal_trace_authorized": False,
    }


def self_test_only() -> dict[str, Any]:
    sources, static_report = static_source_checks()
    simulation = simulation_self_test()
    _, engineering = engineering_records()
    anchors, models, formal = phase991_anchors()
    components = component_self_tests(formal)
    return {
        "passed": True,
        "static_checks": static_report,
        "simulation": simulation,
        "engineering_records_sha256": engineering["engineering_records_sha256"],
        "phase991_freeze_commit_sha256": anchors["freeze"]["freeze_commit_sha256"],
        "models": [entry["model"] for entry in models["models_in_required_order"]],
        "formal_python": formal,
        "component_self_tests": components,
        "source_count": len(sources),
        "gpu_used": False,
        "model_weights_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--independent-source-audit", type=Path)
    arguments = parser.parse_args()
    if arguments.self_test:
        result = self_test_only()
    elif arguments.write:
        result = write_package()
    elif arguments.verify:
        result = verify_package()
    else:
        result = run_independent_source_audit(arguments.independent_source_audit)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
