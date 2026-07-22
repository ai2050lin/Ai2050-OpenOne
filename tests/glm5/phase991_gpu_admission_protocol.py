#!/usr/bin/env python3
"""Freeze the CPU-only Phase991 delayed-binding GPU admission package.

The script may hash model files and load tokenizers, but it must not import an
AutoModel class, load weights into a model, or initialize CUDA.  It physically
separates runtime prompt manifests, freezes the holdout access contract,
builds/audits the truly seeded 128-world extension, verifies the current
tokenizer-runtime migration, and exercises the Phase991 reference resolver.

No formal GPU runner exists at this stage.  A runner may be created only after
the independent Phase991 audit publishes a qualified freeze commit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import builtins
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import phase990_binding_core as p990_core
import phase990_protocol_freeze as p990_protocol
import phase991_gpu_admission_core as core
import phase991_reference_resolver as resolver


EXTENSION_PATH = Path("extension_dataset.json")
EXTENSION_AUDIT_PATH = Path("extension_audit.json")
TOKENIZER_AUDIT_PATH = Path("tokenizer_runtime_migration_and_extension_audit.json")
MODEL_MANIFEST_PATH = Path("model_artifact_manifests.json")
RESOLVER_RECEIPT_PATH = Path("reference_resolver_receipt.json")
HOLDOUT_COMMITMENT_PATH = Path("holdout_access_commitment.json")
ADMISSION_PATH = Path("gpu_admission_preregistration.json")
SELF_TEST_PATH = Path("protocol_self_test.json")
STAGE_COMMIT_PATH = Path("stage_commit.json")

PUBLIC_PROMPT_DIR = Path("runtime_prompts/public")
PRIVATE_PROMPT_DIR = Path("runtime_prompts/private")
PRIVATE_TRUTH_DIR = Path("scoring_truth/private")

MODEL_DIRS = {
    model: core.ROOT / p990_core.MODEL_PATHS[model]
    for model in core.MODEL_ORDER
}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
MODEL_REQUIRED_FILENAMES = {
    "config.json", "generation_config.json", "tokenizer_config.json"
}

GENERATION_CONTRACT = {
    "input_mode": "raw Phase990 prompt; no chat template; add_special_tokens=false",
    "model_order": list(core.MODEL_ORDER),
    "one_model_resident_at_a_time": True,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
    "max_new_tokens": 24,
    "output_scores": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "padding_side": "left",
    "truncation": False,
    "quantization": "bitsandbytes_int8",
    "nonquantized_dtype": "bfloat16",
    "attention_implementation": "sdpa",
    "cpu_or_disk_offload": False,
    "automatic_fallback": False,
    "batch_size": 8,
    "formal_runner_source_exists_at_admission_stage": False,
}

TEACHER_FORCED_CONTRACT = {
    "role": "diagnostic_only_not_replacement_for_natural_generation",
    "prefix": "The retrieved marker is",
    "joiner": "\n",
    "candidate_continuations": {value: f" {value}" for value in core.VALUES},
    "candidate_count": 4,
    "margin": "target continuation logit minus maximum of three foil logits at first continuation token",
    "all_candidates_must_be_single_token_under_each_frozen_tokenizer": True,
    "full_ids_must_equal_context_ids_plus_candidate_ids": True,
}

STOP_RULES = {
    "global_integrity_stop_no_next_model": [
        "source_or_artifact_or_model_hash_mismatch",
        "physical_split_leak_or_manifest_mismatch",
        "holdout_early_access_or_broken_log_chain",
        "runtime_or_tokenizer_drift_outside_migration_receipt",
        "reference_resolver_failure",
        "extension_three_level_overlap_nonzero",
        "extension_generator_or_seed_commitment_mismatch",
        "free_disk_below_80_GiB",
        "lease_or_model_order_violation",
        "GPU_cleanup_or_baseline_recovery_failure",
        "unfrozen_quantization_input_parser_EOS_or_resume_parameter",
    ],
    "scientific_behavior_failure": (
        "block that model from internal trace; after successful GPU cleanup, "
        "continue external behavior testing of the next model"
    ),
    "extension_release": (
        "only after all three models pass the frozen primary natural-behavior gates"
    ),
    "oom_or_interruption": "inconclusive; resume only under a future frozen resume lease",
}


def require(condition: bool, message: str) -> None:
    core.require(condition, message)


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (core.canonical_json(dict(row)) + "\n").encode("utf-8")
        for row in rows
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes(path, core.pretty_json_bytes(dict(payload)))


def _relative_seal(path: Path, base: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased file: {path}")
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": core.sha256_file(path),
    }


def _group_items(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["semantic_world_id"])].append(row)
    return grouped


def _tokenizer_audit_with_expected_count(
    rows: list[Mapping[str, Any]], expected_count: int
) -> dict[str, Any]:
    original_count = p990_core.EXPECTED_ITEM_COUNT
    require(expected_count == len(rows), "tokenizer expected count mismatch")
    try:
        p990_core.EXPECTED_ITEM_COUNT = expected_count
        return p990_protocol.tokenizer_audit(rows)
    finally:
        p990_core.EXPECTED_ITEM_COUNT = original_count


def tokenizer_runtime_audit(
    primary_records: list[Mapping[str, Any]],
    extension_records: list[Mapping[str, Any]],
    created_at_utc: str,
) -> dict[str, Any]:
    old_protocol = core.load_json(core.PHASE990_PROTOCOL)
    old_report = old_protocol["tokenizer_audit"]
    primary_current = _tokenizer_audit_with_expected_count(primary_records, 10240)
    extension_current = _tokenizer_audit_with_expected_count(
        extension_records, core.EXTENSION_RECORD_COUNT
    )

    identity_fields = (
        "tokenizer_class", "vocab_size", "model_max_length",
        "special_token_ids", "teacher_forced_prefix_ids",
        "teacher_forced_context_joiner", "candidate_continuation_text",
        "candidate_continuation_ids", "relation_phrase_ids",
        "natural_answer_target_token_positions", "record_count",
        "min_prompt_tokens", "max_prompt_tokens", "token_sequences_sha256",
        "unexpected_special_token_count", "comparison_counts",
        "comparison_failures", "chat_template_sha256",
    )
    migration: dict[str, Any] = {}
    for model in core.MODEL_ORDER:
        old = old_report["models"][model]
        new = primary_current["models"][model]
        comparisons = {field: old.get(field) == new.get(field) for field in identity_fields}
        # Package version is intentionally outside exact identity.  All token
        # behavior and local tokenizer artifact content must remain identical.
        artifact_same = (
            old["tokenizer_artifact_seal"]["files_sha256"]
            == new["tokenizer_artifact_seal"]["files_sha256"]
        )
        migration[model] = {
            "sealed_transformers_version": old["transformers_version"],
            "current_transformers_version": new["transformers_version"],
            "version_string_changed": old["transformers_version"] != new["transformers_version"],
            "identity_field_checks": comparisons,
            "tokenizer_artifact_files_sha256_equal": artifact_same,
            "accepted_runtime_migration": all(comparisons.values()) and artifact_same,
        }
        require(migration[model]["accepted_runtime_migration"], f"tokenizer migration failed: {model}")

    payload = {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "cpu_only_tokenizer_runtime_migration_and_extension_preflight",
        "phase990_exact_protocol_rebuild_in_current_runtime": False,
        "reason": "sealed transformers version 5.14.1 differs from current 5.12.0",
        "migration_does_not_rewrite_phase990": True,
        "migration_by_model": migration,
        "phase990_primary_current_recomputation": primary_current,
        "phase991_extension_current_recomputation": extension_current,
        "model_order": list(core.MODEL_ORDER),
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    return core.sealed_document(payload, "tokenizer_audit_sha256", created_at_utc)


def _model_file_manifest(model: str) -> dict[str, Any]:
    logical = MODEL_DIRS[model]
    require(logical.exists(), f"missing model directory: {logical}")
    resolved_root = logical.resolve(strict=True)
    files: list[dict[str, Any]] = []
    weight_bytes = 0
    weight_count = 0
    for path in sorted((item for item in logical.rglob("*") if item.is_file()), key=lambda p: str(p).casefold()):
        relative = str(path.relative_to(logical)).replace("\\", "/")
        resolved = path.resolve(strict=True)
        suffix = path.suffix.casefold()
        is_weight = suffix in WEIGHT_SUFFIXES
        size = resolved.stat().st_size
        files.append({
            "relative_path": relative,
            "logical_is_symlink": path.is_symlink(),
            "resolved_path": str(resolved),
            "bytes": size,
            "sha256": core.sha256_file(resolved),
            "is_weight_shard": is_weight,
        })
        if is_weight:
            weight_count += 1
            weight_bytes += size
    names = {entry["relative_path"] for entry in files}
    require(MODEL_REQUIRED_FILENAMES <= names, f"required model metadata missing: {model}")
    require(weight_count > 0 and weight_bytes > 0, f"weight shards missing: {model}")
    return {
        "model": model,
        "logical_path": str(logical),
        "logical_root_is_symlink": logical.is_symlink(),
        "resolved_root": str(resolved_root),
        "file_count": len(files),
        "weight_shard_count": weight_count,
        "weight_bytes": weight_bytes,
        "weight_gib": weight_bytes / 1024**3,
        "files": files,
        "files_manifest_sha256": core.sha256_json(files),
        "weights_loaded": False,
    }


def model_artifact_manifests(created_at_utc: str) -> dict[str, Any]:
    reports = [_model_file_manifest(model) for model in core.MODEL_ORDER]
    payload = {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "content_addressed_model_files_no_model_loading",
        "models_in_required_order": reports,
        "model_order": list(core.MODEL_ORDER),
        "total_weight_bytes": sum(report["weight_bytes"] for report in reports),
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    return core.sealed_document(payload, "model_manifest_sha256", created_at_utc)


def _runtime_environment() -> dict[str, Any]:
    usage = shutil.disk_usage(core.ROOT)
    packages = {}
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    free_gib = usage.free / 1024**3
    require(free_gib >= 80.0, "free disk below frozen 80 GiB minimum")
    return {
        "formal_python": str(Path(sys.executable).resolve()),
        "formal_python_sha256": core.sha256_file(Path(sys.executable).resolve()),
        "packages": packages,
        "nvidia_smi_query": completed.stdout.strip().splitlines(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes_at_freeze": usage.free,
        "disk_free_gib_at_freeze": free_gib,
        "minimum_free_disk_gib": 80,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
            )
        },
        "torch_imported": "torch" in sys.modules,
        "transformers_model_class_imported": False,
        "cuda_initialized": False,
    }


def _prompt_row(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(record["prompt"])
    return {
        "schema_version": "phase991_runtime_prompt.v1",
        "record_id": str(record["record_id"]),
        "semantic_world_id": str(record["semantic_world_id"]),
        "split": str(record["split"]),
        "split_ordinal": int(record["split_ordinal"]),
        "variant_id": str(record["variant_id"]),
        "prompt": prompt,
        "prompt_sha256": core.sha256_bytes(prompt.encode("utf-8")),
        "input_mode": "raw_text_no_chat_template_add_special_tokens_false",
    }


def _truth_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase991_scoring_truth.v1",
        "record_id": str(record["record_id"]),
        "semantic_world_id": str(record["semantic_world_id"]),
        "split": str(record["split"]),
        "split_ordinal": int(record["split_ordinal"]),
        "variant_id": str(record["variant_id"]),
        "semantic_transform": str(record["semantic_transform"]),
        "paraphrase_id": str(record["paraphrase_id"]),
        "fact_order_id": str(record["fact_order_id"]),
        "horizon_id": str(record["horizon_id"]),
        "gold_value": str(record["gold"]["answer_value"]),
        "gold_object": str(record["gold"]["answer_object"]),
        "query_entity": str(record["semantic_state"]["query"]["entity"]),
        "query_relation": str(record["semantic_state"]["query"]["relation"]),
        "semantic_peer_record_ids": dict(record["pair_links"]["semantic_peer_record_ids"]),
    }


def write_runtime_shards(
    root: Path,
    primary_records: list[Mapping[str, Any]],
    extension_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    all_records = [*primary_records, *extension_records]
    reports: dict[str, Any] = {}
    expected = {
        "discovery": 3072,
        "confirmation": 3072,
        "adversarial": 2048,
        "sealed_holdout": 2048,
        core.EXTENSION_SPLIT: core.EXTENSION_RECORD_COUNT,
    }
    seen_ids: set[str] = set()
    for split in core.ALL_RUNTIME_SPLITS:
        rows = [record for record in all_records if record["split"] == split]
        require(len(rows) == expected[split], f"runtime split count: {split}")
        prompt_rows = [_prompt_row(record) for record in rows]
        truth_rows = [_truth_row(record) for record in rows]
        prompt_keys = set().union(*(set(row) for row in prompt_rows))
        forbidden = {
            "gold", "gold_value", "gold_object", "semantic_state", "pair_links",
            "query_entity", "query_relation", "target", "foil",
        }
        require(not (prompt_keys & forbidden), f"truth leaked into prompts: {split}")
        ids = {row["record_id"] for row in prompt_rows}
        require(len(ids) == len(rows) and not (seen_ids & ids), f"split ID overlap: {split}")
        seen_ids.update(ids)
        if split in core.PUBLIC_SPLITS:
            prompt_path = root / PUBLIC_PROMPT_DIR / f"{split}.jsonl"
        else:
            prompt_path = root / PRIVATE_PROMPT_DIR / f"{split}.jsonl"
        truth_path = root / PRIVATE_TRUTH_DIR / f"{split}.jsonl"
        _write_bytes(prompt_path, _jsonl_bytes(prompt_rows))
        _write_bytes(truth_path, _jsonl_bytes(truth_rows))
        reports[split] = {
            "record_count": len(rows),
            "prompt_manifest": _relative_seal(prompt_path, root),
            "truth_manifest": _relative_seal(truth_path, root),
            "record_ids_sha256": core.sha256_json(sorted(ids)),
            "prompt_fields": sorted(prompt_keys),
            "truth_physically_separate": True,
            "model_runner_may_read_truth": False,
        }
    require(len(seen_ids) == 14336, "combined runtime denominator")
    return {
        "splits": reports,
        "combined_record_count": len(seen_ids),
        "record_ids_disjoint": True,
        "holdout_physically_separate_from_public_manifests": True,
        "original_phase990_dataset_remains_combined_and_runner_forbidden": True,
        "python_guard_is_not_os_sandbox": True,
    }


def _representative(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row["paraphrase_id"] == "standard"
        and row["fact_order_id"] == "order_a"
        and row["horizon_id"] == "near"
    ]


def _lookup_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["semantic_state"]["query"]["entity"]),
        str(row["gold"]["answer_object"]),
        str(row["semantic_state"]["query"]["relation"]),
    )


def discovery_fitted_lookup_baseline(
    primary_records: list[Mapping[str, Any]],
    extension_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    training = [
        row for row in _representative(primary_records)
        if row["split"] == "discovery"
    ]
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in training:
        answer = str(row["gold"]["answer_value"])
        counts[_lookup_key(row)][answer] += 1
        global_counts[answer] += 1
    order = {value: index for index, value in enumerate(core.VALUES)}

    def choose(counter: Counter[str]) -> str:
        return min(core.VALUES, key=lambda value: (-counter[value], order[value]))

    fallback = choose(global_counts)
    table = {key: choose(counter) for key, counter in counts.items()}

    def evaluate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        data = list(rows)
        correct = 0
        seen = 0
        for row in data:
            key = _lookup_key(row)
            prediction = table.get(key, fallback)
            seen += int(key in table)
            correct += int(prediction == row["gold"]["answer_value"])
        return {
            "denominator": len(data),
            "correct": correct,
            "accuracy_percent": 100.0 * correct / len(data),
            "seen_feature_rows": seen,
            "unseen_feature_rows": len(data) - seen,
        }

    evaluations = {
        split: evaluate(
            row for row in _representative(primary_records) if row["split"] == split
        )
        for split in (*core.PUBLIC_SPLITS, core.SEALED_SPLIT)
    }
    evaluations[core.EXTENSION_SPLIT] = evaluate(_representative(extension_records))
    return {
        "role": "first-hop-resolved selected-object plus relation lookup baseline",
        "not_a_pure_surface_baseline": True,
        "fit_split": "discovery",
        "training_rows": len(training),
        "feature": "query_name+resolved_selected_object+query_relation",
        "tie_break": "fixed VALUES order red,blue,green,black",
        "unseen_fallback": fallback,
        "table_sha256": core.sha256_json({"|".join(key): value for key, value in sorted(table.items())}),
        "evaluations": evaluations,
        "holdout_was_already_locally_inspectable_and_is_not_blind": True,
        "no_rule_was_tuned_after_holdout_accuracy": True,
        "above_baseline_does_not_prove_internal_two_hop_computation": True,
    }


def _holdout_commitment(
    created_at_utc: str,
    split_reports: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_hash = core.sha256_json(core.CANDIDATE_SET)
    equivalence_hash = core.sha256_json(core.EQUIVALENCE_RULE)
    threshold_hash = core.sha256_json(core.THRESHOLDS)
    search_hash = core.sha256_json(core.SEARCH_CANDIDATE_SET)
    sealed = split_reports[core.SEALED_SPLIT]["prompt_manifest"]
    payload = {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "holdout_first_model_evaluation_access_commitment",
        "candidate_set": deepcopy(core.CANDIDATE_SET),
        "candidate_set_sha256": candidate_hash,
        "search_candidate_set": deepcopy(core.SEARCH_CANDIDATE_SET),
        "search_candidate_set_sha256": search_hash,
        "equivalence_rule": deepcopy(core.EQUIVALENCE_RULE),
        "equivalence_rule_sha256": equivalence_hash,
        "thresholds": deepcopy(core.THRESHOLDS),
        "thresholds_sha256": threshold_hash,
        "freeze_timestamp": created_at_utc,
        "sealed_prompt_manifest": deepcopy(sealed),
        "holdout_semantics": "preregistered_immutable_not_blind",
        "first_human_access_claimed": False,
        "first_model_evaluation_access_status": "not_accessed",
        "first_access_log_genesis_sha256": core.sha256_json({
            "phase": core.PHASE,
            "sealed_manifest_sha256": sealed["sha256"],
            "candidate_set_sha256": candidate_hash,
            "equivalence_rule_sha256": equivalence_hash,
            "thresholds_sha256": threshold_hash,
            "freeze_timestamp": created_at_utc,
        }),
        "future_access_marker_contract": {
            "create_before_open": True,
            "create_exclusive_no_overwrite": True,
            "per_model_markers": list(core.MODEL_ORDER),
            "hash_chain_fields": [
                "previous_head", "run_id", "model", "action", "timestamp",
                "input_manifest_sha256", "output_receipt_sha256", "new_head",
            ],
            "all_three_raw_holdout_outputs_must_seal_before_any_scoring_or_review": True,
            "runner_must_not_contain_direct_holdout_path": True,
            "broker_and_Python_guard_are_not_OS_WORM_or_blindness": True,
        },
    }
    return core.sealed_document(payload, "holdout_commitment_sha256", created_at_utc)


def _file_seals_for_protocol(root: Path, relative_paths: Iterable[Path]) -> dict[str, Any]:
    return {
        str(path).replace("\\", "/"): _relative_seal(root / path, root)
        for path in relative_paths
    }


def build_package(root: Path, created_at_utc: str) -> dict[str, Any]:
    require(not root.exists(), "pending package path already exists")
    root.mkdir(parents=True)
    bridge = core.phase990_bridge()
    primary = core.load_json(core.PHASE990_DATASET)

    extension = core.extension_document(created_at_utc)
    core.verify_self_hash(extension, "extension_dataset_sha256")
    extension_for_audit = dict(extension)
    extension_for_audit.pop("extension_dataset_sha256")
    extension_for_audit.pop("created_at_utc")
    extension_audit_payload = core.audit_extension(extension_for_audit, primary)
    extension_audit = core.sealed_document(
        extension_audit_payload, "extension_audit_sha256", created_at_utc
    )
    _write_json(root / EXTENSION_PATH, extension)
    _write_json(root / EXTENSION_AUDIT_PATH, extension_audit)

    split_report = write_runtime_shards(
        root,
        list(primary["records"]),
        list(extension["records"]),
    )

    tokenizer_audit = tokenizer_runtime_audit(
        list(primary["records"]), list(extension["records"]), created_at_utc
    )
    _write_json(root / TOKENIZER_AUDIT_PATH, tokenizer_audit)

    model_manifests = model_artifact_manifests(created_at_utc)
    _write_json(root / MODEL_MANIFEST_PATH, model_manifests)

    resolver_test = resolver.self_test()
    require(resolver_test["passed"] is True, "reference resolver self-test failed")
    resolver_receipt = core.sealed_document({
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "reference_resolver_positive_and_mutation_receipt",
        "resolver_source_sha256": core.sha256_file(core.ROOT / core.SOURCE_PATHS["resolver"]),
        "self_test": resolver_test,
        "future_qualified_records_require_registry_resolution": True,
        "current_qualified_model_evidence_records": 0,
        "model_weights_loaded": False,
        "cuda_used": False,
    }, "resolver_receipt_sha256", created_at_utc)
    _write_json(root / RESOLVER_RECEIPT_PATH, resolver_receipt)

    holdout = _holdout_commitment(created_at_utc, split_report["splits"])
    _write_json(root / HOLDOUT_COMMITMENT_PATH, holdout)

    lookup = discovery_fitted_lookup_baseline(
        list(primary["records"]), list(extension["records"])
    )
    runtime = _runtime_environment()
    checks = {
        "phase990_cpu_bridge_qualified": bridge["cpu_protocol_qualified"],
        "phase990_gpu_was_not_tested": bridge["gpu_behavior_pre_phase991"] == "not_tested",
        "extension_audit_passed": extension_audit["passed"],
        "extension_worlds_128": extension_audit["counts"]["worlds"] == 128,
        "extension_records_4096": extension_audit["counts"]["records"] == 4096,
        "abstract_overlap_zero": extension_audit["three_level_overlap"]["abstract_semantic"] == 0,
        "observable_overlap_zero": extension_audit["three_level_overlap"]["observable_semantic"] == 0,
        "prompt_overlap_zero": extension_audit["three_level_overlap"]["normalized_prompt"] == 0,
        "runtime_records_14336": split_report["combined_record_count"] == 14336,
        "runtime_record_ids_disjoint": split_report["record_ids_disjoint"],
        "holdout_physically_separate": split_report["holdout_physically_separate_from_public_manifests"],
        "tokenizer_primary_migration_passed": all(
            row["accepted_runtime_migration"]
            for row in tokenizer_audit["migration_by_model"].values()
        ),
        "extension_tokenizers_passed": tokenizer_audit[
            "phase991_extension_current_recomputation"
        ]["passed"],
        "resolver_self_test_passed": resolver_test["passed"],
        "resolver_mutations_all_rejected": (
            resolver_test["mutation_rejection_count"]
            == len(resolver_test["mutation_rejections"])
            and all(resolver_test["mutation_rejections"].values())
        ),
        "weight_manifests_three_models": len(model_manifests["models_in_required_order"]) == 3,
        "disk_at_least_80_gib": runtime["disk_free_gib_at_freeze"] >= 80.0,
        "holdout_not_accessed": holdout["first_model_evaluation_access_status"] == "not_accessed",
        "shortcut_not_claimed_eliminated": extension_audit["shortcut_contract"]["shortcut_claimed_eliminated"] is False,
        "model_weights_not_loaded": model_manifests["model_weights_loaded"] is False,
        "cuda_not_used": model_manifests["cuda_used"] is False,
    }
    require(all(checks.values()), f"Phase991 protocol checks failed: {checks}")
    self_test = core.sealed_document({
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "cpu_only_protocol_self_test",
        "passed": True,
        "checks": checks,
    }, "self_test_sha256", created_at_utc)
    _write_json(root / SELF_TEST_PATH, self_test)

    artifact_paths = [
        EXTENSION_PATH, EXTENSION_AUDIT_PATH, TOKENIZER_AUDIT_PATH,
        MODEL_MANIFEST_PATH, RESOLVER_RECEIPT_PATH, HOLDOUT_COMMITMENT_PATH,
        SELF_TEST_PATH,
    ]
    artifact_paths.extend(
        Path(report[kind]["path"])
        for report in split_report["splits"].values()
        for kind in ("prompt_manifest", "truth_manifest")
    )
    artifact_seals = _file_seals_for_protocol(root, artifact_paths)
    admission_payload = {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "cpu_only_gpu_runner_creation_admission_preregistration",
        "parent_phase990": bridge,
        "source_seals": core.source_seals(core.SOURCE_PATHS),
        "phase990_source_seals": core.source_seals(core.PHASE990_SOURCE_PATHS),
        "artifact_seals": artifact_seals,
        "dataset_identity": {
            "primary_worlds": 320,
            "primary_records": 10240,
            "extension_worlds": 128,
            "extension_records": 4096,
            "independent_unit": "semantic_world_id",
            "32_variants_per_world_are_paired_not_independent": True,
        },
        "runtime_tokenizer_migration_receipt_sha256": tokenizer_audit["tokenizer_audit_sha256"],
        "candidate_set_sha256": holdout["candidate_set_sha256"],
        "search_candidate_set_sha256": holdout["search_candidate_set_sha256"],
        "equivalence_rule_sha256": holdout["equivalence_rule_sha256"],
        "thresholds_sha256": holdout["thresholds_sha256"],
        "generation_contract": deepcopy(GENERATION_CONTRACT),
        "generation_contract_sha256": core.sha256_json(GENERATION_CONTRACT),
        "teacher_forced_contract": deepcopy(TEACHER_FORCED_CONTRACT),
        "teacher_forced_contract_sha256": core.sha256_json(TEACHER_FORCED_CONTRACT),
        "physical_split_contract": split_report,
        "holdout_commitment_sha256": holdout["holdout_commitment_sha256"],
        "extension_generator_contract": {
            "new_distribution_not_same_phase990_finite_generator": True,
            "reason": "the sealed Phase990 finite generator cannot supply 128 disjoint four-transform closures",
            "source_sha256": core.sha256_file(core.ROOT / core.SOURCE_PATHS["core"]),
            "seed": core.EXTENSION_SEED,
            "seed_domain": "uint64",
            "extension_split_assignment_sha256": core.sha256_json([
                (world["semantic_world_id"], world["split_ordinal"])
                for world in extension["worlds"]
            ]),
            "three_level_overlap": extension_audit["three_level_overlap"],
            "candidate_set_equal_primary": True,
            "thresholds_equal_primary": True,
            "lexical_generalization_tested": False,
            "gold_fact_deletion_registered_as_causal_test": False,
            "bijection_missing-item_recovery_limitation_retained": True,
        },
        "shortcut_baselines": {
            "oracle_structure_baseline": extension_audit["shortcut_contract"],
            "discovery_fitted_lookup_baseline": lookup,
            "chosen_option": "B_matched_baseline_and_future_interventions",
        },
        "model_artifact_manifest_sha256": model_manifests["model_manifest_sha256"],
        "runtime_and_precision_contract": {
            **runtime,
            "quantization_changes_scientific_object": True,
            "reported_result_scope": "local INT8 checkpoints under frozen runtime, not full precision",
            "future_runner_must_capture_backend_warnings_and_actual_dtypes": True,
        },
        "serial_execution_contract": {
            "model_order": list(core.MODEL_ORDER),
            "one_model_process_at_a_time": True,
            "GPU_baseline_before_and_cleanup_after_each_model": True,
            "holdout_raw_outputs_all_models_sealed_before_scoring_or_review": True,
            "extension_only_after_all_three_primary_behavior_pass": True,
        },
        "resource_and_resume_contract": {
            "minimum_free_disk_gib": 80,
            "planned_result_quota_gib": [40, 60],
            "exclusive_lease_required": True,
            "no_unfrozen_resume": True,
            "crash_after_holdout_marker_is_fail_closed": True,
        },
        "resolver_receipt_sha256": resolver_receipt["resolver_receipt_sha256"],
        "stop_rules": deepcopy(STOP_RULES),
        "scientific_limits": {
            "phase991_discovers_internal_structure": False,
            "phase991_validates_model_behavior": False,
            "task_truth_graph_is_model_internal_graph": False,
            "behavior_above_baseline_proves_two_hop_mechanism": False,
            "holdout_is_blind": False,
            "Python_access_guard_is_OS_sandbox": False,
            "mechanism_formula_authorized": False,
        },
        "decision": {
            "cpu_admission_package": "awaiting_independent_audit",
            "gpu_runner_creation": "blocked_until_independent_freeze_commit",
            "formal_gpu_model_execution": "not_authorized",
            "internal_trace": "not_authorized",
            "causal_intervention": "not_authorized",
        },
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    admission = core.sealed_document(
        admission_payload, "gpu_admission_sha256", created_at_utc
    )
    _write_json(root / ADMISSION_PATH, admission)

    stage_files = [*artifact_paths, ADMISSION_PATH]
    stage_commit = core.sealed_document({
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "pre_independent_audit_stage_commit",
        "files": _file_seals_for_protocol(root, stage_files),
        "gpu_admission_sha256": admission["gpu_admission_sha256"],
        "independent_audit_run_count": 0,
        "gpu_runner_created": False,
        "gpu_model_run_count": 0,
        "model_weights_loaded": False,
        "cuda_used": False,
    }, "stage_commit_sha256", created_at_utc)
    _write_json(root / STAGE_COMMIT_PATH, stage_commit)
    return {
        "passed": True,
        "gpu_admission_sha256": admission["gpu_admission_sha256"],
        "stage_commit_sha256": stage_commit["stage_commit_sha256"],
        "worlds": 448,
        "records": 14336,
        "formal_gpu_model_execution": "not_authorized",
    }


def write_package() -> dict[str, Any]:
    require(not core.OUT.exists(), f"refusing to overwrite Phase991 output: {core.OUT}")
    core.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    pending = Path(tempfile.mkdtemp(prefix=".phase991-admission-", dir=core.RESULT_ROOT))
    # build_package requires a nonexistent target so it can prove exclusive
    # creation; use a child of the unique pending directory.
    package = pending / "package"
    try:
        result = build_package(package, created)
        os.replace(package, core.OUT)
        pending.rmdir()
        return result
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def verify_package() -> dict[str, Any]:
    require(core.OUT.is_dir() and not core.OUT.is_symlink(), "Phase991 package missing")
    admission = core.load_json(core.OUT / ADMISSION_PATH)
    stage = core.load_json(core.OUT / STAGE_COMMIT_PATH)
    core.verify_self_hash(admission, "gpu_admission_sha256")
    core.verify_self_hash(stage, "stage_commit_sha256")
    require(admission["source_seals"] == core.source_seals(core.SOURCE_PATHS), "source drift")
    for entry in stage["files"].values():
        path = core.OUT / entry["path"]
        require(path.stat().st_size == entry["bytes"], f"size drift: {path}")
        require(core.sha256_file(path) == entry["sha256"], f"hash drift: {path}")
    holdout = core.load_json(core.OUT / HOLDOUT_COMMITMENT_PATH)
    require(holdout["first_model_evaluation_access_status"] == "not_accessed", "holdout accessed")
    return {
        "passed": True,
        "gpu_admission_sha256": admission["gpu_admission_sha256"],
        "stage_commit_sha256": stage["stage_commit_sha256"],
        "source_count": len(admission["source_seals"]),
        "artifact_count": len(stage["files"]),
        "gpu_model_run_count": stage["gpu_model_run_count"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        result = core.self_test()
        result["resolver"] = resolver.self_test()
    elif arguments.write:
        result = write_package()
    else:
        result = verify_package()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
