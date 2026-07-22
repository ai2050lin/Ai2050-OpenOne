#!/usr/bin/env python3
"""Freeze the CPU-only Phase990 tokenizer audit and preregistration.

This script loads tokenizers in the mandatory qwen3 -> glm4 -> deepseek7b
order.  It never calls an AutoModel class, never opens model weight files,
and forces offline CPU-only execution before importing transformers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import builtins
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import phase990_binding_core as core
import phase990_binding_dataset as dataset


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PRIOR_PROMPT_SOURCES = {
    "phase979_natural": (
        "tests/glm5/result/phase979_three_boundary_factorial/rows_natural.jsonl",
        "jsonl",
    ),
    "phase979_truth_development": (
        "tests/glm5/phase979_truth_punctuation_dataset.py",
        "phase979_truth_rebuild",
    ),
    "phase981": (
        "tests/glm5/result/phase981_fresh256_confirmation/dataset.json",
        "json",
    ),
    "phase983": (
        "tests/glm5/result/phase983_cross_model_external_contract/dataset.json",
        "json",
    ),
}

PRIOR_EXECUTION_EVIDENCE = {
    "phase979_truth_development_rows": (
        "tests/glm5/result/phase979_three_boundary_factorial/rows_truth_development.jsonl"
    ),
}

_PROMPT_KEYS = {"prompt", "effective_user_prompt", "problem_prompt"}
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


@contextmanager
def _weight_file_open_guard() -> Iterable[list[str]]:
    attempts: list[str] = []
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def forbidden(file: Any) -> bool:
        if isinstance(file, int):
            return False
        try:
            path = Path(file)
        except TypeError:
            return False
        return path.suffix.casefold() in _WEIGHT_SUFFIXES

    def guarded_builtin(file: Any, *args: Any, **kwargs: Any) -> Any:
        if forbidden(file):
            attempts.append(str(file))
            raise RuntimeError(f"model weight file access rejected: {file}")
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io(file: Any, *args: Any, **kwargs: Any) -> Any:
        if forbidden(file):
            attempts.append(str(file))
            raise RuntimeError(f"model weight file access rejected: {file}")
        return original_io_open(file, *args, **kwargs)

    def guarded_os(file: Any, *args: Any, **kwargs: Any) -> Any:
        if forbidden(file):
            attempts.append(str(file))
            raise RuntimeError(f"model weight file access rejected: {file}")
        return original_os_open(file, *args, **kwargs)

    builtins.open = guarded_builtin
    io.open = guarded_io
    os.open = guarded_os
    try:
        yield attempts
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open
        os.open = original_os_open


def _collect_prompt_values(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PROMPT_KEYS and isinstance(child, str) and child.strip():
                result.add(dataset._normalized_prompt(child))
            result.update(_collect_prompt_values(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_prompt_values(child))
    return result


def prior_prompt_overlap_audit(
    items: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    current = {
        dataset._normalized_prompt(str(item["prompt"])) for item in items
    }
    source_reports: dict[str, Any] = {}
    prior_union: set[str] = set()
    for name, (relative, source_type) in PRIOR_PROMPT_SOURCES.items():
        path = core.ROOT / relative
        core.require(path.is_file(), f"missing prior prompt source: {relative}")
        payload = path.read_bytes()
        prompts: set[str] = set()
        record_count = 0
        if source_type == "json":
            parsed = core.strict_json_from_bytes(payload, name)
            prompts.update(_collect_prompt_values(parsed))
            record_count = 1
        elif source_type == "phase979_truth_rebuild":
            import phase979_truth_punctuation_dataset as phase979_truth

            pairs = phase979_truth.build_pairs("development")
            record_count = len(pairs)
            for pair in pairs:
                rendered = pair.get("prompts")
                core.require(
                    isinstance(rendered, dict) and set(rendered) == {"qA", "qB"},
                    "Phase979 truth prompt reconstruction changed",
                )
                prompts.update(
                    dataset._normalized_prompt(str(rendered[key]))
                    for key in ("qA", "qB")
                )
        else:
            for line_number, line in enumerate(payload.splitlines(), start=1):
                if not line.strip():
                    continue
                parsed = core.strict_json_from_bytes(
                    line, f"{name} line {line_number}"
                )
                prompts.update(_collect_prompt_values(parsed))
                record_count += 1
        overlap = current & prompts
        source_reports[name] = {
            "path": relative.replace("\\", "/"),
            "bytes": len(payload),
            "sha256": core.sha256_bytes(payload),
            "source_type": source_type,
            "records_parsed": record_count,
            "normalized_prompt_count": len(prompts),
            "prompt_text_available": bool(prompts),
            "phase990_overlap_count": len(overlap),
            "phase990_overlap_sha256": core.sha256_json(sorted(overlap)),
        }
        prior_union.update(prompts)
        core.require(prompts, f"prior prompt source produced no text: {name}")
    execution_evidence: dict[str, Any] = {}
    for name, relative in PRIOR_EXECUTION_EVIDENCE.items():
        path = core.ROOT / relative
        core.require(path.is_file(), f"missing prior execution evidence: {relative}")
        payload = path.read_bytes()
        execution_evidence[name] = {
            "path": relative.replace("\\", "/"),
            "bytes": len(payload),
            "sha256": core.sha256_bytes(payload),
        }
    union_overlap = current & prior_union
    report = {
        "passed": not union_overlap,
        "phase990_prompt_count": len(current),
        "prior_prompt_union_count": len(prior_union),
        "overlap_count": len(union_overlap),
        "overlap_sha256": core.sha256_json(sorted(union_overlap)),
        "sources": source_reports,
        "execution_evidence": execution_evidence,
        "sources_without_prompt_text": [],
        "coverage_complete": True,
        "claim_scope": "available_normalized_prompt_text_only",
    }
    core.require(report["passed"], "Phase990 prompt overlaps a prior corpus")
    return report


def tokenizer_artifact_seal(model_key: str) -> dict[str, Any]:
    configured = core.ROOT / core.MODEL_PATHS[model_key]
    core.require(configured.exists(), f"missing model path: {configured}")
    resolved = configured.resolve()
    core.require(resolved.is_dir(), f"model target is not a directory: {resolved}")
    files: dict[str, Any] = {}
    for path in sorted(resolved.iterdir(), key=lambda value: value.name):
        if not path.is_file():
            continue
        include = (
            path.name in _TOKENIZER_FILENAMES
            or path.name.startswith("tokenization_") and path.suffix == ".py"
            or path.name.startswith("configuration_") and path.suffix == ".py"
        )
        if not include:
            continue
        payload = path.read_bytes()
        files[path.name] = {
            "bytes": len(payload),
            "sha256": core.sha256_bytes(payload),
        }
    core.require(
        "tokenizer_config.json" in files
        and any(name in files for name in (
            "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt"
        )),
        f"{model_key} tokenizer artifacts incomplete",
    )
    return {
        "configured_path": core.MODEL_PATHS[model_key],
        "configured_path_is_symlink": configured.is_symlink(),
        "resolved_path": str(resolved),
        "files": files,
        "files_sha256": core.sha256_json(files),
        "weight_files_included_in_tokenizer_seal": False,
    }


def _artifact_file_seal(path: Path) -> dict[str, Any]:
    core.require(path.is_file(), f"missing artifact: {path}")
    core.require(not path.is_symlink(), f"artifact cannot be symlink: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(core.ROOT)).replace("\\", "/"),
        "bytes": len(payload),
        "sha256": core.sha256_bytes(payload),
    }


def _target_token_position(tokenizer: Any, value: str) -> int:
    text = f"The retrieved marker is {value}."
    target_start = len("The retrieved marker is ")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    core.require(isinstance(offsets, list), "tokenizer offsets unavailable")
    positions = [
        index for index, offset in enumerate(offsets)
        if isinstance(offset, (list, tuple))
        and len(offset) == 2
        and int(offset[0]) <= target_start < int(offset[1])
    ]
    core.require(len(positions) == 1, f"target token span ambiguous: {value}")
    return positions[0]


def _tokenizer_one_model(
    model_key: str,
    grouped_items: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    from transformers import AutoTokenizer, __version__ as transformers_version

    answer_contract = core.protocol_static_contract()["answer_contract"]
    prefix = str(answer_contract["teacher_forced_prefix"])
    joiner = str(answer_contract["teacher_forced_context_joiner"])
    continuations = {
        value: str(answer_contract["candidate_continuation_template"]).format(
            value=value
        )
        for value in core.VALUES
    }
    with _weight_file_open_guard() as weight_open_attempts:
        tokenizer = AutoTokenizer.from_pretrained(
            core.ROOT / core.MODEL_PATHS[model_key],
            local_files_only=True,
            trust_remote_code=False,
        )
    core.require(not weight_open_attempts, f"{model_key} tried to open weights")
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    value_ids = {
        value: tokenizer.encode(continuations[value], add_special_tokens=False)
        for value in core.VALUES
    }
    relation_ids = {
        relation: tokenizer.encode(f" {relation}", add_special_tokens=False)
        for relation in core.ATTRIBUTE_RELATIONS
    }
    candidate_lengths = {len(ids) for ids in value_ids.values()}
    relation_lengths = {len(ids) for ids in relation_ids.values()}
    core.require(candidate_lengths == {1}, f"{model_key} value token lengths")
    core.require(relation_lengths == {1}, f"{model_key} relation token lengths")
    core.require(len(prefix_ids) >= 2, f"{model_key} answer prefix too short")
    core.require(
        not (special_ids & set(prefix_ids)),
        f"{model_key} answer prefix contains a special token",
    )
    target_positions = {
        value: _target_token_position(tokenizer, value) for value in core.VALUES
    }
    for value, ids in value_ids.items():
        core.require(
            not (special_ids & set(ids)),
            f"{model_key}/{value} continuation contains a special token",
        )
        reconstructed = tokenizer.decode(
            [*prefix_ids, *ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        core.require(
            reconstructed == prefix + continuations[value],
            f"{model_key}/{value} prefix-continuation is not exact",
        )
        core.require(
            tokenizer.encode(
                prefix + continuations[value], add_special_tokens=False
            ) == [*prefix_ids, *ids],
            f"{model_key}/{value} continuation boundary is non-canonical",
        )
        core.require(
            target_positions[value] >= 2,
            f"{model_key}/{value} is too early in natural answer",
        )

    digest = hashlib.sha256()
    prompt_lengths: list[int] = []
    unexpected_special_count = 0
    comparison_counts = Counter()
    comparison_failures = Counter()
    expected_variants = set(core.VARIANTS)

    for world_id in sorted(grouped_items):
        rows = list(grouped_items[world_id])
        by_variant = {str(row["variant_id"]): row for row in rows}
        core.require(set(by_variant) == expected_variants, "tokenizer grid changed")
        ordered_rows = [by_variant[variant] for variant in core.VARIANTS]
        for row in ordered_rows:
            core.require(
                row.get("teacher_forced_answer_prefix") == prefix
                and row.get("teacher_forced_context_joiner") == joiner
                and row.get("teacher_forced_candidate_continuations")
                == continuations,
                f"{model_key}/{row.get('record_id')} answer contract drift",
            )
        encoded = tokenizer(
            [str(row["prompt"]) for row in ordered_rows],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        contexts = [
            str(row["prompt"]) + joiner + prefix for row in ordered_rows
        ]
        context_ids = tokenizer(
            contexts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        candidate_cases = [
            (row, context, value)
            for row, context in zip(ordered_rows, contexts, strict=True)
            for value in core.VALUES
        ]
        full_ids = tokenizer(
            [
                context + continuations[value]
                for _, context, value in candidate_cases
            ],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        ids_by_variant: dict[str, list[int]] = {}
        for row, ids in zip(ordered_rows, encoded, strict=True):
            token_ids = [int(value) for value in ids]
            ids_by_variant[str(row["variant_id"])] = token_ids
            prompt_lengths.append(len(token_ids))
            unexpected_special_count += len(special_ids & set(token_ids))
            digest.update(core.json_bytes({
                "record_id": row["record_id"],
                "input_ids": token_ids,
            }))
        context_ids_by_record = {
            str(row["record_id"]): ids
            for row, ids in zip(ordered_rows, context_ids, strict=True)
        }
        for (row, _, candidate), full_token_ids in zip(
            candidate_cases, full_ids, strict=True
        ):
            context_token_ids = context_ids_by_record[str(row["record_id"])]
            comparison_counts["teacher_forced_canonical_boundary"] += 1
            if [int(value) for value in full_token_ids] != [
                *[int(value) for value in context_token_ids],
                *value_ids[candidate],
            ]:
                comparison_failures[
                    "teacher_forced_canonical_boundary"
                ] += 1

        for paraphrase in core.PARAPHRASE_IDS:
            for order in core.FACT_ORDER_IDS:
                for horizon in core.HORIZON_IDS:
                    original = dataset.variant_id(
                        "original", paraphrase, order, horizon
                    )
                    for semantic in ("value_swap", "binding_swap"):
                        changed = dataset.variant_id(
                            semantic, paraphrase, order, horizon
                        )
                        comparison_counts["semantic_token_multiset"] += 1
                        if Counter(ids_by_variant[original]) != Counter(
                            ids_by_variant[changed]
                        ):
                            comparison_failures[
                                "semantic_token_multiset"
                            ] += 1
                    relation = dataset.variant_id(
                        "relation_swap", paraphrase, order, horizon
                    )
                    comparison_counts["relation_equal_length"] += 1
                    if len(ids_by_variant[original]) != len(
                        ids_by_variant[relation]
                    ):
                        comparison_failures["relation_equal_length"] += 1

        for semantic in core.SEMANTIC_TRANSFORMS:
            for paraphrase in core.PARAPHRASE_IDS:
                for order in core.FACT_ORDER_IDS:
                    near = dataset.variant_id(
                        semantic, paraphrase, order, "near"
                    )
                    far = dataset.variant_id(
                        semantic, paraphrase, order, "far"
                    )
                    comparison_counts["near_far_token_multiset"] += 1
                    if Counter(ids_by_variant[near]) != Counter(
                        ids_by_variant[far]
                    ):
                        comparison_failures["near_far_token_multiset"] += 1
                for horizon in core.HORIZON_IDS:
                    order_a = dataset.variant_id(
                        semantic, paraphrase, "order_a", horizon
                    )
                    order_b = dataset.variant_id(
                        semantic, paraphrase, "order_b", horizon
                    )
                    comparison_counts["order_token_multiset"] += 1
                    if Counter(ids_by_variant[order_a]) != Counter(
                        ids_by_variant[order_b]
                    ):
                        comparison_failures["order_token_multiset"] += 1

    core.require(not comparison_failures, f"{model_key} tokenizer pair failure")
    core.require(unexpected_special_count == 0, f"{model_key} special token leak")
    core.require(len(prompt_lengths) == core.EXPECTED_ITEM_COUNT,
                 f"{model_key} tokenizer record count")
    core.require(max(prompt_lengths) < 8192, f"{model_key} prompt safety bound")
    chat_template = getattr(tokenizer, "chat_template", None)
    report = {
        "passed": True,
        "model_key": model_key,
        "transformers_version": transformers_version,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": int(tokenizer.model_max_length),
        "special_token_ids": sorted(special_ids),
        "tokenizer_artifact_seal": tokenizer_artifact_seal(model_key),
        "raw_text_input_no_chat_template": True,
        "chat_template_sha256": (
            core.sha256_bytes(str(chat_template).encode("utf-8"))
            if chat_template is not None else None
        ),
        "teacher_forced_prefix_ids": prefix_ids,
        "teacher_forced_context_joiner": joiner,
        "candidate_continuation_text": continuations,
        "candidate_continuation_ids": value_ids,
        "relation_phrase_ids": relation_ids,
        "natural_answer_target_token_positions": target_positions,
        "record_count": len(prompt_lengths),
        "min_prompt_tokens": min(prompt_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "token_sequences_sha256": digest.hexdigest(),
        "unexpected_special_token_count": unexpected_special_count,
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "comparison_failures": dict(sorted(comparison_failures.items())),
        "weight_file_open_guard_attempts": list(weight_open_attempts),
        "trust_remote_code": False,
        "model_loader_api_called": False,
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    del tokenizer
    gc.collect()
    return report


def tokenizer_audit(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[str(item["semantic_world_id"])].append(item)
    reports: dict[str, Any] = {}
    observed_order: list[str] = []
    for model_key in core.MODEL_ORDER:
        observed_order.append(model_key)
        reports[model_key] = _tokenizer_one_model(model_key, grouped)
    report = {
        "passed": all(value["passed"] for value in reports.values()),
        "mandatory_order": list(core.MODEL_ORDER),
        "observed_order": observed_order,
        "records_per_model": len(rows),
        "models": reports,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "TOKENIZERS_PARALLELISM": os.environ.get(
                "TOKENIZERS_PARALLELISM"
            ),
        },
        "model_weights_loaded": False,
        "cuda_used": False,
    }
    core.require(report["passed"], "tokenizer audit failed")
    return report


def protocol_payload() -> dict[str, Any]:
    dataset.verify_artifacts()
    definitions = core.load_json(core.DEFINITIONS_PATH, "definitions")
    corpus = core.load_json(core.DATASET_PATH, "dataset")
    corpus_audit = core.load_json(core.DATASET_AUDIT_PATH, "dataset audit")
    items = corpus["records"]
    tokenizer_report = tokenizer_audit(items)
    overlap_report = prior_prompt_overlap_audit(items)
    payload = {
        **core.protocol_static_contract(),
        "definitions_sha256": definitions["definitions_sha256"],
        "dataset_sha256": corpus["dataset_sha256"],
        "dataset_audit_sha256": corpus_audit["dataset_audit_sha256"],
        "dataset_identity": deepcopy(corpus["identity"]),
        "tokenizer_audit": tokenizer_report,
        "prior_prompt_overlap_audit": overlap_report,
        "source_script_seals": core.file_seals(core.SCRIPT_PATHS),
        "artifact_file_seals": {
            "definitions": _artifact_file_seal(core.DEFINITIONS_PATH),
            "dataset": _artifact_file_seal(core.DATASET_PATH),
            "dataset_audit": _artifact_file_seal(core.DATASET_AUDIT_PATH),
        },
        "phase990_decision": {
            "cpu_protocol_seal": "qualified",
            "gpu_generation_admission": "not_tested",
            "reason_codes": [
                "CPU_SCHEMA_DATA_TOKENIZER_AUDITS_ONLY",
                "MODEL_WEIGHTS_AND_CUDA_NOT_AUTHORIZED",
                "NEW_ADMISSION_ARTIFACT_REQUIRED_BEFORE_GPU",
                "HOLDOUT_MODEL_ACCESS_GATE_NOT_SATISFIED",
                "EXTENSION_GENERATOR_NOT_FROZEN",
                "CLOSURE_REFERENCE_RESOLVER_NOT_IMPLEMENTED",
                "GRAPH_THREAD_REFERENCE_RESOLVER_NOT_IMPLEMENTED",
            ],
        },
    }
    core.require(payload["runtime_boundary"]["formal_generation_admission"] is False,
                 "protocol accidentally admitted GPU generation")
    return payload


def protocol_document(created_at_utc: str | None = None) -> dict[str, Any]:
    return core.sealed_document(
        protocol_payload(), "protocol_sha256", created_at_utc
    )


def _existing_timestamp() -> str | None:
    if not core.PROTOCOL_PATH.is_file():
        return None
    existing = core.load_json(core.PROTOCOL_PATH, "protocol")
    return core.validate_utc_timestamp(existing.get("created_at_utc"), "protocol")


def verify_artifact() -> dict[str, Any]:
    document = core.load_json(core.PROTOCOL_PATH, "protocol")
    core.verify_self_hash(document, "protocol_sha256", "protocol")
    expected = protocol_document(str(document["created_at_utc"]))
    core.verify_exact_document(
        document, expected, "protocol_sha256", "protocol"
    )
    core.require(
        core.PROTOCOL_PATH.read_bytes() == core.json_bytes(expected),
        "protocol bytes are not canonical",
    )
    core.verify_file_seals(
        document.get("source_script_seals"), core.SCRIPT_PATHS, "protocol"
    )
    return {
        "passed": True,
        "files_written": False,
        "protocol_sha256": document["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
        "models": list(core.MODEL_ORDER),
        "records_per_model": core.EXPECTED_ITEM_COUNT,
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def self_test() -> dict[str, Any]:
    payload = protocol_payload()
    checks = {
        "tokenizer_audit_passed": payload["tokenizer_audit"]["passed"],
        "model_order_exact": payload["tokenizer_audit"]["observed_order"]
        == list(core.MODEL_ORDER),
        "all_records_checked_per_model": all(
            report["record_count"] == core.EXPECTED_ITEM_COUNT
            for report in payload["tokenizer_audit"]["models"].values()
        ),
        "all_four_candidate_boundaries_checked_per_model": all(
            report["comparison_counts"].get(
                "teacher_forced_canonical_boundary"
            ) == core.EXPECTED_ITEM_COUNT * len(core.VALUES)
            for report in payload["tokenizer_audit"]["models"].values()
        ),
        "weight_access_guard_clean_per_model": all(
            report["weight_file_open_guard_attempts"] == []
            and report["trust_remote_code"] is False
            and report["model_loader_api_called"] is False
            for report in payload["tokenizer_audit"]["models"].values()
        ),
        "prior_prompt_overlap_zero": payload[
            "prior_prompt_overlap_audit"
        ]["overlap_count"] == 0,
        "model_weights_not_loaded": payload["tokenizer_audit"][
            "model_weights_loaded"
        ] is False,
        "cuda_not_used": payload["tokenizer_audit"]["cuda_used"] is False,
        "gpu_generation_not_admitted": payload["phase990_decision"][
            "gpu_generation_admission"
        ] == "not_tested",
    }
    core.require(all(checks.values()), f"protocol self-test failed: {checks}")
    return {"passed": True, "checks": checks}


def write_artifact() -> dict[str, Any]:
    document = protocol_document(_existing_timestamp())
    installed = core.install_exact(core.PROTOCOL_PATH, core.json_bytes(document))
    verified = verify_artifact()
    return {**verified, "installed": installed}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--self-test"]:
        result = self_test()
    elif arguments == ["--write"]:
        result = write_artifact()
    elif arguments == ["--verify"]:
        result = verify_artifact()
    else:
        raise SystemExit(
            "usage: phase990_protocol_freeze.py [--self-test|--write|--verify]"
        )
    print(core.canonical_json(result))


if __name__ == "__main__":
    main()
