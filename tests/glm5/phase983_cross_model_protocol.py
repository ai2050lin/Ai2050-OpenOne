#!/usr/bin/env python3
"""CPU-only preregistration for Phase 983.

This command authenticates the fresh option-swap corpus, hashes every runtime
script and local model artifact, inspects tokenizers/chat templates without
loading model weights, and freezes the external-contract study.  It authorizes
only a separate engineering qualification; formal generation needs the later
admission artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402
import phase983_option_swap_dataset as dataset_builder  # noqa: E402


GATE_PATH = GLM5 / "phase983_cross_model_gate.py"
_GATE_CONTRACT_CACHE: dict[str, Any] | None = None
_GATE_SCRIPT_SHA256_CACHE: str | None = None


def _integer_set(value: Any, label: str) -> set[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {int(value)}
    core.require(
        isinstance(value, list) and value
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"invalid {label}",
    )
    return {int(item) for item in value}


def _raw_utf8_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _native_generation_prefill(tok: Any) -> dict[str, Any]:
    """Freeze the exact native suffix added by ``add_generation_prompt``.

    Tokenizing the suffix in isolation is not sufficient because tokenization can
    depend on its left boundary.  We therefore require the no-generation render
    and IDs to be exact prefixes of the generation render and IDs, then store the
    actual suffix text and actual suffix token IDs.
    """
    probe = "PHASE983_NATIVE_GENERATION_PREFILL_PROBE"
    messages = [{"role": "user", "content": probe}]
    without_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    with_generation = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    core.require(
        isinstance(without_generation, str) and isinstance(with_generation, str)
        and with_generation.startswith(without_generation),
        "native generation render is not an extension of the no-generation render",
    )
    base_ids = list(tok(
        without_generation, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    full_ids = list(tok(
        with_generation, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    core.require(
        full_ids[:len(base_ids)] == base_ids and len(full_ids) > len(base_ids),
        "native generation prefill is not an exact non-empty token suffix",
    )
    text = with_generation[len(without_generation):]
    isolated_ids = list(tok(
        text, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    input_ids = [int(value) for value in full_ids[len(base_ids):]]
    core.require(text and input_ids == isolated_ids and input_ids,
                 "native assistant prefill is not a strict isolated token suffix")
    return {
        "probe_text": probe,
        "without_generation_prompt_sha256": _raw_utf8_sha256(without_generation),
        "with_generation_prompt_sha256": _raw_utf8_sha256(with_generation),
        "assistant_prefill_text": text,
        "assistant_prefill_text_sha256": _raw_utf8_sha256(text),
        "assistant_prefill_token_ids": input_ids,
        "assistant_prefill_token_ids_sha256": core.sha256_json(input_ids),
    }


def _assistant_prefill_for_user(tok: Any, user: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": user}]
    without = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    with_prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    core.require(isinstance(without, str) and isinstance(with_prompt, str)
                 and with_prompt.startswith(without),
                 "arm-specific native prefill is not a textual suffix")
    without_ids = list(tok(
        without, add_special_tokens=False, return_attention_mask=False).input_ids)
    with_ids = list(tok(
        with_prompt, add_special_tokens=False, return_attention_mask=False).input_ids)
    core.require(with_ids[:len(without_ids)] == without_ids,
                 "arm-specific native prefill is not a token suffix")
    return with_prompt[len(without):], [int(value) for value in with_ids[len(without_ids):]]


def _document_payload(document: dict[str, Any], hash_names: tuple[str, ...]) -> dict[str, Any]:
    blocked = set(hash_names) | {"created_at_utc", "updated_at_utc"}
    return {key: value for key, value in document.items() if key not in blocked}


def _verify_optional_self_hash(document: dict[str, Any], label: str) -> None:
    candidates = [key for key in document if key.endswith("_sha256") and key.startswith(label)]
    for field in candidates:
        expected = core.sha256_json(_document_payload(document, (field,)))
        core.require(document.get(field) == expected, f"{label} self-hash invalid")


def authenticate_dataset() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    dataset = core.load_json(core.DATASET_PATH, "Phase983 dataset")
    audit = core.load_json(core.DATASET_AUDIT_PATH, "Phase983 dataset audit")
    core.require(dataset_builder._verify_dataset_document(dataset),
                 "dataset differs from complete deterministic reconstruction")
    core.require(dataset_builder._verify_audit_document(audit),
                 "dataset audit differs from complete deterministic reconstruction")
    items = dataset.get("items")
    core.require(isinstance(items, list) and len(items) == core.ITEM_COUNT,
                 "fresh dataset is not 256 items")
    core.require(len(core.canonical_json(dataset)) > 0 and len(core.canonical_json(audit)) > 0,
                 "empty dataset artifact")
    contracts = dataset.get("contracts")
    core.require(isinstance(contracts, dict)
                 and contracts.get("phase977_holdout_accessed") is False
                 and audit.get("holdout_accessed") is False
                 and audit.get("holdout_modules_loaded") == [],
                 "fresh dataset reports Phase977 holdout access")
    core.require(dataset.get("dataset_sha256")
                 == audit.get("dataset_document_sha256")
                 and audit.get("dataset_file_sha256")
                 == core.sha256_file(core.DATASET_PATH)
                 and audit.get("passed") is True
                 and audit.get("errors") == [],
                 "dataset/audit self-lineage or pass state changed")

    ids: set[str] = set()
    by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strata = Counter()
    tasks: set[str] = set()
    for item in items:
        core.require(isinstance(item, dict), "dataset item is not an object")
        item_id = str(item.get("id", ""))
        semantic_id = str(item.get("semantic_id", ""))
        task = str(item.get("task", ""))
        difficulty = str(item.get("difficulty", ""))
        swap_side = str(item.get("swap_side", ""))
        answer = str(item.get("answer", ""))
        options = item.get("options")
        prompt = str(item.get("problem_prompt", item.get("prompt", "")))
        core.require(item_id and item_id not in ids, "duplicate/empty item ID")
        ids.add(item_id)
        core.require(semantic_id and task and prompt, "item lacks identity/task/prompt")
        core.require(difficulty in core.DIFFICULTIES
                     and swap_side in core.SWAP_SIDES and answer in core.LABELS,
                     "item stratum outside frozen registry")
        core.require(isinstance(options, dict) and set(options) == set(core.LABELS),
                     "item options malformed")
        core.require("Respond with exactly one label" not in prompt
                     and "Return only A or B" not in prompt,
                     "legacy response instruction conflicts with Phase983 contract")
        by_semantic[semantic_id].append(item)
        strata[(task, difficulty, swap_side, answer)] += 1
        tasks.add(task)

    core.require(len(by_semantic) == core.SEMANTIC_INSTANCE_COUNT,
                 "semantic instance denominator changed")
    for semantic_id, twins in by_semantic.items():
        core.require(len(twins) == 2, f"semantic twin count changed: {semantic_id}")
        twin_by_side = {str(item["swap_side"]): item for item in twins}
        core.require(set(twin_by_side) == set(core.SWAP_SIDES),
                     f"semantic twin sides malformed: {semantic_id}")
        original, swapped = twin_by_side["original"], twin_by_side["swapped"]
        core.require(original.get("spec") == swapped.get("spec")
                     and original.get("truth") == swapped.get("truth")
                     and original.get("distractor") == swapped.get("distractor"),
                     f"option swap changed semantic content: {semantic_id}")
        core.require(original["options"]["A"] == swapped["options"]["B"]
                     and original["options"]["B"] == swapped["options"]["A"],
                     f"option swap did not reverse options: {semantic_id}")
        core.require(original["answer"] != swapped["answer"],
                     f"option swap did not reverse gold label: {semantic_id}")

    core.require(len(tasks) == core.TASK_COUNT, "task denominator changed")
    for task in tasks:
        for difficulty in core.DIFFICULTIES:
            for side in core.SWAP_SIDES:
                for answer in core.LABELS:
                    core.require(strata[(task, difficulty, side, answer)] == 4,
                                 "task/difficulty/swap/label balance changed")
    audit_text = core.canonical_json(audit)
    for required in (
        "holdout", "mechanical", "fresh", "option", "swap", "structural",
    ):
        core.require(required in audit_text.casefold(),
                     f"dataset audit lacks required evidence: {required}")
    freshness = audit.get("freshness_against_prior_public_data")
    core.require(isinstance(freshness, dict)
                 and freshness.get("passed") is True
                 and freshness.get("normalized_prompt_overlap_total_n") == 0
                 and freshness.get("structural_payload_overlap_total_n") == 0
                 and audit.get("mechanically_verified_n") == core.ITEM_COUNT
                 and audit.get("unambiguous_n") == core.ITEM_COUNT
                 and audit.get("strict_option_swap_twin_n")
                 == core.SEMANTIC_INSTANCE_COUNT,
                 "dataset mechanical/freshness/twin audit changed")
    return dataset, audit, items


def artifact_file_registry(model_key: str) -> tuple[Path, list[Path]]:
    root = core.ROOT / core.MODEL_PATHS[model_key]
    core.require(root.is_dir(), f"missing local model directory: {root}")
    required = [
        root / "config.json",
        root / "generation_config.json",
        root / "tokenizer_config.json",
        root / "tokenizer.json",
        root / "model.safetensors.index.json",
    ]
    if (root / "vocab.json").is_file():
        required.append(root / "vocab.json")
    if (root / "merges.txt").is_file():
        required.append(root / "merges.txt")
    weights = sorted(root.glob("*.safetensors"))
    core.require(weights, f"no local weights for {model_key}")
    required.extend(weights)
    core.require(all(path.is_file() for path in required),
                 f"model artifact registry incomplete: {model_key}")
    return root, required


def build_model_artifact_identity(model_key: str) -> dict[str, Any]:
    root, files = artifact_file_registry(model_key)
    registry: dict[str, Any] = {}
    for path in files:
        registry[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
    payload = {
        "model_key": model_key,
        "relative_path": core.MODEL_PATHS[model_key],
        "files": registry,
        "weight_file_count": sum(name.endswith(".safetensors") for name in registry),
        "weight_bytes": sum(
            value["bytes"] for name, value in registry.items()
            if name.endswith(".safetensors")
        ),
        "weights_loaded": False,
        "gpu_accessed": False,
    }
    return {**payload, "identity_sha256": core.sha256_json(payload)}


def verify_model_artifact_identity(identity: Any, model_key: str) -> None:
    core.require(isinstance(identity, dict), f"missing model identity: {model_key}")
    expected = build_model_artifact_identity(model_key)
    core.require(identity == expected, f"model artifact identity changed: {model_key}")


def inspect_tokenizer(model_key: str, probe_item: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = core.ROOT / core.MODEL_PATHS[model_key]
    tok = AutoTokenizer.from_pretrained(
        str(root), trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tok.pad_token_id is None:
        core.require(tok.eos_token_id is not None,
                     f"{model_key} lacks both PAD and tokenizer EOS")
        tok.pad_token = tok.eos_token
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    generation = json.loads(
        (root / "generation_config.json").read_text(encoding="utf-8"))
    eos = set()
    for source, value in (
        ("tokenizer", tok.eos_token_id),
        ("config", config.get("eos_token_id")),
        ("generation", generation.get("eos_token_id")),
    ):
        eos.update(_integer_set(value, f"{model_key} {source} EOS"))
    user_a, rendered_a, ids_a = core.render_prefix(tok, probe_item, core.ARM_A)
    user_b, rendered_b, ids_b = core.render_prefix(tok, probe_item, core.ARM_B)
    core.require(user_a != user_b and rendered_a != rendered_b and ids_a != ids_b,
                 f"{model_key} arms did not produce distinct prefixes")
    core.require(user_a in rendered_a and user_b in rendered_b,
                 f"{model_key} chat template did not preserve user content")
    native_prefill = _native_generation_prefill(tok)
    prefill_a = _assistant_prefill_for_user(tok, user_a)
    prefill_b = _assistant_prefill_for_user(tok, user_b)
    core.require(prefill_a == prefill_b
                 and prefill_a[0] == native_prefill["assistant_prefill_text"]
                 and prefill_a[1] == native_prefill["assistant_prefill_token_ids"],
                 f"{model_key} native assistant prefill differs across arms")
    template = str(getattr(tok, "chat_template", ""))
    core.require(template, f"{model_key} lacks native chat template")
    all_special_ids = sorted({
        int(value) for value in getattr(tok, "all_special_ids", [])
        if isinstance(value, int) and not isinstance(value, bool)
    })
    core.require(all_special_ids and all(0 <= value < len(tok) for value in all_special_ids),
                 f"{model_key} all_special_ids malformed")
    core.require(eos.issubset(set(all_special_ids)),
                 f"{model_key} effective EOS is absent from all_special_ids")
    unexpected_special_ids = sorted(set(all_special_ids) - eos)
    return {
        "model_key": model_key,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "tokenizer_eos_token_id": int(tok.eos_token_id),
        "effective_pad_token_id": int(tok.pad_token_id),
        "effective_eos_token_ids": sorted(eos),
        "all_special_ids": all_special_ids,
        "unexpected_special_token_ids": unexpected_special_ids,
        "chat_template_sha256": _raw_utf8_sha256(template),
        "native_generation_prefill": native_prefill,
        "probe": {
            "item_id": str(probe_item["id"]),
            "arm_A_prefix_sha256": _raw_utf8_sha256(rendered_a),
            "arm_B_prefix_sha256": _raw_utf8_sha256(rendered_b),
            "arm_A_input_ids_sha256": core.sha256_json(ids_a),
            "arm_B_input_ids_sha256": core.sha256_json(ids_b),
            "arm_A_prompt_tokens": len(ids_a),
            "arm_B_prompt_tokens": len(ids_b),
        },
        "native_thinking_switch_used": False,
        "weights_loaded": False,
        "gpu_accessed": False,
    }


def inspect_tokenizer_isolated(
    model_key: str, probe_item: dict[str, Any],
) -> dict[str, Any]:
    """Inspect one tokenizer in a disposable CPU subprocess.

    The dataset verifier deliberately fails if a model runtime has entered its
    process.  Isolation keeps deterministic dataset reconstruction repeatable
    while still freezing the exact native tokenizer/template identity.
    """
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve()),
            "--inspect-tokenizer", model_key,
            "--probe-item-id", str(probe_item["id"]),
        ],
        cwd=str(core.ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="strict", env=environment, timeout=5 * 60, check=False,
    )
    core.require(completed.returncode == 0,
                 f"isolated tokenizer inspection failed: {model_key}")
    try:
        adapter = json.loads(
            completed.stdout, object_pairs_hook=core._pairs_no_duplicates,
            parse_constant=core._reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"isolated tokenizer inspection emitted invalid JSON: {model_key}"
        ) from exc
    verify_tokenizer_adapter(adapter, model_key)
    core.require(adapter["probe"]["item_id"] == str(probe_item["id"]),
                 f"isolated tokenizer probe item changed: {model_key}")
    return adapter


def gate_contract() -> dict[str, Any]:
    """Return the gate module's machine-readable contract without duplication."""
    global _GATE_CONTRACT_CACHE, _GATE_SCRIPT_SHA256_CACHE
    if _GATE_CONTRACT_CACHE is None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, str(GATE_PATH), "--contract"],
            cwd=str(core.ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="strict", env=environment, timeout=5 * 60, check=False,
        )
        core.require(completed.returncode == 0,
                     "sealed gate contract subprocess failed")
        try:
            report = json.loads(
                completed.stdout, object_pairs_hook=core._pairs_no_duplicates,
                parse_constant=core._reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("sealed gate emitted invalid strict JSON") from exc
        contract = report.get("gate_contract") if isinstance(report, dict) else None
        core.require(isinstance(contract, dict)
                     and report.get("passed") is True
                     and report.get("script_sha256") == core.sha256_file(GATE_PATH),
                     "sealed gate self-test/identity failed")
        _GATE_CONTRACT_CACHE = deepcopy(contract)
        _GATE_SCRIPT_SHA256_CACHE = str(report["script_sha256"])
    core.require(_GATE_SCRIPT_SHA256_CACHE == core.sha256_file(GATE_PATH),
                 "gate script changed after contract isolation")
    contract = deepcopy(_GATE_CONTRACT_CACHE)
    core.require(contract.get("phase") == core.PHASE,
                 "gate phase differs from protocol phase")
    core.require(tuple(contract.get("models", [])) == core.MODEL_ORDER,
                 "gate model registry differs from protocol")
    core.require(tuple(contract.get("streams", []))
                 == tuple(f"stream_{stream}" for stream in core.STREAMS),
                 "gate stream registry differs from protocol")
    core.require(tuple(contract.get("states", [])) == core.TERMINAL_STATES,
                 "gate terminal-state registry differs from protocol")
    core.require(contract.get("denominators", {}).get("per_model_stream_arm")
                 == core.ITEM_COUNT,
                 "gate item denominator differs from protocol")
    return contract


def terminal_measurement_contract() -> dict[str, Any]:
    return {
        "V": "EOS and one exact terminal FINAL label matching gold",
        "C": (
            "no EOS at the checkpoint budget; a non-EOS unexpected special token "
            "remains C with CENSORED_WITH_UNEXPECTED_SPECIAL_TOKEN subtype"
        ),
        "I_protocol": (
            "EOS but exact terminal FINAL contract invalid, or EOS with any non-EOS "
            "unexpected special token"
        ),
        "I_sem": "EOS and exact terminal FINAL contract valid but wrong label",
        "unexpected_special_definition": (
            "per-model frozen all_special_ids minus effective_eos_token_ids; prompt "
            "tokens are excluded and only generated trajectory tokens are inspected"
        ),
        "unexpected_special_fields": [
            "unexpected_special_count",
            "unexpected_special_positions",
            "unexpected_special_token_ids",
        ],
        "unexpected_special_with_eos": {
            "terminal_state": "I_protocol",
            "protocol_subtype": "EOS_WITH_UNEXPECTED_SPECIAL_TOKEN",
        },
        "invalid_final_with_eos": {
            "terminal_state": "I_protocol",
            "protocol_subtype": "EOS_WITH_INVALID_FINAL_CONTRACT",
        },
        "unexpected_special_without_eos": {
            "terminal_state": "C",
            "censor_subtype": "CENSORED_WITH_UNEXPECTED_SPECIAL_TOKEN",
        },
        "otherwise_protocol_subtype": None,
        "states_are_external_outcome_bins": True,
        "accounting_identity_only": "V+C+I_protocol+I_sem=N",
    }


def seed_contract() -> dict[str, Any]:
    return {
        "engine_namespace": core.ENGINE_NAMESPACE,
        "dataset_namespace_source": "installed protocol_sha256",
        "crn_block": "one model x one stream x one semantic seed_key",
        "rows_per_crn_block": 4,
        "block_cells": "original/swapped x external arm A/B",
        "included_in_seed": [
            "protocol_sha256", "engine_namespace", "model_key", "stream", "seed_key",
        ],
        "excluded_from_seed": ["arm", "swap_side", "item_id"],
        "model_count": len(core.MODEL_ORDER),
        "stream_count": len(core.STREAMS),
        "unique_seed_key_count": core.SEMANTIC_INSTANCE_COUNT,
        "blocks_per_model": len(core.STREAMS) * core.SEMANTIC_INSTANCE_COUNT,
        "all_model_block_count": (
            len(core.MODEL_ORDER) * len(core.STREAMS) * core.SEMANTIC_INSTANCE_COUNT
        ),
        "seed_registry_materialized_after_protocol_hash_by_admission": True,
        "arm_or_surface_pairing_is_not_counterfactual_causality": True,
    }


def build_payload() -> dict[str, Any]:
    dataset, audit, items = authenticate_dataset()
    script_seals = core.build_file_seals(core.SCRIPT_PATHS)
    dependency_seals = core.build_file_seals(core.DEPENDENCY_PATHS)
    model_identities = {
        model: build_model_artifact_identity(model) for model in core.MODEL_ORDER
    }
    tokenizers = {
        model: inspect_tokenizer_isolated(model, items[0])
        for model in core.MODEL_ORDER
    }
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "study_object": (
            "paired external natural-language instruction configuration bundles"
        ),
        "forbidden_interpretations": [
            "native thinking switch equivalence",
            "equivalent native assistant prefill across model families",
            "internal neural channel",
            "individual counterfactual causality from common random seeds",
            "statistical independence of seed streams",
            "cross-model pooled denominator",
            "shared internal language mechanism",
        ],
        "dataset": {
            "dataset_file_sha256": core.sha256_file(core.DATASET_PATH),
            "dataset_audit_file_sha256": core.sha256_file(core.DATASET_AUDIT_PATH),
            "dataset_content_sha256": core.sha256_json(dataset),
            "dataset_audit_content_sha256": core.sha256_json(audit),
            "item_count": len(items),
            "semantic_instance_count": len({item["semantic_id"] for item in items}),
            "option_swap_twins": True,
            "fresh_against_phase979_and_phase981": True,
            "phase977_holdout_loaded": False,
        },
        "models": list(core.MODEL_ORDER),
        "model_order": list(core.MODEL_ORDER),
        "model_artifact_identities": model_identities,
        "tokenizer_adapters": tokenizers,
        "native_generation_prefill_policy": {
            "frozen_separately_for_each_model": True,
            "must_be_identical_across_arms_within_model": True,
            "must_be_identical_across_models": False,
            "is_part_of_the_model_specific_native_template_bundle": True,
            "is_not_a_native_thinking_switch_intervention": True,
        },
        "arms": deepcopy(core.ARMS),
        "primary_direction": core.PRIMARY_DIRECTION,
        "sampling": deepcopy(core.SAMPLING),
        "seed_contract": seed_contract(),
        "quantization": deepcopy(core.QUANTIZATION),
        "streams": list(core.STREAMS),
        "batch_size": core.BATCH_SIZE,
        "checkpoints": list(core.CHECKPOINTS),
        "unique_decision_checkpoint": core.DECISION_CHECKPOINT,
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "expected_rows_per_model": core.EXPECTED_ROWS_PER_MODEL,
        "expected_rows_all_models": core.EXPECTED_ROWS_ALL_MODELS,
        "terminal_measurement": terminal_measurement_contract(),
        "gate_contract": gate_contract(),
        "gate_contract_sha256": core.sha256_json(gate_contract()),
        "execution_contract": {
            "cpu_only_protocol_freeze": True,
            "model_weights_loaded": False,
            "gpu_used": False,
            "formal_generation_authorized": False,
            "engineering_qualification_authorized": True,
            "qualification_uses_no_formal_dataset_items": True,
            "formal_generation_requires_separate_admission": True,
            "strict_sequential_subprocess_order": list(core.MODEL_ORDER),
            "one_model_resident_at_a_time": True,
            "first_eos_absorbing": True,
            "one_longest_rollout_for_all_checkpoints": True,
            "private_generator_per_row": True,
            "same_semantic_seed_key_model_stream_across_arms_and_swaps": True,
            "model_namespace_in_seed": True,
            "arm_excluded_from_seed": True,
            "resume_and_batch_hashing": True,
            "decision_withheld_for_independent_cpu_audit": True,
        },
        "scope": {
            "holdout": False,
            "holdout_loaded": False,
            "mechanism": False,
            "mechanism_authorized": False,
            "activation_collection": False,
            "internal_intervention": False,
            "multimodal_generalization": False,
        },
        "script_seals": script_seals,
        "dependency_seals": dependency_seals,
        "attachment_lineage": {
            "language_math_attachment_sha256": (
                "7193bb49384fb7314c9dba2ae7366f725a25b82557853a1580a55c57f2759588"
            ),
            "phase983_proposal_attachment_sha256": (
                "140613f8e9b7e91d68751e17d815bbb8930094ed90a665c86ac4591551d9b00b"
            ),
        },
    }
    return payload


def verify_tokenizer_adapter(adapter: Any, model_key: str) -> None:
    core.require(isinstance(adapter, dict), f"missing tokenizer adapter: {model_key}")
    expected_keys = {
        "model_key", "tokenizer_class", "tokenizer_length",
        "tokenizer_eos_token_id", "effective_pad_token_id",
        "effective_eos_token_ids", "all_special_ids",
        "unexpected_special_token_ids", "chat_template_sha256",
        "native_generation_prefill", "probe", "native_thinking_switch_used",
        "weights_loaded", "gpu_accessed",
    }
    core.require(set(adapter) == expected_keys,
                 f"tokenizer adapter schema changed: {model_key}")
    length = adapter.get("tokenizer_length")
    eos = adapter.get("effective_eos_token_ids")
    special = adapter.get("all_special_ids")
    unexpected = adapter.get("unexpected_special_token_ids")
    core.require(
        adapter.get("model_key") == model_key
        and isinstance(adapter.get("tokenizer_class"), str)
        and bool(adapter["tokenizer_class"])
        and isinstance(length, int) and not isinstance(length, bool) and length > 0
        and isinstance(eos, list) and eos == sorted(set(eos)) and bool(eos)
        and isinstance(special, list) and special == sorted(set(special)) and bool(special)
        and all(isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value < length for value in special)
        and set(eos).issubset(set(special))
        and unexpected == sorted(set(special) - set(eos))
        and adapter.get("tokenizer_eos_token_id") in eos
        and isinstance(adapter.get("effective_pad_token_id"), int)
        and 0 <= adapter["effective_pad_token_id"] < length
        and _is_sha256(adapter.get("chat_template_sha256"))
        and adapter.get("native_thinking_switch_used") is False
        and adapter.get("weights_loaded") is False
        and adapter.get("gpu_accessed") is False,
        f"tokenizer adapter identity changed: {model_key}",
    )
    prefill = adapter.get("native_generation_prefill")
    core.require(isinstance(prefill, dict) and set(prefill) == {
        "probe_text", "without_generation_prompt_sha256",
        "with_generation_prompt_sha256", "assistant_prefill_text",
        "assistant_prefill_text_sha256", "assistant_prefill_token_ids",
        "assistant_prefill_token_ids_sha256",
    }, f"native assistant prefill schema changed: {model_key}")
    text = prefill.get("assistant_prefill_text")
    prefill_ids = prefill.get("assistant_prefill_token_ids")
    core.require(
        prefill.get("probe_text") == "PHASE983_NATIVE_GENERATION_PREFILL_PROBE"
        and _is_sha256(prefill.get("without_generation_prompt_sha256"))
        and _is_sha256(prefill.get("with_generation_prompt_sha256"))
        and isinstance(text, str) and bool(text)
        and prefill.get("assistant_prefill_text_sha256") == _raw_utf8_sha256(text)
        and isinstance(prefill_ids, list) and bool(prefill_ids)
        and all(isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value < length for value in prefill_ids)
        and prefill.get("assistant_prefill_token_ids_sha256")
        == core.sha256_json(prefill_ids),
        f"native assistant prefill identity changed: {model_key}",
    )
    probe = adapter.get("probe")
    probe_keys = {
        "item_id", "arm_A_prefix_sha256", "arm_B_prefix_sha256",
        "arm_A_input_ids_sha256", "arm_B_input_ids_sha256",
        "arm_A_prompt_tokens", "arm_B_prompt_tokens",
    }
    core.require(
        isinstance(probe, dict) and set(probe) == probe_keys
        and isinstance(probe.get("item_id"), str) and bool(probe["item_id"])
        and all(_is_sha256(probe.get(field)) for field in (
            "arm_A_prefix_sha256", "arm_B_prefix_sha256",
            "arm_A_input_ids_sha256", "arm_B_input_ids_sha256",
        ))
        and all(isinstance(probe.get(field), int) and not isinstance(probe[field], bool)
                and probe[field] > 0 for field in (
                    "arm_A_prompt_tokens", "arm_B_prompt_tokens"))
        and probe["arm_A_prefix_sha256"] != probe["arm_B_prefix_sha256"]
        and probe["arm_A_input_ids_sha256"] != probe["arm_B_input_ids_sha256"],
        f"tokenizer probe identity changed: {model_key}",
    )


def verify_payload(
    payload: Any, expected_payload: dict[str, Any] | None = None,
) -> None:
    core.require(isinstance(payload, dict), "protocol payload missing")
    core.require(payload.get("phase") == core.PHASE
                 and payload.get("experiment") == core.EXPERIMENT,
                 "protocol phase/experiment changed")
    core.require(payload.get("models") == list(core.MODEL_ORDER)
                 and payload.get("model_order") == list(core.MODEL_ORDER),
                 "model registry/order changed")
    core.require(payload.get("arms") == core.ARMS
                 and payload.get("sampling") == core.SAMPLING
                 and payload.get("quantization") == core.QUANTIZATION,
                 "arm/sampling/quantization contract changed")
    core.require(payload.get("seed_contract") == seed_contract(),
                 "paired seed/CRN block contract changed")
    core.require(
        payload.get("streams") == list(core.STREAMS)
        and payload.get("batch_size") == core.BATCH_SIZE
        and payload.get("checkpoints") == list(core.CHECKPOINTS)
        and payload.get("unique_decision_checkpoint") == core.DECISION_CHECKPOINT
        and payload.get("max_new_tokens") == core.MAX_NEW_TOKENS
        and payload.get("expected_rows_per_model") == core.EXPECTED_ROWS_PER_MODEL
        and payload.get("expected_rows_all_models") == core.EXPECTED_ROWS_ALL_MODELS,
        "protocol grid/horizon changed",
    )
    adapters = payload.get("tokenizer_adapters")
    core.require(isinstance(adapters, dict) and set(adapters) == set(core.MODEL_ORDER),
                 "tokenizer adapter registry changed")
    for model in core.MODEL_ORDER:
        verify_tokenizer_adapter(adapters[model], model)
    core.require(payload.get("native_generation_prefill_policy") == {
        "frozen_separately_for_each_model": True,
        "must_be_identical_across_arms_within_model": True,
        "must_be_identical_across_models": False,
        "is_part_of_the_model_specific_native_template_bundle": True,
        "is_not_a_native_thinking_switch_intervention": True,
    }, "native assistant prefill policy changed")
    core.require(payload.get("terminal_measurement") == terminal_measurement_contract(),
                 "terminal measurement contract changed")
    core.require(payload.get("gate_contract") == gate_contract(),
                 "gate contract changed")
    core.require(payload.get("gate_contract_sha256")
                 == core.sha256_json(gate_contract()),
                 "gate contract machine hash changed")
    execution = payload.get("execution_contract")
    core.require(isinstance(execution, dict)
                 and execution.get("formal_generation_authorized") is False
                 and execution.get("engineering_qualification_authorized") is True
                 and execution.get("one_model_resident_at_a_time") is True,
                 "pre-admission execution boundary changed")
    scope = payload.get("scope")
    core.require(isinstance(scope, dict) and not any(scope.values()),
                 "holdout/mechanism scope widened")
    expected = build_payload() if expected_payload is None else expected_payload
    core.require(isinstance(expected, dict) and payload == expected,
                 "protocol differs from complete deterministic reconstruction")
    core.verify_file_seals(payload.get("script_seals"), core.SCRIPT_PATHS,
                           "Phase983 script")
    core.verify_file_seals(payload.get("dependency_seals"), core.DEPENDENCY_PATHS,
                           "Phase983 dependency")
    identities = payload.get("model_artifact_identities")
    core.require(isinstance(identities, dict) and set(identities) == set(core.MODEL_ORDER),
                 "model artifact registry changed")
    for model in core.MODEL_ORDER:
        identity = identities[model]
        core.require(isinstance(identity, dict)
                     and identity.get("identity_sha256")
                     == core.sha256_json(core.without_fields(identity, "identity_sha256")),
                     f"model artifact identity self-hash changed: {model}")


def install_protocol(
    payload: dict[str, Any], expected_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    verify_payload(payload, expected_payload)
    if core.PROTOCOL_PATH.exists():
        existing = core.load_json(core.PROTOCOL_PATH, "existing Phase983 protocol")
        core.verify_self_hash(existing, "protocol_sha256", "created_at_utc",
                              "Phase983 protocol")
        core.require(core.without_fields(existing, "protocol_sha256", "created_at_utc")
                     == payload, "existing protocol differs from rebuilt payload")
        return existing, False
    document = {
        **payload,
        "protocol_sha256": core.sha256_json(payload),
        "created_at_utc": core.utc_now(),
    }
    core.atomic_write_json(core.PROTOCOL_PATH, document)
    installed = core.load_json(core.PROTOCOL_PATH, "installed Phase983 protocol")
    core.verify_self_hash(installed, "protocol_sha256", "created_at_utc",
                          "installed Phase983 protocol")
    core.require(installed == document,
                 "installed protocol differs after JSON serialization")
    return installed, True


def negative_tests(
    payload: dict[str, Any], expected_payload: dict[str, Any],
) -> dict[str, bool]:
    tests: dict[str, bool] = {}
    mutations = {
        "model_order_rejected": lambda value: value.__setitem__(
            "model_order", list(reversed(core.MODEL_ORDER))),
        "native_arm_rejected": lambda value: value["arms"]["A"].__setitem__(
            "instruction", "/no_think"),
        "threshold_relaxation_rejected": lambda value: value["gate_contract"][
            "per_model_stream_primary"].__setitem__("delta_C_max", 13),
        "gate_hash_rewrite_rejected": lambda value: value.__setitem__(
            "gate_contract_sha256", "0" * 64),
        "special_registry_rejected": lambda value: value["tokenizer_adapters"][
            "qwen3"]["all_special_ids"].pop(),
        "native_prefill_rejected": lambda value: value["tokenizer_adapters"][
            "deepseek7b"]["native_generation_prefill"].__setitem__(
                "assistant_prefill_text", "changed"),
        "seed_surface_exclusion_rejected": lambda value: value[
            "seed_contract"]["excluded_from_seed"].remove("item_id"),
        "unexpected_special_rule_rejected": lambda value: value[
            "terminal_measurement"]["unexpected_special_with_eos"].__setitem__(
                "protocol_subtype", "changed"),
        "formal_pregeneration_rejected": lambda value: value[
            "execution_contract"].__setitem__("formal_generation_authorized", True),
        "mechanism_scope_rejected": lambda value: value["scope"].__setitem__(
            "mechanism", True),
        "empty_script_seals_rejected": lambda value: value.__setitem__(
            "script_seals", {}),
    }
    for name, mutate in mutations.items():
        candidate = deepcopy(payload)
        mutate(candidate)
        try:
            verify_payload(candidate, expected_payload)
        except (RuntimeError, KeyError, TypeError):
            tests[name] = True
        else:
            tests[name] = False
    rehashed_payload = deepcopy(payload)
    rehashed_payload["terminal_measurement"]["unexpected_special_with_eos"][
        "protocol_subtype"] = "REHASHED_TAMPER"
    rehashed_document = {
        **rehashed_payload,
        "protocol_sha256": core.sha256_json(rehashed_payload),
        "created_at_utc": "2000-01-01T00:00:00+00:00",
    }
    core.verify_self_hash(rehashed_document, "protocol_sha256", "created_at_utc",
                          "synthetic rehashed protocol")
    try:
        verify_payload(rehashed_payload, expected_payload)
    except (RuntimeError, KeyError, TypeError):
        tests["self_rehashed_payload_tamper_rejected"] = True
    else:
        tests["self_rehashed_payload_tamper_rejected"] = False
    core.require(all(tests.values()), "protocol negative test failed")
    return tests


def run(write: bool) -> dict[str, Any]:
    payload = build_payload()
    expected_payload = build_payload()
    verify_payload(payload, expected_payload)
    tests = negative_tests(payload, expected_payload)
    result = {
        "phase": core.PHASE,
        "protocol_payload_sha256": core.sha256_json(payload),
        "negative_tests": tests,
        "cpu_only": True,
        "model_weights_loaded": False,
        "gpu_used": False,
        "formal_generation_authorized": False,
        "engineering_qualification_authorized": True,
        "files_written": False,
    }
    if write:
        document, created = install_protocol(payload, expected_payload)
        result.update({
            "protocol_sha256": document["protocol_sha256"],
            "protocol_file_sha256": core.sha256_file(core.PROTOCOL_PATH),
            "files_written": created,
            "existing": not created,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--inspect-tokenizer", choices=core.MODEL_ORDER)
    parser.add_argument("--probe-item-id")
    args = parser.parse_args()
    if args.inspect_tokenizer is not None:
        core.require(isinstance(args.probe_item_id, str) and bool(args.probe_item_id),
                     "isolated tokenizer inspection requires --probe-item-id")
        dataset = core.load_json(core.DATASET_PATH, "Phase983 tokenizer probe dataset")
        matches = [
            item for item in dataset.get("items", [])
            if isinstance(item, dict) and str(item.get("id")) == args.probe_item_id
        ]
        core.require(len(matches) == 1, "isolated tokenizer probe item not unique")
        print(json.dumps(
            inspect_tokenizer(str(args.inspect_tokenizer), matches[0]),
            ensure_ascii=False, indent=2, sort_keys=True,
        ))
        return
    core.require(args.probe_item_id is None,
                 "--probe-item-id is only valid with --inspect-tokenizer")
    print(json.dumps(run(args.write), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
