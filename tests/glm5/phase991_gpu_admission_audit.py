#!/usr/bin/env python3
"""Independent execution audit for the Phase991 CPU admission package.

The audit rehashes every source, artifact, and model file; reconstructs every
physical prompt/truth row; independently re-tokenizes all 10,240 primary and
4,096 extension records for all three tokenizers; rechecks the three overlap
levels and resolver mutation suite; and only then publishes the CPU admission
freeze.  It never imports AutoModel, loads weights, or initializes CUDA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import phase990_binding_core as p990_core
import phase990_binding_dataset as p990_data
import phase991_gpu_admission_core as core
import phase991_gpu_admission_protocol as protocol
import phase991_reference_resolver as resolver


AUDIT_PATH = core.OUT / "independent_execution_audit.json"
FREEZE_PATH = core.OUT / "freeze_commit.json"


def require(condition: bool, message: str) -> None:
    core.require(condition, message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f"JSONL object drift: {path}")
    return rows


def _verify_file_entry(base: Path, entry: Mapping[str, Any]) -> None:
    path = base / str(entry["path"])
    require(path.is_file() and not path.is_symlink(), f"missing/aliased artifact: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"artifact size drift: {path}")
    require(core.sha256_file(path) == entry["sha256"], f"artifact hash drift: {path}")


def _source_and_stage_checks(
    admission: Mapping[str, Any], stage: Mapping[str, Any]
) -> dict[str, bool]:
    for entry in admission["source_seals"].values():
        _verify_file_entry(core.ROOT, entry)
    for entry in admission["phase990_source_seals"].values():
        _verify_file_entry(core.ROOT, entry)
    for entry in stage["files"].values():
        _verify_file_entry(core.OUT, entry)
    return {
        "phase991_sources_rehashed": True,
        "phase990_sources_rehashed": True,
        "stage_artifacts_rehashed": True,
        "stage_admission_hash_matches": (
            stage["gpu_admission_sha256"] == admission["gpu_admission_sha256"]
        ),
    }


def _independent_extension_checks(
    primary: Mapping[str, Any], extension: Mapping[str, Any]
) -> dict[str, Any]:
    old_abstract = {
        str(row["slot_canonical_semantic_sha256"]) for row in primary["records"]
    }
    old_observable = {
        str(row["observable_semantic_variant_sha256"]) for row in primary["records"]
    }
    old_prompts = {
        str(row["normalized_surface_sha256"]) for row in primary["records"]
    }
    new_abstract: set[str] = set()
    new_observable: set[str] = set()
    new_prompts: set[str] = set()
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in extension["records"]:
        state = row["semantic_state"]
        solved = p990_data.solve_state(state)
        require(solved["answer_value"] == row["gold"]["answer_value"], "gold mismatch")
        require(solved["answer_object"] == row["gold"]["answer_object"], "object mismatch")
        abstract = p990_core.sha256_json(p990_data._slot_semantics(state))
        observable = p990_core.sha256_json(p990_data._observable_semantics(state))
        normalized = p990_core.sha256_json(p990_data._normalized_prompt(str(row["prompt"])))
        require(abstract == row["slot_canonical_semantic_sha256"], "abstract hash mismatch")
        require(observable == row["observable_semantic_variant_sha256"], "observable hash mismatch")
        require(normalized == row["normalized_surface_sha256"], "prompt hash mismatch")
        new_abstract.add(abstract)
        new_observable.add(observable)
        new_prompts.add(normalized)
        groups[str(row["semantic_world_id"])].append(row)
    require(len(groups) == 128, "extension world count")
    require(all(len(rows) == 32 for rows in groups.values()), "extension variants")
    require(len(new_abstract) == 512, "extension abstract cardinality")
    require(len(new_observable) == 512, "extension observable cardinality")
    require(len(new_prompts) == 4096, "extension prompt cardinality")
    require(not (old_abstract & new_abstract), "abstract overlap")
    require(not (old_observable & new_observable), "observable overlap")
    require(not (old_prompts & new_prompts), "prompt overlap")

    regenerated = core.extension_document(str(extension["created_at_utc"]))
    require(regenerated == extension, "seeded extension exact regeneration mismatch")
    alternate = core.generate_extension(core.EXTENSION_SEED + 1, primary)
    require(
        alternate["extension_payload_sha256"]
        != core.generate_extension(core.EXTENSION_SEED, primary)["extension_payload_sha256"],
        "seed does not change extension",
    )
    return {
        "worlds": len(groups),
        "records": len(extension["records"]),
        "abstract_states": len(new_abstract),
        "observable_states": len(new_observable),
        "normalized_prompts": len(new_prompts),
        "abstract_overlap": 0,
        "observable_overlap": 0,
        "normalized_prompt_overlap": 0,
        "same_seed_exact_regeneration": True,
        "alternate_seed_changes_payload": True,
    }


def _independent_split_checks(
    primary: Mapping[str, Any], extension: Mapping[str, Any], admission: Mapping[str, Any]
) -> dict[str, Any]:
    source = {
        str(row["record_id"]): row
        for row in [*primary["records"], *extension["records"]]
    }
    expected = {
        "discovery": 3072,
        "confirmation": 3072,
        "adversarial": 2048,
        "sealed_holdout": 2048,
        core.EXTENSION_SPLIT: 4096,
    }
    all_prompt_ids: set[str] = set()
    reports = {}
    split_contract = admission["physical_split_contract"]["splits"]
    for split in core.ALL_RUNTIME_SPLITS:
        prompt_entry = split_contract[split]["prompt_manifest"]
        truth_entry = split_contract[split]["truth_manifest"]
        _verify_file_entry(core.OUT, prompt_entry)
        _verify_file_entry(core.OUT, truth_entry)
        prompts = _read_jsonl(core.OUT / prompt_entry["path"])
        truth = _read_jsonl(core.OUT / truth_entry["path"])
        require(len(prompts) == len(truth) == expected[split], f"split denominator: {split}")
        prompt_by_id = {str(row["record_id"]): row for row in prompts}
        truth_by_id = {str(row["record_id"]): row for row in truth}
        require(len(prompt_by_id) == len(prompts), f"prompt duplicate: {split}")
        require(set(prompt_by_id) == set(truth_by_id), f"prompt/truth registry: {split}")
        require(not (set(prompt_by_id) & all_prompt_ids), f"cross split duplicate: {split}")
        all_prompt_ids.update(prompt_by_id)
        for record_id, prompt in prompt_by_id.items():
            original = source[record_id]
            require(prompt["split"] == split, "prompt split mismatch")
            require(prompt["prompt"] == original["prompt"], "prompt content mismatch")
            require(
                prompt["prompt_sha256"]
                == core.sha256_bytes(str(original["prompt"]).encode("utf-8")),
                "prompt hash mismatch",
            )
            require(
                not ({"gold", "gold_value", "gold_object", "semantic_state", "pair_links"} & set(prompt)),
                "truth leaked into prompt shard",
            )
            target = truth_by_id[record_id]
            require(target["gold_value"] == original["gold"]["answer_value"], "truth value")
            require(target["gold_object"] == original["gold"]["answer_object"], "truth object")
        reports[split] = {"records": len(prompts), "exact_source_rows": len(prompts)}
    require(len(all_prompt_ids) == 14336, "combined physical denominator")
    return {
        "splits": reports,
        "combined_records": len(all_prompt_ids),
        "record_ids_disjoint": True,
        "prompt_truth_exact_join": True,
        "truth_absent_from_runtime_prompts": True,
    }


def _tokenizer_digest(
    tokenizer: Any, rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["semantic_world_id"])].append(row)
    digest = hashlib.sha256()
    lengths: list[int] = []
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    unexpected = 0
    boundary_failures = 0
    candidate_ids = {
        value: tokenizer.encode(f" {value}", add_special_tokens=False)
        for value in core.VALUES
    }
    require({len(value) for value in candidate_ids.values()} == {1}, "candidate lengths")
    for world_id in sorted(grouped):
        by_variant = {str(row["variant_id"]): row for row in grouped[world_id]}
        require(set(by_variant) == set(core.EXTENSION_VARIANTS), "tokenizer variant grid")
        ordered = [by_variant[variant] for variant in core.EXTENSION_VARIANTS]
        prompts = [str(row["prompt"]) for row in ordered]
        encoded = tokenizer(
            prompts, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        contexts = [prompt + "\nThe retrieved marker is" for prompt in prompts]
        context_ids = tokenizer(
            contexts, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        full_texts = [
            context + f" {value}"
            for context in contexts
            for value in core.VALUES
        ]
        full_ids = tokenizer(
            full_texts, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        for row, ids in zip(ordered, encoded, strict=True):
            token_ids = [int(value) for value in ids]
            lengths.append(len(token_ids))
            unexpected += len(special_ids & set(token_ids))
            digest.update(p990_core.json_bytes({
                "record_id": row["record_id"], "input_ids": token_ids
            }))
        index = 0
        for context in context_ids:
            for value in core.VALUES:
                expected = [*[int(token) for token in context], *candidate_ids[value]]
                actual = [int(token) for token in full_ids[index]]
                boundary_failures += int(expected != actual)
                index += 1
    return {
        "record_count": len(lengths),
        "min_prompt_tokens": min(lengths),
        "max_prompt_tokens": max(lengths),
        "token_sequences_sha256": digest.hexdigest(),
        "unexpected_special_token_count": unexpected,
        "teacher_forced_boundary_failures": boundary_failures,
        "candidate_continuation_ids": candidate_ids,
    }


def _independent_tokenizer_checks(
    primary: Mapping[str, Any],
    extension: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    reports: dict[str, Any] = {}
    for model in core.MODEL_ORDER:
        tokenizer = AutoTokenizer.from_pretrained(
            core.ROOT / p990_core.MODEL_PATHS[model],
            local_files_only=True,
            trust_remote_code=False,
        )
        primary_report = _tokenizer_digest(tokenizer, list(primary["records"]))
        extension_report = _tokenizer_digest(tokenizer, list(extension["records"]))
        expected_primary = receipt["phase990_primary_current_recomputation"]["models"][model]
        expected_extension = receipt["phase991_extension_current_recomputation"]["models"][model]
        for actual, expected, label in (
            (primary_report, expected_primary, "primary"),
            (extension_report, expected_extension, "extension"),
        ):
            for field in (
                "record_count", "min_prompt_tokens", "max_prompt_tokens",
                "token_sequences_sha256", "unexpected_special_token_count",
                "candidate_continuation_ids",
            ):
                require(actual[field] == expected[field], f"{model}/{label}/{field}")
            require(actual["teacher_forced_boundary_failures"] == 0, f"{model}/{label}/boundary")
        reports[model] = {
            "primary": primary_report,
            "extension": extension_report,
            "tokenizer_class": type(tokenizer).__name__,
            "recomputed_independently": True,
        }
        del tokenizer
        gc.collect()
    return {"model_order": list(core.MODEL_ORDER), "models": reports, "passed": True}


def _independent_model_manifest_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    reports = []
    for model_report in manifest["models_in_required_order"]:
        total = 0
        weight_count = 0
        for entry in model_report["files"]:
            path = Path(entry["resolved_path"])
            require(path.is_file(), f"model file missing: {path}")
            require(path.stat().st_size == entry["bytes"], f"model size: {path}")
            require(core.sha256_file(path) == entry["sha256"], f"model hash: {path}")
            if entry["is_weight_shard"]:
                total += entry["bytes"]
                weight_count += 1
        require(total == model_report["weight_bytes"], "weight byte total")
        require(weight_count == model_report["weight_shard_count"], "weight count")
        reports.append({
            "model": model_report["model"],
            "weight_bytes": total,
            "weight_shard_count": weight_count,
            "all_files_rehashed": len(model_report["files"]),
        })
    require([row["model"] for row in reports] == list(core.MODEL_ORDER), "model order")
    return {"models": reports, "passed": True}


def audit_payload() -> dict[str, Any]:
    protocol.verify_package()
    admission = core.load_json(core.OUT / protocol.ADMISSION_PATH)
    stage = core.load_json(core.OUT / protocol.STAGE_COMMIT_PATH)
    extension = core.load_json(core.OUT / protocol.EXTENSION_PATH)
    tokenizer_receipt = core.load_json(core.OUT / protocol.TOKENIZER_AUDIT_PATH)
    model_manifest = core.load_json(core.OUT / protocol.MODEL_MANIFEST_PATH)
    holdout = core.load_json(core.OUT / protocol.HOLDOUT_COMMITMENT_PATH)
    primary = core.load_json(core.PHASE990_DATASET)

    source_checks = _source_and_stage_checks(admission, stage)
    extension_checks = _independent_extension_checks(primary, extension)
    split_checks = _independent_split_checks(primary, extension, admission)
    tokenizer_checks = _independent_tokenizer_checks(primary, extension, tokenizer_receipt)
    model_checks = _independent_model_manifest_checks(model_manifest)
    resolver_checks = resolver.self_test()
    require(resolver_checks["passed"], "resolver self-test")
    require(all(resolver_checks["mutation_rejections"].values()), "resolver mutation")

    access_dir = core.OUT / "holdout_access"
    access_files = list(access_dir.rglob("*")) if access_dir.exists() else []
    holdout_checks = {
        "commitment_status_not_accessed": (
            holdout["first_model_evaluation_access_status"] == "not_accessed"
        ),
        "no_holdout_access_files": not access_files,
        "holdout_not_claimed_blind": holdout["holdout_semantics"] == "preregistered_immutable_not_blind",
        "rules_frozen_before_future_access": all(
            isinstance(holdout[field], str) and len(holdout[field]) == 64
            for field in (
                "candidate_set_sha256", "search_candidate_set_sha256",
                "equivalence_rule_sha256", "thresholds_sha256",
            )
        ),
    }
    require(all(holdout_checks.values()), f"holdout checks: {holdout_checks}")

    checks = {
        **source_checks,
        "extension_independent_pass": all(
            extension_checks[key] == 0
            for key in ("abstract_overlap", "observable_overlap", "normalized_prompt_overlap")
        ),
        "physical_split_independent_pass": split_checks["combined_records"] == 14336,
        "tokenizer_independent_pass": tokenizer_checks["passed"],
        "model_files_independent_pass": model_checks["passed"],
        "resolver_independent_pass": resolver_checks["passed"],
        "resolver_all_mutations_rejected": all(resolver_checks["mutation_rejections"].values()),
        "holdout_gate_independent_pass": all(holdout_checks.values()),
        "formal_gpu_runner_absent": not (core.ROOT / "tests/glm5/phase991_gpu_behavior_runner.py").exists(),
        "model_weights_not_loaded": True,
        "cuda_not_used": True,
    }
    require(all(checks.values()), f"independent audit failed: {checks}")
    return {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "independent_cpu_execution_audit_before_gpu_runner_creation",
        "passed": True,
        "checks": checks,
        "source_and_stage": source_checks,
        "extension": extension_checks,
        "physical_splits": split_checks,
        "tokenizers": tokenizer_checks,
        "model_artifacts": model_checks,
        "resolver": resolver_checks,
        "holdout": holdout_checks,
        "scientific_decision": {
            "cpu_gpu_admission_package": "qualified",
            "gpu_runner_creation_authorized_after_freeze_publish": True,
            "formal_gpu_model_execution_authorized": False,
            "reason": "runner source and activation artifact do not yet exist",
            "internal_structure_discovered": False,
            "causal_mechanism_discovered": False,
            "mechanism_formula_authorized": False,
        },
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def run_audit() -> dict[str, Any]:
    require(not AUDIT_PATH.exists() and not FREEZE_PATH.exists(), "audit/freeze already exists")
    created = datetime.now(timezone.utc).isoformat()
    payload = audit_payload()
    audit = core.sealed_document(payload, "independent_audit_sha256", created)
    protocol._write_json(AUDIT_PATH, audit)
    freeze = core.sealed_document({
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "qualified_cpu_admission_freeze_commit",
        "stage_commit": core.artifact_seal(core.OUT / protocol.STAGE_COMMIT_PATH, core.OUT),
        "gpu_admission": core.artifact_seal(core.OUT / protocol.ADMISSION_PATH, core.OUT),
        "independent_audit": core.artifact_seal(AUDIT_PATH, core.OUT),
        "independent_audit_sha256": audit["independent_audit_sha256"],
        "cpu_gpu_admission_package": "qualified",
        "gpu_runner_creation_authorized": True,
        "formal_gpu_model_execution_authorized": False,
        "formal_gpu_runner_created": False,
        "gpu_model_run_count": 0,
        "holdout_model_access_count": 0,
        "internal_trace_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_formula_authorized": False,
        "model_weights_loaded": False,
        "cuda_used": False,
    }, "freeze_commit_sha256", created)
    protocol._write_json(FREEZE_PATH, freeze)
    return {
        "passed": True,
        "independent_audit_sha256": audit["independent_audit_sha256"],
        "freeze_commit_sha256": freeze["freeze_commit_sha256"],
        "gpu_runner_creation_authorized": True,
        "formal_gpu_model_execution_authorized": False,
    }


def verify_audit() -> dict[str, Any]:
    audit = core.load_json(AUDIT_PATH)
    freeze = core.load_json(FREEZE_PATH)
    core.verify_self_hash(audit, "independent_audit_sha256")
    core.verify_self_hash(freeze, "freeze_commit_sha256")
    require(audit["passed"] is True, "audit not passed")
    require(freeze["cpu_gpu_admission_package"] == "qualified", "freeze not qualified")
    for field in ("stage_commit", "gpu_admission", "independent_audit"):
        _verify_file_entry(core.OUT, freeze[field])
    require(
        freeze["independent_audit_sha256"] == audit["independent_audit_sha256"],
        "audit payload bridge mismatch",
    )
    require(freeze["formal_gpu_model_execution_authorized"] is False, "premature GPU auth")
    return {
        "passed": True,
        "independent_audit_sha256": audit["independent_audit_sha256"],
        "freeze_commit_sha256": freeze["freeze_commit_sha256"],
        "gpu_runner_creation_authorized": freeze["gpu_runner_creation_authorized"],
        "formal_gpu_model_execution_authorized": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    result = run_audit() if arguments.run else verify_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
