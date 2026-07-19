#!/usr/bin/env python3
"""Phase 977 exploratory 512->1024 budget audit for frozen discovery failures.

This script is deliberately not a gate.  It selects only the discovery
hard/soft-thinking trajectories that exhausted the preregistered 512-token
budget without EOS, then reruns each original official prompt with the frozen
seed and a 1024-token limit.  A result is accepted only when its first 512
generated token IDs exactly replay the stored discovery trajectory.

All reported rates are conditional on the frozen ``hit512`` subset.  They do
not revise the discovery decision and cannot authorize span, layer, residual,
or cross-time mechanism experiments.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, load_model, release_model
from phase973_conditional_trajectory import get_eos_ids
from phase977_legal_mode_trajectories import (
    CONDITIONS,
    GENERATED_MODE_PARSER_VERSION,
    analyze_generation,
    append_jsonl,
    atomic_write_json,
    generate_stage,
    load_jsonl,
    normalize_item,
    render_prefix,
    sha256_json,
    single_token_id,
    stable_item_seed,
    utc_now,
)


PHASE = 977
MODEL_NAME = "qwen3"
SOURCE_SPLIT = "discovery"
SOURCE_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
SOURCE_BUDGET = 512
AUDIT_BUDGET = 1024
EXPECTED_TRAJECTORIES = 36
EXPECTED_UNIQUE_ITEMS = 25
EXPECTED_PER_CONDITION = 18
BASE_SEED = 977_000
SELECTED_CONDITIONS = ("hard_thinking", "soft_thinking")
STAGE = "exploratory1024"

SOURCE_DIR = (
    ROOT / "tests" / "glm5" / "result" / "phase977_legal_mode_trajectories"
)
SOURCE_MANIFEST = SOURCE_DIR / "manifest_discovery.json"
SOURCE_ROWS = SOURCE_DIR / "rows_discovery.jsonl"
SOURCE_SUMMARY = SOURCE_DIR / "summary_discovery.json"
SOURCE_REAUDIT = SOURCE_DIR / "reaudit_discovery.json"
LEGAL_SCRIPT = ROOT / "tests" / "glm5" / "phase977_legal_mode_trajectories.py"
OUT = ROOT / "tests" / "glm5" / "result" / "phase977_thinking_budget_audit"
MANIFEST_PATH = OUT / "manifest.json"
ROWS_PATH = OUT / "rows.jsonl"
SUMMARY_PATH = OUT / "summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be one JSON object: {path}")
    return value


def read_source_jsonl(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing discovery rows: {path}")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid discovery JSONL line {line_number}: {exc}") from exc
            key = (str(row.get("id")), str(row.get("condition")), str(row.get("stage")))
            if key in records:
                raise RuntimeError(f"duplicate discovery stage key: {key}")
            records[key] = row
    return records


def source_artifact_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in (SOURCE_MANIFEST, SOURCE_ROWS, SOURCE_SUMMARY, SOURCE_REAUDIT)
    }


def local_model_artifact_identity() -> dict[str, Any]:
    model_dir = Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()
    names = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "merges.txt",
        "vocab.json",
    ]
    names.extend(sorted(path.name for path in model_dir.glob("*.safetensors")))
    artifacts: dict[str, Any] = {}
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"required local Qwen3 artifact is absent: {path}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"model_dir": str(model_dir), "files": artifacts}


def verify_reaudit_source_hashes(
    reaudit: dict[str, Any], hashes: dict[str, str]
) -> None:
    source_artifacts = reaudit.get("source_artifacts", {})
    for path in (SOURCE_MANIFEST, SOURCE_ROWS, SOURCE_SUMMARY):
        snapshot = source_artifacts.get(path.name, {})
        current = hashes[path.name]
        if not (
            snapshot.get("unchanged") is True
            and snapshot.get("sha256_before") == current
            and snapshot.get("sha256_after") == current
        ):
            raise RuntimeError(
                f"discovery artifact changed after strict re-audit: {path.name}")


def frozen_hit512_rows(
    records: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in records.values()
        if row.get("stage") == "extended512"
        and row.get("condition") in SELECTED_CONDITIONS
        and row.get("hit512") is True
    ]
    selected.sort(key=lambda row: (str(row["id"]), str(row["condition"])))
    return selected


def audit_source_without_model() -> dict[str, Any]:
    """Authenticate selection and protocol before any model weights are loaded."""
    for path in (
        SOURCE_MANIFEST, SOURCE_ROWS, SOURCE_SUMMARY, SOURCE_REAUDIT, LEGAL_SCRIPT
    ):
        if not path.is_file():
            raise RuntimeError(f"required frozen source is absent: {path}")

    manifest = load_json_object(SOURCE_MANIFEST, "discovery manifest")
    summary = load_json_object(SOURCE_SUMMARY, "discovery summary")
    reaudit = load_json_object(SOURCE_REAUDIT, "strict-v2 discovery re-audit")
    hashes = source_artifact_hashes()
    verify_reaudit_source_hashes(reaudit, hashes)

    downstream = reaudit.get("downstream_gate", {})
    if not (
        reaudit.get("phase") == PHASE
        and reaudit.get("split") == SOURCE_SPLIT
        and reaudit.get("execution_contract", {}).get("cpu_only") is True
        and reaudit.get("execution_contract", {}).get("model_weights_loaded") is False
        and reaudit.get("execution_contract", {}).get("generation_performed") is False
        and reaudit.get("migration", {}).get("target_schema_version") == 2
        and reaudit.get("migration", {}).get("target_parser_version")
        == GENERATED_MODE_PARSER_VERSION
        and reaudit.get("target_identity", {}).get("legal_schema_version") == 2
        and reaudit.get("target_identity", {}).get(
            "generated_mode_parser_version") == GENERATED_MODE_PARSER_VERSION
        and reaudit.get("target_identity", {}).get("legal_script_sha256")
        == sha256_file(LEGAL_SCRIPT)
        and reaudit.get("strict_v2_summary_recomputed", {}).get("complete") is True
        and reaudit.get("strict_v2_summary_recomputed", {}).get(
            "decision_gate", {}).get("passed") is False
        and downstream.get("status") == "NO-GO"
        and downstream.get("no_go") is True
        and downstream.get("strict_v2_gate_passed") is False
    ):
        raise RuntimeError(
            "budget audit is permitted only as an exploratory audit after the "
            "strict-v2 discovery NO-GO")

    if not (
        manifest.get("schema_version") == SOURCE_SCHEMA_VERSION
        and manifest.get("phase") == PHASE
        and manifest.get("model") == MODEL_NAME
        and manifest.get("split") == SOURCE_SPLIT
        and manifest.get("n_items") == 80
        and manifest.get("base_seed") == BASE_SEED
        and manifest.get("budgets") == {"initial": 256, "extended": SOURCE_BUDGET}
        and manifest.get("conditions") == CONDITIONS
        and summary.get("schema_version") == SOURCE_SCHEMA_VERSION
        and summary.get("manifest_sha256") == manifest.get("manifest_sha256")
        and summary.get("complete") is True
        and summary.get("jsonl_stage_rows") == 416
        and summary.get("final_rows_available") == 320
        and summary.get("decision_gate", {}).get("passed") is False
        and reaudit.get("source_identity", {}).get("manifest_self_hash_valid") is True
        and reaudit.get("source_identity", {}).get(
            "stored_summary_reproduced_exact") is True
    ):
        raise RuntimeError("frozen discovery manifest/summary identity audit failed")

    records = read_source_jsonl(SOURCE_ROWS)
    if len(records) != 416:
        raise RuntimeError(f"expected 416 frozen source rows, found {len(records)}")
    if any(
        row.get("manifest_sha256") != manifest.get("manifest_sha256")
        or row.get("schema_version") != SOURCE_SCHEMA_VERSION
        or row.get("split") != SOURCE_SPLIT
        for row in records.values()
    ):
        raise RuntimeError("one or more source rows failed manifest/schema authentication")

    selected = frozen_hit512_rows(records)
    counts = Counter(str(row["condition"]) for row in selected)
    unique_ids = sorted({str(row["id"]) for row in selected})
    if not (
        len(selected) == EXPECTED_TRAJECTORIES
        and len(unique_ids) == EXPECTED_UNIQUE_ITEMS
        and counts == Counter({name: EXPECTED_PER_CONDITION for name in SELECTED_CONDITIONS})
    ):
        raise RuntimeError(
            "frozen hit512 selection drifted: "
            f"n={len(selected)}, unique={len(unique_ids)}, conditions={dict(counts)}")

    for row in selected:
        condition = str(row["condition"])
        item_id = str(row["id"])
        initial = records.get((item_id, condition, "initial256"))
        generated = [int(x) for x in row.get("generated_ids", [])]
        initial_generated = [] if initial is None else [
            int(x) for x in initial.get("generated_ids", [])
        ]
        expected_seed = stable_item_seed(BASE_SEED, SOURCE_SPLIT, item_id)
        if not (
            initial is not None
            and row.get("max_new_tokens") == SOURCE_BUDGET
            and row.get("n_tokens") == SOURCE_BUDGET
            and len(generated) == SOURCE_BUDGET
            and row.get("hit_budget") is True
            and row.get("hit512") is True
            and row.get("has_eos") is False
            and row.get("seed") == expected_seed
            and initial.get("seed") == expected_seed
            and row.get("extension_replayed_initial256_exact") is True
            and generated[:len(initial_generated)] == initial_generated
            and row.get("prompt") == initial.get("prompt")
            and row.get("answer") == initial.get("answer")
            and row.get("alias_groups") == initial.get("alias_groups")
            and row.get("exact") == initial.get("exact")
            and row.get("input_ids") == initial.get("input_ids")
        ):
            raise RuntimeError(f"invalid frozen hit512 source row: {item_id}/{condition}")

    hashes_after = source_artifact_hashes()
    if hashes_after != hashes:
        raise RuntimeError("frozen source artifacts changed during CPU audit")

    return {
        "manifest": manifest,
        "summary": summary,
        "reaudit": reaudit,
        "hashes": hashes,
        "records": records,
        "selected": selected,
        "selected_keys": [f"{row['id']}/{row['condition']}" for row in selected],
        "unique_ids": unique_ids,
        "condition_counts": dict(counts),
    }


def load_preflight_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL_NAME]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def item_from_source(row: dict[str, Any]) -> dict[str, Any]:
    return normalize_item({
        "id": row["id"],
        "task": row["task"],
        "prompt": row["prompt"],
        "answer": row["answer"],
        "alias_groups": row["alias_groups"],
        "exact": row["exact"],
    })


def audit_source_with_tokenizer(source: dict[str, Any], tok) -> dict[str, Any]:
    manifest = source["manifest"]
    eos_ids = [int(x) for x in manifest.get("eos_token_ids", [])]
    think_open_id = single_token_id(tok, "<think>")
    think_close_id = single_token_id(tok, "</think>")
    if not (
        eos_ids
        and manifest.get("special_token_ids", {}).get("think_open") == think_open_id
        and manifest.get("special_token_ids", {}).get("think_close") == think_close_id
    ):
        raise RuntimeError("frozen special-token identity failed")

    strict_source = []
    for row in source["selected"]:
        item = item_from_source(row)
        condition = str(row["condition"])
        user_prompt, rendered, input_ids = render_prefix(tok, item, condition)
        if not (
            input_ids == [int(x) for x in row["input_ids"]]
            and user_prompt == row["effective_user_prompt"]
            and rendered == tok.decode(input_ids, skip_special_tokens=False)
        ):
            raise RuntimeError(f"official prefix mismatch: {row['id']}/{condition}")
        analysis = analyze_generation(
            tok, item, condition, [int(x) for x in row["generated_ids"]],
            eos_ids, think_open_id, think_close_id, SOURCE_BUDGET,
        )
        if not (
            analysis["has_eos"] is False
            and analysis["hit_budget"] is True
            and analysis["n_tokens"] == SOURCE_BUDGET
        ):
            raise RuntimeError(f"strict replay no longer identifies hit512: {row['id']}/{condition}")
        strict_source.append({
            "id": row["id"],
            "condition": condition,
            "mode_valid_at_512": analysis["mode_valid"],
            "think_well_formed_at_512": analysis["think_well_formed"],
            "final_region_valid_at_512": analysis["final_region_valid"],
        })
    return {
        "eos_ids": eos_ids,
        "think_open_id": think_open_id,
        "think_close_id": think_close_id,
        "strict_source": strict_source,
    }


def make_manifest(
    source: dict[str, Any], tokenizer_audit: dict[str, Any], model, tok
) -> dict[str, Any]:
    core = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_thinking_hit512_to1024_budget_audit",
        "evidence_class": "exploratory_conditional_budget_audit_only",
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "tokenizer_class": type(tok).__name__,
        "source_split": SOURCE_SPLIT,
        "source_manifest_sha256": source["manifest"]["manifest_sha256"],
        "source_artifact_file_sha256": source["hashes"],
        "source_generator_legal_script_sha256": source["manifest"]["script_sha256"],
        "current_strict_evaluator_script_sha256": sha256_file(LEGAL_SCRIPT),
        "strict_reaudit_target_script_sha256": source["reaudit"][
            "target_identity"]["legal_script_sha256"],
        "script_sha256": sha256_file(Path(__file__)),
        "source_gate_status": "NO-GO",
        "source_strict_v2_gate_passed": False,
        "source_budget": SOURCE_BUDGET,
        "audit_budget": AUDIT_BUDGET,
        "selection_rule": (
            "frozen discovery extended512 rows where condition is hard_thinking "
            "or soft_thinking and hit512 is true"
        ),
        "selected_conditions": list(SELECTED_CONDITIONS),
        "selected_trajectories": EXPECTED_TRAJECTORIES,
        "selected_unique_items": EXPECTED_UNIQUE_ITEMS,
        "selected_condition_counts": source["condition_counts"],
        "selected_keys": source["selected_keys"],
        "base_seed": BASE_SEED,
        "seed_rule": "reuse exact frozen discovery item seed",
        "generation_strategy": (
            "rerun original official prefix from seed to max_new_tokens=1024; "
            "accept only if generated[:512] exactly equals frozen extended512"
        ),
        "sampling": {
            name: {key: CONDITIONS[name][key] for key in
                   ("temperature", "top_p", "top_k", "min_p")}
            for name in SELECTED_CONDITIONS
        },
        "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
        "eos_token_ids": tokenizer_audit["eos_ids"],
        "special_token_ids": {
            "think_open": tokenizer_audit["think_open_id"],
            "think_close": tokenizer_audit["think_close_id"],
        },
        "runtime_identity": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
            "model_dtype": str(getattr(model, "dtype", "")),
            "attention_implementation": str(
                getattr(model.config, "_attn_implementation", "")),
            "model_config_sha256": sha256_json(model.config.to_dict()),
            "tokenizer_vocab_sha256": sha256_json(tok.get_vocab()),
            "tokenizer_length": len(tok),
            "deterministic_algorithms_enabled":
                torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "local_model_artifacts": local_model_artifact_identity(),
        },
        "endpoint": "P(outcome by 1024 | frozen discovery hit512)",
        "original_discovery_gate_unchanged": True,
        "mechanism_authorized": False,
        "holdout_loaded": False,
    }
    return {
        **core,
        "manifest_sha256": sha256_json(core),
        "created_at_utc": utc_now(),
    }


def install_or_validate_manifest(manifest: dict[str, Any]) -> None:
    if MANIFEST_PATH.exists():
        prior = load_json_object(MANIFEST_PATH, "budget audit manifest")
        prior_core = {
            key: value for key, value in prior.items()
            if key not in {"manifest_sha256", "created_at_utc"}
        }
        if not (
            prior.get("manifest_sha256") == sha256_json(prior_core)
            and prior.get("manifest_sha256") == manifest["manifest_sha256"]
        ):
            raise RuntimeError(
                "budget-audit manifest self-hash/mismatch; refusing to mix runs")
        return
    atomic_write_json(MANIFEST_PATH, manifest)


def validate_output_row(
    row: dict[str, Any], source_row: dict[str, Any], manifest: dict[str, Any],
    tok, eos_ids: list[int], think_open_id: int, think_close_id: int,
) -> None:
    item = item_from_source(source_row)
    condition = str(source_row["condition"])
    generated = [int(x) for x in row.get("generated_ids", [])]
    frozen512 = [int(x) for x in source_row["generated_ids"]]
    recomputed = analyze_generation(
        tok, item, condition, generated, eos_ids,
        think_open_id, think_close_id, AUDIT_BUDGET,
    )
    eos_positions = [int(x) for x in recomputed["eos_positions"]]
    termination_valid = bool(
        SOURCE_BUDGET < len(generated) <= AUDIT_BUDGET
        and (
            (len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
            or (
                not eos_positions
                and len(generated) == AUDIT_BUDGET
                and recomputed["hit_budget"] is True
            )
        )
    )
    expected_sampling = {
        key: CONDITIONS[condition][key]
        for key in ("temperature", "top_p", "top_k", "min_p")
    }
    if not (
        row.get("manifest_sha256") == manifest["manifest_sha256"]
        and row.get("schema_version") == OUTPUT_SCHEMA_VERSION
        and row.get("phase") == PHASE
        and row.get("stage") == STAGE
        and row.get("id") == source_row["id"]
        and row.get("condition") == condition
        and row.get("seed") == source_row["seed"]
        and row.get("source_split") == SOURCE_SPLIT
        and row.get("max_new_tokens") == AUDIT_BUDGET
        and row.get("input_ids") == source_row["input_ids"]
        and row.get("prompt_len") == len(source_row["input_ids"])
        and row.get("task") == source_row["task"]
        and row.get("prompt") == source_row["prompt"]
        and row.get("effective_user_prompt") == source_row["effective_user_prompt"]
        and row.get("answer") == source_row["answer"]
        and row.get("alias_groups") == source_row["alias_groups"]
        and row.get("exact") == source_row["exact"]
        and row.get("sampling") == expected_sampling
        and row.get("source_stage") == "extended512"
        and row.get("source_hit512") is True
        and row.get("replay_512_exact") is True
        and generated[:SOURCE_BUDGET] == frozen512
        and row.get("source_extended512_generated_ids_sha256") == sha256_json(frozen512)
        and row.get("hit1024") == recomputed["hit_budget"]
        and row.get("endpoint_is_conditional_on_hit512") is True
        and row.get("original_discovery_gate_unchanged") is True
        and row.get("mechanism_authorized") is False
        and termination_valid
        and all(row.get(key) == recomputed[key] for key in recomputed)
    ):
        raise RuntimeError(
            f"invalid resumed audit row: {source_row['id']}/{condition}")


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    bool_metrics = (
        "replay_512_exact", "has_eos", "semantic_match", "mode_valid",
        "valid_eos", "valid_mode_eos", "think_well_formed",
        "final_region_valid", "hit1024",
    )
    out: dict[str, Any] = {
        "n": len(rows),
        "unique_items_n": len({str(row["id"]) for row in rows}),
    }
    for key in bool_metrics:
        numerator = sum(bool(row.get(key, False)) for row in rows)
        out[f"{key}_n"] = numerator
        out[f"{key}_rate"] = numerator / len(rows)
    closed = sum(bool(row.get("generated_think_close_positions")) for row in rows)
    out["any_think_close_n"] = closed
    out["any_think_close_rate"] = closed / len(rows)
    eos_steps = [int(row["first_eos_step"]) for row in rows
                 if row.get("first_eos_step") is not None]
    out["mean_first_eos_step_among_eos"] = (
        float(np.mean(eos_steps)) if eos_steps else None)
    out["min_first_eos_step"] = min(eos_steps) if eos_steps else None
    out["max_first_eos_step"] = max(eos_steps) if eos_steps else None
    out["mean_tokens"] = float(np.mean([int(row["n_tokens"]) for row in rows]))
    return out


def write_summary(
    manifest: dict[str, Any], records: dict[tuple[str, str, str], dict[str, Any]],
    selected: list[dict[str, Any]], elapsed_seconds: float,
) -> None:
    expected_keys = {
        (str(row["id"]), str(row["condition"]), STAGE) for row in selected
    }
    rows = [records[key] for key in sorted(expected_keys) if key in records]
    by_condition: dict[str, Any] = {}
    for condition in SELECTED_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_task = {
            task: metric_block([row for row in condition_rows if row["task"] == task])
            for task in sorted({str(row["task"]) for row in condition_rows})
        }
        by_condition[condition] = {
            "overall": metric_block(condition_rows),
            "by_task": by_task,
        }
    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": manifest["experiment"],
        "manifest_sha256": manifest["manifest_sha256"],
        "complete": len(rows) == EXPECTED_TRAJECTORIES,
        "expected_conditional_trajectories": EXPECTED_TRAJECTORIES,
        "completed_conditional_trajectories": len(rows),
        "selected_unique_items": EXPECTED_UNIQUE_ITEMS,
        "endpoint": manifest["endpoint"],
        "overall": metric_block(rows),
        "conditions": by_condition,
        "elapsed_seconds_this_invocation": float(elapsed_seconds),
        "interpretation_contract": {
            "exploratory_only": True,
            "conditional_on_frozen_hit512": True,
            "original_discovery_gate_unchanged": True,
            "mechanism_authorized": False,
            "new_go_no_go_computed": False,
            "holdout_loaded": False,
        },
        "decision_status": "EXPLORATORY_ONLY_NO_GATE",
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(SUMMARY_PATH, summary)


def run(cpu_audit_only: bool) -> None:
    t0 = time.time()
    source = audit_source_without_model()
    preflight_tok = load_preflight_tokenizer()
    try:
        tokenizer_audit = audit_source_with_tokenizer(source, preflight_tok)
    finally:
        del preflight_tok
        gc.collect()

    if cpu_audit_only:
        print(json.dumps({
            "source_audit_passed": True,
            "selected_trajectories": len(source["selected"]),
            "selected_unique_items": len(source["unique_ids"]),
            "condition_counts": source["condition_counts"],
            "strict_parser": GENERATED_MODE_PARSER_VERSION,
            "model_weights_loaded": False,
            "generation_performed": False,
            "original_discovery_gate_unchanged": True,
            "mechanism_authorized": False,
        }, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("formal budget audit requires local CUDA")

    model = None
    try:
        model, tok, device = load_model(MODEL_NAME)
        if getattr(device, "type", str(device).split(":")[0]) != "cuda":
            raise RuntimeError(f"Qwen3 was not loaded on CUDA: {device}")
        runtime_eos = [int(x) for x in get_eos_ids(model, tok)]
        think_open_id = single_token_id(tok, "<think>")
        think_close_id = single_token_id(tok, "</think>")
        if not (
            runtime_eos == tokenizer_audit["eos_ids"]
            and think_open_id == tokenizer_audit["think_open_id"]
            and think_close_id == tokenizer_audit["think_close_id"]
        ):
            raise RuntimeError("runtime token identity differs from frozen discovery")

        manifest = make_manifest(source, tokenizer_audit, model, tok)
        install_or_validate_manifest(manifest)
        records = load_jsonl(ROWS_PATH, manifest["manifest_sha256"])
        expected_keys = {
            (str(row["id"]), str(row["condition"]), STAGE)
            for row in source["selected"]
        }
        unexpected = set(records) - expected_keys
        if unexpected:
            raise RuntimeError(f"unexpected output row keys: {sorted(unexpected)}")
        nonempty_lines = 0
        if ROWS_PATH.exists():
            nonempty_lines = sum(
                bool(line.strip())
                for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            )
        if nonempty_lines != len(records):
            raise RuntimeError("duplicate budget-audit JSONL stage keys detected")

        source_by_key = {
            (str(row["id"]), str(row["condition"]), STAGE): row
            for row in source["selected"]
        }
        for key, row in records.items():
            validate_output_row(
                row, source_by_key[key], manifest, tok, runtime_eos,
                think_open_id, think_close_id,
            )
        write_summary(manifest, records, source["selected"], time.time() - t0)

        for index, source_row in enumerate(source["selected"], 1):
            key = (str(source_row["id"]), str(source_row["condition"]), STAGE)
            if key not in records:
                item = item_from_source(source_row)
                condition = str(source_row["condition"])
                seed = int(source_row["seed"])
                input_ids, generated, user_prompt = generate_stage(
                    model, tok, device, runtime_eos, item, condition, seed,
                    AUDIT_BUDGET,
                )
                frozen512 = [int(x) for x in source_row["generated_ids"]]
                if not (
                    input_ids == [int(x) for x in source_row["input_ids"]]
                    and user_prompt == source_row["effective_user_prompt"]
                    and generated[:SOURCE_BUDGET] == frozen512
                ):
                    raise RuntimeError(
                        f"1024 rerun failed exact 512-prefix replay: "
                        f"{source_row['id']}/{condition}")
                analysis = analyze_generation(
                    tok, item, condition, generated, runtime_eos,
                    think_open_id, think_close_id, AUDIT_BUDGET,
                )
                row = {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "phase": PHASE,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "source_split": SOURCE_SPLIT,
                    "id": item["id"],
                    "task": item["task"],
                    "condition": condition,
                    "stage": STAGE,
                    "seed": seed,
                    "prompt": item["prompt"],
                    "effective_user_prompt": user_prompt,
                    "answer": item["answer"],
                    "alias_groups": item["alias_groups"],
                    "exact": item["exact"],
                    "sampling": {key: CONDITIONS[condition][key] for key in
                                 ("temperature", "top_p", "top_k", "min_p")},
                    "max_new_tokens": AUDIT_BUDGET,
                    "input_ids": [int(x) for x in input_ids],
                    "prompt_len": len(input_ids),
                    **analysis,
                    "source_stage": "extended512",
                    "source_hit512": True,
                    "source_extended512_generated_ids_sha256": sha256_json(frozen512),
                    "replay_512_exact": True,
                    "hit1024": bool(analysis["hit_budget"]),
                    "endpoint_is_conditional_on_hit512": True,
                    "original_discovery_gate_unchanged": True,
                    "mechanism_authorized": False,
                    "recorded_at_utc": utc_now(),
                }
                validate_output_row(
                    row, source_row, manifest, tok, runtime_eos,
                    think_open_id, think_close_id,
                )
                append_jsonl(ROWS_PATH, row)
                records[key] = row

            if index % 4 == 0 or index == EXPECTED_TRAJECTORIES:
                write_summary(
                    manifest, records, source["selected"], time.time() - t0)
                print(
                    f"Phase977 conditional budget audit {index}/{EXPECTED_TRAJECTORIES}",
                    flush=True,
                )

        write_summary(manifest, records, source["selected"], time.time() - t0)
        print(
            f"Phase977 budget audit complete; elapsed={(time.time()-t0)/60:.1f} min; "
            f"rows={ROWS_PATH}",
            flush=True,
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-audit-only", action="store_true",
        help="Authenticate the frozen hit512 selection without loading model weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.cpu_audit_only)
