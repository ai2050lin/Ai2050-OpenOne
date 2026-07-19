#!/usr/bin/env python3
"""One-time sealed-holdout confirmation for Phase978.

The holdout module is lazy-imported only after a frozen, independently
recomputed development admission PASS has been authenticated and an atomic
OPEN receipt has been installed.  Generation then follows the preregistered
256/512/1024/1536 absorbing schedule with exact same-seed prefix replay.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

from model_utils import MODEL_CONFIGS, load_model, release_model  # noqa: E402
import phase978_legal_core as legal  # noqa: E402
import phase978_budget_protocol as protocol_code  # noqa: E402

log = legal.log
get_eos_ids = legal.get_eos_ids


PHASE = 978
SCHEMA_VERSION = 1
MODEL_NAME = "qwen3"
SPLIT = "holdout"
CHECKPOINTS = (256, 512, 1024, 1536)
STAGES = {
    256: "initial256",
    512: "extended512",
    1024: "extended1024",
    1536: "extended1536",
}
EXPECTED_ITEMS = 128
EXPECTED_TASKS = 8
EXPECTED_PER_TASK = 16
BASE_SEED = 977_000

HOLDOUT_SCRIPT = GLM5 / "phase977_holdout_dataset.py"
DEV_DATASET_SCRIPT = GLM5 / "phase977_dev_dataset.py"
DISCOVERY_ROWS_PATH = (
    ROOT / "tests" / "glm5" / "result" /
    "phase977_legal_mode_trajectories" / "rows_discovery.jsonl"
)
PROTOCOL_SCRIPT = GLM5 / "phase978_budget_protocol.py"
DEV_RUNNER_SCRIPT = GLM5 / "phase978_dev_budget_stabilization.py"
DEV_AUDITOR_SCRIPT = GLM5 / "phase978_dev_admission_audit.py"
W_RUNNER_SCRIPT = GLM5 / "phase978_wrong_answer_safety.py"

OUT = ROOT / "tests" / "glm5" / "result" / "phase978_legal_budget_stabilization"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
DEV_ADMISSION_PATH = OUT / "admission_development.json"
OPEN_RECEIPT_PATH = OUT / "holdout_open_receipt.json"
MANIFEST_PATH = OUT / "manifest_holdout.json"
ROWS_PATH = OUT / "rows_holdout.jsonl"
STATUS_PATH = OUT / "generator_status_holdout.json"
SUMMARY_PATH = OUT / "summary_holdout.json"
RUN_LOCK_PATH = OUT / "holdout_run.lock"

PINNED_HOLDOUT_SCRIPT_SHA256 = (
    "d4d630f00a7c0197f6e7ba83704fdcf13121d67b5b09d3a77d649cb3fdff4755"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not one JSON object")
    return value


def without_fields(value: dict[str, Any], *excluded: str) -> dict[str, Any]:
    return {
        key: item for key, item in value.items()
        if key not in set(excluded)
    }


def script_hashes_by_name(document: dict[str, Any]) -> dict[str, str]:
    values = document.get("phase978_script_hashes", {})
    require(isinstance(values, dict), "protocol lacks Phase978 script hashes")
    return {
        Path(value["path"]).name: str(value["sha256"])
        for value in values.values()
        if isinstance(value, dict) and "path" in value and "sha256" in value
    }


def authenticate_protocol() -> dict[str, Any]:
    document = load_json(PROTOCOL_PATH, "Phase978 preregistration")
    core = without_fields(document, "protocol_sha256", "created_at_utc")
    require(document.get("protocol_sha256") == legal.sha256_json(core),
            "Phase978 preregistration self-hash invalid")
    require(document.get("phase") == PHASE and
            document.get("checkpoints") == list(CHECKPOINTS) and
            document.get("decision_checkpoint") == 1536,
            "Phase978 protocol identity/checkpoints mismatch")
    schedule = document.get("generation_schedule", {})
    require(schedule.get("base_seed") == BASE_SEED and
            schedule.get("early_eos_absorbing") is True,
            "Phase978 seed/absorbing schedule mismatch")
    execution = document.get("execution_contract", {})
    require(execution.get("holdout_module_imported") is False and
            execution.get("holdout_module_parsed") is False,
            "protocol freeze reports holdout access")
    hashes = script_hashes_by_name(document)
    for value in document.get("phase978_script_hashes", {}).values():
        require(isinstance(value, dict) and "path" in value and "sha256" in value,
                "invalid Phase978 script commitment")
        script_path = ROOT / str(value["path"])
        require(script_path.is_file() and sha256_file(script_path) == value["sha256"],
                f"Phase978 script changed after freeze: {value.get('path')}")
    for path in (PROTOCOL_SCRIPT, DEV_RUNNER_SCRIPT, DEV_AUDITOR_SCRIPT,
                 W_RUNNER_SCRIPT, Path(__file__)):
        require(hashes.get(path.name) == sha256_file(path),
                f"Phase978 code changed after freeze: {path.name}")
    source_commitments = document.get("phase977_frozen_sources", {})
    for commitment in source_commitments.values():
        require(isinstance(commitment, dict) and "path" in commitment and
                "sha256" in commitment, "invalid frozen source commitment")
        source_path = ROOT / str(commitment["path"])
        require(source_path.is_file() and
                sha256_file(source_path) == commitment["sha256"],
                f"frozen execution/source artifact changed: {commitment.get('path')}")
    holdout_commitment = source_commitments.get("holdout_dataset_module_opaque", {})
    require(holdout_commitment.get("sha256") == PINNED_HOLDOUT_SCRIPT_SHA256 and
            sha256_file(HOLDOUT_SCRIPT) == PINNED_HOLDOUT_SCRIPT_SHA256,
            "opaque holdout module commitment mismatch")
    expected_runtime = document.get("runtime_versions", {})
    current_runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }
    require(current_runtime == expected_runtime,
            "Python/torch/transformers runtime differs from frozen protocol")
    model_identity = document.get("local_model_artifact_identity", {})
    model_root = ROOT / str(model_identity.get("path", ""))
    model_files = model_identity.get("files", {})
    require(model_root.is_dir() and isinstance(model_files, dict) and model_files,
            "protocol lacks local model identity")
    require(Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve() == model_root.resolve(),
            "model loader registry path differs from frozen model path")
    for name, expected in model_files.items():
        path = model_root / name
        require(path.is_file() and path.stat().st_size == expected.get("bytes") and
                sha256_file(path) == expected.get("sha256"),
                f"local model artifact changed after protocol freeze: {name}")
    return document


def authenticate_admission(protocol: dict[str, Any]) -> dict[str, Any]:
    """Re-run the frozen CPU auditor before trusting its installed artifact."""
    admission = load_json(DEV_ADMISSION_PATH, "Phase978 development admission")
    core = without_fields(admission, "admission_sha256", "audited_at_utc")
    require(admission.get("admission_sha256") == legal.sha256_json(core),
            "development admission self-hash invalid")
    require(admission.get("protocol_sha256") == protocol["protocol_sha256"] and
            admission.get("holdout_authorized") is True and
            admission.get("decision_gate", {}).get("passed") is True and
            admission.get("holdout_loaded") is False,
            "development admission does not authorize holdout")

    # Recompute in a separate CPU-only process.  The auditor deliberately hides
    # CUDA; importing it here would mutate this generation process's environment.
    require("phase977_holdout_dataset" not in sys.modules,
            "holdout was imported before admission re-audit")
    audit_env = os.environ.copy()
    audit_env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, str(DEV_AUDITOR_SCRIPT)],
        cwd=str(ROOT), env=audit_env, capture_output=True, text=True,
        encoding="utf-8", timeout=900, check=False,
    )
    require(completed.returncode == 0,
            "independent development admission subprocess failed")
    try:
        recomputed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("independent admission subprocess returned invalid JSON") from exc
    require("phase977_holdout_dataset" not in sys.modules,
            "development re-audit imported holdout")
    require(recomputed.get("admission_sha256") == admission["admission_sha256"] and
            recomputed.get("holdout_authorized") is True and
            recomputed.get("passed") is True,
            "installed and recomputed development admissions disagree")
    return admission


def install_or_validate_open_receipt(
    protocol: dict[str, Any], admission: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "state": "OPEN",
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "development_gate_passed": True,
        "holdout_dataset_module_sha256": sha256_file(HOLDOUT_SCRIPT),
        "holdout_runner_sha256": sha256_file(Path(__file__)),
        "opening_policy": (
            "one atomic receipt after frozen independent development PASS; "
            "all resumes must retain this exact receipt"
        ),
    }
    document = {
        **core,
        "receipt_sha256": legal.sha256_json(core),
        "opened_at_utc": legal.utc_now(),
    }
    if OPEN_RECEIPT_PATH.exists():
        prior = load_json(OPEN_RECEIPT_PATH, "holdout OPEN receipt")
        require(prior.get("receipt_sha256") == legal.sha256_json(
            without_fields(prior, "receipt_sha256", "opened_at_utc")),
            "existing OPEN receipt self-hash invalid")
        require(prior.get("receipt_sha256") == document["receipt_sha256"],
                "existing OPEN receipt differs; refusing a second opening")
        return prior
    OUT.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2)
               + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            OPEN_RECEIPT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # An existing file can only appear here if an external process ignored
        # the run lock. Fail closed instead of treating it as this invocation's open.
        raise RuntimeError("OPEN receipt appeared concurrently; refusing holdout access")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "failed to write exclusive OPEN receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return document


def acquire_run_lock(protocol: dict[str, Any], admission: dict[str, Any]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            RUN_LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "holdout run lock already exists; concurrent run or unaudited stale lock"
        ) from exc
    payload = (legal.canonical_json({
        "pid": os.getpid(),
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "acquired_at_utc": legal.utc_now(),
    }) + "\n").encode("utf-8")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "failed to write holdout run lock")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        RUN_LOCK_PATH.unlink(missing_ok=True)
        raise
    return descriptor


def release_run_lock(descriptor: int) -> None:
    os.close(descriptor)
    RUN_LOCK_PATH.unlink(missing_ok=True)


def load_opened_holdout() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sole function allowed to import and execute the holdout module."""
    require(OPEN_RECEIPT_PATH.is_file(), "holdout cannot load without OPEN receipt")
    payload = DISCOVERY_ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"), "frozen discovery rows are truncated")
    discovery_prompts: dict[str, str] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"frozen discovery row {line_number} is malformed") from exc
        item_id = str(row.get("id"))
        prompt = str(row.get("prompt"))
        if item_id in discovery_prompts:
            require(discovery_prompts[item_id] == prompt,
                    "inconsistent frozen discovery prompt across stages")
        else:
            discovery_prompts[item_id] = prompt
    require(len(discovery_prompts) == 80,
            "frozen discovery prompt denominator is not 80")
    dev_module = importlib.import_module("phase977_dev_dataset")
    previous_prompts = list(discovery_prompts.values())
    previous_prompts.extend(str(item["prompt"]) for item in dev_module.build_dataset())
    module = importlib.import_module("phase977_holdout_dataset")
    try:
        raw_audit = module.audit_dataset(previous_prompts=previous_prompts)
        raw_items = module.build_dataset()
    except Exception as exc:
        raise RuntimeError("opened holdout dataset audit/build failed (details redacted)") from exc
    require(isinstance(raw_audit, dict) and raw_audit.get("ok") is True and
            raw_audit.get("passed") is True and not raw_audit.get("errors"),
            "opened holdout dataset failed its frozen audit")
    items = [legal.normalize_item(item) for item in raw_items]
    local = legal.audit_local_items(items)
    require(len(items) == EXPECTED_ITEMS and local["n"] == EXPECTED_ITEMS and
            local["n_tasks"] == EXPECTED_TASKS,
            "opened holdout size/task count mismatch")
    counts = Counter(item["task"] for item in items)
    require(set(counts) == set(protocol_code.TASKS) and
            all(count == EXPECTED_PER_TASK for count in counts.values()),
            "opened holdout is not 8x16 balanced")
    return items, {
        "frozen_audit_passed": True,
        "local_audit": local,
        "n_items": len(items),
        "task_counts": dict(sorted(counts.items())),
        "cross_set_overlap_n": len(raw_audit.get("cross_set_overlap", [])),
        "duplicate_ids_n": len(raw_audit.get("duplicate_ids", [])),
        "duplicate_prompts_n": len(raw_audit.get("duplicate_prompts", [])),
        "details_redacted_from_runtime_log": True,
    }


def make_manifest(
    protocol: dict[str, Any], admission: dict[str, Any], receipt: dict[str, Any],
    items: list[dict[str, Any]], data_audit: dict[str, Any], model, tok,
    eos_ids: list[int], think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_official_budget_stabilization_holdout_confirmation",
        "split": SPLIT,
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "tokenizer_class": type(tok).__name__,
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "open_receipt_sha256": receipt["receipt_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "holdout_module_sha256": sha256_file(HOLDOUT_SCRIPT),
        "n_items": len(items),
        "item_ids": [item["id"] for item in items],
        "dataset_sha256": legal.dataset_hash(items),
        "data_audit": data_audit,
        "conditions": legal.CONDITIONS,
        "condition_order": list(legal.CONDITIONS),
        "checkpoints": list(CHECKPOINTS),
        "decision_checkpoint": 1536,
        "base_seed": BASE_SEED,
        "seed_rule": protocol["generation_schedule"]["seed_rule"],
        "extension_rule": protocol["generation_schedule"]["extension_rule"],
        "replay_requirement": protocol["generation_schedule"]["replay_requirement"],
        "early_eos_absorbing": True,
        "eos_token_ids": [int(value) for value in eos_ids],
        "special_token_ids": {
            "think_open": int(think_open_id),
            "think_close": int(think_close_id),
        },
        "template_tokens": legal.build_template_tokens(
            tok, items[0], think_open_id, think_close_id),
        "generated_mode_parser_version": legal.GENERATED_MODE_PARSER_VERSION,
        "natural_endpoint_rule": "actual sampled terminal EOS only",
        "interim_condition_metrics_withheld": True,
        "mechanism_intervention": False,
    }
    return {
        **core,
        "manifest_sha256": legal.sha256_json(core),
        "created_at_utc": legal.utc_now(),
    }


def install_or_validate_manifest(manifest: dict[str, Any]) -> None:
    if MANIFEST_PATH.exists():
        prior = load_json(MANIFEST_PATH, "Phase978 holdout manifest")
        require(prior.get("manifest_sha256") == legal.sha256_json(
            without_fields(prior, "manifest_sha256", "created_at_utc")),
            "existing holdout manifest self-hash invalid")
        require(prior.get("manifest_sha256") == manifest["manifest_sha256"],
                "holdout manifest mismatch; only exact resume is allowed")
        return
    legal.atomic_write_json(MANIFEST_PATH, manifest)


def load_rows_strict(manifest_sha256: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not ROWS_PATH.exists():
        return {}
    payload = ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"),
            "holdout JSONL has a truncated tail; refusing implicit repair")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(bool(raw.strip()), f"holdout JSONL blank line {line_number}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed holdout JSONL line {line_number}") from exc
        require(isinstance(row, dict), f"holdout row {line_number} is not object")
        require(row.get("manifest_sha256") == manifest_sha256,
                f"holdout row {line_number} manifest mismatch")
        key = legal.row_key(row)
        require(key not in records, f"duplicate holdout stage key {key}")
        records[key] = row
    return records


def build_row(
    manifest: dict[str, Any], tok, item: dict[str, Any], condition: str,
    seed: int, budget: int, input_ids: list[int], generated_ids: list[int],
    user_prompt: str, eos_ids: list[int], think_open_id: int,
    think_close_id: int, parent: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = legal.analyze_generation(
        tok, item, condition, generated_ids, eos_ids,
        think_open_id, think_close_id, budget,
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "open_receipt_sha256": manifest["open_receipt_sha256"],
        "split": SPLIT,
        "id": item["id"],
        "task": item["task"],
        "condition": condition,
        "stage": STAGES[budget],
        "seed": int(seed),
        "prompt": item["prompt"],
        "effective_user_prompt": user_prompt,
        "answer": item["answer"],
        "alias_groups": item["alias_groups"],
        "exact": item["exact"],
        "enable_thinking": bool(legal.CONDITIONS[condition]["enable_thinking"]),
        "sampling": {name: legal.CONDITIONS[condition][name]
                     for name in ("temperature", "top_p", "top_k", "min_p")},
        "max_new_tokens": budget,
        "input_ids": [int(value) for value in input_ids],
        "prompt_len": len(input_ids),
        "prefilled_think_open_positions": legal.positions_of(
            input_ids, {think_open_id}),
        "prefilled_think_close_positions": legal.positions_of(
            input_ids, {think_close_id}),
        **analysis,
        "source_stage": None if parent is None else parent["stage"],
        "source_budget": None if parent is None else parent["max_new_tokens"],
        "source_generated_ids_sha256": (
            None if parent is None else legal.sha256_json(parent["generated_ids"])),
        "extension_strategy": (
            None if parent is None
            else "full_rerun_original_official_prefix_same_seed"),
        "replay_prior_checkpoint_exact": (
            None if parent is None else
            generated_ids[:int(parent["max_new_tokens"])] == parent["generated_ids"]),
        "hit256": bool(analysis["hit_budget"]) if budget == 256 else True,
        "hit512": bool(analysis["hit_budget"]) if budget == 512 else (
            True if budget > 512 else None),
        "hit1024": bool(analysis["hit_budget"]) if budget == 1024 else (
            True if budget > 1024 else None),
        "hit1536": bool(analysis["hit_budget"]) if budget == 1536 else None,
        "natural_endpoint_is_actual_sampled_eos": True,
        "recorded_at_utc": legal.utc_now(),
    }
    return row


def validate_row(
    row: dict[str, Any], manifest: dict[str, Any], tok, item: dict[str, Any],
    condition: str, budget: int, parent: dict[str, Any] | None,
    eos_ids: list[int], think_open_id: int, think_close_id: int,
) -> None:
    key_text = f"{item['id']}/{condition}/{STAGES[budget]}"
    ids = row.get("generated_ids")
    require(isinstance(ids, list) and ids and
            all(isinstance(value, int) and not isinstance(value, bool) for value in ids),
            f"invalid generated IDs {key_text}")
    user_prompt, _rendered, input_ids = legal.render_prefix(tok, item, condition)
    sampling = {name: legal.CONDITIONS[condition][name]
                for name in ("temperature", "top_p", "top_k", "min_p")}
    require(row.get("schema_version") == SCHEMA_VERSION and row.get("phase") == PHASE and
            row.get("manifest_sha256") == manifest["manifest_sha256"] and
            row.get("protocol_sha256") == manifest["protocol_sha256"] and
            row.get("open_receipt_sha256") == manifest["open_receipt_sha256"] and
            row.get("split") == SPLIT and row.get("id") == item["id"] and
            row.get("task") == item["task"] and row.get("condition") == condition and
            row.get("stage") == STAGES[budget] and
            row.get("seed") == legal.stable_item_seed(BASE_SEED, SPLIT, item["id"]) and
            row.get("prompt") == item["prompt"] and
            row.get("effective_user_prompt") == user_prompt and
            row.get("answer") == item["answer"] and
            row.get("alias_groups") == item["alias_groups"] and
            row.get("exact") == item["exact"] and row.get("sampling") == sampling and
            row.get("max_new_tokens") == budget and row.get("input_ids") == input_ids and
            row.get("prompt_len") == len(input_ids) and
            row.get("natural_endpoint_is_actual_sampled_eos") is True,
            f"holdout row metadata mismatch {key_text}")
    if parent is None:
        require(row.get("source_stage") is None and row.get("source_budget") is None and
                row.get("source_generated_ids_sha256") is None and
                row.get("extension_strategy") is None and
                row.get("replay_prior_checkpoint_exact") is None,
                f"invalid initial parent metadata {key_text}")
    else:
        parent_budget = int(parent["max_new_tokens"])
        require(parent.get("hit_budget") is True and
                row.get("source_stage") == parent["stage"] and
                row.get("source_budget") == parent_budget and
                row.get("source_generated_ids_sha256") ==
                    legal.sha256_json(parent["generated_ids"]) and
                row.get("extension_strategy") ==
                    "full_rerun_original_official_prefix_same_seed" and
                row.get("replay_prior_checkpoint_exact") is True and
                ids[:parent_budget] == parent["generated_ids"],
                f"holdout replay mismatch {key_text}")
    rescored = legal.analyze_generation(
        tok, item, condition, ids, eos_ids,
        think_open_id, think_close_id, budget,
    )
    for field, value in rescored.items():
        require(row.get(field) == value, f"derived field {field} mismatch {key_text}")
    eos = rescored["eos_positions"]
    lower = 0 if parent is None else int(parent["max_new_tokens"])
    require((len(eos) == 1 and eos[0] == len(ids) - 1 and lower < len(ids) <= budget) or
            (not eos and len(ids) == budget and rescored["hit_budget"]),
            f"termination mismatch {key_text}")
    expected_hits = {
        "hit256": bool(rescored["hit_budget"]) if budget == 256 else True,
        "hit512": bool(rescored["hit_budget"]) if budget == 512 else
                  (True if budget > 512 else None),
        "hit1024": bool(rescored["hit_budget"]) if budget == 1024 else
                   (True if budget > 1024 else None),
        "hit1536": bool(rescored["hit_budget"]) if budget == 1536 else None,
    }
    require(all(row.get(key) == value for key, value in expected_hits.items()),
            f"checkpoint flags mismatch {key_text}")


def expected_parent(
    records: dict[tuple[str, str, str], dict[str, Any]],
    item_id: str, condition: str, budget: int,
) -> dict[str, Any] | None:
    if budget == 256:
        return None
    previous = CHECKPOINTS[CHECKPOINTS.index(budget) - 1]
    return records.get((item_id, condition, STAGES[previous]))


def validate_resume_key_set(
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    item_ids = {item["id"] for item in items}
    for item_id, condition, stage in records:
        require(item_id in item_ids and condition in legal.CONDITIONS and
                stage in set(STAGES.values()), f"unexpected holdout row key")
        budget = next(value for value, name in STAGES.items() if name == stage)
        if budget > 256:
            parent = expected_parent(records, item_id, condition, budget)
            require(parent is not None and parent.get("hit_budget") is True,
                    f"holdout child exists without cap-hit parent: {item_id}/{condition}/{stage}")


def write_status(manifest: dict[str, Any], records: dict[tuple[str, str, str], dict[str, Any]],
                 items: list[dict[str, Any]], elapsed: float) -> None:
    complete = 0
    for item in items:
        for condition in legal.CONDITIONS:
            for budget in CHECKPOINTS:
                row = records.get((item["id"], condition, STAGES[budget]))
                if row is None:
                    break
                if row["has_eos"] or budget == 1536:
                    complete += 1
                    break
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "split": SPLIT,
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_trajectories": len(items) * len(legal.CONDITIONS),
        "completed_trajectories": complete,
        "raw_stage_rows": len(records),
        "complete": complete == len(items) * len(legal.CONDITIONS),
        "interim_condition_metrics_withheld": True,
        "elapsed_seconds_this_invocation": float(elapsed),
        "updated_at_utc": legal.utc_now(),
    }
    legal.atomic_write_json(STATUS_PATH, status)


def resolve_endpoint(
    records: dict[tuple[str, str, str], dict[str, Any]],
    item_id: str, condition: str, checkpoint: int,
) -> dict[str, Any]:
    for budget in CHECKPOINTS:
        row = records[(item_id, condition, STAGES[budget])]
        if row["has_eos"] or budget == checkpoint:
            return row
    raise RuntimeError("unreachable endpoint resolver")


def metric_block(rows: list[dict[str, Any]], checkpoint: int,
                 replay_by_item: dict[str, bool]) -> dict[str, Any]:
    require(bool(rows), "cannot summarize empty holdout rows")
    n = len(rows)
    out: dict[str, Any] = {"n": n, "checkpoint": checkpoint}
    for key in ("semantic_match", "mode_valid", "has_eos", "valid_eos",
                "valid_mode_eos", "think_well_formed", "final_region_valid"):
        count = sum(bool(row.get(key, False)) for row in rows)
        out[f"{key}_n"] = count
        out[f"{key}_rate"] = count / n
    hit = sum(row["has_eos"] is False and row["n_tokens"] == checkpoint for row in rows)
    out["hit_cap_n"] = hit
    out["hit_cap_rate"] = hit / n
    if checkpoint == 1536:
        out["hit1536_rate"] = hit / n
    replay_n = sum(bool(replay_by_item[str(row["id"])]) for row in rows)
    out["extension_replay_exact_n"] = replay_n
    out["extension_replay_exact_rate"] = replay_n / n
    return out


def finalize_summary(
    protocol: dict[str, Any], admission: dict[str, Any], receipt: dict[str, Any],
    manifest: dict[str, Any], records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]], tok, eos_ids: list[int],
    think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    item_by_id = {item["id"]: item for item in items}
    expected_keys: set[tuple[str, str, str]] = set()
    replay_by_condition: dict[str, dict[str, bool]] = {
        name: {} for name in legal.CONDITIONS
    }
    for item in items:
        for condition in legal.CONDITIONS:
            parent = None
            trajectory_replay_exact = True
            for budget in CHECKPOINTS:
                key = (item["id"], condition, STAGES[budget])
                expected_keys.add(key)
                row = records.get(key)
                require(row is not None, f"incomplete holdout trajectory: {key}")
                validate_row(row, manifest, tok, item, condition, budget, parent,
                             eos_ids, think_open_id, think_close_id)
                if parent is not None:
                    trajectory_replay_exact = (
                        trajectory_replay_exact
                        and bool(row["replay_prior_checkpoint_exact"])
                    )
                if row["has_eos"]:
                    break
                require(row["hit_budget"] is True,
                        f"non-EOS holdout row did not hit budget: {key}")
                parent = row
            replay_by_condition[condition][item["id"]] = trajectory_replay_exact
    require(set(records) == expected_keys, "holdout final stage key set is not exact")

    checkpoints: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        conditions: dict[str, Any] = {}
        for condition in legal.CONDITIONS:
            rows = [resolve_endpoint(records, item["id"], condition, checkpoint)
                    for item in items]
            replay_by_item = replay_by_condition[condition]
            require(len(replay_by_item) == EXPECTED_ITEMS,
                    f"replay full denominator mismatch: {condition}")
            by_task = {
                task: metric_block(
                    [row for row in rows if row["task"] == task], checkpoint,
                    replay_by_item)
                for task in protocol_code.TASKS
            }
            conditions[condition] = {
                "overall": metric_block(rows, checkpoint, replay_by_item),
                "by_task": by_task,
            }
        checkpoints[str(checkpoint)] = {
            "full_denominator_per_condition": EXPECTED_ITEMS,
            "early_eos_is_absorbing": True,
            "conditions": conditions,
        }

    gate = protocol_code.evaluate_gate(
        split=SPLIT,
        checkpoint=1536,
        complete=True,
        condition_summaries=checkpoints["1536"]["conditions"],
    )
    intervals: dict[str, Any] = {}
    for condition in legal.CONDITIONS:
        counts = Counter()
        for item in items:
            row = resolve_endpoint(records, item["id"], condition, 1536)
            step = row.get("first_eos_step")
            if step is None:
                counts[">1536_censored"] += 1
            elif step <= 256:
                counts["<=256"] += 1
            elif step <= 512:
                counts["257-512"] += 1
            elif step <= 1024:
                counts["513-1024"] += 1
            else:
                counts["1025-1536"] += 1
        intervals[condition] = {key: counts[key] for key in
                                ("<=256", "257-512", "513-1024",
                                 "1025-1536", ">1536_censored")}

    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": manifest["experiment"],
        "split": SPLIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "open_receipt_sha256": receipt["receipt_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(MANIFEST_PATH),
        "rows_sha256": sha256_file(ROWS_PATH),
        "complete": True,
        "expected_trajectories": EXPECTED_ITEMS * len(legal.CONDITIONS),
        "complete_trajectories": EXPECTED_ITEMS * len(legal.CONDITIONS),
        "raw_stage_rows": len(records),
        "all_stage_keys_exact": True,
        "all_rows_strict_v2_recomputed": True,
        "all_terminations_valid": True,
        "all_extension_replays_exact": all(
            all(values.values()) for values in replay_by_condition.values()),
        "checkpoints": checkpoints,
        "eos_time_intervals": intervals,
        "decision_gate": gate,
        "decision_status": "GO" if gate["passed"] else "NO_GO",
        "natural_endpoint_is_actual_sampled_eos": True,
        "greedy_or_logit_gap_used_for_gate": False,
        "wrong_answer_auxiliary_used_for_gate": False,
        "mechanism_result_claimed": False,
        "next_mechanism_phase_eligible": bool(gate["passed"]),
    }
    summary = {
        **core,
        "summary_sha256": legal.sha256_json(core),
        "audited_at_utc": legal.utc_now(),
    }
    if SUMMARY_PATH.exists():
        prior = load_json(SUMMARY_PATH, "existing holdout summary")
        require(prior.get("summary_sha256") == legal.sha256_json(
            without_fields(prior, "summary_sha256", "audited_at_utc")),
            "existing holdout summary self-hash invalid")
        require(prior.get("summary_sha256") == summary["summary_sha256"],
                "existing holdout summary differs; refusing overwrite")
    else:
        legal.atomic_write_json(SUMMARY_PATH, summary)
    return summary


def admission_only() -> dict[str, Any]:
    require("phase977_holdout_dataset" not in sys.modules,
            "holdout unexpectedly imported at process start")
    protocol = authenticate_protocol()
    admission = authenticate_admission(protocol)
    require("phase977_holdout_dataset" not in sys.modules,
            "admission-only path imported holdout")
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "development_gate_passed": True,
        "holdout_authorized": True,
        "open_receipt_created": False,
        "holdout_module_imported": False,
        "model_weights_loaded": False,
    }


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal Phase978 holdout confirmation requires local CUDA")
    t0 = time.time()
    require("phase977_holdout_dataset" not in sys.modules,
            "holdout unexpectedly imported before authorization")
    protocol = authenticate_protocol()
    admission = authenticate_admission(protocol)
    run_lock = acquire_run_lock(protocol, admission)
    model = None
    try:
        receipt = install_or_validate_open_receipt(protocol, admission)
        items, data_audit = load_opened_holdout()
        model, tok, device = load_model(MODEL_NAME)
        require(getattr(device, "type", str(device).split(":")[0]) == "cuda",
                f"Qwen3 did not load on CUDA: {device}")
        eos_ids = [int(value) for value in get_eos_ids(model, tok)]
        think_open_id = legal.single_token_id(tok, "<think>")
        think_close_id = legal.single_token_id(tok, "</think>")
        manifest = make_manifest(
            protocol, admission, receipt, items, data_audit, model, tok,
            eos_ids, think_open_id, think_close_id)
        install_or_validate_manifest(manifest)
        records = load_rows_strict(manifest["manifest_sha256"])
        validate_resume_key_set(records, items)
        item_by_id = {item["id"]: item for item in items}

        # Validate all existing rows before exact same-manifest resume.
        stage_to_budget = {stage: budget for budget, stage in STAGES.items()}
        for key, row in records.items():
            item_id, condition, stage = key
            budget = stage_to_budget[stage]
            parent = expected_parent(records, item_id, condition, budget)
            validate_row(row, manifest, tok, item_by_id[item_id], condition,
                         budget, parent, eos_ids, think_open_id, think_close_id)

        total = len(items) * len(legal.CONDITIONS)
        completed = 0
        for item in items:
            seed = legal.stable_item_seed(BASE_SEED, SPLIT, item["id"])
            for condition in legal.CONDITIONS:
                parent = None
                for budget in CHECKPOINTS:
                    key = (item["id"], condition, STAGES[budget])
                    row = records.get(key)
                    if row is None:
                        input_ids, generated, user_prompt = legal.generate_stage(
                            model, tok, device, eos_ids, item, condition, seed, budget)
                        row = build_row(
                            manifest, tok, item, condition, seed, budget,
                            input_ids, generated, user_prompt, eos_ids,
                            think_open_id, think_close_id, parent)
                        validate_row(row, manifest, tok, item, condition, budget,
                                     parent, eos_ids, think_open_id, think_close_id)
                        legal.append_jsonl(ROWS_PATH, row)
                        records[key] = row
                    if row["has_eos"]:
                        break
                    require(row["hit_budget"] is True,
                            f"non-EOS trajectory did not exhaust checkpoint: {key}")
                    parent = row
                completed += 1
                write_status(manifest, records, items, time.time() - t0)
                if completed % 16 == 0 or completed == total:
                    log(f"  Phase978 holdout trajectories {completed}/{total}")

        write_status(manifest, records, items, time.time() - t0)
        summary = finalize_summary(
            protocol, admission, receipt, manifest, records, items, tok,
            eos_ids, think_open_id, think_close_id)
        log(f"Phase978 holdout complete; decision={summary['decision_status']}; "
            f"elapsed={(time.time()-t0)/60:.1f} min")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        release_run_lock(run_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admission-only", action="store_true",
        help="Authenticate dev PASS without receipt, holdout import, or model load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.admission_only:
        print(json.dumps(admission_only(), ensure_ascii=False, indent=2))
    else:
        run()
