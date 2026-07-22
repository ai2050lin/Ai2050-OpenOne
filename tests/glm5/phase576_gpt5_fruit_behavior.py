#!/usr/bin/env python3
"""Run one Phase576 open split, with models strictly serial and isolated."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import msvcrt
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
GPT5 = ROOT / "tests/gpt5"
GLM5 = ROOT / "tests/glm5"
for path in (GPT5, GLM5):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import phase576_gpt5_fruit_protocol as protocol  # noqa: E402
from phase576_gpt5_fruit_engineering_qualification import (  # noqa: E402
    runtime_identity,
)
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase983_cross_model_engine import (  # noqa: E402
    load_model_adapter,
    release_model_adapter,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale Phase576 behavior temporary artifact: {temporary}")
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale Phase576 behavior temporary artifact: {temporary}")
    raw = "".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
        for row in rows
    ).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    try:
        with temporary.open("xb") as handle:
            handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def stage_dir(stage: str) -> Path:
    return protocol.OUT_DIR / "open_behavior" / stage


def rows_path(model: str, stage: str = "discovery") -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_rows.jsonl.gz"


def summary_path(model: str, stage: str = "discovery") -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_summary.json"


def contract_path(model: str, stage: str = "discovery") -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_contract.json"


def status_path(model: str, stage: str = "discovery") -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_status.json"


def stage_receipt_path(stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{stage}_behavior_execution_receipt.json"


def stage_started_path(stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{stage}_behavior_stage_started.json"


def quarantine_incomplete_stage(stage: str) -> dict[str, Any] | None:
    source = stage_dir(stage)
    if not source.exists():
        return None
    if stage_receipt_path(stage).is_file():
        raise RuntimeError(f"Phase576 {stage} behavior stage is already terminal")
    parent = source.parent.resolve(strict=True)
    if source.is_symlink() or source.resolve(strict=True).parent != parent:
        raise RuntimeError("refusing to quarantine aliased behavior stage")
    inventory = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("refusing to quarantine behavior stage containing symlink")
        if path.is_file():
            inventory.append({
                "path": str(path.relative_to(source)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = parent / f".{source.name}.aborted-{stamp}-pid{os.getpid()}"
    if destination.exists():
        raise RuntimeError("behavior quarantine destination already exists")
    source.rename(destination)
    record = {
        "reason": "nonterminal_prior_stage_atomically_quarantined",
        "path": str(destination.relative_to(protocol.OUT_DIR)).replace("\\", "/"),
        "file_inventory": inventory,
        "file_inventory_sha256": hashlib.sha256(json.dumps(
            inventory, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
    }
    write_json(destination / "phase576_quarantine_receipt.json", record)
    return record


def acquire_stage_lease(stage: str) -> Any:
    path = protocol.OUT_DIR / "open_behavior" / f".phase576_{stage}.lease"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    if path.stat().st_size == 0:
        handle.write(b"0")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Phase576 {stage} behavior stage is actively leased") from exc
    return handle


def release_stage_lease(handle: Any) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


class CudaReleaseError(RuntimeError):
    """The model finished, but the frozen serial CUDA release gate failed."""


def normalized_short_answer(text: str) -> str:
    compact = " ".join(text.strip().split()).casefold()
    return compact.strip(" .,!?:;\"'`()[]{}")


def diagnostic_mentions(row: dict[str, Any], text: str) -> list[str]:
    mentioned = []
    for canonical, aliases in row["candidate_groups"].items():
        if any(re.search(
            rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE
        ) for alias in aliases):
            mentioned.append(canonical)
    return sorted(set(mentioned))


def classify(row: dict[str, Any], generated: str) -> dict[str, Any]:
    normalized = normalized_short_answer(generated)
    exact_owners = {
        canonical
        for canonical, aliases in row["candidate_groups"].items()
        if normalized in {alias.casefold() for alias in aliases}
    }
    selected = next(iter(exact_owners)) if len(exact_owners) == 1 else None
    mentions = diagnostic_mentions(row, generated)
    if len(exact_owners) > 1 or len(mentions) > 1:
        event = "ambiguous_multiple_candidates"
    elif selected == row["target"]:
        event = "target_exact_short_answer"
    elif selected is not None:
        event = "registered_other_exact_short_answer"
    elif mentions:
        event = "candidate_mentioned_but_not_short_answer"
    else:
        event = "unrecoverable"
    correct = selected == row["target"] and len(exact_owners) == 1
    return {
        "generated_text": generated,
        "normalized_generated": normalized,
        "selected_candidate": selected,
        "mentioned_candidates": mentions,
        "semantic_correct": correct,
        "strict_sequence_correct": correct,
        "semantic_event": event,
    }


def verify_engineering_qualification(frozen: dict[str, Any]) -> dict[str, Any]:
    path = protocol.ENGINEERING_QUALIFICATION_PATH
    if not path.exists():
        raise RuntimeError("Phase576 engineering qualification is missing")
    qualification = read_json(path)
    source_key = "tests/glm5/phase576_gpt5_fruit_engineering_qualification.py"
    receipt_relative = qualification.get("execution_receipt_path")
    receipt_path = (
        (ROOT / receipt_relative).resolve()
        if isinstance(receipt_relative, str) else Path("__missing__")
    )
    try:
        receipt_path.relative_to(protocol.OUT_DIR.resolve())
        receipt_in_scope = True
    except (ValueError, OSError):
        receipt_in_scope = False
    receipt = read_json(receipt_path) if receipt_in_scope and receipt_path.is_file() else {}
    expected_qualification_keys = {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "created_at_utc", "terminal_status", "passed",
        "models_in_execution_order", "attempts", "qualified_models",
        "failed_models", "blocked_models", "reports", "execution_contract",
        "execution_contract_sha256", "execution_receipt_path",
        "execution_receipt_sha256", "published_from_pending_execution_receipt",
        "protocol_sha256", "static_audit_sha256", "sealed_commitment_sha256",
        "freeze_commit_sha256", "open_cases_sha256",
        "open_case_sha256_by_split", "qualification_source_sha256",
        "stage_source_seals", "model_artifact_identities",
        "preflight_integrity_checks", "postflight_integrity_checks",
        "runtime_identity", "final_cuda_memory", "formal_case_access",
        "formal_case_content_parsed", "open_case_bytes_hashed_for_integrity",
        "sealed_split_read", "sealed_case_payload_read",
        "prior_sealed_files_read", "activation_persisted",
        "causal_intervention",
    }
    expected_receipt_keys = {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "stage_start_sha256", "created_at_utc", "terminal_status",
        "execution_passed", "required_model_order", "attempts",
        "attempted_models_in_order", "completed_models", "failed_models",
        "not_attempted_models", "fatal_error", "execution_contract",
        "execution_contract_sha256", "preflight_integrity",
        "postflight_integrity", "runtime_identity", "final_cuda_memory",
        "final_cuda_cleanup_pass", "qualification_publish_authorized",
        "qualification_published_at_receipt_commit", "qualification_path",
        "formal_case_access", "formal_case_content_parsed",
        "open_case_bytes_hashed_for_integrity", "sealed_split_read",
        "sealed_case_payload_read", "prior_sealed_files_read",
        "activation_persisted", "causal_intervention",
    }
    attempts = qualification.get("attempts", [])
    reports = qualification.get("reports", [])
    execution_root = (
        protocol.OUT_DIR / "engineering_qualification_execution"
    ).resolve()
    stage_start_path = execution_root / "stage_start.json"
    stage_start_regular = (
        stage_start_path.is_file()
        and not stage_start_path.is_symlink()
        and stage_start_path.resolve(strict=True).parent == execution_root
    )
    stage_start = read_json(stage_start_path) if stage_start_regular else {}
    expected_stage_start_keys = {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "started_at_utc", "required_model_order", "execution_contract",
        "execution_contract_sha256", "qualification_source_sha256",
        "protocol_sha256_observed_at_start",
        "static_audit_sha256_observed_at_start",
        "sealed_commitment_sha256_observed_at_start",
        "freeze_commit_sha256_observed_at_start",
        "formal_case_content_parsed", "sealed_case_payload_read",
        "prior_sealed_files_read", "activation_persisted",
        "causal_intervention",
    }
    expected_running_status_keys = {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "attempt_id", "attempt_index", "model_order_index", "model",
        "status", "started_at_utc", "execution_contract_sha256",
        "stage_start_sha256", "protocol_sha256",
        "qualification_source_sha256", "frozen_model_artifact_identity",
        "cuda_memory_before_load", "formal_case_content_parsed",
        "sealed_case_payload_read", "prior_sealed_files_read",
    }
    expected_terminal_status_keys = {
        "schema_version", "phase_id", "qualification_kind", "run_id",
        "attempt_id", "attempt_index", "model_order_index", "model",
        "status", "started_at_utc", "completed_at_utc",
        "elapsed_seconds", "running_status_sha256",
        "execution_contract_sha256", "cleanup_completed",
        "cuda_memory_after_release", "pytorch_cuda_allocated_after_release",
        "report", "report_sha256", "formal_case_content_parsed",
        "sealed_case_payload_read", "prior_sealed_files_read",
    }
    attempt_artifacts_valid = len(attempts) == len(protocol.MODELS)
    for index, attempt in enumerate(attempts):
        running_relative = attempt.get("running_status_path")
        terminal_relative = attempt.get("terminal_status_path")
        if not isinstance(running_relative, str) or not isinstance(
            terminal_relative, str
        ):
            attempt_artifacts_valid = False
            continue
        running_path = (ROOT / running_relative).resolve()
        terminal_path = (ROOT / terminal_relative).resolve()
        try:
            running_path.relative_to(execution_root)
            terminal_path.relative_to(execution_root)
        except ValueError:
            attempt_artifacts_valid = False
            continue
        if (
            not running_path.is_file()
            or running_path.is_symlink()
            or running_path.resolve(strict=True).parent != execution_root
            or not terminal_path.is_file()
            or terminal_path.is_symlink()
            or terminal_path.resolve(strict=True).parent != execution_root
        ):
            attempt_artifacts_valid = False
            continue
        running = read_json(running_path)
        terminal = read_json(terminal_path)
        report = reports[index] if index < len(reports) else {}
        report_hash = hashlib.sha256(json.dumps(
            report, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        attempt_artifacts_valid = attempt_artifacts_valid and all((
            attempt.get("attempt_index") == index,
            attempt.get("model") == protocol.MODELS[index],
            attempt.get("status") == "complete",
            attempt.get("cleanup_completed") is True,
            attempt.get("pytorch_cuda_allocated_after_release") == 0,
            attempt.get("running_status_sha256") == sha256_file(running_path),
            attempt.get("terminal_status_sha256") == sha256_file(terminal_path),
            attempt.get("report_sha256") == report_hash,
            set(running) == expected_running_status_keys,
            running.get("schema_version")
            == "phase576_repeat_forward_model_status.v1",
            running.get("phase_id") == protocol.PHASE,
            running.get("run_id") == qualification.get("run_id"),
            running.get("attempt_id") == attempt.get("attempt_id"),
            running.get("attempt_index") == index,
            running.get("model_order_index") == index,
            running.get("model") == protocol.MODELS[index],
            running.get("status") == "running",
            running.get("stage_start_sha256") == receipt.get("stage_start_sha256"),
            running.get("execution_contract_sha256")
            == qualification.get("execution_contract_sha256"),
            running.get("protocol_sha256") == qualification.get("protocol_sha256"),
            running.get("qualification_source_sha256")
            == qualification.get("qualification_source_sha256"),
            set(terminal) == expected_terminal_status_keys,
            terminal.get("schema_version")
            == "phase576_repeat_forward_model_status.v1",
            terminal.get("phase_id") == protocol.PHASE,
            terminal.get("run_id") == qualification.get("run_id"),
            terminal.get("attempt_id") == attempt.get("attempt_id"),
            terminal.get("attempt_index") == index,
            terminal.get("model_order_index") == index,
            terminal.get("model") == protocol.MODELS[index],
            terminal.get("status") == "complete",
            terminal.get("execution_contract_sha256")
            == qualification.get("execution_contract_sha256"),
            terminal.get("running_status_sha256")
            == attempt.get("running_status_sha256"),
            terminal.get("cleanup_completed") is True,
            terminal.get("pytorch_cuda_allocated_after_release") == 0,
            terminal.get("report") == report,
            terminal.get("report_sha256") == report_hash,
        ))
    contract_hash = hashlib.sha256(json.dumps(
        qualification.get("execution_contract"), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    checks = {
        "qualification_exact_schema": set(qualification)
        == expected_qualification_keys,
        "receipt_exact_schema": set(receipt) == expected_receipt_keys,
        "stage_start_closure": all((
            stage_start_regular,
            set(stage_start) == expected_stage_start_keys,
            stage_start.get("schema_version")
            == "phase576_repeat_forward_stage_start.v1",
            stage_start.get("phase_id") == protocol.PHASE,
            stage_start.get("qualification_kind")
            == "repeat_forward_engineering_qualification",
            stage_start.get("run_id") == qualification.get("run_id"),
            isinstance(stage_start.get("started_at_utc"), str),
            bool(stage_start.get("started_at_utc")),
            stage_start.get("required_model_order") == list(protocol.MODELS),
            stage_start.get("execution_contract")
            == qualification.get("execution_contract"),
            stage_start.get("execution_contract_sha256")
            == qualification.get("execution_contract_sha256"),
            stage_start.get("qualification_source_sha256")
            == qualification.get("qualification_source_sha256"),
            stage_start.get("protocol_sha256_observed_at_start")
            == sha256_file(protocol.PROTOCOL_PATH),
            stage_start.get("static_audit_sha256_observed_at_start")
            == sha256_file(protocol.STATIC_AUDIT_PATH),
            stage_start.get("sealed_commitment_sha256_observed_at_start")
            == sha256_file(protocol.SEALED_COMMITMENT_PATH),
            stage_start.get("freeze_commit_sha256_observed_at_start")
            == sha256_file(protocol.FREEZE_COMMIT_PATH),
            stage_start.get("formal_case_content_parsed") is False,
            stage_start.get("sealed_case_payload_read") is False,
            stage_start.get("prior_sealed_files_read") is False,
            stage_start.get("activation_persisted") is False,
            stage_start.get("causal_intervention") is False,
            receipt.get("stage_start_sha256") == (
                sha256_file(stage_start_path) if stage_start_regular else None
            ),
        )),
        "schema": qualification.get("schema_version")
        == "phase576_engineering_qualification.v2",
        "phase": qualification.get("phase_id") == protocol.PHASE,
        "kind": qualification.get("qualification_kind")
        == "repeat_forward_engineering_qualification",
        "terminal": qualification.get("terminal_status") == "complete",
        "passed": qualification.get("passed") is True,
        "order": tuple(qualification.get("models_in_execution_order", []))
        == protocol.MODELS,
        "protocol_hash": qualification.get("protocol_sha256")
        == sha256_file(protocol.PROTOCOL_PATH),
        "audit_hash": qualification.get("static_audit_sha256")
        == sha256_file(protocol.STATIC_AUDIT_PATH),
        "commitment_hash": qualification.get("sealed_commitment_sha256")
        == sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "freeze_commit_hash": qualification.get("freeze_commit_sha256")
        == sha256_file(protocol.FREEZE_COMMIT_PATH),
        "open_hash": qualification.get("open_cases_sha256")
        == sha256_file(protocol.OPEN_CASES_PATH),
        "split_hashes": qualification.get("open_case_sha256_by_split")
        == {
            split: sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[split])
            for split in protocol.OPEN_SPLITS
        },
        "source_hash": qualification.get("qualification_source_sha256")
        == frozen["stage_source_seals"][source_key]["sha256"],
        "source_registry": qualification.get("stage_source_seals")
        == frozen["stage_source_seals"],
        "model_registry": qualification.get("model_artifact_identities")
        == frozen["model_artifact_identities"],
        "runtime_identity_matches_current_execution": (
            qualification.get("runtime_identity") == runtime_identity()
        ),
        "formal_case_access": qualification.get("formal_case_access") is False,
        "sealed_access": qualification.get("sealed_split_read") is False,
        "sealed_payload": qualification.get("sealed_case_payload_read") is False,
        "prior_sealed": qualification.get("prior_sealed_files_read") is False,
        "no_activation_persisted": qualification.get("activation_persisted") is False,
        "no_intervention": qualification.get("causal_intervention") is False,
        "report_count": len(reports) == len(protocol.MODELS),
        "qualified_models": qualification.get("qualified_models")
        == list(protocol.MODELS),
        "no_failed_or_blocked": qualification.get("failed_models") == []
        and qualification.get("blocked_models") == [],
        "published_from_pending_receipt": qualification.get(
            "published_from_pending_execution_receipt"
        ) is True,
        "attempt_count": len(attempts) == len(protocol.MODELS),
        "attempt_order": [item.get("model") for item in attempts]
        == list(protocol.MODELS),
        "attempts_complete_clean": all(
            item.get("status") == "complete"
            and item.get("cleanup_completed") is True
            and item.get("pytorch_cuda_allocated_after_release") == 0
            for item in attempts
        ),
        "attempt_artifacts": attempt_artifacts_valid,
        "preflight": isinstance(qualification.get("preflight_integrity_checks"), dict)
        and all(qualification["preflight_integrity_checks"].values()),
        "postflight": isinstance(qualification.get("postflight_integrity_checks"), dict)
        and all(qualification["postflight_integrity_checks"].values()),
        "final_cuda_clean": qualification.get("final_cuda_memory", {}).get(
            "total_allocated_bytes"
        ) == 0,
        "receipt_scope": receipt_in_scope,
        "receipt_hash": receipt_path.is_file()
        and qualification.get("execution_receipt_sha256") == sha256_file(receipt_path),
        "receipt_schema": receipt.get("schema_version")
        == "phase576_repeat_forward_execution_receipt.v1",
        "receipt_identity": receipt.get("phase_id") == protocol.PHASE
        and receipt.get("run_id") == qualification.get("run_id")
        and receipt.get("execution_passed") is True
        and receipt.get("terminal_status")
        == "models_complete_qualification_pending_publish"
        and receipt.get("required_model_order") == list(protocol.MODELS)
        and receipt.get("attempts") == attempts
        and receipt.get("qualification_publish_authorized") is True
        and receipt.get("qualification_published_at_receipt_commit") is False
        and receipt.get("final_cuda_cleanup_pass") is True
        and receipt.get("sealed_split_read") is False
        and receipt.get("sealed_case_payload_read") is False
        and receipt.get("prior_sealed_files_read") is False,
        "receipt_terminal_registries": (
            receipt.get("attempted_models_in_order") == list(protocol.MODELS)
            and receipt.get("completed_models") == list(protocol.MODELS)
            and receipt.get("failed_models") == []
            and receipt.get("not_attempted_models") == []
            and receipt.get("fatal_error") is None
            and receipt.get("runtime_identity")
            == qualification.get("runtime_identity")
        ),
        "receipt_integrity": isinstance(receipt.get("preflight_integrity"), dict)
        and isinstance(receipt.get("postflight_integrity"), dict)
        and all(receipt["preflight_integrity"].get("checks", {}).values())
        and all(receipt["postflight_integrity"].get("checks", {}).values())
        and receipt.get("final_cuda_memory", {}).get("total_allocated_bytes") == 0,
        "contract_identity": qualification.get("execution_contract")
        == receipt.get("execution_contract")
        and qualification.get("execution_contract_sha256") == contract_hash
        and receipt.get("execution_contract_sha256") == contract_hash,
    }
    expected_engineering_checks = {
        "cuda_input", "cuda_only_no_offload", "int8_loaded", "sdpa_loaded",
        "hidden_state_count", "hidden_size", "finite_logits",
        "repeat_logits_exact", "repeat_final_state_exact", "batch_rows_identical",
    }
    for index, (model, report) in enumerate(zip(protocol.MODELS, reports)):
        report_checks = report.get("checks", {})
        detailed = report.get("detailed_checks", {})
        generation = report.get("greedy_generation_capsule", {})
        first_generation = generation.get("first", {})
        second_generation = generation.get("second", {})
        layer_diagnostics = report.get("layer_diagnostics", [])
        direct_logits = report.get("logits_diagnostics", {})
        generation_details_valid = all(
            isinstance(side.get("steps"), list)
            and bool(side["steps"])
            and all(
                step.get("all_layers_valid") is True
                and step.get("logits_finite") is True
                and step.get("logits_batch_rows_identical") is True
                and step.get("logits_contract_valid") is True
                and isinstance(step.get("layers"), list)
                and bool(step["layers"])
                and all(
                    layer.get("finite") is True
                    and layer.get("dtype") == "torch.bfloat16"
                    and isinstance(layer.get("device"), str)
                    and layer["device"].startswith("cuda")
                    and layer.get("batch_rows_identical") is True
                    for layer in step["layers"]
                )
                for step in side["steps"]
            )
            and isinstance(side.get("rows"), list)
            and len(side["rows"]) == 2
            and all(
                row.get("token_ids_in_vocabulary") is True
                and row.get("prompt_prefix_exact") is True
                and row.get("post_eos_tokens_are_absorbing") is True
                and (row.get("eos_seen") is True or row.get("budget_terminated") is True)
                for row in side["rows"]
            )
            for side in (first_generation, second_generation)
        )
        loaded = report.get("loaded_model_identity", {})
        quant = loaded.get("loaded_quantization", {})
        checks[f"{model}_identity"] = (
            report.get("schema_version") == "phase576_repeat_forward_model_report.v1"
            and report.get("phase_id") == protocol.PHASE
            and report.get("qualification_kind")
            == "repeat_forward_engineering_qualification"
            and report.get("model") == model
            and report.get("passed") is True
            and report.get("frozen_model_artifact_identity")
            == frozen["model_artifact_identities"][model]
            and set(report_checks) == expected_engineering_checks
            and all(value is True for value in report_checks.values())
            and isinstance(detailed, dict) and bool(detailed)
            and all(value is True for value in detailed.values())
            and isinstance(layer_diagnostics, list)
            and len(layer_diagnostics) == report.get("hidden_state_count")
            and all(
                row.get("first_finite") is True
                and row.get("second_finite") is True
                and row.get("first_dtype") == "torch.bfloat16"
                and row.get("second_dtype") == "torch.bfloat16"
                and isinstance(row.get("first_device"), str)
                and row["first_device"].startswith("cuda")
                and row.get("first_device") == row.get("second_device")
                and row.get("repeat_exact") is True
                and row.get("first_batch_rows_identical") is True
                and row.get("second_batch_rows_identical") is True
                for row in layer_diagnostics
            )
            and direct_logits.get("first_finite") is True
            and direct_logits.get("second_finite") is True
            and direct_logits.get("repeat_exact") is True
            and direct_logits.get("batch_rows_identical") is True
            and generation.get("max_new_tokens") == protocol.MAX_NEW_TOKENS
            and generation.get("do_sample") is False
            and generation.get("use_cache") is True
            and generation.get("return_dict_in_generate") is True
            and generation.get("output_logits") is True
            and generation.get("output_hidden_states") is True
            and isinstance(first_generation.get("checks"), dict)
            and bool(first_generation["checks"])
            and all(first_generation["checks"].values())
            and isinstance(second_generation.get("checks"), dict)
            and bool(second_generation["checks"])
            and all(second_generation["checks"].values())
            and generation_details_valid
            and isinstance(generation.get("repeat_checks"), dict)
            and bool(generation["repeat_checks"])
            and all(generation.get("repeat_checks", {}).values())
            and loaded.get("cuda_only_no_cpu_or_disk_offload") is True
            and loaded.get("loaded_attn_implementation") == "sdpa"
            and quant.get("load_in_8bit") is True
            and quant.get("floating_parameter_dtypes") == ["torch.bfloat16"]
            and report.get("formal_case_access") is False
            and report.get("sealed_split_read") is False
            and report.get("activation_persisted") is False
            and report.get("causal_intervention") is False
            and report.get("formal_case_content_parsed") is False
            and report.get("sealed_case_payload_read") is False
            and report.get("prior_sealed_files_read") is False
            and report.get("cleanup_completed") is True
            and report.get("pytorch_cuda_allocated_after_release") == 0
            and attempts[index].get("report_sha256")
            == hashlib.sha256(json.dumps(
                report, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
        )
    if not all(checks.values()):
        raise RuntimeError(f"Phase576 engineering qualification invalid: {checks}")
    return qualification


def verify_stage_admission(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in protocol.OPEN_SPLITS:
        raise RuntimeError(f"unsupported Phase576 open stage: {stage}")
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.STATIC_AUDIT_PATH)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    protocol.verify_frozen_source_seals(frozen)
    protocol.verify_frozen_model_artifacts(frozen)
    freeze_commit = protocol._verify_freeze_commit()
    checks = {
        "audit_schema_phase": audit.get("schema_version")
        == "phase576_static_audit.v2" and audit.get("phase_id") == protocol.PHASE,
        "static_valid": audit["valid"] is True and not audit["failures"],
        "audit_cpu_no_models": audit.get("model_weights_loaded") is False
        and audit.get("cuda_used") is False,
        "audit_no_sealed_analysis": audit.get(
            "sealed_model_or_result_read_for_analysis"
        ) is False,
        "audit_grid_valid": audit.get("case_grid_audit", {}).get("valid") is True,
        "audit_cross_model_comparison_rule": isinstance(
            audit.get("cross_model_observational_comparison_rule_audit"), dict
        ) and bool(audit["cross_model_observational_comparison_rule_audit"])
        and all(
            audit["cross_model_observational_comparison_rule_audit"].values()
        ),
        "audit_protocol_hash": audit["protocol_sha256"]
        == sha256_file(protocol.PROTOCOL_PATH),
        "audit_commitment_hash": audit["sealed_commitment_sha256"]
        == sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "frozen_commitment_hash": frozen["sealed_commitment_sha256"]
        == sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "stage_hash": frozen["open_case_sha256_by_split"][stage]
        == sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "audit_stage_hash": audit["open_case_sha256_by_split"][stage]
        == sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "sealed_model_unopened": commitment["sealed_model_opened"] is False,
        "sealed_model_access_zero": commitment["sealed_model_access_count"] == 0,
        "sealed_result_analysis_zero": commitment[
            "sealed_result_analysis_access_count"
        ] == 0,
        "prior_sealed_unread": commitment["prior_sealed_files_read"] is False,
        "required_order": tuple(frozen["models_in_required_execution_order"])
        == protocol.MODELS,
        "freeze_commit_complete": freeze_commit.get("complete") is True,
        "freeze_lock_absent": not protocol.FREEZE_LOCK_PATH.exists(),
        "prior_open_identities": audit.get("prior_open_file_identities")
        == frozen.get("prior_open_file_identities"),
        "current_behavior_decision_absent": not protocol.BEHAVIOR_DECISION_PATHS[
            stage
        ].exists(),
        "current_trace_absent": not (
            protocol.OUT_DIR / "natural_trace" / stage
        ).exists(),
    }
    if stage == "discovery":
        checks["no_future_stage_artifacts"] = not any(path.exists() for path in (
            protocol.DISCOVERY_REGISTRY_PATH,
            protocol.CONFIRMATION_DECISION_PATH,
            protocol.HELDOUT_DECISION_PATH,
            protocol.BEHAVIOR_DECISION_PATHS["confirmation"],
            protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"],
            protocol.OUT_DIR / "open_behavior" / "confirmation",
            protocol.OUT_DIR / "open_behavior" / "heldout_recombination",
            protocol.OUT_DIR / "natural_trace" / "confirmation",
            protocol.OUT_DIR / "natural_trace" / "heldout_recombination",
        ))
    elif stage == "confirmation":
        registry = protocol.verify_discovery_registry(frozen)
        checks["discovery_registry_frozen"] = (
            registry.get("discovery_registry_frozen") is True
            and registry.get("discovery_candidate_pass") is True
            and registry.get("causal_claim_authorized") is False
        )
        checks["no_future_stage_artifacts"] = not any(path.exists() for path in (
            protocol.CONFIRMATION_DECISION_PATH,
            protocol.HELDOUT_DECISION_PATH,
            protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"],
            protocol.OUT_DIR / "open_behavior" / "heldout_recombination",
            protocol.OUT_DIR / "natural_trace" / "heldout_recombination",
        ))
    elif stage == "heldout_recombination":
        decision = protocol.verify_confirmation_decision(frozen)
        checks["confirmation_passed"] = (
            decision.get("structure_confirmation_pass") is True
            and decision.get("causal_claim_authorized") is False
        )
        checks["heldout_decision_absent"] = not protocol.HELDOUT_DECISION_PATH.exists()
    if not all(checks.values()):
        raise RuntimeError(f"Phase576 {stage} admission failed: {checks}")
    qualification = verify_engineering_qualification(frozen)
    return frozen, qualification


def generate_batch(
    adapter: Any,
    model: str,
    stage: str,
    rows: list[dict[str, Any]],
    repeat: str,
) -> list[dict[str, Any]]:
    prompts = [render_chat(adapter.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = adapter.tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False,
        add_special_tokens=True,
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(adapter.input_device) for key, value in encoded.items()}
    eos_ids = [int(value) for value in adapter.eos_identity["effective_eos_token_ids"]]
    with torch.inference_mode():
        generated = adapter.model.generate(
            **encoded,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=adapter.pad_token_id,
            eos_token_id=eos_ids,
        )
    output = []
    for index, row in enumerate(rows):
        suffix = [int(value) for value in generated[index, prompt_width:].tolist()]
        first_eos = next((i for i, value in enumerate(suffix) if value in eos_ids), None)
        content_ids = suffix if first_eos is None else suffix[:first_eos]
        text = adapter.tokenizer.decode(content_ids, skip_special_tokens=True)
        eos_seen = first_eos is not None
        output.append({
            "schema_version": "phase576_open_behavior_row.v2",
            "phase_id": protocol.PHASE,
            "model": model,
            "stage": stage,
            "execution_repeat": repeat,
            "case_id": row["case_id"],
            "split": row["split"],
            "relation": row["relation"],
            "interface": row["interface"],
            "surface_id": row["surface_id"],
            "order": row["order"],
            "focus_object_id": row["focus_object_id"],
            "focus_is_fruit": row["focus_is_fruit"],
            "contrast_group_id": row["contrast_group_id"],
            "contrast_label": row["contrast_label"],
            "independent_unit_id": row["independent_unit_id"],
            "target": row["target"],
            **classify(row, text),
            "generated_token_count_before_eos": len(content_ids),
            "generated_token_ids_before_eos": content_ids,
            "full_generated_suffix_token_ids": suffix,
            "generation_suffix_width": len(suffix),
            "first_eos_index": first_eos,
            "post_eos_token_ids": (
                suffix[first_eos + 1:] if first_eos is not None else []
            ),
            "post_eos_tokens_all_pad": (
                all(value == adapter.pad_token_id for value in suffix[first_eos + 1:])
                if first_eos is not None else True
            ),
            "eos_seen": eos_seen,
            "first_eos_token_id": suffix[first_eos] if first_eos is not None else None,
            "budget_truncated": not eos_seen and len(suffix) >= protocol.MAX_NEW_TOKENS,
            "termination_event": "eos" if eos_seen else "budget" if len(suffix) >= protocol.MAX_NEW_TOKENS else "other",
            "observer_only": True,
            "activation_collected": False,
            "causal": False,
            "sealed_model_access": False,
        })
    del encoded, generated
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["relation"], row["interface"])].append(row)
    for key, bank in sorted(grouped.items()):
        strata["|".join(key)] = {
            "n": len(bank),
            "strict_correct": sum(item["strict_sequence_correct"] for item in bank),
            "strict_rate": sum(item["strict_sequence_correct"] for item in bank) / len(bank),
            "budget_truncated": sum(item["budget_truncated"] for item in bank),
        }
    return {
        "row_count": len(rows),
        "unique_case_count": len({row["case_id"] for row in rows}),
        "strict_correct": sum(row["strict_sequence_correct"] for row in rows),
        "strict_rate": sum(row["strict_sequence_correct"] for row in rows) / len(rows),
        "event_counts": dict(sorted(Counter(row["semantic_event"] for row in rows).items())),
        "termination_counts": dict(sorted(Counter(row["termination_event"] for row in rows).items())),
        "strata": strata,
    }


def run_model(
    model: str,
    stage: str,
    frozen: dict[str, Any],
    stage_cases: list[dict[str, Any]],
    qualification_sha256: str,
    engineering_receipt_sha256: str,
) -> dict[str, Any]:
    for path in (
        rows_path(model, stage), summary_path(model, stage),
        contract_path(model, stage), status_path(model, stage),
    ):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing Phase576 artifact: {path}")
    started_at = now()
    attempt_id = hashlib.sha256(
        f"{protocol.PHASE}|{stage}|{model}|{started_at}".encode("utf-8")
    ).hexdigest()[:24]
    stage_hash = sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    protocol_hash = sha256_file(protocol.PROTOCOL_PATH)
    source_hash = sha256_file(Path(__file__).resolve())
    contract = {
        "schema_version": "phase576_open_behavior_contract.v2",
        "phase_id": protocol.PHASE,
        "created_at_utc": started_at,
        "attempt_id": attempt_id,
        "model": model,
        "stage": stage,
        "model_order_index": protocol.MODELS.index(model),
        "stage_cases_sha256": stage_hash,
        "protocol_sha256": protocol_hash,
        "behavior_source_sha256": source_hash,
        "frozen_stage_source_seals": frozen["stage_source_seals"],
        "model_artifact_identity": frozen["model_artifact_identities"][model],
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": runtime_identity(),
        "batch_size": protocol.BEHAVIOR_BATCH_SIZE,
        "max_new_tokens": protocol.MAX_NEW_TOKENS,
        "repeats": list(protocol.BEHAVIOR_REPEATS),
        "do_sample": False,
        "render_policy": {
            "qwen3_enable_thinking": False,
            "deepseek_empty_think_prefill_closed": True,
            "classification": "exact registered short answer after terminal punctuation trim",
        },
        "sealed_model_access": False,
        "activation_collection": False,
        "causal_intervention": False,
    }
    write_json(contract_path(model, stage), contract)
    contract_hash = sha256_file(contract_path(model, stage))
    write_json(status_path(model, stage), {
        "schema_version": "phase576_behavior_model_status.v1",
        "phase_id": protocol.PHASE, "model": model, "stage": stage,
        "attempt_id": attempt_id,
        "model_order_index": protocol.MODELS.index(model),
        "status": "running", "started_at_utc": started_at,
        "protocol_sha256": protocol_hash,
        "stage_cases_sha256": stage_hash,
        "behavior_source_sha256": source_hash,
        "behavior_contract_sha256": contract_hash,
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "cleanup_completed": False,
        "sealed_model_access": False,
    })
    adapter = None
    output: list[dict[str, Any]] = []
    started = time.time()
    summary: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        protocol.verify_frozen_model_artifacts(frozen, (model,))
        adapter = load_model_adapter(model)
        adapter.tokenizer.padding_side = "left"
        loaded_identity = adapter.identity
        for repeat in protocol.BEHAVIOR_REPEATS:
            for start in range(0, len(stage_cases), protocol.BEHAVIOR_BATCH_SIZE):
                batch = stage_cases[start:start + protocol.BEHAVIOR_BATCH_SIZE]
                output.extend(generate_batch(adapter, model, stage, batch, repeat))
                done = min(start + protocol.BEHAVIOR_BATCH_SIZE, len(stage_cases))
                if start == 0 or done == len(stage_cases) or start // protocol.BEHAVIOR_BATCH_SIZE % 8 == 7:
                    print(f"[{time.strftime('%H:%M:%S')}] {stage}/{model}/{repeat} {done}/{len(stage_cases)}", flush=True)
        summary = {
            "schema_version": "phase576_open_behavior_summary.v2",
            "phase_id": protocol.PHASE,
            "created_at_utc": now(),
            "attempt_id": attempt_id,
            "model": model,
            "stage": stage,
            "elapsed_seconds": time.time() - started,
            "loaded_model_identity": loaded_identity,
            "frozen_model_artifact_identity": frozen["model_artifact_identities"][model],
            **summarize(output),
            "stage_cases_sha256": stage_hash,
            "protocol_sha256": protocol_hash,
            "behavior_contract_sha256": contract_hash,
            "engineering_qualification_sha256": qualification_sha256,
            "engineering_execution_receipt_sha256": engineering_receipt_sha256,
            "runtime_identity": runtime_identity(),
            "sealed_model_access": False,
        }
        write_jsonl_gz(rows_path(model, stage), output)
        summary["rows_sha256"] = sha256_file(rows_path(model, stage))
        write_json(summary_path(model, stage), summary)
    except BaseException as exc:
        failure = exc
        exc.__traceback__ = None
    finally:
        try:
            release_model_adapter(adapter)
        except BaseException as release_exc:
            if failure is None:
                failure = release_exc
                release_exc.__traceback__ = None
        adapter = None
        gc.collect()
        allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        reserved = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        if allocated != 0 and failure is None:
            failure = CudaReleaseError(
                f"nonzero PyTorch CUDA allocation after release: {allocated}"
            )
        if allocated == 0:
            try:
                protocol.verify_frozen_model_artifacts(frozen, (model,))
            except BaseException as identity_exc:
                if failure is None:
                    failure = identity_exc
                    identity_exc.__traceback__ = None

    terminal_common = {
        "schema_version": "phase576_behavior_model_status.v1",
        "phase_id": protocol.PHASE,
        "model": model,
        "stage": stage,
        "attempt_id": attempt_id,
        "model_order_index": protocol.MODELS.index(model),
        "started_at_utc": started_at,
        "elapsed_seconds": time.time() - started,
        "protocol_sha256": protocol_hash,
        "stage_cases_sha256": stage_hash,
        "behavior_source_sha256": source_hash,
        "behavior_contract_sha256": contract_hash,
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": runtime_identity(),
        "cleanup_completed": allocated == 0,
        "pytorch_cuda_allocated_after_release": allocated,
        "pytorch_cuda_reserved_after_release": reserved,
        "sealed_model_access": False,
    }
    if failure is not None:
        write_json(status_path(model, stage), {
            **terminal_common,
            "status": "failed",
            "failure_stage": "model_execution_or_cleanup",
            "failed_at_utc": now(),
            "error_type": type(failure).__name__,
            "error": str(failure),
        })
        raise failure
    if summary is None:
        failure = RuntimeError(f"{stage}/{model}: behavior ended without summary")
        write_json(status_path(model, stage), {
            **terminal_common,
            "status": "failed",
            "failure_stage": "missing_summary",
            "failed_at_utc": now(),
            "error_type": type(failure).__name__,
            "error": str(failure),
        })
        raise failure
    write_json(status_path(model, stage), {
        **terminal_common,
        "status": "complete",
        "completed_at_utc": now(),
        "behavior_rows_sha256": sha256_file(rows_path(model, stage)),
        "behavior_summary_sha256": sha256_file(summary_path(model, stage)),
    })
    return summary


def _run_stage_with_lease(stage: str) -> dict[str, Any]:
    frozen, qualification = verify_stage_admission(stage)
    qualification_sha256 = sha256_file(protocol.ENGINEERING_QUALIFICATION_PATH)
    engineering_receipt_sha256 = qualification["execution_receipt_sha256"]
    cases = read_jsonl(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    if len(cases) != 336 or any(row["split"] != stage or row["sealed"] for row in cases):
        raise RuntimeError(f"Phase576 {stage} case denominator invalid")
    if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
        raise RuntimeError("nonzero PyTorch CUDA allocation at behavior-stage baseline")
    quarantined_attempt = quarantine_incomplete_stage(stage)
    stage_dir(stage).mkdir(parents=True, exist_ok=False)
    write_json(stage_started_path(stage), {
        "schema_version": "phase576_behavior_stage_started.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "stage": stage,
        "models_planned_in_required_order": list(protocol.MODELS),
        "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "behavior_source_sha256": sha256_file(Path(__file__).resolve()),
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": runtime_identity(),
        "quarantined_incomplete_attempt": quarantined_attempt,
        "sealed_model_access": False,
    })
    reports, failures, attempts = [], [], []
    fatal_error: dict[str, Any] | None = None
    for model in protocol.MODELS:
        if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
            fatal_error = {
                "stage": "before_next_model",
                "model": model,
                "error_type": "CudaReleaseError",
                "error": "PyTorch CUDA allocation remains before next model",
            }
            break
        try:
            report = run_model(
                model, stage, frozen, cases,
                qualification_sha256, engineering_receipt_sha256,
            )
            reports.append(report)
            status = read_json(status_path(model, stage))
            attempts.append({
                "model": model,
                "status": "complete",
                "attempt_id": status["attempt_id"],
                "terminal_status_sha256": sha256_file(status_path(model, stage)),
            })
        except Exception as exc:
            status_file = status_path(model, stage)
            if not status_file.is_file():
                synthetic_started = now()
                synthetic_contract_file = contract_path(model, stage)
                synthetic_contract_exists = synthetic_contract_file.is_file()
                synthetic_attempt_id = hashlib.sha256(
                    f"{protocol.PHASE}|{stage}|{model}|{synthetic_started}|precontract".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
                allocated = (
                    int(torch.cuda.memory_allocated())
                    if torch.cuda.is_available() else 0
                )
                reserved = (
                    int(torch.cuda.memory_reserved())
                    if torch.cuda.is_available() else 0
                )
                write_json(status_file, {
                    "schema_version": "phase576_behavior_model_status.v1",
                    "phase_id": protocol.PHASE,
                    "model": model,
                    "stage": stage,
                    "attempt_id": synthetic_attempt_id,
                    "model_order_index": protocol.MODELS.index(model),
                    "started_at_utc": synthetic_started,
                    "elapsed_seconds": 0.0,
                    "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
                    "stage_cases_sha256": sha256_file(
                        protocol.OPEN_SPLIT_CASE_PATHS[stage]
                    ),
                    "behavior_source_sha256": sha256_file(Path(__file__).resolve()),
                    "behavior_contract_sha256": (
                        sha256_file(synthetic_contract_file)
                        if synthetic_contract_exists else None
                    ),
                    "engineering_qualification_sha256": qualification_sha256,
                    "engineering_execution_receipt_sha256": engineering_receipt_sha256,
                    "runtime_identity": runtime_identity(),
                    "cleanup_completed": allocated == 0,
                    "pytorch_cuda_allocated_after_release": allocated,
                    "pytorch_cuda_reserved_after_release": reserved,
                    "sealed_model_access": False,
                    "status": "failed",
                    "failure_stage": (
                        "preexecution_status_publish"
                        if synthetic_contract_exists else "precontract"
                    ),
                    "failed_at_utc": now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            else:
                status = read_json(status_file)
                state = status.get("status")
                if state == "complete":
                    recovered_summary_file = summary_path(model, stage)
                    recovered_rows_file = rows_path(model, stage)
                    recovered_contract_file = contract_path(model, stage)
                    if not all((
                        recovered_summary_file.is_file(),
                        recovered_rows_file.is_file(),
                        recovered_contract_file.is_file(),
                        status.get("cleanup_completed") is True,
                        status.get("pytorch_cuda_allocated_after_release") == 0,
                        status.get("behavior_summary_sha256")
                        == sha256_file(recovered_summary_file),
                        status.get("behavior_rows_sha256")
                        == sha256_file(recovered_rows_file),
                        status.get("behavior_contract_sha256")
                        == sha256_file(recovered_contract_file),
                        not torch.cuda.is_available()
                        or torch.cuda.memory_allocated() == 0,
                    )):
                        raise RuntimeError(
                            f"{stage}/{model}: complete behavior status cannot be recovered"
                        ) from exc
                    recovered_report = read_json(recovered_summary_file)
                    if not any(item.get("model") == model for item in reports):
                        reports.append(recovered_report)
                    attempts.append({
                        "model": model,
                        "status": "complete",
                        "attempt_id": status["attempt_id"],
                        "terminal_status_sha256": sha256_file(status_file),
                    })
                    print(
                        f"{stage}/{model}: recovered complete behavior publication",
                        flush=True,
                    )
                    continue
                if state != "failed":
                    allocated = (
                        int(torch.cuda.memory_allocated())
                        if torch.cuda.is_available() else 0
                    )
                    reserved = (
                        int(torch.cuda.memory_reserved())
                        if torch.cuda.is_available() else 0
                    )
                    contract_file = contract_path(model, stage)
                    write_json(status_file, {
                        "schema_version": "phase576_behavior_model_status.v1",
                        "phase_id": protocol.PHASE,
                        "model": model,
                        "stage": stage,
                        "attempt_id": status.get("attempt_id"),
                        "model_order_index": protocol.MODELS.index(model),
                        "started_at_utc": status.get("started_at_utc"),
                        "elapsed_seconds": status.get("elapsed_seconds", 0.0),
                        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
                        "stage_cases_sha256": sha256_file(
                            protocol.OPEN_SPLIT_CASE_PATHS[stage]
                        ),
                        "behavior_source_sha256": sha256_file(Path(__file__).resolve()),
                        "behavior_contract_sha256": (
                            sha256_file(contract_file) if contract_file.is_file() else None
                        ),
                        "engineering_qualification_sha256": qualification_sha256,
                        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
                        "runtime_identity": runtime_identity(),
                        "cleanup_completed": allocated == 0,
                        "pytorch_cuda_allocated_after_release": allocated,
                        "pytorch_cuda_reserved_after_release": reserved,
                        "sealed_model_access": False,
                        "status": "failed",
                        "failure_stage": "terminal_status_publish",
                        "failed_at_utc": now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
            status = read_json(status_file)
            error_type = status.get("error_type", type(exc).__name__)
            error = status.get("error", str(exc))
            failures.append({"model": model, "error_type": error_type, "error": error})
            attempts.append({
                "model": model, "status": "failed",
                "attempt_id": status.get("attempt_id"),
                "error_type": error_type,
                "terminal_status_sha256": sha256_file(status_path(model, stage)),
            })
            if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
                fatal_error = {
                    "stage": "failed_model_cleanup",
                    "model": model,
                    "error_type": "CudaReleaseError",
                    "error": f"{model} failed and CUDA allocations remain",
                }
                break
    final_allocated = (
        int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    )
    final_reserved = (
        int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
    )
    if final_allocated != 0 and fatal_error is None:
        fatal_error = {
            "stage": "final_cuda_cleanup",
            "model": None,
            "error_type": "CudaReleaseError",
            "error": "PyTorch CUDA allocation remains after behavior stage",
        }
    receipt = {
        "schema_version": "phase576_behavior_execution_receipt.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "stage": stage,
        "models_attempted_in_order": [item["model"] for item in attempts],
        "attempts": attempts,
        "completed_models": [report["model"] for report in reports],
        "failed_models": failures,
        "not_attempted_models": [
            model for model in protocol.MODELS
            if model not in {item["model"] for item in attempts}
        ],
        "fatal_error": fatal_error,
        "terminal_status": "failed" if fatal_error is not None else "complete",
        "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "behavior_source_sha256": sha256_file(Path(__file__).resolve()),
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": runtime_identity(),
        "stage_started_sha256": sha256_file(stage_started_path(stage)),
        "final_pytorch_cuda_allocated": final_allocated,
        "final_pytorch_cuda_reserved": final_reserved,
        "sealed_model_access": False,
    }
    write_json(stage_receipt_path(stage), receipt)
    if fatal_error is not None:
        raise RuntimeError(f"Phase576 {stage} behavior fatal: {fatal_error}")
    return receipt


def run_stage(stage: str) -> dict[str, Any]:
    lease = acquire_stage_lease(stage)
    try:
        return _run_stage_with_lease(stage)
    finally:
        release_stage_lease(lease)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=protocol.OPEN_SPLITS)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.stage), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
