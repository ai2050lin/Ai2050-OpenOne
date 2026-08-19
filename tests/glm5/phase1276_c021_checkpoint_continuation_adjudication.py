#!/usr/bin/env python3
"""Phase1276: read-only continuation and final C021 WP01 adjudication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1274_c021_multitask_free_response_isomorphism as base
import phase1275_c021_multitask_response_isomorphism_execution as execution


PHASE = 1276
CAMPAIGN = "C021"
CONTRACT_ID = "EXP-C021-WP01-003"
OUT = ROOT / "tests/glm5/result/phase1276_c021_checkpoint_continuation_adjudication"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_multitask_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
QUALIFICATION = OUT / "raw/behavior_qualification.jsonl"
MODELS = OUT / "raw/model_response_tensors.jsonl"
PAIRS = OUT / "raw/camera_pair_ledger.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1276_c021_checkpoint_continuation_adjudication_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C021-WP01-003.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_order() -> list[str]:
    return [f"{task}.{architecture}.s{seed_index}" for task in base.TASKS for architecture in base.ARCHITECTURES for seed_index in range(base.SEEDS_PER_CELL)]


def partial_ledgers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return base.read_jsonl(execution.QUALIFICATION), base.read_jsonl(execution.MODELS)


def validate_partial() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    qualification, models = partial_ledgers()
    order = expected_order()
    keys = [row["model_key"] for row in qualification]
    if keys != order[: len(keys)] or len(keys) != 22: raise RuntimeError("partial qualification is not the frozen 22-key prefix")
    legal_measured = {row["model_key"] for row in qualification if row["behavior_passed"]}
    if {row["model_key"] for row in models} != legal_measured or len(models) != 17: raise RuntimeError("partial measured subset mismatch")
    return qualification, models, order[len(keys) :]


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualification, models, remaining = validate_partial()
    predecessor = base.read_json(execution.PROTOCOL)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1276.c021.checkpoint_continuation.v1",
        "claim_type": predecessor["claim_type"],
        "semantic_digest": execution.semantic_digest(predecessor),
        "material_hash": file_sha256(execution.MATERIAL),
        "partial": {"qualification_count": len(qualification), "measured_count": len(models), "qualification_hash": file_sha256(execution.QUALIFICATION), "models_hash": file_sha256(execution.MODELS), "qualification_digest": base.digest(qualification), "models_digest": base.digest(models)},
        "full_order": expected_order(),
        "remaining_keys": remaining,
        "remaining_count": len(remaining),
        "batch_by_architecture": execution.BATCH_BY_ARCHITECTURE,
        "thresholds": predecessor["thresholds"],
        "hard_stops": ["Partial hashes and prefix order are immutable.", "Only remaining_keys may train.", "No prior seed may be retrained or replaced.", "No confirmation result selects a camera or threshold.", "Failure closes C021 synthetic local-isomorphism search."],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR), "contract": file_sha256(CONTRACT), "execution": file_sha256(execution.SCRIPT), "phase1275_protocol": file_sha256(execution.PROTOCOL), "phase1275_qualification": file_sha256(execution.QUALIFICATION), "phase1275_models": file_sha256(execution.MODELS)},
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {"created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "precision": "fp32 training/intervention; fp64 analysis"}


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force: raise RuntimeError("protocol already exists")
    rows = base.read_jsonl(execution.MATERIAL)
    base.write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "registered", "remaining": read_json(PROTOCOL)["remaining_keys"]}))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, rows = read_json(PROTOCOL), base.read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]: raise RuntimeError("frozen protocol/source mismatch")
    if file_sha256(MATERIAL) != file_sha256(execution.MATERIAL): raise RuntimeError("material drift")
    return protocol, rows


def run(device_name: str) -> None:
    protocol, rows = verify_protocol()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda": raise RuntimeError("formal run requires CUDA")
    if any(path.exists() for path in (QUALIFICATION, MODELS, PAIRS, SUMMARY, COMPLETE)): raise RuntimeError("formal outputs already exist")
    qualification, models, remaining = validate_partial()
    base.write_jsonl(QUALIFICATION, qualification)
    base.write_jsonl(MODELS, models)
    started = time.perf_counter()
    for key in remaining:
        task, architecture, seed_part = key.split(".")
        seed_index = int(seed_part[1:])
        config, seed = base.ARCHITECTURES[architecture], base.MODEL_SEEDS[key]
        model, training = base.train_model(task, config, seed, device)
        behavior_passed = bool(training["qualification_accuracy"] >= base.THRESHOLDS["behavior_accuracy_min"])
        qualification.append({"model_key": key, "task": task, "architecture": architecture, "seed_index": seed_index, "seed": seed, "training": training, "behavior_passed": behavior_passed, "qualification_status": "qualified" if behavior_passed else "behavior_rejected"})
        if behavior_passed:
            response_tensor, controls_passed = execution.measure_model_fast(model, task, architecture, rows, device)
            models.append({"model_key": key, "task": task, "architecture": architecture, "depth": config.layers, "seed_index": seed_index, "seed": seed, "controls_passed": controls_passed, "measurement_status": "measured", "claim_status": "abstained", "response_tensor": response_tensor})
        base.write_jsonl(QUALIFICATION, qualification)
        base.write_jsonl(MODELS, models)
        print(canonical_json({"model": key, "accuracy": training["qualification_accuracy"], "behavior": behavior_passed, "controls": models[-1]["controls_passed"] if behavior_passed else None, "steps": training["steps"], "remaining_after": len(expected_order()) - len(qualification)}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if [row["model_key"] for row in qualification] != expected_order(): raise RuntimeError("final order drift")
    ledger = base.build_pair_ledger(models)
    base.write_jsonl(PAIRS, ledger)
    elapsed = time.perf_counter() - started
    summary = {"phase": PHASE, "created_at_utc": utc_now(), "attempted_models": len(qualification), "measured_models": len(models), "continued_models": len(remaining), "pair_count": len(ledger), "elapsed_seconds_continuation": elapsed, "gpu_hours_continuation": elapsed / 3600.0, "device": torch.cuda.get_device_name(0), "protocol_digest": protocol["protocol_digest"], "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS), "run_digest": digest([row["model_key"] for row in qualification]), "pretrained_model_loaded": False}
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"phase": PHASE, "complete": True, "created_at_utc": utc_now(), "run_digest": summary["run_digest"]})


def analyze() -> None:
    verify_protocol()
    qualification, models, ledger = base.read_jsonl(QUALIFICATION), base.read_jsonl(MODELS), base.read_jsonl(PAIRS)
    final, rebuilt = base.analyze_results(qualification, models, ledger)
    if base.digest(ledger) != base.digest(rebuilt): raise RuntimeError("pair ledger drift")
    final.update({"phase": PHASE, "contract_id": CONTRACT_ID, "created_at_utc": utc_now(), "protocol_hash": file_sha256(PROTOCOL), "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS), "phase1274_semantic_digest": execution.semantic_digest(base.read_json(base.PROTOCOL)), "continued_from_phase1275": True})
    final["final_digest"] = digest({key: value for key, value in final.items() if key not in {"created_at_utc", "final_digest"}})
    atomic_json(FINAL, final)
    print(canonical_json({"decision": final["decision"], "passed": final["passed"], "selected": final["selected_camera"], "executable": final["selected_executable_camera"], "behavior": final["behavior_qualified_models"], "controls": final["control_passed_models"], "gates": final["gates"]}))


def run_auditor(mode: str) -> None:
    status = os.spawnv(os.P_WAIT, sys.executable, [sys.executable, str(AUDITOR), mode])
    if status: raise SystemExit(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "preaudit", "run", "analyze", "audit"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mode == "preregister": preregister(args.force)
    elif args.mode == "preaudit": run_auditor("preaudit")
    elif args.mode == "run": run(args.device)
    elif args.mode == "analyze": analyze()
    else: run_auditor("final")


if __name__ == "__main__": main()
