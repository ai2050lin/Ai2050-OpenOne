#!/usr/bin/env python3
"""Phase1275: throughput-safe execution of the frozen Phase1274 science contract."""

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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1274_c021_multitask_free_response_isomorphism as base
import phase1271_c019_cross_layer_micro_write_trajectory as micro


PHASE = 1275
CAMPAIGN = "C021"
CONTRACT_ID = "EXP-C021-WP01-002"
OUT = ROOT / "tests/glm5/result/phase1275_c021_multitask_response_isomorphism_execution"
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
AUDITOR = ROOT / "tests/glm5/phase1275_c021_multitask_response_isomorphism_execution_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C021-WP01-002.json"
BATCH_BY_ARCHITECTURE = {"shallow4": 256, "middle6": 128, "deep8": 64}
SEMANTIC_FIELDS = ("tasks", "discovery_tasks", "heldout_task", "architectures", "seeds_per_cell", "model_seeds", "roles", "readouts", "program_registry", "partitions", "behavior_examples", "cameras", "executable_cameras", "selection_panel", "sealed_panels", "thresholds", "material_seed", "row_count", "material_digest")


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


def semantic_digest(protocol: dict[str, Any]) -> str:
    return digest({field: protocol[field] for field in SEMANTIC_FIELDS})


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = base.read_json(base.PROTOCOL)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1275.c021.multitask_response_isomorphism_execution.v1",
        "claim_type": predecessor["claim_type"],
        "phase1274_dependency": {"formal_complete": base.COMPLETE.exists(), "scientific_outputs_exist": any(path.exists() for path in (base.QUALIFICATION, base.MODELS, base.PAIRS, base.FINAL)), "protocol_hash": file_sha256(base.PROTOCOL), "material_hash": file_sha256(base.MATERIAL), "semantic_digest": semantic_digest(predecessor)},
        **{field: predecessor[field] for field in SEMANTIC_FIELDS},
        "execution_revision": {"batch_by_architecture": BATCH_BY_ARCHITECTURE, "atomic_per_model_checkpoints": True, "scientific_semantics_changed": False},
        "hard_stops": predecessor["hard_stops"] + ["Material and semantic digests must exactly match Phase1274.", "OOM cannot reduce the registered scientific object."],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR), "contract": file_sha256(CONTRACT), "phase1274_executor": file_sha256(base.SCRIPT), "phase1274_protocol": file_sha256(base.PROTOCOL), "phase1274_material": file_sha256(base.MATERIAL)},
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {"created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "precision": "fp32 training/intervention; fp64 analysis"}


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force: raise RuntimeError("protocol already exists")
    predecessor = base.read_json(base.PROTOCOL)
    rows = base.read_jsonl(base.MATERIAL)
    if predecessor["material_digest"] != base.digest([{"row_id": row["row_id"], "row_digest": row["row_digest"]} for row in rows]): raise RuntimeError("Phase1274 material mismatch")
    base.write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "registered", "rows": len(rows), "semantic_digest": semantic_digest(predecessor), "batches": BATCH_BY_ARCHITECTURE}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, rows = base.read_json(PROTOCOL), base.read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]: raise RuntimeError("frozen protocol/source mismatch")
    predecessor = base.read_json(base.PROTOCOL)
    if semantic_digest(protocol) != semantic_digest(predecessor): raise RuntimeError("scientific semantics drift")
    if file_sha256(MATERIAL) != file_sha256(base.MATERIAL): raise RuntimeError("material byte drift")
    return protocol, rows


def measure_model_fast(model, task: str, architecture: str, rows: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], bool]:
    events = base.program_registry(base.ARCHITECTURES[architecture].layers)
    stage_events = {stage: [event for event in events if event["stage"] == stage] for stage in ("attn_write", "mlp_write")}
    event_index = {event["event_id"]: index for index, event in enumerate(events)}
    sums = {partition: np.zeros((len(events), len(base.ROLES), len(base.READOUTS)), dtype=np.float64) for partition in base.PARTITION_COUNTS}
    counts = {partition: 0 for partition in base.PARTITION_COUNTS}
    batch_size = BATCH_BY_ARCHITECTURE[architecture]
    with torch.inference_mode():
        for partition in base.PARTITION_COUNTS:
            selected_rows = [row for row in rows if row["task"] == task and row["partition"] == partition]
            for start in range(0, len(selected_rows), batch_size):
                batch_rows = selected_rows[start : start + batch_size]
                batch = len(batch_rows)
                receiver_ids = torch.tensor([row["receiver_ids"] for row in batch_rows], device=device)
                donor_ids = torch.cat([torch.tensor([row["variants"][role]["ids"] for row in batch_rows], device=device) for role in base.ROLES], dim=0)
                receiver_repeat = receiver_ids.repeat(len(base.ROLES), 1)
                receiver_trace = micro.capture_micro(model, receiver_ids)
                donor_trace = micro.capture_micro(model, donor_ids)
                receiver_trace_repeat = base.repeat_trace(receiver_trace, len(base.ROLES))
                receiver_logits = model(receiver_ids)[:, -1, base.CANDIDATE_SLICE].float()
                donor_logits = model(donor_ids)[:, -1, base.CANDIDATE_SLICE].float().view(len(base.ROLES), batch, 4)
                receiver_answers = torch.tensor([row["receiver_answer"] for row in batch_rows], device=device)
                donor_answers = torch.stack([torch.tensor([row["variants"][role]["answer"] for row in batch_rows], device=device) for role in base.ROLES])
                for stage, stage_rows in stage_events.items():
                    masks = [event["mask"] for event in stage_rows]
                    forward = base.forward_masks_logits(model, receiver_repeat, donor_trace, masks, stage).view(len(masks), len(base.ROLES), batch, 4)
                    reverse = base.forward_masks_logits(model, donor_ids, receiver_trace_repeat, masks, stage).view(len(masks), len(base.ROLES), batch, 4)
                    for local_index, event in enumerate(stage_rows):
                        global_index = event_index[event["event_id"]]
                        f_logits, r_logits = forward[local_index], reverse[local_index]
                        desired = (f_logits.argmax(-1) == donor_answers).float().sum(dim=1).cpu().numpy()
                        reverse_desired = (r_logits.argmax(-1) == receiver_answers[None]).float().sum(dim=1).cpu().numpy()
                        f_switch = (f_logits.argmax(-1) != receiver_answers[None]).float().sum(dim=1).cpu().numpy()
                        r_switch = (r_logits.argmax(-1) != donor_answers).float().sum(dim=1).cpu().numpy()
                        f_strength = torch.stack([base.response_strength(f_logits[role], receiver_logits, donor_logits[role]).sum() for role in range(len(base.ROLES))]).cpu().numpy()
                        r_strength = torch.stack([base.response_strength(r_logits[role], donor_logits[role], receiver_logits).sum() for role in range(len(base.ROLES))]).cpu().numpy()
                        sums[partition][global_index] += np.stack([desired, reverse_desired, f_switch, r_switch, f_strength, r_strength], axis=1)
                counts[partition] += batch
                del receiver_ids, donor_ids, receiver_trace, donor_trace, receiver_trace_repeat, receiver_logits, donor_logits, forward, reverse
            sums[partition] /= float(counts[partition])
    full_event_id = f"attention_prefix.l{base.ARCHITECTURES[architecture].layers - 1}"
    full = sums["confirmation"][event_index[full_event_id]]
    exemplar = next(row for row in rows if row["task"] == task)
    active_by_role = {role: bool(exemplar["variants"][role]["active"]) for role in base.ROLES}
    checks: dict[str, bool] = {}
    for role_index, role in enumerate(base.ROLES):
        if active_by_role[role]: checks[f"{role}.positive"] = bool(min(full[role_index, 0], full[role_index, 1]) >= base.THRESHOLDS["control_positive_min"])
        else: checks[f"{role}.null"] = bool(max(full[role_index, 2], full[role_index, 3]) <= base.THRESHOLDS["control_null_switch_max"])
    passed = bool(all(checks.values()))
    return {"events": events, "roles": list(base.ROLES), "readouts": list(base.READOUTS), "active_by_role": active_by_role, "responses": {partition: sums[partition].tolist() for partition in base.PARTITION_COUNTS}, "control_checks": checks, "controls_passed": passed, "execution_batch": batch_size}, passed


def smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = "deep8"
    base.set_seed(1_275_990_001)
    model = base.TinyCausalTransformer(base.ARCHITECTURES[architecture]).to(device).eval()
    rows = []
    for partition, offset in (("discovery", 7), ("confirmation", 11)):
        rng = np.random.default_rng(1_275_990_001 + offset)
        rows.extend(base.make_case("cyclic", rng, index, partition) for index in range(BATCH_BY_ARCHITECTURE[architecture]))
    started = time.perf_counter()
    response, passed = measure_model_fast(model, "cyclic", architecture, rows, device)
    print(canonical_json({"device": str(device), "architecture": architecture, "batch": BATCH_BY_ARCHITECTURE[architecture], "response_shape": list(np.asarray(response["responses"]["discovery"]).shape), "elapsed_seconds": time.perf_counter() - started, "controls": passed, "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0}))


def run(device_name: str) -> None:
    protocol, rows = verify_protocol()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda": raise RuntimeError("formal run requires CUDA")
    if any(path.exists() for path in (QUALIFICATION, MODELS, PAIRS, SUMMARY, COMPLETE)): raise RuntimeError("formal outputs already exist")
    qualification_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in base.TASKS:
        for architecture, config in base.ARCHITECTURES.items():
            for seed_index in range(base.SEEDS_PER_CELL):
                key = f"{task}.{architecture}.s{seed_index}"
                seed = base.MODEL_SEEDS[key]
                model, training = base.train_model(task, config, seed, device)
                behavior_passed = bool(training["qualification_accuracy"] >= base.THRESHOLDS["behavior_accuracy_min"])
                qualification_rows.append({"model_key": key, "task": task, "architecture": architecture, "seed_index": seed_index, "seed": seed, "training": training, "behavior_passed": behavior_passed, "qualification_status": "qualified" if behavior_passed else "behavior_rejected"})
                if behavior_passed:
                    response_tensor, controls_passed = measure_model_fast(model, task, architecture, rows, device)
                    model_rows.append({"model_key": key, "task": task, "architecture": architecture, "depth": config.layers, "seed_index": seed_index, "seed": seed, "controls_passed": controls_passed, "measurement_status": "measured", "claim_status": "abstained", "response_tensor": response_tensor})
                base.write_jsonl(QUALIFICATION, qualification_rows)
                base.write_jsonl(MODELS, model_rows)
                print(canonical_json({"model": key, "accuracy": training["qualification_accuracy"], "behavior": behavior_passed, "controls": model_rows[-1]["controls_passed"] if behavior_passed else None, "steps": training["steps"], "elapsed_total": time.perf_counter() - started}), flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    ledger = base.build_pair_ledger(model_rows)
    base.write_jsonl(PAIRS, ledger)
    elapsed = time.perf_counter() - started
    summary = {"phase": PHASE, "created_at_utc": utc_now(), "attempted_models": len(qualification_rows), "measured_models": len(model_rows), "pair_count": len(ledger), "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0, "device": torch.cuda.get_device_name(0), "protocol_digest": protocol["protocol_digest"], "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS), "run_digest": digest([row["model_key"] for row in qualification_rows]), "pretrained_model_loaded": False}
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"phase": PHASE, "complete": True, "created_at_utc": utc_now(), "run_digest": summary["run_digest"]})


def analyze() -> None:
    verify_protocol()
    qualification, models, ledger = base.read_jsonl(QUALIFICATION), base.read_jsonl(MODELS), base.read_jsonl(PAIRS)
    final, rebuilt = base.analyze_results(qualification, models, ledger)
    if base.digest(ledger) != base.digest(rebuilt): raise RuntimeError("pair ledger drift")
    final.update({"phase": PHASE, "contract_id": CONTRACT_ID, "created_at_utc": utc_now(), "protocol_hash": file_sha256(PROTOCOL), "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS), "phase1274_semantic_digest": semantic_digest(base.read_json(base.PROTOCOL))})
    final["final_digest"] = digest({key: value for key, value in final.items() if key not in {"created_at_utc", "final_digest"}})
    atomic_json(FINAL, final)
    print(canonical_json({"decision": final["decision"], "passed": final["passed"], "selected": final["selected_camera"], "executable": final["selected_executable_camera"], "behavior": final["behavior_qualified_models"], "controls": final["control_passed_models"]}))


def run_auditor(mode: str) -> None:
    status = os.spawnv(os.P_WAIT, sys.executable, [sys.executable, str(AUDITOR), mode])
    if status: raise SystemExit(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "preregister", "preaudit", "run", "analyze", "audit"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mode == "smoke": smoke()
    elif args.mode == "preregister": preregister(args.force)
    elif args.mode == "preaudit": run_auditor("preaudit")
    elif args.mode == "run": run(args.device)
    elif args.mode == "analyze": analyze()
    else: run_auditor("final")


if __name__ == "__main__": main()
