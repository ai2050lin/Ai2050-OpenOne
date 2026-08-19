from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1274_c021_multitask_free_response_isomorphism as base  # noqa: E402
import phase1277_c022_formation_dynamics_instrument_calibration as instrument  # noqa: E402


PHASE = 1278
CAMPAIGN = "C022"
CONTRACT_ID = "EXP-C022-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1278_c022_free_formation_trajectory_prediction"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment.json"
MATERIAL = OUT / "material/frozen_role_panel.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
DISCOVERY_SUMMARY = OUT / "raw/discovery/run_summary.json"
CONFIRMATION_SUMMARY = OUT / "raw/confirmation/run_summary.json"
PREDICTOR = OUT / "analysis/discovery_predictor_seal.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MAIN = ROOT / "tests/glm5/phase1278_c022_free_formation_trajectory_prediction.py"
AUDITOR = ROOT / "tests/glm5/phase1278_c022_free_formation_trajectory_prediction_audit.py"
P1277_FINAL = instrument.FINAL
P1277_AUDIT = instrument.AUDIT

CELLS = tuple(instrument.CELLS)
SEEDS_PER_CELL = instrument.SEEDS_PER_CELL
DISCOVERY_SEEDS_PER_CELL = instrument.DISCOVERY_SEEDS_PER_CELL
MODEL_SEEDS = instrument.MODEL_SEEDS
BEHAVIOR_STEPS = tuple(instrument.BEHAVIOR_STEPS)
INTERNAL_STEPS = tuple(instrument.INTERNAL_STEPS)
STATE_STEPS = tuple(instrument.STATE_STEPS)
THRESHOLDS = instrument.THRESHOLDS
BASELINE_FEATURES = tuple(instrument.BASELINE_FEATURES)
INTERNAL_FEATURES = tuple(instrument.INTERNAL_FEATURES)
ROLE_PANEL_COUNT = 128
ROLE_PANEL_SEEDS = {task: 1_278_800_000 + 1009 * index + 43 for index, task in enumerate(base.TASKS)}
EVALUATION_SEEDS = {task: 1_278_900_000 + 1009 * index + 47 for index, task in enumerate(base.TASKS)}
EVALUATION_COUNT = 4096
TRAINING_BATCH = 512


def model_key(cell: dict[str, str], seed_index: int) -> str:
    return f"{cell['cell']}.s{seed_index}"


def split_for_seed(seed_index: int) -> str:
    return "discovery" if seed_index < DISCOVERY_SEEDS_PER_CELL else "confirmation"


def model_file(split: str, key: str) -> Path:
    return OUT / f"raw/{split}/models/{key}.json"


def checkpoint_file(split: str, key: str, step: int) -> Path:
    return OUT / f"runs/{split}/checkpoints/{key}.step{step:05d}.pt"


def make_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in base.TASKS:
        rng = np.random.default_rng(ROLE_PANEL_SEEDS[task])
        for index in range(ROLE_PANEL_COUNT):
            rows.append(base.make_case(task, rng, index, "formation"))
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p1277 = instrument.read_json(P1277_FINAL)
    p1277_audit = instrument.read_json(P1277_AUDIT)
    return {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1278.c022.free_formation_trajectory_prediction.v1",
        "claim_type": "deterministic_free_transformer_early_internal_increment_prediction",
        "parent_contract": instrument.CONTRACT_ID,
        "parent_final_digest": p1277["final_digest"],
        "parent_audit_digest": p1277_audit["audit_digest"],
        "cells": list(CELLS),
        "seeds_per_cell": SEEDS_PER_CELL,
        "discovery_seeds_per_cell": DISCOVERY_SEEDS_PER_CELL,
        "model_seeds": MODEL_SEEDS,
        "behavior_steps": list(BEHAVIOR_STEPS),
        "internal_steps": list(INTERNAL_STEPS),
        "state_steps": list(STATE_STEPS),
        "role_panel_count_per_task": ROLE_PANEL_COUNT,
        "role_panel_seeds": ROLE_PANEL_SEEDS,
        "evaluation_count": EVALUATION_COUNT,
        "evaluation_seeds": EVALUATION_SEEDS,
        "training_batch": TRAINING_BATCH,
        "baseline_features": list(BASELINE_FEATURES),
        "internal_features": list(INTERNAL_FEATURES),
        "thresholds": THRESHOLDS,
        "outcome": "Y=1-min(T_stable,H)/H, where T_stable is the second of two adjacent registered behavior observations with accuracy >=0.995 and H=7000; no event gives Y=0",
        "split_order": "discovery training -> discovery predictor seal -> confirmation training -> final score",
        "primary_endpoint": instrument.read_json(instrument.PROTOCOL)["formal_primary_endpoint"],
        "branch_rule": instrument.read_json(instrument.PROTOCOL)["formal_branch_rule"],
        "hard_stops": instrument.read_json(instrument.PROTOCOL)["hard_stops"],
        "material_hash": instrument.digest([row["row_digest"] for row in rows]),
        "source_hashes": {
            "main": instrument.file_sha256(MAIN),
            "auditor": instrument.file_sha256(AUDITOR),
            "phase1277_final": instrument.file_sha256(P1277_FINAL),
            "phase1277_audit": instrument.file_sha256(P1277_AUDIT),
        },
        "created_at_utc": instrument.utc_now(),
    }


def preregister(force: bool) -> None:
    parent = instrument.read_json(P1277_FINAL)
    parent_audit = instrument.read_json(P1277_AUDIT)
    if not parent["formal_execution_authorized"] or not parent_audit["passed"]:
        raise RuntimeError("Phase1277 did not authorize formal execution")
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    if force and any(path.exists() for path in (DISCOVERY_SUMMARY, PREDICTOR, CONFIRMATION_SUMMARY, FINAL)):
        raise RuntimeError("cannot replace protocol after formal evidence exists")
    rows = make_material()
    instrument.write_jsonl(MATERIAL, rows)
    payload = protocol_payload(rows)
    payload["protocol_digest"] = instrument.digest(payload)
    instrument.atomic_json(PROTOCOL, payload)
    instrument.atomic_json(ENVIRONMENT, {
        "created_at_utc": instrument.utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "determinism": "torch deterministic algorithms; math SDP only; TF32 disabled",
        "precision": "fp32 training/intervention; fp64 analysis",
    })
    print(instrument.canonical_json({"status": "registered", "models": len(MODEL_SEEDS), "rows": len(rows), "protocol_digest": payload["protocol_digest"]}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = instrument.read_json(PROTOCOL)
    if protocol["protocol_digest"] != instrument.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest mismatch")
    expected_hashes = {
        "main": instrument.file_sha256(MAIN),
        "auditor": instrument.file_sha256(AUDITOR),
        "phase1277_final": instrument.file_sha256(P1277_FINAL),
        "phase1277_audit": instrument.file_sha256(P1277_AUDIT),
    }
    if protocol["source_hashes"] != expected_hashes:
        raise RuntimeError("protocol source mismatch")
    rows = instrument.read_jsonl(MATERIAL)
    if protocol["material_hash"] != instrument.digest([row["row_digest"] for row in rows]):
        raise RuntimeError("material hash mismatch")
    return protocol, rows


def evaluate(model, task: str, device: torch.device) -> dict[str, float | bool]:
    inputs, labels = base.random_batch(task, EVALUATION_COUNT, EVALUATION_SEEDS[task])
    all_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, EVALUATION_COUNT, 1024):
            logits = model(inputs[start : start + 1024].to(device))[:, -1, base.CANDIDATE_SLICE].float().cpu()
            all_logits.append(logits)
    logits = torch.cat(all_logits, dim=0)
    loss = F.cross_entropy(logits, labels)
    probability = logits.softmax(dim=-1)
    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    margin = correct_logits - other.max(dim=-1).values
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
    return {
        "accuracy": float((logits.argmax(dim=-1) == labels).float().mean().item()),
        "loss": float(loss.item()),
        "margin": float(margin.mean().item()),
        "entropy": float(entropy.mean().item()),
        "all_finite": bool(torch.isfinite(logits).all()),
    }


def prefix_events(layers: int) -> list[dict[str, Any]]:
    return [
        {"event_id": f"{stage}.prefix.l{layer}", "stage": stage, "layer": layer, "mask": list(range(layer + 1))}
        for stage in ("attn_write", "mlp_write")
        for layer in range(layers)
    ]


def measure_internal(model, task: str, architecture: str, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    task_rows = [row for row in rows if row["task"] == task]
    layers = base.ARCHITECTURES[architecture].layers
    events = prefix_events(layers)
    event_index = {event["event_id"]: index for index, event in enumerate(events)}
    stage_events = {stage: [event for event in events if event["stage"] == stage] for stage in ("attn_write", "mlp_write")}
    sums = np.zeros((len(events), len(base.ROLES), len(base.READOUTS)), dtype=np.float64)
    batch_size = 64 if layers <= 4 else 32
    with torch.inference_mode():
        for start in range(0, len(task_rows), batch_size):
            batch_rows = task_rows[start : start + batch_size]
            batch = len(batch_rows)
            receiver_ids = torch.tensor([row["receiver_ids"] for row in batch_rows], device=device)
            donor_ids = torch.cat([
                torch.tensor([row["variants"][role]["ids"] for row in batch_rows], device=device)
                for role in base.ROLES
            ])
            receiver_repeat = receiver_ids.repeat(len(base.ROLES), 1)
            receiver_trace = base.micro.capture_micro(model, receiver_ids)
            donor_trace = base.micro.capture_micro(model, donor_ids)
            receiver_trace_repeat = base.repeat_trace(receiver_trace, len(base.ROLES))
            receiver_logits = model(receiver_ids)[:, -1, base.CANDIDATE_SLICE].float()
            donor_logits = model(donor_ids)[:, -1, base.CANDIDATE_SLICE].float().view(len(base.ROLES), batch, 4)
            receiver_answers = torch.tensor([row["receiver_answer"] for row in batch_rows], device=device)
            donor_answers = torch.stack([
                torch.tensor([row["variants"][role]["answer"] for row in batch_rows], device=device)
                for role in base.ROLES
            ])
            for stage, selected_events in stage_events.items():
                masks = [event["mask"] for event in selected_events]
                forward = base.forward_masks_logits(model, receiver_repeat, donor_trace, masks, stage).view(len(masks), len(base.ROLES), batch, 4)
                reverse = base.forward_masks_logits(model, donor_ids, receiver_trace_repeat, masks, stage).view(len(masks), len(base.ROLES), batch, 4)
                for local_index, event in enumerate(selected_events):
                    target = sums[event_index[event["event_id"]]]
                    for role_index in range(len(base.ROLES)):
                        f_logits = forward[local_index, role_index]
                        r_logits = reverse[local_index, role_index]
                        donor_answer = donor_answers[role_index]
                        target[role_index] += [
                            float((f_logits.argmax(-1) == donor_answer).float().sum().item()),
                            float((r_logits.argmax(-1) == receiver_answers).float().sum().item()),
                            float((f_logits.argmax(-1) != receiver_answers).float().sum().item()),
                            float((r_logits.argmax(-1) != donor_answer).float().sum().item()),
                            float(base.response_strength(f_logits, receiver_logits, donor_logits[role_index]).sum().item()),
                            float(base.response_strength(r_logits, donor_logits[role_index], receiver_logits).sum().item()),
                        ]
            del receiver_ids, donor_ids, receiver_trace, donor_trace, receiver_trace_repeat, receiver_logits, donor_logits
    sums /= float(len(task_rows))
    active_by_role = {
        role: bool(task_rows[0]["variants"][role]["active"])
        for role in base.ROLES
    }
    active_indices = [index for index, role in enumerate(base.ROLES) if active_by_role[role]]
    null_indices = [index for index, role in enumerate(base.ROLES) if not active_by_role[role]]
    stage_scores: dict[str, float] = {}
    for stage in ("attn_write", "mlp_write"):
        full = sums[event_index[f"{stage}.prefix.l{layers - 1}"]]
        active_desired = float(full[active_indices, :2].mean())
        null_switch = float(full[null_indices, 2:4].mean())
        stage_scores[stage] = active_desired - null_switch
    summary = {
        "role_selectivity": 0.5 * (stage_scores["attn_write"] + stage_scores["mlp_write"]),
        "attention_selectivity": stage_scores["attn_write"],
        "mlp_selectivity": stage_scores["mlp_write"],
        "active_role_count": len(active_indices),
        "null_role_count": len(null_indices),
    }
    return {
        "events": events,
        "roles": list(base.ROLES),
        "readouts": list(base.READOUTS),
        "active_by_role": active_by_role,
        "responses": sums.tolist(),
        "summary": summary,
        "all_finite": bool(np.isfinite(sums).all()),
        "panel_count": len(task_rows),
    }


def save_state(model, optimizer, split: str, key: str, task: str, architecture: str, seed: int, step: int) -> dict[str, Any]:
    path = checkpoint_file(split, key, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    payload = {
        "phase": PHASE,
        "model_key": key,
        "task": task,
        "architecture": architecture,
        "seed": seed,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"step": step, "path": str(path.relative_to(ROOT)), "sha256": instrument.file_sha256(path), "bytes": path.stat().st_size}


def stable_event_step(trajectory: list[dict[str, Any]], budget: int) -> int | None:
    ordered = [row for row in sorted(trajectory, key=lambda value: value["step"]) if row["step"] <= budget]
    required = int(THRESHOLDS["stable_adjacent_observations_min"])
    run = 0
    for row in ordered:
        run = run + 1 if row["accuracy"] >= THRESHOLDS["stable_accuracy_min"] else 0
        if run >= required:
            return int(row["step"])
    return None


def train_one(cell: dict[str, str], seed_index: int, split: str, material: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    key = model_key(cell, seed_index)
    seed = MODEL_SEEDS[key]
    task = cell["task"]
    architecture = cell["architecture"]
    instrument.configure_determinism(seed)
    model = base.TinyCausalTransformer(base.ARCHITECTURES[architecture]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    behavior_step_set = set(BEHAVIOR_STEPS)
    internal_step_set = set(INTERNAL_STEPS)
    state_step_set = set(STATE_STEPS)
    trajectory: list[dict[str, Any]] = []
    internal: dict[str, Any] = {}
    checkpoints: list[dict[str, Any]] = []
    last_gradient_norm = 0.0
    started = time.perf_counter()
    extended = True
    max_step = int(THRESHOLDS["extended_budget"])
    for step in range(max_step + 1):
        if step in behavior_step_set:
            model.eval()
            metrics = evaluate(model, task, device)
            trajectory.append({"step": step, "gradient_norm": last_gradient_norm, **metrics})
            model.train()
        if step in internal_step_set:
            model.eval()
            internal[str(step)] = measure_internal(model, task, architecture, material, device)
            model.train()
        if step in state_step_set and (step != int(THRESHOLDS["extended_budget"]) or extended):
            checkpoints.append(save_state(model, optimizer, split, key, task, architecture, seed, step))
        if step == int(THRESHOLDS["fixed_budget"]):
            fixed_event = stable_event_step(trajectory, int(THRESHOLDS["fixed_budget"]))
            extended = fixed_event is None
            if not extended:
                break
        if step == max_step:
            break
        if step == 3500:
            optimizer.param_groups[0]["lr"] = 2.0e-4
        inputs, labels = base.random_batch(task, TRAINING_BATCH, seed + 10_000 + step)
        logits = model(inputs.to(device))[:, -1, base.CANDIDATE_SLICE].float()
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        last_gradient_norm = float(gradient_norm.item())
        optimizer.step()
    fixed_event = stable_event_step(trajectory, int(THRESHOLDS["fixed_budget"]))
    extended_event = stable_event_step(trajectory, int(THRESHOLDS["extended_budget"]))
    fixed_budget = float(THRESHOLDS["fixed_budget"])
    formation_progress = 0.0 if fixed_event is None else 1.0 - float(fixed_event) / fixed_budget
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "model_key": key,
        "cell": cell["cell"],
        "task": task,
        "architecture": architecture,
        "depth": base.ARCHITECTURES[architecture].layers,
        "seed_index": seed_index,
        "seed": seed,
        "split": split,
        "trajectory": trajectory,
        "internal": internal,
        "checkpoints": checkpoints,
        "fixed_event_step": fixed_event,
        "extended_event_step": extended_event,
        "fixed_success": fixed_event is not None,
        "delayed_success": fixed_event is None and extended_event is not None,
        "extended_censored": extended_event is None,
        "formation_progress": formation_progress,
        "all_finite": all(bool(row["all_finite"]) and all(math.isfinite(float(row[field])) for field in ("accuracy", "loss", "margin", "entropy", "gradient_norm")) for row in trajectory) and all(value["all_finite"] for value in internal.values()),
        "trained_steps": int(THRESHOLDS["extended_budget"] if extended else THRESHOLDS["fixed_budget"]),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "deterministic_execution": True,
    }
    result["model_digest"] = instrument.digest({key_: value for key_, value in result.items() if key_ != "elapsed_seconds"})
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def split_models(split: str) -> list[tuple[dict[str, str], int]]:
    return [
        (cell, seed_index)
        for cell in CELLS
        for seed_index in range(SEEDS_PER_CELL)
        if split_for_seed(seed_index) == split
    ]


def run_split(split: str) -> None:
    verify_protocol()
    if split == "discovery" and PREDICTOR.exists():
        raise RuntimeError("discovery cannot run after predictor seal")
    if split == "confirmation" and not PREDICTOR.exists():
        raise RuntimeError("confirmation requires frozen predictor")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _, material = verify_protocol()
    device = torch.device("cuda")
    started_at = instrument.utc_now()
    started = time.perf_counter()
    expected = split_models(split)
    for ordinal, (cell, seed_index) in enumerate(expected, start=1):
        key = model_key(cell, seed_index)
        path = model_file(split, key)
        if path.exists():
            existing = instrument.read_json(path)
            if existing["seed"] != MODEL_SEEDS[key] or existing["split"] != split:
                raise RuntimeError(f"existing model mismatch: {key}")
            print(instrument.canonical_json({"split": split, "model": key, "status": "resumed", "ordinal": ordinal, "total": len(expected)}), flush=True)
            continue
        result = train_one(cell, seed_index, split, material, device)
        instrument.atomic_json(path, result)
        print(instrument.canonical_json({
            "split": split,
            "model": key,
            "ordinal": ordinal,
            "total": len(expected),
            "fixed_event": result["fixed_event_step"],
            "extended_event": result["extended_event_step"],
            "seconds": result["elapsed_seconds"],
        }), flush=True)
    paths = [model_file(split, model_key(cell, seed_index)) for cell, seed_index in expected]
    models = [instrument.read_json(path) for path in paths]
    summary_path = DISCOVERY_SUMMARY if split == "discovery" else CONFIRMATION_SUMMARY
    summary = {
        "phase": PHASE,
        "split": split,
        "started_at_utc": started_at,
        "completed_at_utc": instrument.utc_now(),
        "model_count": len(models),
        "fixed_event_count": sum(model["fixed_success"] for model in models),
        "delayed_event_count": sum(model["delayed_success"] for model in models),
        "extended_censored_count": sum(model["extended_censored"] for model in models),
        "all_finite": all(model["all_finite"] for model in models),
        "model_hashes": {path.stem: instrument.file_sha256(path) for path in paths},
        "model_digests": {model["model_key"]: model["model_digest"] for model in models},
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "protocol_digest": instrument.read_json(PROTOCOL)["protocol_digest"],
    }
    summary["summary_digest"] = instrument.digest(summary)
    instrument.atomic_json(summary_path, summary)
    print(instrument.canonical_json({"split": split, "models": len(models), "events": summary["fixed_event_count"], "delayed": summary["delayed_event_count"], "censored": summary["extended_censored_count"]}))


def row_at(model: dict[str, Any], step: int) -> dict[str, Any]:
    return next(row for row in model["trajectory"] if int(row["step"]) == step)


def features(model: dict[str, Any], augmented: bool) -> tuple[list[str], list[float]]:
    step512 = row_at(model, 512)
    step768 = row_at(model, 768)
    names = [f"cell::{cell['cell']}" for cell in CELLS] + list(BASELINE_FEATURES)
    values = [float(model["cell"] == cell["cell"]) for cell in CELLS] + [
        float(step512["accuracy"]),
        float(step768["accuracy"]),
        float(step768["accuracy"] - step512["accuracy"]),
        float(step512["loss"]),
        float(step768["loss"]),
        float(step768["loss"] - step512["loss"]),
        float(step768["margin"]),
        float(step768["entropy"]),
        float(step768["gradient_norm"]),
    ]
    if augmented:
        internal512 = model["internal"]["512"]["summary"]
        internal768 = model["internal"]["768"]["summary"]
        names.extend(INTERNAL_FEATURES)
        values.extend([
            float(internal512["role_selectivity"]),
            float(internal768["role_selectivity"]),
            float(internal768["role_selectivity"] - internal512["role_selectivity"]),
            float(internal768["attention_selectivity"]),
            float(internal768["mlp_selectivity"]),
        ])
    return names, values


def load_models(split: str) -> list[dict[str, Any]]:
    return [
        instrument.read_json(model_file(split, model_key(cell, seed_index)))
        for cell, seed_index in split_models(split)
    ]


def object_gate(models: list[dict[str, Any]], split: str) -> dict[str, Any]:
    events = [model for model in models if model["fixed_success"]]
    censored = [model for model in models if not model["fixed_success"]]
    informative = []
    per_cell: dict[str, Any] = {}
    for cell in CELLS:
        selected = [model for model in models if model["cell"] == cell["cell"]]
        event_count = sum(model["fixed_success"] for model in selected)
        censored_count = len(selected) - event_count
        if event_count and censored_count:
            informative.append(cell["cell"])
        per_cell[cell["cell"]] = {"count": len(selected), "event_count": event_count, "censored_count": censored_count}
    event_min = int(THRESHOLDS[f"{split}_event_min"])
    censored_min = int(THRESHOLDS[f"{split}_censored_min"])
    gates = {
        "model_count": len(models) == len(CELLS) * (DISCOVERY_SEEDS_PER_CELL if split == "discovery" else SEEDS_PER_CELL - DISCOVERY_SEEDS_PER_CELL),
        "event_count": len(events) >= event_min,
        "censored_count": len(censored) >= censored_min,
        "informative_cells": len(informative) >= THRESHOLDS["informative_cells_per_split_min"],
        "all_finite": all(model["all_finite"] for model in models),
    }
    return {
        "split": split,
        "event_count": len(events),
        "censored_count": len(censored),
        "informative_cells": informative,
        "per_cell": per_cell,
        "gates": gates,
        "passed": all(gates.values()),
    }


def fit_discovery() -> None:
    verify_protocol()
    if not DISCOVERY_SUMMARY.exists():
        raise RuntimeError("discovery run is incomplete")
    if PREDICTOR.exists():
        raise RuntimeError("predictor already sealed")
    confirmation_existing = list((OUT / "raw/confirmation/models").glob("*.json")) if (OUT / "raw/confirmation/models").exists() else []
    if confirmation_existing or CONFIRMATION_SUMMARY.exists():
        raise RuntimeError("confirmation exists before predictor seal")
    models = load_models("discovery")
    object_decision = object_gate(models, "discovery")
    baseline_names, baseline_rows = [], []
    augmented_names, augmented_rows = [], []
    outcomes = []
    for model in models:
        baseline_name, baseline_value = features(model, False)
        augmented_name, augmented_value = features(model, True)
        baseline_names.append(baseline_name)
        augmented_names.append(augmented_name)
        baseline_rows.append(baseline_value)
        augmented_rows.append(augmented_value)
        outcomes.append(float(model["formation_progress"]))
    if any(names != baseline_names[0] for names in baseline_names) or any(names != augmented_names[0] for names in augmented_names):
        raise RuntimeError("feature order drift")
    y = np.asarray(outcomes, dtype=np.float64)
    baseline_x = np.asarray(baseline_rows, dtype=np.float64)
    augmented_x = np.asarray(augmented_rows, dtype=np.float64)
    cell_means = {
        cell["cell"]: float(np.mean([model["formation_progress"] for model in models if model["cell"] == cell["cell"]]))
        for cell in CELLS
    }
    seal: dict[str, Any] = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": instrument.utc_now(),
        "confirmation_absent_at_seal": True,
        "confirmation_model_count_at_seal": 0,
        "object_gate": object_decision,
        "feature_cutoff": THRESHOLDS["prediction_cutoff"],
        "baseline_feature_names": baseline_names[0],
        "augmented_feature_names": augmented_names[0],
        "cell_prior": cell_means,
        "models": None,
        "confirmation_authorized": object_decision["passed"],
        "protocol_digest": instrument.read_json(PROTOCOL)["protocol_digest"],
    }
    if object_decision["passed"]:
        baseline_model = instrument.fit_ridge(baseline_x, y, THRESHOLDS["ridge_alpha"])
        augmented_model = instrument.fit_ridge(augmented_x, y, THRESHOLDS["ridge_alpha"])
        seal["models"] = {"baseline": baseline_model, "augmented": augmented_model}
        seal["discovery_scores"] = {
            "cell_prior_mae": instrument.mae(y, np.asarray([cell_means[model["cell"]] for model in models])),
            "baseline_mae": instrument.mae(y, instrument.apply_ridge(baseline_model, baseline_x)),
            "augmented_mae": instrument.mae(y, instrument.apply_ridge(augmented_model, augmented_x)),
        }
    seal["predictor_digest"] = instrument.digest(seal)
    instrument.atomic_json(PREDICTOR, seal)
    print(instrument.canonical_json({"object_pass": object_decision["passed"], "events": object_decision["event_count"], "censored": object_decision["censored_count"], "confirmation_authorized": seal["confirmation_authorized"]}))


def score_predictions(models: list[dict[str, Any]], predictor: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray([model["formation_progress"] for model in models], dtype=np.float64)
    baseline_x = np.asarray([features(model, False)[1] for model in models], dtype=np.float64)
    augmented_x = np.asarray([features(model, True)[1] for model in models], dtype=np.float64)
    cell_prediction = np.asarray([predictor["cell_prior"][model["cell"]] for model in models], dtype=np.float64)
    baseline_prediction = instrument.apply_ridge(predictor["models"]["baseline"], baseline_x)
    augmented_prediction = instrument.apply_ridge(predictor["models"]["augmented"], augmented_x)
    cell_mae = instrument.mae(y, cell_prediction)
    baseline_mae = instrument.mae(y, baseline_prediction)
    augmented_mae = instrument.mae(y, augmented_prediction)
    best_behavior = min(cell_mae, baseline_mae)
    relative = (best_behavior - augmented_mae) / best_behavior if best_behavior > 0 else 0.0
    groups = [model["cell"] for model in models]
    baseline_order = instrument.pair_order_accuracy(y, baseline_prediction, groups)
    augmented_order = instrument.pair_order_accuracy(y, augmented_prediction, groups)
    baseline_order_accuracy = baseline_order["accuracy"] if baseline_order["accuracy"] is not None else 0.0
    augmented_order_accuracy = augmented_order["accuracy"] if augmented_order["accuracy"] is not None else 0.0
    per_cell: dict[str, Any] = {}
    for cell in CELLS:
        indices = [index for index, model in enumerate(models) if model["cell"] == cell["cell"]]
        cell_y = y[indices]
        cell_baseline = baseline_prediction[indices]
        cell_augmented = augmented_prediction[indices]
        per_cell[cell["cell"]] = {
            "count": len(indices),
            "baseline_mae": instrument.mae(cell_y, cell_baseline),
            "augmented_mae": instrument.mae(cell_y, cell_augmented),
            "augmented_wins": instrument.mae(cell_y, cell_augmented) < instrument.mae(cell_y, cell_baseline),
        }
    return {
        "cell_prior_mae": cell_mae,
        "baseline_mae": baseline_mae,
        "augmented_mae": augmented_mae,
        "best_behavior_mae": best_behavior,
        "relative_mae_improvement": relative,
        "baseline_pair_order": baseline_order,
        "augmented_pair_order": augmented_order,
        "pair_order_advantage": augmented_order_accuracy - baseline_order_accuracy,
        "per_cell": per_cell,
        "cell_win_count": sum(value["augmented_wins"] for value in per_cell.values()),
        "lookup_depth_win_count": sum(per_cell[name]["augmented_wins"] for name in ("context_lookup.shallow4", "context_lookup.deep8")),
        "predictions": [
            {
                "model_key": model["model_key"],
                "cell": model["cell"],
                "outcome": float(y[index]),
                "cell_prior": float(cell_prediction[index]),
                "baseline": float(baseline_prediction[index]),
                "augmented": float(augmented_prediction[index]),
            }
            for index, model in enumerate(models)
        ],
    }


def finalize() -> None:
    protocol, _ = verify_protocol()
    predictor = instrument.read_json(PREDICTOR)
    discovery_models = load_models("discovery")
    if not predictor["confirmation_authorized"]:
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "contract_id": CONTRACT_ID,
            "stage": "discovery_object_gate_failure",
            "discovery_object_gate": predictor["object_gate"],
            "passed": False,
            "decision": "formation_prediction_not_tested_due_to_discovery_object_failure",
            "causal_branch_authorized": False,
            "small_synthetic_camera_escalation_closed": True,
            "protocol_digest": protocol["protocol_digest"],
            "predictor_digest": predictor["predictor_digest"],
            "created_at_utc": instrument.utc_now(),
        }
    else:
        if not CONFIRMATION_SUMMARY.exists():
            raise RuntimeError("confirmation run is incomplete")
        confirmation_models = load_models("confirmation")
        confirmation_object = object_gate(confirmation_models, "confirmation")
        scores = score_predictions(confirmation_models, predictor)
        augmented_order = scores["augmented_pair_order"]["accuracy"] or 0.0
        gates = {
            "discovery_object": predictor["object_gate"]["passed"],
            "confirmation_object": confirmation_object["passed"],
            "relative_mae_increment": scores["relative_mae_improvement"] >= THRESHOLDS["augmented_relative_mae_improvement_min"],
            "pair_order_absolute": augmented_order >= THRESHOLDS["augmented_pair_order_accuracy_min"],
            "pair_order_advantage": scores["pair_order_advantage"] >= THRESHOLDS["augmented_pair_order_advantage_min"],
            "cell_breadth": scores["cell_win_count"] >= THRESHOLDS["confirmation_cell_win_min"],
            "lookup_depth_breadth": scores["lookup_depth_win_count"] >= THRESHOLDS["lookup_depth_win_min"],
            "all_finite": all(model["all_finite"] for model in discovery_models + confirmation_models),
            "confirmation_after_predictor_seal": instrument.read_json(CONFIRMATION_SUMMARY)["started_at_utc"] > predictor["created_at_utc"],
        }
        passed = all(gates.values())
        extended = {
            split: {
                "fixed_success": sum(model["fixed_success"] for model in models),
                "delayed_success": sum(model["delayed_success"] for model in models),
                "extended_censored": sum(model["extended_censored"] for model in models),
            }
            for split, models in (("discovery", discovery_models), ("confirmation", confirmation_models))
        }
        final = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "contract_id": CONTRACT_ID,
            "stage": "confirmation_prediction",
            "model_count": len(discovery_models) + len(confirmation_models),
            "discovery_object_gate": predictor["object_gate"],
            "confirmation_object_gate": confirmation_object,
            "scores": scores,
            "extended_budget": extended,
            "gates": gates,
            "passed": passed,
            "decision": "early_internal_formation_increment_confirmed" if passed else "early_internal_formation_increment_not_confirmed",
            "causal_branch_authorized": passed,
            "small_synthetic_camera_escalation_closed": not passed,
            "pretrained_model_authorized": False,
            "protocol_digest": protocol["protocol_digest"],
            "predictor_digest": predictor["predictor_digest"],
            "discovery_summary_hash": instrument.file_sha256(DISCOVERY_SUMMARY),
            "confirmation_summary_hash": instrument.file_sha256(CONFIRMATION_SUMMARY),
            "created_at_utc": instrument.utc_now(),
        }
    final["final_digest"] = instrument.digest(final)
    instrument.atomic_json(FINAL, final)
    print(instrument.canonical_json({"stage": final["stage"], "passed": final["passed"], "decision": final["decision"], "causal": final["causal_branch_authorized"]}))


def smoke() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    material = make_material()
    cell = CELLS[0]
    seed = MODEL_SEEDS[model_key(cell, 0)]
    instrument.configure_determinism(seed)
    device = torch.device("cuda")
    model = base.TinyCausalTransformer(base.ARCHITECTURES[cell["architecture"]]).to(device).eval()
    behavior = evaluate(model, cell["task"], device)
    internal = measure_internal(model, cell["task"], cell["architecture"], material, device)
    print(instrument.canonical_json({"behavior": behavior, "internal_shape": list(np.asarray(internal["responses"]).shape), "summary": internal["summary"], "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20}))


def run_auditor(mode: str) -> None:
    completed = subprocess.run([sys.executable, str(AUDITOR), mode], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "preregister", "preaudit", "smoke", "run-discovery", "fit-discovery",
        "run-confirmation", "finalize", "audit", "all",
    ))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"preregister", "all"}:
        preregister(args.force)
    if args.command in {"preaudit", "all"}:
        run_auditor("pre")
    if args.command == "smoke":
        smoke()
    if args.command in {"run-discovery", "all"}:
        run_split("discovery")
    if args.command in {"fit-discovery", "all"}:
        fit_discovery()
    if args.command in {"run-confirmation", "all"}:
        predictor = instrument.read_json(PREDICTOR)
        if predictor["confirmation_authorized"]:
            run_split("confirmation")
    if args.command in {"finalize", "all"}:
        finalize()
    if args.command in {"audit", "all"}:
        run_auditor("final")


if __name__ == "__main__":
    main()
