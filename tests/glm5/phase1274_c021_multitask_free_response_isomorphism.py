#!/usr/bin/env python3
"""Phase1274: three-task free-Transformer response quotient and isomorphism test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer
import phase1271_c019_cross_layer_micro_write_trajectory as micro
import phase1273_c021_response_isomorphism_camera_calibration as camera


PHASE = 1274
CAMPAIGN = "C021"
CONTRACT_ID = "EXP-C021-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1274_c021_multitask_free_response_isomorphism"
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
AUDITOR = ROOT / "tests/glm5/phase1274_c021_multitask_free_response_isomorphism_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C021-WP01-001.json"
PHASE1273_FINAL = ROOT / "tests/glm5/result/phase1273_c021_response_isomorphism_camera_calibration/analysis/final.json"
PHASE1273_AUDIT = ROOT / "tests/glm5/result/phase1273_c021_response_isomorphism_camera_calibration/audit/independent_final_audit.json"

BOS, TASK, REC, SEP, MAP, QUERY, ANSWER = range(7)
ENTITY_START, CODE_START, OP_START, LABEL_START, PAD = 7, 9, 13, 17, 21
VOCAB, LENGTH = 22, 23
ANSWER_POSITION = 22
CANDIDATE_SLICE = slice(LABEL_START, LABEL_START + 4)
TASKS = ("cyclic", "xor", "context_lookup")
DISCOVERY_TASKS = ("cyclic", "xor")
HELDOUT_TASK = "context_lookup"
ROLES = ("source_target", "source_wrong", "operator_target", "payload_target", "distractor_null")
READOUTS = ("forward_desired", "reverse_desired", "forward_switch", "reverse_switch", "forward_strength", "reverse_strength")
CAMERAS = camera.CAMERAS
EXECUTABLE_CAMERAS = ("identity_coordinate", "monotone_depth_warp")
ARCHITECTURES = {
    "shallow4": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
    "middle6": ModelConfig(layers=6, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
    "deep8": ModelConfig(layers=8, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
}
SEEDS_PER_CELL = 3
MODEL_SEEDS = {
    f"{task}.{architecture}.s{seed_index}": 1_274_000_000 + 100_000 * task_index + 10_000 * list(ARCHITECTURES).index(architecture) + 101 * seed_index + 17
    for task_index, task in enumerate(TASKS)
    for architecture in ARCHITECTURES
    for seed_index in range(SEEDS_PER_CELL)
}
MATERIAL_SEEDS = {"discovery": 1_274_700_001, "confirmation": 1_274_800_001}
PARTITION_COUNTS = {"discovery": 1024, "confirmation": 1024}
BEHAVIOR_EXAMPLES = 16384
TRAINING_BATCH = 512
TRAINING_STEPS_MAX = 7000
CAMERA_GRID = 8
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "behavior_models_min": 24,
    "behavior_per_task_depth_min": 2,
    "control_positive_min": 0.90,
    "control_null_switch_max": 0.10,
    "control_models_min": 18,
    "control_per_task_depth_min": 1,
    "unseen_task_balanced_accuracy_min": 0.80,
    "unseen_task_auc_min": 0.85,
    "unseen_seed_balanced_accuracy_min": 0.80,
    "joint_balanced_accuracy_min": 0.78,
    "mapping_positive_cosine_min": 0.85,
    "mapping_task_advantage_min": 0.08,
    "false_authorizations_max": 0,
}


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def answer_for(task: str, source: int, operator: int, payload: list[int]) -> int:
    if task == "cyclic": return (source + operator) % 4
    if task == "xor": return source ^ operator
    if task == "context_lookup": return int(payload[source])
    raise ValueError(task)


def build_sequence(source: int, distractor: int, operator: int, payload: list[int]) -> list[int]:
    values = [BOS, TASK, REC, ENTITY_START, CODE_START + source, SEP, REC, ENTITY_START + 1, CODE_START + distractor, SEP, MAP, OP_START + operator]
    for code in range(4):
        values.extend([CODE_START + code, LABEL_START + int(payload[code])])
    values.extend([QUERY, ENTITY_START, ANSWER])
    if len(values) != LENGTH: raise RuntimeError("sequence length drift")
    return values


def make_case(task: str, rng: np.random.Generator, index: int, partition: str) -> dict[str, Any]:
    source = index % 4
    operator = (index // 4) % 4
    distractor = int(rng.integers(4))
    payload = rng.permutation(4).astype(int).tolist()
    source_target = (source + 1 + (index // 16) % 3) % 4
    if source_target == source: source_target = (source + 1) % 4
    source_wrong = next(value for value in range(4) if value not in {source, source_target})
    operator_target = (operator + 1) % 4
    payload_target = list(payload)
    swap_code = (source + 1) % 4
    payload_target[source], payload_target[swap_code] = payload_target[swap_code], payload_target[source]
    distractor_target = (distractor + 1) % 4
    receiver_answer = answer_for(task, source, operator, payload)
    variants = {
        "source_target": (source_target, distractor, operator, payload),
        "source_wrong": (source_wrong, distractor, operator, payload),
        "operator_target": (source, distractor, operator_target, payload),
        "payload_target": (source, distractor, operator, payload_target),
        "distractor_null": (source, distractor_target, operator, payload),
    }
    row_variants: dict[str, Any] = {}
    for role, (variant_source, variant_distractor, variant_operator, variant_payload) in variants.items():
        variant_answer = answer_for(task, variant_source, variant_operator, variant_payload)
        row_variants[role] = {
            "ids": build_sequence(variant_source, variant_distractor, variant_operator, variant_payload),
            "answer": variant_answer,
            "active": variant_answer != receiver_answer,
        }
    row = {
        "row_id": f"{task}.{partition}.{index:04d}",
        "task": task,
        "partition": partition,
        "source": source,
        "operator": operator,
        "payload": payload,
        "receiver_ids": build_sequence(source, distractor, operator, payload),
        "receiver_answer": receiver_answer,
        "variants": row_variants,
    }
    row["row_digest"] = digest(row)
    return row


def make_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, count in PARTITION_COUNTS.items():
        for task_index, task in enumerate(TASKS):
            rng = np.random.default_rng(MATERIAL_SEEDS[partition] + 1009 * task_index)
            rows.extend(make_case(task, rng, index, partition) for index in range(count))
    return rows


def random_batch(task: str, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs, labels = [], []
    for _ in range(count):
        source, distractor, operator = (int(rng.integers(4)) for _ in range(3))
        payload = rng.permutation(4).astype(int).tolist()
        inputs.append(build_sequence(source, distractor, operator, payload))
        labels.append(answer_for(task, source, operator, payload))
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_behavior(model: TinyCausalTransformer, task: str, seed: int, device: torch.device, count: int = BEHAVIOR_EXAMPLES) -> float:
    correct = 0
    with torch.inference_mode():
        for start in range(0, count, 1024):
            size = min(1024, count - start)
            inputs, labels = random_batch(task, size, seed + start)
            predicted = model(inputs.to(device))[:, -1, CANDIDATE_SLICE].argmax(-1).cpu()
            correct += int((predicted == labels).sum())
    return correct / float(count)


def train_model(task: str, config: ModelConfig, seed: int, device: torch.device) -> tuple[TinyCausalTransformer, dict[str, Any]]:
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    started = time.perf_counter()
    screening = 0.0
    for step in range(TRAINING_STEPS_MAX):
        if step == 3500: optimizer.param_groups[0]["lr"] = 2.0e-4
        inputs, labels = random_batch(task, TRAINING_BATCH, seed + 10_000 + step)
        logits = model(inputs.to(device))[:, -1, CANDIDATE_SLICE].float()
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= 1800 and step % 100 == 99:
            screening = evaluate_behavior(model.eval(), task, seed + 8_000_000, device, 4096)
            model.train()
            if screening >= 0.997: break
    model.eval()
    final_accuracy = evaluate_behavior(model, task, seed + 9_000_000, device)
    return model, {"steps": step + 1, "screening_accuracy": screening, "qualification_accuracy": final_accuracy, "elapsed_seconds": time.perf_counter() - started, "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def program_registry(layers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, prefix in (("attn_write", "attention"), ("mlp_write", "mlp")):
        for kind in ("single", "prefix", "suffix"):
            for layer in range(layers):
                if kind == "single": mask = [layer]
                elif kind == "prefix": mask = list(range(layer + 1))
                else: mask = list(range(layer, layers))
                rows.append({"event_id": f"{prefix}_{kind}.l{layer}", "event_type": f"{prefix}_{kind}", "stage": stage, "layer": layer, "relative_layer": layer / max(layers - 1, 1), "mask": mask})
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration, audit = read_json(PHASE1273_FINAL), read_json(PHASE1273_AUDIT)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1274.c021.multitask_free_response_isomorphism.v1",
        "claim_type": "controlled_free_transformer_response_quotient_and_executable_mapping",
        "phase1273_dependency": {"passed": calibration.get("passed"), "decision": calibration.get("decision"), "audit_passed": audit.get("passed"), "final_hash": file_sha256(PHASE1273_FINAL), "audit_hash": file_sha256(PHASE1273_AUDIT)},
        "tasks": list(TASKS),
        "discovery_tasks": list(DISCOVERY_TASKS),
        "heldout_task": HELDOUT_TASK,
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "seeds_per_cell": SEEDS_PER_CELL,
        "model_seeds": MODEL_SEEDS,
        "planned_models": len(MODEL_SEEDS),
        "roles": list(ROLES),
        "readouts": list(READOUTS),
        "program_registry": {name: program_registry(config.layers) for name, config in ARCHITECTURES.items()},
        "partitions": PARTITION_COUNTS,
        "behavior_examples": BEHAVIOR_EXAMPLES,
        "training": {"batch": TRAINING_BATCH, "max_steps": TRAINING_STEPS_MAX, "precision": "fp32"},
        "cameras": list(CAMERAS),
        "executable_cameras": list(EXECUTABLE_CAMERAS),
        "selection_panel": "cyclic/xor, seed indices 0/1, discovery scores only",
        "sealed_panels": ["unseen_task", "unseen_seed", "joint_unseen_task_seed"],
        "thresholds": THRESHOLDS,
        "material_seed": MATERIAL_SEEDS,
        "row_count": len(rows),
        "material_digest": digest([{"row_id": row["row_id"], "row_digest": row["row_digest"]} for row in rows]),
        "hard_stops": [
            "Behavior-failed models are recorded and never replaced.",
            "Only behavior-qualified models may enter response measurement.",
            "No confirmation score selects a camera or threshold.",
            "Spectrum classification cannot authorize rescue without an executable mapping camera.",
            "Failure closes C021 synthetic local isomorphism search.",
            "No pretrained model is loaded or authorized.",
        ],
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR), "contract": file_sha256(CONTRACT), "model": file_sha256(ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"), "micro_executor": file_sha256(ROOT / "tests/glm5/phase1271_c019_cross_layer_micro_write_trajectory.py"), "camera": file_sha256(ROOT / "tests/glm5/phase1273_c021_response_isomorphism_camera_calibration.py")},
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {"created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "precision": "fp32 training and intervention; fp64 analysis"}


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force: raise RuntimeError("protocol already exists")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "registered", "rows": len(rows), "models": len(MODEL_SEEDS)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, rows = read_json(PROTOCOL), read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["protocol_digest"] != expected["protocol_digest"]: raise RuntimeError("frozen protocol/source mismatch")
    for row in rows:
        value, stored = dict(row), row["row_digest"]
        value.pop("row_digest")
        if digest(value) != stored: raise RuntimeError("material digest mismatch")
    return protocol, rows


def masks_tensor(masks: list[list[int]], layers: int, device: torch.device) -> torch.Tensor:
    values = torch.zeros((len(masks), layers), dtype=torch.bool, device=device)
    for index, mask in enumerate(masks):
        if mask: values[index, torch.tensor(mask, device=device)] = True
    return values


def repeat_trace(trace: list[dict[str, torch.Tensor]], repeats: int) -> list[dict[str, torch.Tensor]]:
    return [{key: value.repeat(repeats, 1, 1) for key, value in layer.items()} for layer in trace]


def forward_masks_logits(model: TinyCausalTransformer, receiver_ids: torch.Tensor, donor: list[dict[str, torch.Tensor]], masks: list[list[int]], stage: str) -> torch.Tensor:
    selected_masks = masks_tensor(masks, len(model.blocks), receiver_ids.device)
    mask_count, batch = len(masks), receiver_ids.shape[0]
    hidden = model.embed(receiver_ids).repeat(mask_count, 1, 1)
    causal = torch.triu(torch.ones(receiver_ids.shape[1], receiver_ids.shape[1], dtype=torch.bool, device=receiver_ids.device), diagonal=1)
    for layer, block in enumerate(model.blocks):
        attn_write, after_attn, mlp_write = micro.block_parts(block, hidden, causal)
        selected = selected_masks[:, layer].repeat_interleave(batch)
        if stage == "attn_write" and bool(selected.any()):
            replacement = donor[layer]["attn_write"][:, ANSWER_POSITION].repeat(mask_count, 1)
            attn_write = attn_write.clone()
            attn_write[:, ANSWER_POSITION] = torch.where(selected[:, None], replacement, attn_write[:, ANSWER_POSITION])
            after_attn = hidden + attn_write
            mlp_write = block.mlp(block.mlp_norm(after_attn))
        if stage == "mlp_write" and bool(selected.any()):
            replacement = donor[layer]["mlp_write"][:, ANSWER_POSITION].repeat(mask_count, 1)
            mlp_write = mlp_write.clone()
            mlp_write[:, ANSWER_POSITION] = torch.where(selected[:, None], replacement, mlp_write[:, ANSWER_POSITION])
        hidden = after_attn + mlp_write
    return model.lm_head(model.final_norm(hidden))[:, -1, CANDIDATE_SLICE].float().view(mask_count, batch, 4)


def response_strength(patched: torch.Tensor, baseline: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    patched_c = patched - patched.mean(dim=-1, keepdim=True)
    baseline_c = baseline - baseline.mean(dim=-1, keepdim=True)
    donor_c = donor - donor.mean(dim=-1, keepdim=True)
    numerator = torch.linalg.vector_norm(patched_c - baseline_c, dim=-1)
    scale = 0.5 * (torch.linalg.vector_norm(baseline_c, dim=-1) + torch.linalg.vector_norm(donor_c, dim=-1)) + 1.0e-6
    return torch.clamp(numerator / scale, 0.0, 2.0) / 2.0


def measure_model(model: TinyCausalTransformer, task: str, architecture: str, rows: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], bool]:
    events = program_registry(ARCHITECTURES[architecture].layers)
    stage_events = {stage: [event for event in events if event["stage"] == stage] for stage in ("attn_write", "mlp_write")}
    event_index = {event["event_id"]: index for index, event in enumerate(events)}
    sums = {partition: np.zeros((len(events), len(ROLES), len(READOUTS)), dtype=np.float64) for partition in PARTITION_COUNTS}
    counts = {partition: 0 for partition in PARTITION_COUNTS}
    with torch.inference_mode():
        for partition in PARTITION_COUNTS:
            selected_rows = [row for row in rows if row["task"] == task and row["partition"] == partition]
            for start in range(0, len(selected_rows), 32):
                batch_rows = selected_rows[start : start + 32]
                batch = len(batch_rows)
                receiver_ids = torch.tensor([row["receiver_ids"] for row in batch_rows], device=device)
                donor_ids = torch.cat([torch.tensor([row["variants"][role]["ids"] for row in batch_rows], device=device) for role in ROLES], dim=0)
                receiver_repeat = receiver_ids.repeat(len(ROLES), 1)
                receiver_trace = micro.capture_micro(model, receiver_ids)
                donor_trace = micro.capture_micro(model, donor_ids)
                receiver_trace_repeat = repeat_trace(receiver_trace, len(ROLES))
                receiver_logits = model(receiver_ids)[:, -1, CANDIDATE_SLICE].float()
                donor_logits = model(donor_ids)[:, -1, CANDIDATE_SLICE].float().view(len(ROLES), batch, 4)
                receiver_answers = torch.tensor([row["receiver_answer"] for row in batch_rows], device=device)
                donor_answers = torch.stack([torch.tensor([row["variants"][role]["answer"] for row in batch_rows], device=device) for role in ROLES])
                for stage, stage_rows in stage_events.items():
                    masks = [event["mask"] for event in stage_rows]
                    forward = forward_masks_logits(model, receiver_repeat, donor_trace, masks, stage).view(len(masks), len(ROLES), batch, 4)
                    reverse = forward_masks_logits(model, donor_ids, receiver_trace_repeat, masks, stage).view(len(masks), len(ROLES), batch, 4)
                    for local_index, event in enumerate(stage_rows):
                        global_index = event_index[event["event_id"]]
                        for role_index in range(len(ROLES)):
                            f_logits, r_logits = forward[local_index, role_index], reverse[local_index, role_index]
                            d_answer = donor_answers[role_index]
                            desired = (f_logits.argmax(-1) == d_answer).float().sum().item()
                            reverse_desired = (r_logits.argmax(-1) == receiver_answers).float().sum().item()
                            f_switch = (f_logits.argmax(-1) != receiver_answers).float().sum().item()
                            r_switch = (r_logits.argmax(-1) != d_answer).float().sum().item()
                            f_strength = response_strength(f_logits, receiver_logits, donor_logits[role_index]).sum().item()
                            r_strength = response_strength(r_logits, donor_logits[role_index], receiver_logits).sum().item()
                            sums[partition][global_index, role_index] += [desired, reverse_desired, f_switch, r_switch, f_strength, r_strength]
                counts[partition] += batch
                del receiver_ids, donor_ids, receiver_trace, donor_trace, receiver_trace_repeat, receiver_logits, donor_logits
            sums[partition] /= float(counts[partition])
    full_event_id = f"attention_prefix.l{ARCHITECTURES[architecture].layers - 1}"
    full = sums["confirmation"][event_index[full_event_id]]
    active_by_role = {role: bool(next(row for row in rows if row["task"] == task)["variants"][role]["active"]) for role in ROLES}
    control_checks: dict[str, bool] = {}
    for role_index, role in enumerate(ROLES):
        if active_by_role[role]:
            control_checks[f"{role}.positive"] = bool(min(full[role_index, 0], full[role_index, 1]) >= THRESHOLDS["control_positive_min"])
        else:
            control_checks[f"{role}.null"] = bool(max(full[role_index, 2], full[role_index, 3]) <= THRESHOLDS["control_null_switch_max"])
    passed = bool(all(control_checks.values()))
    payload = {"events": events, "roles": list(ROLES), "readouts": list(READOUTS), "active_by_role": active_by_role, "responses": {partition: sums[partition].tolist() for partition in PARTITION_COUNTS}, "control_checks": control_checks, "controls_passed": passed}
    return payload, passed


def event_blocks(row: dict[str, Any], partition: str) -> dict[str, np.ndarray]:
    responses = np.asarray(row["response_tensor"]["responses"][partition], dtype=np.float64)
    blocks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for index, event in enumerate(row["response_tensor"]["events"]):
        blocks.setdefault(event["event_type"], []).append((int(event["layer"]), responses[index].reshape(-1)))
    return {kind: np.stack([value for _, value in sorted(values)]) for kind, values in sorted(blocks.items())}


def identity_score(left: dict[str, Any], right: dict[str, Any], partition: str) -> float:
    a, b = event_blocks(left, partition), event_blocks(right, partition)
    values_a = np.concatenate([camera.resample(a[kind], CAMERA_GRID) for kind in sorted(a)], axis=0)
    values_b = np.concatenate([camera.resample(b[kind], CAMERA_GRID) for kind in sorted(b)], axis=0)
    return camera.cosine(values_a, values_b)


def layer_features(row: dict[str, Any], partition: str) -> np.ndarray:
    blocks = event_blocks(row, partition)
    return np.concatenate([blocks[kind] for kind in sorted(blocks)], axis=1)


def monotone_score(left: dict[str, Any], right: dict[str, Any], partition: str, path: list[tuple[int, int]] | None = None) -> tuple[float, list[tuple[int, int]]]:
    if path is None: path = camera.dtw_path(layer_features(left, "discovery"), layer_features(right, "discovery"))
    a, b = layer_features(left, partition), layer_features(right, partition)
    return camera.cosine(np.stack([a[i] for i, _ in path]), np.stack([b[j] for _, j in path])), path


def spectrum_signature(row: dict[str, Any], partition: str) -> np.ndarray:
    blocks = event_blocks(row, partition)
    stacked = np.concatenate([blocks[kind] for kind in sorted(blocks)], axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    singular = np.pad(singular[:24], (0, max(0, 24 - len(singular))))
    moments = np.concatenate([np.r_[block.mean(axis=0), block.std(axis=0)] for block in blocks.values()])
    return np.concatenate([stacked.mean(axis=0), stacked.std(axis=0), singular, moments])


def graph_signature(row: dict[str, Any], partition: str) -> np.ndarray:
    events = row["response_tensor"]["events"]
    responses = np.asarray(row["response_tensor"]["responses"][partition], dtype=np.float64).reshape(len(events), -1)
    adjacency = np.zeros((len(events), len(events)), dtype=np.float64)
    for i, left in enumerate(events):
        for j in range(i + 1, len(events)):
            right = events[j]
            linked = (left["event_type"] == right["event_type"] and abs(left["layer"] - right["layer"]) == 1) or (left["layer"] == right["layer"] and left["event_type"] != right["event_type"])
            if linked: adjacency[i, j] = adjacency[j, i] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    graph_eigen = np.linalg.eigvalsh(laplacian)
    graph_eigen = np.interp(np.linspace(0, len(graph_eigen) - 1, 24), np.arange(len(graph_eigen)), graph_eigen)
    response_eigen = np.linalg.eigvalsh(responses @ responses.T)
    response_eigen = np.interp(np.linspace(0, len(response_eigen) - 1, 24), np.arange(len(response_eigen)), response_eigen)
    degree = np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
    smooth = adjacency @ responses / degree
    typed = np.concatenate([block.mean(axis=0) for block in event_blocks(row, partition).values()])
    return np.concatenate([graph_eigen, response_eigen, responses.mean(axis=0), responses.std(axis=0), smooth.mean(axis=0), typed])


def pair_score(name: str, left: dict[str, Any], right: dict[str, Any], partition: str, path: list[tuple[int, int]] | None = None) -> tuple[float, list[tuple[int, int]] | None]:
    if name == "identity_coordinate": return identity_score(left, right, partition), None
    if name == "monotone_depth_warp": return monotone_score(left, right, partition, path)
    if name == "response_spectrum": return camera.cosine(spectrum_signature(left, partition), spectrum_signature(right, partition)), None
    if name == "gated_causal_graph": return camera.cosine(graph_signature(left, partition), graph_signature(right, partition)), None
    raise ValueError(name)


def build_pair_ledger(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [row for row in models if row.get("controls_passed")]
    ledger: list[dict[str, Any]] = []
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            item = {"pair_id": f'{left["model_key"]}__{right["model_key"]}', "left": left["model_key"], "right": right["model_key"], "left_task": left["task"], "right_task": right["task"], "left_seed_index": left["seed_index"], "right_seed_index": right["seed_index"], "same_task": int(left["task"] == right["task"]), "selection": left["task"] in DISCOVERY_TASKS and right["task"] in DISCOVERY_TASKS and left["seed_index"] < 2 and right["seed_index"] < 2, "unseen_task": HELDOUT_TASK in {left["task"], right["task"]}, "unseen_seed": 2 in {left["seed_index"], right["seed_index"]}, "joint": HELDOUT_TASK in {left["task"], right["task"]} and 2 in {left["seed_index"], right["seed_index"]}, "scores": {}}
            for name in CAMERAS:
                discovery, path = pair_score(name, left, right, "discovery")
                confirmation, _ = pair_score(name, left, right, "confirmation", path)
                item["scores"][name] = {"discovery": discovery, "confirmation": confirmation, "path": path}
            ledger.append(item)
    return ledger


def panel_metrics(rows: list[dict[str, Any]], camera_name: str, threshold: float) -> dict[str, Any]:
    if not rows or len({row["same_task"] for row in rows}) < 2: return {"eligible": False, "count": len(rows), "balanced_accuracy": 0.0, "auc": 0.0, "positive_mean": 0.0, "negative_mean": 0.0, "advantage": 0.0}
    labels = np.asarray([row["same_task"] for row in rows], dtype=int)
    scores = np.asarray([row["scores"][camera_name]["confirmation"] for row in rows])
    positive, negative = float(scores[labels == 1].mean()), float(scores[labels == 0].mean())
    return {"eligible": True, "count": len(rows), "positive_count": int((labels == 1).sum()), "negative_count": int((labels == 0).sum()), "balanced_accuracy": camera.balanced_accuracy(labels, (scores >= threshold).astype(int)), "auc": camera.auc(labels, scores), "positive_mean": positive, "negative_mean": negative, "advantage": positive - negative}


def analyze_results(qualification: list[dict[str, Any]], models: list[dict[str, Any]], ledger: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if ledger is None: ledger = build_pair_ledger(models)
    selection = [row for row in ledger if row["selection"]]
    selection_results: dict[str, Any] = {}
    for name in CAMERAS:
        if not selection or len({row["same_task"] for row in selection}) < 2:
            selection_results[name] = {"threshold": 1.0, "balanced_accuracy": 0.0, "auc": 0.0}
            continue
        labels = np.asarray([row["same_task"] for row in selection], dtype=int)
        scores = np.asarray([row["scores"][name]["discovery"] for row in selection])
        bacc, threshold = camera.choose_threshold(labels, scores)
        selection_results[name] = {"threshold": threshold, "balanced_accuracy": bacc, "auc": camera.auc(labels, scores)}
    priority = {name: -index for index, name in enumerate(CAMERAS)}
    selected = max(CAMERAS, key=lambda name: (selection_results[name]["balanced_accuracy"], selection_results[name]["auc"], priority[name]))
    executable = max(EXECUTABLE_CAMERAS, key=lambda name: (selection_results[name]["balanced_accuracy"], selection_results[name]["auc"], priority[name]))
    panels = {panel: [row for row in ledger if row[panel]] for panel in ("unseen_task", "unseen_seed", "joint")}
    evaluations = {name: {panel: panel_metrics(rows, name, selection_results[name]["threshold"]) for panel, rows in panels.items()} for name in CAMERAS}
    qualified = [row for row in qualification if row["behavior_passed"]]
    controls = [row for row in models if row.get("controls_passed")]
    per_cell_behavior = {f"{task}.{architecture}": sum(row["behavior_passed"] and row["task"] == task and row["architecture"] == architecture for row in qualification) for task in TASKS for architecture in ARCHITECTURES}
    per_cell_control = {f"{task}.{architecture}": sum(row.get("controls_passed") and row["task"] == task and row["architecture"] == architecture for row in models) for task in TASKS for architecture in ARCHITECTURES}
    unseen_task = evaluations[selected]["unseen_task"]
    unseen_seed = evaluations[selected]["unseen_seed"]
    joint = evaluations[selected]["joint"]
    mapping_joint = evaluations[executable]["joint"]
    gates = {
        "behavior_breadth": len(qualified) >= THRESHOLDS["behavior_models_min"] and min(per_cell_behavior.values()) >= THRESHOLDS["behavior_per_task_depth_min"],
        "control_breadth": len(controls) >= THRESHOLDS["control_models_min"] and min(per_cell_control.values()) >= THRESHOLDS["control_per_task_depth_min"],
        "unseen_task_accuracy": unseen_task["eligible"] and unseen_task["balanced_accuracy"] >= THRESHOLDS["unseen_task_balanced_accuracy_min"],
        "unseen_task_auc": unseen_task["eligible"] and unseen_task["auc"] >= THRESHOLDS["unseen_task_auc_min"],
        "unseen_seed_accuracy": unseen_seed["eligible"] and unseen_seed["balanced_accuracy"] >= THRESHOLDS["unseen_seed_balanced_accuracy_min"],
        "joint_accuracy": joint["eligible"] and joint["balanced_accuracy"] >= THRESHOLDS["joint_balanced_accuracy_min"],
        "executable_mapping_positive": mapping_joint["eligible"] and mapping_joint["positive_mean"] >= THRESHOLDS["mapping_positive_cosine_min"],
        "executable_mapping_specific": mapping_joint["eligible"] and mapping_joint["advantage"] >= THRESHOLDS["mapping_task_advantage_min"],
        "false_authorizations": True,
    }
    passed = all(gates.values())
    final = {"phase": PHASE, "contract_id": CONTRACT_ID, "qualification_models": len(qualification), "behavior_qualified_models": len(qualified), "control_passed_models": len(controls), "per_cell_behavior": per_cell_behavior, "per_cell_control": per_cell_control, "pair_count": len(ledger), "selection_pair_count": len(selection), "selection_results": selection_results, "selected_camera": selected, "selected_executable_camera": executable, "evaluations": evaluations, "gates": gates, "passed": passed, "decision": "cross_task_functional_response_isomorphism_confirmed" if passed else "cross_task_functional_response_isomorphism_not_confirmed", "rescue_authorized": passed, "pretrained_authorized": False, "synthetic_local_isomorphism_search_closed": not passed}
    return final, ledger


def run(device_name: str) -> None:
    protocol, rows = verify_protocol()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda": raise RuntimeError("formal run requires CUDA")
    qualification_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in TASKS:
        for architecture, config in ARCHITECTURES.items():
            for seed_index in range(SEEDS_PER_CELL):
                key = f"{task}.{architecture}.s{seed_index}"
                seed = MODEL_SEEDS[key]
                model, training = train_model(task, config, seed, device)
                behavior_passed = training["qualification_accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
                qualification = {"model_key": key, "task": task, "architecture": architecture, "seed_index": seed_index, "seed": seed, "training": training, "behavior_passed": behavior_passed, "qualification_status": "qualified" if behavior_passed else "behavior_rejected"}
                qualification_rows.append(qualification)
                if behavior_passed:
                    response_tensor, controls_passed = measure_model(model, task, architecture, rows, device)
                    model_rows.append({"model_key": key, "task": task, "architecture": architecture, "depth": config.layers, "seed_index": seed_index, "seed": seed, "controls_passed": controls_passed, "measurement_status": "measured", "claim_status": "abstained", "response_tensor": response_tensor})
                print(canonical_json({"model": key, "accuracy": training["qualification_accuracy"], "behavior": behavior_passed, "controls": model_rows[-1]["controls_passed"] if behavior_passed else None, "steps": training["steps"]}), flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    write_jsonl(QUALIFICATION, qualification_rows)
    write_jsonl(MODELS, model_rows)
    ledger = build_pair_ledger(model_rows)
    write_jsonl(PAIRS, ledger)
    elapsed = time.perf_counter() - started
    summary = {"phase": PHASE, "created_at_utc": utc_now(), "attempted_models": len(qualification_rows), "measured_models": len(model_rows), "pair_count": len(ledger), "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0, "device": torch.cuda.get_device_name(0), "protocol_digest": protocol["protocol_digest"], "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS), "run_digest": digest([row["model_key"] for row in qualification_rows]), "pretrained_model_loaded": False}
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"phase": PHASE, "complete": True, "created_at_utc": utc_now(), "run_digest": summary["run_digest"]})


def analyze() -> None:
    verify_protocol()
    qualification, models, ledger = read_jsonl(QUALIFICATION), read_jsonl(MODELS), read_jsonl(PAIRS)
    final, recomputed_ledger = analyze_results(qualification, models, ledger)
    if digest(ledger) != digest(recomputed_ledger): raise RuntimeError("pair ledger drift")
    final.update({"created_at_utc": utc_now(), "protocol_hash": file_sha256(PROTOCOL), "qualification_hash": file_sha256(QUALIFICATION), "models_hash": file_sha256(MODELS), "pairs_hash": file_sha256(PAIRS)})
    final["final_digest"] = digest({key: value for key, value in final.items() if key not in {"created_at_utc", "final_digest"}})
    atomic_json(FINAL, final)
    print(canonical_json({"decision": final["decision"], "passed": final["passed"], "selected": final["selected_camera"], "executable": final["selected_executable_camera"], "behavior": final["behavior_qualified_models"], "controls": final["control_passed_models"]}))


def smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, config, seed = "cyclic", ModelConfig(layers=4, width=32, heads=4, mlp_width=64, max_length=LENGTH, vocab_size=VOCAB), 1274001
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device).eval()
    rows = []
    for partition, seed_offset in (("discovery", 1), ("confirmation", 2)):
        rng = np.random.default_rng(seed_offset)
        rows.extend(make_case(task, rng, index, partition) for index in range(16))
    payload, passed = measure_model(model, task, "shallow4", rows, device)
    inputs, labels = random_batch(task, 32, seed)
    logits = model(inputs.to(device))[:, -1, CANDIDATE_SLICE]
    print(canonical_json({"device": str(device), "shape": list(logits.shape), "labels": list(labels.shape), "programs": len(program_registry(4)), "response_shape": list(np.asarray(payload["responses"]["discovery"]).shape), "passed": passed}))


def probe() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development_seeds = {"cyclic": 1_274_990_011, "xor": 1_274_990_021, "context_lookup": 1_274_990_031}
    outputs = []
    for task in TASKS:
        model, training = train_model(task, ARCHITECTURES["shallow4"], development_seeds[task], device)
        rows = []
        for partition, offset in (("discovery", 41), ("confirmation", 43)):
            rng = np.random.default_rng(development_seeds[task] + offset)
            rows.extend(make_case(task, rng, index, partition) for index in range(128))
        response, controls = measure_model(model, task, "shallow4", rows, device)
        outputs.append({"task": task, "training": training, "controls": controls, "control_checks": response["control_checks"]})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(outputs, indent=2))


def run_auditor(mode: str) -> None:
    status = os.spawnv(os.P_WAIT, sys.executable, [sys.executable, str(AUDITOR), mode])
    if status: raise SystemExit(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "probe", "preregister", "preaudit", "run", "analyze", "audit"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mode == "smoke": smoke()
    elif args.mode == "probe": probe()
    elif args.mode == "preregister": preregister(args.force)
    elif args.mode == "preaudit": run_auditor("preaudit")
    elif args.mode == "run": run(args.device)
    elif args.mode == "analyze": analyze()
    else: run_auditor("final")


if __name__ == "__main__": main()
