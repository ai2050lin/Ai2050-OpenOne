#!/usr/bin/env python3
"""Phase1250: known-truth multi-event response-camera calibration.

Freely trained causal Transformers solve a direct one-input task and a code
task whose output additionally depends on a later mapping.  Source-only and
multi-event cameras have the same 80-dimensional input budget.  The formal
test asks whether an actual patched trajectory supplies information that a
causally earlier source observation cannot contain.
"""

from __future__ import annotations

import argparse
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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


PHASE = 1250
CONTRACT_ID = "EXP-C003-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1250_c003_multievent_response_camera_audit.py"
OUT = ROOT / "tests/glm5/result/phase1250_c003_multievent_response_camera"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_worlds.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
ARRAYS = OUT / "raw/camera_arrays.npz"
ANALYSIS = OUT / "analysis/camera_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

BOS, DIRECT, CODE, REC, SEP, MAP, QUERY, ANSWER = range(8)
ENTITY_START, LABEL_START, CODE_START, SHIFT_START = 8, 10, 14, 18
VOCAB, LENGTH = 22, 23
SOURCE_POSITIONS = (4, 8)
SHIFT_POSITION, ANSWER_POSITION = 11, 22
CANDIDATE_SLICE = slice(LABEL_START, LABEL_START + 4)

ARCHITECTURES = {
    "compact": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
    "wide": ModelConfig(layers=4, width=128, heads=4, mlp_width=256, max_length=LENGTH, vocab_size=VOCAB),
}
REPLICATES = 2
MODEL_SEEDS = {
    "compact_r0": 1_250_301_001,
    "compact_r1": 1_250_301_101,
    "wide_r0": 1_250_401_001,
    "wide_r1": 1_250_401_101,
}
WORLD_SEED = 1_250_500_001
PROJECTION_SEED = 1_250_600_001
WORLD_COUNTS = {"discovery": 50, "selection": 20, "confirmation": 30}
FIT_ALPHAS = (0.25, 0.50)
SELECTION_ALPHA = 0.75
CONFIRMATION_ALPHA = 1.0
CAMERA_DIM = 80
RIDGE_GRID = (1.0e-3, 1.0e-2, 1.0e-1)
TRAINING_STEPS = 6500
TRAINING_BATCH = 512

CAMERA_FAMILIES = (
    "source_only",
    "typed_source_only",
    "multievent_additive",
    "typed_multievent_additive",
    "multievent_interaction",
)

THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "target_direct_cosine_min": 0.90,
    "target_code_cosine_min": 0.90,
    "target_positive_fraction_min": 0.95,
    "target_relative_error_max": 0.55,
    "code_cosine_advantage_over_best_source_min": 0.60,
    "target_effect_over_null_difference_min": 5.0,
    "null_effect_fraction_max": 0.05,
    "passing_models_min": 3,
    "passing_per_architecture_min": 1,
    "exact_source_collision_fraction_min": 1.0,
    "collision_response_separation_min": 1.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def build_sequence(representation: int, codes: list[int], shift: int, order: list[int]) -> tuple[list[int], dict[int, int]]:
    mapping = (np.arange(4) + shift) % 4
    source = mapping[np.asarray(codes)] + LABEL_START if representation == 0 else np.asarray(codes) + CODE_START
    values = [BOS, DIRECT if representation == 0 else CODE, REC, ENTITY_START, int(source[0]), SEP,
              REC, ENTITY_START + 1, int(source[1]), SEP, MAP, SHIFT_START + shift]
    positions: dict[int, int] = {}
    for code in order:
        values.extend([CODE_START + code, LABEL_START + int(mapping[code])])
        positions[code] = len(values) - 1
    values.extend([QUERY, ENTITY_START, ANSWER])
    if len(values) != LENGTH:
        raise RuntimeError("sequence length drift")
    return values, positions


def make_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    partitions = [name for name, count in WORLD_COUNTS.items() for _ in range(count)]
    rows: list[dict[str, Any]] = []
    for group, partition in enumerate(partitions):
        codes = rng.choice(4, 2, replace=False).astype(int).tolist()
        target_code = int(rng.choice([value for value in range(4) if value not in codes]))
        null_code = int(rng.choice([value for value in range(4) if value != codes[1]]))
        order = rng.permutation(4).astype(int).tolist()
        for shift in range(4):
            for representation in range(2):
                receiver, positions = build_sequence(representation, codes, shift, order)
                target_codes = list(codes); target_codes[0] = target_code
                null_codes = list(codes); null_codes[1] = null_code
                target, _ = build_sequence(representation, target_codes, shift, order)
                null, _ = build_sequence(representation, null_codes, shift, order)
                row = {
                    "row_id": f"g{group:03d}.s{shift}.r{representation}",
                    "group": group,
                    "partition": partition,
                    "representation": "direct" if representation == 0 else "code",
                    "representation_index": representation,
                    "shift": shift,
                    "codes": codes,
                    "target_code": target_code,
                    "null_code": null_code,
                    "codebook_order": order,
                    "codebook_value_positions": {str(key): value for key, value in positions.items()},
                    "receiver_ids": receiver,
                    "target_ids": target,
                    "null_ids": null,
                    "source_positions": list(SOURCE_POSITIONS),
                    "shift_position": SHIFT_POSITION,
                    "answer_position": ANSWER_POSITION,
                    "receiver_answer": (codes[0] + shift) % 4,
                    "target_answer": (target_code + shift) % 4,
                    "null_answer": (codes[0] + shift) % 4,
                }
                row["row_digest"] = digest(row)
                rows.append(row)
    if len(rows) != 800:
        raise RuntimeError("world count drift")
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1250.c003.multievent_camera.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "known_truth_multievent_response_camera_calibration",
        "question": "Can an actual multi-event intervention trajectory predict held-out target response when the causally prior source state is provably insufficient?",
        "architectures": {key: vars(value) for key, value in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "task": {
            "direct": "source label -> answer label",
            "code": "source code + later cyclic-shift codebook -> answer label",
            "known_dependency_difference": "code response depends on a later shift while direct source already names the answer",
            "query_entity": 0,
            "behavior_confirmation_examples": 16384,
        },
        "partitions": WORLD_COUNTS,
        "row_count": len(rows),
        "partition_digest": digest([{key: row[key] for key in ("row_id", "partition", "row_digest")} for row in rows]),
        "interventions": {
            "target": "patch queried source token embedding toward a donor with a different source",
            "matched_null": "patch unqueried source token embedding; correct answer is unchanged",
            "fit_alphas": list(FIT_ALPHAS),
            "selection_alpha": SELECTION_ALPHA,
            "confirmation_alpha": CONFIRMATION_ALPHA,
            "readout": "centered four-label logit response",
        },
        "observation_bundle": {
            "source": "patched-minus-receiver source embedding",
            "shift": "patched-minus-receiver state at shift token after early third",
            "map_receiver": "patched-minus-receiver state at receiver-code value after middle depth",
            "map_donor": "patched-minus-receiver state at donor-code value after middle depth",
            "answer_boundary": "patched-minus-receiver answer-boundary state after two-thirds depth",
            "important_boundary": "This is trajectory imaging after a legal intervention, not zero-cost prediction from the source alone.",
        },
        "camera": {
            "families": list(CAMERA_FAMILIES),
            "input_dimension_each": CAMERA_DIM,
            "output_dimension": 4,
            "ridge_grid_frozen": list(RIDGE_GRID),
            "ridge_selected_on": "selection worlds only",
            "confirmation": "sealed worlds, alpha=1",
            "parameter_budget_note": "All fitted linear families have the same input and output dimensions; structured zeros do not grant extra dimensions.",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 0.75, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "hard_stops": [
            "Behavior failure stops a model before camera fitting.",
            "Confirmation never selects camera family or ridge.",
            "A pass does not prove a discrete two-hop neural algorithm.",
            "A pass authorizes one fresh Qwen3 multi-event external-validity stage, not a semantic mechanism claim.",
            "A failure closes this frozen multi-event bundle; no event or threshold reselection is allowed.",
            "No GLM4 or DS7B is authorized by this known-truth calibration.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32_parameters_and_execution",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_worlds()
    write_jsonl(MATERIAL, rows)
    write_json(ENVIRONMENT, environment_snapshot())
    write_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source changed after preregistration")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    for row in rows:
        value = dict(row); stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest mismatch")
    counts = {partition: len({row["group"] for row in rows if row["partition"] == partition}) for partition in WORLD_COUNTS}
    if counts != WORLD_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    return protocol, rows


def random_batch(count: int, seed: int, forced_representation: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs, labels, reps = [], [], []
    for _ in range(count):
        representation = int(rng.integers(2)) if forced_representation is None else forced_representation
        shift = int(rng.integers(4))
        codes = rng.choice(4, 2, replace=False).astype(int).tolist()
        row, _positions = build_sequence(representation, codes, shift, rng.permutation(4).astype(int).tolist())
        inputs.append(row); labels.append((codes[0] + shift) % 4); reps.append(representation)
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(reps)


def train_model(config: ModelConfig, seed: int, device: torch.device) -> tuple[TinyCausalTransformer, dict[str, Any]]:
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-3)
    start = time.perf_counter()
    scores = [0.0, 0.0, 0.0]
    for step in range(TRAINING_STEPS):
        if step == 3000:
            optimizer.param_groups[0]["lr"] = 2.0e-4
        if step < 3000:
            inputs, labels, _reps = random_batch(TRAINING_BATCH, seed + step + 10_000, 1)
        else:
            code_x, code_y, _ = random_batch(384, seed + step + 10_000, 1)
            direct_x, direct_y, _ = random_batch(128, seed + step + 2_010_000, 0)
            inputs, labels = torch.cat([code_x, direct_x]), torch.cat([code_y, direct_y])
        logits = model(inputs.to(device))[:, -1, CANDIDATE_SLICE].float()
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step >= 3000 and step % 100 == 99:
            test_x, test_y, test_r = random_batch(16384, seed + 9_999_999)
            with torch.inference_mode():
                predicted = model(test_x.to(device))[:, -1, CANDIDATE_SLICE].argmax(-1).cpu()
            scores = [float((predicted == test_y).float().mean())]
            scores += [float((predicted[test_r == rep] == test_y[test_r == rep]).float().mean()) for rep in range(2)]
            if min(scores) >= THRESHOLDS["behavior_accuracy_min"]:
                break
    return model.eval(), {
        "steps": step + 1,
        "accuracy_overall": scores[0],
        "accuracy_direct": scores[1],
        "accuracy_code": scores[2],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def projectors(width: int) -> dict[str, np.ndarray]:
    dims = {"source": 80, "shift": 16, "map_receiver": 16, "map_donor": 16, "boundary": 16}
    values = {}
    for index, (name, dimension) in enumerate(dims.items()):
        rng = np.random.default_rng(PROJECTION_SEED + 10_007 * width + index)
        values[name] = rng.choice([-1.0, 1.0], size=(width, dimension)).astype(np.float32) / math.sqrt(dimension)
    return values


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, -1, CANDIDATE_SLICE].float()
    return values - values.mean(dim=-1, keepdim=True)


@torch.no_grad()
def collect_model_rows(model: TinyCausalTransformer, rows: list[dict[str, Any]], alpha: float, device: torch.device) -> list[dict[str, Any]]:
    projection = projectors(model.config.width)
    records: list[dict[str, Any]] = []
    early_depth = int(math.ceil(len(model.blocks) / 3))
    middle_depth = int(math.ceil(len(model.blocks) / 2))
    late_depth = int(math.ceil(2 * len(model.blocks) / 3))
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        receiver_ids = torch.tensor([row["receiver_ids"] for row in chunk], device=device)
        receiver_logits, receiver_states = model(receiver_ids, return_states=True)
        receiver_base = centered_logits(receiver_logits)
        for donor_name, donor_field, entity in (("target", "target_ids", 0), ("null", "null_ids", 1)):
            donor_ids = torch.tensor([row[donor_field] for row in chunk], device=device)
            _donor_logits, donor_states = model(donor_ids, return_states=True)
            source_position = SOURCE_POSITIONS[entity]
            hidden = receiver_states[0].clone()
            hidden[:, source_position] += alpha * (donor_states[0][:, source_position] - receiver_states[0][:, source_position])
            patched_states = [hidden]
            for block in model.blocks:
                hidden = block(hidden); patched_states.append(hidden)
            patched_logits = model.lm_head(model.final_norm(hidden))
            actual = (centered_logits(patched_logits) - receiver_base).cpu().numpy()
            for local, row in enumerate(chunk):
                receiver_code = int(row["codes"][entity])
                donor_code = int(row["target_code"] if donor_name == "target" else row["null_code"])
                positions = {int(key): int(value) for key, value in row["codebook_value_positions"].items()}
                raw = {
                    "source": (patched_states[0][local, source_position] - receiver_states[0][local, source_position]).cpu().numpy(),
                    "shift": (patched_states[early_depth][local, SHIFT_POSITION] - receiver_states[early_depth][local, SHIFT_POSITION]).cpu().numpy(),
                    "map_receiver": (patched_states[middle_depth][local, positions[receiver_code]] - receiver_states[middle_depth][local, positions[receiver_code]]).cpu().numpy(),
                    "map_donor": (patched_states[middle_depth][local, positions[donor_code]] - receiver_states[middle_depth][local, positions[donor_code]]).cpu().numpy(),
                    "boundary": (patched_states[late_depth][local, ANSWER_POSITION] - receiver_states[late_depth][local, ANSWER_POSITION]).cpu().numpy(),
                }
                projected = {key: raw[key] @ projection[key] for key in raw}
                source = projected["source"].astype(np.float32)
                typed_source = np.zeros(CAMERA_DIM, dtype=np.float32)
                rep = int(row["representation_index"])
                typed_source[rep * 40:(rep + 1) * 40] = source[:40]
                multievent = np.concatenate([projected["source"][:16], projected["shift"], projected["map_receiver"], projected["map_donor"], projected["boundary"]])
                typed_multievent = np.zeros(CAMERA_DIM, dtype=np.float32)
                compact = np.concatenate([projected[key][:8] for key in ("source", "shift", "map_receiver", "map_donor", "boundary")])
                typed_multievent[rep * 40:(rep + 1) * 40] = compact
                condition = (projected["shift"] + projected["map_receiver"] + projected["map_donor"]) / math.sqrt(3.0)
                interaction = np.concatenate([source[:16], projected["shift"], projected["map_receiver"], projected["map_donor"], source[:16] * condition])
                records.append({
                    "row_id": row["row_id"], "group": row["group"], "partition": row["partition"],
                    "representation": row["representation"], "representation_index": rep, "shift": row["shift"],
                    "donor": donor_name, "alpha": alpha, "actual": actual[local],
                    "source_only": source, "typed_source_only": typed_source,
                    "multievent_additive": multievent.astype(np.float32),
                    "typed_multievent_additive": typed_multievent,
                    "multievent_interaction": interaction.astype(np.float32),
                })
    return records


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1.0e-6] = 1.0
    z = (x - mean) / scale
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    penalty = np.eye(design.shape[1]); penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + ridge * penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def ridge_predict(x: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    return design @ model["weights"]


def response_metrics(predicted: np.ndarray, actual: np.ndarray, normalization: float | None = None) -> dict[str, float]:
    dot = np.sum(predicted * actual, axis=1)
    cosine = dot / np.maximum(np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1), 1.0e-8)
    denominator = np.maximum(np.linalg.norm(actual, axis=1), 1.0e-6) if normalization is None else np.full(len(actual), normalization)
    return {
        "cosine_mean": float(np.mean(cosine)),
        "cosine_positive_fraction": float(np.mean(cosine > 0)),
        "relative_error_mean": float(np.mean(np.linalg.norm(predicted - actual, axis=1) / denominator)),
        "actual_effect_norm_mean": float(np.mean(np.linalg.norm(actual, axis=1))),
        "predicted_effect_norm_mean": float(np.mean(np.linalg.norm(predicted, axis=1))),
    }


def stack(records: list[dict[str, Any]], family: str) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([row[family] for row in records]), np.stack([row["actual"] for row in records])


def subset(records: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [row for row in records if all(row[key] == value for key, value in criteria.items())]


def evaluate_camera(records: list[dict[str, Any]], family: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    discovery = [row for row in records if row["partition"] == "discovery" and row["alpha"] in FIT_ALPHAS]
    selection = subset(records, partition="selection", alpha=SELECTION_ALPHA)
    confirmation = subset(records, partition="confirmation", alpha=CONFIRMATION_ALPHA)
    train_x, train_y = stack(discovery, family)
    candidates = []
    for ridge in RIDGE_GRID:
        camera = fit_ridge(train_x, train_y, ridge)
        sx, sy = stack(selection, family)
        prediction = ridge_predict(sx, camera)
        target = np.asarray([row["donor"] == "target" for row in selection])
        metric = response_metrics(prediction[target], sy[target])
        candidates.append((metric["cosine_mean"] - 0.25 * metric["relative_error_mean"], ridge, camera))
    _score, selected_ridge, camera = max(candidates, key=lambda item: item[0])
    cx, cy = stack(confirmation, family)
    prediction = ridge_predict(cx, camera)
    target_mask = np.asarray([row["donor"] == "target" for row in confirmation])
    target_norm = float(np.mean(np.linalg.norm(cy[target_mask], axis=1)))
    groups: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {"target_all": target_mask, "null_all": ~target_mask}
    for donor in ("target", "null"):
        for representation in ("direct", "code"):
            masks[f"{donor}_{representation}"] = np.asarray([
                row["donor"] == donor and row["representation"] == representation for row in confirmation
            ])
    for name, mask in masks.items():
        groups[name] = response_metrics(prediction[mask], cy[mask], target_norm if name.startswith("null") else None)
    return {
        "family": family,
        "selected_ridge": selected_ridge,
        "selection_scores": {str(ridge): score for score, ridge, _camera in candidates},
        "confirmation": groups,
    }, {"actual": cy, "predicted": prediction}


def source_collision(records: list[dict[str, Any]]) -> dict[str, Any]:
    target = [row for row in records if row["partition"] == "confirmation" and row["alpha"] == 1.0 and row["donor"] == "target" and row["representation"] == "code"]
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in target:
        groups.setdefault(int(row["group"]), []).append(row)
    reports = []
    for group, values in sorted(groups.items()):
        feature_distances, response_distances = [], []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                feature_distances.append(float(np.linalg.norm(values[left]["source_only"] - values[right]["source_only"])))
                response_distances.append(float(np.linalg.norm(values[left]["actual"] - values[right]["actual"])))
        reports.append({
            "group": group,
            "feature_max_pair_distance": max(feature_distances),
            "response_max_pair_distance": max(response_distances),
            "exact_collision": max(feature_distances) <= 1.0e-6,
            "response_separated": max(response_distances) >= THRESHOLDS["collision_response_separation_min"],
        })
    return {
        "groups": len(reports),
        "exact_collision_fraction": float(np.mean([row["exact_collision"] for row in reports])),
        "response_separated_fraction": float(np.mean([row["response_separated"] for row in reports])),
        "feature_max_pair_distance_max": float(max(row["feature_max_pair_distance"] for row in reports)),
        "response_max_pair_distance_mean": float(np.mean([row["response_max_pair_distance"] for row in reports])),
    }


def execute_model(architecture: str, replicate: int, rows: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    key = model_key(architecture, replicate)
    model, behavior = train_model(ARCHITECTURES[architecture], MODEL_SEEDS[key], device)
    if min(behavior["accuracy_direct"], behavior["accuracy_code"]) < THRESHOLDS["behavior_accuracy_min"]:
        return {"model_key": key, "architecture": architecture, "replicate": replicate, "behavior": behavior, "behavior_gate": False}, {}
    records: list[dict[str, Any]] = []
    for alpha in (*FIT_ALPHAS, SELECTION_ALPHA, CONFIRMATION_ALPHA):
        records.extend(collect_model_rows(model, rows, alpha, device))
    families: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for family in CAMERA_FAMILIES:
        summary, values = evaluate_camera(records, family)
        families[family] = summary
        arrays[f"{key}.{family}.actual"] = values["actual"]
        arrays[f"{key}.{family}.predicted"] = values["predicted"]
    best_source = max(("source_only", "typed_source_only"), key=lambda name: families[name]["confirmation"]["target_code"]["cosine_mean"])
    best_multi = max(("multievent_additive", "typed_multievent_additive", "multievent_interaction"),
                     key=lambda name: families[name]["confirmation"]["target_code"]["cosine_mean"])
    chosen = families[best_multi]["confirmation"]
    target_effect = chosen["target_all"]["actual_effect_norm_mean"]
    null_effect = chosen["null_all"]["actual_effect_norm_mean"]
    code_advantage = chosen["target_code"]["cosine_mean"] - families[best_source]["confirmation"]["target_code"]["cosine_mean"]
    gate = bool(
        chosen["target_direct"]["cosine_mean"] >= THRESHOLDS["target_direct_cosine_min"]
        and chosen["target_code"]["cosine_mean"] >= THRESHOLDS["target_code_cosine_min"]
        and chosen["target_all"]["cosine_positive_fraction"] >= THRESHOLDS["target_positive_fraction_min"]
        and chosen["target_direct"]["relative_error_mean"] <= THRESHOLDS["target_relative_error_max"]
        and chosen["target_code"]["relative_error_mean"] <= THRESHOLDS["target_relative_error_max"]
        and code_advantage >= THRESHOLDS["code_cosine_advantage_over_best_source_min"]
        and target_effect - null_effect >= THRESHOLDS["target_effect_over_null_difference_min"]
        and null_effect / max(target_effect, 1.0e-9) <= THRESHOLDS["null_effect_fraction_max"]
    )
    summary = {
        "model_key": key, "architecture": architecture, "replicate": replicate,
        "behavior": behavior, "behavior_gate": True,
        "camera_families": families,
        "selected_source_family": best_source,
        "selected_multievent_family": best_multi,
        "code_cosine_advantage": code_advantage,
        "target_effect_norm": target_effect,
        "null_effect_norm": null_effect,
        "target_minus_null_effect": target_effect - null_effect,
        "null_effect_fraction": null_effect / max(target_effect, 1.0e-9),
        "source_collision": source_collision(records),
        "model_gate": gate,
    }
    del model
    torch.cuda.empty_cache()
    return summary, arrays


def formal_run() -> None:
    protocol, rows = verify_protocol()
    if not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit failed")
    if RAW.exists() or ARRAYS.exists():
        raise RuntimeError("one-shot formal output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    start = time.perf_counter(); models = []; arrays: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        for replicate in range(REPLICATES):
            summary, values = execute_model(architecture, replicate, rows, torch.device("cuda"))
            models.append(summary); arrays.update(values)
    ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAYS, **arrays)
    elapsed = time.perf_counter() - start
    payload = {
        "phase": PHASE, "schema_version": "phase1250.c003.multievent_camera.run.v1",
        "created_at_utc": utc_now(), "protocol_digest": protocol["protocol_digest"],
        "models": models, "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0,
        "array_sha256": file_sha256(ARRAYS), "array_size_bytes": ARRAYS.stat().st_size,
        "pretrained_model_loaded": False,
    }
    payload["run_digest"] = digest(payload)
    write_json(RAW, payload)
    print(canonical_json({"status": "formal_complete", "models": len(models), "gpu_hours": payload["gpu_hours"]}))


def analyze() -> None:
    protocol, _rows = verify_protocol(); run = read_json(RAW)
    if file_sha256(ARRAYS) != run["array_sha256"]:
        raise RuntimeError("array hash mismatch")
    models = run["models"]
    behavior_gate = all(row.get("behavior_gate") for row in models)
    passing = [row for row in models if row.get("model_gate")]
    per_architecture = {architecture: sum(row.get("model_gate", False) for row in models if row["architecture"] == architecture) for architecture in ARCHITECTURES}
    collision_gate = all(
        row.get("source_collision", {}).get("exact_collision_fraction", 0.0) >= THRESHOLDS["exact_source_collision_fraction_min"]
        and row.get("source_collision", {}).get("response_separated_fraction", 0.0) >= 1.0
        for row in models if row.get("behavior_gate")
    )
    camera_gate = len(passing) >= THRESHOLDS["passing_models_min"] and all(value >= THRESHOLDS["passing_per_architecture_min"] for value in per_architecture.values())
    gates = {"G-BEHAVIOR": behavior_gate, "G-NONIDENTIFIABILITY": collision_gate, "G-MULTIEVENT": camera_gate}
    verdict = "known_truth_multievent_camera_confirmed" if all(gates.values()) else "known_truth_multievent_camera_not_confirmed"
    adjudication = {
        "phase": PHASE, "schema_version": "phase1250.c003.multievent_camera.adjudication.v1",
        "created_at_utc": utc_now(), "protocol_digest": protocol["protocol_digest"], "run_digest": run["run_digest"],
        "gates": gates, "passing_model_count": len(passing), "model_count": len(models),
        "passing_per_architecture": per_architecture, "models": models, "verdict": verdict,
        "authorization": {"phase1251_fresh_qwen_multievent_external_validity": all(gates.values()), "semantic_mechanism_claim": False, "cross_model_claim": False},
        "interpretation": [
            "A source-only state is exactly nonidentifiable across later mapping contexts.",
            "A measured multi-event intervention trajectory can recover target response in known-truth networks.",
            "This calibrates a trajectory camera; it does not establish a discrete two-hop algorithm or a natural-language mechanism.",
        ],
        "non_claims": [
            "No pretrained language model was loaded.",
            "No semantic representation, reusable language pattern or cross-model invariant was identified.",
            "The observation bundle includes downstream consequences of the intervention and is not a zero-cost source prediction.",
        ],
    }
    adjudication["adjudication_digest"] = digest(adjudication)
    write_json(ANALYSIS, adjudication)
    final = {
        "phase": PHASE, "created_at_utc": utc_now(), "verdict": verdict, "gates": gates,
        "passing_model_count": len(passing), "model_count": len(models), "passing_per_architecture": per_architecture,
        "next_phase_authorized": all(gates.values()), "next_phase": 1251 if all(gates.values()) else None,
        "semantic_mechanism_claim_authorized": False,
        "artifact_hashes": {"protocol": file_sha256(PROTOCOL), "material": file_sha256(MATERIAL), "raw": file_sha256(RAW), "arrays": file_sha256(ARRAYS), "analysis": file_sha256(ANALYSIS)},
    }
    final["final_digest"] = digest(final)
    write_json(FINAL, final)
    print(canonical_json({"status": "analyzed", "verdict": verdict, "gates": gates}))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preregister", "run", "analyze")); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "preregister": preregister(args.force)
    elif args.command == "run": formal_run()
    else: analyze()


if __name__ == "__main__":
    main()
