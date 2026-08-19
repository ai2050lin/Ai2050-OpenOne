#!/usr/bin/env python3
"""Phase1251: known-truth causal-slice object competition.

The experiment separates three questions that Phase1250 mixed together:
whether explicit later conditions permit ex-ante prediction, whether one late
post-context state is sufficient, and whether a measured multi-event response
contains non-redundant information.  Every fitted camera has an 80-dimensional
input and is selected without reading the sealed confirmation partition.
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


PHASE = 1251
CONTRACT_ID = "EXP-C004-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition_audit.py"
MODEL_DEPENDENCY = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
OUT = ROOT / "tests/glm5/result/phase1251_c004_causal_slice_competition"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_worlds.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
ARRAYS = OUT / "raw/camera_arrays.npz"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/causal_slice_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

BOS, DIRECT, CODE, REC, SEP, MAP, QUERY, ANSWER = range(8)
ENTITY_START, LABEL_START, CODE_START, SHIFT_START = 8, 10, 14, 18
VOCAB, LENGTH = 22, 23
SOURCE_POSITIONS = (4, 8)
SHIFT_POSITION, ANSWER_POSITION = 11, 22
CANDIDATE_SLICE = slice(LABEL_START, LABEL_START + 4)

ARCHITECTURES = {
    "shallow4": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
    "middle6": ModelConfig(layers=6, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
    "deep8": ModelConfig(layers=8, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB),
}
REPLICATES = 2
MODEL_SEEDS = {
    "shallow4_r0": 1_251_401_001,
    "shallow4_r1": 1_251_401_101,
    "middle6_r0": 1_251_601_001,
    "middle6_r1": 1_251_601_101,
    "deep8_r0": 1_251_801_001,
    "deep8_r1": 1_251_801_101,
}
WORLD_SEED = 1_251_900_001
PROJECTION_SEED = 1_251_910_001
BOOTSTRAP_SEED = 1_251_920_001
WORLD_COUNTS = {"discovery": 64, "selection": 32, "confirmation": 64}
FIT_ALPHAS = (0.25, 0.50)
SELECTION_ALPHA = 0.75
CONFIRMATION_ALPHA = 1.0
CAMERA_DIM = 80
RIDGE_GRID = (1.0e-3, 1.0e-2, 1.0e-1)
BOOTSTRAP_REPLICATES = 4000
TRAINING_STEPS = 6500
TRAINING_BATCH = 512
BEHAVIOR_EXAMPLES = 32768

SINGLE_FAMILIES = (
    "shift_early",
    "map_receiver_middle",
    "map_donor_middle",
    "boundary_early",
    "boundary_middle",
    "boundary_late",
    "boundary_final",
)
NONBOUNDARY_SINGLE_FAMILIES = (
    "shift_early",
    "map_receiver_middle",
    "map_donor_middle",
)
EXANTE_FAMILIES = (
    "condition_only",
    "source_condition",
    "precut_interaction",
)
LOO_FAMILIES = (
    "loo_source",
    "loo_shift",
    "loo_map_receiver",
    "loo_map_donor",
    "loo_boundary",
)
CAMERA_FAMILIES = (
    "source_only",
    *EXANTE_FAMILIES,
    *SINGLE_FAMILIES,
    "multievent_full",
    "multievent_no_boundary",
    *LOO_FAMILIES,
)

THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "camera_direct_cosine_min": 0.90,
    "camera_code_cosine_min": 0.90,
    "camera_positive_fraction_min": 0.95,
    "camera_relative_error_max": 0.55,
    "target_effect_over_null_difference_min": 5.0,
    "null_effect_fraction_max": 0.08,
    "equivalence_margin": 0.03,
    "distributed_advantage_min": 0.05,
    "no_boundary_code_cosine_min": 0.85,
    "breadth_models_min": 4,
    "breadth_per_depth_min": 1,
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


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
    values = [
        BOS,
        DIRECT if representation == 0 else CODE,
        REC,
        ENTITY_START,
        int(source[0]),
        SEP,
        REC,
        ENTITY_START + 1,
        int(source[1]),
        SEP,
        MAP,
        SHIFT_START + shift,
    ]
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
                target_codes = list(codes)
                target_codes[0] = target_code
                null_codes = list(codes)
                null_codes[1] = null_code
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
    expected = sum(WORLD_COUNTS.values()) * 8
    if len(rows) != expected:
        raise RuntimeError(f"world count drift: {len(rows)} != {expected}")
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1251.c004.causal_slice.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "known_truth_causal_slice_object_competition",
        "question": "Does a late single state, an explicit-condition predictor, or a distributed measured trajectory best account for held-out intervention response?",
        "architectures": {key: vars(value) for key, value in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "task": {
            "direct": "source label -> answer label",
            "code": "source code + later cyclic-shift codebook -> answer label",
            "behavior_examples_per_model": BEHAVIOR_EXAMPLES,
            "known_limitation": "The frozen task uses one cyclic-shift family; this phase adjudicates observation objects, not mapping-family generalization.",
        },
        "partitions": WORLD_COUNTS,
        "row_count": len(rows),
        "partition_digest": digest([{key: row[key] for key in ("row_id", "partition", "row_digest")} for row in rows]),
        "interventions": {
            "target": "patch queried source embedding toward a different source code",
            "matched_null": "patch unqueried source embedding while preserving the queried answer",
            "fit_alphas": list(FIT_ALPHAS),
            "selection_alpha": SELECTION_ALPHA,
            "confirmation_alpha": CONFIRMATION_ALPHA,
            "readout": "centered four-label logit response",
        },
        "object_ledger": {
            "source_only": "causally prior diagnostic delta",
            "condition_only": "external task variables only; no hidden state",
            "source_condition": "source delta plus external later condition; ex-ante",
            "precut_interaction": "source delta plus unpatched answer-boundary cut states; ex-ante",
            "single_event": list(SINGLE_FAMILIES),
            "multievent_no_boundary": "four measured deltas with answer boundary excluded",
            "multievent_full": "five measured deltas including a late answer boundary",
            "leave_one_out": list(LOO_FAMILIES),
            "semantic_boundary": "Post-intervention cameras are diagnostic. Feature deletion tests observational non-redundancy, not physical causal necessity.",
        },
        "camera": {
            "families": list(CAMERA_FAMILIES),
            "input_dimension_each": CAMERA_DIM,
            "output_dimension": 4,
            "ridge_grid": list(RIDGE_GRID),
            "single_and_exante_family_selection": "selection target-code score only",
            "confirmation": "sealed groups at alpha=1",
            "bootstrap": {"unit": "world group", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "complexity_control": "All cameras expose 80 columns; effective ridge degrees of freedom are reported.",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 1.5, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDITOR),
            "model_dependency": file_sha256(MODEL_DEPENDENCY),
        },
        "hard_stops": [
            "Any behavior-unqualified model is retained as a failure and is not replaced.",
            "Confirmation cannot select a ridge, event, single-state family, ex-ante family, or threshold.",
            "Late-single non-inferiority rejects observational multi-event necessity; it does not prove a one-node causal mechanism.",
            "A multi-event predictive advantage is not a causal-path claim without physical blocking and rescue.",
            "No pretrained language model is authorized by this phase alone.",
            "Failure cannot be followed by extra seeds, relaxed thresholds, or a larger event bundle under this contract.",
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
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "groups": sum(WORLD_COUNTS.values())}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source changed after preregistration")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest mismatch")
    counts = {name: len({row["group"] for row in rows if row["partition"] == name}) for name in WORLD_COUNTS}
    if counts != WORLD_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    return protocol, rows


def random_batch(count: int, seed: int, forced_representation: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inputs, labels, representations = [], [], []
    for _ in range(count):
        representation = int(rng.integers(2)) if forced_representation is None else forced_representation
        shift = int(rng.integers(4))
        codes = rng.choice(4, 2, replace=False).astype(int).tolist()
        sequence, _positions = build_sequence(representation, codes, shift, rng.permutation(4).astype(int).tolist())
        inputs.append(sequence)
        labels.append((codes[0] + shift) % 4)
        representations.append(representation)
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(representations)


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
            inputs, labels, _ = random_batch(TRAINING_BATCH, seed + step + 10_000, 1)
        else:
            code_x, code_y, _ = random_batch(384, seed + step + 10_000, 1)
            direct_x, direct_y, _ = random_batch(128, seed + step + 2_010_000, 0)
            inputs, labels = torch.cat([code_x, direct_x]), torch.cat([code_y, direct_y])
        logits = model(inputs.to(device))[:, -1, CANDIDATE_SLICE].float()
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= 3000 and step % 100 == 99:
            test_x, test_y, test_r = random_batch(BEHAVIOR_EXAMPLES, seed + 9_999_999)
            with torch.inference_mode():
                predicted = model(test_x.to(device))[:, -1, CANDIDATE_SLICE].argmax(-1).cpu()
            scores = [float((predicted == test_y).float().mean())]
            scores.extend(float((predicted[test_r == rep] == test_y[test_r == rep]).float().mean()) for rep in range(2))
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


def projection(width: int, name: str, dimension: int) -> np.ndarray:
    name_seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(PROJECTION_SEED + 1009 * width + name_seed + 7919 * dimension)
    return rng.choice([-1.0, 1.0], size=(width, dimension)).astype(np.float32) / math.sqrt(dimension)


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, -1, CANDIDATE_SLICE].float()
    return values - values.mean(dim=-1, keepdim=True)


def external_condition(row: dict[str, Any], donor_name: str, entity: int, donor_code: int, alpha: float) -> np.ndarray:
    receiver_code = int(row["codes"][entity])
    receiver_answer = (receiver_code + int(row["shift"])) % 4
    donor_answer = (donor_code + int(row["shift"])) % 4
    representation = int(row["representation_index"])
    donor_index = 0 if donor_name == "target" else 1
    positions = {int(key): int(value) for key, value in row["codebook_value_positions"].items()}
    output = np.zeros(CAMERA_DIM, dtype=np.float32)
    output[receiver_answer * 4 + donor_answer] = 1.0
    output[16 + representation * 4 + donor_answer] = 1.0
    output[24 + donor_index * 4 + donor_answer] = 1.0
    output[32 + receiver_code * 4 + donor_code] = 1.0
    output[48 + int(row["shift"])] = 1.0
    output[52 + receiver_code] = 1.0
    output[56 + donor_code] = 1.0
    output[60 + receiver_answer] = 1.0
    output[64 + donor_answer] = 1.0
    output[68 + representation] = 1.0
    output[70 + donor_index] = 1.0
    output[72 + donor_answer] = alpha
    output[76] = alpha
    output[77] = alpha * alpha
    output[78] = positions[receiver_code] / float(LENGTH - 1)
    output[79] = positions[donor_code] / float(LENGTH - 1)
    return output


def concatenate_projected(raw: dict[str, np.ndarray], names: tuple[str, ...], dimension: int) -> np.ndarray:
    return np.concatenate([raw[name] @ projection(len(raw[name]), name, dimension) for name in names]).astype(np.float32)


@torch.no_grad()
def collect_model_rows(model: TinyCausalTransformer, rows: list[dict[str, Any]], alpha: float, device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    depths = len(model.blocks)
    early_depth = int(math.ceil(depths / 3))
    middle_depth = int(math.ceil(depths / 2))
    late_depth = int(math.ceil(2 * depths / 3))
    single_matrices = {
        name: projection(model.config.width, name, CAMERA_DIM)
        for name in ("source", "shift_early", "map_receiver_middle", "map_donor_middle", "boundary_early", "boundary_middle", "boundary_late", "boundary_final")
    }
    precut_matrices = {
        name: projection(model.config.width, f"precut_{name}", 16)
        for name in ("source", "boundary_early", "boundary_middle", "boundary_late")
    }
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
                hidden = block(hidden)
                patched_states.append(hidden)
            patched_logits = model.lm_head(model.final_norm(hidden))
            actual = (centered_logits(patched_logits) - receiver_base).cpu().numpy()
            for local, row in enumerate(chunk):
                receiver_code = int(row["codes"][entity])
                donor_code = int(row["target_code"] if donor_name == "target" else row["null_code"])
                positions = {int(key): int(value) for key, value in row["codebook_value_positions"].items()}
                raw = {
                    "source": (patched_states[0][local, source_position] - receiver_states[0][local, source_position]).cpu().numpy(),
                    "shift_early": (patched_states[early_depth][local, SHIFT_POSITION] - receiver_states[early_depth][local, SHIFT_POSITION]).cpu().numpy(),
                    "map_receiver_middle": (patched_states[middle_depth][local, positions[receiver_code]] - receiver_states[middle_depth][local, positions[receiver_code]]).cpu().numpy(),
                    "map_donor_middle": (patched_states[middle_depth][local, positions[donor_code]] - receiver_states[middle_depth][local, positions[donor_code]]).cpu().numpy(),
                    "boundary_early": (patched_states[early_depth][local, ANSWER_POSITION] - receiver_states[early_depth][local, ANSWER_POSITION]).cpu().numpy(),
                    "boundary_middle": (patched_states[middle_depth][local, ANSWER_POSITION] - receiver_states[middle_depth][local, ANSWER_POSITION]).cpu().numpy(),
                    "boundary_late": (patched_states[late_depth][local, ANSWER_POSITION] - receiver_states[late_depth][local, ANSWER_POSITION]).cpu().numpy(),
                    "boundary_final": (patched_states[-1][local, ANSWER_POSITION] - receiver_states[-1][local, ANSWER_POSITION]).cpu().numpy(),
                }
                condition = external_condition(row, donor_name, entity, donor_code, alpha)
                source_single = (raw["source"] @ single_matrices["source"]).astype(np.float32)
                source_condition = np.concatenate([source_single[:40], condition[:40]]).astype(np.float32)
                pre_source = raw["source"] @ precut_matrices["source"]
                pre_early = receiver_states[early_depth][local, ANSWER_POSITION].cpu().numpy() @ precut_matrices["boundary_early"]
                pre_middle = receiver_states[middle_depth][local, ANSWER_POSITION].cpu().numpy() @ precut_matrices["boundary_middle"]
                pre_late = receiver_states[late_depth][local, ANSWER_POSITION].cpu().numpy() @ precut_matrices["boundary_late"]
                pre_interaction = pre_source * np.tanh((pre_early + pre_middle + pre_late) / math.sqrt(3.0))
                precut = np.concatenate([pre_source, pre_early, pre_middle, pre_late, pre_interaction]).astype(np.float32)
                full_names = ("source", "shift_early", "map_receiver_middle", "map_donor_middle", "boundary_late")
                no_boundary_names = ("source", "shift_early", "map_receiver_middle", "map_donor_middle")
                features: dict[str, np.ndarray] = {
                    "source_only": source_single,
                    "condition_only": condition,
                    "source_condition": source_condition,
                    "precut_interaction": precut,
                    "multievent_full": concatenate_projected(raw, full_names, 16),
                    "multievent_no_boundary": concatenate_projected(raw, no_boundary_names, 20),
                }
                for family in SINGLE_FAMILIES:
                    features[family] = (raw[family] @ single_matrices[family]).astype(np.float32)
                loo_drop = {
                    "loo_source": "source",
                    "loo_shift": "shift_early",
                    "loo_map_receiver": "map_receiver_middle",
                    "loo_map_donor": "map_donor_middle",
                    "loo_boundary": "boundary_late",
                }
                for family, dropped in loo_drop.items():
                    features[family] = concatenate_projected(raw, tuple(name for name in full_names if name != dropped), 20)
                if set(features) != set(CAMERA_FAMILIES):
                    raise RuntimeError("camera feature family drift")
                records.append({
                    "row_id": row["row_id"],
                    "group": row["group"],
                    "partition": row["partition"],
                    "representation": row["representation"],
                    "donor": donor_name,
                    "alpha": alpha,
                    "actual": actual[local].astype(np.float32),
                    **features,
                })
    return records


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, np.ndarray | float]:
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1.0e-6] = 1.0
    z = (x - mean) / scale
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + ridge * penalty, design.T @ y)
    singular = np.linalg.svd(z, compute_uv=False)
    effective_df = float(1.0 + np.sum((singular * singular) / (singular * singular + ridge)))
    return {"mean": mean, "scale": scale, "weights": weights, "effective_df": effective_df}


def ridge_predict(x: np.ndarray, camera: dict[str, np.ndarray | float]) -> np.ndarray:
    z = (x - camera["mean"]) / camera["scale"]
    design = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    return design @ camera["weights"]


def row_cosine(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    dot = np.sum(predicted * actual, axis=1)
    return dot / np.maximum(np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1), 1.0e-8)


def response_metrics(predicted: np.ndarray, actual: np.ndarray, normalization: float | None = None) -> dict[str, float]:
    cosine = row_cosine(predicted, actual)
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


def confirmation_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if row["partition"] == "confirmation" and row["alpha"] == CONFIRMATION_ALPHA]


def evaluate_camera(records: list[dict[str, Any]], family: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    discovery = [row for row in records if row["partition"] == "discovery" and row["alpha"] in FIT_ALPHAS]
    selection = [row for row in records if row["partition"] == "selection" and row["alpha"] == SELECTION_ALPHA]
    confirmation = confirmation_records(records)
    train_x, train_y = stack(discovery, family)
    selection_mask = np.asarray([row["donor"] == "target" and row["representation"] == "code" for row in selection])
    candidates = []
    sx, sy = stack(selection, family)
    for ridge in RIDGE_GRID:
        camera = fit_ridge(train_x, train_y, ridge)
        prediction = ridge_predict(sx, camera)
        metric = response_metrics(prediction[selection_mask], sy[selection_mask])
        score = metric["cosine_mean"] - 0.25 * metric["relative_error_mean"]
        candidates.append((score, ridge, camera))
    selection_score, selected_ridge, camera = max(candidates, key=lambda item: item[0])
    cx, cy = stack(confirmation, family)
    prediction = ridge_predict(cx, camera)
    target_mask = np.asarray([row["donor"] == "target" for row in confirmation])
    target_norm = float(np.mean(np.linalg.norm(cy[target_mask], axis=1)))
    masks: dict[str, np.ndarray] = {"target_all": target_mask, "null_all": ~target_mask}
    for donor in ("target", "null"):
        for representation in ("direct", "code"):
            masks[f"{donor}_{representation}"] = np.asarray([
                row["donor"] == donor and row["representation"] == representation for row in confirmation
            ])
    groups = {
        name: response_metrics(prediction[mask], cy[mask], target_norm if name.startswith("null") else None)
        for name, mask in masks.items()
    }
    return {
        "family": family,
        "selected_ridge": selected_ridge,
        "selection_target_code_score": selection_score,
        "selection_scores": {str(ridge): score for score, ridge, _camera in candidates},
        "effective_degrees_of_freedom": float(camera["effective_df"]),
        "confirmation": groups,
    }, {"actual": cy.astype(np.float32), "predicted": prediction.astype(np.float32)}


def bootstrap_cosine_difference(
    records: list[dict[str, Any]],
    actual: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    mask = np.asarray([row["donor"] == "target" and row["representation"] == "code" for row in records])
    groups = np.asarray([int(row["group"]) for row in records])[mask]
    difference = row_cosine(left[mask], actual[mask]) - row_cosine(right[mask], actual[mask])
    unique = np.unique(groups)
    group_means = np.asarray([difference[groups == group].mean() for group in unique])
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(0, len(group_means), size=(BOOTSTRAP_REPLICATES, len(group_means)))
    bootstrap = group_means[draws].mean(axis=1)
    return {
        "left_minus_right_mean": float(group_means.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "independent_groups": int(len(group_means)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def breadth(flags: dict[str, bool], models: list[dict[str, Any]]) -> tuple[bool, dict[str, int]]:
    per_depth = {
        architecture: sum(flags.get(row["model_key"], False) for row in models if row["architecture"] == architecture)
        for architecture in ARCHITECTURES
    }
    passed = sum(flags.values()) >= THRESHOLDS["breadth_models_min"] and all(
        value >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values()
    )
    return bool(passed), per_depth


def execute_model(architecture: str, replicate: int, rows: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    key = model_key(architecture, replicate)
    model, behavior = train_model(ARCHITECTURES[architecture], MODEL_SEEDS[key], device)
    if min(behavior["accuracy_direct"], behavior["accuracy_code"]) < THRESHOLDS["behavior_accuracy_min"]:
        del model
        torch.cuda.empty_cache()
        return {
            "model_key": key,
            "architecture": architecture,
            "replicate": replicate,
            "behavior": behavior,
            "behavior_gate": False,
        }, {}
    records: list[dict[str, Any]] = []
    for alpha in (*FIT_ALPHAS, SELECTION_ALPHA, CONFIRMATION_ALPHA):
        records.extend(collect_model_rows(model, rows, alpha, device))
    families: dict[str, Any] = {}
    values: dict[str, dict[str, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    for family in CAMERA_FAMILIES:
        summary, family_values = evaluate_camera(records, family)
        families[family] = summary
        values[family] = family_values
        arrays[f"{key}.{family}.actual"] = family_values["actual"]
        arrays[f"{key}.{family}.predicted"] = family_values["predicted"]
    selected_single = max(SINGLE_FAMILIES, key=lambda name: families[name]["selection_target_code_score"])
    selected_nonboundary = max(NONBOUNDARY_SINGLE_FAMILIES, key=lambda name: families[name]["selection_target_code_score"])
    selected_exante = max(EXANTE_FAMILIES, key=lambda name: families[name]["selection_target_code_score"])
    confirmation = confirmation_records(records)
    actual = values["multievent_full"]["actual"]
    comparisons = {
        "full_minus_best_single": bootstrap_cosine_difference(
            confirmation, actual, values["multievent_full"]["predicted"], values[selected_single]["predicted"], 101 + replicate
        ),
        "no_boundary_minus_best_nonboundary_single": bootstrap_cosine_difference(
            confirmation, actual, values["multievent_no_boundary"]["predicted"], values[selected_nonboundary]["predicted"], 201 + replicate
        ),
        "full_minus_best_exante": bootstrap_cosine_difference(
            confirmation, actual, values["multievent_full"]["predicted"], values[selected_exante]["predicted"], 301 + replicate
        ),
    }
    full = families["multievent_full"]["confirmation"]
    single = families[selected_single]["confirmation"]
    no_boundary = families["multievent_no_boundary"]["confirmation"]
    exante = families[selected_exante]["confirmation"]
    target_effect = full["target_all"]["actual_effect_norm_mean"]
    null_effect = full["null_all"]["actual_effect_norm_mean"]
    intervention_gate = bool(
        target_effect - null_effect >= THRESHOLDS["target_effect_over_null_difference_min"]
        and null_effect / max(target_effect, 1.0e-9) <= THRESHOLDS["null_effect_fraction_max"]
    )
    full_quality = bool(
        full["target_direct"]["cosine_mean"] >= THRESHOLDS["camera_direct_cosine_min"]
        and full["target_code"]["cosine_mean"] >= THRESHOLDS["camera_code_cosine_min"]
        and full["target_all"]["cosine_positive_fraction"] >= THRESHOLDS["camera_positive_fraction_min"]
        and full["target_direct"]["relative_error_mean"] <= THRESHOLDS["camera_relative_error_max"]
        and full["target_code"]["relative_error_mean"] <= THRESHOLDS["camera_relative_error_max"]
        and intervention_gate
    )
    late_single_sufficient = bool(
        single["target_code"]["cosine_mean"] >= THRESHOLDS["camera_code_cosine_min"]
        and comparisons["full_minus_best_single"]["ci95_high"] <= THRESHOLDS["equivalence_margin"]
    )
    exante_sufficient = bool(
        exante["target_code"]["cosine_mean"] >= THRESHOLDS["camera_code_cosine_min"]
        and comparisons["full_minus_best_exante"]["ci95_high"] <= THRESHOLDS["equivalence_margin"]
    )
    distributed_advantage = bool(
        full_quality
        and comparisons["full_minus_best_single"]["ci95_low"] >= THRESHOLDS["distributed_advantage_min"]
        and no_boundary["target_code"]["cosine_mean"] >= THRESHOLDS["no_boundary_code_cosine_min"]
        and comparisons["no_boundary_minus_best_nonboundary_single"]["ci95_low"] >= THRESHOLDS["distributed_advantage_min"]
    )
    summary = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "behavior": behavior,
        "behavior_gate": True,
        "camera_families": families,
        "selected_single_family": selected_single,
        "selected_nonboundary_single_family": selected_nonboundary,
        "selected_exante_family": selected_exante,
        "comparisons": comparisons,
        "target_effect_norm": target_effect,
        "null_effect_norm": null_effect,
        "target_minus_null_effect": target_effect - null_effect,
        "null_effect_fraction": null_effect / max(target_effect, 1.0e-9),
        "intervention_gate": intervention_gate,
        "full_camera_quality": full_quality,
        "late_single_sufficient": late_single_sufficient,
        "exante_sufficient": exante_sufficient,
        "distributed_predictive_advantage": distributed_advantage,
    }
    del records, model
    torch.cuda.empty_cache()
    return summary, arrays


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def formal_run() -> None:
    protocol, rows = verify_protocol()
    if not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit failed")
    if any(path.exists() for path in (RAW, ARRAYS, COMPLETE)):
        raise RuntimeError("one-shot formal output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    start = time.perf_counter()
    models: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        for replicate in range(REPLICATES):
            summary, values = execute_model(architecture, replicate, rows, torch.device("cuda"))
            models.append(summary)
            arrays.update(values)
            print(canonical_json({"status": "model_complete", "model": summary["model_key"], "behavior_gate": summary["behavior_gate"]}), flush=True)
    atomic_npz(ARRAYS, arrays)
    elapsed = time.perf_counter() - start
    payload = {
        "phase": PHASE,
        "schema_version": "phase1251.c004.causal_slice.run.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "models": models,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "array_sha256": file_sha256(ARRAYS),
        "array_size_bytes": ARRAYS.stat().st_size,
        "pretrained_model_loaded": False,
    }
    payload["run_digest"] = digest(payload)
    atomic_json(RAW, payload)
    marker = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": payload["run_digest"],
        "raw_sha256": file_sha256(RAW),
        "arrays_sha256": file_sha256(ARRAYS),
        "status": "formal_run_complete",
    }
    marker["marker_digest"] = digest(marker)
    atomic_json(COMPLETE, marker)
    print(canonical_json({"status": "formal_complete", "models": len(models), "gpu_hours": payload["gpu_hours"]}))


def analyze() -> None:
    protocol, _rows = verify_protocol()
    run = read_json(RAW)
    marker = read_json(COMPLETE)
    if file_sha256(ARRAYS) != run["array_sha256"] or marker["arrays_sha256"] != run["array_sha256"]:
        raise RuntimeError("array hash mismatch")
    if marker["run_digest"] != run["run_digest"] or marker["raw_sha256"] != file_sha256(RAW):
        raise RuntimeError("completion marker mismatch")
    models = run["models"]
    behavior_flags = {row["model_key"]: bool(row.get("behavior_gate")) for row in models}
    full_flags = {row["model_key"]: bool(row.get("full_camera_quality")) for row in models}
    single_flags = {row["model_key"]: bool(row.get("late_single_sufficient")) for row in models}
    exante_flags = {row["model_key"]: bool(row.get("exante_sufficient")) for row in models}
    distributed_flags = {row["model_key"]: bool(row.get("distributed_predictive_advantage")) for row in models}
    behavior_gate, behavior_per_depth = breadth(behavior_flags, models)
    full_gate, full_per_depth = breadth(full_flags, models)
    single_gate, single_per_depth = breadth(single_flags, models)
    exante_gate, exante_per_depth = breadth(exante_flags, models)
    distributed_gate, distributed_per_depth = breadth(distributed_flags, models)
    if not behavior_gate:
        verdict = "behavior_qualification_failed"
    elif not full_gate:
        verdict = "trajectory_camera_not_reproduced"
    elif single_gate or exante_gate:
        verdict = "distributed_observation_necessity_rejected"
    elif distributed_gate:
        verdict = "distributed_multievent_predictive_advantage_confirmed_not_causal"
    else:
        verdict = "causal_slice_object_competition_unresolved"
    gates = {
        "G-BEHAVIOR-BREADTH": behavior_gate,
        "G-FULL-CAMERA-BREADTH": full_gate,
        "G-LATE-SINGLE-SUFFICIENCY": single_gate,
        "G-EXANTE-SUFFICIENCY": exante_gate,
        "G-DISTRIBUTED-PREDICTIVE-ADVANTAGE": distributed_gate,
    }
    adjudication = {
        "phase": PHASE,
        "schema_version": "phase1251.c004.causal_slice.adjudication.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": run["run_digest"],
        "verdict": verdict,
        "gates": gates,
        "per_depth": {
            "behavior": behavior_per_depth,
            "full_camera": full_per_depth,
            "late_single": single_per_depth,
            "exante": exante_per_depth,
            "distributed": distributed_per_depth,
        },
        "models": models,
        "authorization": {
            "physical_path_blocking_phase": bool(distributed_gate and not single_gate and not exante_gate),
            "pretrained_language_model_phase": False,
            "semantic_mechanism_claim": False,
            "causal_path_claim": False,
        },
        "interpretation_rules": [
            "Late-single or ex-ante non-inferiority rejects the claim that multiple measured post-intervention events are observationally necessary.",
            "A distributed predictive advantage authorizes physical path blocking and rescue, not a causal-path claim.",
            "Leave-one-out camera deletion measures predictive redundancy only.",
            "No result in this known-truth task is language-mechanism or cross-model evidence.",
        ],
    }
    adjudication["adjudication_digest"] = digest(adjudication)
    atomic_json(ANALYSIS, adjudication)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "gates": gates,
        "per_depth": adjudication["per_depth"],
        "authorization": adjudication["authorization"],
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "material": file_sha256(MATERIAL),
            "environment": file_sha256(ENVIRONMENT),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "arrays": file_sha256(ARRAYS),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"status": "analyzed", "verdict": verdict, "gates": gates}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        formal_run()
    else:
        analyze()


if __name__ == "__main__":
    main()
