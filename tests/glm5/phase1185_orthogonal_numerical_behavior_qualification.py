#!/usr/bin/env python3
"""Phase1185: orthogonal numerical and behavioral qualification.

Numerical support is sealed from every finite discovery checkpoint without
conditioning on holdout behavior. Behavioral qualification is computed in a
separate ledger. Only the final science-eligibility decision intersects the
two ledgers. No mechanism camera is evaluated in this phase.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402
import phase1183_gauge_exact_prospective_mechanism_closure as p1183  # noqa: E402
import phase1184_numerical_domain_qualification as p1184  # noqa: E402


PHASE = 1185
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1185_orthogonal_numerical_behavior_qualification_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1185_orthogonal_numerical_behavior_qualification"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
DOMAIN_PATH = OUT_ROOT / "analysis/numerical_support_seal.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

MODULUS = 61
WIDTH = 128
REPLICATES = 8
TRAIN_FRACTION = 0.50
ENDPOINT_STEP = 12_000
NATURAL_GAUGE_TRANSFORMS = 8
ENGINEERED_GAUGE_TRANSFORMS = 4
SAFETY_RADIUS_MULTIPLIER = 2.0
FP32_U = float(2.0 ** -24)
FP64_U = float(2.0 ** -53)
ENGINEERED_SCALES = (1e-4, 0.02, 0.2, 2.0)
ENGINEERED_STRUCTURES = ("full", "half_duplicate")

TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": ENDPOINT_STEP,
}

PROFILE_METRICS = (
    "parameter_l2",
    "parameter_max_abs",
    "hidden_weight_rms",
    "hidden_weight_max_abs",
    "output_weight_rms",
    "output_weight_max_abs",
    "hidden_activation_rms",
    "hidden_activation_max_abs",
    "squared_hidden_rms",
    "squared_hidden_max_abs",
    "logit_rms",
    "logit_max_abs",
    "margin_rms",
    "margin_max_abs",
    "hidden_stable_rank",
    "hidden_active_condition",
)

THRESHOLDS = {
    "numerical_discovery_system_count_min": 32,
    "numerical_confirmation_system_count_min": 32,
    "confirmation_safety_coverage_min": 0.90,
    "algebraic_feature_error_max": 1e-12,
    "fp64_scaled_error_max": 128.0 * FP64_U,
    "fp32_scaled_error_max": 32.0 * FP32_U,
    "fp64_absolute_floor": 256.0 * FP64_U,
    "fp32_absolute_floor": 128.0 * FP32_U,
    "fp64_relative_multiplier": 128.0 * FP64_U,
    "fp32_relative_multiplier": 32.0 * FP32_U,
    "decision_eligible_fraction_min": 0.95,
    "decision_agreement_min": 1.0,
    "margin_sign_eligible_fraction_min": 0.95,
    "margin_sign_agreement_min": 1.0,
    "natural_gauge_pass_fraction_min": 1.0,
    "engineered_gauge_pass_fraction_min": 1.0,
    "positive_feature_error_min": 1e-4,
    "positive_scaled_error_min": 1e-3,
    "positive_decision_agreement_max": 0.95,
    "positive_control_pass_fraction_min": 0.95,
    "behavior_train_accuracy_min": 0.95,
    "behavior_holdout_accuracy_min": 0.90,
    "behavior_qualified_per_task_min": 6,
    "behavior_passing_task_count_min": 3,
    "behavior_qualified_system_count_min": 24,
    "science_intersection_task_count_min": 3,
    "science_intersection_system_count_min": 24,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str
    family: str
    formula: str


TASK_SPECS = (
    TaskSpec("axis_affine_a", "discovery", "affine", "(19*a+23*b+7) mod 61"),
    TaskSpec("axis_square_sum_a", "discovery", "quadratic", "2*(a+3)^2+7*(b+11)^2+5 mod 61"),
    TaskSpec("axis_left_cube_a", "discovery", "cubic", "(a+8)^3+13*b+17 mod 61"),
    TaskSpec("axis_xor_a", "discovery", "bitwise", "(a xor b)+21 mod 61"),
    TaskSpec("axis_affine_b", "confirmation", "affine", "(29*a+31*b+11) mod 61"),
    TaskSpec("axis_square_sum_b", "confirmation", "quadratic", "5*(a+6)^2+3*(b+14)^2+19 mod 61"),
    TaskSpec("axis_left_cube_b", "confirmation", "cubic", "2*(a+5)^3+17*b+23 mod 61"),
    TaskSpec("axis_xor_b", "confirmation", "bitwise", "(a xor b)+33 mod 61"),
)


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "axis_affine_a": lambda a, b: (19 * a + 23 * b + 7) % p,
        "axis_square_sum_a": lambda a, b: (2 * ((a + 3) % p) ** 2 + 7 * ((b + 11) % p) ** 2 + 5) % p,
        "axis_left_cube_a": lambda a, b: (((a + 8) % p) ** 3 + 13 * b + 17) % p,
        "axis_xor_a": lambda a, b: (((a ^ b) % p) + 21) % p,
        "axis_affine_b": lambda a, b: (29 * a + 31 * b + 11) % p,
        "axis_square_sum_b": lambda a, b: (5 * ((a + 6) % p) ** 2 + 3 * ((b + 14) % p) ** 2 + 19) % p,
        "axis_left_cube_b": lambda a, b: (2 * ((a + 5) % p) ** 3 + 17 * b + 23) % p,
        "axis_xor_b": lambda a, b: (((a ^ b) % p) + 33) % p,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(task_index: int, replicate: int) -> int:
    return 11850000 + task_index * 100_003 + replicate * 1_009


def split_tasks(split: str) -> list[TaskSpec]:
    return [task for task in TASK_SPECS if task.split == split]


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray(
        [[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)],
        dtype=np.int64,
    )


def task_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    row_counts = sorted(tuple(map(int, sorted(np.bincount(row, minlength=MODULUS)))) for row in table)
    column_counts = sorted(tuple(map(int, sorted(np.bincount(column, minlength=MODULUS)))) for column in table.T)
    payload = {
        "global_counts": sorted(int(value) for value in np.bincount(table.ravel(), minlength=MODULUS)),
        "row_counts": row_counts,
        "column_counts": column_counts,
        "distinct_rows": len({tuple(map(int, row)) for row in table.tolist()}),
        "distinct_columns": len({tuple(map(int, row)) for row in table.T.tolist()}),
    }
    return {"table_digest": digest(table.tolist()), "quotient_digest": digest(payload)}


def make_data(task_name: str, seed: int) -> dict[str, torch.Tensor]:
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    labels = task_table(task_name).reshape(-1)
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(TRAIN_FRACTION * len(pairs)))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "holdout_x": x[~mask_t],
        "holdout_y": y[~mask_t],
        "all_x": x,
        "all_y": y,
    }


def checkpoint_path(split: str, task_name: str, replicate: int, seed: int) -> Path:
    return OUT_ROOT / "runs" / split / "checkpoints" / f"{task_name}_r{replicate}_s{seed}_step{ENDPOINT_STEP}.pt"


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def behavior_metrics(model, data: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "holdout"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(data[f"{split}_x"].to(device)).float()
        finite = bool(torch.isfinite(logits).all().item())
        result[f"{split}_all_finite"] = finite
        result[f"{split}_accuracy"] = (
            float((logits.argmax(1).cpu() == data[f"{split}_y"]).float().mean().item()) if finite else 0.0
        )
    result["qualified"] = bool(
        result["train_all_finite"]
        and result["holdout_all_finite"]
        and result["train_accuracy"] >= THRESHOLDS["behavior_train_accuracy_min"]
        and result["holdout_accuracy"] >= THRESHOLDS["behavior_holdout_accuracy_min"]
    )
    return result


def rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(math.sqrt(max(float(np.mean(values * values)), 0.0)))


@torch.inference_mode()
def numerical_profile(model, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, float]:
    logits_t, hidden_t = p1181.fp32_state(model, x, device)
    logits = logits_t.detach().cpu().double().numpy()
    hidden = hidden_t.detach().cpu().double().numpy()
    squared = hidden * hidden
    parameters = np.concatenate([parameter.detach().cpu().double().numpy().ravel() for parameter in model.parameters()])
    hidden_w = model.hidden.weight.detach().cpu().double().numpy()
    output_w = model.output.weight.detach().cpu().double().numpy()
    margins = p1181.correct_margin(logits_t, y.to(device)).detach().cpu().double().numpy()
    singular = np.linalg.svd(hidden, compute_uv=False)
    active = singular[singular > max(float(singular[0]) * 1e-10, 1e-30)]
    profile = {
        "parameter_l2": float(np.linalg.norm(parameters)),
        "parameter_max_abs": float(np.max(np.abs(parameters))),
        "hidden_weight_rms": rms(hidden_w),
        "hidden_weight_max_abs": float(np.max(np.abs(hidden_w))),
        "output_weight_rms": rms(output_w),
        "output_weight_max_abs": float(np.max(np.abs(output_w))),
        "hidden_activation_rms": rms(hidden),
        "hidden_activation_max_abs": float(np.max(np.abs(hidden))),
        "squared_hidden_rms": rms(squared),
        "squared_hidden_max_abs": float(np.max(np.abs(squared))),
        "logit_rms": rms(logits),
        "logit_max_abs": float(np.max(np.abs(logits))),
        "margin_rms": rms(margins),
        "margin_max_abs": float(np.max(np.abs(margins))),
        "hidden_stable_rank": float(np.sum(singular * singular) / max(float(singular[0] ** 2), 1e-30)),
        "hidden_active_condition": float(singular[0] / active[-1]) if len(active) else 1.0,
    }
    if set(profile) != set(PROFILE_METRICS):
        raise RuntimeError("profile metric mismatch")
    if not all(math.isfinite(value) and value > 0.0 for value in profile.values()):
        raise RuntimeError("nonfinite or nonpositive numerical profile")
    return profile


def profile_vector(profile: dict[str, float]) -> np.ndarray:
    return np.log(np.asarray([profile[metric] for metric in PROFILE_METRICS], dtype=np.float64))


def l_inf_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1185 is already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("audit script must exist before registration")
    signatures = {task.name: task_signature(task.name) for task in TASK_SPECS}
    if len({item["table_digest"] for item in signatures.values()}) != len(TASK_SPECS):
        raise RuntimeError("task tables must be unique")
    phase1184_final = p1184.OUT_ROOT / "analysis/final_stop.json"
    protocol = {
        "phase": PHASE,
        "registered_at_utc": utc_now(),
        "scientific_object": (
            "Orthogonally qualify the numerical gauge instrument and task behavior on fresh networks, "
            "without conditioning numerical support on holdout generalization."
        ),
        "independence": {
            "phase1184_reopened": False,
            "new_modulus": MODULUS,
            "new_tasks": True,
            "new_seeds": True,
            "discovery_support_uses_holdout_filter": False,
            "behavior_and_numerical_ledgers_parallel": True,
            "mechanism_camera_evaluated": False,
        },
        "claim_exclusions": [
            "Orthogonal means distinct constructs and ledgers, not statistical independence.",
            "A numerical pass does not confirm K165 or any natural mechanism.",
            "The support cloud is an engineering applicability envelope, not a natural manifold.",
            "The restricted output-layer witness is not full-network backward error.",
            "No result transfers directly to Transformers or language models.",
        ],
        "dimensions": {
            "modulus": MODULUS,
            "width": WIDTH,
            "replicates_per_task": REPLICATES,
            "train_fraction": TRAIN_FRACTION,
            "endpoint_step": ENDPOINT_STEP,
            "natural_gauge_transforms": NATURAL_GAUGE_TRANSFORMS,
            "engineered_gauge_transforms": ENGINEERED_GAUGE_TRANSFORMS,
        },
        "tasks": [asdict(task) | {"signature": signatures[task.name]} for task in TASK_SPECS],
        "training": TRAINING,
        "numerical_axis": {
            "profile_metrics": list(PROFILE_METRICS),
            "support_source": "all finite discovery checkpoints regardless of holdout behavior",
            "natural_support": "sealed discovery log-profile point cloud",
            "safety_envelope": (
                "union of L-infinity balls around discovery points; radius equals twice the maximum "
                "leave-one-out nearest-neighbor distance"
            ),
            "known_gauge_truth_scales": list(ENGINEERED_SCALES),
            "known_gauge_truth_structures": list(ENGINEERED_STRUCTURES),
            "stress": "trained checkpoint weight x100 plus half-channel duplication; descriptive only",
        },
        "behavior_axis": {
            "qualification": "train>=0.95 and holdout>=0.90 with finite logits",
            "effect_on_numerical_axis": "none",
            "effect_on_mechanism_authorization": "intersect only after both ledgers are finalized",
        },
        "error_contract": {
            "absolute": "max|z_g-z| <= floor + relative_multiplier*max|z|",
            "scaled": "max|z_g-z|/max|z| and RMS(z_g-z)/RMS(z)",
            "decision": (
                "For natural trained networks, argmax and label-margin sign must agree on at least 95% "
                "of cases separated from the frozen error envelope. Engineered tiny-scale strata may "
                "abstain when their margins are below that envelope; their algebraic and forward-error "
                "contracts remain mandatory."
            ),
            "restricted_backward_witness": "minimum-norm output-layer perturbation, descriptive only",
        },
        "thresholds": THRESHOLDS,
        "sequential_dependencies": [
            "discovery training seal",
            "numerical support seal from all finite discovery checkpoints",
            "confirmation training seal independent of behavior ledger",
            "parallel numerical and behavior ledgers",
            "science authorization from final intersection only",
        ],
        "auto_continue_rule": (
            "Only numerical-axis pass, behavior-axis pass, science-intersection pass, and independent audit "
            "jointly authorize one separately preregistered fresh three-evidence mechanism confirmation."
        ),
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1171_source": file_sha256(Path(p1171.__file__)),
            "phase1181_source": file_sha256(Path(p1181.__file__)),
            "phase1183_source": file_sha256(Path(p1183.__file__)),
            "phase1184_source": file_sha256(Path(p1184.__file__)),
            "phase1184_stop": file_sha256(phase1184_final),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    stored = protocol.pop("protocol_digest")
    if digest(protocol) != stored:
        raise RuntimeError("protocol digest mismatch")
    protocol["protocol_digest"] = stored
    paths = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1171_source": Path(p1171.__file__),
        "phase1181_source": Path(p1181.__file__),
        "phase1183_source": Path(p1183.__file__),
        "phase1184_source": Path(p1184.__file__),
        "phase1184_stop": p1184.OUT_ROOT / "analysis/final_stop.json",
    }
    for name, path in paths.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen source changed: {name}")
    return protocol


def train_split(split: str) -> None:
    protocol = validate_protocol()
    if split == "confirmation" and not read_json(DOMAIN_PATH)["numerical_support_pass"]:
        raise RuntimeError("confirmation denied before numerical support seal")
    seal_path = OUT_ROOT / "runs" / split / "training_seal.json"
    if seal_path.exists():
        raise RuntimeError(f"{split} training already sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for task in split_tasks(split):
        task_index = list(TASK_SPECS).index(task)
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            set_seed(seed)
            data = make_data(task.name, seed + 17)
            model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=WIDTH)).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]
            )
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
            final_loss = math.nan
            for step in range(1, ENDPOINT_STEP + 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(train_x).float(), train_y)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"nonfinite loss: {task.name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.item())
            path = checkpoint_path(split, task.name, replicate, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "split": split,
                "task_name": task.name,
                "task_index": task_index,
                "replicate": replicate,
                "seed": seed,
                "step": ENDPOINT_STEP,
                "config": {"modulus": MODULUS, "width": WIDTH},
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            torch.save(payload, path)
            hashes[path.name] = file_sha256(path)
            rows.append(
                {
                    "task_name": task.name,
                    "task_index": task_index,
                    "replicate": replicate,
                    "seed": seed,
                    "endpoint_loss": final_loss,
                    "train_pair_digest": digest(data["train_x"].tolist()),
                    "holdout_pair_digest": digest(data["holdout_x"].tolist()),
                    "holdout_labels_unread": True,
                }
            )
            print(canonical_json({"trained": task.name, "replicate": replicate, "split": split}), flush=True)
            del model, optimizer, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()
    metrics_path = OUT_ROOT / "runs" / split / "training_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    seal = {
        "phase": PHASE,
        "split": split,
        "sealed_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(rows),
        "checkpoint_hashes": hashes,
        "training_metrics_sha256": file_sha256(metrics_path),
        "all_holdout_labels_unread": all(row["holdout_labels_unread"] for row in rows),
    }
    seal["seal_digest"] = digest(seal)
    write_json(seal_path, seal)
    print(canonical_json(seal))


def verify_training_seal(split: str, protocol_digest: str) -> dict[str, Any]:
    seal = read_json(OUT_ROOT / "runs" / split / "training_seal.json")
    copy = dict(seal)
    stored = copy.pop("seal_digest")
    if digest(copy) != stored or seal["protocol_digest"] != protocol_digest:
        raise RuntimeError(f"invalid {split} seal")
    metrics_path = OUT_ROOT / "runs" / split / "training_metrics.jsonl"
    if file_sha256(metrics_path) != seal["training_metrics_sha256"]:
        raise RuntimeError(f"{split} metrics changed")
    for name, expected in seal["checkpoint_hashes"].items():
        if file_sha256(OUT_ROOT / "runs" / split / "checkpoints" / name) != expected:
            raise RuntimeError(f"checkpoint changed: {name}")
    return seal


def build_system_rows(split: str, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((OUT_ROOT / "runs" / split / "checkpoints").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        data = make_data(payload["task_name"], payload["seed"] + 17)
        behavior = behavior_metrics(model, data, device)
        profile = numerical_profile(model, data["all_x"], data["all_y"], device)
        rows.append(
            {
                "checkpoint": path.name,
                "checkpoint_sha256": file_sha256(path),
                "task_name": payload["task_name"],
                "task_index": payload["task_index"],
                "replicate": payload["replicate"],
                "seed": payload["seed"],
                "numerical_finite": True,
                "profile": profile,
                "behavior": behavior,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def behavior_summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in split_tasks(split):
        selected = [row for row in rows if row["task_name"] == task.name]
        qualified = [row for row in selected if row["behavior"]["qualified"]]
        tasks[task.name] = {
            "system_count": len(selected),
            "qualified_count": len(qualified),
            "minimum_train_accuracy": min(row["behavior"]["train_accuracy"] for row in selected),
            "minimum_holdout_accuracy": min(row["behavior"]["holdout_accuracy"] for row in selected),
            "task_pass": len(qualified) >= THRESHOLDS["behavior_qualified_per_task_min"],
        }
    qualified_count = sum(row["behavior"]["qualified"] for row in rows)
    passing_tasks = sum(item["task_pass"] for item in tasks.values())
    return {
        "qualified_system_count": qualified_count,
        "passing_task_count": passing_tasks,
        "tasks": tasks,
        "behavior_axis_pass": bool(
            qualified_count >= THRESHOLDS["behavior_qualified_system_count_min"]
            and passing_tasks >= THRESHOLDS["behavior_passing_task_count_min"]
        ),
    }


def seal_numerical_support() -> None:
    protocol = validate_protocol()
    if DOMAIN_PATH.exists():
        raise RuntimeError("numerical support already sealed")
    verify_training_seal("discovery", protocol["protocol_digest"])
    device = torch.device("cuda")
    rows = build_system_rows("discovery", device)
    rows_path = OUT_ROOT / "runs/discovery/systems.jsonl"
    write_jsonl(rows_path, rows)
    finite_rows = [row for row in rows if row["numerical_finite"]]
    cloud = np.stack([profile_vector(row["profile"]) for row in finite_rows])
    nearest: list[float] = []
    for index in range(len(cloud)):
        nearest.append(min(l_inf_distance(cloud[index], cloud[j]) for j in range(len(cloud)) if j != index))
    radius = SAFETY_RADIUS_MULTIPLIER * max(nearest)
    support_pass = bool(
        len(finite_rows) >= THRESHOLDS["numerical_discovery_system_count_min"]
        and math.isfinite(radius)
        and radius > 0.0
    )
    behavior = behavior_summary(rows, "discovery")
    seal = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(rows),
        "finite_profile_count": len(finite_rows),
        "profile_metrics": list(PROFILE_METRICS),
        "log_profile_cloud": cloud.tolist(),
        "leave_one_out_nearest_distances": nearest,
        "safety_radius": radius,
        "safety_radius_multiplier": SAFETY_RADIUS_MULTIPLIER,
        "behavior_ledger_descriptive_only": behavior,
        "rows_sha256": file_sha256(rows_path),
        "numerical_support_pass": support_pass,
    }
    seal["support_digest"] = digest(seal)
    write_json(DOMAIN_PATH, seal)
    print(canonical_json({key: value for key, value in seal.items() if key != "log_profile_cloud"}))


def support_distance(profile: dict[str, float], cloud: np.ndarray) -> float:
    vector = profile_vector(profile)
    return min(l_inf_distance(vector, point) for point in cloud)


def forward_metrics(reference: np.ndarray, changed: np.ndarray, precision: str) -> dict[str, float]:
    unit = FP64_U if precision == "fp64" else FP32_U
    multiplier = 128.0 if precision == "fp64" else 32.0
    floor = (256.0 if precision == "fp64" else 128.0) * unit
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    scale_max = max(float(np.max(np.abs(reference))), 1e-30)
    scale_rms = max(rms(reference), 1e-30)
    absolute = float(np.max(np.abs(delta)))
    return {
        "absolute_max": absolute,
        "reference_max_abs": scale_max,
        "scaled_max": absolute / scale_max,
        "rms_relative": rms(delta) / scale_rms,
        "mixed_absolute_bound": floor + multiplier * unit * scale_max,
    }


def decision_metrics(reference: torch.Tensor, changed: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    max_abs = max(float(reference.abs().max().item()), 1e-30)
    uncertainty = 2.0 * (THRESHOLDS["fp32_absolute_floor"] + THRESHOLDS["fp32_relative_multiplier"] * max_abs)
    top = torch.topk(reference, k=2, dim=1).values
    gaps = top[:, 0] - top[:, 1]
    eligible = gaps > uncertainty
    predictions_equal = reference.argmax(1) == changed.argmax(1)
    decision_fraction = float(eligible.float().mean().item())
    decision_agreement = float(predictions_equal[eligible].float().mean().item()) if bool(eligible.any()) else 1.0
    margin_reference = p1181.correct_margin(reference, targets.to(reference.device))
    margin_changed = p1181.correct_margin(changed, targets.to(reference.device))
    margin_eligible = margin_reference.abs() > uncertainty
    margin_equal = (margin_reference >= 0) == (margin_changed >= 0)
    margin_fraction = float(margin_eligible.float().mean().item())
    margin_agreement = float(margin_equal[margin_eligible].float().mean().item()) if bool(margin_eligible.any()) else 1.0
    return {
        "uncertainty_bound": uncertainty,
        "decision_eligible_fraction": decision_fraction,
        "decision_agreement": decision_agreement,
        "margin_sign_eligible_fraction": margin_fraction,
        "margin_sign_agreement": margin_agreement,
    }


def gauge_case(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: torch.device,
    *,
    require_decision_coverage: bool,
    design: np.ndarray,
    design_pinv: np.ndarray,
) -> dict[str, Any]:
    transformed = p1183.gauge_model(model, seed, device)
    original32, hidden32 = p1181.fp32_state(model, x, device)
    changed32, _ = p1181.fp32_state(transformed, x, device)
    _, original64 = p1183.cpu_hidden_and_logits(model, x)
    _, changed64 = p1183.cpu_hidden_and_logits(transformed, x)
    feature = np.asarray(p1183.algebraic_internal_features(model, x), dtype=np.float64)
    feature_changed = np.asarray(p1183.algebraic_internal_features(transformed, x), dtype=np.float64)
    fp32 = forward_metrics(original32.detach().cpu().double().numpy(), changed32.detach().cpu().double().numpy(), "fp32")
    fp64 = forward_metrics(original64, changed64, "fp64")
    decision = decision_metrics(original32, changed32, y)
    delta = changed32.detach().cpu().double().numpy() - original32.detach().cpu().double().numpy()
    delta_w_t = design_pinv @ delta
    reconstructed = design @ delta_w_t
    output_norm = float(np.linalg.norm(model.output.weight.detach().cpu().double().numpy()))
    feature_error = float(np.max(np.abs(feature - feature_changed)))
    decision_pass = bool(
        decision["decision_agreement"] >= THRESHOLDS["decision_agreement_min"]
        and decision["margin_sign_agreement"] >= THRESHOLDS["margin_sign_agreement_min"]
        and (
            not require_decision_coverage
            or (
                decision["decision_eligible_fraction"] >= THRESHOLDS["decision_eligible_fraction_min"]
                and decision["margin_sign_eligible_fraction"]
                >= THRESHOLDS["margin_sign_eligible_fraction_min"]
            )
        )
    )
    passed = bool(
        feature_error <= THRESHOLDS["algebraic_feature_error_max"]
        and fp64["absolute_max"] <= fp64["mixed_absolute_bound"]
        and fp64["scaled_max"] <= THRESHOLDS["fp64_scaled_error_max"]
        and fp64["rms_relative"] <= THRESHOLDS["fp64_scaled_error_max"]
        and fp32["absolute_max"] <= fp32["mixed_absolute_bound"]
        and fp32["scaled_max"] <= THRESHOLDS["fp32_scaled_error_max"]
        and fp32["rms_relative"] <= THRESHOLDS["fp32_scaled_error_max"]
        and decision_pass
    )
    result = {
        "seed": seed,
        "feature_error": feature_error,
        "fp64": fp64,
        "fp32": fp32,
        "decision": decision,
        "decision_coverage_required": require_decision_coverage,
        "decision_contract_pass": decision_pass,
        "restricted_output_backward_witness": {
            "relative_output_weight_norm": float(np.linalg.norm(delta_w_t.T) / max(output_norm, 1e-30)),
            "relative_reconstruction_residual": rms(delta - reconstructed) / max(rms(delta), 1e-30),
            "claim_status": "descriptive_not_full_network_backward_error",
        },
        "gauge_pass": passed,
    }
    del transformed
    return result


def positive_control(
    model,
    x: torch.Tensor,
    seed: int,
    device: torch.device,
    *,
    require_decision_difference: bool,
) -> dict[str, Any]:
    broken = p1183.gauge_model(model, seed, device, broken_output=True)
    original, _ = p1181.fp32_state(model, x, device)
    changed, _ = p1181.fp32_state(broken, x, device)
    feature = np.asarray(p1183.algebraic_internal_features(model, x), dtype=np.float64)
    feature_changed = np.asarray(p1183.algebraic_internal_features(broken, x), dtype=np.float64)
    fp32 = forward_metrics(original.detach().cpu().double().numpy(), changed.detach().cpu().double().numpy(), "fp32")
    agreement = float((original.argmax(1) == changed.argmax(1)).float().mean().item())
    feature_error = float(np.max(np.abs(feature - feature_changed)))
    passed = bool(
        feature_error >= THRESHOLDS["positive_feature_error_min"]
        and fp32["scaled_max"] >= THRESHOLDS["positive_scaled_error_min"]
        and (
            not require_decision_difference
            or agreement <= THRESHOLDS["positive_decision_agreement_max"]
        )
    )
    del broken
    return {
        "seed": seed,
        "feature_error": feature_error,
        "fp32": fp32,
        "decision_agreement": agreement,
        "decision_difference_required": require_decision_difference,
        "positive_control_pass": passed,
    }


def engineered_model(scale: float, structure: str, seed: int, device: torch.device):
    set_seed(seed)
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=WIDTH)).to(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale / 0.02)
        if structure == "half_duplicate":
            half = WIDTH // 2
            model.hidden.weight[half:].copy_(model.hidden.weight[:half])
            model.output.weight[:, half:].copy_(model.output.weight[:, :half])
    model.eval()
    return model


def engineered_calibration(device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = torch.tensor([(a, b) for a in range(MODULUS) for b in range(MODULUS)], dtype=torch.long)
    y = torch.tensor(task_table("axis_affine_b").reshape(-1), dtype=torch.long)
    gauge_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    case_index = 0
    for scale in ENGINEERED_SCALES:
        for structure in ENGINEERED_STRUCTURES:
            model = engineered_model(scale, structure, 11858000 + case_index, device)
            _, hidden32 = p1181.fp32_state(model, x, device)
            design = hidden32.detach().cpu().double().numpy() ** 2
            design_pinv = np.linalg.pinv(design, rcond=1e-10)
            for transform in range(ENGINEERED_GAUGE_TRANSFORMS):
                gauge_rows.append(
                    {
                        "case": case_index,
                        "scale": scale,
                        "structure": structure,
                        "transform": transform,
                        **gauge_case(
                            model,
                            x,
                            y,
                            11858100 + case_index * 100 + transform,
                            device,
                            require_decision_coverage=False,
                            design=design,
                            design_pinv=design_pinv,
                        ),
                    }
                )
            positive_rows.append(
                {
                    "case": case_index,
                    "scale": scale,
                    "structure": structure,
                    **positive_control(
                        model,
                        x,
                        11858900 + case_index,
                        device,
                        require_decision_difference=False,
                    ),
                }
            )
            del model
            case_index += 1
            gc.collect()
            torch.cuda.empty_cache()
    return gauge_rows, positive_rows


def stressed_copy(model, device: torch.device):
    stressed = p1171.RoleSquareNetwork(model.config).to(device)
    stressed.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in stressed.parameters():
            parameter.mul_(100.0)
        half = WIDTH // 2
        stressed.hidden.weight[half:].copy_(stressed.hidden.weight[:half])
        stressed.output.weight[:, half:].copy_(stressed.output.weight[:, :half])
    stressed.eval()
    return stressed


def stress_tier(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    selected = [next(row for row in rows if row["task_name"] == task.name) for task in split_tasks("confirmation")]
    output: list[dict[str, Any]] = []
    for row in selected:
        payload = torch.load(OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"], map_location="cpu", weights_only=False)
        original = load_model(payload, device)
        model = stressed_copy(original, device)
        data = make_data(payload["task_name"], payload["seed"] + 17)
        reference, _ = p1181.fp32_state(model, data["all_x"], device)
        reference_np = reference.detach().cpu().double().numpy()
        for transform in range(4):
            changed_model = p1183.gauge_model(model, 11859500 + payload["task_index"] * 10 + transform, device)
            changed, _ = p1181.fp32_state(changed_model, data["all_x"], device)
            output.append(
                {
                    "task_name": row["task_name"],
                    "transform": transform,
                    "fp32": forward_metrics(reference_np, changed.detach().cpu().double().numpy(), "fp32"),
                    "decision_agreement": float((reference.argmax(1) == changed.argmax(1)).float().mean().item()),
                    "all_finite": bool(torch.isfinite(reference).all().item() and torch.isfinite(changed).all().item()),
                }
            )
            del changed_model
        del model, original
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "status": "descriptive_non_gating",
        "row_count": len(output),
        "maximum_absolute_error": max(row["fp32"]["absolute_max"] for row in output),
        "maximum_scaled_error": max(row["fp32"]["scaled_max"] for row in output),
        "minimum_decision_agreement": min(row["decision_agreement"] for row in output),
        "all_finite": all(row["all_finite"] for row in output),
        "rows": output,
    }


def evaluate_confirmation() -> None:
    protocol = validate_protocol()
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1185 already finalized")
    support = read_json(DOMAIN_PATH)
    if not support["numerical_support_pass"]:
        raise RuntimeError("numerical support did not pass")
    verify_training_seal("confirmation", protocol["protocol_digest"])
    device = torch.device("cuda")
    rows = build_system_rows("confirmation", device)
    cloud = np.asarray(support["log_profile_cloud"], dtype=np.float64)
    for row in rows:
        row["support_distance"] = support_distance(row["profile"], cloud)
        row["inside_safety_envelope"] = row["support_distance"] <= support["safety_radius"]
    systems_path = OUT_ROOT / "runs/confirmation/systems.jsonl"
    write_jsonl(systems_path, rows)
    finite_rows = [row for row in rows if row["numerical_finite"]]
    coverage = sum(row["inside_safety_envelope"] for row in finite_rows) / max(len(finite_rows), 1)
    coverage_pass = bool(
        len(finite_rows) >= THRESHOLDS["numerical_confirmation_system_count_min"]
        and coverage >= THRESHOLDS["confirmation_safety_coverage_min"]
    )

    natural_gauge: list[dict[str, Any]] = []
    natural_positive: list[dict[str, Any]] = []
    for row in finite_rows:
        payload = torch.load(OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"], map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        data = make_data(payload["task_name"], payload["seed"] + 17)
        _, hidden32 = p1181.fp32_state(model, data["all_x"], device)
        design = hidden32.detach().cpu().double().numpy() ** 2
        design_pinv = np.linalg.pinv(design, rcond=1e-10)
        for transform in range(NATURAL_GAUGE_TRANSFORMS):
            natural_gauge.append(
                {
                    "checkpoint": row["checkpoint"],
                    "task_name": row["task_name"],
                    "replicate": row["replicate"],
                    "transform": transform,
                    **gauge_case(
                        model,
                        data["all_x"],
                        data["all_y"],
                        11857000 + payload["task_index"] * 10_000 + payload["replicate"] * 100 + transform,
                        device,
                        require_decision_coverage=True,
                        design=design,
                        design_pinv=design_pinv,
                    ),
                }
            )
        natural_positive.append(
            {
                "checkpoint": row["checkpoint"],
                "task_name": row["task_name"],
                "replicate": row["replicate"],
                **positive_control(
                    model,
                    data["all_x"],
                    11857900 + payload["task_index"] * 100 + payload["replicate"],
                    device,
                    require_decision_difference=True,
                ),
            }
        )
        print(canonical_json({"gauged": row["task_name"], "replicate": row["replicate"]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    engineered_gauge, engineered_positive = engineered_calibration(device)
    paths_rows = {
        "natural_gauge": (OUT_ROOT / "analysis/natural_gauge_rows.jsonl", natural_gauge),
        "natural_positive": (OUT_ROOT / "analysis/natural_positive_rows.jsonl", natural_positive),
        "engineered_gauge": (OUT_ROOT / "analysis/engineered_gauge_rows.jsonl", engineered_gauge),
        "engineered_positive": (OUT_ROOT / "analysis/engineered_positive_rows.jsonl", engineered_positive),
    }
    hashes: dict[str, str] = {}
    for name, (path, content) in paths_rows.items():
        write_jsonl(path, content)
        hashes[name] = file_sha256(path)

    natural_fraction = sum(row["gauge_pass"] for row in natural_gauge) / max(len(natural_gauge), 1)
    engineered_fraction = sum(row["gauge_pass"] for row in engineered_gauge) / max(len(engineered_gauge), 1)
    natural_positive_fraction = sum(row["positive_control_pass"] for row in natural_positive) / max(len(natural_positive), 1)
    engineered_positive_fraction = sum(row["positive_control_pass"] for row in engineered_positive) / max(len(engineered_positive), 1)
    numerical_axis_pass = bool(
        support["numerical_support_pass"]
        and coverage_pass
        and natural_fraction >= THRESHOLDS["natural_gauge_pass_fraction_min"]
        and engineered_fraction >= THRESHOLDS["engineered_gauge_pass_fraction_min"]
        and natural_positive_fraction >= THRESHOLDS["positive_control_pass_fraction_min"]
        and engineered_positive_fraction >= THRESHOLDS["positive_control_pass_fraction_min"]
    )
    behavior = behavior_summary(rows, "confirmation")
    intersection = [row for row in rows if row["behavior"]["qualified"] and row["inside_safety_envelope"]]
    intersection_tasks = len({row["task_name"] for row in intersection if sum(
        item["task_name"] == row["task_name"] for item in intersection
    ) >= THRESHOLDS["behavior_qualified_per_task_min"]})
    intersection_pass = bool(
        len(intersection) >= THRESHOLDS["science_intersection_system_count_min"]
        and intersection_tasks >= THRESHOLDS["science_intersection_task_count_min"]
    )
    stress = stress_tier(rows, device)
    authorized = bool(numerical_axis_pass and behavior["behavior_axis_pass"] and intersection_pass)
    all_gauge = natural_gauge + engineered_gauge
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_object": "orthogonal_numerical_and_behavioral_qualification_only",
        "support_digest": support["support_digest"],
        "numerical_axis": {
            "confirmation_finite_count": len(finite_rows),
            "safety_coverage": coverage,
            "coverage_pass": coverage_pass,
            "natural_gauge_row_count": len(natural_gauge),
            "natural_gauge_pass_fraction": natural_fraction,
            "engineered_gauge_row_count": len(engineered_gauge),
            "engineered_gauge_pass_fraction": engineered_fraction,
            "natural_positive_fraction": natural_positive_fraction,
            "engineered_positive_fraction": engineered_positive_fraction,
            "maximum_feature_error": max(row["feature_error"] for row in all_gauge),
            "maximum_fp64_scaled_error": max(row["fp64"]["scaled_max"] for row in all_gauge),
            "maximum_fp32_scaled_error": max(row["fp32"]["scaled_max"] for row in all_gauge),
            "maximum_fp32_rms_relative": max(row["fp32"]["rms_relative"] for row in all_gauge),
            "minimum_decision_eligible_fraction": min(row["decision"]["decision_eligible_fraction"] for row in all_gauge),
            "minimum_decision_agreement": min(row["decision"]["decision_agreement"] for row in all_gauge),
            "maximum_restricted_backward_relative_norm": max(
                row["restricted_output_backward_witness"]["relative_output_weight_norm"] for row in all_gauge
            ),
            "numerical_axis_pass": numerical_axis_pass,
            "artifact_hashes": hashes,
        },
        "behavior_axis": behavior,
        "science_intersection": {
            "system_count": len(intersection),
            "passing_task_count": intersection_tasks,
            "intersection_pass": intersection_pass,
        },
        "stress_tier": stress,
        "mechanism_camera_status": "not_tested_by_design",
        "phase1184_status": "unchanged_closed_failure",
        "mechanism_confirmation_authorized": authorized,
        "auto_continue": {
            "authorized": authorized,
            "next": "one_fresh_three_evidence_mechanism_confirmation" if authorized else None,
        },
        "systems_sha256": file_sha256(systems_path),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preregister",
            "train-discovery",
            "seal-numerical-support",
            "train-confirmation",
            "evaluate-confirmation",
        ),
    )
    args = parser.parse_args()
    commands = {
        "preregister": preregister,
        "train-discovery": lambda: train_split("discovery"),
        "seal-numerical-support": seal_numerical_support,
        "train-confirmation": lambda: train_split("confirmation"),
        "evaluate-confirmation": evaluate_confirmation,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
