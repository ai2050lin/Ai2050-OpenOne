"""Phase1183: gauge-exact prospective mechanism closure on fresh networks.

This is a new registry.  It does not reopen Phase1182 and never reads its
untouched confirmation panel.  The order of operations is frozen:

1. qualify algebraic features against the declared signed-permutation gauge;
2. train and seal fresh discovery networks;
3. establish response material, endpoint/prefix prediction, and donor rescue;
4. only after a discovery pass, train and inspect fresh confirmation tasks;
5. close this registry after the first formal decision, positive or negative.

The algebraic camera uses power sums and mixed power sums of sign-invariant
per-channel quantities.  Centering/unit normalization of the response target
is descriptive normalization, not a claimed physical gauge transformation.
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
import phase1182_quotient_response_camera_and_rescue as p1182  # noqa: E402


PHASE = 1183
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1183_gauge_exact_prospective_mechanism_closure_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1183_gauge_exact_prospective_mechanism_closure"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREFLIGHT_PATH = OUT_ROOT / "instrument/preflight.json"
CAMERA_NPZ = OUT_ROOT / "analysis/camera_seal.npz"
CAMERA_META = OUT_ROOT / "analysis/camera_seal.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

MODULUS = 31
WIDTH = 128
REPLICATES = 6
TRAIN_FRACTION = 0.50
HISTORY_STEPS = (25, 50, 75, 100, 150)
ENDPOINT_STEP = 10_000
CHECKPOINT_STEPS = (*HISTORY_STEPS, ENDPOINT_STEP)
ENDPOINT_L2 = 100.0
PREFIX_L2 = 1.0
INJURY_CHANNEL_COUNT = 32
FUTURE_MASK_COUNT = 32
FUTURE_MASK_SIZE = 8
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": ENDPOINT_STEP,
}

THRESHOLDS = {
    "instrument_feature_max_error_max": 1e-12,
    "instrument_fp64_logit_max_error_max": 1e-10,
    "instrument_fp32_logit_max_error_max": 1e-4,
    "instrument_positive_sentinel_error_min": 1e-4,
    "train_accuracy_min": 0.95,
    "holdout_accuracy_min": 0.90,
    "response_centered_norm_min": 1e-6,
    "replay_max_error_max": 1e-7,
    "gauge_fp32_logit_max_error_max": 1e-4,
    "gauge_ordered_response_max_error_max": 1e-5,
    "qualified_system_count_per_task_min": 5,
    "within_task_median_unit_shape_distance_min": 0.08,
    "behavior_matched_pair_count_per_task_min": 4,
    "behavior_matched_median_response_distance_min": 0.08,
    "absolute_behavior_response_distance_correlation_max": 0.75,
    "response_scale_coefficient_of_variation_min": 0.05,
    "discovery_passing_task_count_min": 8,
    "confirmation_passing_task_count_min": 4,
    "endpoint_joint_residual_cosine_min": 0.35,
    "endpoint_residual_cosine_improvement_min": 0.02,
    "endpoint_residual_risk_improvement_min": 0.0,
    "prefix_joint_residual_cosine_min": 0.15,
    "prefix_residual_cosine_improvement_min": 0.04,
    "prefix_residual_risk_improvement_min": 0.0,
    "injury_accuracy_drop_min": 0.50,
    "correct_rescue_accuracy_gap_from_baseline_max": 0.05,
    "wrong_rescue_accuracy_gap_from_baseline_max": 0.05,
    "correct_wrong_accuracy_difference_max": 0.03,
    "future_response_error_advantage_min": 0.10,
    "recipient_positive_future_advantage_fraction_min": 0.75,
    "discovery_positive_task_count_min": 2,
    "confirmation_positive_task_count_min": 3,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str
    family: str
    formula: str


TASK_SPECS = (
    TaskSpec("affine_fresh", "discovery", "affine", "(9*a+14*b+17) mod 31"),
    TaskSpec("product_fresh", "discovery", "bilinear", "(a+4)*(b+7)+12 mod 31"),
    TaskSpec("left_square_fresh", "discovery", "left_quadratic", "(a+5)^2+6*b+8 mod 31"),
    TaskSpec("right_square_fresh", "discovery", "right_quadratic", "10*a+(b+3)^2+4 mod 31"),
    TaskSpec("square_sum_fresh", "discovery", "separable_quadratic", "(a+2)^2+7*(b+8)^2+9 mod 31"),
    TaskSpec("maximum_fresh", "discovery", "ordered_max", "max(a+6,b+10)+3 mod 31"),
    TaskSpec("xor_fresh", "discovery", "bitwise_xor", "(a xor b)+11 mod 31"),
    TaskSpec("or_fresh", "discovery", "bitwise_or", "(a or b)+13 mod 31"),
    TaskSpec("left_cube_fresh", "confirmation", "left_cubic", "(a+6)^3+11*b+2 mod 31"),
    TaskSpec("right_cube_fresh", "confirmation", "right_cubic", "4*a+(b+9)^3+7 mod 31"),
    TaskSpec("and_fresh", "confirmation", "bitwise_and", "(a and b)+9 mod 31"),
    TaskSpec("cube_square_fresh", "confirmation", "separable_cubic_quadratic", "(a+7)^3+5*(b+4)^2+13 mod 31"),
)
TASK_BY_NAME = {task.name: task for task in TASK_SPECS}


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "affine_fresh": lambda a, b: (9 * a + 14 * b + 17) % p,
        "product_fresh": lambda a, b: (((a + 4) % p) * ((b + 7) % p) + 12) % p,
        "left_square_fresh": lambda a, b: (((a + 5) % p) ** 2 + 6 * b + 8) % p,
        "right_square_fresh": lambda a, b: (10 * a + ((b + 3) % p) ** 2 + 4) % p,
        "square_sum_fresh": lambda a, b: (((a + 2) % p) ** 2 + 7 * ((b + 8) % p) ** 2 + 9) % p,
        "maximum_fresh": lambda a, b: (max((a + 6) % p, (b + 10) % p) + 3) % p,
        "xor_fresh": lambda a, b: (((a ^ b) % p) + 11) % p,
        "or_fresh": lambda a, b: (((a | b) % p) + 13) % p,
        "left_cube_fresh": lambda a, b: (((a + 6) % p) ** 3 + 11 * b + 2) % p,
        "right_cube_fresh": lambda a, b: (4 * a + ((b + 9) % p) ** 3 + 7) % p,
        "and_fresh": lambda a, b: (((a & b) % p) + 9) % p,
        "cube_square_fresh": lambda a, b: (((a + 7) % p) ** 3 + 5 * ((b + 4) % p) ** 2 + 13) % p,
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
    return 11830000 + task_index * 100_003 + replicate * 1_009


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray(
        [[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)],
        dtype=np.int64,
    )


def task_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    size = table.shape[0]
    global_counts = np.bincount(table.ravel(), minlength=size)
    row_histograms = np.sort(
        np.stack([np.bincount(row, minlength=size) for row in table]), axis=1
    )
    column_histograms = np.sort(
        np.stack([np.bincount(column, minlength=size) for column in table.T]), axis=1
    )
    row_agreement: list[int] = []
    column_agreement: list[int] = []
    for first in range(size):
        for second in range(first + 1, size):
            row_agreement.append(int(np.sum(table[first] == table[second])))
            column_agreement.append(int(np.sum(table[:, first] == table[:, second])))
    payload = {
        "global_output_multiplicities": sorted(int(value) for value in global_counts),
        "row_histogram_multiset": sorted(tuple(map(int, row)) for row in row_histograms.tolist()),
        "column_histogram_multiset": sorted(tuple(map(int, row)) for row in column_histograms.tolist()),
        "row_pair_agreement_multiset": sorted(row_agreement),
        "column_pair_agreement_multiset": sorted(column_agreement),
        "distinct_row_count": len({tuple(map(int, row)) for row in table.tolist()}),
        "distinct_column_count": len({tuple(map(int, row)) for row in table.T.tolist()}),
    }
    return {
        "table_digest": digest(table.tolist()),
        "quotient_digest": digest(payload),
        "global_output_multiplicity_range": [
            min(payload["global_output_multiplicities"]),
            max(payload["global_output_multiplicities"]),
        ],
        "distinct_row_count": payload["distinct_row_count"],
        "distinct_column_count": payload["distinct_column_count"],
    }


def make_data(task_name: str, seed: int) -> dict[str, torch.Tensor]:
    table = task_table(task_name)
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(table.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "holdout_x": x[~mask_t],
        "holdout_y": y[~mask_t],
    }


def load_panel(payload: dict[str, Any]) -> p1181.DataPanel:
    data = make_data(str(payload["task_name"]), int(payload["seed"]) + 17)
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    train_mask[: len(data["train_x"])] = True
    return p1181.DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def checkpoint_path(split: str, task_name: str, replicate: int, seed: int, step: int) -> Path:
    return OUT_ROOT / "runs" / split / "checkpoints" / (
        f"{task_name}_r{replicate}_s{seed}_step{step:05d}.pt"
    )


def endpoint_paths(split: str) -> list[Path]:
    root = OUT_ROOT / "runs" / split / "checkpoints"
    return sorted(root.glob(f"*step{ENDPOINT_STEP:05d}.pt"))


def task_specs(split: str) -> list[TaskSpec]:
    return [task for task in TASK_SPECS if task.split == split]


def canonical_sum(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return float(math.fsum(float(value) for value in ordered))


def stable_moment(values: np.ndarray, power: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    return canonical_sum(np.power(array, power)) / max(array.size, 1)


@torch.inference_mode()
def cpu_hidden_and_logits(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    left_e = model.left_embedding.weight.detach().cpu().double().numpy()
    right_e = model.right_embedding.weight.detach().cpu().double().numpy()
    hidden_w = model.hidden.weight.detach().cpu().double().numpy()
    output_w = model.output.weight.detach().cpu().double().numpy()
    ids = x.cpu().numpy()
    hidden = (left_e[ids[:, 0]] + right_e[ids[:, 1]]) @ hidden_w.T
    logits = (hidden * hidden) @ output_w.T
    return hidden, logits


@torch.inference_mode()
def algebraic_internal_features(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
) -> list[float]:
    """Functions of generators invariant to signed hidden-channel permutation."""

    left_e = model.left_embedding.weight.detach().cpu().double().numpy()
    right_e = model.right_embedding.weight.detach().cpu().double().numpy()
    hidden_w = model.hidden.weight.detach().cpu().double().numpy()
    output_w = model.output.weight.detach().cpu().double().numpy()
    ids = x.cpu().numpy()
    left_projection = left_e @ hidden_w.T
    right_projection = right_e @ hidden_w.T
    hidden = (left_e[ids[:, 0]] + right_e[ids[:, 1]]) @ hidden_w.T

    def column_moment(matrix: np.ndarray, power: int) -> np.ndarray:
        powered = np.power(matrix, power)
        return np.asarray(
            [canonical_sum(powered[:, channel]) / powered.shape[0] for channel in range(powered.shape[1])],
            dtype=np.float64,
        )

    channels = np.stack(
        (
            np.sum(hidden_w * hidden_w, axis=1),
            np.sum(output_w * output_w, axis=0),
            np.sum(left_projection * left_projection, axis=0),
            np.sum(right_projection * right_projection, axis=0),
            column_moment(hidden, 2),
            column_moment(hidden, 4),
            column_moment(output_w, 4),
            column_moment(left_projection * right_projection, 1),
        ),
        axis=1,
    )
    scales = np.sqrt(
        np.asarray([stable_moment(channels[:, index], 2) for index in range(channels.shape[1])])
    )
    normalized = channels / np.maximum(scales[None, :], 1e-30)
    features: list[float] = []
    for index in range(channels.shape[1]):
        features.append(float(np.log1p(scales[index])))
        features.extend(stable_moment(normalized[:, index], power) for power in (1, 2, 3, 4))
    for left in range(channels.shape[1]):
        for right in range(left + 1, channels.shape[1]):
            features.append(stable_moment(normalized[:, left] * normalized[:, right], 1))
            features.append(stable_moment(normalized[:, left] ** 2 * normalized[:, right], 1))
    return features


def gauge_model(
    model: p1171.RoleSquareNetwork,
    seed: int,
    device: torch.device,
    broken_output: bool = False,
) -> p1171.RoleSquareNetwork:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(model.config.width, generator=generator)
    signs = torch.where(
        torch.rand(model.config.width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    transformed = p1171.RoleSquareNetwork(model.config).to(device)
    with torch.no_grad():
        transformed.left_embedding.weight.copy_(model.left_embedding.weight)
        transformed.right_embedding.weight.copy_(model.right_embedding.weight)
        transformed.hidden.weight.copy_(
            signs[:, None].to(device) * model.hidden.weight[permutation.to(device)]
        )
        if broken_output:
            transformed.output.weight.copy_(model.output.weight)
        else:
            transformed.output.weight.copy_(model.output.weight[:, permutation.to(device)])
    transformed.eval()
    return transformed


def instrument_preflight() -> None:
    protocol = validate_protocol(require_preflight=False)
    if PREFLIGHT_PATH.exists():
        raise RuntimeError("instrument preflight already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    x = torch.tensor([(a, b) for a in range(MODULUS) for b in range(MODULUS)], dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for case_index, scale in enumerate((1e-4, 0.02, 2.0)):
        set_seed(11833000 + case_index)
        model = p1171.RoleSquareNetwork(
            p1171.RoleSquareConfig(modulus=MODULUS, width=WIDTH)
        ).to(device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale / 0.02)
            if case_index == 2:
                model.hidden.weight[64:].copy_(model.hidden.weight[:64])
                model.output.weight[:, 64:].copy_(model.output.weight[:, :64])
        reference_features = np.asarray(algebraic_internal_features(model, x), dtype=np.float64)
        reference_hidden, reference_logits = cpu_hidden_and_logits(model, x)
        cuda_logits, _ = p1181.fp32_state(model, x, device)
        reverse_features = np.asarray(algebraic_internal_features(model, x.flip(0)), dtype=np.float64)
        rows.append(
            {
                "case": case_index,
                "kind": "batch_reverse",
                "feature_error": float(np.max(np.abs(reference_features - reverse_features))),
                "fp64_logit_error": 0.0,
                "fp32_logit_error": 0.0,
            }
        )
        current = model
        for transform_index in range(8):
            transformed = gauge_model(current, 11834000 + case_index * 100 + transform_index, device)
            features = np.asarray(algebraic_internal_features(transformed, x), dtype=np.float64)
            _, logits = cpu_hidden_and_logits(transformed, x)
            transformed_cuda_logits, _ = p1181.fp32_state(transformed, x, device)
            rows.append(
                {
                    "case": case_index,
                    "kind": "signed_permutation",
                    "iteration": transform_index,
                    "feature_error": float(np.max(np.abs(reference_features - features))),
                    "fp64_logit_error": float(np.max(np.abs(reference_logits - logits))),
                    "fp32_logit_error": float((cuda_logits - transformed_cuda_logits).abs().max().item()),
                }
            )
            if current is not model:
                del current
            current = transformed
        broken = gauge_model(model, 11835000 + case_index, device, broken_output=True)
        broken_features = np.asarray(algebraic_internal_features(broken, x), dtype=np.float64)
        _, broken_logits = cpu_hidden_and_logits(broken, x)
        rows.append(
            {
                "case": case_index,
                "kind": "leak_positive_sentinel",
                "feature_error": float(np.max(np.abs(reference_features - broken_features))),
                "fp64_logit_error": float(np.max(np.abs(reference_logits - broken_logits))),
                "fp32_logit_error": None,
            }
        )
        if current is not model:
            del current
        del model, broken, reference_hidden
        torch.cuda.empty_cache()
    valid = [row for row in rows if row["kind"] != "leak_positive_sentinel"]
    sentinels = [row for row in rows if row["kind"] == "leak_positive_sentinel"]
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "case_count": len(rows),
        "feature_max_error": max(row["feature_error"] for row in valid),
        "fp64_logit_max_error": max(row["fp64_logit_error"] for row in valid),
        "fp32_logit_max_error": max(row["fp32_logit_error"] for row in valid),
        "positive_sentinel_min_error": min(row["feature_error"] for row in sentinels),
        "rows": rows,
    }
    thresholds = protocol["thresholds"]
    summary["preflight_pass"] = bool(
        summary["feature_max_error"] <= thresholds["instrument_feature_max_error_max"]
        and summary["fp64_logit_max_error"] <= thresholds["instrument_fp64_logit_max_error_max"]
        and summary["fp32_logit_max_error"] <= thresholds["instrument_fp32_logit_max_error_max"]
        and summary["positive_sentinel_min_error"] >= thresholds["instrument_positive_sentinel_error_min"]
    )
    summary["summary_digest"] = digest(summary)
    write_json(PREFLIGHT_PATH, summary)
    print(canonical_json(summary))


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1183 is already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError(f"missing audit script: {AUDIT_SCRIPT}")
    signatures = {task.name: task_signature(task.name) for task in TASK_SPECS}
    if len({item["table_digest"] for item in signatures.values()}) != len(TASK_SPECS):
        raise RuntimeError("formal task tables are not unique")
    if len({item["quotient_digest"] for item in signatures.values()}) != len(TASK_SPECS):
        raise RuntimeError("formal task quotient signatures are not unique")
    protocol = {
        "phase": PHASE,
        "registered_at_utc": utc_now(),
        "scientific_object": (
            "One-shot prospective test of whether algebraic signed-permutation invariants predict fresh "
            "single-channel causal-response quotients before intervention and guide response-matched donor rescue."
        ),
        "registry_independence": {
            "phase1182_confirmation_read": False,
            "networks": "freshly trained after registration",
            "tasks": "new modulus-31 tables with pairwise distinct frozen quotient signatures",
            "seeds": "new Phase1183 seed namespace",
        },
        "claim_exclusions": [
            "A pass does not establish semantic identity, a language mechanism, or a complete mechanism quotient.",
            "Response-matched donor is not a ground-truth algorithm label.",
            "Response centering and unit normalization are descriptive normalizations, not physical gauge transformations.",
            "The declared gauge group is limited to hidden-channel signed permutations.",
        ],
        "gauge": {
            "group": "S_128 semidirect product (Z_2)^128",
            "network_action": "permute hidden rows and matching output columns; hidden-row signs vanish through square",
            "feature_generators": "even channel quantities, power sums p_k, and mixed power sums",
            "identity": "phi(g dot H) = phi(H) for every declared signed permutation",
        },
        "dimensions": {
            "modulus": MODULUS,
            "width": WIDTH,
            "replicates_per_task": REPLICATES,
            "train_fraction": TRAIN_FRACTION,
        },
        "tasks": [asdict(task) | {"signature": signatures[task.name]} for task in TASK_SPECS],
        "splits": {
            "discovery": {
                "fit_tasks": [task.name for task in TASK_SPECS[:6]],
                "test_and_rescue_tasks": [task.name for task in TASK_SPECS[6:8]],
            },
            "confirmation": {
                "tasks": [task.name for task in TASK_SPECS if task.split == "confirmation"],
            },
        },
        "training": TRAINING,
        "history_steps": list(HISTORY_STEPS),
        "endpoint_step": ENDPOINT_STEP,
        "camera": {
            "endpoint_l2": ENDPOINT_L2,
            "prefix_l2": PREFIX_L2,
            "target": "sorted, centered, unit-normalized single-channel correct-margin drops",
            "null": "output/behavior moments and their frozen prefix trajectory",
            "joint": "null plus algebraic internal invariant generators",
        },
        "rescue": {
            "calibration_future_split": "holdout parity split frozen in code",
            "injury_channel_count": INJURY_CHANNEL_COUNT,
            "future_mask_count": FUTURE_MASK_COUNT,
            "future_mask_size": FUTURE_MASK_SIZE,
            "donor_pool": "four behavior-nearest same-task donors",
            "correct_label": "smallest calibration response distance in donor pool",
            "wrong_label": "largest calibration response distance in donor pool",
        },
        "thresholds": THRESHOLDS,
        "sequential_gates": [
            "instrument_preflight",
            "fresh_discovery_material",
            "discovery_endpoint_prediction",
            "discovery_prefix_prediction",
            "discovery_donor_rescue",
            "fresh_confirmation_material",
            "confirmation_endpoint_prediction",
            "confirmation_prefix_prediction",
            "confirmation_donor_rescue",
        ],
        "failure_action": (
            "Stop at the first failed gate, leave later split unread, close this registry, and do not retune "
            "features, thresholds, precision, task list, steps, injury, masks, or regularization."
        ),
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1171_source": file_sha256(Path(p1171.__file__)),
            "phase1181_source": file_sha256(Path(p1181.__file__)),
            "phase1182_source": file_sha256(Path(p1182.__file__)),
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


def validate_protocol(require_preflight: bool = True) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1183 is not preregistered")
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
        "phase1182_source": Path(p1182.__file__),
    }
    for name, path in paths.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen script changed: {name}")
    if require_preflight:
        if not PREFLIGHT_PATH.exists() or not read_json(PREFLIGHT_PATH)["preflight_pass"]:
            raise RuntimeError("instrument preflight did not pass")
    return protocol


@torch.inference_mode()
def endpoint_accuracy(model, x, y, device: torch.device) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x.to(device)).float()
    return float((logits.argmax(dim=1).cpu() == y).float().mean().item())


def train_split(split: str) -> None:
    protocol = validate_protocol()
    if split == "confirmation":
        discovery = OUT_ROOT / "runs/discovery/summary.json"
        if not discovery.exists() or not read_json(discovery)["discovery_pass"]:
            raise RuntimeError("confirmation training denied because discovery did not pass")
    seal_path = OUT_ROOT / "runs" / split / "training_seal.json"
    if seal_path.exists():
        raise RuntimeError(f"{split} training is already sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    tasks = task_specs(split)
    for task in tasks:
        task_index = list(TASK_SPECS).index(task)
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            set_seed(seed)
            data = make_data(task.name, seed + 17)
            model = p1171.RoleSquareNetwork(
                p1171.RoleSquareConfig(modulus=MODULUS, width=WIDTH)
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=TRAINING["learning_rate"],
                weight_decay=TRAINING["weight_decay"],
            )
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
            final_loss = math.nan
            for step in range(1, ENDPOINT_STEP + 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(train_x).float(), train_y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss: {task.name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.item())
                if step not in CHECKPOINT_STEPS:
                    continue
                path = checkpoint_path(split, task.name, replicate, seed, step)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "phase": PHASE,
                    "protocol_digest": protocol["protocol_digest"],
                    "split": split,
                    "task_name": task.name,
                    "task_index": task_index,
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
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
                    "endpoint_train_accuracy": endpoint_accuracy(model, data["train_x"], data["train_y"], device),
                    "endpoint_loss": final_loss,
                    "train_pair_digest": digest(data["train_x"].tolist()),
                    "holdout_pair_digest": digest(data["holdout_x"].tolist()),
                    "holdout_labels_unread": True,
                }
            )
            print(canonical_json({"split": split, "trained": task.name, "replicate": replicate}), flush=True)
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
        "trajectory_count": len(rows),
        "checkpoint_count": len(hashes),
        "training_metrics_sha256": file_sha256(metrics_path),
        "checkpoint_hashes": hashes,
        "all_holdout_labels_unread": all(row["holdout_labels_unread"] for row in rows),
    }
    seal["seal_digest"] = digest(seal)
    write_json(seal_path, seal)
    print(canonical_json(seal))


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def scalar_moments(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    scale = math.sqrt(max(stable_moment(values, 2), 1e-30))
    normalized = values / scale
    return [float(np.log1p(scale)), *(stable_moment(normalized, k) for k in (1, 2, 3, 4))]


@torch.inference_mode()
def output_features(model, panel: p1181.DataPanel, device: torch.device) -> list[float]:
    logits, _ = p1181.fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    margins = p1181.correct_margin(logits, targets)
    probabilities = torch.softmax(logits, dim=1)
    arrays = (
        margins,
        probabilities.max(dim=1).values,
        probabilities.gather(1, targets[:, None]).squeeze(1),
        logits.norm(dim=1),
    )
    features: list[float] = []
    for mask in (panel.train_mask, panel.holdout_mask):
        selected = mask.to(device)
        features.extend(
            (
                float((logits[selected].argmax(dim=1) == targets[selected]).float().mean().item()),
                float(F.cross_entropy(logits[selected], targets[selected]).item()),
            )
        )
        for array in arrays:
            features.extend(scalar_moments(array[selected].cpu().numpy()))
    for parameter in model.parameters():
        features.append(float(parameter.detach().float().norm().item()))
    return features


def trajectory_summary(rows: list[list[float]]) -> list[float]:
    return p1182.trajectory_summary(rows)


def gauge_check(model, panel, response, seed, device) -> dict[str, float]:
    transformed = gauge_model(model, seed, device)
    original_logits, _ = p1181.fp32_state(model, panel.x, device)
    transformed_logits, _ = p1181.fp32_state(transformed, panel.x, device)
    transformed_response = p1181.response_spectrum(transformed, panel, device)
    original_feature = np.asarray(algebraic_internal_features(model, panel.x), dtype=np.float64)
    transformed_feature = np.asarray(algebraic_internal_features(transformed, panel.x), dtype=np.float64)
    result = {
        "fp32_logit_maximum_error": float((original_logits - transformed_logits).abs().max().item()),
        "ordered_response_maximum_error": float(
            np.max(np.abs(np.asarray(response["ordered"]) - np.asarray(transformed_response["ordered"])))
        ),
        "algebraic_feature_maximum_error": float(np.max(np.abs(original_feature - transformed_feature))),
    }
    del transformed
    return result


def build_record(path: Path, split: str, gauge: bool, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = load_model(payload, device)
    panel = load_panel(payload)
    behavior = p1181.behavior_metrics(model, panel, device)
    response = p1181.response_spectrum(model, panel, device)
    replay = p1181.response_spectrum(model, panel, device)
    replay_error = float(np.max(np.abs(np.asarray(response["ordered"]) - np.asarray(replay["ordered"]))))
    endpoint_internal = algebraic_internal_features(model, panel.x)
    endpoint_output = output_features(model, panel, device)
    prefix_internal_rows: list[list[float]] = []
    prefix_output_rows: list[list[float]] = []
    for step in HISTORY_STEPS:
        history = torch.load(
            checkpoint_path(split, payload["task_name"], payload["replicate"], payload["seed"], step),
            map_location="cpu",
            weights_only=False,
        )
        history_model = load_model(history, device)
        prefix_internal_rows.append(algebraic_internal_features(history_model, panel.x))
        prefix_output_rows.append(output_features(history_model, panel, device))
        del history_model
    gauge_result = gauge_check(model, panel, response, 11836000 + payload["task_index"], device) if gauge else None
    record = {
        "phase": PHASE,
        "split": split,
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "task_name": payload["task_name"],
        "task_index": payload["task_index"],
        "replicate": payload["replicate"],
        "seed": payload["seed"],
        "behavior": behavior,
        "response": response,
        "target": response["unit_shape"],
        "endpoint_null": endpoint_output + trajectory_summary(prefix_output_rows),
        "endpoint_internal": endpoint_internal,
        "prefix_null": trajectory_summary(prefix_output_rows),
        "prefix_internal": trajectory_summary(prefix_internal_rows),
        "replay_maximum_error": replay_error,
        "gauge": gauge_result,
    }
    del model
    torch.cuda.empty_cache()
    return record


def material_summary(rows: list[dict[str, Any]], split: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    summary = p1181.summarize(rows, split, thresholds)
    summary["phase"] = PHASE
    algebraic_errors = [
        row["gauge"]["algebraic_feature_maximum_error"]
        for row in rows
        if row["gauge"] is not None
    ]
    summary["maximum_algebraic_feature_error"] = max(algebraic_errors)
    summary["algebraic_feature_pass"] = bool(
        summary["maximum_algebraic_feature_error"]
        <= thresholds["instrument_feature_max_error_max"]
    )
    summary["split_pass"] = bool(summary["split_pass"] and summary["algebraic_feature_pass"])
    summary.pop("summary_digest", None)
    summary["summary_digest"] = digest(summary)
    return summary


def qualified(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return bool(
        row["behavior"]["all_logits_finite"] == 1.0
        and row["behavior"]["train_accuracy"] >= thresholds["train_accuracy_min"]
        and row["behavior"]["holdout_accuracy"] >= thresholds["holdout_accuracy_min"]
        and row["response"]["centered_norm"] >= thresholds["response_centered_norm_min"]
    )


def fit_camera(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fit_names = {task.name for task in TASK_SPECS[:6]}
    fit_rows = [row for row in rows if row["task_name"] in fit_names and qualified(row, THRESHOLDS)]
    return {
        "endpoint": p1182.fit_stage(fit_rows, "endpoint", ENDPOINT_L2),
        "prefix": p1182.fit_stage(fit_rows, "prefix", PREFIX_L2),
    }


def save_camera(camera: dict[str, Any], protocol_digest: str) -> None:
    arrays: dict[str, np.ndarray] = {}
    for stage, stage_cameras in camera.items():
        for label, seal in stage_cameras.items():
            for key, value in seal.items():
                arrays[f"{stage}__{label}__{key}"] = value
    CAMERA_NPZ.parent.mkdir(parents=True, exist_ok=True)
    temporary = CAMERA_NPZ.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, CAMERA_NPZ)
    metadata = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol_digest,
        "npz_sha256": file_sha256(CAMERA_NPZ),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    metadata["metadata_digest"] = digest(metadata)
    write_json(CAMERA_META, metadata)


def load_camera() -> dict[str, Any]:
    metadata = read_json(CAMERA_META)
    if file_sha256(CAMERA_NPZ) != metadata["npz_sha256"]:
        raise RuntimeError("camera seal hash mismatch")
    arrays = np.load(CAMERA_NPZ)
    return {
        stage: {
            label: {
                key: arrays[f"{stage}__{label}__{key}"]
                for key in ("mean", "scale", "weights")
            }
            for label in ("null", "joint")
        }
        for stage in ("endpoint", "prefix")
    }


def future_masks(task_name: str) -> list[np.ndarray]:
    seed = int(hashlib.sha256(("phase1183:" + task_name).encode("utf-8")).hexdigest()[:16], 16)
    generator = np.random.default_rng(seed)
    return [
        np.sort(generator.choice(WIDTH, size=FUTURE_MASK_SIZE, replace=False))
        for _ in range(FUTURE_MASK_COUNT)
    ]


def rescue_task(task_name: str, rows: list[dict[str, Any]], split: str, device: torch.device) -> dict[str, Any]:
    bundles: list[dict[str, Any]] = []
    by_name = {path.name: path for path in endpoint_paths(split)}
    for row in rows:
        payload = torch.load(by_name[row["checkpoint"]], map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        panel = load_panel(payload)
        bundle = p1182.rescue_bundle(model, panel, device)
        bundle.update({"row": row, "model": model, "panel": panel})
        bundles.append(bundle)
    behavior = np.stack([bundle["behavior_vector"] for bundle in bundles])
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    masks = future_masks(task_name)
    output_rows: list[dict[str, Any]] = []
    for recipient_index, recipient in enumerate(bundles):
        candidates = [index for index in range(len(bundles)) if index != recipient_index]
        candidates.sort(key=lambda index: float(np.linalg.norm(behavior[recipient_index] - behavior[index])))
        pool = candidates[:4]
        recipient_ordered = np.sort(recipient["calibration_response"])
        distances = {
            index: float(np.linalg.norm(np.sort(bundles[index]["calibration_response"]) - recipient_ordered))
            for index in pool
        }
        correct_index = min(pool, key=lambda index: distances[index])
        wrong_index = max(pool, key=lambda index: distances[index])
        recipient_order = np.argsort(recipient["calibration_response"])
        recipient_rank = np.empty(len(recipient_order), dtype=np.int64)
        recipient_rank[recipient_order] = np.arange(len(recipient_order))
        injured_channels = np.argsort(np.abs(recipient["calibration_response"]))[-INJURY_CHANNEL_COUNT:]
        baseline = p1182.evaluate_hybrid(
            recipient["q_evaluation"], recipient["weight"], recipient["targets_evaluation"], masks, device
        )
        injured_q = recipient["q_evaluation"].clone()
        injured_q[:, injured_channels] = 0.0
        injured = p1182.evaluate_hybrid(
            injured_q, recipient["weight"], recipient["targets_evaluation"], masks, device
        )
        rescues: dict[str, Any] = {}
        for label, donor_index in (("correct", correct_index), ("wrong", wrong_index)):
            donor = bundles[donor_index]
            _, donor_hidden = p1181.fp32_state(donor["model"], recipient["panel"].x, device)
            donor_q = donor_hidden.square()[recipient["evaluation_mask"].to(device)].cpu()
            donor_order = np.argsort(donor["calibration_response"])
            hybrid_q = recipient["q_evaluation"].clone()
            hybrid_weight = recipient["weight"].clone()
            for recipient_channel in injured_channels:
                donor_channel = donor_order[recipient_rank[recipient_channel]]
                hybrid_q[:, recipient_channel] = donor_q[:, donor_channel]
                hybrid_weight[:, recipient_channel] = donor["weight"][:, donor_channel]
            evaluated = p1182.evaluate_hybrid(
                hybrid_q, hybrid_weight, recipient["targets_evaluation"], masks, device
            )
            evaluated["future_response_error"] = p1182.response_error(
                evaluated["future_response"], baseline["future_response"]
            )
            evaluated["donor_replicate"] = donor["row"]["replicate"]
            evaluated["calibration_response_distance"] = distances[donor_index]
            rescues[label] = evaluated
        injured["future_response_error"] = p1182.response_error(
            injured["future_response"], baseline["future_response"]
        )
        output_rows.append(
            {
                "task_name": task_name,
                "recipient_replicate": recipient["row"]["replicate"],
                "baseline": baseline,
                "injured": injured,
                "correct": rescues["correct"],
                "wrong": rescues["wrong"],
            }
        )
    for bundle in bundles:
        del bundle["model"]
    torch.cuda.empty_cache()
    return {"task_name": task_name, "rows": output_rows}


def scan_split(split: str) -> None:
    protocol = validate_protocol()
    summary_path = OUT_ROOT / "runs" / split / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"{split} scan already exists")
    seal = read_json(OUT_ROOT / "runs" / split / "training_seal.json")
    expected_count = len(task_specs(split)) * REPLICATES
    paths = endpoint_paths(split)
    if len(paths) != expected_count or seal["trajectory_count"] != expected_count:
        raise RuntimeError(f"invalid {split} training seal")
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
        if not discovery["discovery_pass"]:
            raise RuntimeError("confirmation scan denied")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    gauged: set[str] = set()
    for index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task_name = payload["task_name"]
        rows.append(build_record(path, split, task_name not in gauged, device))
        gauged.add(task_name)
        print(canonical_json({"split": split, "scanned": index + 1, "total": len(paths)}), flush=True)
    material = material_summary(rows, split, protocol["thresholds"])
    rows_path = OUT_ROOT / "runs" / split / "systems.jsonl"
    write_jsonl(rows_path, rows)
    if split == "discovery":
        if not material["split_pass"]:
            summary = {
                "phase": PHASE,
                "split": split,
                "created_at_utc": utc_now(),
                "material": material,
                "discovery_pass": False,
                "stop_reason": "fresh discovery material gate failed",
                "confirmation_status": "not_run",
            }
            summary["summary_digest"] = digest(summary)
            write_json(summary_path, summary)
            print(canonical_json(summary))
            return
        camera = fit_camera(rows)
        test_names = {task.name for task in TASK_SPECS[6:8]}
        test_rows = [row for row in rows if row["task_name"] in test_names and qualified(row, THRESHOLDS)]
        endpoint = p1182.score_stage(test_rows, "endpoint", camera["endpoint"])
        prefix = p1182.score_stage(test_rows, "prefix", camera["prefix"])
        endpoint["gate_pass"] = p1182.camera_gate("endpoint", endpoint, THRESHOLDS)
        prefix["gate_pass"] = p1182.camera_gate("prefix", prefix, THRESHOLDS)
        rescue_tasks = [
            rescue_task(name, [row for row in test_rows if row["task_name"] == name], split, device)
            for name in sorted(test_names)
        ]
        write_json(OUT_ROOT / "runs" / split / "rescue_raw.json", {"tasks": rescue_tasks})
        rescue = p1182.rescue_summary(rescue_tasks, split, THRESHOLDS)
        discovery_pass = bool(endpoint["gate_pass"] and prefix["gate_pass"] and rescue["gate_pass"])
        if discovery_pass:
            save_camera(camera, protocol["protocol_digest"])
        summary = {
            "phase": PHASE,
            "split": split,
            "created_at_utc": utc_now(),
            "material": material,
            "endpoint": endpoint,
            "prefix": prefix,
            "rescue": rescue,
            "discovery_pass": discovery_pass,
            "camera_sealed": discovery_pass,
            "confirmation_status": "authorized_not_run" if discovery_pass else "not_run",
            "rows_digest": digest(rows),
        }
    else:
        if not material["split_pass"]:
            summary = {
                "phase": PHASE,
                "split": split,
                "created_at_utc": utc_now(),
                "material": material,
                "confirmation_pass": False,
                "stop_reason": "fresh confirmation material gate failed",
                "rows_digest": digest(rows),
            }
            summary["summary_digest"] = digest(summary)
            write_json(summary_path, summary)
            print(canonical_json(summary))
            return
        camera = load_camera()
        confirmation_rows = [row for row in rows if qualified(row, THRESHOLDS)]
        endpoint = p1182.score_stage(confirmation_rows, "endpoint", camera["endpoint"])
        prefix = p1182.score_stage(confirmation_rows, "prefix", camera["prefix"])
        endpoint["gate_pass"] = p1182.camera_gate("endpoint", endpoint, THRESHOLDS)
        prefix["gate_pass"] = p1182.camera_gate("prefix", prefix, THRESHOLDS)
        rescue_tasks = [
            rescue_task(
                task.name,
                [row for row in confirmation_rows if row["task_name"] == task.name],
                split,
                device,
            )
            for task in task_specs(split)
        ]
        write_json(OUT_ROOT / "runs" / split / "rescue_raw.json", {"tasks": rescue_tasks})
        rescue = p1182.rescue_summary(rescue_tasks, split, THRESHOLDS)
        confirmation_pass = bool(
            material["split_pass"]
            and endpoint["gate_pass"]
            and prefix["gate_pass"]
            and rescue["gate_pass"]
        )
        summary = {
            "phase": PHASE,
            "split": split,
            "created_at_utc": utc_now(),
            "material": material,
            "endpoint": endpoint,
            "prefix": prefix,
            "rescue": rescue,
            "confirmation_pass": confirmation_pass,
            "rows_digest": digest(rows),
        }
    summary["summary_digest"] = digest(summary)
    write_json(summary_path, summary)
    print(canonical_json(summary))


def finalize() -> None:
    protocol = validate_protocol()
    discovery_path = OUT_ROOT / "runs/discovery/summary.json"
    if not discovery_path.exists():
        raise RuntimeError("discovery summary is missing")
    discovery = read_json(discovery_path)
    confirmation_path = OUT_ROOT / "runs/confirmation/summary.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    primary = bool(
        discovery.get("discovery_pass", False)
        and confirmation is not None
        and confirmation.get("confirmation_pass", False)
    )
    if not discovery.get("discovery_pass", False):
        status = "discovery_gate_stop_confirmation_unread"
    elif confirmation is None:
        status = "confirmation_authorized_not_run"
    elif primary:
        status = "prospective_micro_network_response_mechanism_candidate_confirmed"
    else:
        status = "independent_confirmation_failed"
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_status": status,
        "primary_pass": primary,
        "instrument_preflight": read_json(PREFLIGHT_PATH),
        "discovery": discovery,
        "confirmation": confirmation,
        "claim_scope": (
            "A pass establishes prospective prediction and selective response-spectrum rescue only in this "
            "fresh RoleSquareNetwork panel under the declared signed-permutation quotient. It does not establish "
            "semantic identity, behavioral necessity, a complete gauge quotient, or language-model transfer."
        ),
        "registry": "closed_after_one_formal_decision",
        "auto_continue": {
            "authorized": primary,
            "next": "separately preregistered freely trained micro-Transformer transfer" if primary else None,
            "reason": "Only an independently confirmed three-evidence conjunction authorizes transfer.",
        },
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
            "instrument-preflight",
            "train-discovery",
            "scan-discovery",
            "train-confirmation",
            "scan-confirmation",
            "finalize",
        ),
    )
    args = parser.parse_args()
    commands = {
        "preregister": preregister,
        "instrument-preflight": instrument_preflight,
        "train-discovery": lambda: train_split("discovery"),
        "scan-discovery": lambda: scan_split("discovery"),
        "train-confirmation": lambda: train_split("confirmation"),
        "scan-confirmation": lambda: scan_split("confirmation"),
        "finalize": finalize,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
