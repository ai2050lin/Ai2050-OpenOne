#!/usr/bin/env python3
"""Phase1184: independent numerical applicability-domain qualification.

This phase does not reopen Phase1183 and does not evaluate a mechanism camera.
It first estimates a natural numerical support from fresh trained networks,
seals a safety domain, and then prospectively tests independent tasks/seeds.
Adversarial scaling is reported in a separate, non-gating stress tier.
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


PHASE = 1184
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1184_numerical_domain_qualification_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1184_numerical_domain_qualification"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
DOMAIN_PATH = OUT_ROOT / "analysis/discovery_domain_seal.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

MODULUS = 31
WIDTH = 128
REPLICATES = 8
TRAIN_FRACTION = 0.50
ENDPOINT_STEP = 10_000
GAUGE_TRANSFORMS = 8
SAFETY_MULTIPLIER = 4.0
FP32_U = float(2.0 ** -24)
FP64_U = float(2.0 ** -53)

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
    "train_accuracy_min": 0.95,
    "holdout_accuracy_min": 0.90,
    "qualified_system_count_per_task_min": 6,
    "passing_task_count_per_split_min": 3,
    "qualified_system_count_per_split_min": 24,
    "confirmation_safety_domain_fraction_min": 0.90,
    "algebraic_feature_max_error_max": 1e-12,
    "fp64_scaled_forward_error_max": 128.0 * FP64_U,
    "fp32_scaled_forward_error_max": 32.0 * FP32_U,
    "fp64_absolute_floor": 256.0 * FP64_U,
    "fp32_absolute_floor": 128.0 * FP32_U,
    "fp64_relative_multiplier": 128.0 * FP64_U,
    "fp32_relative_multiplier": 32.0 * FP32_U,
    "decision_agreement_min": 1.0,
    "margin_sign_agreement_min": 1.0,
    "valid_gauge_fraction_min": 1.0,
    "positive_feature_error_min": 1e-4,
    "positive_scaled_forward_error_min": 1e-3,
    "positive_decision_agreement_max": 0.95,
    "positive_control_fraction_min": 0.95,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str
    family: str
    formula: str


TASK_SPECS = (
    TaskSpec("domain_affine_a", "discovery", "affine", "(13*a+7*b+4) mod 31"),
    TaskSpec("domain_product_a", "discovery", "bilinear", "(a+5)*(b+9)+3 mod 31"),
    TaskSpec("domain_left_square_a", "discovery", "quadratic", "(a+4)^2+11*b+8 mod 31"),
    TaskSpec("domain_xor_a", "discovery", "bitwise", "(a xor b)+6 mod 31"),
    TaskSpec("domain_affine_b", "confirmation", "affine", "(17*a+12*b+9) mod 31"),
    TaskSpec("domain_square_sum_b", "confirmation", "quadratic", "(a+7)^2+3*(b+5)^2+14 mod 31"),
    TaskSpec("domain_left_cube_b", "confirmation", "cubic", "(a+2)^3+9*b+11 mod 31"),
    TaskSpec("domain_xor_b", "confirmation", "bitwise", "(a xor b)+15 mod 31"),
)


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "domain_affine_a": lambda a, b: (13 * a + 7 * b + 4) % p,
        "domain_product_a": lambda a, b: (((a + 5) % p) * ((b + 9) % p) + 3) % p,
        "domain_left_square_a": lambda a, b: (((a + 4) % p) ** 2 + 11 * b + 8) % p,
        "domain_xor_a": lambda a, b: (((a ^ b) % p) + 6) % p,
        "domain_affine_b": lambda a, b: (17 * a + 12 * b + 9) % p,
        "domain_square_sum_b": lambda a, b: (((a + 7) % p) ** 2 + 3 * ((b + 5) % p) ** 2 + 14) % p,
        "domain_left_cube_b": lambda a, b: (((a + 2) % p) ** 3 + 9 * b + 11) % p,
        "domain_xor_b": lambda a, b: (((a ^ b) % p) + 15) % p,
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
    return 11840000 + task_index * 100_003 + replicate * 1_009


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray(
        [[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)],
        dtype=np.int64,
    )


def task_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    row_counts = sorted(
        tuple(map(int, sorted(np.bincount(row, minlength=MODULUS)))) for row in table
    )
    column_counts = sorted(
        tuple(map(int, sorted(np.bincount(column, minlength=MODULUS)))) for column in table.T
    )
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


def split_tasks(split: str) -> list[TaskSpec]:
    return [task for task in TASK_SPECS if task.split == split]


def checkpoint_path(split: str, task_name: str, replicate: int, seed: int) -> Path:
    return OUT_ROOT / "runs" / split / "checkpoints" / f"{task_name}_r{replicate}_s{seed}_step{ENDPOINT_STEP}.pt"


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def accuracy(model: p1171.RoleSquareNetwork, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, Any]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x.to(device)).float()
    finite = bool(torch.isfinite(logits).all().item())
    return {
        "accuracy": float((logits.argmax(dim=1).cpu() == y).float().mean().item()) if finite else 0.0,
        "all_logits_finite": finite,
    }


def positive_rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(math.sqrt(max(float(np.mean(values * values)), 0.0)))


@torch.inference_mode()
def numerical_profile(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
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
    condition = float(singular[0] / active[-1]) if len(active) else 1.0
    stable_rank = float(np.sum(singular * singular) / max(float(singular[0] ** 2), 1e-30))
    result = {
        "parameter_l2": float(np.linalg.norm(parameters)),
        "parameter_max_abs": float(np.max(np.abs(parameters))),
        "hidden_weight_rms": positive_rms(hidden_w),
        "hidden_weight_max_abs": float(np.max(np.abs(hidden_w))),
        "output_weight_rms": positive_rms(output_w),
        "output_weight_max_abs": float(np.max(np.abs(output_w))),
        "hidden_activation_rms": positive_rms(hidden),
        "hidden_activation_max_abs": float(np.max(np.abs(hidden))),
        "squared_hidden_rms": positive_rms(squared),
        "squared_hidden_max_abs": float(np.max(np.abs(squared))),
        "logit_rms": positive_rms(logits),
        "logit_max_abs": float(np.max(np.abs(logits))),
        "margin_rms": positive_rms(margins),
        "margin_max_abs": float(np.max(np.abs(margins))),
        "hidden_stable_rank": stable_rank,
        "hidden_active_condition": condition,
    }
    if set(result) != set(PROFILE_METRICS) or not all(math.isfinite(value) and value > 0.0 for value in result.values()):
        raise RuntimeError("invalid numerical profile")
    return result


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1184 is already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("audit script must exist before registration")
    signatures = {task.name: task_signature(task.name) for task in TASK_SPECS}
    if len({item["table_digest"] for item in signatures.values()}) != len(TASK_SPECS):
        raise RuntimeError("task tables must be unique")
    phase1183_stop = p1183.OUT_ROOT / "analysis/final_stop.json"
    protocol = {
        "phase": PHASE,
        "registered_at_utc": utc_now(),
        "scientific_object": (
            "Independent qualification of the finite-precision applicability domain for the declared "
            "signed-permutation gauge on fresh RoleSquare networks; no mechanism-camera outcome is measured."
        ),
        "independence": {
            "phase1183_registry_reopened": False,
            "phase1183_scientific_results_read": False,
            "new_tasks": True,
            "new_seed_namespace": True,
            "discovery_defines_domain_only": True,
            "confirmation_is_untouched_until_domain_seal": True,
        },
        "claim_scope": {
            "allowed": (
                "Numerical replay qualification only for width-128 modulus-31 RoleSquare networks trained "
                "under the frozen optimizer and lying inside the sealed safety domain."
            ),
            "excluded": [
                "A pass does not confirm K165 or any mechanism camera.",
                "A pass does not establish a complete functional-equivalence group.",
                "The restricted output-layer backward witness is not full-network backward error.",
                "Adversarial stress outcomes do not gate the natural-domain claim.",
                "No claim transfers directly to Transformers, language models, or brains.",
            ],
        },
        "dimensions": {
            "modulus": MODULUS,
            "width": WIDTH,
            "replicates_per_task": REPLICATES,
            "train_fraction": TRAIN_FRACTION,
            "gauge_transforms_per_qualified_confirmation_system": GAUGE_TRANSFORMS,
        },
        "tasks": [asdict(task) | {"signature": signatures[task.name]} for task in TASK_SPECS],
        "training": TRAINING,
        "domain_definition": {
            "profile_metrics": list(PROFILE_METRICS),
            "natural_support": "componentwise min/max over behavior-qualified discovery systems",
            "safety_domain": f"componentwise [natural_min/{SAFETY_MULTIPLIER}, natural_max*{SAFETY_MULTIPLIER}]",
            "confirmation_gate": "fraction of qualified confirmation systems inside every safety bound",
            "stress_tier": "weight x100 plus duplicated half channels; descriptive and non-gating",
        },
        "error_contract": {
            "absolute_forward": "max |z_g-z|; gate uses absolute_floor + relative_multiplier*max|z|",
            "scaled_forward": "max |z_g-z| / max(max|z|, tiny)",
            "rms_relative": "RMS(z_g-z) / max(RMS(z), tiny)",
            "decision": "argmax and correct-margin-sign agreement",
            "restricted_backward_witness": (
                "minimum-norm output-layer perturbation against the original squared-hidden design; descriptive only"
            ),
        },
        "thresholds": THRESHOLDS,
        "sequential_gates": [
            "fresh_discovery_training_seal",
            "discovery_behavior_qualification_and_domain_seal",
            "fresh_confirmation_training_seal",
            "confirmation_behavior_and_domain_coverage",
            "algebraic_and_finite_precision_gauge_replay",
            "broken_compensation_positive_control",
        ],
        "failure_action": (
            "Stop at the first failed sequential gate. Do not edit thresholds, bounds, task list, precision, "
            "or stress-tier status. A failure is numerical-domain evidence, not mechanism evidence."
        ),
        "auto_continue_rule": (
            "Only a fully passing Phase1184 authorizes one separately preregistered fresh three-evidence "
            "mechanism confirmation. Otherwise stop."
        ),
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1171_source": file_sha256(Path(p1171.__file__)),
            "phase1181_source": file_sha256(Path(p1181.__file__)),
            "phase1183_source": file_sha256(Path(p1183.__file__)),
            "phase1183_stop": file_sha256(phase1183_stop),
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
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1184 is not preregistered")
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
        "phase1183_stop": p1183.OUT_ROOT / "analysis/final_stop.json",
    }
    for name, path in paths.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen source changed: {name}")
    return protocol


def train_split(split: str) -> None:
    protocol = validate_protocol()
    if split == "confirmation":
        if not DOMAIN_PATH.exists() or not read_json(DOMAIN_PATH)["domain_seal_pass"]:
            raise RuntimeError("confirmation denied before passing discovery domain seal")
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
            train_result = accuracy(model, data["train_x"], data["train_y"], device)
            rows.append(
                {
                    "task_name": task.name,
                    "task_index": task_index,
                    "replicate": replicate,
                    "seed": seed,
                    "endpoint_train_accuracy": train_result["accuracy"],
                    "all_train_logits_finite": train_result["all_logits_finite"],
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
    seal_path = OUT_ROOT / "runs" / split / "training_seal.json"
    seal = read_json(seal_path)
    stored = seal.pop("seal_digest")
    if digest(seal) != stored:
        raise RuntimeError(f"{split} training seal digest mismatch")
    seal["seal_digest"] = stored
    if seal["protocol_digest"] != protocol_digest or not seal["all_holdout_labels_unread"]:
        raise RuntimeError(f"invalid {split} training seal")
    metrics_path = OUT_ROOT / "runs" / split / "training_metrics.jsonl"
    if file_sha256(metrics_path) != seal["training_metrics_sha256"]:
        raise RuntimeError(f"{split} training metrics changed")
    for name, expected in seal["checkpoint_hashes"].items():
        path = OUT_ROOT / "runs" / split / "checkpoints" / name
        if file_sha256(path) != expected:
            raise RuntimeError(f"checkpoint changed: {name}")
    return seal


def build_behavior_profile_rows(split: str, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((OUT_ROOT / "runs" / split / "checkpoints").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        data = make_data(payload["task_name"], payload["seed"] + 17)
        train_result = accuracy(model, data["train_x"], data["train_y"], device)
        holdout_result = accuracy(model, data["holdout_x"], data["holdout_y"], device)
        profile = numerical_profile(model, data["all_x"], data["all_y"], device)
        qualified = bool(
            train_result["all_logits_finite"]
            and holdout_result["all_logits_finite"]
            and train_result["accuracy"] >= THRESHOLDS["train_accuracy_min"]
            and holdout_result["accuracy"] >= THRESHOLDS["holdout_accuracy_min"]
        )
        rows.append(
            {
                "checkpoint": path.name,
                "checkpoint_sha256": file_sha256(path),
                "task_name": payload["task_name"],
                "task_index": payload["task_index"],
                "replicate": payload["replicate"],
                "seed": payload["seed"],
                "train_accuracy": train_result["accuracy"],
                "holdout_accuracy": holdout_result["accuracy"],
                "all_logits_finite": bool(train_result["all_logits_finite"] and holdout_result["all_logits_finite"]),
                "qualified": qualified,
                "profile": profile,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def task_qualification(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for task in split_tasks(split):
        selected = [row for row in rows if row["task_name"] == task.name]
        qualified = [row for row in selected if row["qualified"]]
        summaries[task.name] = {
            "system_count": len(selected),
            "qualified_system_count": len(qualified),
            "minimum_train_accuracy": min(row["train_accuracy"] for row in selected),
            "minimum_holdout_accuracy": min(row["holdout_accuracy"] for row in selected),
            "task_pass": len(qualified) >= THRESHOLDS["qualified_system_count_per_task_min"],
        }
    return summaries


def seal_discovery_domain() -> None:
    protocol = validate_protocol()
    if DOMAIN_PATH.exists():
        raise RuntimeError("discovery domain already sealed")
    verify_training_seal("discovery", protocol["protocol_digest"])
    device = torch.device("cuda")
    rows = build_behavior_profile_rows("discovery", device)
    rows_path = OUT_ROOT / "runs/discovery/systems.jsonl"
    write_jsonl(rows_path, rows)
    qualified = [row for row in rows if row["qualified"]]
    task_summaries = task_qualification(rows, "discovery")
    passing_tasks = sum(int(item["task_pass"]) for item in task_summaries.values())
    behavior_pass = bool(
        passing_tasks >= THRESHOLDS["passing_task_count_per_split_min"]
        and len(qualified) >= THRESHOLDS["qualified_system_count_per_split_min"]
    )
    bounds: dict[str, Any] = {}
    if behavior_pass:
        for metric in PROFILE_METRICS:
            values = [row["profile"][metric] for row in qualified]
            lower, upper = min(values), max(values)
            bounds[metric] = {
                "natural_min": lower,
                "natural_max": upper,
                "safety_min": lower / SAFETY_MULTIPLIER,
                "safety_max": upper * SAFETY_MULTIPLIER,
            }
    seal = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(rows),
        "qualified_system_count": len(qualified),
        "passing_task_count": passing_tasks,
        "task_summaries": task_summaries,
        "behavior_pass": behavior_pass,
        "bounds": bounds,
        "rows_sha256": file_sha256(rows_path),
        "domain_seal_pass": behavior_pass and len(bounds) == len(PROFILE_METRICS),
    }
    seal["domain_seal_digest"] = digest(seal)
    write_json(DOMAIN_PATH, seal)
    print(canonical_json(seal))


def in_safety_domain(profile: dict[str, float], bounds: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = [
        metric
        for metric in PROFILE_METRICS
        if not (bounds[metric]["safety_min"] <= profile[metric] <= bounds[metric]["safety_max"])
    ]
    return len(failures) == 0, failures


def forward_metrics(reference: np.ndarray, changed: np.ndarray, unit: float, precision: str) -> dict[str, float]:
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    scale_max = max(float(np.max(np.abs(reference))), 1e-30)
    scale_rms = max(positive_rms(reference), 1e-30)
    multiplier = 128.0 if precision == "fp64" else 32.0
    absolute_floor = (256.0 if precision == "fp64" else 128.0) * unit
    absolute_max = float(np.max(np.abs(delta)))
    return {
        "absolute_max": absolute_max,
        "reference_max_abs": scale_max,
        "scaled_max": absolute_max / scale_max,
        "rms_relative": positive_rms(delta) / scale_rms,
        "mixed_absolute_bound": absolute_floor + multiplier * unit * scale_max,
    }


def gauge_row(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: torch.device,
    pinv_design: np.ndarray,
    design: np.ndarray,
) -> dict[str, Any]:
    transformed = p1183.gauge_model(model, seed, device)
    original_cuda, original_hidden = p1181.fp32_state(model, x, device)
    changed_cuda, _ = p1181.fp32_state(transformed, x, device)
    _, original_fp64 = p1183.cpu_hidden_and_logits(model, x)
    _, changed_fp64 = p1183.cpu_hidden_and_logits(transformed, x)
    original_feature = np.asarray(p1183.algebraic_internal_features(model, x), dtype=np.float64)
    changed_feature = np.asarray(p1183.algebraic_internal_features(transformed, x), dtype=np.float64)
    fp32_reference = original_cuda.detach().cpu().double().numpy()
    fp32_changed = changed_cuda.detach().cpu().double().numpy()
    fp32 = forward_metrics(fp32_reference, fp32_changed, FP32_U, "fp32")
    fp64 = forward_metrics(original_fp64, changed_fp64, FP64_U, "fp64")
    predictions_reference = original_cuda.argmax(dim=1)
    predictions_changed = changed_cuda.argmax(dim=1)
    margin_reference = p1181.correct_margin(original_cuda, y.to(device))
    margin_changed = p1181.correct_margin(changed_cuda, y.to(device))
    delta = fp32_changed - fp32_reference
    delta_w_t = pinv_design @ delta
    reconstructed = design @ delta_w_t
    backward_residual = positive_rms(delta - reconstructed) / max(positive_rms(delta), 1e-30)
    output_norm = float(np.linalg.norm(model.output.weight.detach().cpu().double().numpy()))
    feature_error = float(np.max(np.abs(original_feature - changed_feature)))
    decision_agreement = float((predictions_reference == predictions_changed).float().mean().item())
    margin_sign_agreement = float(((margin_reference >= 0) == (margin_changed >= 0)).float().mean().item())
    valid = bool(
        feature_error <= THRESHOLDS["algebraic_feature_max_error_max"]
        and fp64["absolute_max"] <= fp64["mixed_absolute_bound"]
        and fp64["scaled_max"] <= THRESHOLDS["fp64_scaled_forward_error_max"]
        and fp64["rms_relative"] <= THRESHOLDS["fp64_scaled_forward_error_max"]
        and fp32["absolute_max"] <= fp32["mixed_absolute_bound"]
        and fp32["scaled_max"] <= THRESHOLDS["fp32_scaled_forward_error_max"]
        and fp32["rms_relative"] <= THRESHOLDS["fp32_scaled_forward_error_max"]
        and decision_agreement >= THRESHOLDS["decision_agreement_min"]
        and margin_sign_agreement >= THRESHOLDS["margin_sign_agreement_min"]
    )
    result = {
        "seed": seed,
        "feature_error": feature_error,
        "fp64": fp64,
        "fp32": fp32,
        "decision_agreement": decision_agreement,
        "margin_sign_agreement": margin_sign_agreement,
        "restricted_output_backward_witness": {
            "relative_output_weight_norm": float(np.linalg.norm(delta_w_t.T) / max(output_norm, 1e-30)),
            "relative_reconstruction_residual": backward_residual,
            "claim_status": "descriptive_not_full_network_backward_error",
        },
        "gauge_pass": valid,
    }
    del transformed, original_hidden
    return result


def positive_control_row(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    broken = p1183.gauge_model(model, seed, device, broken_output=True)
    original_logits, _ = p1181.fp32_state(model, x, device)
    broken_logits, _ = p1181.fp32_state(broken, x, device)
    original_feature = np.asarray(p1183.algebraic_internal_features(model, x), dtype=np.float64)
    broken_feature = np.asarray(p1183.algebraic_internal_features(broken, x), dtype=np.float64)
    metrics = forward_metrics(
        original_logits.detach().cpu().double().numpy(),
        broken_logits.detach().cpu().double().numpy(),
        FP32_U,
        "fp32",
    )
    decision = float((original_logits.argmax(dim=1) == broken_logits.argmax(dim=1)).float().mean().item())
    feature_error = float(np.max(np.abs(original_feature - broken_feature)))
    passed = bool(
        feature_error >= THRESHOLDS["positive_feature_error_min"]
        and metrics["scaled_max"] >= THRESHOLDS["positive_scaled_forward_error_min"]
        and decision <= THRESHOLDS["positive_decision_agreement_max"]
    )
    del broken
    return {
        "seed": seed,
        "feature_error": feature_error,
        "fp32": metrics,
        "decision_agreement": decision,
        "positive_control_pass": passed,
    }


def stress_copy(model: p1171.RoleSquareNetwork, device: torch.device) -> p1171.RoleSquareNetwork:
    stressed = p1171.RoleSquareNetwork(model.config).to(device)
    stressed.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in stressed.parameters():
            parameter.mul_(100.0)
        half = stressed.config.width // 2
        stressed.hidden.weight[half:].copy_(stressed.hidden.weight[:half])
        stressed.output.weight[:, half:].copy_(stressed.output.weight[:, :half])
    stressed.eval()
    return stressed


def stress_tier(qualified_rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for task in split_tasks("confirmation"):
        candidates = [row for row in qualified_rows if row["task_name"] == task.name]
        if candidates:
            selected.append(candidates[0])
    stress_rows: list[dict[str, Any]] = []
    for row in selected:
        path = OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        base_model = load_model(payload, device)
        model = stress_copy(base_model, device)
        data = make_data(payload["task_name"], payload["seed"] + 17)
        x = data["all_x"]
        reference, _ = p1181.fp32_state(model, x, device)
        reference_np = reference.detach().cpu().double().numpy()
        for index in range(GAUGE_TRANSFORMS):
            transformed = p1183.gauge_model(model, 11849000 + payload["task_index"] * 100 + index, device)
            changed, _ = p1181.fp32_state(transformed, x, device)
            changed_np = changed.detach().cpu().double().numpy()
            metrics = forward_metrics(reference_np, changed_np, FP32_U, "fp32")
            stress_rows.append(
                {
                    "task_name": payload["task_name"],
                    "replicate": payload["replicate"],
                    "transform": index,
                    "fp32": metrics,
                    "decision_agreement": float((reference.argmax(1) == changed.argmax(1)).float().mean().item()),
                    "all_finite": bool(torch.isfinite(reference).all().item() and torch.isfinite(changed).all().item()),
                }
            )
            del transformed
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "status": "descriptive_non_gating",
        "system_count": len(selected),
        "row_count": len(stress_rows),
        "maximum_absolute_error": max((row["fp32"]["absolute_max"] for row in stress_rows), default=None),
        "maximum_scaled_error": max((row["fp32"]["scaled_max"] for row in stress_rows), default=None),
        "minimum_decision_agreement": min((row["decision_agreement"] for row in stress_rows), default=None),
        "all_finite": all(row["all_finite"] for row in stress_rows),
        "rows": stress_rows,
    }


def evaluate_confirmation() -> None:
    protocol = validate_protocol()
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1184 final already exists")
    domain = read_json(DOMAIN_PATH)
    if not domain["domain_seal_pass"]:
        raise RuntimeError("discovery domain did not pass")
    verify_training_seal("confirmation", protocol["protocol_digest"])
    device = torch.device("cuda")
    rows = build_behavior_profile_rows("confirmation", device)
    for row in rows:
        row["inside_safety_domain"], row["safety_domain_failures"] = in_safety_domain(row["profile"], domain["bounds"])
    systems_path = OUT_ROOT / "runs/confirmation/systems.jsonl"
    write_jsonl(systems_path, rows)
    qualified = [row for row in rows if row["qualified"]]
    task_summaries = task_qualification(rows, "confirmation")
    passing_tasks = sum(int(item["task_pass"]) for item in task_summaries.values())
    behavior_pass = bool(
        passing_tasks >= THRESHOLDS["passing_task_count_per_split_min"]
        and len(qualified) >= THRESHOLDS["qualified_system_count_per_split_min"]
    )
    safety_fraction = sum(int(row["inside_safety_domain"]) for row in qualified) / max(len(qualified), 1)
    domain_coverage_pass = bool(safety_fraction >= THRESHOLDS["confirmation_safety_domain_fraction_min"])

    gauge_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    if behavior_pass and domain_coverage_pass:
        for row in qualified:
            path = OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = load_model(payload, device)
            data = make_data(payload["task_name"], payload["seed"] + 17)
            _, hidden = p1181.fp32_state(model, data["all_x"], device)
            design = hidden.detach().cpu().double().numpy() ** 2
            pinv_design = np.linalg.pinv(design, rcond=1e-10)
            for transform_index in range(GAUGE_TRANSFORMS):
                result = gauge_row(
                    model,
                    data["all_x"],
                    data["all_y"],
                    11848000 + payload["task_index"] * 10_000 + payload["replicate"] * 100 + transform_index,
                    device,
                    pinv_design,
                    design,
                )
                gauge_rows.append(
                    {
                        "checkpoint": row["checkpoint"],
                        "task_name": row["task_name"],
                        "replicate": row["replicate"],
                        "transform": transform_index,
                        **result,
                    }
                )
            positive_rows.append(
                {
                    "checkpoint": row["checkpoint"],
                    "task_name": row["task_name"],
                    "replicate": row["replicate"],
                    **positive_control_row(
                        model,
                        data["all_x"],
                        11848500 + payload["task_index"] * 100 + payload["replicate"],
                        device,
                    ),
                }
            )
            print(canonical_json({"gauged": row["task_name"], "replicate": row["replicate"]}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    gauge_path = OUT_ROOT / "analysis/gauge_rows.jsonl"
    positive_path = OUT_ROOT / "analysis/positive_control_rows.jsonl"
    write_jsonl(gauge_path, gauge_rows)
    write_jsonl(positive_path, positive_rows)
    valid_gauge_fraction = sum(int(row["gauge_pass"]) for row in gauge_rows) / max(len(gauge_rows), 1)
    positive_fraction = sum(int(row["positive_control_pass"]) for row in positive_rows) / max(len(positive_rows), 1)
    gauge_pass = bool(
        len(gauge_rows) == len(qualified) * GAUGE_TRANSFORMS
        and valid_gauge_fraction >= THRESHOLDS["valid_gauge_fraction_min"]
    )
    positive_pass = bool(
        len(positive_rows) == len(qualified)
        and positive_fraction >= THRESHOLDS["positive_control_fraction_min"]
    )
    stress = stress_tier(qualified, device) if behavior_pass else {"status": "not_run_after_behavior_failure"}
    primary_pass = bool(behavior_pass and domain_coverage_pass and gauge_pass and positive_pass)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_object": "finite_precision_numerical_domain_qualification_only",
        "discovery_domain_seal_digest": domain["domain_seal_digest"],
        "confirmation": {
            "system_count": len(rows),
            "qualified_system_count": len(qualified),
            "passing_task_count": passing_tasks,
            "task_summaries": task_summaries,
            "behavior_pass": behavior_pass,
            "safety_domain_fraction": safety_fraction,
            "domain_coverage_pass": domain_coverage_pass,
        },
        "gauge": {
            "row_count": len(gauge_rows),
            "valid_fraction": valid_gauge_fraction,
            "maximum_feature_error": max((row["feature_error"] for row in gauge_rows), default=None),
            "maximum_fp64_absolute_error": max((row["fp64"]["absolute_max"] for row in gauge_rows), default=None),
            "maximum_fp64_scaled_error": max((row["fp64"]["scaled_max"] for row in gauge_rows), default=None),
            "maximum_fp32_absolute_error": max((row["fp32"]["absolute_max"] for row in gauge_rows), default=None),
            "maximum_fp32_scaled_error": max((row["fp32"]["scaled_max"] for row in gauge_rows), default=None),
            "maximum_fp32_rms_relative": max((row["fp32"]["rms_relative"] for row in gauge_rows), default=None),
            "minimum_decision_agreement": min((row["decision_agreement"] for row in gauge_rows), default=None),
            "minimum_margin_sign_agreement": min((row["margin_sign_agreement"] for row in gauge_rows), default=None),
            "maximum_restricted_backward_relative_norm": max(
                (row["restricted_output_backward_witness"]["relative_output_weight_norm"] for row in gauge_rows),
                default=None,
            ),
            "maximum_restricted_backward_residual": max(
                (row["restricted_output_backward_witness"]["relative_reconstruction_residual"] for row in gauge_rows),
                default=None,
            ),
            "gauge_pass": gauge_pass,
            "rows_sha256": file_sha256(gauge_path),
        },
        "positive_control": {
            "row_count": len(positive_rows),
            "valid_fraction": positive_fraction,
            "minimum_feature_error": min((row["feature_error"] for row in positive_rows), default=None),
            "minimum_scaled_forward_error": min((row["fp32"]["scaled_max"] for row in positive_rows), default=None),
            "maximum_decision_agreement": max((row["decision_agreement"] for row in positive_rows), default=None),
            "positive_control_pass": positive_pass,
            "rows_sha256": file_sha256(positive_path),
        },
        "stress_tier": stress,
        "primary_pass": primary_pass,
        "mechanism_camera_status": "not_tested_by_design",
        "phase1183_status": "unchanged_frozen_failure",
        "auto_continue": {
            "authorized": primary_pass,
            "next": "one_separately_preregistered_fresh_three_evidence_mechanism_confirmation" if primary_pass else None,
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
            "train-discovery",
            "seal-discovery-domain",
            "train-confirmation",
            "evaluate-confirmation",
        ),
    )
    args = parser.parse_args()
    commands = {
        "preregister": preregister,
        "train-discovery": lambda: train_split("discovery"),
        "seal-discovery-domain": seal_discovery_domain,
        "train-confirmation": lambda: train_split("confirmation"),
        "evaluate-confirmation": evaluate_confirmation,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
