#!/usr/bin/env python3
"""One-shot causal fork for the source of late affine relation regularization.

Phase1174 found that an affine relation-transfer/closure camera had endpoint
external validity but usually appeared long after behavioral generalization.
This phase does not reopen the failed formation-prediction branch.  Instead,
each all-new trajectory is trained to a fixed branch point and cloned into
three optimizer regimes: continued AdamW decay, decay disabled, and decay
disabled with global parameter norm matched to the continued-decay arm.

The sole primary question is whether continued weight decay selectively
promotes the late affine camera while behavior remains matched.  No nonlinear
camera search, hidden-feature selection, or relation causal-use claim is
authorized by this experiment.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402
import phase1174_training_inferred_relation_event_prediction as p1174  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1175_late_relation_weight_decay_fork_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1175_late_relation_weight_decay_fork"
P1174_FINAL = ROOT / "tests/glm5/result/phase1174_training_inferred_relation_event_prediction/analysis/final.json"
P1174_AUDIT = ROOT / "tests/glm5/result/phase1174_training_inferred_relation_event_prediction/audit/independent_audit.json"
MATERIAL_PROBE = ROOT / "tests/glm5_temp/phase1175_weight_decay_task_probe.json"

PHASE = 1175
MODULUS = 61
MODEL_WIDTH = 128
TRAIN_FRACTION = 0.50
REPLICATES = 8
BRANCH_STEP = 2500
MAX_STEP = 12000
ARMS = ("continued_decay", "decay_off", "norm_matched_no_decay")
ARM_WEIGHT_DECAY = {
    "continued_decay": 1.0,
    "decay_off": 0.0,
    "norm_matched_no_decay": 0.0,
}
PARENT_CHECKPOINT_STEPS = (
    25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000,
    1250, 1500, 1750, 2000, 2250, 2500,
)
ARM_CHECKPOINT_STEPS = (
    2750, 3000, 3500, 4000, 4500, 5000, 5500,
    6000, 7000, 8000, 9000, 10000, 12000,
)
LATE_ENDPOINT_STEPS = (9000, 10000, 12000)
TRAINING = {
    "learning_rate": 0.001,
    "parent_weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch_deterministic",
    "branch_step": BRANCH_STEP,
    "maximum_step": MAX_STEP,
}
THRESHOLDS = {
    "train_fit_accuracy_min": 0.99,
    "stable_generalization_accuracy_min": 0.90,
    "stable_adjacent_checkpoint_count": 2,
    "camera_score_min": 0.15,
    "camera_advantage_min": 0.10,
    "camera_adjacent_checkpoint_count": 2,
    "minimum_branch_quiet_behavior_matched_trajectory_count": 24,
    "minimum_branch_quiet_behavior_match_fraction": 0.75,
    "minimum_task_class_breadth": 4,
    "maximum_median_final_holdout_gap": 0.02,
    "maximum_median_generalization_time_gap": 500,
    "minimum_median_late_camera_effect": 0.10,
    "minimum_per_class_late_camera_effect": 0.05,
    "minimum_late_camera_effect_class_breadth": 4,
    "minimum_actual_over_random_effect_advantage": 0.08,
    "maximum_norm_match_relative_error": 1.0e-5,
    "norm_control_equivalence_tolerance": 0.05,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    formula: str


TASK_SPECS = (
    TaskSpec("add_power6", "power_fiber_6", "a + b^6 mod 61"),
    TaskSpec("add_power20", "power_fiber_20", "a + b^20 mod 61"),
    TaskSpec("add_poly3", "polynomial_fiber_3", "a + b^3 + 2b mod 61"),
    TaskSpec("add_poly4", "polynomial_fiber_4", "a + b^4 + 3b mod 61"),
    TaskSpec("add_poly5", "polynomial_fiber_5", "a + b^5 + 2b^2 mod 61"),
    TaskSpec("add_factor3", "factorized_cubic_fiber", "a + b(b-1)(b-2) mod 61"),
)
TASK_BY_NAME = {task.name: task for task in TASK_SPECS}


def task_functions() -> dict[str, Callable[[int, int], int]]:
    p = MODULUS
    return {
        "add_power6": lambda a, b: (a + pow(b, 6, p)) % p,
        "add_power20": lambda a, b: (a + pow(b, 20, p)) % p,
        "add_poly3": lambda a, b: (a + pow(b, 3, p) + 2 * b) % p,
        "add_poly4": lambda a, b: (a + pow(b, 4, p) + 3 * b) % p,
        "add_poly5": lambda a, b: (a + pow(b, 5, p) + 2 * b * b) % p,
        "add_factor3": lambda a, b: (a + b * (b - 1) * (b - 2)) % p,
    }


def task_table(task_name: str) -> np.ndarray:
    function = task_functions()[task_name]
    return np.asarray(
        [[function(a, b) for b in range(MODULUS)] for a in range(MODULUS)],
        dtype=np.int64,
    )


def quotient_signature(task_name: str) -> dict[str, Any]:
    table = task_table(task_name)
    payload = p1172.quotient_invariant_payload(table)
    return {
        "digest": base.digest(payload),
        "table_digest": base.digest(table.tolist()),
        "distinct_row_count": payload["distinct_row_count"],
        "distinct_column_count": payload["distinct_column_count"],
        "row_distinct_output_range": [
            min(payload["row_distinct_output_counts"]),
            max(payload["row_distinct_output_counts"]),
        ],
        "column_distinct_output_range": [
            min(payload["column_distinct_output_counts"]),
            max(payload["column_distinct_output_counts"]),
        ],
    }


def model_seed(task_index: int, replicate: int) -> int:
    return 11_750_000 + task_index * 100_003 + replicate * 1_009


def make_data(task_name: str, seed: int) -> dict[str, Any]:
    table = task_table(task_name)
    pairs = np.asarray(
        [(a, b) for a in range(MODULUS) for b in range(MODULUS)],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    train_mask_flat = np.zeros(len(pairs), dtype=bool)
    train_mask_flat[order[: int(round(len(pairs) * TRAIN_FRACTION))]] = True
    backgrounds = rng.permutation(MODULUS)
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(table.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(train_mask_flat, dtype=torch.bool)
    return {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "holdout_x": x[~mask_t],
        "holdout_y": y[~mask_t],
        "train_mask": train_mask_flat.reshape(MODULUS, MODULUS),
        "contexts": {
            "key": backgrounds[: p1174.KEY_CONTEXT_COUNT].astype(np.int64),
            "fit": backgrounds[
                p1174.KEY_CONTEXT_COUNT : p1174.KEY_CONTEXT_COUNT + p1174.FIT_CONTEXT_COUNT
            ].astype(np.int64),
            "test": backgrounds[
                p1174.KEY_CONTEXT_COUNT + p1174.FIT_CONTEXT_COUNT :
            ].astype(np.int64),
        },
    }


def tensor_state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def optimizer_moment_digest(optimizer: torch.optim.Optimizer) -> str:
    payload = optimizer.state_dict()["state"]
    digest = hashlib.sha256()
    for parameter_id in sorted(payload):
        digest.update(str(parameter_id).encode("ascii"))
        for name in sorted(payload[parameter_id]):
            value = payload[parameter_id][name]
            digest.update(name.encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def parameter_l2_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(parameter.detach().float().square().sum().item())
        for parameter in model.parameters()
    ))


def gradient_l2_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    ))


def rescale_to_norm(model: torch.nn.Module, target_norm: float) -> float:
    current = parameter_l2_norm(model)
    if current <= 0.0:
        raise RuntimeError("cannot norm-match a zero-parameter model")
    scale = target_norm / current
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)
    return abs(parameter_l2_norm(model) - target_norm) / max(target_norm, 1.0e-12)


def material_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_signatures = {
        f"1172:{task.name}": p1172.quotient_signature(task.name)["digest"]
        for task in p1172.TASK_SPECS
    }
    old_signatures.update({
        f"1174:{task.name}": p1174.quotient_signature(task.name)["digest"]
        for task in p1174.TASK_SPECS
    })
    old_digests = set(old_signatures.values())
    signatures = {task.name: quotient_signature(task.name) for task in TASK_SPECS}
    manifest = []
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = make_data(task.name, seed + 17)
            relation_key = p1174.infer_relation_key(data)
            manifest.append({
                "trajectory_id": f"{task.name}_r{replicate}_s{seed}",
                "task_name": task.name,
                "task_index": task_index,
                "replicate": replicate,
                "seed": seed,
                "relation_key": relation_key,
                "train_pair_digest": base.digest(data["train_x"].tolist()),
                "train_label_digest": base.digest(data["train_y"].tolist()),
                "context_digest": base.digest({
                    name: values.tolist() for name, values in data["contexts"].items()
                }),
            })
    random_nulls = [p1174.random_relation_null(11_759_000 + index * 103) for index in range(16)]
    probe = base.read_json(MATERIAL_PROBE)
    checks = {
        "candidate_signatures_unique": len({
            value["digest"] for value in signatures.values()
        }) == len(TASK_SPECS),
        "no_phase1172_or_1174_collision": all(
            value["digest"] not in old_digests for value in signatures.values()
        ),
        "all_training_only_relation_keys_identified": all(
            row["relation_key"]["eligible_count"] == 3 for row in manifest
        ),
        "random_table_nulls_abstain": all(
            key["eligible_count"] == 0 for key in random_nulls
        ),
        "probe_scope_is_zero_training": probe["scope"].startswith("zero-training"),
        "formal_tasks_pass_probe": all(
            task.name in probe["eligible_no_collision_names"] for task in TASK_SPECS
        ),
        "relation_keys_are_training_only": all(
            not row["relation_key"]["uses_holdout_inputs"]
            and not row["relation_key"]["uses_holdout_labels"]
            and not row["relation_key"]["uses_task_name_or_formula"]
            and not row["relation_key"]["uses_future_generalization"]
            for row in manifest
        ),
    }
    return manifest, {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "signatures": signatures,
        "old_signatures": old_signatures,
        "random_null_eligible_counts": [key["eligible_count"] for key in random_nulls],
        "material_probe_digest": probe["digest"],
        "material_probe_sha256": base.sha256_file(MATERIAL_PROBE),
    }


def protocol_command() -> None:
    path = OUT_ROOT / "protocol/preregistration.json"
    if path.exists():
        raise RuntimeError("Phase1175 protocol already exists")
    prior = base.read_json(P1174_FINAL)
    prior_audit = base.read_json(P1174_AUDIT)
    if prior["decision"]["primary_endpoint_pass"]:
        raise RuntimeError("Phase1174 failure boundary changed")
    if not prior["decision"]["free_network_endpoint_camera_externality"]:
        raise RuntimeError("Phase1174 endpoint externality boundary changed")
    if prior["decision"]["nonlinear_camera_search_authorized"]:
        raise RuntimeError("Phase1174 nonlinear-camera hard stop changed")
    if not prior_audit["passed"] or prior_audit["passed_count"] != 46:
        raise RuntimeError("Phase1174 independent audit boundary mismatch")
    manifest, material = material_manifest()
    if not material["pass"]:
        raise RuntimeError(f"material gate failed: {material['checks']}")
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": base.utc_now(),
        "title": "One-shot post-formation weight-decay fork for late affine relation regularization",
        "script_sha256": base.sha256_file(SCRIPT),
        "audit_script_sha256": base.sha256_file(AUDIT_SCRIPT),
        "prior_phase1174_final_digest": prior["final_digest"],
        "prior_phase1174_audit_digest": prior_audit["audit_digest"],
        "hypothesis_source": (
            "Phase1174 prospectively observed median behavioral generalization near step 200 and "
            "median affine-camera onset near step 7000; the present causal source hypothesis was "
            "registered before any Phase1175 formal trajectory was trained."
        ),
        "separation_from_closed_branch": {
            "formation_prediction_reopened": False,
            "nonlinear_camera_search": False,
            "relation_causal_use_test": False,
            "question": "Does continued AdamW decay promote the already-calibrated late affine normal form?",
        },
        "task_specs": [
            asdict(task) | {"quotient_signature": material["signatures"][task.name]}
            for task in TASK_SPECS
        ],
        "manifest": manifest,
        "material_gate": material,
        "task_count": len(TASK_SPECS),
        "parent_trajectory_count": len(TASK_SPECS) * REPLICATES,
        "arm_trajectory_count": len(TASK_SPECS) * REPLICATES * len(ARMS),
        "replicates": REPLICATES,
        "model": {
            "architecture": "RoleSquareNetwork",
            "modulus": MODULUS,
            "width": MODEL_WIDTH,
            "parameter_count": 39_808,
        },
        "training": TRAINING,
        "branch": {
            "step": BRANCH_STEP,
            "parent_regime": "AdamW weight_decay=1.0",
            "arms": {
                "continued_decay": "clone parent model and optimizer; continue AdamW weight_decay=1.0",
                "decay_off": "clone parent model and optimizer; set AdamW weight_decay=0.0",
                "norm_matched_no_decay": (
                    "clone parent model and optimizer; set AdamW weight_decay=0.0; after every step, "
                    "globally rescale parameters to the continued-decay arm's L2 norm; optimizer "
                    "moments are not rescaled"
                ),
            },
            "identical_model_and_optimizer_moments_required": True,
            "holdout_blind_at_branch": True,
        },
        "parent_checkpoint_steps": PARENT_CHECKPOINT_STEPS,
        "arm_checkpoint_steps": ARM_CHECKPOINT_STEPS,
        "late_endpoint_steps": LATE_ENDPOINT_STEPS,
        "camera": {
            "implementation": "exact frozen Phase1174 training-inferred relation camera",
            "map_family": "affine ridge only",
            "whitening": "operator-fit backgrounds only",
            "random_pairing_control": True,
            "camera_search": False,
        },
        "thresholds": THRESHOLDS,
        "primary_endpoint": (
            "Among branch-quiet trajectories with all three arms behavior-matched, continued decay "
            "must exceed decay-off late affine-camera score by median >=0.10, by >=0.05 in at least "
            "4/6 task classes, and by >=0.08 beyond the matched random-pairing arm effect. At least "
            "24 trajectories across 4 task classes and >=75% of branch-quiet trajectories must be "
            "behavior-matched; final behavior and generalization time must remain within frozen gaps."
        ),
        "secondary_endpoint": (
            "The norm-matched no-decay arm distinguishes whether equality of global parameter norm "
            "is sufficient to recover the continued-decay camera trajectory."
        ),
        "hard_stops": [
            "A behavior mismatch makes the late-regularization causal test inconclusive, not positive.",
            "Failure closes this weight-decay source hypothesis; do not tune branch time, decay dose, or camera.",
            "Success supports only a source of late affine regularization, not the driver of early generalization.",
            "No nonlinear camera, hidden-feature search, relation causal-use intervention, or natural-language claim.",
            "Auto-continuation is false regardless of outcome; an early mechanism requires an independent known-truth hypothesis.",
        ],
        "claim_scope": "Controlled free RoleSquareNetwork trajectories and this optimizer intervention only.",
    }
    protocol["protocol_digest"] = base.digest(protocol)
    base.write_json(path, protocol)
    print(json.dumps({
        "protocol_digest": protocol["protocol_digest"],
        "material_gate_pass": material["pass"],
        "tasks": len(TASK_SPECS),
        "parent_trajectories": protocol["parent_trajectory_count"],
        "arm_trajectories": protocol["arm_trajectory_count"],
    }))


def smoke_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if not protocol["material_gate"]["pass"]:
        raise RuntimeError("material gate is not closed")
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH))
    if sum(parameter.numel() for parameter in model.parameters()) != 39_808:
        raise RuntimeError("parameter count mismatch")
    state = copy.deepcopy(model.state_dict())
    clones = []
    for _ in ARMS:
        clone = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH))
        clone.load_state_dict(state)
        clones.append(clone)
    digests = [tensor_state_digest(clone.state_dict()) for clone in clones]
    if len(set(digests)) != 1:
        raise RuntimeError("branch clone identity smoke failed")
    for task_index, task in enumerate(TASK_SPECS):
        data = make_data(task.name, model_seed(task_index, 0) + 17)
        if len(data["train_x"]) != 1860 or len(data["holdout_x"]) != 1861:
            raise RuntimeError(f"split mismatch: {task.name}")
        if p1174.infer_relation_key(data)["eligible_count"] != 3:
            raise RuntimeError(f"relation-key mismatch: {task.name}")
    print(json.dumps({
        "smoke_pass": True,
        "task_count": len(TASK_SPECS),
        "parameter_count": 39_808,
        "clone_identity": True,
    }))


def checkpoint_payload(
    model: p1171.RoleSquareNetwork,
    task: TaskSpec,
    task_index: int,
    replicate: int,
    seed: int,
    step: int,
    arm: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task.name,
        "task_index": task_index,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "arm": arm,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> p1171.RoleSquareNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def train_step(
    model: p1171.RoleSquareNetwork,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    trajectory_label: str,
    step: int,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(train_x).float()
        loss = F.cross_entropy(logits, train_y)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"nonfinite loss: {trajectory_label}/{step}")
    loss.backward()
    grad_norm = gradient_l2_norm(model)
    optimizer.step()
    return float(loss.item()), grad_norm


def record_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_hashes: dict[str, str],
    model: p1171.RoleSquareNetwork,
    data: dict[str, Any],
    relation_key: dict[str, Any],
    task: TaskSpec,
    task_index: int,
    replicate: int,
    seed: int,
    trajectory_id: str,
    step: int,
    arm: str,
    loss: float,
    grad_norm: float,
    device: torch.device,
    norm_match_relative_error: float | None,
) -> None:
    train_metrics = p1171.evaluate(model, data["train_x"], data["train_y"], device)
    structure = p1172.training_only_structure(model, data, device)
    camera = p1174.relation_camera(model, data, relation_key, device, seed + 95_001)
    checkpoint_id = f"{trajectory_id}_{arm}_step{step:05d}"
    checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(model, task, task_index, replicate, seed, step, arm),
        checkpoint_path,
    )
    checkpoint_hash = base.sha256_file(checkpoint_path)
    checkpoint_hashes[checkpoint_id] = checkpoint_hash
    rows.append({
        "trajectory_id": trajectory_id,
        "checkpoint_id": checkpoint_id,
        "task_name": task.name,
        "task_index": task_index,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "arm": arm,
        "loss": loss,
        "gradient_l2_norm": grad_norm,
        "train": train_metrics,
        "training_only_structure": structure,
        "relation_key": relation_key,
        "relation_camera": camera,
        "optimizer_weight_decay": 1.0 if arm == "parent" else ARM_WEIGHT_DECAY[arm],
        "norm_match_relative_error": norm_match_relative_error,
        "train_pair_digest": base.digest(data["train_x"].tolist()),
        "train_label_digest": base.digest(data["train_y"].tolist()),
        "sealed_holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
        "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
        "holdout_evaluated_during_training": False,
        "holdout_used_by_gradient": False,
        "camera_used_holdout": False,
        "checkpoint_sha256": checkpoint_hash,
    })


def make_arm(
    parent_state: dict[str, torch.Tensor],
    parent_optimizer_state: dict[str, Any],
    weight_decay: float,
    device: torch.device,
) -> tuple[p1171.RoleSquareNetwork, torch.optim.AdamW]:
    model = p1171.RoleSquareNetwork(
        p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH)
    ).to(device)
    model.load_state_dict(parent_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=weight_decay
    )
    optimizer.load_state_dict(copy.deepcopy(parent_optimizer_state))
    for group in optimizer.param_groups:
        group["weight_decay"] = weight_decay
    return model, optimizer


def train_and_seal_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout exists before training seal")
    manifest_by_id = {row["trajectory_id"]: row for row in protocol["manifest"]}
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    branch_snapshot_hashes: dict[str, str] = {}
    branch_identity_rows: list[dict[str, Any]] = []
    norm_match_errors: list[float] = []
    for task_index, task in enumerate(TASK_SPECS):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            base.set_seed(seed)
            data = make_data(task.name, seed + 17)
            relation_key = p1174.infer_relation_key(data)
            trajectory_id = f"{task.name}_r{replicate}_s{seed}"
            if relation_key["key_digest"] != manifest_by_id[trajectory_id]["relation_key"]["key_digest"]:
                raise RuntimeError(f"relation-key drift: {trajectory_id}")
            parent = p1171.RoleSquareNetwork(
                p1171.RoleSquareConfig(modulus=MODULUS, width=MODEL_WIDTH)
            ).to(device)
            parent_optimizer = torch.optim.AdamW(
                parent.parameters(),
                lr=TRAINING["learning_rate"],
                weight_decay=TRAINING["parent_weight_decay"],
            )
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
            for step in range(1, BRANCH_STEP + 1):
                loss, grad_norm = train_step(
                    parent, parent_optimizer, train_x, train_y, trajectory_id, step
                )
                if step in PARENT_CHECKPOINT_STEPS:
                    record_checkpoint(
                        rows, checkpoint_hashes, parent, data, relation_key, task,
                        task_index, replicate, seed, trajectory_id, step, "parent",
                        loss, grad_norm, device, None,
                    )

            parent_state = copy.deepcopy(parent.state_dict())
            parent_optimizer_state = copy.deepcopy(parent_optimizer.state_dict())
            arm_models: dict[str, p1171.RoleSquareNetwork] = {}
            arm_optimizers: dict[str, torch.optim.AdamW] = {}
            for arm in ARMS:
                model, optimizer = make_arm(
                    parent_state, parent_optimizer_state, ARM_WEIGHT_DECAY[arm], device
                )
                arm_models[arm] = model
                arm_optimizers[arm] = optimizer
            model_digests = {
                arm: tensor_state_digest(model.state_dict())
                for arm, model in arm_models.items()
            }
            optimizer_digests = {
                arm: optimizer_moment_digest(optimizer)
                for arm, optimizer in arm_optimizers.items()
            }
            branch_exact = len(set(model_digests.values())) == 1 and len(set(optimizer_digests.values())) == 1
            if not branch_exact:
                raise RuntimeError(f"branch identity failed: {trajectory_id}")
            snapshot_root = OUT_ROOT / "runs/training/branch_snapshots"
            snapshot_root.mkdir(parents=True, exist_ok=True)
            for arm in ARMS:
                snapshot_path = snapshot_root / f"{trajectory_id}_{arm}.pt"
                torch.save({
                    "phase": PHASE,
                    "trajectory_id": trajectory_id,
                    "arm": arm,
                    "step": BRANCH_STEP,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in arm_models[arm].state_dict().items()
                    },
                    "optimizer_state_dict": arm_optimizers[arm].state_dict(),
                    "model_digest": model_digests[arm],
                    "optimizer_moment_digest": optimizer_digests[arm],
                    "weight_decay": ARM_WEIGHT_DECAY[arm],
                }, snapshot_path)
                branch_snapshot_hashes[f"{trajectory_id}:{arm}"] = base.sha256_file(snapshot_path)
            branch_identity_rows.append({
                "trajectory_id": trajectory_id,
                "model_digests": model_digests,
                "optimizer_moment_digests": optimizer_digests,
                "exact": branch_exact,
            })
            del parent, parent_optimizer

            last_metrics: dict[str, tuple[float, float]] = {}
            for step in range(BRANCH_STEP + 1, MAX_STEP + 1):
                for arm in ARMS:
                    last_metrics[arm] = train_step(
                        arm_models[arm], arm_optimizers[arm], train_x, train_y,
                        f"{trajectory_id}/{arm}", step,
                    )
                target_norm = parameter_l2_norm(arm_models["continued_decay"])
                norm_error = rescale_to_norm(arm_models["norm_matched_no_decay"], target_norm)
                norm_match_errors.append(norm_error)
                if step not in ARM_CHECKPOINT_STEPS:
                    continue
                for arm in ARMS:
                    loss, grad_norm = last_metrics[arm]
                    record_checkpoint(
                        rows, checkpoint_hashes, arm_models[arm], data, relation_key,
                        task, task_index, replicate, seed, trajectory_id, step, arm,
                        loss, grad_norm, device,
                        norm_error if arm == "norm_matched_no_decay" else None,
                    )
            print(json.dumps({
                "trained": trajectory_id,
                "branch_exact": branch_exact,
                "final_camera": {
                    arm: rows[-len(ARMS) + index]["relation_camera"]["actual"]["score"]
                    for index, arm in enumerate(ARMS)
                },
                "final_norm_match_error": norm_match_errors[-1],
            }), flush=True)
            del arm_models, arm_optimizers, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()

    metrics_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    base.write_jsonl(metrics_path, rows)
    expected_rows = (
        len(TASK_SPECS) * REPLICATES * len(PARENT_CHECKPOINT_STEPS)
        + len(TASK_SPECS) * REPLICATES * len(ARMS) * len(ARM_CHECKPOINT_STEPS)
    )
    seal: dict[str, Any] = {
        "phase": PHASE,
        "sealed_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "parent_trajectory_count": len(TASK_SPECS) * REPLICATES,
        "arm_trajectory_count": len(TASK_SPECS) * REPLICATES * len(ARMS),
        "checkpoint_count": len(rows),
        "expected_checkpoint_count": expected_rows,
        "training_metrics_sha256": base.sha256_file(metrics_path),
        "checkpoint_hashes": checkpoint_hashes,
        "branch_snapshot_hashes": branch_snapshot_hashes,
        "branch_identity_rows": branch_identity_rows,
        "all_branch_models_and_optimizer_moments_exact": all(
            row["exact"] for row in branch_identity_rows
        ),
        "maximum_norm_match_relative_error": max(norm_match_errors),
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "no_holdout_camera": all(not row["camera_used_holdout"] for row in rows),
        "strict_inductive_whitening": all(
            row["relation_camera"]["actual"]["whitening_fit_background_only"]
            and not row["relation_camera"]["actual"]["test_background_used_for_whitening"]
            for row in rows
        ),
        "all_training_logits_exactly_finite": all(row["train"]["exact_all_finite"] for row in rows),
        "checkpoint_count_exact": len(rows) == expected_rows,
        "training_sealed": True,
    }
    seal["seal_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({
        "seal_digest": seal["seal_digest"],
        "parent_trajectories": seal["parent_trajectory_count"],
        "arm_trajectories": seal["arm_trajectory_count"],
        "checkpoints": seal["checkpoint_count"],
        "branch_identity": seal["all_branch_models_and_optimizer_moments_exact"],
        "max_norm_match_error": seal["maximum_norm_match_relative_error"],
    }))


def reveal_holdout_command() -> None:
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    output_root = OUT_ROOT / "runs/holdout"
    if output_root.exists():
        raise RuntimeError("holdout already revealed")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    rows = []
    for training_row in training_rows:
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{training_row['checkpoint_id']}.pt"
        if base.sha256_file(checkpoint_path) != training_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {training_row['checkpoint_id']}")
        data = make_data(training_row["task_name"], training_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
        rows.append({
            "trajectory_id": training_row["trajectory_id"],
            "checkpoint_id": training_row["checkpoint_id"],
            "task_name": training_row["task_name"],
            "task_index": training_row["task_index"],
            "replicate": training_row["replicate"],
            "seed": training_row["seed"],
            "step": training_row["step"],
            "arm": training_row["arm"],
            "train": training_row["train"],
            "holdout": holdout,
        })
        del model
    path = output_root / "holdout_metrics.jsonl"
    base.write_jsonl(path, rows)
    summary: dict[str, Any] = {
        "phase": PHASE,
        "evaluated_at_utc": base.utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(rows),
        "parent_trajectory_count": len({
            row["trajectory_id"] for row in rows
        }),
        "all_holdout_logits_exactly_finite": all(
            row["holdout"]["exact_all_finite"] for row in rows
        ),
        "holdout_metrics_sha256": base.sha256_file(path),
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_json(output_root / "summary.json", summary)
    print(json.dumps({
        "rows": len(rows),
        "all_finite": summary["all_holdout_logits_exactly_finite"],
        "summary_digest": summary["summary_digest"],
    }))


def first_stable_index(flags: list[bool], count: int) -> int | None:
    if len(flags) < count:
        return None
    return next((
        index for index in range(len(flags) - count + 1)
        if all(flags[index : index + count])
    ), None)


def trajectory_summary(
    parent_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    holdout_by_checkpoint: dict[str, dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    sequence = sorted(parent_rows + arm_rows, key=lambda row: row["step"])
    generalization_flags = [
        row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]
        and holdout_by_checkpoint[row["checkpoint_id"]]["holdout"]["accuracy"]
        >= THRESHOLDS["stable_generalization_accuracy_min"]
        for row in sequence
    ]
    generalization_index = first_stable_index(
        generalization_flags, THRESHOLDS["stable_adjacent_checkpoint_count"]
    )
    camera_flags = [
        row["relation_camera"]["status"] == "EligibleRelation"
        and row["relation_camera"]["actual"]["score"] >= THRESHOLDS["camera_score_min"]
        and row["relation_camera"]["score_advantage"] >= THRESHOLDS["camera_advantage_min"]
        for row in sequence
    ]
    camera_index = first_stable_index(
        camera_flags, THRESHOLDS["camera_adjacent_checkpoint_count"]
    )
    branch = next(row for row in parent_rows if row["step"] == BRANCH_STEP)
    late = [row for row in arm_rows if row["step"] in LATE_ENDPOINT_STEPS]
    final = sequence[-1]
    final_holdout = holdout_by_checkpoint[final["checkpoint_id"]]["holdout"]
    return {
        "trajectory_id": sequence[0]["trajectory_id"],
        "task_name": sequence[0]["task_name"],
        "task_index": sequence[0]["task_index"],
        "replicate": sequence[0]["replicate"],
        "seed": sequence[0]["seed"],
        "arm": arm,
        "stable_generalization_step": (
            sequence[generalization_index]["step"] if generalization_index is not None else None
        ),
        "relation_camera_step": (
            sequence[camera_index]["step"] if camera_index is not None else None
        ),
        "branch_camera_score": branch["relation_camera"]["actual"]["score"],
        "branch_camera_advantage": branch["relation_camera"]["score_advantage"],
        "branch_camera_quiet": bool(
            branch["relation_camera"]["actual"]["score"] < THRESHOLDS["camera_score_min"]
            or branch["relation_camera"]["score_advantage"] < THRESHOLDS["camera_advantage_min"]
        ),
        "late_camera_score": float(np.median([
            row["relation_camera"]["actual"]["score"] for row in late
        ])),
        "late_random_score": float(np.median([
            row["relation_camera"]["random_pairing"]["score"] for row in late
        ])),
        "late_camera_advantage": float(np.median([
            row["relation_camera"]["score_advantage"] for row in late
        ])),
        "final_train_accuracy": final["train"]["accuracy"],
        "final_holdout_accuracy": final_holdout["accuracy"],
        "final_parameter_l2_norm": final["training_only_structure"]["parameter_l2_norm"],
        "final_camera_score": final["relation_camera"]["actual"]["score"],
        "final_random_score": final["relation_camera"]["random_pairing"]["score"],
        "all_train_logits_finite": all(row["train"]["exact_all_finite"] for row in sequence),
        "all_holdout_logits_finite": all(
            holdout_by_checkpoint[row["checkpoint_id"]]["holdout"]["exact_all_finite"]
            for row in sequence
        ),
    }


def all_trajectory_summaries() -> list[dict[str, Any]]:
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    holdout_rows = base.read_jsonl(OUT_ROOT / "runs/holdout/holdout_metrics.jsonl")
    holdout_by_checkpoint = {row["checkpoint_id"]: row for row in holdout_rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    summaries = []
    for trajectory_id in sorted(grouped):
        rows = grouped[trajectory_id]
        parent = [row for row in rows if row["arm"] == "parent"]
        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            summaries.append(trajectory_summary(parent, arm_rows, holdout_by_checkpoint, arm))
    return summaries


def build_score() -> dict[str, Any]:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    holdout_summary = base.read_json(OUT_ROOT / "runs/holdout/summary.json")
    summaries = all_trajectory_summaries()
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(row["trajectory_id"], {})[row["arm"]] = row
    pair_rows = []
    for trajectory_id in sorted(grouped):
        arms = grouped[trajectory_id]
        if set(arms) != set(ARMS):
            raise RuntimeError(f"arm set mismatch: {trajectory_id}")
        baseline = arms["continued_decay"]
        off = arms["decay_off"]
        norm = arms["norm_matched_no_decay"]
        all_finite = all(
            row["all_train_logits_finite"] and row["all_holdout_logits_finite"]
            for row in arms.values()
        )
        final_behavior_pass = all(
            row["stable_generalization_step"] is not None
            and row["final_train_accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]
            and row["final_holdout_accuracy"] >= THRESHOLDS["stable_generalization_accuracy_min"]
            for row in arms.values()
        )
        final_holdout_gap = max(
            row["final_holdout_accuracy"] for row in arms.values()
        ) - min(row["final_holdout_accuracy"] for row in arms.values())
        generalization_steps = [row["stable_generalization_step"] for row in arms.values()]
        generalization_gap = (
            max(generalization_steps) - min(generalization_steps)
            if all(step is not None for step in generalization_steps)
            else None
        )
        behavior_matched = bool(
            all_finite
            and final_behavior_pass
            and final_holdout_gap <= THRESHOLDS["maximum_median_final_holdout_gap"]
            and generalization_gap is not None
            and generalization_gap <= THRESHOLDS["maximum_median_generalization_time_gap"]
        )
        pair_rows.append({
            "trajectory_id": trajectory_id,
            "task_name": baseline["task_name"],
            "replicate": baseline["replicate"],
            "seed": baseline["seed"],
            "branch_camera_quiet": baseline["branch_camera_quiet"],
            "all_finite": all_finite,
            "final_behavior_pass": final_behavior_pass,
            "final_holdout_gap": final_holdout_gap,
            "generalization_time_gap": generalization_gap,
            "behavior_matched": behavior_matched,
            "late_camera_effect_continued_minus_off": (
                baseline["late_camera_score"] - off["late_camera_score"]
            ),
            "late_random_effect_continued_minus_off": (
                baseline["late_random_score"] - off["late_random_score"]
            ),
            "late_advantage_effect_continued_minus_off": (
                baseline["late_camera_advantage"] - off["late_camera_advantage"]
            ),
            "late_camera_continued_minus_norm_matched": (
                baseline["late_camera_score"] - norm["late_camera_score"]
            ),
            "late_camera_norm_matched_minus_off": (
                norm["late_camera_score"] - off["late_camera_score"]
            ),
            "camera_event_delay_off_minus_continued": (
                (off["relation_camera_step"] or (MAX_STEP + 1))
                - (baseline["relation_camera_step"] or (MAX_STEP + 1))
            ),
            "arms": arms,
        })
    branch_quiet = [row for row in pair_rows if row["branch_camera_quiet"]]
    matched = [row for row in branch_quiet if row["behavior_matched"]]
    task_names = sorted({row["task_name"] for row in matched})
    behavior_match_fraction = len(matched) / max(len(branch_quiet), 1)
    actual_effects = [row["late_camera_effect_continued_minus_off"] for row in matched]
    random_effects = [row["late_random_effect_continued_minus_off"] for row in matched]
    event_delays = [row["camera_event_delay_off_minus_continued"] for row in matched]
    holdout_gaps = [row["final_holdout_gap"] for row in branch_quiet]
    generalization_gaps = [
        row["generalization_time_gap"]
        for row in branch_quiet if row["generalization_time_gap"] is not None
    ]
    per_task = []
    for task_name in task_names:
        task_rows = [row for row in matched if row["task_name"] == task_name]
        baseline_scores = [
            row["arms"]["continued_decay"]["late_camera_score"] for row in task_rows
        ]
        baseline_advantages = [
            row["arms"]["continued_decay"]["late_camera_advantage"] for row in task_rows
        ]
        median_effect = float(np.median([
            row["late_camera_effect_continued_minus_off"] for row in task_rows
        ]))
        per_task.append({
            "task_name": task_name,
            "trajectory_count": len(task_rows),
            "median_continued_late_camera_score": float(np.median(baseline_scores)),
            "median_continued_late_camera_advantage": float(np.median(baseline_advantages)),
            "median_late_camera_effect": median_effect,
            "baseline_endpoint_pass": bool(
                np.median(baseline_scores) >= THRESHOLDS["camera_score_min"]
                and np.median(baseline_advantages) >= THRESHOLDS["camera_advantage_min"]
            ),
            "effect_pass": median_effect >= THRESHOLDS["minimum_per_class_late_camera_effect"],
        })
    baseline_endpoint_breadth = sum(row["baseline_endpoint_pass"] for row in per_task)
    effect_class_breadth = sum(row["effect_pass"] for row in per_task)
    median_actual_effect = float(np.median(actual_effects)) if actual_effects else 0.0
    median_random_effect = float(np.median(random_effects)) if random_effects else 0.0
    median_norm_base_gap = float(np.median([
        abs(row["late_camera_continued_minus_norm_matched"]) for row in matched
    ])) if matched else math.inf
    median_norm_off_difference = float(np.median([
        row["late_camera_norm_matched_minus_off"] for row in matched
    ])) if matched else 0.0
    median_base_norm_difference = float(np.median([
        row["late_camera_continued_minus_norm_matched"] for row in matched
    ])) if matched else 0.0
    median_abs_norm_off_gap = float(np.median([
        abs(row["late_camera_norm_matched_minus_off"]) for row in matched
    ])) if matched else math.inf
    tolerance = THRESHOLDS["norm_control_equivalence_tolerance"]
    if median_norm_base_gap <= tolerance and median_norm_off_difference >= tolerance:
        norm_control_classification = "global_norm_matching_recovers_late_camera"
    elif median_abs_norm_off_gap <= tolerance and median_base_norm_difference >= tolerance:
        norm_control_classification = "global_norm_equality_is_not_sufficient"
    else:
        norm_control_classification = "mixed_or_unresolved"
    checks = {
        "protocol_and_material_closed": protocol["material_gate"]["pass"],
        "branch_identity_exact": seal["all_branch_models_and_optimizer_moments_exact"],
        "norm_match_numerically_exact": (
            seal["maximum_norm_match_relative_error"]
            <= THRESHOLDS["maximum_norm_match_relative_error"]
        ),
        "matched_trajectory_count": (
            len(matched)
            >= THRESHOLDS["minimum_branch_quiet_behavior_matched_trajectory_count"]
        ),
        "behavior_match_fraction": (
            behavior_match_fraction
            >= THRESHOLDS["minimum_branch_quiet_behavior_match_fraction"]
        ),
        "task_class_breadth": len(task_names) >= THRESHOLDS["minimum_task_class_breadth"],
        "final_holdout_equivalence": bool(
            holdout_gaps
            and float(np.median(holdout_gaps))
            <= THRESHOLDS["maximum_median_final_holdout_gap"]
        ),
        "generalization_time_equivalence": bool(
            generalization_gaps
            and float(np.median(generalization_gaps))
            <= THRESHOLDS["maximum_median_generalization_time_gap"]
        ),
        "continued_decay_endpoint_object_breadth": (
            baseline_endpoint_breadth >= THRESHOLDS["minimum_task_class_breadth"]
        ),
        "median_late_camera_effect": (
            median_actual_effect >= THRESHOLDS["minimum_median_late_camera_effect"]
        ),
        "late_camera_effect_class_breadth": (
            effect_class_breadth >= THRESHOLDS["minimum_late_camera_effect_class_breadth"]
        ),
        "actual_effect_exceeds_random_pairing_effect": (
            median_actual_effect - median_random_effect
            >= THRESHOLDS["minimum_actual_over_random_effect_advantage"]
        ),
    }
    primary_pass = bool(all(checks.values()))
    behavior_gate_names = (
        "matched_trajectory_count", "behavior_match_fraction", "task_class_breadth",
        "final_holdout_equivalence", "generalization_time_equivalence",
    )
    behavior_interpretable = bool(all(checks[name] for name in behavior_gate_names))
    return {
        "phase": PHASE,
        "scored_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "primary_endpoint_pass": primary_pass,
        "behavior_interpretable": behavior_interpretable,
        "checks": checks,
        "parent_trajectory_count": len(grouped),
        "branch_quiet_trajectory_count": len(branch_quiet),
        "behavior_matched_trajectory_count": len(matched),
        "behavior_match_fraction": behavior_match_fraction,
        "behavior_matched_task_class_count": len(task_names),
        "median_final_holdout_gap": float(np.median(holdout_gaps)) if holdout_gaps else None,
        "median_generalization_time_gap": (
            float(np.median(generalization_gaps)) if generalization_gaps else None
        ),
        "median_late_camera_effect": median_actual_effect,
        "median_late_random_effect": median_random_effect,
        "median_actual_over_random_effect_advantage": median_actual_effect - median_random_effect,
        "effect_class_breadth": effect_class_breadth,
        "continued_decay_endpoint_class_breadth": baseline_endpoint_breadth,
        "median_camera_event_delay_off_minus_continued": (
            float(np.median(event_delays)) if event_delays else None
        ),
        "norm_control": {
            "classification": norm_control_classification,
            "median_absolute_continued_minus_norm_matched": median_norm_base_gap,
            "median_norm_matched_minus_off": median_norm_off_difference,
            "median_continued_minus_norm_matched": median_base_norm_difference,
            "median_absolute_norm_matched_minus_off": median_abs_norm_off_gap,
            "maximum_parameter_norm_match_relative_error": seal["maximum_norm_match_relative_error"],
        },
        "per_task": per_task,
        "pair_rows": pair_rows,
        "trajectory_summaries": summaries,
        "interpretation": {
            "if_pass": (
                "Continued AdamW decay causally promotes the late affine relation normal form under "
                "matched behavior in these controlled networks. This does not make that normal form "
                "the cause of early generalization."
            ),
            "if_behavior_fails": (
                "The optimizer intervention changed behavior enough that the representation-source "
                "contrast is causally confounded and must be marked inconclusive."
            ),
            "if_effect_fails": (
                "Continued weight decay is not a sufficient explanation of the late affine camera "
                "under the frozen dose, branch time, and task panel; close this source hypothesis."
            ),
            "scope": "Controlled RoleSquareNetwork formation only.",
        },
    }


def score_command() -> None:
    path = OUT_ROOT / "analysis/score.json"
    if path.exists():
        raise RuntimeError("score already exists")
    score = build_score()
    score["score_digest"] = base.digest(score)
    base.write_json(path, score)
    print(json.dumps({
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "behavior_interpretable": score["behavior_interpretable"],
        "matched_trajectories": score["behavior_matched_trajectory_count"],
        "median_late_camera_effect": score["median_late_camera_effect"],
        "effect_class_breadth": score["effect_class_breadth"],
        "norm_control": score["norm_control"]["classification"],
        "score_digest": score["score_digest"],
    }))


def finalize_command() -> None:
    path = OUT_ROOT / "analysis/final.json"
    if path.exists():
        raise RuntimeError("final already exists")
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    score = base.read_json(OUT_ROOT / "analysis/score.json")
    if score["primary_endpoint_pass"]:
        status = "late_affine_regularization_weight_decay_supported"
    elif not score["behavior_interpretable"]:
        status = "inconclusive_due_to_behavior_mismatch"
    else:
        status = "weight_decay_source_hypothesis_not_supported"
    final: dict[str, Any] = {
        "phase": PHASE,
        "finalized_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "status": status,
            "primary_endpoint_pass": score["primary_endpoint_pass"],
            "behavior_interpretable": score["behavior_interpretable"],
            "late_affine_weight_decay_source_supported": score["primary_endpoint_pass"],
            "early_generalization_driver_identified": False,
            "relation_causal_use_authorized": False,
            "nonlinear_camera_search_authorized": False,
            "hidden_feature_search_authorized": False,
            "branch_time_or_decay_tuning_authorized": False,
            "auto_continue": False,
            "authorized_next": None,
        },
        "claims": [
            "All three arms began from tensor-identical model states and optimizer moments at a frozen branch step.",
            "Holdout labels were absent from training, branching, norm matching, and relation-camera computation.",
            "Six all-new quotient classes and eight seeds per class were used; no Phase1172/1174 quotient signature collided.",
            "The norm-matched arm controls equality of global parameter L2 norm but not optimizer moments after the branch.",
        ],
        "hard_boundary": (
            "This one-shot fork can classify a source of the late affine normal form only. It cannot "
            "identify the early generalization mechanism, prove semantic encoding, or reopen the "
            "closed affine formation-prediction branch."
        ),
        "next_research_requirement": (
            "A new known-truth early implementation family with a preregistered camera that precedes "
            "behavior; no automatic metric or camera search is authorized."
        ),
    }
    final["final_digest"] = base.digest(final)
    base.write_json(path, final)
    print(json.dumps({
        "status": status,
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "auto_continue": False,
        "final_digest": final["final_digest"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "protocol", "smoke", "train-and-seal", "reveal-holdout", "score", "finalize",
    ))
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "smoke": smoke_command,
        "train-and-seal": train_and_seal_command,
        "reveal-holdout": reveal_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
