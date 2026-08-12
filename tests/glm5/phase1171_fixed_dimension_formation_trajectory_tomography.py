#!/usr/bin/env python3
"""Sealed fixed-dimension confirmation of formation-trajectory regimes.

Phase1170 showed that a binary delayed-transition label mixed direct/left-
censored generalizers with trajectories that never generalized stably.  It
also changed model and data dimensions with the modulus.  Phase1171 fixes the
modulus, vocabulary, output size, parameter count, optimizer, and schedule.
Eight randomly frozen asymmetric affine rules are learned by one role-separated
square architecture with dense early checkpoints.  All checkpoints and
training-only summaries are sealed before held-out evaluation.

The primary endpoint asks whether delayed, direct/left-censored, and nonstable
regimes form a broad three-regime panel.  It does not fit a predictor and does
not claim a mechanism.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1171_fixed_dimension_formation_trajectory_tomography_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography"
P1170_FINAL = ROOT / "tests/glm5/result/phase1170_natural_rule_selection_breadth_confirmation/analysis/final.json"
P1170_AUDIT = ROOT / "tests/glm5/result/phase1170_natural_rule_selection_breadth_confirmation/audit/independent_audit.json"
P1170_FINITE_AUDIT = ROOT / "tests/glm5/result/phase1170_natural_rule_selection_breadth_confirmation/audit/finite_fraction_exact_recompute.json"
PILOT31_SCRIPT = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_pilot.py"
PILOT31_RESULT = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_pilot.json"
PILOT61_SCRIPT = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_p61_pilot.py"
PILOT61_RESULT = ROOT / "tests/glm5_temp/phase1171_fixed_dimension_role_square_p61_pilot.json"

PHASE = 1171
MODULUS = 61
MODEL_WIDTH = 128
TRAIN_FRACTION = 0.50
REPLICATES = 8
TASK_SELECTION_SEED = 11710017
FORMAL_TASK_COUNT = 8
CHECKPOINT_STEPS = (25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 10000)
PILOT_OPERATION = (2, 3, 5)
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": max(CHECKPOINT_STEPS),
}
THRESHOLDS = {
    "train_fit_accuracy_min": 0.99,
    "stable_generalization_accuracy_min": 0.90,
    "stable_generalization_adjacent_checkpoint_count": 2,
    "global_count_per_regime_min": 8,
    "mixed_task_count_min": 5,
    "count_per_present_regime_in_mixed_task_min": 2,
    "distinct_regimes_per_mixed_task_min": 2,
    "all_trajectories_must_fit": True,
    "all_logits_must_be_finite": True,
}
REGIMES = ("delayed", "direct_left_censored", "nonstable")


def eligible_operations() -> list[tuple[int, int, int]]:
    return [
        (alpha, beta, gamma)
        for alpha in range(1, MODULUS)
        for beta in range(1, MODULUS)
        if alpha != beta and (alpha + beta) % MODULUS != 0
        for gamma in range(MODULUS)
        if (alpha, beta, gamma) != PILOT_OPERATION
    ]


def sampled_operations() -> tuple[tuple[int, int, int], ...]:
    return tuple(random.Random(TASK_SELECTION_SEED).sample(eligible_operations(), 12))


OPERATION_SAMPLE = sampled_operations()
FORMAL_OPERATIONS = OPERATION_SAMPLE[:FORMAL_TASK_COUNT]
RESERVED_OPERATIONS = OPERATION_SAMPLE[FORMAL_TASK_COUNT:]
TASKS = {f"affine_{index:02d}_a{op[0]}_b{op[1]}_g{op[2]}": op for index, op in enumerate(FORMAL_OPERATIONS)}


def model_seed(task_index: int, replicate: int) -> int:
    return 11710000 + int(task_index) * 100_003 + int(replicate) * 1_009


@dataclass(frozen=True)
class RoleSquareConfig:
    modulus: int = MODULUS
    width: int = MODEL_WIDTH


class RoleSquareNetwork(nn.Module):
    """Fixed-size square learner that can represent asymmetric operand roles."""

    def __init__(self, config: RoleSquareConfig) -> None:
        super().__init__()
        self.config = config
        self.left_embedding = nn.Embedding(config.modulus, config.width)
        self.right_embedding = nn.Embedding(config.modulus, config.width)
        self.hidden = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.modulus, bias=False)
        for module in (self.left_embedding, self.right_embedding, self.hidden, self.output):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        summed = self.left_embedding(input_ids[:, 0]) + self.right_embedding(input_ids[:, 1])
        return self.output(self.hidden(summed).square())


def make_data(operation: tuple[int, int, int], seed: int) -> dict[str, torch.Tensor]:
    alpha, beta, gamma = operation
    pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    order = np.random.default_rng(seed).permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    train_mask_array = np.zeros(len(pairs), dtype=bool)
    train_mask_array[order[:cutoff]] = True
    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor([(alpha * a + beta * b + gamma) % MODULUS for a, b in pairs], dtype=torch.long)
    train_mask = torch.tensor(train_mask_array, dtype=torch.bool)
    return {
        "train_x": x[train_mask],
        "train_y": y[train_mask],
        "holdout_x": x[~train_mask],
        "holdout_y": y[~train_mask],
    }


@torch.inference_mode()
def evaluate(model: RoleSquareNetwork, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs.to(device)).float()
    finite = torch.isfinite(logits)
    finite_count = int(finite.sum(dtype=torch.int64).item())
    total_count = finite.numel()
    exact_all_finite = bool(finite.all().item())
    predicted = logits.argmax(dim=-1).cpu()
    result: dict[str, Any] = {
        "case_count": len(targets),
        "accuracy": float((predicted == targets).float().mean().item()),
        "exact_finite_count": finite_count,
        "total_logit_count": total_count,
        "finite_fraction": finite_count / total_count,
        "exact_all_finite": exact_all_finite,
    }
    if exact_all_finite:
        probabilities = torch.softmax(logits, dim=-1)
        target_probability = probabilities.gather(1, targets.to(device)[:, None]).squeeze(1)
        result["mean_target_probability"] = float(target_probability.mean().item())
        result["minimum_target_probability"] = float(target_probability.min().item())
    else:
        result["mean_target_probability"] = None
        result["minimum_target_probability"] = None
    return result


@torch.inference_mode()
def local_rule_scores(
    model: RoleSquareNetwork,
    train_x: torch.Tensor,
    operation: tuple[int, int, int],
    device: torch.device,
) -> dict[str, Any]:
    alpha, beta, _ = operation
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(train_x.to(device)).float().cpu()
    centered = logits - logits.mean(dim=1, keepdim=True)
    lookup = torch.full((MODULUS, MODULUS), -1, dtype=torch.long)
    row_indices = torch.arange(len(train_x), dtype=torch.long)
    lookup[train_x[:, 0], train_x[:, 1]] = row_indices
    left_indices = lookup[(train_x[:, 0] + 1) % MODULUS, train_x[:, 1]]
    right_indices = lookup[train_x[:, 0], (train_x[:, 1] + 1) % MODULUS]
    left_mask = left_indices >= 0
    right_mask = right_indices >= 0
    left_cos = F.cosine_similarity(centered[left_indices[left_mask]], torch.roll(centered[left_mask], shifts=alpha, dims=1), dim=1)
    right_cos = F.cosine_similarity(centered[right_indices[right_mask]], torch.roll(centered[right_mask], shifts=beta, dims=1), dim=1)
    both = left_mask & right_mask
    left_aligned = torch.roll(centered[left_indices[both]], shifts=-alpha, dims=1)
    right_aligned = torch.roll(centered[right_indices[both]], shifts=-beta, dims=1)
    path_cos = F.cosine_similarity(left_aligned, right_aligned, dim=1)
    return {
        "local_equivariance_cosine": float(torch.cat((left_cos, right_cos)).mean().item()),
        "local_equivariance_edge_count": int(left_mask.sum().item() + right_mask.sum().item()),
        "path_consistency_cosine": float(path_cos.mean().item()),
        "path_consistency_cell_count": int(both.sum().item()),
    }


def training_only_structure(
    model: RoleSquareNetwork,
    data: dict[str, torch.Tensor],
    operation: tuple[int, int, int],
    device: torch.device,
) -> dict[str, Any]:
    left = F.linear(model.left_embedding.weight.detach().float(), model.hidden.weight.detach().float())
    right = F.linear(model.right_embedding.weight.detach().float(), model.hidden.weight.detach().float())
    output = model.output.weight.detach().float()
    total_norm = sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters()) ** 0.5
    result = {
        "left_embedding_circulant_gram": base.circulant_gram_score(left),
        "right_embedding_circulant_gram": base.circulant_gram_score(right),
        "mean_embedding_circulant_gram": 0.5 * (base.circulant_gram_score(left) + base.circulant_gram_score(right)),
        "output_circulant_gram": base.circulant_gram_score(output),
        "left_embedding_fourier_top4_share": base.fourier_top_share(left),
        "right_embedding_fourier_top4_share": base.fourier_top_share(right),
        "mean_embedding_fourier_top4_share": 0.5 * (base.fourier_top_share(left) + base.fourier_top_share(right)),
        "output_fourier_top4_share": base.fourier_top_share(output),
        "parameter_l2_norm": total_norm,
    }
    result.update(local_rule_scores(model, data["train_x"], operation, device))
    return result


def checkpoint_payload(
    model: RoleSquareNetwork,
    task_name: str,
    task_index: int,
    operation: tuple[int, int, int],
    replicate: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task_name,
        "task_index": task_index,
        "operation": operation,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> RoleSquareNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RoleSquareNetwork(RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["step"])
    fit_rows = [row for row in ordered if row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]]
    generalizer = [
        row["train"]["accuracy"] >= THRESHOLDS["train_fit_accuracy_min"]
        and row["holdout"]["accuracy"] >= THRESHOLDS["stable_generalization_accuracy_min"]
        for row in ordered
    ]
    stable_start = next(
        (ordered[index] for index in range(len(ordered) - 1) if generalizer[index] and generalizer[index + 1]),
        None,
    )
    fit_step = fit_rows[0]["step"] if fit_rows else None
    stable_step = stable_start["step"] if stable_start else None
    if fit_step is None:
        regime = "unfit"
    elif stable_step is None:
        regime = "nonstable"
    elif stable_step == fit_step:
        regime = "direct_left_censored"
    elif stable_step > fit_step:
        regime = "delayed"
    else:
        raise RuntimeError("stable generalization cannot precede the fit event under the frozen definition")
    return {
        "trajectory_id": ordered[0]["trajectory_id"],
        "task_name": ordered[0]["task_name"],
        "task_index": ordered[0]["task_index"],
        "operation": ordered[0]["operation"],
        "replicate": ordered[0]["replicate"],
        "seed": ordered[0]["seed"],
        "fit_step": fit_step,
        "stable_generalization_step": stable_step,
        "observed_delay_steps": stable_step - fit_step if stable_step is not None and fit_step is not None else None,
        "regime": regime,
        "maximum_holdout_accuracy": max(row["holdout"]["accuracy"] for row in ordered),
        "final_holdout_accuracy": ordered[-1]["holdout"]["accuracy"],
        "all_train_logits_finite": all(row["train"]["exact_all_finite"] for row in ordered),
        "all_holdout_logits_finite": all(row["holdout"]["exact_all_finite"] for row in ordered),
    }


def task_summary(trajectories: list[dict[str, Any]], task_name: str, operation: tuple[int, int, int]) -> dict[str, Any]:
    selected = [row for row in trajectories if row["task_name"] == task_name]
    counts = {regime: sum(row["regime"] == regime for row in selected) for regime in (*REGIMES, "unfit")}
    present = sum(counts[regime] >= THRESHOLDS["count_per_present_regime_in_mixed_task_min"] for regime in REGIMES)
    mixed = present >= THRESHOLDS["distinct_regimes_per_mixed_task_min"]
    delays = [row["observed_delay_steps"] for row in selected if row["observed_delay_steps"] is not None]
    return {
        "task_name": task_name,
        "operation": operation,
        "trajectory_count": len(selected),
        "regime_counts": counts,
        "qualifying_distinct_regime_count": present,
        "mixed_regime_task": mixed,
        "median_observed_delay_steps": statistics.median(delays) if delays else None,
        "median_maximum_holdout_accuracy": statistics.median(row["maximum_holdout_accuracy"] for row in selected),
    }


def endpoint_decision(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    task_summaries = [task_summary(trajectories, task_name, operation) for task_name, operation in TASKS.items()]
    global_counts = {regime: sum(row["regime"] == regime for row in trajectories) for regime in (*REGIMES, "unfit")}
    mixed_tasks = sum(row["mixed_regime_task"] for row in task_summaries)
    all_finite = all(row["all_train_logits_finite"] and row["all_holdout_logits_finite"] for row in trajectories)
    pass_conditions = {
        "all_trajectories_fit": global_counts["unfit"] == 0,
        "all_logits_finite": all_finite,
        "each_target_regime_has_global_support": all(global_counts[regime] >= THRESHOLDS["global_count_per_regime_min"] for regime in REGIMES),
        "mixed_task_breadth": mixed_tasks >= THRESHOLDS["mixed_task_count_min"],
    }
    return {
        "task_summaries": task_summaries,
        "global_regime_counts": global_counts,
        "mixed_task_count": mixed_tasks,
        "pass_conditions": pass_conditions,
        "primary_endpoint_pass": all(pass_conditions.values()),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite existing Phase1171 output")
    prior_final = base.read_json(P1170_FINAL)
    prior_audit = base.read_json(P1170_AUDIT)
    finite_audit = base.read_json(P1170_FINITE_AUDIT)
    failed_checks = [key for key, value in prior_audit["checks"].items() if not value]
    if prior_final["decision"]["auto_continue"] is not False:
        raise RuntimeError("Phase1170 was expected to stop automatically")
    if not (prior_audit["passed"] == 29 and prior_audit["total"] == 30 and failed_checks == ["finite"]):
        raise RuntimeError("unexpected Phase1170 audit state")
    if not finite_audit["overall_pass"]:
        raise RuntimeError("Phase1170 exact finite recomputation did not pass")
    expected_sample = (
        (26, 29, 60), (53, 47, 7), (29, 60, 37), (12, 14, 2),
        (48, 9, 42), (18, 42, 26), (35, 8, 52), (21, 32, 57),
        (6, 22, 40), (20, 52, 27), (30, 25, 33), (27, 51, 42),
    )
    if OPERATION_SAMPLE != expected_sample:
        raise RuntimeError("operation sample changed")
    allocation = []
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = make_data(operation, seed + 17)
            allocation.append({
                "task_name": task_name,
                "task_index": task_index,
                "operation": operation,
                "replicate": replicate,
                "seed": seed,
                "train_pair_digest": base.digest(data["train_x"].tolist()),
                "holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
                "train_label_digest": base.digest(data["train_y"].tolist()),
                "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
                "train_case_count": len(data["train_x"]),
                "holdout_case_count": len(data["holdout_x"]),
            })
    protocol = {
        "phase": PHASE,
        "created_at_utc": base.utc_now(),
        "question": "Do three preregistered formation-time regimes form a broad panel when task and model dimensions are fixed?",
        "authorization": "The user's post-Phase1170 request explicitly authorizes a new fixed-dimension trajectory-tomography phase; it does not amend Phase1170.",
        "prerequisite": {
            "phase1170_final_sha256": base.sha256_file(P1170_FINAL),
            "phase1170_audit_sha256": base.sha256_file(P1170_AUDIT),
            "phase1170_exact_finite_audit_sha256": base.sha256_file(P1170_FINITE_AUDIT),
            "phase1170_primary_pass": prior_final["decision"]["primary_endpoint_pass"],
            "phase1170_original_audit": {"passed": prior_audit["passed"], "total": prior_audit["total"], "failed_checks": failed_checks},
            "phase1170_exact_finite_pass": finite_audit["overall_pass"],
        },
        "engineering_calibration_excluded_from_evidence": {
            "p31_script_sha256": base.sha256_file(PILOT31_SCRIPT),
            "p31_result_sha256": base.sha256_file(PILOT31_RESULT),
            "p61_script_sha256": base.sha256_file(PILOT61_SCRIPT),
            "p61_result_sha256": base.sha256_file(PILOT61_RESULT),
            "p61_pilot_operation": PILOT_OPERATION,
            "rule": "Pilot tasks, seeds, trajectories, and outcomes cannot enter any Phase1171 count or claim.",
        },
        "source_hashes": {
            "primary_script": base.sha256_file(SCRIPT),
            "audit_script": base.sha256_file(AUDIT_SCRIPT),
        },
        "task_selection": {
            "modulus": MODULUS,
            "eligible_operation_count": len(eligible_operations()),
            "eligibility": "alpha,beta in 1..60; alpha!=beta; alpha+beta!=0 mod61; exclude the p61 pilot operation",
            "selection_seed": TASK_SELECTION_SEED,
            "sampled_operations": OPERATION_SAMPLE,
            "formal_operations": FORMAL_OPERATIONS,
            "reserved_fresh_operations": RESERVED_OPERATIONS,
        },
        "tasks": TASKS,
        "replicates_per_task": REPLICATES,
        "trajectory_count": len(TASKS) * REPLICATES,
        "allocation": allocation,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "train_fraction": TRAIN_FRACTION,
        "model": {
            "class": "RoleSquareNetwork",
            "modulus": MODULUS,
            "width": MODEL_WIDTH,
            "parameter_count_fixed_across_tasks": sum(parameter.numel() for parameter in RoleSquareNetwork(RoleSquareConfig()).parameters()),
            "role_separated_embeddings": True,
        },
        "training": TRAINING,
        "regime_definitions": {
            "fit_step": "first saved checkpoint with train accuracy >= 0.99",
            "stable_generalization_step": "first of two adjacent saved checkpoints with train accuracy >= 0.99 and holdout accuracy >= 0.90",
            "delayed": "stable_generalization_step > fit_step",
            "direct_left_censored": "stable_generalization_step == fit_step; this is an observation category, not proof that no earlier memory state existed",
            "nonstable": "fit occurs but no stable_generalization_step exists in the observation window",
            "unfit": "no saved checkpoint reaches train accuracy >= 0.99; disqualifying rather than a target regime",
        },
        "thresholds": THRESHOLDS,
        "sealed_rules": [
            "No held-out logits, losses, labels, accuracies, event times, or regimes are computed during training.",
            "All 1088 checkpoints and training-only summaries are sealed before the held-out directory exists.",
            "Dimensions, operation sample, formal/reserved split, seeds, checkpoints, thresholds, and branch logic cannot change after protocol creation.",
            "The direct_left_censored category is allowed to disappear under denser sampling; disappearance is a valid negative result.",
            "K144 features are recorded but cannot select tasks, labels, thresholds, or predictors in Phase1171.",
            "A pass only authorizes a separately preregistered prospective predictor on untouched reserved operations.",
            "A fail forbids predictor fitting, operation search, checkpoint redefinition, hidden scans, and causal intervention in this registry.",
        ],
    }
    protocol["protocol_digest"] = base.digest(protocol)
    base.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    print(json.dumps({
        "protocol_digest": protocol["protocol_digest"],
        "formal_operations": FORMAL_OPERATIONS,
        "reserved_operations": RESERVED_OPERATIONS,
        "trajectories": protocol["trajectory_count"],
    }))


def train_and_seal_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training is already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout outcomes exist before training seal")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            base.set_seed(seed)
            data = make_data(operation, seed + 17)
            model = RoleSquareNetwork(RoleSquareConfig()).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
            train_x_device = data["train_x"].to(device)
            train_y_device = data["train_y"].to(device)
            for step in range(1, max(CHECKPOINT_STEPS) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(train_x_device).float()
                    loss = F.cross_entropy(logits, train_y_device)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss: {task_name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                if step not in CHECKPOINT_STEPS:
                    continue
                train_metrics = evaluate(model, data["train_x"], data["train_y"], device)
                structure = training_only_structure(model, data, operation, device)
                trajectory_id = f"{task_name}_r{replicate}_s{seed}"
                checkpoint_id = f"{trajectory_id}_step{step:05d}"
                checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, task_name, task_index, operation, replicate, seed, step), checkpoint_path)
                checkpoint_hash = base.sha256_file(checkpoint_path)
                checkpoint_hashes[checkpoint_id] = checkpoint_hash
                rows.append({
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": checkpoint_id,
                    "task_name": task_name,
                    "task_index": task_index,
                    "operation": operation,
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.item()),
                    "train": train_metrics,
                    "training_only_structure": structure,
                    "train_pair_digest": base.digest(data["train_x"].tolist()),
                    "sealed_holdout_pair_digest": base.digest(data["holdout_x"].tolist()),
                    "train_label_digest": base.digest(data["train_y"].tolist()),
                    "sealed_holdout_label_digest": base.digest(data["holdout_y"].tolist()),
                    "holdout_evaluated_during_training": False,
                    "holdout_used_by_gradient": False,
                    "checkpoint_sha256": checkpoint_hash,
                })
            print(json.dumps({"trained": trajectory_id, "checkpoints": len(CHECKPOINT_STEPS)}), flush=True)
            del model, optimizer, train_x_device, train_y_device
            gc.collect()
            torch.cuda.empty_cache()
    metrics_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    base.write_jsonl(metrics_path, rows)
    seal = {
        "phase": PHASE,
        "sealed_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": len(TASKS) * REPLICATES,
        "checkpoint_count": len(rows),
        "training_metrics_sha256": base.sha256_file(metrics_path),
        "checkpoint_hashes": checkpoint_hashes,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "all_training_logits_exactly_finite": all(row["train"]["exact_all_finite"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({"seal_digest": seal["seal_digest"], "trajectories": seal["trajectory_count"], "checkpoints": seal["checkpoint_count"]}))


def evaluate_holdout_command() -> None:
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    holdout_root = OUT_ROOT / "runs/holdout"
    if holdout_root.exists():
        raise RuntimeError("refusing to overwrite held-out outcomes")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    rows = []
    for training_row in training_rows:
        checkpoint_id = training_row["checkpoint_id"]
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
        if base.sha256_file(checkpoint_path) != training_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_id}")
        operation = tuple(training_row["operation"])
        data = make_data(operation, training_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        holdout_metrics = evaluate(model, data["holdout_x"], data["holdout_y"], device)
        rows.append({
            "trajectory_id": training_row["trajectory_id"],
            "checkpoint_id": checkpoint_id,
            "task_name": training_row["task_name"],
            "task_index": training_row["task_index"],
            "operation": operation,
            "replicate": training_row["replicate"],
            "seed": training_row["seed"],
            "step": training_row["step"],
            "train": training_row["train"],
            "training_only_structure": training_row["training_only_structure"],
            "holdout": holdout_metrics,
        })
        del model
    output_path = holdout_root / "holdout_metrics.jsonl"
    base.write_jsonl(output_path, rows)
    summary = {
        "phase": PHASE,
        "evaluated_at_utc": base.utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(rows),
        "all_holdout_logits_exactly_finite": all(row["holdout"]["exact_all_finite"] for row in rows),
        "holdout_metrics_sha256": base.sha256_file(output_path),
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_json(holdout_root / "summary.json", summary)
    print(json.dumps({"summary_digest": summary["summary_digest"], "rows": len(rows), "all_finite": summary["all_holdout_logits_exactly_finite"]}))


def score_command() -> None:
    holdout_summary = base.read_json(OUT_ROOT / "runs/holdout/summary.json")
    rows = base.read_jsonl(OUT_ROOT / "runs/holdout/holdout_metrics.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    trajectories = [trajectory_summary(group) for group in grouped.values()]
    decision = endpoint_decision(trajectories)
    score = {
        "phase": PHASE,
        "scored_at_utc": base.utc_now(),
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "trajectory_count": len(trajectories),
        "trajectories": sorted(trajectories, key=lambda row: (row["task_index"], row["replicate"])),
        **decision,
        "interpretation": {
            "if_pass": "Three formation-time regimes have sufficient cross-task support to authorize one separately preregistered training-only prospective predictor test on reserved operations.",
            "if_fail": "The three-regime taxonomy did not form a broad panel under fixed dimensions; no predictor, operation search, hidden scan, or intervention is authorized.",
            "scope": "The endpoint concerns event-time behavior in a controlled role-square learner, not a language code or causal formation mechanism.",
        },
    }
    score["score_digest"] = base.digest(score)
    base.write_json(OUT_ROOT / "analysis/score.json", score)
    print(json.dumps({
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "global_regime_counts": score["global_regime_counts"],
        "mixed_task_count": score["mixed_task_count"],
        "score_digest": score["score_digest"],
    }))


def finalize_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    score = base.read_json(OUT_ROOT / "analysis/score.json")
    passed = bool(score["primary_endpoint_pass"])
    final = {
        "phase": PHASE,
        "finalized_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "primary_endpoint_pass": passed,
            "three_regime_panel_confirmed": passed,
            "prospective_predictor_phase_authorized": passed,
            "hidden_scan_authorized": False,
            "causal_intervention_authorized": False,
            "operation_search_authorized": False,
            "auto_continue": passed,
            "authorized_next": "Phase1172: freeze simple training-only event-time predictors on Phase1171, then test once on the untouched reserved operations" if passed else None,
        },
        "claims": [
            "All formal tasks share exactly the same modulus, dimensions, architecture, and parameter count.",
            "Asymmetric affine rules are representable because operand roles use separate embedding tables.",
            "Formation regimes are defined by fit and stable-generalization event order, not by a posthoc memory-accuracy cutoff.",
            "A positive taxonomy endpoint would authorize prediction only; it would not establish mechanism identity or language external validity.",
        ],
    }
    final["final_digest"] = base.digest(final)
    base.write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({"final_digest": final["final_digest"], "auto_continue": final["decision"]["auto_continue"]}))


def smoke_command() -> None:
    expected = (
        (26, 29, 60), (53, 47, 7), (29, 60, 37), (12, 14, 2),
        (48, 9, 42), (18, 42, 26), (35, 8, 52), (21, 32, 57),
        (6, 22, 40), (20, 52, 27), (30, 25, 33), (27, 51, 42),
    )
    if OPERATION_SAMPLE != expected:
        raise RuntimeError("unexpected operation sample")
    if set(FORMAL_OPERATIONS).intersection(RESERVED_OPERATIONS):
        raise RuntimeError("formal and reserved operations overlap")
    if PILOT_OPERATION in OPERATION_SAMPLE:
        raise RuntimeError("pilot operation leaked into the sampled registry")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    parameter_count = sum(parameter.numel() for parameter in RoleSquareNetwork(RoleSquareConfig()).parameters())
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        seed = model_seed(task_index, 0)
        data = make_data(operation, seed + 17)
        overlap = set(map(tuple, data["train_x"].tolist())).intersection(map(tuple, data["holdout_x"].tolist()))
        if overlap:
            raise RuntimeError("train/holdout overlap")
        print(json.dumps({
            "task_name": task_name,
            "operation": operation,
            "train": len(data["train_x"]),
            "holdout": len(data["holdout_x"]),
            "parameter_count": parameter_count,
        }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "train-and-seal", "evaluate-holdout", "score", "finalize", "smoke"))
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "train-and-seal": train_and_seal_command,
        "evaluate-holdout": evaluate_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
