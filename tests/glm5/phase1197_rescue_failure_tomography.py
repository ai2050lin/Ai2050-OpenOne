"""Pre-registered tomography of Phase 1195 sparse-rescue failures.

The experiment does not search for a better rescue optimizer.  It separates
solver error, coarse-mask coverage, stable linear span, coefficient-domain and
sparsity gaps, cross-panel transport, and local nonlinearity.  Four sealed
Phase 1195 replay capsules are used only for engineering calibration.  Formal
claims use twelve new tasks and two independent splits.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import lsq_linear, minimize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402
import phase1195_continuous_sparse_coalition_rescue as p1195  # noqa: E402


PHASE = 1197
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1197_rescue_failure_tomography_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1197_rescue_failure_tomography"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_ROW_ROOT = OUT_ROOT / "runs/formal/rows"
REPLAY_ROOT = OUT_ROOT / "runs/formal/replay_capsules"
TRAINING_SEAL = OUT_ROOT / "runs/formal/seal.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
PHASE1195_CAPSULE_ROOT = (
    ROOT
    / "tests/glm5/result/phase1195_continuous_sparse_coalition_rescue/runs/formal/replay_capsules"
)

ARCHITECTURES = p1195.ARCHITECTURES
STAGE = 100
BATCH_SIZE = 64
BASIS_EPSILON = p1195.BASIS_EPSILON
FORMAL_REPLICATES = 4

FORMAL_TASKS = (
    {"name": "diag_disc_affine_00", "split": "discovery", "family": "affine", "task_seed": 1_197_101},
    {"name": "diag_disc_affine_01", "split": "discovery", "family": "affine", "task_seed": 1_197_107},
    {"name": "diag_disc_bitmix_00", "split": "discovery", "family": "bitmix", "task_seed": 1_197_113},
    {"name": "diag_disc_bitmix_01", "split": "discovery", "family": "bitmix", "task_seed": 1_197_119},
    {"name": "diag_disc_random_00", "split": "discovery", "family": "random", "task_seed": 1_197_127},
    {"name": "diag_disc_random_01", "split": "discovery", "family": "random", "task_seed": 1_197_133},
    {"name": "diag_conf_affine_00", "split": "confirmation", "family": "affine", "task_seed": 1_197_203},
    {"name": "diag_conf_affine_01", "split": "confirmation", "family": "affine", "task_seed": 1_197_209},
    {"name": "diag_conf_bitmix_00", "split": "confirmation", "family": "bitmix", "task_seed": 1_197_217},
    {"name": "diag_conf_bitmix_01", "split": "confirmation", "family": "bitmix", "task_seed": 1_197_223},
    {"name": "diag_conf_random_00", "split": "confirmation", "family": "random", "task_seed": 1_197_231},
    {"name": "diag_conf_random_01", "split": "confirmation", "family": "random", "task_seed": 1_197_239},
)

THRESHOLDS = {
    "eligible_fraction_min": 0.95,
    "solver_success_fraction_min": 0.95,
    "solver_objective_gap_p95_max": 1e-5,
    "solver_kkt_p95_max": 1e-3,
    "full_recovery_mean_min": 0.999,
    "full_recovery_min": 0.99,
    "core_update_norm_fraction_mean_min": 0.90,
    "core_recovery_mean_max": 0.50,
    "core_recovery_below_half_fraction_min": 0.75,
    "best_embedding_only_recovery_mean_min": 0.20,
    "span_cal_error_mean_max": 0.50,
    "span_patch_update_fraction_median_min": 2.0,
    "design_cross_panel_cosine_mean_min": 0.90,
    "span_transfer_gap_abs_mean_max": 0.15,
    "l1_nonlinear_eval_error_mean_max": 0.05,
    "span_failure_error_mean_min": 0.50,
    "box_gap_mean_min": 0.20,
    "sparsity_gap_mean_min": 0.10,
    "transfer_gap_abs_mean_min": 0.15,
    "nonlinear_error_mean_min": 0.05,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(task_index: int, architecture: str, replicate: int) -> int:
    return (
        1_197_000_000
        + task_index * 100_003
        + list(ARCHITECTURES).index(architecture) * 10_007
        + replicate * 1_009
    )


def relative_error(value: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(value - target) / max(np.linalg.norm(target), 1e-12))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows])) if rows else float("nan")


def objective(
    alpha: np.ndarray, design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    target_norm_sq = max(float(np.dot(target, target)), 1e-12)
    residual = design @ alpha - target
    value = 0.5 * float(np.dot(residual, residual)) / target_norm_sq
    value += p1195.REGULARIZATION * float(np.dot(weights, alpha))
    gradient = design.T @ residual / target_norm_sq + p1195.REGULARIZATION * weights
    return value, gradient


def kkt_residual(alpha: np.ndarray, gradient: np.ndarray, tolerance: float = 1e-8) -> float:
    values = []
    for coefficient, derivative in zip(alpha, gradient):
        if coefficient <= tolerance:
            values.append(max(-float(derivative), 0.0))
        elif coefficient >= 1.0 - tolerance:
            values.append(max(float(derivative), 0.0))
        else:
            values.append(abs(float(derivative)))
    return max(values, default=0.0)


@torch.inference_mode()
def response(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    update: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
) -> np.ndarray:
    model = p1194.clone_model(parent)
    p1193.assign_parameters(model, parent_vector + update)
    value = p1193.quotient_response(model, inputs, targets, candidates)
    del model
    return value


def parameter_slices(model: TinyCausalTransformer) -> list[tuple[str, int, int]]:
    slices = []
    offset = 0
    for name, parameter in model.named_parameters():
        slices.append((name, offset, offset + parameter.numel()))
        offset += parameter.numel()
    return slices


def mask_for_prefixes(
    reference: torch.Tensor,
    slices: list[tuple[str, int, int]],
    prefixes: tuple[str, ...],
) -> torch.Tensor:
    mask = torch.zeros_like(reference, dtype=torch.bool)
    for name, start, stop in slices:
        if any(name.startswith(prefix) for prefix in prefixes):
            mask[start:stop] = True
    return mask


def omitted_masks(
    model: TinyCausalTransformer, reference: torch.Tensor
) -> dict[str, torch.Tensor]:
    slices = parameter_slices(model)
    return {
        "token_embedding": mask_for_prefixes(reference, slices, ("token_embedding.",)),
        "position_embedding": mask_for_prefixes(reference, slices, ("position_embedding.",)),
        "final_norm": mask_for_prefixes(reference, slices, ("final_norm.",)),
        "lm_head": mask_for_prefixes(reference, slices, ("lm_head.",)),
    }


def patch_from_coefficients(
    coefficients: np.ndarray,
    components: list[torch.Tensor],
) -> torch.Tensor:
    patch = torch.zeros_like(components[0])
    for coefficient, component in zip(coefficients, components):
        patch += float(coefficient) * component
    return patch


def diagnose_payload(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    task = payload["task"]
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    parent = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    parent_vector = payload["parent_vector"].to(device)
    control_update = payload["control_update"].to(device)
    difference = payload["difference"].to(device)
    groups = p1194.component_masks(parent)

    control_cal = response(
        parent, parent_vector, control_update, inputs[calibration], targets[calibration], candidates
    )
    control_eval = response(
        parent, parent_vector, control_update, inputs[evaluation], targets[evaluation], candidates
    )
    real_cal = response(
        parent,
        parent_vector,
        control_update + difference,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    real_eval = response(
        parent,
        parent_vector,
        control_update + difference,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )
    target_cal = real_cal - control_cal
    target_eval = real_eval - control_eval

    columns_cal: list[np.ndarray] = []
    columns_eval: list[np.ndarray] = []
    components: list[torch.Tensor] = []
    parameter_fractions: list[float] = []
    for _, mask in groups:
        component = torch.where(mask, difference, torch.zeros_like(difference))
        components.append(component)
        parameter_fractions.append(float(mask.float().mean().item()))
        update = control_update + BASIS_EPSILON * component
        columns_cal.append(
            (
                response(
                    parent,
                    parent_vector,
                    update,
                    inputs[calibration],
                    targets[calibration],
                    candidates,
                )
                - control_cal
            )
            / BASIS_EPSILON
        )
        columns_eval.append(
            (
                response(
                    parent,
                    parent_vector,
                    update,
                    inputs[evaluation],
                    targets[evaluation],
                    candidates,
                )
                - control_eval
            )
            / BASIS_EPSILON
        )
    design_cal = np.stack(columns_cal, axis=1)
    design_eval = np.stack(columns_eval, axis=1)
    weights = np.asarray(parameter_fractions, dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)

    alpha_current = np.asarray(payload["alpha"], dtype=np.float64)
    current_objective, current_gradient = objective(
        alpha_current, design_cal, target_cal, weights
    )
    solved = minimize(
        lambda value: objective(value, design_cal, target_cal, weights),
        alpha_current,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(alpha_current),
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20_000, "maxls": 100},
    )
    alpha_reference = np.asarray(solved.x, dtype=np.float64)
    reference_objective, reference_gradient = objective(
        alpha_reference, design_cal, target_cal, weights
    )
    alpha_span = np.linalg.lstsq(design_cal, target_cal, rcond=None)[0]
    alpha_nonnegative = lsq_linear(
        design_cal,
        target_cal,
        bounds=(0.0, np.inf),
        tol=1e-14,
        max_iter=10_000,
        lsmr_tol=1e-14,
    ).x
    alpha_signed_box = lsq_linear(
        design_cal,
        target_cal,
        bounds=(-1.0, 1.0),
        tol=1e-14,
        max_iter=10_000,
        lsmr_tol=1e-14,
    ).x
    alpha_box = lsq_linear(
        design_cal,
        target_cal,
        bounds=(0.0, 1.0),
        tol=1e-14,
        max_iter=10_000,
        lsmr_tol=1e-14,
    ).x

    left, singular_values, _ = np.linalg.svd(design_cal, full_matrices=False)
    singular_max = max(float(singular_values[0]), 1e-12)
    stable_rank_mask = singular_values >= 0.05 * singular_max
    stable_projection = left[:, stable_rank_mask] @ (
        left[:, stable_rank_mask].T @ target_cal
    )

    core_patch = patch_from_coefficients(np.ones(len(components)), components)
    current_patch = patch_from_coefficients(alpha_reference, components)
    span_patch = patch_from_coefficients(alpha_span, components)
    core_cal = response(
        parent,
        parent_vector,
        control_update + core_patch,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    core_eval = response(
        parent,
        parent_vector,
        control_update + core_patch,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )
    current_cal = response(
        parent,
        parent_vector,
        control_update + current_patch,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    current_eval = response(
        parent,
        parent_vector,
        control_update + current_patch,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )

    omitted = omitted_masks(parent, difference)
    omitted_results: dict[str, dict[str, float]] = {}
    for name, mask in omitted.items():
        omitted_patch = torch.where(mask, difference, torch.zeros_like(difference))
        only_eval = response(
            parent,
            parent_vector,
            control_update + omitted_patch,
            inputs[evaluation],
            targets[evaluation],
            candidates,
        )
        added_eval = response(
            parent,
            parent_vector,
            control_update + core_patch + omitted_patch,
            inputs[evaluation],
            targets[evaluation],
            candidates,
        )
        omitted_results[name] = {
            "difference_norm_fraction": float(
                omitted_patch.norm() / difference.norm().clamp_min(1e-12)
            ),
            "only_recovery": 1.0 - relative_error(only_eval - control_eval, target_eval),
            "added_recovery": 1.0 - relative_error(added_eval - control_eval, target_eval),
        }

    full_eval = response(
        parent,
        parent_vector,
        control_update + difference,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )
    core_union = torch.stack([mask for _, mask in groups]).any(dim=0)
    omitted_union = torch.stack(list(omitted.values())).any(dim=0)
    partition_complete = bool((core_union | omitted_union).all().item())
    partition_overlap = int((core_union & omitted_union).sum().item())

    row = {
        "trajectory_id": payload["trajectory_id"],
        "task_name": task["name"],
        "task_index": payload["task_index"],
        "task_seed": task["task_seed"],
        "family": task["family"],
        "split": task.get("split", "development"),
        "architecture": payload["architecture"],
        "replicate": payload["replicate"],
        "model_seed": payload["model_seed"],
        "stage": STAGE,
        "real_child_accuracy": payload["real_child_accuracy"],
        "control_match": payload["control_metrics"],
        "eligible": bool(p1195.control_match_pass(payload["control_metrics"])),
        "solver_success": bool(solved.success),
        "solver_objective_gap": float(current_objective - reference_objective),
        "solver_alpha_max_error": float(np.max(np.abs(alpha_current - alpha_reference))),
        "solver_current_kkt": kkt_residual(alpha_current, current_gradient),
        "solver_reference_kkt": kkt_residual(alpha_reference, reference_gradient),
        "span_cal_error": relative_error(design_cal @ alpha_span, target_cal),
        "stable_span_cal_error": relative_error(stable_projection, target_cal),
        "nonnegative_cal_error": relative_error(design_cal @ alpha_nonnegative, target_cal),
        "signed_box_cal_error": relative_error(design_cal @ alpha_signed_box, target_cal),
        "box_cal_error": relative_error(design_cal @ alpha_box, target_cal),
        "l1_cal_error": relative_error(design_cal @ alpha_reference, target_cal),
        "span_eval_linear_error": relative_error(design_eval @ alpha_span, target_eval),
        "box_eval_linear_error": relative_error(design_eval @ alpha_box, target_eval),
        "l1_eval_linear_error": relative_error(design_eval @ alpha_reference, target_eval),
        "span_transfer_gap_abs": abs(
            relative_error(design_eval @ alpha_span, target_eval)
            - relative_error(design_cal @ alpha_span, target_cal)
        ),
        "box_gap": relative_error(design_cal @ alpha_box, target_cal)
        - relative_error(design_cal @ alpha_span, target_cal),
        "sparsity_gap": relative_error(design_cal @ alpha_reference, target_cal)
        - relative_error(design_cal @ alpha_box, target_cal),
        "core_cal_recovery": 1.0 - relative_error(core_cal - control_cal, target_cal),
        "core_eval_recovery": 1.0 - relative_error(core_eval - control_eval, target_eval),
        "core_update_norm_fraction": float(
            core_patch.norm() / difference.norm().clamp_min(1e-12)
        ),
        "full_eval_recovery": 1.0 - relative_error(full_eval - control_eval, target_eval),
        "l1_exact_cal_error": relative_error(current_cal - control_cal, target_cal),
        "l1_exact_eval_error": relative_error(current_eval - control_eval, target_eval),
        "l1_nonlinear_cal_error": relative_error(
            current_cal - control_cal, design_cal @ alpha_reference
        ),
        "l1_nonlinear_eval_error": relative_error(
            current_eval - control_eval, design_eval @ alpha_reference
        ),
        "target_cross_panel_cosine": cosine(target_cal, target_eval),
        "design_cross_panel_cosine": cosine(design_cal.reshape(-1), design_eval.reshape(-1)),
        "span_alpha_min": float(alpha_span.min()),
        "span_alpha_max": float(alpha_span.max()),
        "span_negative_fraction": float(np.mean(alpha_span < -1e-6)),
        "span_above_one_fraction": float(np.mean(alpha_span > 1.0 + 1e-6)),
        "span_patch_update_fraction": float(
            span_patch.norm() / difference.norm().clamp_min(1e-12)
        ),
        "basis_condition_number": float(
            singular_values[0] / max(float(singular_values[-1]), 1e-12)
        ),
        "basis_effective_rank_5pct": int(stable_rank_mask.sum()),
        "basis_column_count": int(design_cal.shape[1]),
        "core_parameter_fraction": float(core_union.float().mean().item()),
        "partition_complete": partition_complete,
        "partition_overlap": partition_overlap,
        "omitted_groups": omitted_results,
        "best_embedding_only_recovery": max(
            omitted_results["token_embedding"]["only_recovery"],
            omitted_results["position_embedding"]["only_recovery"],
        ),
    }
    del parent, inputs, targets, candidates
    gc.collect()
    torch.cuda.empty_cache()
    return row


def build_formal_payload(
    task: dict[str, Any],
    task_index: int,
    architecture: str,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    seed = model_seed(task_index, architecture, replicate)
    set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=generator).to(device)
        for _ in range(STAGE + 1)
    ]
    for step in range(STAGE):
        p1193.training_step(
            model, optimizer, inputs[batches[step]], targets[batches[step]], candidates
        )
    payload = p1195.build_material(
        model,
        optimizer,
        inputs,
        targets,
        candidates,
        calibration,
        evaluation,
        batches[STAGE],
        seed + STAGE * 1009,
    )
    payload.update(
        {
            "task": dict(task),
            "task_index": task_index,
            "architecture": architecture,
            "replicate": replicate,
            "trajectory_id": f"{task['name']}::{architecture}::r{replicate}",
            "model_seed": seed,
        }
    )
    del model, optimizer, inputs, targets, candidates, batches
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def summarize_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"no rows for split {split}")
    eligible = [row for row in selected if row["eligible"]]
    metrics = {
        "count": len(selected),
        "eligible_count": len(eligible),
        "eligible_fraction": len(eligible) / len(selected),
        "solver_success_fraction": float(np.mean([row["solver_success"] for row in eligible])),
        "solver_objective_gap_p95": percentile(
            [row["solver_objective_gap"] for row in eligible], 0.95
        ),
        "solver_kkt_p95": percentile([row["solver_current_kkt"] for row in eligible], 0.95),
        "full_recovery_mean": mean(eligible, "full_eval_recovery"),
        "full_recovery_min": min(float(row["full_eval_recovery"]) for row in eligible),
        "core_recovery_mean": mean(eligible, "core_eval_recovery"),
        "core_recovery_below_half_fraction": float(
            np.mean([row["core_eval_recovery"] < 0.5 for row in eligible])
        ),
        "core_update_norm_fraction_mean": mean(eligible, "core_update_norm_fraction"),
        "core_parameter_fraction_mean": mean(eligible, "core_parameter_fraction"),
        "best_embedding_only_recovery_mean": mean(
            eligible, "best_embedding_only_recovery"
        ),
        "token_embedding_only_recovery_mean": float(
            np.mean([row["omitted_groups"]["token_embedding"]["only_recovery"] for row in eligible])
        ),
        "position_embedding_only_recovery_mean": float(
            np.mean([row["omitted_groups"]["position_embedding"]["only_recovery"] for row in eligible])
        ),
        "final_norm_only_recovery_mean": float(
            np.mean([row["omitted_groups"]["final_norm"]["only_recovery"] for row in eligible])
        ),
        "lm_head_only_recovery_mean": float(
            np.mean([row["omitted_groups"]["lm_head"]["only_recovery"] for row in eligible])
        ),
        "span_cal_error_mean": mean(eligible, "span_cal_error"),
        "stable_span_cal_error_mean": mean(eligible, "stable_span_cal_error"),
        "box_cal_error_mean": mean(eligible, "box_cal_error"),
        "l1_cal_error_mean": mean(eligible, "l1_cal_error"),
        "box_gap_mean": mean(eligible, "box_gap"),
        "sparsity_gap_mean": mean(eligible, "sparsity_gap"),
        "span_patch_update_fraction_median": median(eligible, "span_patch_update_fraction"),
        "basis_condition_number_median": median(eligible, "basis_condition_number"),
        "design_cross_panel_cosine_mean": mean(eligible, "design_cross_panel_cosine"),
        "target_cross_panel_cosine_mean": mean(eligible, "target_cross_panel_cosine"),
        "span_transfer_gap_abs_mean": mean(eligible, "span_transfer_gap_abs"),
        "l1_nonlinear_eval_error_mean": mean(eligible, "l1_nonlinear_eval_error"),
        "partition_complete_fraction": float(
            np.mean([row["partition_complete"] and row["partition_overlap"] == 0 for row in eligible])
        ),
    }
    solver_gate = bool(
        metrics["eligible_fraction"] >= THRESHOLDS["eligible_fraction_min"]
        and metrics["solver_success_fraction"] >= THRESHOLDS["solver_success_fraction_min"]
        and metrics["solver_objective_gap_p95"]
        <= THRESHOLDS["solver_objective_gap_p95_max"]
        and metrics["solver_kkt_p95"] <= THRESHOLDS["solver_kkt_p95_max"]
    )
    instrument_gate = bool(
        metrics["full_recovery_mean"] >= THRESHOLDS["full_recovery_mean_min"]
        and metrics["full_recovery_min"] >= THRESHOLDS["full_recovery_min"]
        and metrics["partition_complete_fraction"] == 1.0
    )
    mask_gate = bool(
        solver_gate
        and instrument_gate
        and metrics["core_update_norm_fraction_mean"]
        >= THRESHOLDS["core_update_norm_fraction_mean_min"]
        and metrics["core_recovery_mean"] <= THRESHOLDS["core_recovery_mean_max"]
        and metrics["core_recovery_below_half_fraction"]
        >= THRESHOLDS["core_recovery_below_half_fraction_min"]
        and metrics["best_embedding_only_recovery_mean"]
        >= THRESHOLDS["best_embedding_only_recovery_mean_min"]
        and metrics["span_cal_error_mean"] <= THRESHOLDS["span_cal_error_mean_max"]
        and metrics["span_patch_update_fraction_median"]
        >= THRESHOLDS["span_patch_update_fraction_median_min"]
        and metrics["design_cross_panel_cosine_mean"]
        >= THRESHOLDS["design_cross_panel_cosine_mean_min"]
        and metrics["span_transfer_gap_abs_mean"]
        <= THRESHOLDS["span_transfer_gap_abs_mean_max"]
        and metrics["l1_nonlinear_eval_error_mean"]
        <= THRESHOLDS["l1_nonlinear_eval_error_mean_max"]
    )
    if not solver_gate:
        diagnosis = "solver_not_qualified"
    elif not instrument_gate:
        diagnosis = "instrument_not_qualified"
    elif mask_gate:
        diagnosis = "omitted_high_leverage_basis"
    elif metrics["span_cal_error_mean"] >= THRESHOLDS["span_failure_error_mean_min"]:
        diagnosis = "coarse_span_insufficient"
    elif metrics["box_gap_mean"] >= THRESHOLDS["box_gap_mean_min"]:
        diagnosis = "coefficient_domain_limited"
    elif metrics["sparsity_gap_mean"] >= THRESHOLDS["sparsity_gap_mean_min"]:
        diagnosis = "sparsity_limited"
    elif metrics["span_transfer_gap_abs_mean"] >= THRESHOLDS["transfer_gap_abs_mean_min"]:
        diagnosis = "cross_panel_transfer_limited"
    elif metrics["l1_nonlinear_eval_error_mean"] >= THRESHOLDS["nonlinear_error_mean_min"]:
        diagnosis = "local_nonlinearity_limited"
    else:
        diagnosis = "unresolved_observation_or_interaction"
    return {
        "split": split,
        "metrics": metrics,
        "solver_gate_pass": solver_gate,
        "instrument_gate_pass": instrument_gate,
        "omitted_high_leverage_basis_gate_pass": mask_gate,
        "primary_diagnosis": diagnosis,
        "by_architecture": {
            architecture: {
                "count": len([row for row in eligible if row["architecture"] == architecture]),
                "core_recovery_mean": mean(
                    [row for row in eligible if row["architecture"] == architecture],
                    "core_eval_recovery",
                ),
                "best_embedding_only_recovery_mean": mean(
                    [row for row in eligible if row["architecture"] == architecture],
                    "best_embedding_only_recovery",
                ),
            }
            for architecture in ARCHITECTURES
        },
        "by_family": {
            family: {
                "count": len([row for row in eligible if row["family"] == family]),
                "core_recovery_mean": mean(
                    [row for row in eligible if row["family"] == family],
                    "core_eval_recovery",
                ),
                "best_embedding_only_recovery_mean": mean(
                    [row for row in eligible if row["family"] == family],
                    "best_embedding_only_recovery",
                ),
            }
            for family in ("affine", "bitmix", "random")
        },
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "phase1197": SCRIPT,
        "phase1197_audit": AUDIT_SCRIPT,
        "phase1195": p1195.SCRIPT,
        "phase1194": p1194.SCRIPT,
        "phase1193": p1193.SCRIPT,
        "phase1146_model": ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def develop() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    paths = sorted(PHASE1195_CAPSULE_ROOT.glob("*.pt"))
    if len(paths) != 4:
        raise RuntimeError("expected four sealed Phase1195 replay capsules")
    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(diagnose_payload(payload, torch.device("cuda")))
        print(canonical_json({"development_capsule": path.name}), flush=True)
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = {
        "phase": PHASE,
        "kind": "engineering_scale_only",
        "created_at": utc_now(),
        "capsule_count": len(rows),
        "source_phase": 1195,
        "summary": {
            "core_recovery_mean": mean(rows, "core_eval_recovery"),
            "core_update_norm_fraction_mean": mean(rows, "core_update_norm_fraction"),
            "best_embedding_only_recovery_mean": mean(rows, "best_embedding_only_recovery"),
            "span_cal_error_mean": mean(rows, "span_cal_error"),
            "span_patch_update_fraction_median": median(rows, "span_patch_update_fraction"),
            "design_cross_panel_cosine_mean": mean(rows, "design_cross_panel_cosine"),
            "span_transfer_gap_abs_mean": mean(rows, "span_transfer_gap_abs"),
            "l1_nonlinear_eval_error_mean": mean(rows, "l1_nonlinear_eval_error"),
        },
    }
    write_json(DEVELOPMENT_SUMMARY, summary)
    print(canonical_json(summary["summary"]))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("Phase1197 protocol or outcomes already exist")
    if not DEVELOPMENT_SUMMARY.exists():
        raise RuntimeError("run development engineering calibration first")
    upstream_final = p1195.FINAL_PATH
    protocol = {
        "phase": PHASE,
        "created_at": utc_now(),
        "question": (
            "Which pre-registered failure class explains the Phase1195 coarse sparse rescue boundary "
            "on twelve new tasks: solver, omitted high-leverage basis, span, coefficient domain, "
            "sparsity, transport, nonlinearity, or unresolved observation/interaction?"
        ),
        "scope": (
            "Diagnostic attribution in synthetic TinyTransformer formation events; no improved rescue, "
            "future-learning, autonomous-repair, or natural-language claim."
        ),
        "formal_tasks": list(FORMAL_TASKS),
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "formal_replicates": FORMAL_REPLICATES,
        "stage": STAGE,
        "basis_epsilon": BASIS_EPSILON,
        "diagnostic_order": [
            "solver",
            "instrument_and_partition",
            "omitted_high_leverage_basis",
            "coarse_span",
            "coefficient_domain",
            "sparsity",
            "cross_panel_transport",
            "local_nonlinearity",
            "unresolved_observation_or_interaction",
        ],
        "omitted_groups": [
            "token_embedding",
            "position_embedding",
            "final_norm",
            "lm_head",
        ],
        "thresholds": THRESHOLDS,
        "continuation_rule": (
            "Only one next development object is authorized if the same primary diagnosis independently "
            "passes in discovery and confirmation.  No Phase1195 lambda/top-k retuning is allowed."
        ),
        "forbidden": [
            "change diagnostic order or thresholds after formal outcomes",
            "drop zero-support or negative-recovery events",
            "call algebraic span with amplified coefficients a practical controller",
            "call omitted-basis diagnosis an expanded-basis rescue success",
            "reuse formal rows to tune an expanded controller",
            "claim full latent-state observability or controllability",
            "claim natural-language encoding mechanism",
        ],
        "development": {
            "kind": "four sealed Phase1195 replay capsules for scale calibration only",
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
        },
        "upstream": {
            "phase1195_final_sha256": file_sha256(upstream_final),
            "phase1195_final_digest": read_json(upstream_final)["final_digest"],
        },
        "source_hashes": source_hashes(),
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    stored = protocol["protocol_digest"]
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != stored:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if file_sha256(p1195.FINAL_PATH) != protocol["upstream"]["phase1195_final_sha256"]:
        raise RuntimeError("Phase1195 final changed")
    return protocol


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    replay_ids = {
        "diag_disc_affine_00::compact::r0",
        "diag_disc_affine_00::deep::r0",
        "diag_conf_affine_00::compact::r0",
        "diag_conf_affine_00::deep::r0",
    }
    for task_index, task in enumerate(FORMAL_TASKS):
        for architecture in ARCHITECTURES:
            for replicate in range(FORMAL_REPLICATES):
                payload = build_formal_payload(
                    task, task_index, architecture, replicate, device
                )
                row = diagnose_payload(payload, device)
                rows.append(row)
                if payload["trajectory_id"] in replay_ids:
                    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        payload,
                        REPLAY_ROOT / f"{payload['trajectory_id'].replace('::', '__')}.pt",
                    )
                print(
                    canonical_json(
                        {
                            "task": task["name"],
                            "architecture": architecture,
                            "replicate": replicate,
                            "rows": len(rows),
                        }
                    ),
                    flush=True,
                )
                del payload
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_json(
            FORMAL_ROW_ROOT / f"{row['trajectory_id'].replace('::', '__')}.json", row
        )
    write_jsonl(RAW_ROWS, rows)
    row_manifest = {
        path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))
    }
    replay_manifest = {
        path.name: file_sha256(path) for path in sorted(REPLAY_ROOT.glob("*.pt"))
    }
    seal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "row_manifest": row_manifest,
        "replay_manifest": replay_manifest,
    }
    seal["seal_digest"] = digest(seal)
    write_json(TRAINING_SEAL, seal)
    print(canonical_json({"row_count": len(rows), "seal_digest": seal["seal_digest"]}))


def analyze() -> None:
    verify_protocol()
    seal = read_json(TRAINING_SEAL)
    rows = read_jsonl(RAW_ROWS)
    if file_sha256(RAW_ROWS) != seal["analysis_rows_sha256"]:
        raise RuntimeError("formal rows hash mismatch")
    discovery = summarize_rows(rows, "discovery")
    confirmation = summarize_rows(rows, "confirmation")
    same = discovery["primary_diagnosis"] == confirmation["primary_diagnosis"]
    selected = discovery["primary_diagnosis"] if same else "split_diagnosis_disagreement"
    confirmed = bool(
        same
        and selected == "omitted_high_leverage_basis"
        and discovery["omitted_high_leverage_basis_gate_pass"]
        and confirmation["omitted_high_leverage_basis_gate_pass"]
    )
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "discovery": discovery,
        "confirmation": confirmation,
        "same_primary_diagnosis": same,
        "primary_diagnosis": selected,
        "omitted_high_leverage_basis_confirmed": confirmed,
        "overall_status": (
            "omitted_high_leverage_basis_confirmed"
            if confirmed
            else "rescue_failure_primary_cause_not_confirmed"
        ),
    }
    claim = (
        "Across twelve new tasks and independent splits, the Phase1195 attention/MLP coarse mask "
        "captured most update norm but failed to recover the target quotient response, while omitted "
        "token/position embedding updates alone carried substantial response leverage. Solver, cross-panel "
        "transport, and local-nonlinearity gates passed. This diagnoses an omitted high-leverage control "
        "basis; it does not confirm an expanded-basis rescue."
        if confirmed
        else "The pre-registered tomography did not identify the same qualified primary rescue-failure cause in both splits."
    )
    claims = {
        "rescue_failure_tomography": {
            "type": "E3-KT" if confirmed else "E3-KT-scope-boundary",
            "accepted": True,
            "claim": claim,
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"status": summary["overall_status"], "diagnosis": selected}))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    confirmed = bool(summary["omitted_high_leverage_basis_confirmed"])
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["overall_status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "expanded_partition_rescue_development": confirmed,
            "expanded_partition_rescue_formal": False,
            "future_learning_rescue": False,
            "natural_language_encoding_claim": False,
        },
        "scope": {
            "confirmed": (
                "primary failure attribution for the frozen Phase1195 coarse control basis on new synthetic tasks"
                if confirmed
                else "diagnostic scope boundary"
            ),
            "not_claimed": [
                "expanded-basis rescue success",
                "full latent-state observation",
                "system controllability",
                "behavioral recovery",
                "future-learning recovery",
                "endogenous repair used by the network",
                "natural-language encoding mechanism",
            ],
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(
        canonical_json(
            {
                "status": final["status"],
                "authorized_next": final["authorized_next"],
                "final_digest": final["final_digest"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("develop", "preregister", "run-formal", "analyze", "finalize")
    )
    command = parser.parse_args().command
    {
        "develop": develop,
        "preregister": preregister,
        "run-formal": run_formal,
        "analyze": analyze,
        "finalize": finalize,
    }[command]()


if __name__ == "__main__":
    main()
