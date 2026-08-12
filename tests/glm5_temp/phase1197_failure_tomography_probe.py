"""Engineering probe for Phase 1197 rescue-failure diagnostics.

This script reads only the four sealed Phase 1195 CUDA replay capsules.  It is
not a formal scientific result; it checks metric scales before preregistration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear, minimize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402
import phase1195_continuous_sparse_coalition_rescue as p1195  # noqa: E402


CAPSULE_ROOT = (
    ROOT
    / "tests/glm5/result/phase1195_continuous_sparse_coalition_rescue/runs/formal/replay_capsules"
)


def relative_error(value: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(value - target) / max(np.linalg.norm(target), 1e-12))


def objective(
    alpha: np.ndarray, design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    norm_sq = max(float(np.dot(target, target)), 1e-12)
    residual = design @ alpha - target
    value = 0.5 * float(np.dot(residual, residual)) / norm_sq
    value += p1195.REGULARIZATION * float(np.dot(weights, alpha))
    gradient = design.T @ residual / norm_sq + p1195.REGULARIZATION * weights
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


def inspect_capsule(path: Path, device: torch.device) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    task = payload["task"]
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    parent = TinyCausalTransformer(p1195.ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    parent_vector = payload["parent_vector"].to(device)
    control_update = payload["control_update"].to(device)
    difference = payload["difference"].to(device)
    groups = p1194.component_masks(parent)
    parameter_slices = []
    offset = 0
    for name, parameter in parent.named_parameters():
        parameter_slices.append((name, offset, offset + parameter.numel()))
        offset += parameter.numel()

    def named_mask(prefixes: tuple[str, ...]) -> torch.Tensor:
        mask = torch.zeros_like(difference, dtype=torch.bool)
        for name, start, stop in parameter_slices:
            if any(name.startswith(prefix) for prefix in prefixes):
                mask[start:stop] = True
        return mask

    omitted_groups = {
        "token_embedding": named_mask(("token_embedding.",)),
        "position_embedding": named_mask(("position_embedding.",)),
        "final_norm": named_mask(("final_norm.",)),
        "lm_head": named_mask(("lm_head.",)),
    }

    control_cal = response(
        parent,
        parent_vector,
        control_update,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    control_eval = response(
        parent,
        parent_vector,
        control_update,
        inputs[evaluation],
        targets[evaluation],
        candidates,
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

    columns_cal = []
    columns_eval = []
    components = []
    parameter_fractions = []
    for _, mask in groups:
        component = torch.where(mask, difference, torch.zeros_like(difference))
        components.append(component)
        parameter_fractions.append(float(mask.float().mean().item()))
        probe_update = control_update + p1195.BASIS_EPSILON * component
        columns_cal.append(
            (
                response(
                    parent,
                    parent_vector,
                    probe_update,
                    inputs[calibration],
                    targets[calibration],
                    candidates,
                )
                - control_cal
            )
            / p1195.BASIS_EPSILON
        )
        columns_eval.append(
            (
                response(
                    parent,
                    parent_vector,
                    probe_update,
                    inputs[evaluation],
                    targets[evaluation],
                    candidates,
                )
                - control_eval
            )
            / p1195.BASIS_EPSILON
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
    singular_values = np.linalg.svd(design_cal, compute_uv=False)
    singular_max = max(float(singular_values[0]), 1e-12)
    stable_rank_mask = singular_values >= 0.05 * singular_max
    left, _, _ = np.linalg.svd(design_cal, full_matrices=False)
    stable_projection = left[:, stable_rank_mask] @ (
        left[:, stable_rank_mask].T @ target_cal
    )

    mask_patch = torch.stack(components).sum(dim=0)
    mask_cal = response(
        parent,
        parent_vector,
        control_update + mask_patch,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    mask_eval = response(
        parent,
        parent_vector,
        control_update + mask_patch,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )
    omitted_exact = {}
    for name, mask in omitted_groups.items():
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
            control_update + mask_patch + omitted_patch,
            inputs[evaluation],
            targets[evaluation],
            candidates,
        )
        omitted_exact[name] = {
            "difference_norm_fraction": float(
                omitted_patch.norm() / difference.norm().clamp_min(1e-12)
            ),
            "only_recovery": 1.0 - relative_error(
                only_eval - control_eval, target_eval
            ),
            "added_recovery": 1.0 - relative_error(
                added_eval - control_eval, target_eval
            ),
        }
    reference_patch = sum(
        (float(coefficient) * component for coefficient, component in zip(alpha_reference, components)),
        torch.zeros_like(difference),
    )
    span_patch = sum(
        (float(coefficient) * component for coefficient, component in zip(alpha_span, components)),
        torch.zeros_like(difference),
    )
    signed_box_patch = sum(
        (
            float(coefficient) * component
            for coefficient, component in zip(alpha_signed_box, components)
        ),
        torch.zeros_like(difference),
    )
    reference_cal = response(
        parent,
        parent_vector,
        control_update + reference_patch,
        inputs[calibration],
        targets[calibration],
        candidates,
    )
    reference_eval = response(
        parent,
        parent_vector,
        control_update + reference_patch,
        inputs[evaluation],
        targets[evaluation],
        candidates,
    )

    output = {
        "trajectory_id": payload["trajectory_id"],
        "architecture": payload["architecture"],
        "solver_success": bool(solved.success),
        "solver_objective_gap": current_objective - reference_objective,
        "solver_alpha_max_error": float(np.max(np.abs(alpha_current - alpha_reference))),
        "solver_current_kkt": kkt_residual(alpha_current, current_gradient),
        "solver_reference_kkt": kkt_residual(alpha_reference, reference_gradient),
        "span_cal_error": relative_error(design_cal @ alpha_span, target_cal),
        "stable_span_cal_error": relative_error(stable_projection, target_cal),
        "nonnegative_cal_error": relative_error(
            design_cal @ alpha_nonnegative, target_cal
        ),
        "signed_box_cal_error": relative_error(
            design_cal @ alpha_signed_box, target_cal
        ),
        "box_cal_error": relative_error(design_cal @ alpha_box, target_cal),
        "l1_cal_error": relative_error(design_cal @ alpha_reference, target_cal),
        "span_eval_linear_error": relative_error(design_eval @ alpha_span, target_eval),
        "nonnegative_eval_linear_error": relative_error(
            design_eval @ alpha_nonnegative, target_eval
        ),
        "signed_box_eval_linear_error": relative_error(
            design_eval @ alpha_signed_box, target_eval
        ),
        "box_eval_linear_error": relative_error(design_eval @ alpha_box, target_eval),
        "l1_eval_linear_error": relative_error(design_eval @ alpha_reference, target_eval),
        "mask_cal_error": relative_error(mask_cal - control_cal, target_cal),
        "mask_eval_error": relative_error(mask_eval - control_eval, target_eval),
        "mask_update_norm_fraction": float(mask_patch.norm() / difference.norm().clamp_min(1e-12)),
        "l1_exact_cal_error": relative_error(reference_cal - control_cal, target_cal),
        "l1_exact_eval_error": relative_error(reference_eval - control_eval, target_eval),
        "l1_nonlinear_cal_error": relative_error(
            reference_cal - control_cal, design_cal @ alpha_reference
        ),
        "l1_nonlinear_eval_error": relative_error(
            reference_eval - control_eval, design_eval @ alpha_reference
        ),
        "target_cross_panel_cosine": p1194.cosine(target_cal, target_eval),
        "design_cross_panel_cosine": p1194.cosine(design_cal.reshape(-1), design_eval.reshape(-1)),
        "span_alpha_min": float(alpha_span.min()),
        "span_alpha_max": float(alpha_span.max()),
        "span_negative_fraction": float(np.mean(alpha_span < -1e-6)),
        "span_above_one_fraction": float(np.mean(alpha_span > 1.0 + 1e-6)),
        "span_patch_update_fraction": float(
            span_patch.norm() / difference.norm().clamp_min(1e-12)
        ),
        "signed_box_patch_update_fraction": float(
            signed_box_patch.norm() / difference.norm().clamp_min(1e-12)
        ),
        "signed_box_boundary_fraction": float(
            np.mean(np.abs(alpha_signed_box) >= 1.0 - 1e-6)
        ),
        "basis_condition_number": float(
            singular_values[0] / max(float(singular_values[-1]), 1e-12)
        ),
        "basis_effective_rank_5pct": int(stable_rank_mask.sum()),
        "basis_column_count": int(design_cal.shape[1]),
        "omitted_exact": omitted_exact,
    }
    del parent, inputs, targets, candidates
    torch.cuda.empty_cache()
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows = [inspect_capsule(path, device) for path in sorted(CAPSULE_ROOT.glob("*.pt"))]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
