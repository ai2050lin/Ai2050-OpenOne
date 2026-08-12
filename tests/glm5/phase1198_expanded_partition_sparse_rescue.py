"""Expanded-partition sparse rescue after the Phase 1197 mask diagnosis.

The only experimental change from Phase 1195 is the control basis.  The
original per-layer attention/MLP groups are retained and four previously
omitted parameter blocks are added: token embedding, position embedding,
final normalization, and LM head.  Solver, coefficient domain, regularizer,
negative controls, complexity limits, and rescue thresholds stay frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import TinyCausalTransformer  # noqa: E402
import phase1193_tiny_transformer_quotient_causal_bridge as p1193  # noqa: E402
import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402
import phase1195_continuous_sparse_coalition_rescue as p1195  # noqa: E402
import phase1197_rescue_failure_tomography as p1197  # noqa: E402


PHASE = 1198
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1198_expanded_partition_sparse_rescue_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1198_expanded_partition_sparse_rescue"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
DEVELOPMENT_AUDIT = OUT_ROOT / "development/independent_audit.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FORMAL_ROW_ROOT = OUT_ROOT / "runs/formal/rows"
REPLAY_ROOT = OUT_ROOT / "runs/formal/replay_capsules"
TRAINING_SEAL = OUT_ROOT / "runs/formal/seal.json"
RAW_ROWS = OUT_ROOT / "analysis/rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

ARCHITECTURES = p1195.ARCHITECTURES
RESCUE_STAGE = p1195.RESCUE_STAGE
WRONG_TIME_STAGE = p1195.WRONG_TIME_STAGE
MAX_STEP = RESCUE_STAGE
BATCH_SIZE = p1195.BATCH_SIZE
BASIS_EPSILON = p1195.BASIS_EPSILON
REGULARIZATION = p1195.REGULARIZATION
SOLVER_ITERATIONS = p1195.SOLVER_ITERATIONS
SUPPORT_EPSILON = p1195.SUPPORT_EPSILON
DEVELOPMENT_REPLICATES = 2
FORMAL_REPLICATES = 4

DEVELOPMENT_TASKS = (
    {"name": "expand_dev_affine_00", "family": "affine", "task_seed": 1_198_011},
    {"name": "expand_dev_affine_01", "family": "affine", "task_seed": 1_198_017},
    {"name": "expand_dev_bitmix_00", "family": "bitmix", "task_seed": 1_198_023},
    {"name": "expand_dev_bitmix_01", "family": "bitmix", "task_seed": 1_198_029},
    {"name": "expand_dev_random_00", "family": "random", "task_seed": 1_198_037},
    {"name": "expand_dev_random_01", "family": "random", "task_seed": 1_198_043},
)

FORMAL_TASKS = (
    {"name": "expand_disc_affine_00", "split": "discovery", "family": "affine", "task_seed": 1_198_101},
    {"name": "expand_disc_affine_01", "split": "discovery", "family": "affine", "task_seed": 1_198_107},
    {"name": "expand_disc_bitmix_00", "split": "discovery", "family": "bitmix", "task_seed": 1_198_113},
    {"name": "expand_disc_bitmix_01", "split": "discovery", "family": "bitmix", "task_seed": 1_198_119},
    {"name": "expand_disc_random_00", "split": "discovery", "family": "random", "task_seed": 1_198_127},
    {"name": "expand_disc_random_01", "split": "discovery", "family": "random", "task_seed": 1_198_133},
    {"name": "expand_conf_affine_00", "split": "confirmation", "family": "affine", "task_seed": 1_198_203},
    {"name": "expand_conf_affine_01", "split": "confirmation", "family": "affine", "task_seed": 1_198_209},
    {"name": "expand_conf_bitmix_00", "split": "confirmation", "family": "bitmix", "task_seed": 1_198_217},
    {"name": "expand_conf_bitmix_01", "split": "confirmation", "family": "bitmix", "task_seed": 1_198_223},
    {"name": "expand_conf_random_00", "split": "confirmation", "family": "random", "task_seed": 1_198_231},
    {"name": "expand_conf_random_01", "split": "confirmation", "family": "random", "task_seed": 1_198_239},
)

CONTROL_THRESHOLDS = dict(p1195.CONTROL_THRESHOLDS)
RESCUE_THRESHOLDS = {
    **p1195.RESCUE_THRESHOLDS,
    "expanded_minus_coarse_recovery_mean_min": 0.05,
    "expanded_better_fraction_min": 0.70,
    "architecture_gain_min": 0.02,
    "family_gain_min": 0.00,
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


def model_seed(task_index: int, architecture: str, replicate: int, corpus: str) -> int:
    base = 1_198_900_000 if corpus == "development" else 1_198_000_000
    return base + task_index * 100_003 + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def expanded_component_masks(
    model: TinyCausalTransformer,
) -> list[tuple[str, torch.Tensor]]:
    reference = p1193.flatten_parameters(model)
    groups = list(p1194.component_masks(model))
    groups.extend(list(p1197.omitted_masks(model, reference).items()))
    return groups


def partition_metrics(groups: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    stacked = torch.stack([mask for _, mask in groups]).to(torch.int16)
    coverage = stacked.sum(dim=0)
    return {
        "complete": bool((coverage == 1).all().item()),
        "uncovered": int((coverage == 0).sum().item()),
        "overlap": int((coverage > 1).sum().item()),
        "parameter_fraction_sum": float(sum(mask.float().mean().item() for _, mask in groups)),
    }


@torch.inference_mode()
def solve_expanded_coalition(
    parent: TinyCausalTransformer,
    parent_vector: torch.Tensor,
    control_update: torch.Tensor,
    difference: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    calibration: torch.Tensor,
) -> dict[str, Any]:
    groups = expanded_component_masks(parent)
    partition = partition_metrics(groups)
    if not partition["complete"]:
        raise RuntimeError(f"expanded partition is not exact: {partition}")

    control = p1194.clone_model(parent)
    p1193.assign_parameters(control, parent_vector + control_update)
    control_q = p1193.quotient_response(control, inputs[calibration], targets[calibration], candidates)
    real = p1194.clone_model(parent)
    p1193.assign_parameters(real, parent_vector + control_update + difference)
    real_q = p1193.quotient_response(real, inputs[calibration], targets[calibration], candidates)
    target = real_q - control_q

    columns: list[np.ndarray] = []
    components: list[torch.Tensor] = []
    parameter_fractions: list[float] = []
    for _, mask in groups:
        component = torch.where(mask, difference, torch.zeros_like(difference))
        probe = p1194.clone_model(parent)
        p1193.assign_parameters(probe, parent_vector + control_update + BASIS_EPSILON * component)
        value = p1193.quotient_response(probe, inputs[calibration], targets[calibration], candidates)
        columns.append((value - control_q) / BASIS_EPSILON)
        components.append(component)
        parameter_fractions.append(float(mask.float().mean().item()))
        del probe

    design = np.stack(columns, axis=1)
    target_norm_sq = max(float(np.dot(target, target)), 1e-12)
    lipschitz = float(np.linalg.norm(design, ord=2) ** 2 / target_norm_sq)
    step = 1.0 / max(lipschitz, 1e-9)
    weights = np.asarray(parameter_fractions, dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)
    alpha = np.zeros(len(groups), dtype=np.float64)
    for _ in range(SOLVER_ITERATIONS):
        gradient = design.T @ (design @ alpha - target) / target_norm_sq
        alpha = np.clip(alpha - step * gradient - step * REGULARIZATION * weights, 0.0, 1.0)

    patch = torch.zeros_like(difference)
    support = torch.zeros_like(difference, dtype=torch.bool)
    for coefficient, component, (_, mask) in zip(alpha, components, groups):
        patch += float(coefficient) * component
        if coefficient > SUPPORT_EPSILON:
            support |= mask
    fit = design @ alpha
    del control, real
    return {
        "patch": patch,
        "alpha": alpha,
        "support": support,
        "group_names": [name for name, _ in groups],
        "support_count": int(np.sum(alpha > SUPPORT_EPSILON)),
        "support_parameter_fraction": float(support.float().mean().item()),
        "coefficient_l1": float(np.sum(alpha)),
        "coefficient_max": float(np.max(alpha)),
        "patch_update_fraction": float(patch.norm() / difference.norm().clamp_min(1e-12)),
        "calibration_cosine": p1194.cosine(fit, target),
        "calibration_relative_error": float(
            np.linalg.norm(fit - target) / max(np.linalg.norm(target), 1e-12)
        ),
        "partition": partition,
    }


def expanded_local_nulls(
    parent: TinyCausalTransformer,
    difference: torch.Tensor,
    alpha: np.ndarray,
    support: torch.Tensor,
    correct_patch: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    groups = expanded_component_masks(parent)
    shifted = np.roll(alpha, len(alpha) // 2)
    wrong_component = torch.zeros_like(difference)
    for coefficient, (_, mask) in zip(shifted, groups):
        wrong_component += float(coefficient) * torch.where(mask, difference, torch.zeros_like(difference))
    target_norm = correct_patch.norm()
    wrong_component = p1194.scaled_like(wrong_component, target_norm)
    generator = torch.Generator(device=difference.device).manual_seed(seed)
    random_patch = torch.zeros_like(difference)
    random_values = torch.randn(int(support.sum().item()), generator=generator, device=difference.device)
    random_patch[support] = p1194.scaled_like(random_values, target_norm)
    return {
        "wrong_component": wrong_component,
        "negative": -correct_patch,
        "random": random_patch,
    }


def solve_payload(
    payload: dict[str, Any], device: torch.device, seed: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    task = payload["task"]
    inputs, targets, candidates, calibration, _ = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    parent = TinyCausalTransformer(ARCHITECTURES[payload["architecture"]]).to(device)
    parent.load_state_dict(payload["parent_state"])
    solution = solve_expanded_coalition(
        parent,
        payload["parent_vector"].to(device),
        payload["control_update"].to(device),
        payload["difference"].to(device),
        inputs,
        targets,
        candidates,
        calibration,
    )
    nulls = expanded_local_nulls(
        parent,
        payload["difference"].to(device),
        solution["alpha"],
        solution["support"],
        solution["patch"],
        seed,
    )
    del parent, inputs, targets, candidates
    return solution, nulls


def trajectory(
    task: dict[str, Any],
    task_index: int,
    architecture: str,
    replicate: int,
    corpus: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = model_seed(task_index, architecture, replicate, corpus)
    set_seed(seed)
    inputs, targets, candidates, calibration, evaluation = p1194.make_data(
        int(task["task_seed"]), str(task["family"]), device
    )
    model = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    optimizer = p1193.optimizer_for(model)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    batches = [
        torch.randint(0, len(inputs), (BATCH_SIZE,), generator=generator).to(device)
        for _ in range(MAX_STEP + 1)
    ]
    stage_payloads: dict[int, dict[str, Any]] = {}
    stage_solutions: dict[int, dict[str, Any]] = {}
    for step in range(MAX_STEP + 1):
        if step in (WRONG_TIME_STAGE, RESCUE_STAGE):
            payload = p1195.build_material(
                model,
                optimizer,
                inputs,
                targets,
                candidates,
                calibration,
                evaluation,
                batches[step],
                seed + step * 1009,
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
            solution, _ = solve_payload(payload, device, seed + step * 2003 + 43)
            stage_payloads[step] = payload
            stage_solutions[step] = solution
        if step < MAX_STEP:
            p1193.training_step(model, optimizer, inputs[batches[step]], targets[batches[step]], candidates)

    payload = stage_payloads[RESCUE_STAGE]
    solution = stage_solutions[RESCUE_STAGE]
    parent = TinyCausalTransformer(ARCHITECTURES[architecture]).to(device)
    parent.load_state_dict(payload["parent_state"])
    nulls = expanded_local_nulls(
        parent,
        payload["difference"].to(device),
        solution["alpha"],
        solution["support"],
        solution["patch"],
        seed + RESCUE_STAGE * 2003 + 43,
    )
    coarse_patch = payload["correct_patch"].clone()
    coarse_alpha = list(payload["alpha"])
    coarse_groups = list(payload["group_names"])
    wrong_time = p1194.scaled_like(
        stage_solutions[WRONG_TIME_STAGE]["patch"], solution["patch"].norm()
    ).cpu()
    payload.update(
        {
            "coarse_correct_patch": coarse_patch,
            "coarse_alpha": coarse_alpha,
            "coarse_group_names": coarse_groups,
            "correct_patch": solution["patch"].detach().cpu(),
            "wrong_component_patch": nulls["wrong_component"].detach().cpu(),
            "wrong_time_patch": wrong_time,
            "negative_patch": nulls["negative"].detach().cpu(),
            "random_patch": nulls["random"].detach().cpu(),
            "alpha": solution["alpha"].tolist(),
            "group_names": solution["group_names"],
            "support_count": solution["support_count"],
            "support_parameter_fraction": solution["support_parameter_fraction"],
            "coefficient_l1": solution["coefficient_l1"],
            "coefficient_max": solution["coefficient_max"],
            "patch_update_fraction": solution["patch_update_fraction"],
            "calibration_cosine": solution["calibration_cosine"],
            "calibration_relative_error": solution["calibration_relative_error"],
            "partition": solution["partition"],
        }
    )
    row = {
        "trajectory_id": payload["trajectory_id"],
        "event_id": f"{payload['trajectory_id']}::s{RESCUE_STAGE}",
        "task_name": task["name"],
        "task_index": task_index,
        "task_seed": task["task_seed"],
        "family": task["family"],
        "split": task.get("split", "development"),
        "architecture": architecture,
        "replicate": replicate,
        "model_seed": seed,
        "stage": RESCUE_STAGE,
        "event_loss": payload["event_loss"],
        "real_child_accuracy": payload["real_child_accuracy"],
        "control_match": payload["control_metrics"],
        "alpha": payload["alpha"],
        "group_names": payload["group_names"],
        "coarse_alpha": coarse_alpha,
        "coarse_group_names": coarse_groups,
        "support_count": payload["support_count"],
        "support_parameter_fraction": payload["support_parameter_fraction"],
        "coefficient_l1": payload["coefficient_l1"],
        "coefficient_max": payload["coefficient_max"],
        "patch_update_fraction": payload["patch_update_fraction"],
        "calibration_cosine": payload["calibration_cosine"],
        "calibration_relative_error": payload["calibration_relative_error"],
        "partition": payload["partition"],
    }
    del parent, model, optimizer, inputs, targets, candidates, batches, stage_payloads, stage_solutions
    gc.collect()
    torch.cuda.empty_cache()
    return row, payload


def attach_metrics(
    rows: list[dict[str, Any]], payloads: list[dict[str, Any]], device: torch.device
) -> None:
    by_trajectory = {payload["trajectory_id"]: payload for payload in payloads}
    by_cell = {
        (
            payload["task"].get("split", "development"),
            payload["architecture"],
            payload["replicate"],
            payload["task_index"],
        ): payload
        for payload in payloads
    }
    split_indices: dict[str, list[int]] = {}
    for payload in payloads:
        split_indices.setdefault(payload["task"].get("split", "development"), []).append(payload["task_index"])
    split_indices = {key: sorted(set(value)) for key, value in split_indices.items()}
    for row in rows:
        payload = by_trajectory[row["trajectory_id"]]
        indices = split_indices[row["split"]]
        next_index = indices[(indices.index(row["task_index"]) + 1) % len(indices)]
        wrong_task_payload = by_cell[(row["split"], row["architecture"], row["replicate"], next_index)]
        correct_norm = payload["correct_patch"].norm()
        wrong_task = p1194.scaled_like(wrong_task_payload["correct_patch"], correct_norm)
        payload["wrong_task_patch"] = wrong_task
        payload["wrong_task_trajectory_id"] = wrong_task_payload["trajectory_id"]
        zero = torch.zeros_like(payload["correct_patch"])
        variants = {
            "control": zero,
            "correct": payload["correct_patch"],
            "coarse": payload["coarse_correct_patch"],
            "wrong_component": payload["wrong_component_patch"],
            "wrong_time": payload["wrong_time_patch"],
            "wrong_task": wrong_task,
            "negative": payload["negative_patch"],
            "random": payload["random_patch"],
        }
        measured = {name: p1195.variant_metrics(payload, patch, device) for name, patch in variants.items()}
        control_error = measured["control"]["response_error"]
        for metrics in measured.values():
            metrics["response_recovery"] = (
                control_error - metrics["response_error"]
            ) / max(control_error, 1e-12)
        null_names = ("wrong_component", "wrong_time", "wrong_task", "negative", "random")
        correct_recovery = measured["correct"]["response_recovery"]
        coarse_recovery = measured["coarse"]["response_recovery"]
        null_recovery = max(measured[name]["response_recovery"] for name in null_names)
        eligible = bool(
            control_error >= CONTROL_THRESHOLDS["control_error_min"]
            and row["support_parameter_fraction"] <= CONTROL_THRESHOLDS["support_parameter_fraction_max"]
            and row["patch_update_fraction"] <= CONTROL_THRESHOLDS["patch_update_fraction_max"]
            and p1195.control_match_pass(row["control_match"])
            and row["partition"]["complete"]
        )
        row.update(
            {
                "wrong_task_trajectory_id": wrong_task_payload["trajectory_id"],
                "rescue_variants": measured,
                "rescue_control_error": control_error,
                "rescue_correct_recovery": correct_recovery,
                "coarse_recovery": coarse_recovery,
                "expanded_recovery_gain": correct_recovery - coarse_recovery,
                "rescue_null_recovery": null_recovery,
                "rescue_advantage": correct_recovery - null_recovery,
                "rescue_eligible": eligible,
            }
        )


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def rescue_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    eligible = [row for row in rows if row["rescue_eligible"]]
    return {
        "count": len(rows),
        "eligible_count": len(eligible),
        "correct_recovery_mean": mean(eligible, "rescue_correct_recovery"),
        "coarse_recovery_mean": mean(eligible, "coarse_recovery"),
        "expanded_recovery_gain_mean": mean(eligible, "expanded_recovery_gain"),
        "expanded_better_fraction": float(np.mean([row["expanded_recovery_gain"] > 0 for row in eligible])) if eligible else 0.0,
        "null_recovery_mean": mean(eligible, "rescue_null_recovery"),
        "advantage_mean": mean(eligible, "rescue_advantage"),
        "positive_fraction": float(np.mean([row["rescue_advantage"] > 0 for row in eligible])) if eligible else 0.0,
        "support_count_mean": mean(eligible, "support_count"),
        "support_parameter_fraction_mean": mean(eligible, "support_parameter_fraction"),
        "patch_update_fraction_mean": mean(eligible, "patch_update_fraction"),
        "calibration_cosine_mean": mean(eligible, "calibration_cosine"),
    }


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"no rows for split {split}")
    overall = rescue_group(selected)
    overall["eligible_fraction"] = overall["eligible_count"] / max(overall["count"], 1)
    by_architecture = {
        architecture: rescue_group([row for row in selected if row["architecture"] == architecture])
        for architecture in ARCHITECTURES
    }
    by_family = {
        family: rescue_group([row for row in selected if row["family"] == family])
        for family in ("affine", "bitmix", "random")
    }
    gate = bool(
        overall["eligible_fraction"] >= CONTROL_THRESHOLDS["eligible_fraction_min"]
        and overall["support_parameter_fraction_mean"] <= CONTROL_THRESHOLDS["support_parameter_fraction_mean_max"]
        and overall["patch_update_fraction_mean"] <= CONTROL_THRESHOLDS["patch_update_fraction_mean_max"]
        and overall["correct_recovery_mean"] >= RESCUE_THRESHOLDS["correct_recovery_mean_min"]
        and overall["advantage_mean"] >= RESCUE_THRESHOLDS["advantage_mean_min"]
        and overall["positive_fraction"] >= RESCUE_THRESHOLDS["positive_fraction_min"]
        and overall["expanded_recovery_gain_mean"] >= RESCUE_THRESHOLDS["expanded_minus_coarse_recovery_mean_min"]
        and overall["expanded_better_fraction"] >= RESCUE_THRESHOLDS["expanded_better_fraction_min"]
        and all(
            group["correct_recovery_mean"] >= RESCUE_THRESHOLDS["architecture_recovery_min"]
            and group["advantage_mean"] >= RESCUE_THRESHOLDS["architecture_advantage_min"]
            and group["positive_fraction"] >= RESCUE_THRESHOLDS["architecture_positive_fraction_min"]
            and group["expanded_recovery_gain_mean"] >= RESCUE_THRESHOLDS["architecture_gain_min"]
            for group in by_architecture.values()
        )
        and all(
            group["advantage_mean"] >= RESCUE_THRESHOLDS["family_advantage_min"]
            and group["expanded_recovery_gain_mean"] >= RESCUE_THRESHOLDS["family_gain_min"]
            for group in by_family.values()
        )
    )
    return {
        "split": split,
        "row_count": len(selected),
        "trajectory_count": len({row["trajectory_id"] for row in selected}),
        "rescue": overall,
        "rescue_by_architecture": by_architecture,
        "rescue_by_family": by_family,
        "rescue_gate_pass": gate,
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "phase1198": SCRIPT,
        "phase1198_audit": AUDIT_SCRIPT,
        "phase1197": p1197.SCRIPT,
        "phase1195": p1195.SCRIPT,
        "phase1194": p1194.SCRIPT,
        "phase1193": p1193.SCRIPT,
        "phase1146_model": ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def run_corpus(
    tasks: tuple[dict[str, Any], ...], replicates: int, corpus: str, device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for architecture in ARCHITECTURES:
            for replicate in range(replicates):
                row, payload = trajectory(task, task_index, architecture, replicate, corpus, device)
                rows.append(row)
                payloads.append(payload)
                print(canonical_json({"corpus": corpus, "task": task["name"], "architecture": architecture, "replicate": replicate, "rows": len(rows)}), flush=True)
    attach_metrics(rows, payloads, device)
    if corpus == "formal":
        replay_ids = {
            "expand_disc_affine_00::compact::r0",
            "expand_disc_affine_00::deep::r0",
            "expand_conf_affine_00::compact::r0",
            "expand_conf_affine_00::deep::r0",
        }
        REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            if payload["trajectory_id"] in replay_ids:
                torch.save(payload, REPLAY_ROOT / f"{payload['trajectory_id'].replace('::', '__')}.pt")
    del payloads
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def develop() -> None:
    if DEVELOPMENT_ROWS.exists() or DEVELOPMENT_SUMMARY.exists():
        raise RuntimeError("Phase1198 development outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    upstream = read_json(p1197.FINAL_PATH)
    if not upstream["authorized_next"]["expanded_partition_rescue_development"]:
        raise RuntimeError("Phase1197 did not authorize this development")
    rows = run_corpus(DEVELOPMENT_TASKS, DEVELOPMENT_REPLICATES, "development", torch.device("cuda"))
    summary = summarize(rows, "development")
    output = {
        "phase": PHASE,
        "kind": "authorized_development_only",
        "created_at": utc_now(),
        "development": summary,
        "development_gate_pass": summary["rescue_gate_pass"],
        "authorized_next": {"formal_preregistration": summary["rescue_gate_pass"]},
    }
    write_jsonl(DEVELOPMENT_ROWS, rows)
    write_json(DEVELOPMENT_SUMMARY, output)
    print(canonical_json({"development_gate_pass": output["development_gate_pass"], "rescue": summary["rescue"]}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("Phase1198 protocol or formal outcomes already exist")
    development = read_json(DEVELOPMENT_SUMMARY)
    development_audit = read_json(DEVELOPMENT_AUDIT)
    if not development["development_gate_pass"] or not development_audit.get("gate_pass", False):
        raise RuntimeError("development or its independent audit did not pass")
    upstream = read_json(p1197.FINAL_PATH)
    protocol = {
        "phase": PHASE,
        "created_at": utc_now(),
        "question": "Does adding the four Phase1197-identified omitted blocks to the otherwise frozen Phase1195 sparse controller produce selective immediate quotient-response rescue on independent tasks?",
        "scope": "Synthetic 32-class TinyTransformer update forks; immediate fixed-optimizer response only.",
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "formal_tasks": list(FORMAL_TASKS),
        "formal_replicates": FORMAL_REPLICATES,
        "rescue_stage": RESCUE_STAGE,
        "wrong_time_stage": WRONG_TIME_STAGE,
        "batch_size": BATCH_SIZE,
        "only_changed_variable": "control partition adds token_embedding, position_embedding, final_norm, and lm_head",
        "expanded_group_order": "original layer attention/MLP groups followed by token_embedding, position_embedding, final_norm, lm_head",
        "solver": {
            "basis_epsilon": BASIS_EPSILON,
            "regularization": REGULARIZATION,
            "iterations": SOLVER_ITERATIONS,
            "coefficient_domain": "0 <= alpha_g <= 1",
            "objective": "Phase1195 normalized calibration linear-response error plus weighted L1 coefficient cost",
            "support_epsilon": SUPPORT_EPSILON,
        },
        "nulls": [
            "same_norm_shifted_component_weights",
            "same_norm_wrong_time_expanded_coalition",
            "same_norm_wrong_task_expanded_coalition",
            "sign_reversed_correct_coalition",
            "same_support_same_norm_random",
        ],
        "control_thresholds": CONTROL_THRESHOLDS,
        "rescue_thresholds": RESCUE_THRESHOLDS,
        "continuation_rule": "Only a minimality/role-decomposition development study is authorized if the full rescue and gain gates pass independently in discovery and confirmation.",
        "forbidden": [
            "change solver, regularization, coefficient bounds, nulls, or thresholds after formal outcomes",
            "post-hoc top-k or lambda search",
            "drop an architecture, family, or negative-recovery event",
            "call immediate quotient-response rescue behavior or future-learning rescue",
            "call the selected coalition an endogenous network repair mechanism",
            "claim natural-language encoding mechanism",
        ],
        "development": {
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "audit_sha256": file_sha256(DEVELOPMENT_AUDIT),
        },
        "upstream": {
            "phase1197_final_sha256": file_sha256(p1197.FINAL_PATH),
            "phase1197_final_digest": upstream["final_digest"],
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
    if file_sha256(p1197.FINAL_PATH) != protocol["upstream"]["phase1197_final_sha256"]:
        raise RuntimeError("Phase1197 final changed")
    for key, path in (("rows_sha256", DEVELOPMENT_ROWS), ("summary_sha256", DEVELOPMENT_SUMMARY), ("audit_sha256", DEVELOPMENT_AUDIT)):
        if file_sha256(path) != protocol["development"][key]:
            raise RuntimeError(f"development asset changed: {path}")
    return protocol


def run_formal() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or RAW_ROWS.exists():
        raise RuntimeError("Phase1198 formal outcomes already exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = run_corpus(FORMAL_TASKS, FORMAL_REPLICATES, "formal", torch.device("cuda"))
    FORMAL_ROW_ROOT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_json(FORMAL_ROW_ROOT / f"{row['event_id'].replace('::', '__')}.json", row)
    write_jsonl(RAW_ROWS, rows)
    seal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "row_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "analysis_rows_sha256": file_sha256(RAW_ROWS),
        "row_manifest": {path.name: file_sha256(path) for path in sorted(FORMAL_ROW_ROOT.glob("*.json"))},
        "replay_manifest": {path.name: file_sha256(path) for path in sorted(REPLAY_ROOT.glob("*.pt"))},
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
    discovery = summarize(rows, "discovery")
    confirmation = summarize(rows, "confirmation")
    positive = discovery["rescue_gate_pass"] and confirmation["rescue_gate_pass"]
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "discovery": discovery,
        "confirmation": confirmation,
        "rescue_decision": "positive" if positive else "not_confirmed",
        "overall_status": "expanded_partition_sparse_rescue_confirmed" if positive else "expanded_partition_sparse_rescue_not_confirmed",
    }
    claims = {
        "expanded_partition_sparse_rescue": {
            "type": "E3-KT" if positive else "E3-KT-scope-boundary",
            "accepted": True,
            "claim": (
                "With solver, controls, and complexity gates frozen, adding the four omitted parameter blocks yields selective immediate quotient-response rescue and improves over the original coarse controller across both architectures, all task families, and independent splits."
                if positive
                else "The expanded partition did not satisfy the complete pre-registered rescue, selectivity, complexity, and coarse-improvement gate in both independent splits."
            ),
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"rescue": summary["rescue_decision"], "status": summary["overall_status"]}))


def replay_capsule(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    solution, _ = solve_payload(payload, device, int(payload["model_seed"]) + RESCUE_STAGE * 2003 + 43)
    variants = {
        "control": torch.zeros_like(payload["correct_patch"]),
        "correct": solution["patch"].cpu(),
        "coarse": payload["coarse_correct_patch"],
        "wrong_component": payload["wrong_component_patch"],
        "wrong_time": payload["wrong_time_patch"],
        "wrong_task": payload["wrong_task_patch"],
        "negative": payload["negative_patch"],
        "random": payload["random_patch"],
    }
    measured = {name: p1195.variant_metrics(payload, patch, device) for name, patch in variants.items()}
    return {
        "trajectory_id": payload["trajectory_id"],
        "alpha_max_error": float(np.max(np.abs(solution["alpha"] - np.asarray(payload["alpha"])))),
        "patch_relative_error": float((solution["patch"].cpu() - payload["correct_patch"]).norm() / payload["correct_patch"].norm().clamp_min(1e-12)),
        "measured": measured,
    }


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    positive = summary["rescue_decision"] == "positive"
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["overall_status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "formal_summary": summary,
        "authorized_next": {
            "expanded_basis_minimality_role_decomposition_development": positive,
            "fixed_optimizer_future_learning_rescue": False,
            "self_consistent_optimizer_rescue": False,
            "natural_language_encoding_claim": False,
        },
        "scope": {
            "confirmed": "immediate disjoint-panel quotient-response rescue by the expanded controller only if both formal gates passed",
            "not_claimed": [
                "behavior recovery",
                "future-learning recovery",
                "system controllability",
                "endogenous repair used by the network",
                "natural-language encoding mechanism",
            ],
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "run-formal", "analyze", "finalize"))
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
