from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1188_terminal_three_evidence_confirmation as p1188  # noqa: E402


PHASE = 1189
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1189_quotient_formation_operator_calibration_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1189_quotient_formation_operator_calibration"
DEVELOPMENT_ROWS = OUT_ROOT / "development/rows.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
RAW_ROWS = OUT_ROOT / "analysis/formation_rows.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

DEVELOPMENT_SOURCE = p1171.OUT_ROOT / "runs/training/checkpoints"
FORMAL_SOURCE = p1188.CHECKPOINT_ROOT
SOURCE_STEP = 10_000
EXPECTED_SYSTEMS = 64
EXPECTED_TASKS = 8
PROGRESS_SCALE = 1.02
REDISTRIBUTION = 0.30
PANEL_SEED_OFFSET = 1_189
CLASSIFICATION_THRESHOLD = 0.05
THRESHOLDS = {
    "source_system_count": EXPECTED_SYSTEMS,
    "source_task_count": EXPECTED_TASKS,
    "logit_equivalence_max": 2e-4,
    "prediction_agreement_min": 1.0,
    "loss_pair_difference_max": 1e-5,
    "update_norm_relative_error_max": 1e-5,
    "parameter_norm_relative_gap_max": 5e-3,
    "positive_calibration_transition_norm_min": 0.10,
    "positive_evaluation_transition_norm_min": 0.10,
    "control_transition_norm_max": 1e-4,
    "positive_transfer_cosine_min": 0.90,
    "positive_transfer_relative_error_max": 0.35,
    "gauge_transition_error_max": 2e-5,
    "classification_accuracy_min": 1.0,
    "systems_per_split_min": 32,
    "tasks_per_split_min": 4,
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def endpoint_paths(root: Path) -> list[Path]:
    return sorted(root.glob(f"*step{SOURCE_STEP:05d}.pt"))


def checkpoint_manifest(paths: list[Path]) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in paths}


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def panel_from_payload(payload: dict[str, Any]) -> p1181.DataPanel:
    data = p1171.make_data(
        tuple(payload["operation"]), int(payload["seed"]) + PANEL_SEED_OFFSET
    )
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    calibration = torch.zeros(len(x), dtype=torch.bool)
    calibration[: len(data["train_x"])] = True
    return p1181.DataPanel(
        x=x,
        y=y,
        train_mask=calibration,
        holdout_mask=~calibration,
    )


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def clone_model(
    model: p1171.RoleSquareNetwork, device: torch.device
) -> p1171.RoleSquareNetwork:
    cloned = p1171.RoleSquareNetwork(model.config).to(device)
    cloned.load_state_dict(model.state_dict())
    cloned.eval()
    return cloned


def expand_duplicate_pairs(
    model: p1171.RoleSquareNetwork, device: torch.device
) -> p1171.RoleSquareNetwork:
    old_width = model.config.width
    expanded = p1171.RoleSquareNetwork(
        p1171.RoleSquareConfig(model.config.modulus, old_width * 2)
    ).to(device)
    with torch.no_grad():
        expanded.left_embedding.weight.zero_()
        expanded.right_embedding.weight.zero_()
        expanded.left_embedding.weight[:, :old_width].copy_(model.left_embedding.weight)
        expanded.right_embedding.weight[:, :old_width].copy_(model.right_embedding.weight)
        expanded.hidden.weight.zero_()
        expanded.hidden.weight[0::2, :old_width].copy_(model.hidden.weight)
        expanded.hidden.weight[1::2, :old_width].copy_(model.hidden.weight)
        expanded.output.weight[:, 0::2].copy_(0.5 * model.output.weight)
        expanded.output.weight[:, 1::2].copy_(0.5 * model.output.weight)
    expanded.eval()
    return expanded


def positive_transition(
    base: p1171.RoleSquareNetwork,
    seed: int,
    device: torch.device,
) -> p1171.RoleSquareNetwork:
    target = clone_model(base, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(
        torch.rand(base.config.width // 2, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(device)
    with torch.no_grad():
        pair_total = base.output.weight[:, 0::2] + base.output.weight[:, 1::2]
        left_share = 0.5 + REDISTRIBUTION * signs
        target.output.weight[:, 0::2].copy_(
            PROGRESS_SCALE * pair_total * left_share[None, :]
        )
        target.output.weight[:, 1::2].copy_(
            PROGRESS_SCALE * pair_total * (1.0 - left_share)[None, :]
        )
    target.eval()
    return target


def update_norm(
    base: p1171.RoleSquareNetwork, target: p1171.RoleSquareNetwork
) -> float:
    total = sum(
        (right.detach().float() - left.detach().float()).square().sum()
        for left, right in zip(base.parameters(), target.parameters())
    )
    return float(torch.sqrt(total).item())


def parameter_norm(model: p1171.RoleSquareNetwork) -> float:
    total = sum(parameter.detach().float().square().sum() for parameter in model.parameters())
    return float(torch.sqrt(total).item())


def matched_gauge_control(
    base: p1171.RoleSquareNetwork,
    target_update_norm: float,
    target_parameter_norm: float,
    seed: int,
    device: torch.device,
) -> tuple[p1171.RoleSquareNetwork, tuple[float, float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(
        torch.rand(base.config.width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(device)

    def build(amplitude: float, embedding_log_scale: float) -> p1171.RoleSquareNetwork:
        target = clone_model(base, device)
        scale = torch.exp(amplitude * signs)
        embedding_scale = math.exp(embedding_log_scale)
        with torch.no_grad():
            target.left_embedding.weight.copy_(embedding_scale * base.left_embedding.weight)
            target.right_embedding.weight.copy_(embedding_scale * base.right_embedding.weight)
            target.hidden.weight.copy_(scale[:, None] * base.hidden.weight)
            target.output.weight.copy_(
                PROGRESS_SCALE
                * base.output.weight
                / (embedding_scale**2 * scale.square()[None, :])
            )
        target.eval()
        return target

    hidden_energy = base.hidden.weight.detach().float().square().sum(dim=1).cpu().numpy()
    output_energy = base.output.weight.detach().float().square().sum(dim=0).cpu().numpy()
    embedding_energy = float(
        base.left_embedding.weight.detach().float().square().sum().item()
        + base.right_embedding.weight.detach().float().square().sum().item()
    )
    sign_array = signs.cpu().numpy().astype(np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        scale = np.exp(values[0] * sign_array)
        embedding_scale = math.exp(float(values[1]))
        update_squared = float(
            (embedding_scale - 1.0) ** 2 * embedding_energy
            + np.sum((scale - 1.0) ** 2 * hidden_energy)
            + np.sum(
                (PROGRESS_SCALE / (embedding_scale**2 * scale**2) - 1.0) ** 2
                * output_energy
            )
        )
        final_squared = float(
            embedding_scale**2 * embedding_energy
            + np.sum(scale**2 * hidden_energy)
            + np.sum(
                PROGRESS_SCALE**2 / (embedding_scale**4 * scale**4) * output_energy
            )
        )
        return np.asarray(
            [
                (update_squared - target_update_norm**2) / target_update_norm**2,
                (final_squared - target_parameter_norm**2) / target_parameter_norm**2,
            ],
            dtype=np.float64,
        )

    starts = [
        np.asarray([first, second], dtype=np.float64)
        for first in (-0.30, -0.15, 0.0, 0.15, 0.30)
        for second in (-0.30, -0.15, 0.15, 0.30)
    ]
    fits = [
        least_squares(
            residual,
            x0=start,
            bounds=(np.asarray([-1.5, -1.5]), np.asarray([1.5, 1.5])),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=1_000,
        )
        for start in starts
    ]
    fit = min(fits, key=lambda item: float(np.linalg.norm(residual(item.x))))
    fit_error = float(np.max(np.abs(residual(fit.x))))
    if fit_error > 1e-8:
        raise RuntimeError(f"joint nuisance matching failed: {fit_error}")
    amplitudes = (float(fit.x[0]), float(fit.x[1]))
    return build(*amplitudes), amplitudes


@torch.inference_mode()
def fp32_logits(model: p1171.RoleSquareNetwork, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return p1181.fp32_state(model, x, device)[0]


@torch.inference_mode()
def response_unit_shape(
    model: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    selected_mask: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    logits, hidden = p1181.fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    selected = selected_mask.to(device)
    baseline_margin = p1181.correct_margin(logits, targets)
    squared = hidden.square()
    output_weight = model.output.weight.detach().float()
    responses: list[float] = []
    for start in range(0, hidden.shape[1], 32):
        stop = min(start + 32, hidden.shape[1])
        channels = torch.arange(start, stop, device=device)
        contribution = (
            squared[:, channels].transpose(0, 1)[:, :, None]
            * output_weight[:, channels].transpose(0, 1)[:, None, :]
        )
        changed_logits = logits[None] - contribution
        flat_targets = targets.repeat(stop - start)
        changed_margin = p1181.correct_margin(
            changed_logits.reshape(-1, changed_logits.shape[-1]), flat_targets
        ).reshape(stop - start, -1)
        effect = (baseline_margin[None, selected] - changed_margin[:, selected]).mean(dim=1)
        responses.extend(float(value) for value in effect.cpu())
    ordered = np.sort(np.asarray(responses, dtype=np.float64))
    centered = ordered - ordered.mean()
    return centered / max(float(np.linalg.norm(centered)), 1e-12)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


def relative_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(float(np.linalg.norm(target)), 1e-12))


def gauge_model(
    model: p1171.RoleSquareNetwork, seed: int, device: torch.device
) -> p1171.RoleSquareNetwork:
    return p1181.gauge_model(model, seed, device)


def transition_record(
    label: str,
    base: p1171.RoleSquareNetwork,
    target: p1171.RoleSquareNetwork,
    panel: p1181.DataPanel,
    base_calibration: np.ndarray,
    base_evaluation: np.ndarray,
    gauge_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    calibration = response_unit_shape(target, panel, panel.train_mask, device) - base_calibration
    evaluation = response_unit_shape(target, panel, panel.holdout_mask, device) - base_evaluation
    gauged_base = gauge_model(base, gauge_seed, device)
    gauged_target = gauge_model(target, gauge_seed, device)
    gauged_calibration = (
        response_unit_shape(gauged_target, panel, panel.train_mask, device)
        - response_unit_shape(gauged_base, panel, panel.train_mask, device)
    )
    gauged_evaluation = (
        response_unit_shape(gauged_target, panel, panel.holdout_mask, device)
        - response_unit_shape(gauged_base, panel, panel.holdout_mask, device)
    )
    result = {
        "label": label,
        "calibration_delta": calibration.tolist(),
        "evaluation_delta": evaluation.tolist(),
        "calibration_norm": float(np.linalg.norm(calibration)),
        "evaluation_norm": float(np.linalg.norm(evaluation)),
        "wasserstein2_calibration": float(np.linalg.norm(calibration) / math.sqrt(len(calibration))),
        "wasserstein2_evaluation": float(np.linalg.norm(evaluation) / math.sqrt(len(evaluation))),
        "calibration_to_evaluation_cosine": cosine(calibration, evaluation),
        "calibration_to_evaluation_relative_error": relative_error(calibration, evaluation),
        "gauge_calibration_max_error": float(np.max(np.abs(calibration - gauged_calibration))),
        "gauge_evaluation_max_error": float(np.max(np.abs(evaluation - gauged_evaluation))),
        "classified_positive": bool(float(np.linalg.norm(calibration)) >= CLASSIFICATION_THRESHOLD),
    }
    del gauged_base, gauged_target
    return result


def loss_for(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, device: torch.device) -> float:
    return float(F.cross_entropy(logits[mask.to(device)], targets.to(device)[mask.to(device)]).item())


def build_record(path: Path, corpus: str, device: torch.device) -> dict[str, Any]:
    payload = load_payload(path)
    panel = panel_from_payload(payload)
    original = load_model(payload, device)
    base = expand_duplicate_pairs(original, device)
    index_seed = int(payload["task_index"]) * 10_000 + int(payload["replicate"])
    positive = positive_transition(base, 1_189_100_000 + index_seed, device)
    positive_update_norm = update_norm(base, positive)
    positive_parameter_norm = parameter_norm(positive)
    control, control_amplitude = matched_gauge_control(
        base,
        positive_update_norm,
        positive_parameter_norm,
        1_189_200_000 + index_seed,
        device,
    )
    base_logits = fp32_logits(base, panel.x, device)
    positive_logits = fp32_logits(positive, panel.x, device)
    control_logits = fp32_logits(control, panel.x, device)
    expected_logits = PROGRESS_SCALE * base_logits
    targets = panel.y.to(device)
    base_calibration = response_unit_shape(base, panel, panel.train_mask, device)
    base_evaluation = response_unit_shape(base, panel, panel.holdout_mask, device)
    positive_record = transition_record(
        "mechanism_changing_redistribution",
        base,
        positive,
        panel,
        base_calibration,
        base_evaluation,
        1_189_300_000 + index_seed,
        device,
    )
    control_record = transition_record(
        "mechanism_preserving_channel_gauge",
        base,
        control,
        panel,
        base_calibration,
        base_evaluation,
        1_189_400_000 + index_seed,
        device,
    )
    control_parameter_norm = parameter_norm(control)
    positive_control_parameter_gap = abs(positive_parameter_norm - control_parameter_norm)
    result = {
        "corpus": corpus,
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "task_name": str(payload["task_name"]),
        "task_index": int(payload["task_index"]),
        "operation": [int(value) for value in payload["operation"]],
        "replicate": int(payload["replicate"]),
        "seed": int(payload["seed"]),
        "split": "development" if corpus == "development" else (
            "discovery" if int(payload["task_index"]) < 4 else "confirmation"
        ),
        "base_expansion_logit_error": float(
            (fp32_logits(original, panel.x, device) - base_logits).abs().max().item()
        ),
        "positive_expected_logit_error": float((positive_logits - expected_logits).abs().max().item()),
        "control_expected_logit_error": float((control_logits - expected_logits).abs().max().item()),
        "positive_control_logit_error": float((positive_logits - control_logits).abs().max().item()),
        "positive_prediction_agreement": float(
            (positive_logits.argmax(dim=1) == base_logits.argmax(dim=1)).float().mean().item()
        ),
        "control_prediction_agreement": float(
            (control_logits.argmax(dim=1) == base_logits.argmax(dim=1)).float().mean().item()
        ),
        "base_calibration_loss": loss_for(base_logits, targets, panel.train_mask, device),
        "base_evaluation_loss": loss_for(base_logits, targets, panel.holdout_mask, device),
        "positive_calibration_loss": loss_for(positive_logits, targets, panel.train_mask, device),
        "positive_evaluation_loss": loss_for(positive_logits, targets, panel.holdout_mask, device),
        "control_calibration_loss": loss_for(control_logits, targets, panel.train_mask, device),
        "control_evaluation_loss": loss_for(control_logits, targets, panel.holdout_mask, device),
        "positive_update_norm": positive_update_norm,
        "control_update_norm": update_norm(base, control),
        "update_norm_relative_error": abs(update_norm(base, control) - positive_update_norm)
        / max(positive_update_norm, 1e-12),
        "base_parameter_norm": parameter_norm(base),
        "positive_parameter_norm": positive_parameter_norm,
        "control_parameter_norm": control_parameter_norm,
        "positive_control_parameter_norm_relative_gap": positive_control_parameter_gap
        / max(positive_parameter_norm, 1e-12),
        "control_amplitude": control_amplitude,
        "positive": positive_record,
        "control": control_record,
    }
    del original, base, positive, control
    return result


def system_pass(row: dict[str, Any]) -> bool:
    t = THRESHOLDS
    return all(
        (
            row["base_expansion_logit_error"] <= t["logit_equivalence_max"],
            row["positive_expected_logit_error"] <= t["logit_equivalence_max"],
            row["control_expected_logit_error"] <= t["logit_equivalence_max"],
            row["positive_control_logit_error"] <= t["logit_equivalence_max"],
            row["positive_prediction_agreement"] >= t["prediction_agreement_min"],
            row["control_prediction_agreement"] >= t["prediction_agreement_min"],
            abs(row["positive_calibration_loss"] - row["control_calibration_loss"])
            <= t["loss_pair_difference_max"],
            abs(row["positive_evaluation_loss"] - row["control_evaluation_loss"])
            <= t["loss_pair_difference_max"],
            row["update_norm_relative_error"] <= t["update_norm_relative_error_max"],
            row["positive_control_parameter_norm_relative_gap"]
            <= t["parameter_norm_relative_gap_max"],
            row["positive"]["calibration_norm"]
            >= t["positive_calibration_transition_norm_min"],
            row["positive"]["evaluation_norm"]
            >= t["positive_evaluation_transition_norm_min"],
            row["control"]["calibration_norm"] <= t["control_transition_norm_max"],
            row["control"]["evaluation_norm"] <= t["control_transition_norm_max"],
            row["positive"]["calibration_to_evaluation_cosine"]
            >= t["positive_transfer_cosine_min"],
            row["positive"]["calibration_to_evaluation_relative_error"]
            <= t["positive_transfer_relative_error_max"],
            row["positive"]["gauge_calibration_max_error"] <= t["gauge_transition_error_max"],
            row["positive"]["gauge_evaluation_max_error"] <= t["gauge_transition_error_max"],
            row["control"]["gauge_calibration_max_error"] <= t["gauge_transition_error_max"],
            row["control"]["gauge_evaluation_max_error"] <= t["gauge_transition_error_max"],
            row["positive"]["classified_positive"],
            not row["control"]["classified_positive"],
        )
    )


def summarize_rows(rows: list[dict[str, Any]], expected_split: str | None = None) -> dict[str, Any]:
    selected = rows if expected_split is None else [row for row in rows if row["split"] == expected_split]
    if not selected:
        return {"split": expected_split, "system_count": 0, "task_count": 0, "gate_pass": False}
    classification_correct = sum(
        int(row["positive"]["classified_positive"]) + int(not row["control"]["classified_positive"])
        for row in selected
    )
    system_pass_count = sum(system_pass(row) for row in selected)
    task_names = sorted({row["task_name"] for row in selected})
    result = {
        "split": expected_split or "all",
        "system_count": len(selected),
        "task_count": len(task_names),
        "task_names": task_names,
        "system_pass_count": system_pass_count,
        "classification_correct": classification_correct,
        "classification_total": 2 * len(selected),
        "classification_accuracy": classification_correct / (2 * len(selected)),
        "max_logit_equivalence_error": max(
            max(
                row["base_expansion_logit_error"],
                row["positive_expected_logit_error"],
                row["control_expected_logit_error"],
                row["positive_control_logit_error"],
            )
            for row in selected
        ),
        "min_prediction_agreement": min(
            min(row["positive_prediction_agreement"], row["control_prediction_agreement"])
            for row in selected
        ),
        "max_loss_pair_difference": max(
            max(
                abs(row["positive_calibration_loss"] - row["control_calibration_loss"]),
                abs(row["positive_evaluation_loss"] - row["control_evaluation_loss"]),
            )
            for row in selected
        ),
        "max_update_norm_relative_error": max(row["update_norm_relative_error"] for row in selected),
        "max_parameter_norm_relative_gap": max(
            row["positive_control_parameter_norm_relative_gap"] for row in selected
        ),
        "positive_calibration_norm_min": min(
            row["positive"]["calibration_norm"] for row in selected
        ),
        "positive_calibration_norm_mean": float(
            np.mean([row["positive"]["calibration_norm"] for row in selected])
        ),
        "positive_evaluation_norm_min": min(row["positive"]["evaluation_norm"] for row in selected),
        "positive_evaluation_norm_mean": float(
            np.mean([row["positive"]["evaluation_norm"] for row in selected])
        ),
        "control_transition_norm_max": max(
            max(row["control"]["calibration_norm"], row["control"]["evaluation_norm"])
            for row in selected
        ),
        "positive_transfer_cosine_min": min(
            row["positive"]["calibration_to_evaluation_cosine"] for row in selected
        ),
        "positive_transfer_cosine_mean": float(
            np.mean([row["positive"]["calibration_to_evaluation_cosine"] for row in selected])
        ),
        "positive_transfer_relative_error_max": max(
            row["positive"]["calibration_to_evaluation_relative_error"] for row in selected
        ),
        "positive_transfer_relative_error_mean": float(
            np.mean(
                [row["positive"]["calibration_to_evaluation_relative_error"] for row in selected]
            )
        ),
        "gauge_transition_error_max": max(
            max(
                row[k][field]
                for k in ("positive", "control")
                for field in ("gauge_calibration_max_error", "gauge_evaluation_max_error")
            )
            for row in selected
        ),
        "mean_calibration_loss_change": float(
            np.mean([row["positive_calibration_loss"] - row["base_calibration_loss"] for row in selected])
        ),
        "mean_evaluation_loss_change": float(
            np.mean([row["positive_evaluation_loss"] - row["base_evaluation_loss"] for row in selected])
        ),
    }
    minimum_systems = EXPECTED_SYSTEMS if expected_split in (None, "development") else THRESHOLDS["systems_per_split_min"]
    minimum_tasks = EXPECTED_TASKS if expected_split in (None, "development") else THRESHOLDS["tasks_per_split_min"]
    result["gate_pass"] = bool(
        len(selected) >= minimum_systems
        and len(task_names) >= minimum_tasks
        and system_pass_count == len(selected)
        and result["classification_accuracy"] >= THRESHOLDS["classification_accuracy_min"]
    )
    return result


def run_corpus(paths: list[Path], corpus: str, output: Path) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        rows.append(build_record(path, corpus, device))
        print(
            canonical_json(
                {
                    "corpus": corpus,
                    "completed": index + 1,
                    "total": len(paths),
                    "checkpoint": path.name,
                    "system_pass": system_pass(rows[-1]),
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    write_jsonl(output, rows)
    return rows


def source_hashes() -> dict[str, str]:
    paths = [SCRIPT, AUDIT_SCRIPT, Path(p1171.__file__), Path(p1181.__file__), Path(p1187.__file__), Path(p1188.__file__)]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def develop() -> None:
    paths = endpoint_paths(DEVELOPMENT_SOURCE)
    if len(paths) != EXPECTED_SYSTEMS:
        raise RuntimeError(f"expected {EXPECTED_SYSTEMS} development endpoints, found {len(paths)}")
    rows = run_corpus(paths, "development", DEVELOPMENT_ROWS)
    summary = summarize_rows(rows, "development")
    summary.update(
        {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "source_manifest": checkpoint_manifest(paths),
            "source_manifest_digest": digest(checkpoint_manifest(paths)),
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "thresholds": THRESHOLDS,
            "formal_data_read": False,
            "interpretation": "Historical Phase1171 systems calibrate a fixed finite quotient-transition instrument only.",
        }
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(DEVELOPMENT_SUMMARY, summary)
    if not summary["gate_pass"]:
        raise RuntimeError("development calibration failed; formal reveal is forbidden")


def preregister() -> None:
    if not DEVELOPMENT_SUMMARY.exists():
        raise RuntimeError("development calibration is missing")
    development = read_json(DEVELOPMENT_SUMMARY)
    if not development["gate_pass"]:
        raise RuntimeError("development gate did not pass")
    paths = endpoint_paths(FORMAL_SOURCE)
    if len(paths) != EXPECTED_SYSTEMS:
        raise RuntimeError(f"expected {EXPECTED_SYSTEMS} formal endpoints, found {len(paths)}")
    payloads = [load_payload(path) for path in paths]
    tasks = sorted({str(payload["task_name"]) for payload in payloads})
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError("formal task count changed")
    manifest = checkpoint_manifest(paths)
    protocol = {
        "phase": PHASE,
        "title": "Quotient formation operator known-truth calibration",
        "created_at_utc": utc_now(),
        "scientific_question": (
            "Can a gauge-invariant finite response transition distinguish a behavior-, loss-, output-, "
            "and update-norm-matched mechanism-changing update from a mechanism-preserving gauge update, "
            "and transfer from a calibration half to an unseen evaluation half?"
        ),
        "object": {
            "quotient_state": "sorted centered unit channel-ablation response spectrum",
            "formation_event": "finite quantile-matched endpoint difference; no derivative claim",
            "positive": "duplicate-channel load redistribution with exact pairwise function preservation",
            "negative": (
                "positive hidden-channel plus global embedding rescaling with exact output compensation; "
                "two gauge degrees jointly match update and endpoint parameter norms"
            ),
            "common_factor": f"both endpoints multiply logits by {PROGRESS_SCALE}",
            "nuisance_matching": [
                "pointwise final logits",
                "predictions",
                "cross-entropy",
                "raw update L2 norm",
                "final parameter norm within a frozen relative tolerance",
            ],
            "gauge_group": "signed hidden-channel permutations",
        },
        "development": {
            "source": str(DEVELOPMENT_SOURCE.relative_to(ROOT)),
            "summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "summary_digest": development["summary_digest"],
            "system_count": development["system_count"],
            "task_count": development["task_count"],
        },
        "formal": {
            "source": str(FORMAL_SOURCE.relative_to(ROOT)),
            "checkpoint_manifest": manifest,
            "checkpoint_manifest_digest": digest(manifest),
            "system_count": len(paths),
            "task_names": tasks,
            "discovery_task_names": tasks[:4],
            "confirmation_task_names": tasks[4:],
            "allocation_digest": digest({"discovery": tasks[:4], "confirmation": tasks[4:]}),
            "outcomes_absent": not RAW_ROWS.exists(),
        },
        "constants": {
            "progress_scale": PROGRESS_SCALE,
            "redistribution": REDISTRIBUTION,
            "panel_seed_offset": PANEL_SEED_OFFSET,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
        },
        "thresholds": THRESHOLDS,
        "source_hashes": source_hashes(),
        "evidence_contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "hard_stops": [
            "No formal task, seed, endpoint, response, or label may tune a threshold or transformation.",
            "A failed formal split closes this exact formation-operator camera without feature search.",
            "Sorting is finite one-dimensional optimal matching, not a differentiable rank identity.",
            "Raw parameter updates are nuisance descriptors, not cross-gauge mechanism coordinates.",
            "A pass calibrates an instrument in a duplicated-channel RoleSquare known-truth system only.",
            "Frozen Qwen3, GLM4, and DS7B cannot test training formation and are excluded.",
        ],
        "authorization": {
            "phase1190_free_training_preregistration": "both formal splits and independent audit must pass",
            "transformer_or_llm_transfer": False,
            "theory_closure": False,
        },
    }
    protocol["protocol_digest"] = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    write_json(PROTOCOL_PATH, protocol)


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    expected_digest = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    if expected_digest != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source code changed after preregistration")
    if file_sha256(DEVELOPMENT_SUMMARY) != protocol["development"]["summary_sha256"]:
        raise RuntimeError("development summary changed")
    if file_sha256(DEVELOPMENT_ROWS) != protocol["development"]["rows_sha256"]:
        raise RuntimeError("development rows changed")
    if file_sha256(p1187.CONTRACT_PATH) != protocol["evidence_contract_sha256"]:
        raise RuntimeError("evidence contract changed")
    paths = endpoint_paths(FORMAL_SOURCE)
    if checkpoint_manifest(paths) != protocol["formal"]["checkpoint_manifest"]:
        raise RuntimeError("formal checkpoint manifest changed")
    return protocol


def bounded(value: float, threshold: float, comparator: str) -> dict[str, Any]:
    return {
        "claim_type": "bounded_float",
        "gating": True,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "dtype": "float64",
    }


def universal(agree: int, eligible: int) -> dict[str, Any]:
    return {
        "claim_type": "universal_boolean",
        "gating": True,
        "agree_count": int(agree),
        "eligible_count": int(eligible),
        "abstained": eligible == 0,
    }


def compile_claims(summary: dict[str, Any]) -> dict[str, Any]:
    contract = read_json(p1187.CONTRACT_PATH)
    raw: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "confirmation"):
        current = summary[split]
        prefix = split + "."
        raw[prefix + "all_systems"] = universal(current["system_pass_count"], current["system_count"])
        raw[prefix + "classification"] = bounded(
            current["classification_accuracy"], THRESHOLDS["classification_accuracy_min"], ">="
        )
        raw[prefix + "positive_calibration"] = bounded(
            current["positive_calibration_norm_min"],
            THRESHOLDS["positive_calibration_transition_norm_min"],
            ">=",
        )
        raw[prefix + "positive_evaluation"] = bounded(
            current["positive_evaluation_norm_min"],
            THRESHOLDS["positive_evaluation_transition_norm_min"],
            ">=",
        )
        raw[prefix + "control_null"] = bounded(
            current["control_transition_norm_max"], THRESHOLDS["control_transition_norm_max"], "<="
        )
        raw[prefix + "transfer_cosine"] = bounded(
            current["positive_transfer_cosine_min"], THRESHOLDS["positive_transfer_cosine_min"], ">="
        )
        raw[prefix + "transfer_error"] = bounded(
            current["positive_transfer_relative_error_max"],
            THRESHOLDS["positive_transfer_relative_error_max"],
            "<=",
        )
        raw[prefix + "gauge_invariance"] = bounded(
            current["gauge_transition_error_max"], THRESHOLDS["gauge_transition_error_max"], "<="
        )
    compiled = {name: p1187.compile_claim(claim, contract) for name, claim in raw.items()}
    values = [bool(claim["authorizes"]) for claim in compiled.values()]
    conjunction = p1187.compile_claim(
        {"claim_type": "conjunction", "gating": True, "values": values}, contract
    )
    return {
        "contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "raw": raw,
        "compiled": compiled,
        "conjunction": conjunction,
        "gate_pass": bool(conjunction["authorizes"]),
    }


def confirm() -> None:
    protocol = verify_protocol()
    paths = endpoint_paths(FORMAL_SOURCE)
    rows = run_corpus(paths, "formal", RAW_ROWS)
    discovery = summarize_rows(rows, "discovery")
    confirmation = summarize_rows(rows, "confirmation")
    overall = summarize_rows(rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "checkpoint_manifest_digest": protocol["formal"]["checkpoint_manifest_digest"],
        "raw_rows_sha256": file_sha256(RAW_ROWS),
        "discovery": discovery,
        "confirmation": confirmation,
        "overall": overall,
        "formal_gate_pass": bool(discovery["gate_pass"] and confirmation["gate_pass"]),
    }
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(SUMMARY_PATH, summary)
    claims = compile_claims(summary)
    write_json(CLAIMS_PATH, claims)


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit_pass = bool(AUDIT_PATH.exists() and read_json(AUDIT_PATH).get("gate_pass"))
    main_pass = bool(summary["formal_gate_pass"] and claims["gate_pass"] and audit_pass)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "known_truth_formation_operator_calibrated" if main_pass else (
            "awaiting_independent_audit" if summary["formal_gate_pass"] and claims["gate_pass"] and not AUDIT_PATH.exists()
            else "known_truth_formation_operator_failed"
        ),
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(CLAIMS_PATH),
        "audit_digest": read_json(AUDIT_PATH).get("audit_digest") if AUDIT_PATH.exists() else None,
        "formal_gate_pass": summary["formal_gate_pass"],
        "typed_claim_gate_pass": claims["gate_pass"],
        "independent_audit_pass": audit_pass,
        "main_gate_pass": main_pass,
        "evidence_grade": "E3_KT_instrument" if main_pass else "no_upgrade",
        "authorized_next": {
            "phase1190_free_training_formation_preregistration": main_pass,
            "automatic_unpreregistered_execution": False,
            "transformer_or_language_model_transfer": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "A finite signed-permutation-invariant response transition distinguishes and transfers a constructed "
            "mechanism-changing update from a matched mechanism-preserving gauge update in duplicated-channel "
            "RoleSquare systems. It does not show that natural SGD uses this transition or that the same object "
            "exists in Transformers."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(FINAL_PATH, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "confirm", "finalize", "all"))
    args = parser.parse_args()
    if args.command in ("develop", "all"):
        develop()
    if args.command in ("preregister", "all"):
        preregister()
    if args.command in ("confirm", "all"):
        confirm()
    if args.command in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
