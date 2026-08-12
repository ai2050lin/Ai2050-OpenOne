from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_gate as p1181  # noqa: E402


SOURCE = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography/runs/training/checkpoints"
DEVICE = torch.device("cuda")
PROGRESS_SCALE = 1.02
REDISTRIBUTION = 0.30


def load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def panel(payload: dict) -> p1181.DataPanel:
    data = p1171.make_data(tuple(payload["operation"]), int(payload["seed"]))
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    train_mask[: len(data["train_x"])] = True
    return p1181.DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def load_model(payload: dict) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def expand_pairs(model: p1171.RoleSquareNetwork) -> p1171.RoleSquareNetwork:
    old_width = model.config.width
    expanded = p1171.RoleSquareNetwork(
        p1171.RoleSquareConfig(modulus=model.config.modulus, width=old_width * 2)
    ).to(DEVICE)
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


def clone_model(model: p1171.RoleSquareNetwork) -> p1171.RoleSquareNetwork:
    cloned = p1171.RoleSquareNetwork(model.config).to(DEVICE)
    cloned.load_state_dict(model.state_dict())
    cloned.eval()
    return cloned


def positive_transition(base: p1171.RoleSquareNetwork, seed: int) -> p1171.RoleSquareNetwork:
    target = clone_model(base)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.where(
        torch.rand(base.config.width // 2, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(DEVICE)
    with torch.no_grad():
        pair_total = base.output.weight[:, 0::2] + base.output.weight[:, 1::2]
        left_share = 0.5 + REDISTRIBUTION * signs
        target.output.weight[:, 0::2].copy_(PROGRESS_SCALE * pair_total * left_share[None, :])
        target.output.weight[:, 1::2].copy_(PROGRESS_SCALE * pair_total * (1.0 - left_share)[None, :])
    return target


def update_norm(base: p1171.RoleSquareNetwork, target: p1171.RoleSquareNetwork) -> float:
    return float(
        torch.sqrt(
            sum(
                (right.detach().float() - left.detach().float()).square().sum()
                for left, right in zip(base.parameters(), target.parameters())
            )
        ).item()
    )


def rescaled_control(
    base: p1171.RoleSquareNetwork,
    target_norm: float,
    target_parameter_norm: float,
    seed: int,
) -> tuple[p1171.RoleSquareNetwork, tuple[float, float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs_a = torch.where(
        torch.rand(base.config.width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).to(DEVICE)
    def build(amplitude: float, embedding_log_scale: float) -> p1171.RoleSquareNetwork:
        target = clone_model(base)
        scale = torch.exp(amplitude * signs_a)
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
        return target

    hidden_energy = base.hidden.weight.detach().float().square().sum(dim=1).cpu().numpy()
    output_energy = base.output.weight.detach().float().square().sum(dim=0).cpu().numpy()
    embedding_energy = float(
        base.left_embedding.weight.detach().float().square().sum().item()
        + base.right_embedding.weight.detach().float().square().sum().item()
    )
    sign_a = signs_a.cpu().numpy().astype(np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        scale = np.exp(values[0] * sign_a)
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
                (update_squared - target_norm**2) / target_norm**2,
                (final_squared - target_parameter_norm**2) / target_parameter_norm**2,
            ]
        )

    starts = [
        np.asarray([a, b], dtype=np.float64)
        for a in (-0.3, -0.15, 0.0, 0.15, 0.3)
        for b in (-0.3, -0.15, 0.15, 0.3)
    ]
    fits = [
        least_squares(
            residual,
            x0=start,
            bounds=(np.asarray([-1.5, -1.5]), np.asarray([1.5, 1.5])),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=1000,
        )
        for start in starts
    ]
    fit = min(fits, key=lambda item: float(np.linalg.norm(residual(item.x))))
    error = float(np.max(np.abs(residual(fit.x))))
    if error > 1e-8:
        raise RuntimeError(f"unable to jointly match nuisance norms: {error}")
    result = build(float(fit.x[0]), float(fit.x[1]))
    return result, (float(fit.x[0]), float(fit.x[1]))


@torch.inference_mode()
def logits(model: p1171.RoleSquareNetwork, x: torch.Tensor) -> torch.Tensor:
    return model(x.to(DEVICE)).float()


@torch.inference_mode()
def response(
    model: p1171.RoleSquareNetwork,
    data: p1181.DataPanel,
    mask: torch.Tensor,
) -> np.ndarray:
    raw_logits, hidden = p1181.fp32_state(model, data.x, DEVICE)
    targets = data.y.to(DEVICE)
    selected = mask.to(DEVICE)
    baseline = p1181.correct_margin(raw_logits, targets)
    squared = hidden.square()
    output = model.output.weight.detach().float()
    values: list[float] = []
    for start in range(0, model.config.width, 32):
        stop = min(start + 32, model.config.width)
        channels = torch.arange(start, stop, device=DEVICE)
        contribution = (
            squared[:, channels].transpose(0, 1)[:, :, None]
            * output[:, channels].transpose(0, 1)[:, None, :]
        )
        changed = raw_logits[None] - contribution
        flat_targets = targets.repeat(stop - start)
        margins = p1181.correct_margin(
            changed.reshape(-1, changed.shape[-1]), flat_targets
        ).reshape(stop - start, -1)
        effects = (baseline[None, selected] - margins[:, selected]).mean(dim=1)
        values.extend(float(value) for value in effects.cpu())
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    centered = ordered - ordered.mean()
    return centered / max(float(np.linalg.norm(centered)), 1e-12)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denom, 1e-12))


def parameter_norm(model: p1171.RoleSquareNetwork) -> float:
    return float(
        torch.sqrt(
            sum(parameter.detach().float().square().sum() for parameter in model.parameters())
        ).item()
    )


def main() -> None:
    paths = sorted(SOURCE.glob("*step10000.pt"))[:8]
    rows = []
    for index, path in enumerate(paths):
        payload = load_payload(path)
        data = panel(payload)
        original = load_model(payload)
        base = expand_pairs(original)
        positive = positive_transition(base, 11890000 + index)
        target_norm = update_norm(base, positive)
        target_parameter_norm = parameter_norm(positive)
        control, amplitude = rescaled_control(
            base, target_norm, target_parameter_norm, 11891000 + index
        )
        base_logits = logits(base, data.x)
        positive_logits = logits(positive, data.x)
        control_logits = logits(control, data.x)
        base_train = response(base, data, data.train_mask)
        base_holdout = response(base, data, data.holdout_mask)
        for label, target in (("positive", positive), ("control", control)):
            delta_train = response(target, data, data.train_mask) - base_train
            delta_holdout = response(target, data, data.holdout_mask) - base_holdout
            rows.append(
                {
                    "checkpoint": path.name,
                    "label": label,
                    "train_norm": float(np.linalg.norm(delta_train)),
                    "holdout_norm": float(np.linalg.norm(delta_holdout)),
                    "train_holdout_cosine": cosine(delta_train, delta_holdout),
                    "update_norm": update_norm(base, target),
                    "positive_control_norm_error": abs(update_norm(base, target) - target_norm),
                    "parameter_norm": parameter_norm(target),
                    "positive_control_parameter_norm_error": abs(
                        parameter_norm(target) - target_parameter_norm
                    ),
                    "logit_error": float(
                        (logits(target, data.x) - PROGRESS_SCALE * base_logits).abs().max().item()
                    ),
                    "prediction_agreement": float(
                        (
                            logits(target, data.x).argmax(dim=1)
                            == base_logits.argmax(dim=1)
                        ).float().mean().item()
                    ),
                    "control_amplitude": amplitude if label == "control" else None,
                }
            )
        del original, base, positive, control, base_logits, positive_logits, control_logits
        torch.cuda.empty_cache()
    print(json.dumps(rows, indent=2))
    for label in ("positive", "control"):
        subset = [row for row in rows if row["label"] == label]
        print(
            label,
            {
                key: float(np.mean([row[key] for row in subset]))
                for key in ("train_norm", "holdout_norm", "train_holdout_cosine", "logit_error")
            },
        )


if __name__ == "__main__":
    main()
