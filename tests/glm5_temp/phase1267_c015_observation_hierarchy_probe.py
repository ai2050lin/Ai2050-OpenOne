from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266
from phase1146_learned_composition_benchmark import ModelConfig


RANK_COMPONENT = 8
RANK_OUTPUT = 16
RFF_WIDTH = 256
RIDGE = 1.0e-5


def basis(x: torch.Tensor, rank: int) -> torch.Tensor:
    x = x.double()
    _u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() == 0 or float(s[0]) <= 1.0e-12:
        return torch.zeros((x.shape[1], 0), dtype=torch.float64, device=x.device)
    keep = min(rank, int((s > s[0] * 1.0e-7).sum().item()))
    return vh[:keep].T.contiguous()


def raw_observation(capture: dict, layer: int, level: str) -> list[torch.Tensor]:
    states = capture["states"]
    if level == "delta":
        return [states["h10"][:, layer] - states["h00"][:, layer], states["h01"][:, layer] - states["h00"][:, layer]]
    if level == "state":
        return [states["h00"][:, layer], states["h10"][:, layer] - states["h00"][:, layer], states["h01"][:, layer] - states["h00"][:, layer]]
    if level == "trajectory":
        values = []
        for index in range(layer + 1):
            values.extend([states["h00"][:, index], states["h10"][:, index] - states["h00"][:, index], states["h01"][:, index] - states["h00"][:, index]])
        return values
    raise ValueError(level)


def feature(z: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((z.shape[0], 1), dtype=torch.float64, device=z.device)
    phi = math.sqrt(1.0 / RFF_WIDTH) * torch.cat((torch.sin(z @ omega + phase), torch.cos(z @ omega + phase)), dim=1)
    return torch.cat((ones, z, phi), dim=1)


def fit(discovery: dict, layer: int, level: str) -> dict:
    components = raw_observation(discovery, layer, level)
    bases = [basis(value, RANK_COMPONENT) for value in components]
    z = torch.cat([value.double() @ component_basis for value, component_basis in zip(components, bases)], dim=1)
    mean, std = z.mean(0), z.std(0).clamp_min(1.0e-5)
    z = (z - mean) / std
    generator = torch.Generator(device=z.device)
    generator.manual_seed(1_267_070 + 101 * layer + 7 * len(bases))
    omega = torch.randn((z.shape[1], RFF_WIDTH), generator=generator, dtype=torch.float64, device=z.device) * 0.35
    phase = torch.rand((RFF_WIDTH,), generator=generator, dtype=torch.float64, device=z.device) * (2.0 * math.pi)
    factorial = p1266.factorial_at_layer(discovery, layer)
    target = (factorial["A1"] - factorial["A0"]).double()
    out_basis = basis(target, RANK_OUTPUT)
    y = target @ out_basis
    x = feature(z, omega, phase)
    gram = x.T @ x
    scale = float(torch.trace(gram).item()) / max(1, gram.shape[0])
    weights = torch.linalg.solve(gram + RIDGE * max(scale, 1.0) * torch.eye(gram.shape[0], dtype=torch.float64, device=x.device), x.T @ y)
    return {"bases": bases, "mean": mean, "std": std, "omega": omega, "phase": phase, "out_basis": out_basis, "weights": weights}


def predict(model: dict, capture: dict, layer: int, level: str) -> torch.Tensor:
    components = raw_observation(capture, layer, level)
    z = torch.cat([value.double() @ component_basis for value, component_basis in zip(components, model["bases"])], dim=1)
    z = (z - model["mean"]) / model["std"]
    interaction = feature(z, model["omega"], model["phase"]) @ model["weights"] @ model["out_basis"].T
    return p1266.factorial_at_layer(capture, layer)["A0"].double() + interaction


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    config = ModelConfig(layers=3, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22)
    seed = 1_267_301_001
    p1266.set_seed(seed)
    model, training = p1266.task_module.train_model(config, seed, device)
    rows = p1266.make_material()
    discovery = p1266.capture_partition(model, p1266.partition_rows(rows, "discovery"), device)
    oracle = p1266.capture_partition(model, p1266.partition_rows(rows, "oracle"), device)
    print({"training": training, "oracle_accuracy": oracle["accuracies"]})
    for layer in range(config.layers):
        truth = p1266.factorial_at_layer(oracle, layer)["A1"]
        risks = {}
        for level in ("delta", "state", "trajectory"):
            camera = fit(discovery, layer, level)
            pred = predict(camera, oracle, layer, level)
            risks[level] = float(p1266.bounded_loss_vector(pred, truth).mean().item())
        print({"layer": layer, "risks": risks})


if __name__ == "__main__":
    main()
