#!/usr/bin/env python3
"""Engineering-only pilot for a sealed modular-addition trajectory protocol.

This file is not evidential output.  It is allowed to inspect the pilot test
split so that one fixed training regime can be selected before fresh formal
tasks and seeds are preregistered.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import (  # noqa: E402
    ModelConfig,
    TinyCausalTransformer,
)


class SquareMLP(nn.Module):
    """Minimal symmetric network that can discover a Fourier addition circuit."""

    def __init__(self, modulus: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(modulus, width)
        self.hidden = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, modulus, bias=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.hidden.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        operands = input_ids[:, (1, 3)]
        hidden = self.embedding(operands).sum(dim=1)
        hidden = self.hidden(hidden)
        return self.output(hidden.square())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data(modulus: int, fraction: float, seed: int) -> tuple[torch.Tensor, ...]:
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    cutoff = int(round(len(pairs) * fraction))
    train_indices = set(order[:cutoff].tolist())
    bos, plus, equals = modulus, modulus + 1, modulus + 2
    rows, labels, train_mask = [], [], []
    for index, (a, b) in enumerate(pairs):
        rows.append([bos, a, plus, b, equals])
        labels.append((a + b) % modulus)
        train_mask.append(index in train_indices)
    x = torch.tensor(rows, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(train_mask, dtype=torch.bool)
    return x[mask], y[mask], x[~mask], y[~mask]


@torch.inference_mode()
def accuracy(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    modulus: int,
) -> float:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = model(x.to(device))
        logits = raw.float() if raw.ndim == 2 else raw[:, -1, :modulus].float()
    return float((logits.argmax(dim=-1).cpu() == y).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=31)
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1169001)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model", choices=("transformer", "square_mlp"), default="transformer")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    set_seed(args.seed)
    train_x, train_y, test_x, test_y = make_data(args.modulus, args.fraction, args.seed + 17)
    if args.model == "transformer":
        config = ModelConfig(
            layers=args.layers,
            width=args.width,
            heads=4,
            mlp_width=args.width * 4,
            max_length=5,
            vocab_size=args.modulus + 3,
        )
        model: torch.nn.Module = TinyCausalTransformer(config).to(device)
        config_payload = asdict(config)
    else:
        model = SquareMLP(args.modulus, args.width).to(device)
        config_payload = {"model": "square_mlp", "width": args.width}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.perf_counter()
    trace = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw = model(train_x.to(device))
            logits = raw.float() if raw.ndim == 2 else raw[:, -1, : args.modulus].float()
            loss = F.cross_entropy(logits, train_y.to(device))
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.interval == 0:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "train_accuracy": accuracy(model, train_x, train_y, device, args.modulus),
                "test_accuracy": accuracy(model, test_x, test_y, device, args.modulus),
                "elapsed_seconds": time.perf_counter() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
    output = {
        "engineering_only": True,
        "arguments": vars(args),
        "config": config_payload,
        "train_count": len(train_x),
        "test_count": len(test_x),
        "trace": trace,
    }
    output_path = ROOT / "tests/glm5_temp/phase1169_modular_grokking_pilot.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
