#!/usr/bin/env python3
"""Disjoint engineering pilot for the Phase 1159 free-Transformer protocol."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


ROWS = 4
COLS = 4
CONTEXTS = 2
VOCAB = 48
TEMPLATES = tuple(itertools.permutations(("row", "col", "context")))


def make_lexicon(seed: int) -> dict[str, list[int] | int]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(2, 44)).tolist()
    return {
        "bos": 0,
        "query": 1,
        "row": ids[:4],
        "col": ids[4:8],
        "context": ids[8:10],
        "answer": ids[10:42],
    }


def target(row: int, col: int, context: int) -> int:
    return context * 16 + row * 4 + col


def dataset(lexicon: dict[str, list[int] | int]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[list[int]] = []
    targets: list[int] = []
    for template in TEMPLATES:
        for context in range(CONTEXTS):
            for row in range(ROWS):
                for col in range(COLS):
                    values = {"row": row, "col": col, "context": context}
                    inputs.append(
                        [
                            int(lexicon["bos"]),
                            *[int(lexicon[role][values[role]]) for role in template],
                            int(lexicon["query"]),
                        ]
                    )
                    targets.append(target(row, col, context))
    return torch.tensor(inputs), torch.tensor(targets)


def train(config: ModelConfig, seed: int) -> dict[str, float | int]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    lexicon = make_lexicon(seed + 11)
    inputs, targets = dataset(lexicon)
    model = TinyCausalTransformer(config).to(device)
    answer_ids = torch.tensor(lexicon["answer"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 29)
    final_step = 0
    for step in range(1, 1201):
        index = torch.randint(0, len(inputs), (128,), generator=generator)
        x = inputs[index].to(device)
        y = targets[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[:, -1].index_select(-1, answer_ids)
            loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_step = step
        if step % 50 == 0:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inputs.to(device))[:, -1].float().index_select(-1, answer_ids)
                probabilities = torch.softmax(logits, dim=-1)
                accuracy = float((logits.argmax(-1).cpu() == targets).float().mean())
                minimum = float(probabilities.gather(1, targets.to(device)[:, None]).min())
            if accuracy == 1.0 and minimum >= 0.98:
                break
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs.to(device))[:, -1].float().index_select(-1, answer_ids)
        probabilities = torch.softmax(logits, dim=-1)
    return {
        "layers": config.layers,
        "width": config.width,
        "steps": final_step,
        "accuracy": float((logits.argmax(-1).cpu() == targets).float().mean()),
        "minimum_probability": float(probabilities.gather(1, targets.to(device)[:, None]).min()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    rows = [
        train(ModelConfig(layers=4, width=64, heads=4, mlp_width=128, max_length=5, vocab_size=VOCAB), 915901),
        train(ModelConfig(layers=8, width=96, heads=4, mlp_width=192, max_length=5, vocab_size=VOCAB), 915902),
    ]
    print(json.dumps({"phase": 1159, "pilot_only": True, "models": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
