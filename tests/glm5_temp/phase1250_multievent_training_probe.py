#!/usr/bin/env python3
"""Development-only behavior probe for the Phase1250 direct/code task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


BOS, DIRECT, CODE, REC, SEP, MAP, QUERY, ANSWER = range(8)
ENTITY_START = 8
LABEL_START = 10
CODE_START = 14
SHIFT_START = 18
VOCAB = 22
LENGTH = 23


def batch(count: int, seed: int, forced_representation: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    rows = np.empty((count, LENGTH), dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    reps = np.empty(count, dtype=np.int64)
    for index in range(count):
        representation = int(rng.integers(0, 2)) if forced_representation is None else forced_representation
        shift = int(rng.integers(0, 4))
        mapping = (np.arange(4) + shift) % 4
        codes = rng.choice(4, size=2, replace=False)
        query = 0
        order = rng.permutation(4)
        source = mapping[codes] + LABEL_START if representation == 0 else codes + CODE_START
        sequence = [BOS, DIRECT if representation == 0 else CODE, REC, ENTITY_START, int(source[0]), SEP,
                    REC, ENTITY_START + 1, int(source[1]), SEP, MAP, SHIFT_START + shift]
        for code in order:
            sequence.extend([CODE_START + int(code), LABEL_START + int(mapping[code])])
        sequence.extend([QUERY, ENTITY_START + query, ANSWER])
        rows[index] = sequence
        targets[index] = int(mapping[codes[query]])
        reps[index] = representation
    return torch.tensor(rows), torch.tensor(targets), torch.tensor(reps)


def accuracy(model: TinyCausalTransformer, inputs: torch.Tensor, labels: torch.Tensor, reps: torch.Tensor) -> tuple[float, float, float]:
    with torch.inference_mode():
        pred = model(inputs.cuda())[:, -1, LABEL_START:LABEL_START + 4].argmax(-1).cpu()
    return float((pred == labels).float().mean()), float((pred[reps == 0] == labels[reps == 0]).float().mean()), float((pred[reps == 1] == labels[reps == 1]).float().mean())


def main() -> None:
    torch.manual_seed(1250001)
    model = TinyCausalTransformer(ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=LENGTH, vocab_size=VOCAB)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    test_x, test_y, test_r = batch(8192, 1250999)
    for step in range(6500):
        if step == 3000:
            for group in optimizer.param_groups:
                group["lr"] = 2.0e-4
        if step < 3000:
            x, y, _ = batch(512, 1251000 + step, 1)
        else:
            code_x, code_y, _ = batch(384, 1251000 + step, 1)
            direct_x, direct_y, _ = batch(128, 2251000 + step, 0)
            x, y = torch.cat([code_x, direct_x]), torch.cat([code_y, direct_y])
        logits = model(x.cuda())[:, -1, LABEL_START:LABEL_START + 4]
        loss = F.cross_entropy(logits.float(), y.cuda())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 99:
            score = accuracy(model, test_x, test_y, test_r)
            print(step + 1, float(loss), score, flush=True)
            if min(score) >= 0.995:
                break


if __name__ == "__main__":
    main()
