#!/usr/bin/env python3
"""Development-only probe for a hidden-observation -> patch-response camera."""

from __future__ import annotations

import contextlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rows(seed: int, count: int, device: torch.device):
    rng = np.random.default_rng(seed)
    recv, donor, targets0, targets1, positions = [], [], [], [], []
    for _ in range(count):
        values = rng.choice(8, 2, replace=False)
        query_slot = int(rng.integers(0, 2))
        base = [4 + int(values[0]), 4 + int(values[1]), 2 + query_slot]
        swapped = [4 + int(values[1]), 4 + int(values[0]), 2 + query_slot]
        recv.append(base)
        donor.append(swapped)
        targets0.append(int(values[query_slot]))
        targets1.append(int(values[1 - query_slot]))
        positions.append(query_slot)
    return (
        torch.tensor(recv, dtype=torch.long, device=device),
        torch.tensor(donor, dtype=torch.long, device=device),
        torch.tensor(targets0, dtype=torch.long, device=device),
        torch.tensor(targets1, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
    )


def training_data(device: torch.device):
    values = []
    targets = []
    for v0 in range(8):
        for v1 in range(8):
            if v0 == v1:
                continue
            for q in (0, 1):
                values.append([4 + v0, 4 + v1, 2 + q])
                targets.append(v0 if q == 0 else v1)
    return torch.tensor(values, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def train(seed: int, config: ModelConfig, device: torch.device):
    seed_all(seed)
    model = TinyCausalTransformer(config).to(device)
    x, y = training_data(device)
    candidate = torch.arange(4, 12, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    for step in range(1000):
        idx = torch.randint(0, x.shape[0], (256,), generator=generator).to(device)
        logits = model(x[idx])[:, -1].index_select(-1, candidate)
        loss = F.cross_entropy(logits.float(), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 49:
            with torch.inference_mode():
                pred = torch.argmax(model(x)[:, -1].index_select(-1, candidate), dim=-1)
                acc = float((pred == y).float().mean())
            if acc >= 0.9999:
                return model.eval(), step + 1, acc
    return model.eval(), 1000, acc


def modules(model):
    out = {}
    for depth, block in enumerate(model.blocks):
        out[f"residual_d{depth+1}"] = block
        out[f"attention_d{depth+1}"] = block.attn
        out[f"mlp_d{depth+1}"] = block.mlp
    return out


@torch.no_grad()
def candidate_logits(model, ids):
    return model(ids)[:, -1, 4:12].float()


@torch.no_grad()
def capture(model, module, ids, position):
    store = {}
    def hook(_m, _a, output):
        value = output[0] if isinstance(output, tuple) else output
        batch = torch.arange(value.shape[0], device=value.device)
        store["value"] = value[batch, position].detach().clone()
        return output
    handle = module.register_forward_hook(hook)
    try:
        candidate_logits(model, ids)
    finally:
        handle.remove()
    return store["value"]


@torch.no_grad()
def patched(model, module, recv, position, source, alpha):
    calls = 0
    def hook(_m, _a, output):
        nonlocal calls
        value = output[0] if isinstance(output, tuple) else output
        result = value.clone()
        batch = torch.arange(value.shape[0], device=value.device)
        base = value[batch, position]
        result[batch, position] = base + alpha * (source.to(value) - base)
        calls += 1
        return (result,) + output[1:] if isinstance(output, tuple) else result
    handle = module.register_forward_hook(hook)
    try:
        value = candidate_logits(model, recv)
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError(f"patch calls={calls}")
    return value


def ridge(x, y, lam=1e-2):
    xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    reg = np.eye(xb.shape[1]) * lam
    reg[-1, -1] = 0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def predict(x, w):
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ w


def cosine_rows(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    config = ModelConfig(layers=4, width=64, heads=4, mlp_width=128, max_length=3, vocab_size=16)
    model, steps, accuracy = train(1247001, config, device)
    recv_d, donor_d, _, _, pos_d = rows(1247111, 192, device)
    recv_c, donor_c, _, _, pos_c = rows(1247999, 192, device)
    results = []
    for name, module in modules(model).items():
        for role in ("target", "boundary"):
            pd = pos_d if role == "target" else torch.full_like(pos_d, 2)
            pc = pos_c if role == "target" else torch.full_like(pos_c, 2)
            rd = capture(model, module, recv_d, pd)
            dd = capture(model, module, donor_d, pd)
            rc = capture(model, module, recv_c, pc)
            dc = capture(model, module, donor_c, pc)
            base_d = candidate_logits(model, recv_d)
            base_c = candidate_logits(model, recv_c)
            x_parts, y_parts = [], []
            for alpha in (0.25, 0.5):
                x_parts.append((alpha * (dd - rd)).cpu().numpy())
                y_parts.append((patched(model, module, recv_d, pd, dd, alpha) - base_d).cpu().numpy())
            x = np.concatenate(x_parts)
            y = np.concatenate(y_parts)
            w = ridge(x, y)
            xc = (dc - rc).cpu().numpy()
            actual = (patched(model, module, recv_c, pc, dc, 1.0) - base_c).cpu().numpy()
            predicted = predict(xc, w)
            cos = cosine_rows(predicted, actual)
            rel = np.linalg.norm(predicted - actual, axis=1) / np.maximum(np.linalg.norm(actual, axis=1), 1e-8)
            results.append({
                "event": name,
                "role": role,
                "effect_norm": float(np.mean(np.linalg.norm(actual, axis=1))),
                "cosine_mean": float(np.mean(cos)),
                "cosine_positive": float(np.mean(cos > 0)),
                "relative_error_mean": float(np.mean(rel)),
            })
    results.sort(key=lambda z: (-z["cosine_mean"], z["relative_error_mean"]))
    print({"steps": steps, "accuracy": accuracy, "top": results[:8], "bottom": results[-4:]})


if __name__ == "__main__":
    main()
