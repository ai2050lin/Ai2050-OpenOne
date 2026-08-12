#!/usr/bin/env python3
"""Disposable pilot for training-domain-only composition formation arms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1166_cross_task_predictive_order_confirmation as p  # noqa: E402


ARMS = {
    "baseline": (False, False),
    "factor_aux": (True, False),
    "equivariance": (False, True),
    "factor_aux_equivariance": (True, True),
}


def examples(lexicon: dict) -> tuple[torch.Tensor, ...]:
    train_x, train_y, eval_x, eval_y = p.task_examples(p.COMPOSITION_TASK, lexicon)
    rows, cols, contexts = [], [], []
    pairs_a, pairs_b, pair_permutations = [], [], []
    lookup = {}
    index = 0
    for template in range(len(p.source.TEMPLATES)):
        for context in range(2):
            for row in range(4):
                for col in range(4):
                    if p.composition_holdout(row, col, context):
                        continue
                    lookup[(template, row, col, context)] = index
                    rows.append(row)
                    cols.append(col)
                    contexts.append(context)
                    index += 1
    for (template, row, col, context), source_index in lookup.items():
        transforms = (
            ((row + 1) % 4, col, context, [((value % 4) + 1) % 4 + 4 * (value // 4) for value in range(8)]),
            (row, (col + 1) % 4, context, [((value % 4) + 1) % 4 + 4 * (value // 4) for value in range(8)]),
            (row, col, 1 - context, [value ^ 4 for value in range(8)]),
        )
        for new_row, new_col, new_context, permutation in transforms:
            target_index = lookup.get((template, new_row, new_col, new_context))
            if target_index is not None:
                pairs_a.append(source_index)
                pairs_b.append(target_index)
                pair_permutations.append(permutation)
    return (
        train_x,
        train_y,
        eval_x,
        eval_y,
        torch.tensor(rows),
        torch.tensor(cols),
        torch.tensor(contexts),
        torch.tensor(pairs_a),
        torch.tensor(pairs_b),
        torch.tensor(pair_permutations),
    )


def run(arm: str, architecture: str, replicate: int) -> dict:
    use_aux, use_equivariance = ARMS[arm]
    device = torch.device("cuda")
    config = p.ARCHITECTURES[architecture]
    seed = 1167100 + list(ARMS).index(arm) * 10000 + list(p.ARCHITECTURES).index(architecture) * 1009 + replicate * 107
    p.source.set_seed(seed)
    lexicon = p.make_lexicon(seed + 18017)
    model = p.source.TinyCausalTransformer(config).to(device)
    row_head = nn.Linear(config.width, 4).to(device)
    col_head = nn.Linear(config.width, 4).to(device)
    context_head = nn.Linear(config.width, 2).to(device)
    tensors = examples(lexicon)
    train_x, train_y, eval_x, eval_y, rows, cols, contexts, pair_a, pair_b, permutations = tensors
    candidates = p.answer_ids(p.COMPOSITION_TASK, lexicon, device)
    parameters = list(model.parameters())
    if use_aux:
        parameters += list(row_head.parameters()) + list(col_head.parameters()) + list(context_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=1e-3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 31)
    trace = []
    for step in range(1, 3001):
        model.train()
        indices = torch.randint(0, len(train_x), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw, states = model(train_x[indices].to(device), return_states=True)
            logits = raw[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), train_y[indices].to(device))
            if use_aux:
                hidden = model.final_norm(states[-1])[:, -1]
                aux = (
                    F.cross_entropy(row_head(hidden).float(), rows[indices].to(device))
                    + F.cross_entropy(col_head(hidden).float(), cols[indices].to(device))
                    + F.cross_entropy(context_head(hidden).float(), contexts[indices].to(device))
                ) / 3.0
                loss = loss + 0.3 * aux
            if use_equivariance:
                pair_indices = torch.randint(0, len(pair_a), (128,), generator=generator)
                a = pair_a[pair_indices]
                b = pair_b[pair_indices]
                permutation = permutations[pair_indices].to(device)
                logits_a = model(train_x[a].to(device))[:, -1].index_select(-1, candidates).float()
                logits_b = model(train_x[b].to(device))[:, -1].index_select(-1, candidates).float()
                logits_a = logits_a - logits_a.mean(dim=1, keepdim=True)
                logits_b = logits_b - logits_b.mean(dim=1, keepdim=True)
                mapped_b = logits_b.gather(1, permutation)
                equivariance = F.mse_loss(
                    mapped_b / (mapped_b.std(dim=1, keepdim=True) + 1e-5),
                    logits_a / (logits_a.std(dim=1, keepdim=True) + 1e-5),
                )
                loss = loss + 0.3 * equivariance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step in (500, 1000, 1500, 2000, 2500, 3000):
            train = p.evaluate(model, train_x, train_y, p.COMPOSITION_TASK, lexicon)
            holdout = p.evaluate(model, eval_x, eval_y, p.COMPOSITION_TASK, lexicon)
            trace.append({"step": step, "train": train, "holdout": holdout})
    result = {
        "arm": arm,
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "training_case_count": len(train_x),
        "holdout_case_count": len(eval_x),
        "equivariance_pair_count": len(pair_a),
        "trace": trace,
    }
    del model, row_head, col_head, context_head
    torch.cuda.empty_cache()
    return result


def main() -> None:
    results = [
        run(arm, architecture, replicate)
        for arm in ARMS
        for architecture in p.ARCHITECTURES
        for replicate in (0, 1)
    ]
    payload = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
    output = ROOT / "tests/glm5_temp/phase1167_training_arm_pilot_output.json"
    output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
