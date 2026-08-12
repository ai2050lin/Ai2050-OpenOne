#!/usr/bin/env python3
"""Disposable pilot: continue Phase1166 composition models without holdout gradients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1166_cross_task_predictive_order_confirmation as p1166  # noqa: E402


def main() -> None:
    device = torch.device("cuda")
    rows = p1166.p1163.read_jsonl(
        p1166.OUT_ROOT / "runs/models/model_metrics.jsonl"
    )
    selected = [
        row
        for row in rows
        if row["task"] == p1166.COMPOSITION_TASK
        and row["replicate"] in (0, 2)
    ]
    results = []
    for row in selected:
        checkpoint = (
            p1166.OUT_ROOT
            / "runs/models"
            / p1166.COMPOSITION_TASK
            / "checkpoints"
            / f"{row['model_id']}.pt"
        )
        model, _, lexicon = p1166.load_checkpoint(checkpoint, device)
        train_x, train_y, eval_x, eval_y = p1166.task_examples(
            p1166.COMPOSITION_TASK, lexicon
        )
        candidates = p1166.answer_ids(p1166.COMPOSITION_TASK, lexicon, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, weight_decay=1e-2
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(row["seed"]) + 1167001)
        trace = []
        for step in range(1, 20001):
            model.train()
            indices = torch.randint(
                0, len(train_x), (128,), generator=generator
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(train_x[indices].to(device))[:, -1].index_select(
                    -1, candidates
                )
                loss = F.cross_entropy(logits.float(), train_y[indices].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 500 == 0:
                train = p1166.evaluate(
                    model,
                    train_x,
                    train_y,
                    p1166.COMPOSITION_TASK,
                    lexicon,
                )
                holdout = p1166.evaluate(
                    model,
                    eval_x,
                    eval_y,
                    p1166.COMPOSITION_TASK,
                    lexicon,
                )
                trace.append(
                    {
                        "step": step,
                        "train_accuracy": train["accuracy"],
                        "holdout_accuracy": holdout["accuracy"],
                        "holdout_mean_probability": holdout["mean_probability"],
                    }
                )
        results.append(
            {
                "model_id": row["model_id"],
                "architecture": row["architecture"],
                "replicate": row["replicate"],
                "trace": trace,
            }
        )
        del model
        torch.cuda.empty_cache()
    print(json.dumps(results, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
