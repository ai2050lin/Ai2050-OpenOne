#!/usr/bin/env python3
"""Isolated BF16 worker for Phase2090/C556 cross-model execution.

Each invocation loads exactly one model. Process isolation guarantees that GPU
and CPU mappings from the previous model are gone before the next worker starts.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2076_c542_c559_typed_operation_response_passport_campaign as campaign


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = None
    try:
        authorized = campaign.final("C555")["headline"]["causal_types"]
        rows = [
            row for row in campaign.read_rows(campaign.material_path())
            if row.get("operation_type") in authorized
        ]
        model, tokenizer, device, placement = campaign.previous.model_base().load_bf16(args.model)
        compile_base = campaign.previous.prior.previous.parent.previous.prior.compile_base
        compiled = compile_base.compile_qwen(tokenizer, rows)
        sample = compiled[:min(48, len(compiled))]
        correct = 0
        checkpoint_counts: list[int] = []
        with torch.inference_mode():
            for row in sample:
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                output = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    output_hidden_states=True, return_dict=True,
                )
                scores = [float(output.logits[0, -1, candidate[0]]) for candidate in row["candidate_ids"]]
                correct += int(int(scores[1] > scores[0]) == int(row["gold_position"]))
                checkpoint_counts.append(len(output.hidden_states))
        save(args.output, {
            "status": "closed", "model": args.model, "sample_rows": len(sample),
            "behavior_accuracy": correct / max(len(sample), 1),
            "checkpoint_count": int(np.median(checkpoint_counts)), "placement": placement,
        })
    except Exception as exc:
        save(args.output, {
            "status": "worker_exception", "model": args.model,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        campaign.previous.model_base().release_bf16(model)


if __name__ == "__main__":
    main()
