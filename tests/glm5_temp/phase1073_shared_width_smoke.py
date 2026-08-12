#!/usr/bin/env python3
"""One-unit Qwen3 audit for the Phase1073 shared-width repair."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, release_fp16
import phase1069_local_coordinate_scan as previous
import phase1073_late_query_protocol as protocol
import phase1073_late_query_scan as scan


def main() -> None:
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "cases.qwen3.jsonl"
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["operation_unit_id"]].append(row)
    values = next(iter(grouped.values()))
    by_task = defaultdict(dict)
    for row in values:
        by_task[row["task_family"]][row["state"]] = row
    states = list(protocol.STATES)
    state_positions = {state: index for index, state in enumerate(states)}
    width = max(
        len(by_task[task][state]["input_ids"])
        for task in protocol.TASK_FAMILIES
        for state in states
    )
    model = capture = None
    try:
        model, _tokenizer, device, _placement = load_fp16("qwen3")
        layers = list(get_layers(model))
        capture = previous.ResidualRoleCapture(model, layers)
        capture.register()
        task_stats = {}
        with torch.inference_mode():
            for task in protocol.TASK_FAMILIES:
                task_rows = [by_task[task][state] for state in states]
                ids, mask, _lengths, positions = scan.pad_rows(
                    task_rows, 0, device, fixed_width=width
                )
                capture.begin(positions)
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )
                capture.validate()
                task_stats[task] = {
                    depth: (
                        scan.did_vectors(
                            capture.values[depth].float(),
                            state_positions,
                        ).detach(),
                        torch.linalg.vector_norm(
                            capture.values[depth].float(), dim=-1
                        ).mean(dim=0).detach(),
                    )
                    for depth in range(len(layers) + 1)
                }
                del output, ids, mask, positions
                capture.values = {}
        role_indices = {
            role: protocol.CAPTURE_ROLES.index(role)
            for role in protocol.PRE_BRANCH_HARD_NEGATIVE_ROLES
        }
        maximum = 0.0
        embedding = 0.0
        for depth in range(len(layers) + 1):
            left, left_den = task_stats["transitive"][depth]
            right, right_den = task_stats["key_copy"][depth]
            contrast = left - right
            for contrast_vector in contrast:
                relative = scan.relative_from_denominator(
                    contrast_vector, (left_den + right_den) / 2.0
                )
                for role_index in role_indices.values():
                    maximum = max(
                        maximum,
                        float(relative[role_index].item()),
                    )
                if depth == 0:
                    embedding = max(
                        embedding, float(relative.max().item())
                    )
        print({
            "shared_width": width,
            "pre_branch_operation_contrast_max": maximum,
            "embedding_operation_contrast_max": embedding,
        })
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    main()
