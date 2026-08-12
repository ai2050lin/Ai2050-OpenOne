#!/usr/bin/env python3
"""One-unit runtime audit for Phase1080 causal role ordering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, release_fp16
from phase1065_multimode_response_atlas_scan import RoleCapture, event_definitions
import phase1079_output_orthogonal_pattern_scan as scan_math
import phase1080_natural_relevance_atlas_protocol as protocol


scan_math.protocol = protocol


def main() -> None:
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "cases.qwen3.jsonl"
    )
    unit_id = rows[0]["unit_id"]
    selected = {
        row["state"]: row for row in rows if row["unit_id"] == unit_id
    }
    ordered = [selected[state] for state in protocol.STATES]
    model = capture = None
    try:
        model, _, device, _ = load_fp16("qwen3")
        layers = list(get_layers(model))
        capture = RoleCapture(model, layers)
        capture.register()
        pad_id = int(model.config.pad_token_id or model.config.eos_token_id)
        input_ids, attention_mask, _, positions = scan_math.pad_rows(
            ordered, pad_id, device
        )
        capture.begin(positions)
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        capture.validate()
        state_index = {
            state: index for index, state in enumerate(protocol.STATES)
        }
        role_index = {
            role: index for index, role in enumerate(protocol.CAPTURE_ROLES)
        }
        maxima = {role: 0.0 for role in protocol.PRE_BRANCH_ROLES}
        for event in event_definitions(len(layers)):
            key = (str(event["component"]), int(event["depth"]))
            values = capture.values[key].float()
            for template in (0, 1):
                for answer in (0, 1):
                    for surface in (0, 1):
                        branch_values = [
                            values[state_index[
                                f"t{template}_b{branch}_a{answer}_l{surface}"
                            ]]
                            for branch in protocol.BRANCHES
                        ]
                        for role in protocol.PRE_BRANCH_ROLES:
                            index = role_index[role]
                            reference = branch_values[0][index]
                            for value in branch_values[1:]:
                                maxima[role] = max(
                                    maxima[role],
                                    float((value[index] - reference).abs().max()),
                                )
        result = {
            "unit_id": unit_id,
            "protocol_digest": protocol.read_json(
                protocol.OUT_ROOT / "protocol" / "preregistration.json"
            )["protocol_digest"],
            "pre_branch_max_abs": maxima,
            "passed": max(maxima.values()) == 0.0,
        }
        print(json.dumps(result))
        if not result["passed"]:
            raise RuntimeError(result)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    main()
