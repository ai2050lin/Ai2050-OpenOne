#!/usr/bin/env python3
"""Run the preregistered Phase1084 middle-band response scan."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1084_two_entity_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol
import phase1082_semantic_output_operation_world_scan as engine
from phase1065_multimode_response_atlas_scan import (
    RoleCapture as FullRoleCapture,
    event_definitions as full_event_definitions,
)


def targeted_event_definitions(n_layers: int):
    rows = [
        dict(row)
        for row in full_event_definitions(n_layers)
        if (
            protocol.TARGET_RELATIVE_DEPTH_MIN
            <= float(row["relative_depth"])
            <= protocol.TARGET_RELATIVE_DEPTH_MAX
        )
    ]
    for index, row in enumerate(rows):
        row["event_index"] = index
    return rows


class MiddleBandRoleCapture(FullRoleCapture):
    """Capture only the preregistered middle-third component events."""

    def __init__(self, model, layers):
        super().__init__(model, layers)
        self.allowed = {
            (str(row["component"]), int(row["depth"]))
            for row in targeted_event_definitions(len(layers))
        }

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, 1):
            if ("residual", depth) in self.allowed:
                self.handles.append(layer.register_forward_hook(
                    self._hook("residual", depth)
                ))
            if ("attention_output", depth) in self.allowed:
                self.handles.append(layer.self_attn.register_forward_hook(
                    self._hook("attention_output", depth)
                ))
            if ("mlp_output", depth) in self.allowed:
                self.handles.append(layer.mlp.register_forward_hook(
                    self._hook("mlp_output", depth)
                ))

    def validate(self) -> None:
        missing = self.allowed - set(self.values)
        repeated = {
            str(key): count
            for key, count in self.counts.items()
            if count != 1
        }
        unexpected = set(self.values) - self.allowed
        if missing or repeated or unexpected:
            raise RuntimeError(
                f"targeted capture drift missing={list(missing)[:5]} "
                f"unexpected={list(unexpected)[:5]} repeated={repeated}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    engine.protocol = protocol
    engine.event_definitions = targeted_event_definitions
    engine.RoleCapture = MiddleBandRoleCapture
    engine.run(args.model)


if __name__ == "__main__":
    main()
