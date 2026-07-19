#!/usr/bin/env python3
"""Run the frozen Phase556 layer-input boundary sweep on GLM4."""

from __future__ import annotations

import argparse
from pathlib import Path

import phase556_fruit_layer_input_boundary as base


base.MODEL = "glm4"
base.BEHAVIOR_PATH = base.OUT_DIR / "phase556_glm4_behavior_rows.jsonl"
base.LAYER_GRID = (
    0, 4, 8, 12, 16, 20, 22, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39,
)


def output_path(split: str) -> Path:
    return (
        base.OUT_DIR / "layer_input_boundary" / "glm4" / split
        / "phase556_boundary_rows.jsonl"
    )


def summary_path(split: str) -> Path:
    return (
        base.OUT_DIR / "layer_input_boundary" / "glm4" / split
        / "phase556_boundary_execution_summary.json"
    )


base.output_path = output_path
base.summary_path = summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=tuple(base.SPLIT_OFFSETS))
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    base.run(args.split, args.restart, args.batch_size)
