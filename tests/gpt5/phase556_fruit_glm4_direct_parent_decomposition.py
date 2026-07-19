#!/usr/bin/env python3
"""Run the frozen Phase556 direct-parent decomposition on GLM4."""

from __future__ import annotations

import argparse

import phase556_fruit_direct_parent_decomposition as base


base.MODEL = "glm4"
base.BEHAVIOR_PATH = base.OUT_DIR / "phase556_glm4_behavior_rows.jsonl"
base.BOUNDARY_PATH = base.OUT_DIR / "phase556_glm4_layer_input_boundary_analysis.json"
base.OUTPUT = (
    base.OUT_DIR / "direct_parent_decomposition" / "glm4"
    / "phase556_direct_parent_rows.jsonl"
)
base.SUMMARY = (
    base.OUT_DIR / "direct_parent_decomposition" / "glm4"
    / "phase556_direct_parent_execution_summary.json"
)
base.ANCHOR_SLICE = (36, 44)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    base.run(args.restart, args.batch_size)
