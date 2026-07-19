#!/usr/bin/env python3
"""Analyze the replicated Phase556 GLM4 layer-input boundaries."""

from __future__ import annotations

import phase556_fruit_layer_input_boundary_analysis as base


base.MODEL = "glm4"
base.BOUNDARY_DIR = base.OUT_DIR / "layer_input_boundary" / "glm4"
base.OUTPUT = base.OUT_DIR / "phase556_glm4_layer_input_boundary_analysis.json"


if __name__ == "__main__":
    base.analyze()
