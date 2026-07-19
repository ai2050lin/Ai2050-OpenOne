#!/usr/bin/env python3
"""Analyze the Phase556 GLM4 direct-parent response cards."""

from __future__ import annotations

import phase556_fruit_direct_parent_analysis as base


base.MODEL = "glm4"
base.ROWS = (
    base.OUT_DIR / "direct_parent_decomposition" / "glm4"
    / "phase556_direct_parent_rows.jsonl"
)
base.OUTPUT = base.OUT_DIR / "phase556_glm4_direct_parent_analysis.json"


if __name__ == "__main__":
    base.analyze()
