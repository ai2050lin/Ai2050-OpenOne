#!/usr/bin/env python3
"""Behavior-only Phase1082 revision-2 pilot with balanced semantic outputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5_temp"))

import phase1081_behavior_pilot as engine
import phase1082_semantic_output_operation_world_protocol as protocol


engine.protocol = protocol


def balanced_candidate_rows(rows):
    return [
        row for row in rows
        if row["template"] == 0 and row["panel"] == "active"
    ]


def balanced_generation_rows(rows):
    """Cover both semantic outcomes, both queries, and both code assignments."""
    selected = []
    states = (
        "t0_cactive_m0_q0_w0",
        "t0_cactive_m0_q1_w1",
        "t1_cactive_m1_q0_w1",
        "t1_cactive_m1_q1_w0",
    )
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            units = sorted({
                row["unit_id"] for row in rows
                if row["family"] == family and row["split"] == split
            })[:len(states)]
            for unit_id, state in zip(units, states):
                row = next(
                    row for row in rows
                    if row["unit_id"] == unit_id and row["state"] == state
                )
                selected.append({
                    **row,
                    "semantic_case_index": int(row["case_index"]),
                })
    return selected


engine.generation_rows = balanced_generation_rows
engine.selected_rows = balanced_candidate_rows


if __name__ == "__main__":
    engine.main()
