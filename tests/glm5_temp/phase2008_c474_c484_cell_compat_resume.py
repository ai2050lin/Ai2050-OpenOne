#!/usr/bin/env python3
"""Resume C474-C484 with the legacy behavior runner's redundant cell field."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase2005_c471_c484_program_guard_hypergraph_campaign as campaign


_material_lookup = campaign.material_lookup


def compatible_material_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows, _ = _material_lookup()
    rows = [{**row, "cell": "".join(str(bit) for bit in row["bits"])} for row in rows]
    return rows, {row["case_id"]: row for row in rows}


campaign.material_lookup = compatible_material_lookup

for name in [f"C{value}" for value in range(474, 485)]:
    campaign.RUNNERS[name]()
