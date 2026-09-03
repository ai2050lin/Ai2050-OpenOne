#!/usr/bin/env python3
"""Independent structural audit for C273."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1807_c273_full_field_response_ecology"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c273_response_ecology_atlas.json"


def main() -> None:
    final = json.loads((OUT / "analysis/final.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol/preregistration.json").read_text(encoding="utf-8"))
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    ecology = np.load(OUT / "analysis/full_coordinate_ecology_counts.uint16.npy", mmap_mode="r")
    checks = {
        "phase_campaign": final["phase"] == 1807 and final["campaign"] == "C273",
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "contract_is_observational": bool(protocol["status"] == "full_field_failure_ecology_frozen" and protocol["interpretive_gates"]),
        "ecology_shape": list(ecology.shape) == [6, 36, 6, 8, 2560],
        "asset_full_coordinates": asset["dimensions"] == list(range(2560)) and all(len(row["values"]) == 2560 for row in asset["rows"]),
        "no_markdown_result": not any(OUT.rglob("*.md")),
    }
    audit = {"phase": 1807, "campaign": "C273", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    (OUT / "audit/independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
