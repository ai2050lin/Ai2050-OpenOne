#!/usr/bin/env python3
"""Independent structural audit for C275."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1809_c275_joint_relational_state_observation"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c275_cross_role_reuse_atlas.json"


def main() -> None:
    final = json.loads((OUT / "analysis/final.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol/preregistration.json").read_text(encoding="utf-8"))
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    same = np.load(OUT / "analysis/same_sign_source_counts.uint32.npy", mmap_mode="r")
    checks = {
        "phase_campaign": final["phase"] == 1809 and final["campaign"] == "C275",
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "two_material_observation": protocol["materials"] == ["third", "fourth"],
        "count_shape": list(same.shape) == [2, 6, 6, 6, 2560],
        "asset_full_coordinates": asset["dimensions"] == list(range(2560)) and all(len(row["values"]) == 2560 for row in asset["rows"]),
        "no_markdown_result": not any(OUT.rglob("*.md")),
    }
    audit = {"phase": 1809, "campaign": "C275", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    (OUT / "audit/independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
