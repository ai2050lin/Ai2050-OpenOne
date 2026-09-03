#!/usr/bin/env python3
"""Independent structural audit for C276."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1810_c276_prospective_cross_role_reuse_prediction"


def main() -> None:
    final = json.loads((OUT / "analysis/final.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol/preregistration.json").read_text(encoding="utf-8"))
    source_map = np.load(OUT / "analysis/frozen_source_map.int8.npy", mmap_mode="r")
    counts = np.load(OUT / "analysis/coordinate_correct_union_counts.uint16.npy", mmap_mode="r")
    checks = {
        "phase_campaign": final["phase"] == 1810 and final["campaign"] == "C276",
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "prospective_split": protocol["training"].startswith("C248") and protocol["prospective_test"].startswith("C264"),
        "no_patching": "patch" not in protocol["prediction"].lower(),
        "source_map_shape": list(source_map.shape) == [6, 36, 6],
        "full_coordinate_counts": list(counts.shape) == [6, 36, 6, 4, 2, 2560],
        "no_markdown_result": not any(OUT.rglob("*.md")),
    }
    audit = {"phase": 1810, "campaign": "C276", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    (OUT / "audit/independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
