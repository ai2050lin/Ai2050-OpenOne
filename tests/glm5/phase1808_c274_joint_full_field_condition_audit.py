#!/usr/bin/env python3
"""Independent structural audit for C274."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1808_c274_joint_full_field_condition"


def main() -> None:
    final = json.loads((OUT / "analysis/final.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol/preregistration.json").read_text(encoding="utf-8"))
    maps = [np.load(OUT / f"analysis/{name}_pred_sign.int8.npy", mmap_mode="r") for name in ("role_joint", "all_role_joint")]
    checks = {
        "phase_campaign": final["phase"] == 1808 and final["campaign"] == "C274",
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "prospective_split": protocol["training"].startswith("C248") and protocol["prospective_holdout"].startswith("C264"),
        "two_frozen_candidates": set(protocol["candidates"]) == {"role_joint", "all_role_joint"},
        "full_shapes": all(list(item.shape) == [6, 36, 6, 8, 2560] for item in maps),
        "no_markdown_result": not any(OUT.rglob("*.md")),
    }
    audit = {"phase": 1808, "campaign": "C274", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    (OUT / "audit/independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
