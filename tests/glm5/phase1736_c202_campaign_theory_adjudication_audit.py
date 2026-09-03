#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1736_c202_campaign_theory_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = core.load(OUT / "analysis/adjudication.json")
    asset_meta = core.load(OUT / "analysis/public_asset.json")
    asset_path = ROOT / asset_meta["path"]
    asset = core.load(asset_path)
    pairs = report["c201_topology_reaudit"]["pairs"]
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "upstream": all(report["upstream_independent_audits"].values()),
        "pair_nulls": len(pairs) == 3 and all(row["role_permutations"] == 720 and 0 < row["exact_upper_p"] <= 1 for row in pairs),
        "typed_not_tested": report["evidence_ledger"]["C200"]["natural_deletion_rescue_tested"] is False,
        "no_new_math_overclaim": report["new_mathematics_upgrade_gate"]["gate_passed"] is False,
        "asset": core.sha(asset_path) == asset_meta["sha256"] and len(asset["dimensions"]) == 2560 and len(asset["rows"]) == asset_meta["rows"],
        "asset_finite": bool(np.isfinite(np.asarray([row["values"] for row in asset["rows"]], dtype=np.float32)).all()),
        "producer_hash": core.sha(Path(__file__).with_name("phase1736_c202_campaign_theory_adjudication.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1736, "campaign": "C202", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
