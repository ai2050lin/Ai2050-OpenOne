#!/usr/bin/env python3
"""Independent audit for C192 multi-program response equivalence."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1726_c192_multi_program_response_equivalence"; PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c192_multi_program_response_equivalence.json"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/multi_program_equivalence.json"); payload = core.load(PUBLIC); producer = Path(__file__).with_name("phase1726_c192_multi_program_response_equivalence.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "accounting": report["observed_cells"] + len(report["registered_missing"]) == 112, "all_2560": len(payload["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in payload["rows"]), "four_programs": len(report["by_source_program"]) == 4, "constrained_support": report["constrained_nearest_neighbor"]["support"] == report["observed_cells"], "finite": bool(np.isfinite([report["constrained_nearest_neighbor"][key] for key in ("same_family_rate", "available_peer_baseline", "advantage")]).all()), "asset_hash": core.sha(PUBLIC) == final["asset"]["sha256"], "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1726, "campaign": "C192", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
