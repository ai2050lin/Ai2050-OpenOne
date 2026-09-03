#!/usr/bin/env python3
"""Independent audit for C193 program-scaffold failure decomposition."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1727_c193_program_scaffold_failure_decomposition"; PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c193_program_centered_response_residual.json"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/failure_decomposition.json"); payload = core.load(PUBLIC); producer = Path(__file__).with_name("phase1727_c193_program_scaffold_failure_decomposition.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "six_atlases": len(report["raw_nearest_atlases"]) == 6, "support": report["program_centered_strict"]["support"] == 112, "all_2560": len(payload["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in payload["rows"]), "finite": bool(np.isfinite([report["program_centered_strict"][key] for key in ("same_family_rate", "family_baseline", "family_advantage")]).all()), "asset_hash": core.sha(PUBLIC) == final["asset"]["sha256"], "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1727, "campaign": "C193", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
