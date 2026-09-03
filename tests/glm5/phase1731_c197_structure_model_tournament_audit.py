#!/usr/bin/env python3
"""Independent audit for C197 structure model tournament."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1731_c197_structure_model_tournament"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/tournament.json"); producer = Path(__file__).with_name("phase1731_c197_structure_model_tournament.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "seven_models": len(report["ranking"]) == 7 and set(report["ranking"]) == set(protocol["models"]), "winner": report["winner"] == report["ranking"][0], "finite": bool(np.isfinite([[report["confirmation"][m]["nrmse"], report["joint_stimulus"][m]["nrmse"]] for m in report["ranking"]]).all()), "operator_files": all((OUT / f"analysis/operators/{m}.float32.npy").exists() for m in report["ranking"]), "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1731, "campaign": "C197", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
