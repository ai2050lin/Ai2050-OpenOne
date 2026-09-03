#!/usr/bin/env python3
"""Independent audit for Phase1483."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1483_existing_observation_asset_registry"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    registry = core.load(OUT / "analysis/asset_registry.json")
    py_compile.compile(str(TESTS / "phase1483_existing_observation_asset_registry.py"), doraise=True)
    unsigned = {key: value for key, value in registry.items() if key != "registry_sha256"}
    live = [(ROOT / row["path"]).exists() and (ROOT / row["path"]).stat().st_size == row["bytes"] for row in registry["assets"]]
    checks = {
        "digest": registry["registry_sha256"] == core.digest(unsigned),
        "files": all(live),
        "recorded_hashes": all(row["sha256"] == row["expected_sha256"] and row["hash_valid"] for row in registry["assets"]),
        "asset_count": len(registry["assets"]) == 6,
        "selected": {row["campaign"] for row in registry["assets"] if row["selected_for_c084"]} == {"C079", "C082"},
        "missing": {row["campaign"] for row in registry["predefined_missingness"]} == {"C080", "C081", "C083"},
        "final": final["status"] == "legal_assets_and_missingness_registered" and not final["model_run"],
    }
    result = {"phase": 1483, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
