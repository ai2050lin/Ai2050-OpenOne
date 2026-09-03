#!/usr/bin/env python3
"""Independent audit for C189 synthesis heatmap."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1723_c189_campaign_synthesis_extended_heatmap"; PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c189_new_material_response_scaffold_heatmap.json"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); payload = core.load(PUBLIC); producer = Path(__file__).with_name("phase1723_c189_campaign_synthesis_extended_heatmap.py")
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "schema": payload["schema"] == "c189_new_material_response_scaffold_heatmap.v1", "58_rows": len(payload["rows"]) == 58, "all_2560": len(payload["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in payload["rows"]), "energy_and_signed": {row["kind"] for row in payload["rows"]} >= {"target_energy_profile", "signed_mean_response"}, "asset_hash": core.sha(PUBLIC) == final["asset"]["sha256"], "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1723, "campaign": "C189", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
