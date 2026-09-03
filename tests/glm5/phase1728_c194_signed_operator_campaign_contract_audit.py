#!/usr/bin/env python3
"""Independent audit for C194 signed-operator campaign contract."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1728_c194_signed_operator_campaign_contract"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); producer = Path(__file__).with_name("phase1728_c194_signed_operator_campaign_contract.py")
    rows = core.rows(OUT / "material/natural_cases.jsonl"); compiled = core.rows(OUT / "compiled/qwen3_natural.jsonl")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "material": len(rows) == 320 and len(compiled) == 320,
        "ten_programs": len({r["program"] for r in rows}) == 10,
        "balanced": sum(r["gold_position"] == 0 for r in rows) == 160 and sum(r["surface"] == 0 for r in rows) == 160,
        "roles": all(set(r["role_positions"]) == {"primary", "secondary", "relation", "context", "query", "boundary"} for r in compiled),
        "audit_boundaries": len(protocol["evidence_audit"]["corrected_overclaims"]) == 4,
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {"phase": 1728, "campaign": "C194", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
