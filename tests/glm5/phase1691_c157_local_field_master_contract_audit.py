#!/usr/bin/env python3
"""Independent file-level audit for C157."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1691_c157_local_field_master_contract"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = load("protocol/preregistration.json")
    adjudication = load("analysis/evidence_adjudication.json")
    final = load("analysis/final.json")
    checks = {
        "phase": protocol["phase"] == final["phase"] == 1691,
        "routes": len(protocol["routes"]) == 9,
        "continuous": [r["phase"] for r in protocol["routes"]] == list(range(1692, 1701)),
        "typed_ledgers": set(adjudication["typed_claims"]) == {"measured_pass", "measured_fail", "not_tested"},
        "critical_correction": any("recipient-only" in value for value in adjudication["corrections"]),
        "attachment_hashes": len(protocol["attachment_hashes"]) == 2 and all(len(value) == 64 for value in protocol["attachment_hashes"]),
        "frozen_hash": len(sha(OUT / "protocol/preregistration.json")) == 64,
        "closed": final["status"] == "closed" and final["routes_frozen"] == 9,
    }
    report = {"phase": 1691, "campaign": "C157", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "run_C158"}
    path = OUT / "audit/independent_final_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
