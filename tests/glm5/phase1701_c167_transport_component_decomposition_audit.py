#!/usr/bin/env python3
"""Independent artifact audit for C167."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1701_c167_transport_component_decomposition"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    protocol = load(OUT / "protocol/preregistration.json")
    report = load(OUT / "analysis/decomposition.json")
    final = load(OUT / "analysis/final.json")
    tensor = np.load(OUT / "analysis/top_relation_component_fields.float16.npy", mmap_mode="r")
    coordinates = load(OUT / "analysis/top_relation_source_coordinates.json")
    checks = {
        "contract": load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "final": final["all_checks_passed"],
        "retrospective": protocol["epistemic_status"].startswith("retrospective") and final["epistemic_status"] == "retrospective observation",
        "components": set(report["component_replication"]) == {"shared", "panel", "relation", "interaction"},
        "energy": all(abs(sum(row.values()) - 1.0) < 2e-4 for row in report["energy_fractions"].values()),
        "tensor": bool(list(tensor.shape) == [2, 4, 16, 6, 2560] and np.isfinite(tensor).all()),
        "coordinates": len(coordinates["coordinates"]) == 64 and len(set(coordinates["coordinates"])) == 64,
        "claim_boundary": "not prospective confirmation" in report["claim_boundary"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "weights", "PCA")),
    }
    audit = {
        "phase": 1701,
        "campaign": "C167",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "memo_then_new_data_campaign",
    }
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
