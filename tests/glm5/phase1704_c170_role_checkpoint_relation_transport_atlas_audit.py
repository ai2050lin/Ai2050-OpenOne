#!/usr/bin/env python3
"""Independent audit for C170."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1704_c170_role_checkpoint_relation_transport_atlas"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    protocol = load(OUT / "protocol/preregistration.json")
    report = load(OUT / "analysis/atlas.json")
    final = load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/role_checkpoint_response.float16.npy", mmap_mode="r")
    field = np.load(OUT / "analysis/fresh_relation_components.float16.npy", mmap_mode="r")
    checks = {
        "contract": load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "run": load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
        "analysis": load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "final": final["all_checks_passed"],
        "raw": list(raw.shape) == [9, 24, 16, 6, 2560] and np.isfinite(raw).all(),
        "field": list(field.shape) == [9, 4, 16, 6, 2560] and np.isfinite(field).all(),
        "settings": len(report["settings"]) == 9 and len({(r["source_checkpoint"], r["source_role"]) for r in report["settings"]}) == 9,
        "labels": sum(report["label_counts"].values()) == 9,
        "all_executed": report["campaign_complete"] and protocol["campaign_policy"].startswith("evaluate all nine"),
        "coordinates": len(protocol["source_coordinates"]) == 16 and len(set(protocol["source_coordinates"])) == 16,
        "boundary": "not guaranteed optimal elsewhere" in report["claim_boundary"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "weights", "PCA")),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    audit = {"phase": 1704, "campaign": "C170", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "memo_and_stage_synthesis"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
