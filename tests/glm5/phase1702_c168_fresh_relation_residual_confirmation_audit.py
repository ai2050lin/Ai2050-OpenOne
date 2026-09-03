#!/usr/bin/env python3
"""Independent audit for C168."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1702_c168_fresh_relation_residual_confirmation"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = load(OUT / "protocol/preregistration.json")
    report = load(OUT / "analysis/confirmation.json")
    final = load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/fresh_q24_q25_response.float16.npy", mmap_mode="r")
    component = np.load(OUT / "analysis/fresh_relation_components.float16.npy", mmap_mode="r")
    checks = {
        "contract": load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "run": load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
        "analysis": load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "final": final["all_checks_passed"],
        "raw_shape": list(raw.shape) == [8, 64, 6, 2560] and np.isfinite(raw).all(),
        "component_shape": list(component.shape) == [4, 64, 6, 2560] and np.isfinite(component).all(),
        "fresh_material": sum(1 for _ in open(OUT / "material/fresh_anchors.jsonl", encoding="utf-8")) == 8,
        "source_lock": len(protocol["source"]["coordinates"]) == 64 and len(set(protocol["source"]["coordinates"])) == 64,
        "gates": report["passed"] == all(report["gates"].values()) == final["headline"]["passed"],
        "source_hash": protocol["source_hashes"]["C167_lock"] == sha(ROOT / "tests/glm5/result/phase1701_c167_transport_component_decomposition/analysis/top_relation_source_coordinates.json"),
        "boundary": "not whole-network" in report["claim_boundary"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "weights", "PCA")),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    audit = {"phase": 1702, "campaign": "C168", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_passed": report["passed"], "authorization": "memo_and_C169_visualization"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
