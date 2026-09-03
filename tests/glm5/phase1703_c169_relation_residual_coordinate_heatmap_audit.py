#!/usr/bin/env python3
"""Independent audit for C169."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1703_c169_relation_residual_coordinate_heatmap"
FRONTEND = ROOT / "frontend/public/vis_data/research_kernel/c167_c168_relation_residual_heatmap.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = load(OUT / "protocol/preregistration.json")
    asset = load(OUT / "analysis/heatmap.json")
    report = load(OUT / "analysis/synthesis.json")
    final = load(OUT / "analysis/final.json")
    rows = asset["rows"]
    checks = {
        "contract": load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "build": load(OUT / "audit/internal_build_audit.json")["all_checks_passed"],
        "final": final["all_checks_passed"],
        "frontend_equal": sha(FRONTEND) == sha(OUT / "analysis/heatmap.json") == report["asset_sha256"],
        "schema": asset["schema"] == "c167_c168_relation_residual_heatmap.v1",
        "dimensions": asset["dimensions"] == list(range(2560)),
        "rows": len(rows) == 194 and all(len(row["values"]) == 2560 for row in rows),
        "finite": all(np.isfinite(np.asarray(row["values"], np.float32)).all() for row in rows),
        "relations": set(row.get("relation") for row in rows if row["kind"] == "relation_component") == {"is_a", "part_of", "located_in", "precedes"},
        "splits": set(row.get("split") for row in rows if row["kind"] == "relation_component") == {"old_reference", "fresh"},
        "source_coordinates": len(asset["source_coordinates"]) == 64,
        "semantics": "q24 relation-role" in asset["coordinate_semantics"] and "q25 target-role" in asset["coordinate_semantics"],
        "forbidden": all(value in protocol["forbidden"] for value in ("attention", "MLP", "weights", "PCA")),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    audit = {"phase": 1703, "campaign": "C169", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "memo_and_stage_close"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
