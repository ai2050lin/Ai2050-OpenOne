#!/usr/bin/env python3
"""Phase1598 / C106: append nested coordinate supports to the client heatmap and close the campaign."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    audit = core.load(OUT / "audit/independent_final_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    if final["authorization"] != "audit_export_and_close_c106" or not audit["all_checks_passed"]:
        raise RuntimeError("C106 heatmap authorization missing")
    canonical = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(canonical)
    family_rows = core.rows(OUT / "analysis/minimal_coordinate_coalition_by_family.jsonl")
    support_rows = []
    for row in family_rows:
        k = row["minimal_all_four_controlled_k"]
        rank = protocol["rankings"][row["family"]]
        selected = set(rank[:k])
        support_rows.append({"family": row["family"], "k": k, "values": [1 if coordinate in selected else 0 for coordinate in range(2560)]})
    payload["phase"] = 1598
    payload["campaign"] = "C104-C106"
    payload["title"] = "C104-C106 Upstream Role-State Barcode and Nested Causal Coordinate Field"
    payload["support_rows"] = support_rows
    payload["coalition_rows"] = core.rows(OUT / "analysis/nested_coordinate_intervention_summary.jsonl")
    payload["coalition_family_rows"] = family_rows
    payload["headline"]["minimal_k"] = final["minimal_k"]
    payload["source"]["c106_final_sha256"] = core.sha(OUT / "analysis/final.json")
    payload["source"]["independent_audits"].append("10/10 C106 nested intervention")
    payload["claim_boundary"] = "4/4 upstream barcodes replicated; corrected whole-role transport closes attribute binding and agent-patient. C106 first reaches all four formal causal cells at K=256 and K=128 respectively, but agent-patient is non-monotonic at K=1024. These are frozen sufficient activation-coordinate coalitions, not necessary sets, neurons, weights or a universal semantic code."
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    checks = {
        "source": audit["all_checks_passed"],
        "supports": len(support_rows) == 2 and all(len(row["values"]) == 2560 for row in support_rows),
        "counts": {row["family"]: sum(row["values"]) for row in support_rows} == final["minimal_k"],
        "curve": len(payload["coalition_rows"]) == 80 and len(payload["coalition_family_rows"]) == 2,
        "nonmonotonic": next(row for row in family_rows if row["family"] == "agent_patient")["nested_results"][6]["controlled_cells"] == 3,
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "scope": "not necessary sets" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    closure = {
        "phase": 1598, "campaign": "C106", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "minimal_coordinate_coalition_observation_intervention_heatmap_complete",
        "checks": checks, "heatmap_sha256": core.sha(canonical), "heatmap_bytes": canonical.stat().st_size,
        "minimal_k": final["minimal_k"],
        "new_puzzles": {
            "K279": "within prospectively replicated causal role states, first tested sufficient nested supports use 256/2560 coordinates for attribute binding and 128/2560 for agent-patient across confirmation/lockbox and both code strata",
            "K280": "nested coordinate sufficiency need not be monotonic: agent-patient passes at K=128,256,512, fails one cell at K=1024, then passes again, implying mixed facilitating and interfering coordinates or control sensitivity",
        },
        "claim_boundary": final["interpretation"],
        "next_authorization": "C107 observation-first coordinate interaction map: use existing C106 outputs to identify facilitating versus interfering coordinate bands, then preregister disjoint rescue and sabotage coalitions on new lexical units",
    }
    core.save(OUT / "analysis/closure.json", closure)
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    main()
