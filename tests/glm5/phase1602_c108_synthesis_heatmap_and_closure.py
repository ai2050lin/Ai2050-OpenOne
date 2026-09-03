#!/usr/bin/env python3
"""Phase1602 / C108: synthesize the fresh causal result, update heatmap, and close."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1600_c108_fresh_coordinate_causality"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    audit = core.load(OUT / "audit/independent_intervention_audit.json")
    if not audit["all_checks_passed"] or audit["authorization"] != "run_phase1602_c108_synthesis_heatmap_and_closure":
        raise RuntimeError("C108 closure authorization missing")
    summaries = core.rows(OUT / "analysis/fresh_coordinate_intervention_summary.jsonl")
    family_rows = core.rows(OUT / "analysis/fresh_coordinate_family_rollup.jsonl")
    canonical = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(canonical)
    payload["phase"] = 1602
    payload["campaign"] = "C104-C108"
    payload["title"] = "C104-C108 Upstream Truth-Response Coordinate Field"
    payload["fresh_c108"] = {
        "materials": {"units": 24, "cases": 384, "partitions": ["prospective_confirmation", "independent_lockbox"]},
        "behavior": final["behavior"],
        "summary_rows": summaries,
        "family_rollup": family_rows,
    }
    payload["headline"]["fresh_c108_truth_write_delete"] = {
        row["family"]: [row["truth_direction_write_cells"], row["truth_direction_delete_cells"]] for row in family_rows
    }
    payload["headline"]["fresh_c108_task_write_delete"] = {
        row["family"]: [row["code_aligned_task_write_cells"], row["code_aligned_task_delete_cells"]] for row in family_rows
    }
    payload["claim_boundary"] = (
        "C108 prospectively confirms the frozen attribute-binding K=256 support as a controlled raw truth-direction "
        "write and delete candidate in 4/4 fresh cells, but it flips only 13.5% of false cases on average and closes "
        "only 2/4 code-aligned cells. The agent-patient K=128 support fails fresh write replication (0/4) and reaches "
        "3/4 delete cells. Neither set is minimal, necessary, functionally sufficient, a weight set, or a universal semantic code."
    )
    payload["source"]["c108_results_sha256"] = final["results_sha256"]
    payload["source"]["independent_audits"].append("12/12 C108 fresh write-delete intervention audit")
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {
        "phase": 1602, "campaign": "C108", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_coordinate_causality_stage_closed",
        "evidence": {
            "attribute_binding": "K=256: raw write 4/4, raw delete 4/4, task-aligned write/delete 2/4, mean truth flip 0.1354167",
            "agent_patient": "K=128: raw write 0/4, raw delete 3/4, task-aligned write 0/4 and delete 1/4",
        },
        "claim_adjudication": {
            "K281": "fresh bidirectional causal-response candidate: attribute-binding K=256 has controlled write and delete effects across both new partitions and both codes, but is partial rather than functionally sufficient",
            "K282": "support identity boundary: agent-patient K=128 does not prospectively reproduce the controlled write effect; a first passing K on reused data is not a stable coalition identity",
        },
        "heatmap": {"sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "path": str(PUBLIC.relative_to(ROOT))},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C109 observation-first semantic constellation: collect full embedding-to-HiddenState fields for relative graph-role changes before freezing any new causal support; preserve C108 attribute K=256 only as a positive calibration candidate",
    }
    core.save(OUT / "analysis/closure.json", closure)
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    main()
