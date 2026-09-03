#!/usr/bin/env python3
"""Independent audit for Phase1624 / C114 structural atlas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1623_c114_existing_data_structural_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "audit/internal_closure_audit.json")
    atlas = core.load(OUT / "analysis/structural_atlas.json")
    closure = core.load(OUT / "analysis/closure.json")
    payload = core.load(PUBLIC)
    canonical = OUT / "visualization/c109_c114_structural_atlas.json"
    checks = {
        "internal": report["all_checks_passed"],
        "producer": report["producer_sha256"] == core.sha(TESTS / "phase1624_c114_structural_atlas_and_prediction_freeze.py"),
        "atlas": len(atlas["cells"]) == 16 and core.sha(OUT / "analysis/structural_atlas.json") == report["atlas_sha256"],
        "counts": atlas["rollups"]["attribute_binding"]["beats_permutation_median_cells"] == 8 and atlas["rollups"]["attribute_binding"]["strictly_beats_all_permutations_cells"] == 6 and atlas["rollups"]["agent_patient"]["path_gt_query_cells"] == 8,
        "identity": core.sha(canonical) == core.sha(PUBLIC) == report["asset_sha256"] == closure["heatmap"]["sha256"],
        "payload": payload["phase"] == 1624 and "c114_structural_atlas" in payload,
        "puzzles": set(closure["new_puzzles"]) == {"K298-OBS", "K299-OBS", "K300-BOUNDARY"},
        "boundary": "existing exposed" in closure["claim_boundary"] and "descriptive" in closure["status"],
        "authorization": closure["next_authorization"].startswith("C115 fifth-lexicon prospective test"),
    }
    audit = {"phase": 1624, "campaign": "C114", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": "append_c114_memo_build_client_and_close"}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/independent_closure_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
