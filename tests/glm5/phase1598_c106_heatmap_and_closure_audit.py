#!/usr/bin/env python3
"""Independent C106 heatmap and closure audit."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1598_c106_heatmap_and_closure.py"
    py_compile.compile(str(producer), doraise=True)
    closure = core.load(OUT / "analysis/closure.json")
    asset = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    payload = core.load(asset)
    checks = {
        "producer": py_compile.compile(str(producer), doraise=True) is not None,
        "source": all(closure["checks"].values()),
        "minimal": closure["minimal_k"] == {"attribute_binding": 256, "agent_patient": 128},
        "supports": [(row["family"], row["k"], sum(row["values"])) for row in payload["support_rows"]] == [("attribute_binding", 256, 256), ("agent_patient", 128, 128)],
        "curve": len(payload["coalition_rows"]) == 80,
        "identity": core.sha(asset) == core.sha(PUBLIC) == closure["heatmap_sha256"],
        "puzzles": set(closure["new_puzzles"]) == {"K279", "K280"},
        "authorization": closure["next_authorization"].startswith("C107 observation-first"),
    }
    result = {"phase": 1598, "campaign": "C106", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_closure_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
