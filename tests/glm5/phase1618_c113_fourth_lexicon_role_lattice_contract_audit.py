#!/usr/bin/env python3
"""Independent audit for Phase1618 / C113."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_pre_model_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1618_c113_fourth_lexicon_role_lattice_contract.py"),
        "material_digest": protocol["material_digest"] == core.digest([*units, *cases]),
        "counts": len(units) == 24 and len(cases) == 384 and len(compiled) == 384 and len(manifest) == protocol["occurrences"],
        "factorial": all(value == 6 for value in Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases).values()),
        "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "predictions": set(protocol["frozen_predictions"]) == {"K292_attribute_coordinate_assignment", "K294_agent_path_increment", "agent_all_role_increment", "leave_query_anchor_candidate", "leave_query_focus_candidate"},
        "coalitions": len(protocol["role_coalitions"]) == 8 and all(f"path_without_{role}" in protocol["role_coalitions"] for role in ("focus_record", "focus_post", "query_focus", "query_anchor")),
        "observation_first": protocol["observation_first"].startswith("capture and adjudicate"),
        "boundary": "not weights" in protocol["claim_boundary"] and "natural-route" in protocol["claim_boundary"],
        "authorization": protocol["authorization"] == "execute_phase1619_c113_exact_field_capture",
    }
    report = {"phase": 1618, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "protocol_sha256": core.sha(OUT / "protocol/preregistration.json"), "authorization": protocol["authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/independent_pre_model_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
