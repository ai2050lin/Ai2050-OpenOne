#!/usr/bin/env python3
"""Independent filesystem audit for C434-C445 / Phase1968-1979."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PRODUCER = TESTS / "phase1968_c434_c445_guarded_response_graph_campaign.py"

PHASES = {
    f"C{campaign}": (1968 + campaign - 434, slug)
    for campaign, slug in (
        (434, "evidence_adjudication_and_guarded_graph_contract"),
        (435, "fresh_language_graph_material_and_zero_models"),
        (436, "qwen_multifamily_behavior_qualification"),
        (437, "qualified_full_coordinate_and_token_field"),
        (438, "guarded_signed_event_hypergraph_discovery"),
        (439, "unseen_lexicon_construction_event_prediction"),
        (440, "typed_state_distance_tournament"),
        (441, "repaired_binary_graph_behavior_interface"),
        (442, "qualified_graph_field_and_depth_prediction"),
        (443, "expanded_known_truth_writer_calibration"),
        (444, "registered_cross_model_functional_topology"),
        (445, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    digest = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    checks = {}
    for name, (phase, slug) in PHASES.items():
        out = RESULT / f"phase{phase}_{name.lower()}_{slug}"
        prereg = load(out / "protocol/preregistration.json")
        final = load(out / "analysis/final.json")
        checks[f"{name}_closed"] = final["all_checks_passed"] and prereg["producer_sha256"] == digest
        checks[f"{name}_phase"] = prereg["phase"] == phase == final["phase"]
    c435 = RESULT / "phase1969_c435_fresh_language_graph_material_and_zero_models"
    material_rows = sum(1 for line in (c435 / "material/cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    checks["fresh_material_rows"] = material_rows == 2880
    c441 = RESULT / "phase1975_c441_repaired_binary_graph_behavior_interface"
    graph_rows = sum(1 for line in (c441 / "material/cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    checks["graph_material_rows"] = graph_rows == 2304
    visual = ROOT / "frontend/public/vis_data/research_kernel/c445_guarded_response_graph.json"
    payload = load(visual)
    checks["visual_schema"] = payload["schema"] == "c445.guarded-response-graph.v1"
    checks["visual_full_coordinates"] = bool(payload["rows"]) and all(len(row["values"]) == 2560 for row in payload["rows"])
    cleanup = load(RESULT / "phase1979_c445_campaign_synthesis_visual_cleanup_and_audit/audit/cleanup.json")
    checks["cleanup_verified"] = all(row["removed"] and len(row["sha256"]) == 64 for row in cleanup)
    checks["claim_boundary"] = all(
        "strict_interpretation" in load(RESULT / f"phase{phase}_{name.lower()}_{slug}/analysis/final.json")["headline"]
        or name in ("C438",)
        for name, (phase, slug) in PHASES.items()
    )
    report = {
        "phase": 1979, "campaign": "C434-C445", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    destination = RESULT / "phase1979_c445_campaign_synthesis_visual_cleanup_and_audit/audit/independent_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise AssertionError({key: value for key, value in checks.items() if not value})


if __name__ == "__main__":
    main()
