#!/usr/bin/env python3
"""Phase1478: close C082 and authorize fresh-material validation."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1478_c082_campaign_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1478 exists")
    stages = {phase: RESULT / f"phase{phase}_c082_{name}" for phase, name in (
        (1475, "coordinate_atlas_contract"),
        (1476, "coordinate_atlas"),
        (1477, "atlas_synthesis"),
    )}
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in stages.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in stages.items()}
    metadata = core.load(stages[1476] / "analysis/atlas_metadata.json")
    manifest = core.load(stages[1477] / "frozen/future_prediction_manifest.json")
    scripts = []
    for phase in range(1475, 1478):
        for path in sorted(TESTS.glob(f"phase{phase}_c082_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    checks = {
        "audits": all(value["all_checks_passed"] for value in audits.values()),
        "chain": finals[1475]["authorization"] == "run_phase1476_c082_coordinate_atlas" and finals[1476]["authorization"] == "run_phase1477_c082_atlas_audit_and_synthesis" and finals[1477]["authorization"] == "run_phase1478_c082_campaign_closure",
        "atlas_hash": metadata["files"]["mean_effect.float32.npy"]["sha256"] == core.sha(stages[1476] / "atlas/mean_effect.float32.npy"),
        "sign_hash": metadata["files"]["sign_consistency.float16.npy"]["sha256"] == core.sha(stages[1476] / "atlas/sign_consistency.float16.npy"),
        "freeze": manifest["freeze_sha256"] == finals[1477]["freeze_sha256"] == core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}),
        "predictions": len(manifest["future_fresh_material_predictions"]) == 5 and manifest["not_confirmed_here"],
        "scripts_compile": len(scripts) == 6,
        "no_model": metadata["model_run"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1478,
        "campaign": "C082",
        "status": "closed_with_retrospective_lexical_to_common_boundary_convergence_candidate",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "retained_candidate": "relation-specific query-token difference directions converge late to a distributed shared boundary decision response with a small recurrent coordinate scaffold",
        "evidence_level": "exploratory retrospective candidate only",
        "not_established": ["fresh-material replication", "natural unlabeled semantics", "causal necessity or sufficiency", "cross-model invariance", "attention/MLP/parameter mechanism", "new mathematics"],
        "next": "C083 tests the five frozen P082 predictions on completely fresh material with the historically qualified C079 task interface",
        "authorization": "preregister_c083_fresh_material_validation_of_lexical_to_common_boundary_convergence",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
