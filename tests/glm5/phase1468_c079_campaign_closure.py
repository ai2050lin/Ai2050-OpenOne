#!/usr/bin/env python3
"""Phase1468: close the complete C079 observation campaign."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1468_c079_campaign_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1468 exists")
    stages = {
        1463: RESULT / "phase1463_c079_aggregate_observation_contract",
        1464: RESULT / "phase1464_c079_behavior",
        1465: RESULT / "phase1465_c079_discovery_full_field_capture",
        1466: RESULT / "phase1466_c079_discovery_basic_observation_and_freeze",
        1467: RESULT / "phase1467_c079_holdout_capture_and_validation",
    }
    audits = {phase: core.load(path / "audit/independent_final_audit.json") for phase, path in stages.items()}
    finals = {phase: core.load(path / "analysis/final.json") for phase, path in stages.items()}
    behavior = core.load(stages[1464] / "analysis/behavior_summary.json")
    discovery_meta = core.load(stages[1465] / "analysis/capture_metadata.json")
    manifest = core.load(stages[1466] / "frozen/candidate_manifest.json")
    holdout = core.load(stages[1467] / "analysis/holdout_summary.json")
    validation = core.rows(stages[1467] / "analysis/candidate_holdout_validation.jsonl")
    scripts = []
    for phase in range(1463, 1468):
        for path in sorted(TESTS.glob(f"phase{phase}_c079_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    robust = holdout["robust_candidates"]
    candidate_by_id = {row["candidate_id"]: row for row in manifest["candidates"]}
    checks = {
        "audits": all(value["all_checks_passed"] for value in audits.values()),
        "authorizations": finals[1463]["authorization"] == "run_phase1464_c079_behavior" and finals[1464]["authorization"] == "run_phase1465_c079_discovery_full_field_capture" and finals[1465]["authorization"] == "run_phase1466_c079_discovery_basic_observation_and_freeze" and finals[1466]["authorization"] == "run_phase1467_c079_holdout_capture_and_validation" and finals[1467]["authorization"] == "run_phase1468_c079_campaign_closure",
        "behavior": behavior["behavior_qualified"] and behavior["eligible_count"] == 207,
        "discovery_hash": core.sha(stages[1465] / "raw/discovery_role_field.float16.npy") == discovery_meta["raw_sha256"],
        "holdout_hash": core.sha(stages[1467] / "raw/holdout_role_field.float16.npy") == holdout["raw_sha256"],
        "freeze": manifest["freeze_sha256"] == holdout["discovery_freeze_sha256"],
        "validation": len(validation) == 36 and all(row["split_passed"] for row in validation),
        "robust": len(robust) == 18 and Counter(candidate_by_id[value]["role"] for value in robust) == {"boundary": 6, "query_label": 6, "query_relation": 6},
        "boundary": all(candidate_by_id[value]["state"] in (32, 33) for value in robust if candidate_by_id[value]["role"] == "boundary"),
        "scripts_compile": len(scripts) == 10,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1468,
        "campaign": "C079",
        "status": "closed_with_cross_split_explicit_label_trajectory_regularities",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "retained": [
            "aggregate behavior qualification on third fresh material set",
            "complete raw discovery and two-split holdout fields",
            "causal-mask zero effect at upstream record roles for a query-only contrast",
            "stable early lexical carrier effects at query label and query relation roles",
            "six cross-split late-boundary full-vector regularities at states 32-33",
        ],
        "not_established": [
            "equality interaction separated from token main effects",
            "unlabeled natural relation semantics",
            "causal necessity or sufficiency of the observed vectors",
            "neurons, attention heads, MLP circuits, or parameter mechanisms",
            "cross-model invariance, algebraic closure, or new mathematics",
        ],
        "next_object": {
            "name": "balanced label-equality interaction field",
            "formula": "I_AB = 0.5 * (H_AA + H_BB - H_AB - H_BA)",
            "reason": "cancels additive main effects of the individual record and query labels while retaining their equality/pairing interaction",
            "sequence": ["known-truth raw observation", "discovery interaction description", "independent holdout prediction", "explicit-label withdrawal to natural verbs", "only then weak causal tests"],
        },
        "authorization": "preregister_c080_balanced_equality_interaction_and_label_withdrawal_campaign",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
