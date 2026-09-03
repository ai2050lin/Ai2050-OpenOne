#!/usr/bin/env python3
"""Phase1459: recomputable adjudication of the C074-C077 analysis."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1459_c074_c077_analysis_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1459 exists")
    closure_paths = {
        "c074": RESULT / "phase1449_c074_campaign_closure",
        "c075": RESULT / "phase1452_c075_behavior_gate_closure",
        "c076": RESULT / "phase1455_c076_behavior_gate_closure",
        "c077": RESULT / "phase1458_c077_behavior_gate_closure",
    }
    closures = {key: core.load(path / "analysis/final.json") for key, path in closure_paths.items()}
    audits = {key: core.load(path / "audit/independent_final_audit.json") for key, path in closure_paths.items()}
    active = core.rows(RESULT / "phase1445_c074_directional_domain_contract/material/active_cases.jsonl")
    composition = core.rows(RESULT / "phase1445_c074_directional_domain_contract/material/composition_sets.jsonl")
    behavior = core.rows(RESULT / "phase1446_c074_behavior/raw/active_behavior.jsonl")
    references = {value for row in composition for value in row.values() if isinstance(value, str) and value.startswith("c074-a-")}
    errors = {row["case_id"] for row in behavior if not row["correct"]}
    scripts = []
    for phase in range(1445, 1459):
        for path in sorted(TESTS.glob(f"phase{phase}_*.py")):
            py_compile.compile(str(path), doraise=True)
            scripts.append(path.name)
    checks = {
        "closure_audits": all(value["all_checks_passed"] for value in audits.values()),
        "statuses": closures["c074"]["status"] == "closed_with_sparse_directional_transport_domain" and all("behavior_gate" in closures[key]["status"] for key in ("c075", "c076", "c077")),
        "c074_cells": Counter(row["cell"] for row in active) == {cell: 720 for cell in ("aa", "ab", "ac", "ad", "ba", "bb", "bc", "bd")},
        "c074_truth": Counter(row["truth"] for row in active) == {True: 1440, False: 4320},
        "composition": len(composition) == 72 and len(references) == 720,
        "error_mapping": len(errors) == 3 and not (errors & references),
        "hidden_scope": all(not (RESULT / path).exists() for path in ("phase1451_c075_full_field_capture", "phase1454_c076_discovery_full_field_capture", "phase1458_c077_discovery_full_field_capture")),
        "scripts_compile": len(scripts) >= 28,
        "authorization": closures["c077"]["authorization"] == "preregister_c078_colon_label_observation_campaign",
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    result = {
        "phase": 1459,
        "campaign": "C074-C077",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "accepted": [
            "C074 found ten contract-robust direct directed whole-state transport edges, not a global transport law",
            "C074 transport was strongly surface-order and answer-direction dependent",
            "C075-C077 failed behavior gates and did not access relation Hidden States",
            "behavior stability and internal information are distinct questions",
            "C077 surface B failed mainly on equal-label truth cases while surface A was nearly perfect",
        ],
        "corrections": [
            "C074 eight material cells are named distractor constructions, not eight Boolean semantic truth combinations",
            "the robust domain is a contract-qualified output transport domain, not a natural mathematical domain",
            "direct robust edges do not establish closure, transitivity, inverse laws, or composition",
            "C074 errors were absent from the preregistered composition references; this is an explicit mapping fact, not post-hoc deletion",
            "prompt changes alter the full trajectory; evidence does not isolate a separate noisy logits manifold",
        ],
        "rejected": [
            "C075-C077 prove or suggest that relations were perfectly encoded internally",
            "C074 establishes a universal single-track law",
            "behavior gates should be abandoned",
            "TDA, persistent homology, PCA, or advanced manifold claims are currently authorized",
            "new mathematics is required by these phases",
        ],
        "c078_constraints": {
            "fresh_material": True,
            "surfaces": "two colon-delimited label surfaces frozen before behavior",
            "behavior_first": True,
            "eligible_set_rule": "only preregistered complete behavior-correct factorial sets may enter Hidden-State observation",
            "observables": ["embeddings", "all Hidden States", "yes/no logits"],
            "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes"],
            "goal": "observe full-dimensional labeled-carrier trajectories before freezing simple regularities",
        },
        "authorization": "run_phase1460_c078_contract",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
