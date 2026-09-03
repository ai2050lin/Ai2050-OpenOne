#!/usr/bin/env python3
"""Phase1528: freeze same-shape right-padded quartet calibration for C090."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1527_c090_singleton_numeric_calibration"
SOURCE = RESULT / "phase1526_c089_full_dimensional_diagnostics"
OUT = RESULT / "phase1528_c090_right_padded_group_calibration_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1528 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_phase1528_c090_right_padded_group_calibration" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1527 authorization missing")
    selected = core.rows(SOURCE / "protocol/singleton_calibration_cases.jsonl")
    batches = []
    for set_id in sorted({row["set_id"] for row in selected}):
        for surface in ("a_question", "b_question"):
            rows = sorted((row for row in selected if row["set_id"] == set_id and row["surface"] == surface), key=lambda row: ("aa", "ab", "ba", "bb").index(row["cell"]))
            batches.append({"batch_id": f"{set_id}__{surface}", "set_id": set_id, "surface": surface, "case_ids": [row["case_id"] for row in rows], "cells": [row["cell"] for row in rows]})
    batch_path = OUT / "protocol/right_padded_calibration_batches.jsonl"
    core.write_rows(batch_path, batches)
    protocol = {
        "phase": 1528, "campaign": "C090", "schema": "c090.right_padded_quartet_numeric_calibration.v1",
        "source_cases_sha256": core.sha(SOURCE / "protocol/singleton_calibration_cases.jsonl"),
        "batches_sha256": core.sha(batch_path), "case_count": 72, "batch_count": 18, "batch_size": 4,
        "engine": {
            "padding": "right", "batch_unit": "one set_id and one surface with aa,ab,ba,bb",
            "position_ids": "attention_mask cumulative positions", "logits": "full logits gathered at each row's true final prompt token",
            "role_pooling": "mean over registered token span without offsets", "repeat_batches": 3,
        },
        "gates": {"repeat_hidden_max_abs": 1e-6, "repeat_logit_max_abs": 1e-6, "causal_prefix_relative_l2": 1e-6},
        "failure_rule": "if any gate fails, do not authorize full recapture; preserve both singleton and right-padding failures",
        "allowed_observables": ["input embeddings", "all Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "PCA", "learned probes", "threshold mutation after execution"],
        "hidden_semantic_claim": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    checks = {
        "parent": parent_audit["all_checks_passed"], "cases": len(selected) == 72,
        "batches": len(batches) == 18 and all(len(row["case_ids"]) == 4 for row in batches),
        "cells": all(row["cells"] == ["aa", "ab", "ba", "bb"] for row in batches),
        "coverage": Counter(case_id for batch in batches for case_id in batch["case_ids"]) == Counter(row["case_id"] for row in selected),
        "no_model": True, "semantic_block": not protocol["hidden_semantic_claim"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_model_engine_audit.json", {"phase": 1528, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    core.save(OUT / "analysis/final.json", {"phase": 1528, "campaign": "C090", "status": "right_padded_group_calibration_contract_frozen", "contract_sha256": protocol["contract_sha256"], "authorization": "run_phase1529_c090_right_padded_group_calibration"})
    print(json.dumps({"protocol": protocol, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
