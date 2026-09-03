#!/usr/bin/env python3
"""Independent audit for Phase1619 / C113 exact capture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    field_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    field = np.load(field_path, mmap_mode="r")
    logits = np.load(logits_path, mmap_mode="r")
    index = core.rows(index_path)
    checks = {
        "producer": report["producer_sha256"] == core.sha(TESTS / "phase1619_c113_exact_field_capture.py"),
        "field": list(field.shape) == protocol["archive"]["shape"] and field.dtype == np.uint16 and core.sha(field_path) == report["raw_sha256"],
        "logits": logits.shape == (384, 2) and logits.dtype == np.float32 and bool(np.isfinite(logits).all()) and core.sha(logits_path) == report["logits_sha256"],
        "index": len(index) == 384 and core.sha(index_path) == report["index_sha256"],
        "numeric": all(value == 0.0 for value in report["numeric"].values()),
        "runtime": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "checks": all(report["checks"].values()),
        "authorization": report["authorization"] == "run_phase1620_c113_field_adjudication",
    }
    audit = {"phase": 1619, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": report["authorization"]}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/independent_capture_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
