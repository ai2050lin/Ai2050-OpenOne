#!/usr/bin/env python3
"""Independent audit for the C104 exact-BF16 full-field archive."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1590_c104_qwen_full_field_capture.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/capture_adapter.json")
    report = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    contract = core.load(OUT / "protocol/preregistration.json")
    raw_path = OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy"
    index_path = OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl"
    field = np.load(raw_path, mmap_mode="r")
    index = core.rows(index_path)
    checks = {
        "producer": core.sha(producer) == adapter["producer_sha256"],
        "contract": core.sha(OUT / "protocol/preregistration.json") == adapter["contract_sha256"],
        "barcode": contract["barcode_sha256"] == adapter["barcode_sha256"],
        "source_checks": all(report["checks"].values()),
        "shape": tuple(field.shape) == (37, contract["storage"]["total_valid_tokens"], 2560),
        "dtype": field.dtype == np.uint16,
        "index": len(index) == 576 and index[0]["token_start"] == 0 and index[-1]["token_end"] == field.shape[1],
        "contiguous": all(row["token_end"] - row["token_start"] == row["prompt_length"] for row in index),
        "hashes": core.sha(raw_path) == report["raw_sha256"] and core.sha(index_path) == report["index_sha256"],
        "authorization": report["authorization"] == "run_phase1591_c104_frozen_candidate_validation",
    }
    result = {"phase": 1590, "campaign": "C104", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_full_field_capture_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
