#!/usr/bin/env python3
"""Independent audit for Phase1582 C102 predictions."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"

import sys
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1582_c102_c101_field_discovery.py"
    py_compile.compile(str(producer), doraise=True)
    prediction = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    summary = core.load(OUT / "analysis/c101_field_discovery_summary.json")
    graph_path = ROOT / prediction["barcodes"]["graph_path"]
    breadth_path = ROOT / prediction["barcodes"]["breadth_path"]
    graph = np.load(graph_path, mmap_mode="r")
    breadth = np.load(breadth_path, mmap_mode="r")
    rankings = [row["coordinate_rank"] for row in prediction["selectors"]]
    checks = {
        "producer": core.sha(producer) == prediction["producer_sha256"],
        "pre_capture": not (OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy").exists(),
        "families": summary["families"] == len(prediction["selectors"]) == 8,
        "shapes": graph.shape == (4, 3, 2560) and breadth.shape == (4, 3, 2560),
        "hashes": core.sha(graph_path) == prediction["barcodes"]["graph_sha256"] and core.sha(breadth_path) == prediction["barcodes"]["breadth_sha256"],
        "finite": bool(np.isfinite(graph).all() and np.isfinite(breadth).all()),
        "rankings": all(len(rank) == 2560 and sorted(rank) == list(range(2560)) for rank in rankings),
        "selectors": all(0 <= row["selector"]["state"] < 37 and row["selector"]["minimum_partition_cosine"] <= 1.000001 for row in prediction["selectors"]),
        "authorization": prediction["authorization"] == "run_phase1583_c102_qwen_full_field_capture",
    }
    result = {"phase": 1582, "campaign": "C102", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_c101_discovery_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
