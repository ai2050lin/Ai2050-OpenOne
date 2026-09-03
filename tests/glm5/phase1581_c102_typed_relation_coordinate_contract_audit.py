#!/usr/bin/env python3
"""Independent audit for the frozen C102 contract."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1581_c102_typed_relation_coordinate_contract.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    source = core.load(OUT / "audit/pre_model_material_semantic_audit.json")
    graph = core.rows(OUT / "compiled/qwen3_graph.jsonl")
    breadth = core.rows(OUT / "compiled/qwen3_breadth.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "source": source["all_checks_passed"] and source["passed"] == source["total"] == 18,
        "counts": len(graph) == len(breadth) == 576,
        "partitions": Counter(row["partition"] for row in [*graph, *breadth]) == {partition: 384 for partition in protocol["materials"]["partitions"]},
        "factorial": source["factorial_rank"] == 16,
        "semantic": all((row["path_count"] == 1) == row["truth"] for row in graph),
        "storage": protocol["storage"]["archive_dtype"] == "uint16 exact BF16 bits" and protocol["storage"]["expected_raw_bytes"] < 40 * 1024 ** 3,
        "scope": "activation coordinate is not a model parameter" in protocol["analysis_correction"]["corrected"],
        "authorization": protocol["authorization"] == "run_phase1582_c102_c101_field_discovery",
    }
    result = {"phase": 1581, "campaign": "C102", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_contract_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
