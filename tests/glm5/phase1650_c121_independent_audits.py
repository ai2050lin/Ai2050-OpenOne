#!/usr/bin/env python3
"""Independent audits for C121 contract and behavior qualification."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1650_c121_structured_comparison_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1650_c121_structured_comparison_common as c121


def save(name: str, phase: int, checks: dict, authorization: str) -> None:
    report = {"phase": phase, "campaign": "C121", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": authorization}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report)
    print(json.dumps(report, indent=2))


def contract() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    c120_values = {str(value).casefold() for row in core.rows(TESTS / "result/phase1647_c120_controlled_comparison_observation_campaign/material/units.jsonl") for value in row["values"]}
    c121_values = {str(value).casefold() for row in units for value in row["values"]}
    cells = Counter((row["partition"], row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["output_format"]) for row in cases)
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1650_c121_structured_comparison_common.py"),
        "digest": protocol["material_digest"] == core.digest([*units, *cases]),
        "counts": (len(units), len(cases), len(compiled), len(manifest)) == (24, 1152, 1152, protocol["occurrences"]),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in c121.PARTITIONS},
        "factorial": cells == {(partition, dimension, truth, gap, surface, output_format): 8 for partition in c121.PARTITIONS for dimension, truth, gap, surface, output_format in itertools.product(c121.DIMENSIONS, (1, -1), (1, -1), (1, -1), (1, -1))},
        "truth": all(row["truth_factor"] == (1 if row["scores"]["A"][row["dimension"]] > row["scores"]["B"][row["dimension"]] else -1) for row in cases),
        "fresh": not (c120_values & c121_values),
        "roles": all(set(row["role_positions"]) == set(c121.ROLES) for row in compiled),
        "candidates": all(len(candidate) == 1 for row in compiled for candidate in row["candidate_ids"]),
        "zero_models": all(value == 0.5 for key, value in protocol["zero_models"].items() if key != "integer_comparison_oracle"),
        "behavior_first": "no HiddenState archive" in protocol["behavior_first"],
        "boundary": all(term in protocol["claim_boundary"] for term in ("no HiddenState", "attention/MLP", "new mathematics")),
        "authorization": protocol["authorization"] == "execute_phase1651_c121_behavior_qualification",
    }
    save("independent_contract_audit", 1650, checks, protocol["authorization"])


def behavior() -> None:
    report = core.load(OUT / "analysis/behavior_qualification.json")
    logits = np.load(OUT / "raw/qwen3_behavior_candidate_logits.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    overall = sum(row["correct"] for row in rows) / len(rows)
    by_dimension = {name: sum(row["correct"] for row in rows if row["dimension"] == name) / 384 for name in c121.DIMENSIONS}
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "logits": list(logits.shape) == [1152, 2] and bool(np.isfinite(logits).all()) and core.sha(OUT / "raw/qwen3_behavior_candidate_logits.float32.npy") == report["logits_sha256"],
        "index": len(rows) == 1152 and core.sha(OUT / "raw/qwen3_behavior_index.jsonl") == report["index_sha256"],
        "overall": abs(overall - report["behavior"]["overall"]) < 1e-12,
        "dimensions": by_dimension == report["behavior"]["by_dimension"],
        "gate": report["gate_passed"] == all(report["gate_checks"].values()),
        "repeat": report["repeat_logits_max_abs"] == 0,
        "bf16": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "no_hidden_archive": not (OUT / "raw/qwen3_role_subtoken_all_states.uint16.npy").exists(),
        "authorization": report["authorization"] == ("freeze_phase1652_c121_all_coordinate_capture" if report["gate_passed"] else "close_C121_behavior_route"),
    }
    save("independent_behavior_audit", 1651, checks, report["authorization"])


STAGES = {"contract": contract, "behavior": behavior}
if __name__ == "__main__": STAGES[sys.argv[1]]()
