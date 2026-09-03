#!/usr/bin/env python3
"""Phase1578: remove the non-scientific unit_index dependency from C101 analysis."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1575_c101_dual_arm"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as base


def prepare() -> None:
    old = core.load(OUT / "protocol/analysis_adapter.json")
    raw = core.load(OUT / "analysis/qwen_capture_summary.json")
    failure = {
        "phase": 1577,
        "campaign": "C101",
        "status": "analysis_not_started_index_metadata_dependency",
        "error": "KeyError: unit_index",
        "scientific_result": "none",
        "walsh_coefficients_completed": False,
        "raw_field_modified": False,
        "authorization": "freeze_phase1578_index_adapter",
    }
    checks = {
        "old_frozen": old["producer_sha256"] == core.sha(TESTS / "phase1577_c101_dual_arm_analysis.py"),
        "raw": old["source_raw_sha256"] == raw["raw_sha256"],
        "index": old["source_index_sha256"] == raw["index_sha256"],
        "no_final": not (OUT / "analysis/final.json").exists(),
        "unit_id_available": all("unit_id" in row for row in core.rows(OUT / "raw/qwen3_registered_role_index.jsonl")),
        "unit_index_absent": all("unit_index" not in row for row in core.rows(OUT / "raw/qwen3_registered_role_index.jsonl")[:8]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    adapter = {
        "phase": 1578,
        "campaign": "C101",
        "status": "index_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "source_raw_sha256": raw["raw_sha256"],
        "source_index_sha256": raw["index_sha256"],
        "single_change": "derive unit identity from unit_id; do not require omitted unit_index metadata",
        "unchanged": {"primary": old["primary"], "null": old["null"], "breadth": old["breadth"]},
        "authorization": "execute_phase1578_analysis_adapter",
    }
    core.save(OUT / "analysis/phase1577_adapter_failure.json", failure)
    core.save(OUT / "protocol/analysis_index_adapter.json", adapter)
    core.save(OUT / "audit/pre_analysis_index_adapter_audit.json", {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "adapter": adapter}, indent=2))


def corrected_compute(field: np.ndarray, index: list[dict[str, Any]], arm: str, roles: tuple[str, ...], effects: tuple[str, ...], masks: tuple[tuple[int, ...], ...] | None):
    rows = [row for row in index if row["arm"] == arm]
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[row["unit_id"]].append(row)
    units = []
    for unit_id, unit_rows in by_unit.items():
        first = unit_rows[0]
        units.append({key: first[key] for key in ("unit_id", "arm", "family", "world", "partition", "surface")})
    units.sort(key=lambda row: row["unit_id"])
    path = OUT / f"raw/qwen3_{arm}_walsh_coefficients_v2.float32.npy"
    coeff = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), len(effects), base.STATES, len(roles), base.DIM))
    for unit_index, unit in enumerate(units):
        unit_rows = by_unit[unit["unit_id"]]
        values = np.stack([np.stack([base.role_vector(field, row, role) for role in roles], axis=1) for row in unit_rows], axis=0)
        for effect_index, effect in enumerate(effects):
            if arm == "confirmation":
                signs = np.asarray([base.graph_base.effect_sign(row, effect) for row in unit_rows], dtype=np.float32)
            else:
                assert masks is not None
                names = [base.BREADTH_FACTORS[i] for i in masks[effect_index]]
                key_for = {"truth": "truth_factor", "surface": "surface_factor", "distractor": "distractor_factor", "code": "code"}
                signs = np.asarray([np.prod([row[key_for[name]] for name in names]) for row in unit_rows], dtype=np.float32)
            coeff[unit_index, effect_index] = np.einsum("c,csrd->srd", signs, values, optimize=True) / 16.0
        if (unit_index + 1) % 12 == 0:
            print(f"[phase1578] {arm} coefficients {unit_index + 1}/{len(units)}", flush=True)
    coeff.flush()
    del coeff
    core.write_rows(OUT / f"raw/qwen3_{arm}_walsh_index_v2.jsonl", [{"row_index": i, **row} for i, row in enumerate(units)])
    return path, units


def analyze() -> None:
    adapter = core.load(OUT / "protocol/analysis_index_adapter.json")
    audit = core.load(OUT / "audit/pre_analysis_index_adapter_audit.json")
    if adapter["authorization"] != "execute_phase1578_analysis_adapter" or not audit["all_checks_passed"]:
        raise RuntimeError("adapter not authorized")
    if adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("adapter changed after freeze")
    base.compute_coefficients = corrected_compute
    base.analyze()
    final = core.load(OUT / "analysis/final.json")
    report = {
        "phase": 1578,
        "campaign": "C101",
        "status": "index_adapter_analysis_complete",
        "single_change": adapter["single_change"],
        "scientific_final_sha256": core.sha(OUT / "analysis/final.json"),
        "scientific_checks_passed": final["all_checks_passed"],
        "authorization": final["result"]["authorization"],
    }
    core.save(OUT / "analysis/phase1578_adapter_final.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "analyze"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else analyze()


if __name__ == "__main__":
    main()
