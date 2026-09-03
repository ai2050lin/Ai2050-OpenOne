#!/usr/bin/env python3
"""Independent recomputation audits for C118."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1640_c118_identifiable_default_override_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1640_c118_default_override_common as c118


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def support(values: np.ndarray, k: int) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def save(name: str, phase: int, checks: dict, authorization: str) -> None:
    report = {"phase": phase, "campaign": "C118", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": authorization}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report)
    print(json.dumps(report, indent=2))


def contract() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    cells = Counter((row["partition"], row["default_factor"], row["hit_factor"], row["conflict_factor"], row["surface_factor"], row["output_format"]) for row in cases)
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1640_c118_default_override_common.py"),
        "digest": protocol["material_digest"] == core.digest([*units, *cases]),
        "counts": (len(units), len(cases), len(compiled), len(manifest)) == (24, 768, 768, protocol["occurrences"]),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in c118.PARTITIONS},
        "factorial": cells == {(partition, *cell): 8 for partition in c118.PARTITIONS for cell in __import__("itertools").product((1, -1), repeat=5)},
        "truth": all(row["truth_factor"] == (row["default_factor"] if row["hit_factor"] == -1 else row["default_factor"] * row["conflict_factor"]) for row in cases),
        "unique": len({row["prompt"] for row in cases}) == 768,
        "roles": all(set(row["role_positions"]) == set(c118.ROLES) for row in compiled),
        "candidates": all(len(value) == 1 for row in compiled for value in row["candidate_ids"]),
        "object": protocol["object"] == "default inheritance and item-specific conflicting exception override",
        "boundary": all(term in protocol["claim_boundary"] for term in ("no weights", "attention/MLP", "common module", "new mathematics")),
        "authorization": protocol["authorization"] == "execute_phase1641_c118_cuda_capture",
    }
    save("independent_contract_audit", 1640, checks, protocol["authorization"])


def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    logits = np.load(OUT / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "shape": list(raw.shape) == protocol["archive"]["shape"] and raw.dtype == np.uint16,
        "hash": core.sha(OUT / protocol["archive"]["path"]) == report["raw_sha256"],
        "logits": list(logits.shape) == [768, 2] and bool(np.isfinite(logits).all()) and core.sha(OUT / "raw/qwen3_candidate_logits.float32.npy") == report["logits_sha256"],
        "index": len(index) == 768 and core.sha(OUT / "raw/qwen3_behavior_index.jsonl") == report["index_sha256"],
        "repeat": all(value == 0 for value in report["numeric"].values()),
        "behavior": report["behavior_gate_passed"] == all(report["behavior_gate_checks"].values()),
        "numeric": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "authorization": report["authorization"] == ("run_phase1642_c118_discovery" if report["behavior_gate_passed"] else "close_hidden_state_route_and_continue_campaign_missingness_ledger"),
    }
    save("independent_capture_audit", 1641, checks, report["authorization"])


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/discovery_freeze.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    fields = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    candidates = core.rows(OUT / "analysis/discovery_candidate_table.jsonl")
    e, r, state = c118.EFFECTS.index("override"), c118.ROLES.index(nomination["role"]), int(nomination["state"])
    left = np.mean(fields[:4, e, r, state], axis=0, dtype=np.float32)
    right = np.mean(fields[4:, e, r, state], axis=0, dtype=np.float32)
    score = cosine(left, right) * min(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    eligible = [row for row in candidates if row["score"] is not None]
    winner = sorted(eligible, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], c118.ROLES.index(row["role"])))[0]
    checks = {
        "capture": core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"],
        "shape": list(fields.shape) == [8, 5, 9, 37, 2560],
        "hash": core.sha(OUT / "analysis/discovery_unit_effect_role_state.float32.npy") == report["field_sha256"],
        "candidates": len(candidates) == 270 and winner["role"] == nomination["role"] and winner["state"] == nomination["state"],
        "score": abs(score - nomination["score"]) < 1e-7,
        "support": len(nomination["support"]) == 256 and len(set(nomination["support"])) == 256,
        "partition": all(int(value.rsplit("-", 1)[1]) < 8 for value in nomination["discovery_units"]),
        "authorization": report["authorization"] == "execute_phase1643_c118_validation",
    }
    save("independent_discovery_audit", 1642, checks, report["authorization"])


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/validation_adjudication.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    fields = np.load(OUT / "analysis/validation_unit_effect_role_state.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "analysis/validation_intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/validation_summary.jsonl")
    e, r, state = c118.EFFECTS.index("override"), c118.ROLES.index(nomination["role"]), int(nomination["state"])
    d = np.mean(discovery[:, e, r, state], axis=0, dtype=np.float32)
    conf = np.mean(fields[:8, e, r, state], axis=0, dtype=np.float32)
    lock = np.mean(fields[8:, e, r, state], axis=0, dtype=np.float32)
    expected = {
        "confirmation_lockbox_cosine": cosine(conf, lock), "confirmation_to_discovery_cosine": cosine(conf, d), "lockbox_to_discovery_cosine": cosine(lock, d),
        "confirmation_support_overlap": len(support(conf, 256) & set(nomination["support"])) / 256,
        "lockbox_support_overlap": len(support(lock, 256) & set(nomination["support"])) / 256,
    }
    expected_predictions = {
        "field_passed": all(report["field_checks"].values()),
        "coordinate_assignment_cells": sum(row["frozen_support_gt_permutation_median"] for row in summaries),
        "strict_assignment_cells": sum(row["frozen_support_gt_all_permutations"] for row in summaries),
        "common_positive_cells": sum(row["mode_median_gains"]["boundary_common_only"] > 0 for row in summaries),
        "residual_positive_cells": sum(row["mode_median_gains"]["boundary_residual_only"] > 0 for row in summaries),
        "full_gt_each_component_cells": sum(row["mode_median_gains"]["boundary_full"] > max(row["mode_median_gains"]["boundary_common_only"], row["mode_median_gains"]["boundary_residual_only"]) for row in summaries),
    }
    checks = {
        "discovery": core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"],
        "shape": list(fields.shape) == [16, 5, 9, 37, 2560],
        "counts": len(rows) == 128 and len(summaries) == 4 and all(row["pairs"] == 32 and row["independent_units"] == 8 for row in summaries),
        "metrics": all(abs(expected[key] - report["field_metrics"][key]) < 1e-7 for key in expected),
        "predictions": expected_predictions == report["predictions"],
        "l2": report["max_l2_relative_error"] <= 0.02,
        "finite": all(math.isfinite(row["modes"][mode]["target_direction_gain"]) for row in rows for mode in row["modes"]),
        "hashes": core.sha(OUT / "analysis/validation_unit_effect_role_state.float32.npy") == report["field_sha256"] and core.sha(OUT / "analysis/validation_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/validation_summary.jsonl") == report["summary_sha256"],
        "boundary": report["leave_c118_out_common_residual"]["definition"].startswith("leave-C118-out"),
        "authorization": report["authorization"] == "run_phase1644_c118_synthesis_visualization_and_closure",
    }
    save("independent_validation_audit", 1643, checks, report["authorization"])


def closure() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    internal = core.load(OUT / "audit/internal_closure_audit.json")
    payload = core.load(PUBLIC)
    effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C118"]
    raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C118"]
    checks = {
        "internal": internal["all_checks_passed"],
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == internal["asset_sha256"],
        "effects": len(effects) > 0 and all(len(row["values"]) == 2560 for row in effects),
        "raw": len(raw) == 120 and all(len(row["values"]) == 2560 for row in raw),
        "embedding": any(row["state"] == 0 for row in effects) and any(row["state"] == 0 for row in raw),
        "batch": payload["campaign"] == "C109-C118" and "c118_batch" in payload,
        "boundary": all(term in closure["claim_boundary"] for term in ("does not identify weights", "attention/MLP", "semantic residual", "new mathematics")),
        "authorization": closure["next_authorization"].startswith("C119"),
    }
    save("independent_closure_audit", 1644, checks, "append_C118_memo_and_verify_client")


STAGES = {"contract": contract, "capture": capture, "discover": discover, "validate": validate, "closure": closure}


if __name__ == "__main__":
    STAGES[sys.argv[1]]()
