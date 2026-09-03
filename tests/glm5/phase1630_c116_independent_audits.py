#!/usr/bin/env python3
"""Independent audits for C116 negation-scope observation campaign."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1630_c116_negation_scope_observation_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if den <= 1e-12 else float(np.dot(left, right) / den)


def topk(values: np.ndarray, k: int) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def save(name: str, phase: int, checks: dict, authorization: str) -> None:
    report = {"phase": phase, "campaign": "C116", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": authorization}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report); print(json.dumps(report, indent=2))


def contract_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); internal = core.load(OUT / "audit/internal_pre_model_audit.json"); units = core.rows(OUT / "material/units.jsonl"); cases = core.rows(OUT / "material/cases.jsonl"); compiled = core.rows(OUT / "compiled/qwen3.jsonl"); manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    checks = {"internal": internal["all_checks_passed"], "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1630_c116_negation_scope_common.py"), "digest": protocol["material_digest"] == core.digest([*units, *cases]), "counts": (len(units), len(cases), len(compiled), len(manifest)) == (36, 576, 576, protocol["occurrences"]), "partitions": Counter(row["partition"] for row in units) == {p: 12 for p in protocol["partitions"]}, "unique": len({row["prompt"] for row in cases}) == 576, "roles": all(set(row["role_positions"]) == set(protocol["roles"]) for row in compiled), "discovery_lock": protocol["discovery_rule"]["partitions_allowed"] == ["discovery"] and protocol["discovery_rule"]["support_k"] == 256, "sources": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()), "boundary": "no natural-language universality" in protocol["claim_boundary"] and "attention/MLP" in protocol["claim_boundary"], "authorization": protocol["authorization"] == "execute_phase1631_c116_exact_field_capture"}
    save("independent_pre_model_audit", 1630, checks, protocol["authorization"])


def capture_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/capture_summary.json"); field = np.load(OUT / protocol["archive"]["path"], mmap_mode="r"); logits = np.load(OUT / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r"); index = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    checks = {"contract": core.load(OUT / "audit/independent_pre_model_audit.json")["all_checks_passed"], "shape": list(field.shape) == protocol["archive"]["shape"] and field.dtype == np.uint16, "hash": core.sha(OUT / protocol["archive"]["path"]) == report["raw_sha256"], "logits": list(logits.shape) == [576, 2] and bool(np.isfinite(logits).all()) and core.sha(OUT / "raw/qwen3_candidate_logits.float32.npy") == report["logits_sha256"], "index": len(index) == 576 and core.sha(OUT / "raw/qwen3_behavior_index.jsonl") == report["index_sha256"], "repeat": all(value == 0 for value in report["numeric"].values()), "numeric": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"], "authorization": report["authorization"] == "run_phase1632_c116_discovery_freeze"}
    save("independent_capture_audit", 1631, checks, report["authorization"])


def discovery_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/discovery_freeze.json"); nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json"); amendment = core.load(OUT / "protocol/phase1632_execution_amendment.json"); fields = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r"); candidates = core.rows(OUT / "analysis/discovery_candidate_table.jsonl")
    role_i = protocol["roles"].index(nomination["role"]); state = nomination["state"]; left = np.mean(fields[:6, role_i, state], axis=0, dtype=np.float32); right = np.mean(fields[6:, role_i, state], axis=0, dtype=np.float32); expected_score = cosine(left, right) * min(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    eligible = [row for row in candidates if row["score"] is not None]; winner = sorted(eligible, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], protocol["roles"].index(row["role"])))[0]
    checks = {"capture": core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"], "amendment": amendment["original_producer_sha256"] == protocol["producer_sha256"] and set(amendment["unchanged"]) >= {"materials", "partitions", "validation gates"}, "shape": list(fields.shape) == [12, 7, 37, 2560], "hash": core.sha(OUT / "analysis/discovery_unit_truth_role_state.float32.npy") == report["discovery_sha256"], "candidates": len(candidates) == 210 and winner["role"] == nomination["role"] and winner["state"] == nomination["state"], "score": abs(expected_score - nomination["score"]) < 1e-7, "support": len(nomination["support"]) == 256 and len(set(nomination["support"])) == 256, "partition": all(value.startswith("c116-negation-") and int(value.rsplit("-", 1)[1]) < 12 for value in nomination["discovery_units"]), "authorization": report["authorization"] == "execute_phase1633_c116_confirmation_lockbox_validation"}
    save("independent_discovery_audit", 1632, checks, report["authorization"])


def validation_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/validation_adjudication.json"); nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json"); amendment = core.load(OUT / "protocol/phase1633_execution_amendment.json"); discovery = np.load(OUT / "analysis/discovery_unit_truth_role_state.float32.npy", mmap_mode="r"); fields = np.load(OUT / "analysis/validation_unit_truth_role_state.float32.npy", mmap_mode="r"); rows = core.rows(OUT / "analysis/validation_intervention_results.jsonl"); summaries = core.rows(OUT / "analysis/validation_summary.jsonl")
    role_i, state = protocol["roles"].index(nomination["role"]), nomination["state"]; d = np.mean(discovery[:, role_i, state], axis=0, dtype=np.float32); c = np.mean(fields[:12, role_i, state], axis=0, dtype=np.float32); l = np.mean(fields[12:, role_i, state], axis=0, dtype=np.float32); metrics = report["field_metrics"]
    recomputed = {"confirmation_lockbox_cosine": cosine(c, l), "confirmation_to_discovery_cosine": cosine(c, d), "lockbox_to_discovery_cosine": cosine(l, d), "confirmation_support_overlap": len(topk(c, 256) & set(nomination["support"])) / 256, "lockbox_support_overlap": len(topk(l, 256) & set(nomination["support"])) / 256}
    expected_predictions = {"field_passed": all(report["field_checks"].values()), "correct_movement_gt_permutation_median_cells": sum(row["frozen_support_gt_permutation_median"] for row in summaries), "strict_win_cells_descriptive": sum(row["frozen_support_gt_all_permutations"] for row in summaries), "path_gt_query_cells": sum(row["path_gt_query"] for row in summaries), "selected_role_positive_cells": sum(row["selected_role_positive"] for row in summaries)}
    checks = {"discovery": core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"], "amendment": amendment["nomination_sha256"] == core.sha(OUT / "protocol/frozen_discovery_nomination.json") and "all gates" in amendment["unchanged"], "shape": list(fields.shape) == [24, 7, 37, 2560], "counts": len(rows) == 192 and len(summaries) == 4 and all(row["pairs"] == 48 and row["independent_units"] == 12 for row in summaries), "metrics": all(abs(recomputed[key] - metrics[key]) < 1e-7 for key in recomputed), "predictions": expected_predictions == report["predictions"], "l2": report["max_l2_relative_error"] <= protocol["numeric"]["movement_l2_relative_tolerance"], "finite": all(math.isfinite(row["modes"][mode]["truth_direction_gain"]) for row in rows for mode in row["modes"]), "hashes": core.sha(OUT / "analysis/validation_unit_truth_role_state.float32.npy") == report["field_sha256"] and core.sha(OUT / "analysis/validation_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/validation_summary.jsonl") == report["summary_sha256"], "authorization": report["authorization"] == "run_phase1634_c116_synthesis_heatmap_and_closure"}
    save("independent_validation_audit", 1633, checks, report["authorization"])


def closure_audit() -> None:
    closure = core.load(OUT / "analysis/closure.json"); internal = core.load(OUT / "audit/internal_closure_audit.json"); payload = core.load(PUBLIC); effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C116"]; raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C116"]
    nomination = payload["c116_batch"]["nomination"]
    checks = {"internal": internal["all_checks_passed"], "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == internal["asset_sha256"], "effects": len(effects) == 231 and all(len(row["values"]) == 2560 for row in effects), "raw": len(raw) == 24 and all(len(row["values"]) == 2560 for row in raw), "candidate_visible": any(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in effects) and sum(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in raw) == 3, "batch": payload["campaign"] == "C109-C116" and "c115_batch" in payload and "c116_batch" in payload and len(payload["c116_batch"]["summaries"]) == 4, "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"], "authorization": closure["next_authorization"].startswith("C117 observation-first fourth-family whole-part exception campaign")}
    save("independent_closure_audit", 1634, checks, "append_c115_c116_memos_build_frontend")


STAGES = {"contract": contract_audit, "capture": capture_audit, "discover": discovery_audit, "validate": validation_audit, "closure": closure_audit}
def main(stage: str) -> None: STAGES[stage]()
