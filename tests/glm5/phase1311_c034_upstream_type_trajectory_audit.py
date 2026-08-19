#!/usr/bin/env python3
"""Independent pre/post audit for Phase1311 C034 typed trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PHASE = 1311
CAMPAIGN = "C034"
OUT = T / "result/phase1311_c034_upstream_type_trajectory"
PARENT = T / "result/phase1310_c034_qwen3_typed_behavior"
CONTRACT = T / "result/phase1309_c034_typed_response_camera_contract"
MATERIAL = CONTRACT / "material/frozen_typed_response_pairs.jsonl"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_trajectory_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/trajectory_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/trajectory_summary.json"
SELECTED = OUT / "analysis/selected_cell.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1311_c034_upstream_type_trajectory.py"
SCRIPT = Path(__file__).resolve()

PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
QUERY_DEPTHS = (8, 14, 20, 26, 32)
CELLS = tuple(("query_end", d) for d in QUERY_DEPTHS) + (("answer_boundary", 26),)
EPS = 1e-12


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def eligible(metric: dict[str, float], th: dict[str, float]) -> bool:
    return (
        metric["same_attribute_cross_surface_cosine_median"] >= th["same_attribute_cross_surface_cosine_median_min"]
        and metric["type_gap_median"] >= th["type_gap_median_min"]
        and metric["type_gap_positive_fraction"] >= th["type_gap_positive_fraction_min"]
        and metric["active_to_null_norm_ratio"] >= th["active_to_null_norm_ratio_min"]
    )


def cell_metrics(same: np.ndarray, wrong: np.ndarray, active: np.ndarray, null: np.ndarray,
                 metadata: list[dict[str, Any]], partition: str, cell_i: int) -> dict[str, float]:
    idx = [i for i, x in enumerate(metadata) if x["partition"] == partition]
    gap = same[idx, cell_i] - wrong[idx, cell_i]
    return {
        "same_attribute_cross_surface_cosine_median": float(np.median(same[idx, cell_i])),
        "wrong_attribute_cosine_median": float(np.median(wrong[idx, cell_i])),
        "type_gap_median": float(np.median(gap)),
        "type_gap_positive_fraction": float(np.mean(gap > 0)),
        "active_to_null_norm_ratio": float(np.median(active[idx, cell_i])) / max(float(np.median(null[idx, cell_i])), EPS),
    }


def recompute(arrays: Any, metadata: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    same = arrays["same_attribute_cosine"]
    wrong = arrays["wrong_attribute_cosine"]
    active = arrays["active_delta_norm"]
    null = arrays["null_delta_norm"]
    behavior = arrays["behavior_correct"]
    metrics = {
        p: {f"{role}@{depth}": cell_metrics(same, wrong, active, null, metadata, p, ci)
            for ci, (role, depth) in enumerate(CELLS)}
        for p in PARTITIONS
    }
    candidates: list[tuple[float, int, int, int]] = []
    for ci, depth in enumerate(QUERY_DEPTHS):
        metric = metrics["discovery"][f"query_end@{depth}"]
        if eligible(metric, th):
            candidates.append((metric["type_gap_median"], -depth, ci, depth))
    candidates.sort(reverse=True)
    selected = None
    gates = {
        "finite": bool(all(np.isfinite(arrays[name]).all() for name in
                           ("same_attribute_cosine", "wrong_attribute_cosine", "active_delta_norm", "null_delta_norm"))),
        "behavior_replay": float(np.mean(behavior)) >= th["behavior_replay_accuracy_min"],
        "discovery_candidate_exists": bool(candidates),
    }
    if candidates:
        _, _, selected_i, selected_depth = candidates[0]
        selected = {
            "role": "query_end", "depth": selected_depth, "cell_index": selected_i,
            "discovery_metric": metrics["discovery"][f"query_end@{selected_depth}"],
        }
        for partition in ("confirmation", "holdout"):
            selected_metric = metrics[partition][f"query_end@{selected_depth}"]
            late_metric = metrics[partition]["answer_boundary@26"]
            gates[f"{partition}_selected_cell"] = eligible(selected_metric, th)
            gates[f"{partition}_upstream_over_late"] = (
                selected_metric["type_gap_median"] - late_metric["type_gap_median"]
                >= th["upstream_over_late_type_gap_min"]
            )
    else:
        for partition in ("confirmation", "holdout"):
            gates[f"{partition}_selected_cell"] = False
            gates[f"{partition}_upstream_over_late"] = False
    return {
        "partition_cell_metrics": metrics,
        "selected_cell": selected,
        "behavior_replay_accuracy": float(np.mean(behavior)),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "parent_authorization",
        load(PARENT / "analysis/final.json").get("authorization") == "phase1311_typed_trajectory_only"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"), "Phase1310")
    contract = load(CONTRACT / "protocol/preregistration.json")
    add(checks, "frozen_thresholds", protocol["thresholds"] == contract["trajectory"]["thresholds"], protocol["thresholds"])
    add(checks, "frozen_cells", protocol["cells"] == [{"role": r, "depth": d} for r, d in CELLS], protocol["cells"])
    add(checks, "dependencies",
        protocol["dependencies"] == {
            "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
            "parent_final": sha(PARENT / "analysis/final.json"),
            "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
            "contract": sha(CONTRACT / "protocol/preregistration.json"),
            "material": sha(MATERIAL), "manifest": sha(M),
        }, protocol["dependencies"])
    manifest = rows(M)
    add(checks, "manifest_hash_count",
        protocol["manifest"]["sha256"] == sha(M) and len(manifest) == 288,
        {"sha256": sha(M), "count": len(manifest)})
    counts = Counter(x["partition"] for x in manifest)
    add(checks, "partition_balance", counts == Counter({p: 96 for p in PARTITIONS}), dict(counts))
    add(checks, "case_uniqueness", len({x["case_key"] for x in manifest}) == len(manifest), len(manifest))
    structure_ok = True
    detail = {"cases": len(manifest), "attribute_surface_cells": Counter()}
    for x in manifest:
        identities = x["identity_positions"]
        expected_wrong = ATTRS[(ATTRS.index(x["attribute"]) + 1) % len(ATTRS)]
        structure_ok &= (
            len(identities) == 2 and identities[0] != identities[1]
            and x["wrong_attribute"] == expected_wrong
            and x["opposite_surface"] == SURFACES[1 - SURFACES.index(x["anchor_surface"])]
            and [s["gold_position"] for s in x["target_states"]] == identities
            and [s["gold_position"] for s in x["same_attribute_states"]] == identities
            and [s["gold_position"] for s in x["wrong_attribute_states"]] == identities
            and x["target_states"][0]["candidate_ids"] == x["same_attribute_states"][0]["candidate_ids"]
            and x["target_states"][0]["candidate_ids"] == x["wrong_attribute_states"][0]["candidate_ids"]
            and x["null_states"][0]["gold_position"] == x["null_states"][1]["gold_position"]
            and all(s["positions"]["query_end"] < s["positions"]["answer_boundary"] < len(s["ids"])
                    for key in ("target_states", "same_attribute_states", "wrong_attribute_states", "null_states")
                    for s in x[key])
        )
        detail["attribute_surface_cells"][(x["partition"], x["attribute"], x["anchor_surface"])] += 1
    detail["attribute_surface_cells"] = {"|".join(k): v for k, v in detail["attribute_surface_cells"].items()}
    structure_ok &= all(v == 8 for v in detail["attribute_surface_cells"].values()) and len(detail["attribute_surface_cells"]) == 36
    add(checks, "identity_aligned_factorial_structure", structure_ok, detail)
    add(checks, "formal_budget_and_stops",
        protocol["formal_run_budget"] == 1 and protocol["hard_stops"] == [
            "No intervention", "No component selection", "No nonregistered depth or role",
            "No post-unblinding threshold or pairing change", "No second formal model run"],
        {"budget": protocol["formal_run_budget"], "hard_stops": protocol["hard_stops"]})
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit_stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
        "all_checks_passed": passed, "authorization": authorization if passed else "none",
        "protocol_digest": load(P)["protocol_digest"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": result["passed_count"], "total": result["total_count"],
                     "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    add(checks, "formal_outputs_absent", not any(x.exists() for x in (A, META, S, SELECTED, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1311_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    arrays = np.load(A, allow_pickle=False)
    metadata = load(META)
    case_metadata = metadata["case_metadata"]
    add(checks, "array_schema",
        set(arrays.files) == {"same_attribute_cosine", "wrong_attribute_cosine", "active_delta_norm", "null_delta_norm", "behavior_correct"}
        and arrays["same_attribute_cosine"].shape == (288, 6)
        and arrays["wrong_attribute_cosine"].shape == (288, 6)
        and arrays["active_delta_norm"].shape == (288, 6)
        and arrays["null_delta_norm"].shape == (288, 6)
        and arrays["behavior_correct"].shape == (288, 2),
        {name: list(arrays[name].shape) for name in arrays.files})
    add(checks, "raw_hashes", metadata["array_sha256"] == sha(A) and metadata["manifest_sha256"] == sha(M),
        {"array": sha(A), "manifest": sha(M)})
    manifest = rows(M)
    expected_metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "wrong_attribute", "anchor_surface", "opposite_surface")} for x in manifest]
    add(checks, "metadata_alignment", case_metadata == expected_metadata, len(case_metadata))
    result = recompute(arrays, case_metadata, protocol["thresholds"])
    summary = load(S)
    add(checks, "independent_metrics", summary["partition_cell_metrics"] == result["partition_cell_metrics"]
        and summary["behavior_replay_accuracy"] == result["behavior_replay_accuracy"],
        {"behavior_replay_accuracy": result["behavior_replay_accuracy"], "selected": result["selected_cell"]})
    add(checks, "independent_selection", summary["selected_cell"] == result["selected_cell"]
        and load(SELECTED)["selected_cell"] == result["selected_cell"], result["selected_cell"])
    add(checks, "independent_gates", summary["gates"] == result["gates"]
        and summary["all_gates_passed"] == result["all_gates_passed"], result["gates"])
    authorization = "phase1312_upstream_selective_rescue_only" if result["all_gates_passed"] else "close_c034_without_causal"
    final = load(F)
    add(checks, "verdict_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"], final)
    qa = metadata["model_audit"]
    add(checks, "fp16_cuda", qa["has_fp16_parameters"] and not qa["has_quantized_modules"]
        and metadata["cuda_peak_allocated_bytes"] > 0, {"audit": qa, "peak": metadata["cuda_peak_allocated_bytes"]})
    complete = load(C)
    add(checks, "formal_budget_consumed", complete["formal_runs_consumed"] == 1
        and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
