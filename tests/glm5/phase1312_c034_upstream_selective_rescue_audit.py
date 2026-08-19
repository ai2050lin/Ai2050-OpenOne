#!/usr/bin/env python3
"""Independent pre/post audit for Phase1312; never imports the execution script."""
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
PHASE = 1312
CAMPAIGN = "C034"
OUT = T / "result/phase1312_c034_upstream_selective_rescue"
PARENT = T / "result/phase1311_c034_upstream_type_trajectory"
CONTRACT = T / "result/phase1309_c034_typed_response_camera_contract"
MATERIAL = CONTRACT / "material/frozen_typed_response_pairs.jsonl"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_rescue_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/rescue_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/rescue_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1312_c034_upstream_selective_rescue.py"
SCRIPT = Path(__file__).resolve()

PARTITIONS = ("confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
ARMS = ("baseline", "block_only", "correct_cross_surface", "matched_null_cross_surface",
        "wrong_attribute_cross_surface", "self_retention")
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


def recompute(margins: np.ndarray, answers: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]):
    baseline, blocked, correct, null, wrong = [margins[:, i] for i in range(5)]
    correct_gain, null_gain, wrong_gain = correct - blocked, null - blocked, wrong - blocked
    denominator = baseline - blocked
    recovery = correct_gain / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    partitions = {}
    for partition in PARTITIONS:
        idx = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        partitions[partition] = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "blocked_target_identity_accuracy": float(np.mean(answers[idx, 1])),
            "correct_rescue_accuracy": float(np.mean(answers[idx, 2])),
            "self_retention_accuracy": float(np.mean(answers[idx, 5])),
        }
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])),
        "blocked_target_identity_accuracy": float(np.mean(answers[:, 1])),
        "correct_rescue_accuracy": float(np.mean(answers[:, 2])),
        "self_retention_accuracy": float(np.mean(answers[:, 5])),
        "correct_gain_median": float(np.median(correct_gain)),
        "null_gain_median": float(np.median(null_gain)),
        "wrong_attribute_gain_median": float(np.median(wrong_gain)),
        "recovery_fraction_median": float(np.nanmedian(recovery)),
        "valid_recovery_fraction": float(np.mean(np.isfinite(recovery))),
        "correct_over_null_margin_ratio": float(np.median(correct_gain)) / max(abs(float(np.median(null_gain))), EPS),
        "pairwise_correct_win_fraction": float(np.mean(correct_gain > np.maximum(null_gain, wrong_gain))),
        "natural_retention": float(np.mean(answers[:, [0, 5]])),
    }
    gates = {
        "finite": bool(np.isfinite(margins).all()),
        "recovery_defined": metrics["valid_recovery_fraction"] == 1.0,
        "recovery": metrics["recovery_fraction_median"] >= th["recovery_fraction_median_min"],
        "null_ratio": metrics["correct_over_null_margin_ratio"] >= th["correct_over_null_margin_ratio_min"],
        "pairwise_correct_win": metrics["pairwise_correct_win_fraction"] >= th["pairwise_correct_win_fraction_min"],
        "natural_retention": metrics["natural_retention"] >= th["natural_retention_min"],
    }
    for partition in PARTITIONS:
        cell = partitions[partition]
        gates[f"{partition}_baseline"] = cell["baseline_accuracy"] >= th["baseline_accuracy_min"]
        gates[f"{partition}_blocked"] = cell["blocked_target_identity_accuracy"] <= th["blocked_target_identity_accuracy_max"]
        gates[f"{partition}_correct_rescue"] = cell["correct_rescue_accuracy"] >= th["correct_rescue_accuracy_min"]
        gates[f"{partition}_retention"] = cell["self_retention_accuracy"] >= th["natural_retention_min"]
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "parent_authorization",
        load(PARENT / "analysis/final.json").get("authorization") == "phase1312_upstream_selective_rescue_only"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"), "Phase1311")
    selected = load(PARENT / "analysis/selected_cell.json")["selected_cell"]
    add(checks, "frozen_selected_cell", selected["role"] == "query_end" and selected["depth"] == 14
        and selected["cell_index"] == 1 and protocol["block_depth"] == 14 and protocol["rescue_depth"] == 15,
        {"selected": selected, "block": protocol["block_depth"], "rescue": protocol["rescue_depth"]})
    contract = load(CONTRACT / "protocol/preregistration.json")
    add(checks, "frozen_thresholds", protocol["thresholds"] == contract["causal"]["thresholds"], protocol["thresholds"])
    expected_dependencies = {
        "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
        "parent_manifest": sha(PARENT / "protocol/frozen_trajectory_manifest.jsonl"),
        "parent_arrays": sha(PARENT / "raw/trajectory_arrays.npz"),
        "parent_selected": sha(PARENT / "analysis/selected_cell.json"),
        "parent_final": sha(PARENT / "analysis/final.json"),
        "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
        "contract": sha(CONTRACT / "protocol/preregistration.json"), "material": sha(MATERIAL), "manifest": sha(M),
    }
    add(checks, "dependencies", protocol["dependencies"] == expected_dependencies, protocol["dependencies"])
    manifest = rows(M)
    add(checks, "manifest_hash_count", len(manifest) == 192 and protocol["manifest"]["sha256"] == sha(M),
        {"count": len(manifest), "sha256": sha(M)})
    add(checks, "partition_balance", Counter(x["partition"] for x in manifest) == Counter({p: 96 for p in PARTITIONS}),
        dict(Counter(x["partition"] for x in manifest)))
    add(checks, "surface_balance", Counter(x["target_surface"] for x in manifest) == Counter({s: 96 for s in SURFACES}),
        dict(Counter(x["target_surface"] for x in manifest)))
    add(checks, "case_uniqueness", len({x["case_key"] for x in manifest}) == 192, len(manifest))

    trajectory = {x["case_key"]: x for x in rows(PARENT / "protocol/frozen_trajectory_manifest.jsonl") if x["partition"] in PARTITIONS}
    material = {x["group_id"]: x for x in rows(MATERIAL)}
    structure_ok = True
    for x in manifest:
        identities = x["identity_positions"]
        source = trajectory[x["source_keys"]["trajectory"]]
        null_source = material[x["source_keys"]["null"]]
        expected_wrong = ATTRS[(ATTRS.index(x["attribute"]) + 1) % len(ATTRS)]
        structure_ok &= (
            x["partition"] in PARTITIONS and x["wrong_attribute"] == expected_wrong
            and x["target_surface"] != x["donor_surface"]
            and x["target_state0"] == source["target_states"][0] and x["target_state1"] == source["target_states"][1]
            and x["correct_state0"] == source["same_attribute_states"][0] and x["correct_state1"] == source["same_attribute_states"][1]
            and x["wrong_state0"] == source["wrong_attribute_states"][0] and x["wrong_state1"] == source["wrong_attribute_states"][1]
            and x["null_state0"] == null_source["states"][0] and x["null_state1"] == null_source["states"][1]
            and null_source["surface"] == x["donor_surface"] and null_source["attribute"] == x["attribute"]
            and [x["target_state0"]["gold_position"], x["target_state1"]["gold_position"]] == identities
            and [x["correct_state0"]["gold_position"], x["correct_state1"]["gold_position"]] == identities
            and [x["wrong_state0"]["gold_position"], x["wrong_state1"]["gold_position"]] == identities
            and x["null_state0"]["gold_position"] == x["null_state1"]["gold_position"]
            and all(state["candidate_ids"] == x["target_state0"]["candidate_ids"] for state in
                    (x["target_state1"], x["correct_state0"], x["correct_state1"], x["wrong_state0"], x["wrong_state1"],
                     x["null_state0"], x["null_state1"]))
            and all(state["positions"]["query_end"] < state["positions"]["answer_boundary"] < len(state["ids"])
                    for key in ("target_state0", "target_state1", "correct_state0", "correct_state1",
                                "null_state0", "null_state1", "wrong_state0", "wrong_state1") for state in (x[key],))
        )
    add(checks, "typed_donor_semantics", structure_ok, "correct/null/wrong donors are opposite-surface and identity aligned")
    add(checks, "arms", protocol["arms"] == list(ARMS), protocol["arms"])
    add(checks, "formal_budget_and_stops", protocol["formal_run_budget"] == 1 and protocol["hard_stops"] == [
        "No discovery partition", "No depth, role, donor, wrong-attribute, or threshold change",
        "No head, MLP, neuron, or subspace search", "No second formal model run",
        "C034 closes after Phase1312 regardless of verdict"],
        {"budget": protocol["formal_run_budget"], "hard_stops": protocol["hard_stops"]})
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit_stage": stage,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
              "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
              "all_checks_passed": passed, "authorization": authorization if passed else "none",
              "protocol_digest": load(P)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": result["passed_count"], "total": result["total_count"],
                     "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    add(checks, "formal_outputs_absent", not any(x.exists() for x in (A, META, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1312_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    margins, answers = z["identity1_minus_identity0_margin"], z["target_identity_correct"]
    add(checks, "array_schema", set(z.files) == {"identity1_minus_identity0_margin", "target_identity_correct"}
        and margins.shape == answers.shape == (192, 6), {name: list(z[name].shape) for name in z.files})
    meta = load(META)
    add(checks, "raw_hashes", meta["array_sha256"] == sha(A) and meta["manifest_sha256"] == sha(M),
        {"array": sha(A), "manifest": sha(M)})
    expected_metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "wrong_attribute", "target_surface", "donor_surface")} for x in rows(M)]
    add(checks, "metadata_alignment", meta["case_metadata"] == expected_metadata, len(meta["case_metadata"]))
    result = recompute(margins, answers, meta["case_metadata"], protocol["thresholds"])
    summary = load(S)
    add(checks, "independent_metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "independent_partitions", summary["partitions"] == result["partitions"], result["partitions"])
    add(checks, "independent_gates", summary["gates"] == result["gates"]
        and summary["all_gates_passed"] == result["all_gates_passed"], result["gates"])
    authorization = ("close_c034_with_upstream_typed_rescue_candidate" if result["all_gates_passed"]
                     else "close_c034_at_upstream_rescue_boundary")
    final = load(F)
    add(checks, "verdict_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"] and final["c034_closed"] is True, final)
    qa = meta["model_audit"]
    add(checks, "fp16_cuda", qa["has_fp16_parameters"] and not qa["has_quantized_modules"]
        and meta["cuda_peak_allocated_bytes"] > 0, {"audit": qa, "peak": meta["cuda_peak_allocated_bytes"]})
    complete = load(C)
    add(checks, "formal_budget_consumed", complete["formal_runs_consumed"] == 1
        and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
