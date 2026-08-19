#!/usr/bin/env python3
"""Independent pre/post audit for Phase1316; never imports the execution script."""
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
PHASE = 1316
CAMPAIGN = "C035"
OUT = T / "result/phase1316_c035_typed_multireadout_rescue"
PARENT = T / "result/phase1315_c035_multisite_position_cut"
CONTRACT = T / "result/phase1313_c035_semantic_position_cut_contract"
MATERIAL = CONTRACT / "material/frozen_position_cut_pairs.jsonl"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_typed_rescue_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/typed_rescue_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/typed_rescue_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1316_c035_typed_multireadout_rescue.py"
SCRIPT = Path(__file__).resolve()
PARTITIONS = ("confirmation", "holdout")
ATTRS = ("temperature", "texture", "origin", "condition", "category", "priority")
SURFACES = ("registry_prose", "registry_ledger")
ROLES = ("query_attribute", "query_value", "query_end", "record_entities", "record_queried_values")
ARMS = ("baseline", "block_only", "self_retention") + tuple(f"active_{a}" for a in ATTRS) + tuple(f"null_{a}" for a in ATTRS)
EPS = 1e-12


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def recompute(margins: np.ndarray, answers: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    active, null = margins[:, 3:9], margins[:, 9:15]
    active_answers, null_answers = answers[:, 3:9], answers[:, 9:15]
    attr_index = np.array([ATTRS.index(x["receiver_attribute"]) for x in metadata])
    correct = active[np.arange(len(metadata)), attr_index]
    correct_answers = active_answers[np.arange(len(metadata)), attr_index]
    correct_gain = correct - margins[:, 1]
    denominator = margins[:, 0] - margins[:, 1]
    recovery = correct_gain / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    wrong_gain = np.stack([active[i, np.arange(6) != ai] - margins[i, 1] for i, ai in enumerate(attr_index)])
    wrong_answers = np.stack([active_answers[i, np.arange(6) != ai] for i, ai in enumerate(attr_index)])
    null_gain = null - margins[:, 1, None]
    partitions = {}
    gates = {"finite": bool(np.isfinite(margins).all())}
    for partition in PARTITIONS:
        idx = np.array([i for i, x in enumerate(metadata) if x["partition"] == partition])
        cell = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "block_accuracy": float(np.mean(answers[idx, 1])),
            "self_retention": float(np.mean(answers[idx, 2])),
            "correct_rescue_accuracy": float(np.mean(correct_answers[idx])),
            "correct_recovery_fraction_median": float(np.nanmedian(recovery[idx])),
            "valid_recovery_fraction": float(np.mean(np.isfinite(recovery[idx]))),
            "own_attribute_win_fraction": float(np.mean(correct_gain[idx] > np.max(wrong_gain[idx], axis=1))),
            "wrong_attribute_exclusion_fraction": float(1.0 - np.mean(wrong_answers[idx])),
            "null_exclusion_fraction": float(1.0 - np.mean(null_answers[idx])),
            "correct_gain_median": float(np.median(correct_gain[idx])),
            "max_wrong_gain_median": float(np.median(np.max(wrong_gain[idx], axis=1))),
            "max_null_gain_median": float(np.median(np.max(null_gain[idx], axis=1))),
        }
        partitions[partition] = cell
        gates[f"{partition}_recovery_defined"] = cell["valid_recovery_fraction"] == 1.0
        gates[f"{partition}_correct_accuracy"] = cell["correct_rescue_accuracy"] >= th["correct_rescue_accuracy_min"]
        gates[f"{partition}_recovery"] = cell["correct_recovery_fraction_median"] >= th["correct_recovery_fraction_median_min"]
        gates[f"{partition}_own_win"] = cell["own_attribute_win_fraction"] >= th["own_attribute_win_fraction_min"]
        gates[f"{partition}_wrong_exclusion"] = cell["wrong_attribute_exclusion_fraction"] >= th["wrong_attribute_exclusion_fraction_min"]
        gates[f"{partition}_null_exclusion"] = cell["null_exclusion_fraction"] >= th["null_exclusion_fraction_min"]
        gates[f"{partition}_self"] = cell["self_retention"] >= th["self_retention_min"]
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])), "block_accuracy": float(np.mean(answers[:, 1])),
        "self_retention": float(np.mean(answers[:, 2])), "correct_rescue_accuracy": float(np.mean(correct_answers)),
        "correct_recovery_fraction_median": float(np.nanmedian(recovery)),
        "own_attribute_win_fraction": float(np.mean(correct_gain > np.max(wrong_gain, axis=1))),
        "wrong_attribute_exclusion_fraction": float(1.0 - np.mean(wrong_answers)),
        "null_exclusion_fraction": float(1.0 - np.mean(null_answers)),
        "correct_gain_median": float(np.median(correct_gain)),
        "max_wrong_gain_median": float(np.median(np.max(wrong_gain, axis=1))),
        "max_null_gain_median": float(np.median(np.max(null_gain, axis=1))),
    }
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)},
        protocol["source_hashes"])
    add(checks, "parent_authorization",
        load(PARENT / "analysis/final.json").get("authorization") == "phase1316_typed_rescue_only"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"), "Phase1315")
    contract = load(CONTRACT / "protocol/preregistration.json")
    add(checks, "frozen_contract", protocol["block_depth"] == 14 and protocol["rescue_depth"] == 15
        and protocol["roles"] == list(ROLES) and protocol["arms"] == list(ARMS)
        and protocol["thresholds"] == contract["typed_rescue"]["thresholds"],
        {"depths": [protocol["block_depth"], protocol["rescue_depth"]], "roles": protocol["roles"]})
    expected_dependencies = {
        "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
        "parent_manifest": sha(PARENT / "protocol/frozen_cut_manifest.jsonl"),
        "parent_arrays": sha(PARENT / "raw/cut_arrays.npz"),
        "parent_final": sha(PARENT / "analysis/final.json"),
        "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
        "contract": sha(CONTRACT / "protocol/preregistration.json"),
        "material": sha(MATERIAL), "manifest": sha(M),
    }
    add(checks, "dependencies", protocol["dependencies"] == expected_dependencies, protocol["dependencies"])
    manifest = rows(M)
    add(checks, "manifest_hash_count", len(manifest) == 144 and protocol["manifest"]["sha256"] == sha(M),
        {"count": len(manifest), "sha": sha(M)})
    add(checks, "partition_balance", Counter(x["partition"] for x in manifest) == Counter({p: 72 for p in PARTITIONS}),
        dict(Counter(x["partition"] for x in manifest)))
    add(checks, "receiver_balance", all(sum(x["receiver_attribute"] == a and x["target_surface"] == s for x in manifest) == 12
        for a in ATTRS for s in SURFACES), "12 per receiver-attribute/surface")
    structure = True
    for item in manifest:
        identities = item["identity_positions"]
        structure &= item["target_surface"] != item["donor_surface"] and set(item["active_donors"]) == set(ATTRS)
        structure &= set(item["null_donors"]) == set(ATTRS) and len(item["receiver_states"]) == 2
        for attr in ATTRS:
            active = item["active_donors"][attr]
            null = item["null_donors"][attr]
            structure &= [x["gold_position"] for x in active] == identities
            structure &= null[0]["gold_position"] == null[1]["gold_position"]
            for role in ROLES:
                structure &= len(active[0]["positions"][role]) == len(active[1]["positions"][role])
                structure &= len(null[0]["positions"][role]) == len(null[1]["positions"][role])
                structure &= len(active[0]["positions"][role]) == len(item["receiver_states"][1]["positions"][role])
    add(checks, "typed_donor_alignment", structure, "opposite-surface identity-oriented active and matched-null donors")
    add(checks, "budget_stops", protocol["formal_run_budget"] == 1 and len(protocol["hard_stops"]) == 5
        and protocol["success_authorization"].startswith("close_c035")
        and protocol["failure_authorization"].startswith("close_c035"), protocol["hard_stops"])
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
    checks = base(load(P))
    add(checks, "formal_outputs_absent", not any(path.exists() for path in (A, META, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1316_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    margins, answers = z["gold_minus_max_nongold_margin"], z["gold_correct"]
    add(checks, "array_schema", set(z.files) == {"gold_minus_max_nongold_margin", "gold_correct"}
        and margins.shape == answers.shape == (144, 15), {name: list(z[name].shape) for name in z.files})
    meta = load(META)
    add(checks, "raw_hashes", meta["array_sha256"] == sha(A) and meta["manifest_sha256"] == sha(M),
        {"array": sha(A), "manifest": sha(M)})
    expected_metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "receiver_attribute",
                                             "target_surface", "donor_surface")} for x in rows(M)]
    add(checks, "metadata_alignment", meta["case_metadata"] == expected_metadata, len(meta["case_metadata"]))
    result = recompute(margins, answers, meta["case_metadata"], protocol["thresholds"])
    summary = load(S)
    add(checks, "independent_metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "independent_partitions", summary["partitions"] == result["partitions"], result["partitions"])
    add(checks, "independent_gates", summary["gates"] == result["gates"]
        and summary["all_gates_passed"] == result["all_gates_passed"], result["gates"])
    authorization = ("close_c035_with_typed_multisite_rescue_candidate" if result["all_gates_passed"]
                     else "close_c035_with_multisite_dependence_without_type_selectivity")
    final = load(F)
    add(checks, "verdict_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"] and final["c035_closed"] is True, final)
    qa = meta["model_audit"]
    add(checks, "fp16_cuda", qa["has_fp16_parameters"] and not qa["has_quantized_modules"]
        and meta["cuda_peak_allocated_bytes"] > 0, {"qa": qa, "peak": meta["cuda_peak_allocated_bytes"]})
    complete = load(C)
    add(checks, "formal_budget_consumed", complete["formal_runs_consumed"] == 1
        and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
