#!/usr/bin/env python3
"""Independent pre/post audit for Phase1315; never imports the execution script."""
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
PHASE = 1315
CAMPAIGN = "C035"
OUT = T / "result/phase1315_c035_multisite_position_cut"
PARENT = T / "result/phase1314_c035_qwen3_behavior"
CONTRACT = T / "result/phase1313_c035_semantic_position_cut_contract"
MATERIAL = CONTRACT / "material/frozen_position_cut_pairs.jsonl"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_cut_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/cut_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/cut_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1315_c035_multisite_position_cut.py"
SCRIPT = Path(__file__).resolve()
PARTITIONS = ("confirmation", "holdout")
ARMS = ("baseline", "query_end_only", "query_bundle", "record_bundle", "full_registered", "self_retention")
ROLE_SETS = {
    "query_end_only": ("query_end",),
    "query_bundle": ("query_attribute", "query_value", "query_end"),
    "record_bundle": ("record_entities", "record_queried_values"),
    "full_registered": ("query_attribute", "query_value", "query_end", "record_entities", "record_queried_values"),
}
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
    drops = margins[:, 0, None] - margins
    partitions = {}
    gates = {"finite": bool(np.isfinite(margins).all())}
    for partition in PARTITIONS:
        idx = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        cell = {
            "baseline_accuracy": float(np.mean(answers[idx, 0])),
            "query_end_accuracy": float(np.mean(answers[idx, 1])),
            "query_bundle_accuracy": float(np.mean(answers[idx, 2])),
            "record_bundle_accuracy": float(np.mean(answers[idx, 3])),
            "full_cut_accuracy": float(np.mean(answers[idx, 4])),
            "self_retention": float(np.mean(answers[idx, 5])),
            "query_end_margin_drop_median": float(np.median(drops[idx, 1])),
            "full_cut_margin_drop_median": float(np.median(drops[idx, 4])),
        }
        cell["full_over_qend_drop_ratio"] = cell["full_cut_margin_drop_median"] / max(
            abs(cell["query_end_margin_drop_median"]), EPS)
        partitions[partition] = cell
        gates[f"{partition}_baseline"] = cell["baseline_accuracy"] >= th["baseline_accuracy_min"]
        gates[f"{partition}_self"] = cell["self_retention"] >= th["self_retention_min"]
        gates[f"{partition}_full_accuracy"] = cell["full_cut_accuracy"] <= th["full_cut_accuracy_max"]
        gates[f"{partition}_full_drop"] = cell["full_cut_margin_drop_median"] >= th["full_cut_margin_drop_median_min"]
        gates[f"{partition}_ratio"] = cell["full_over_qend_drop_ratio"] >= th["full_over_qend_drop_ratio_min"]
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])),
        "arm_accuracy": {arm: float(np.mean(answers[:, i])) for i, arm in enumerate(ARMS)},
        "margin_median": {arm: float(np.median(margins[:, i])) for i, arm in enumerate(ARMS)},
        "margin_drop_median": {arm: float(np.median(drops[:, i])) for i, arm in enumerate(ARMS)},
    }
    return {"metrics": metrics, "partitions": partitions, "gates": gates, "all_gates_passed": all(gates.values())}


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)},
        protocol["source_hashes"])
    add(checks, "parent_authorization",
        load(PARENT / "analysis/final.json").get("authorization") == "phase1315_multisite_cut_only"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"), "Phase1314")
    contract = load(CONTRACT / "protocol/preregistration.json")
    add(checks, "frozen_contract", protocol["depth"] == 14
        and protocol["thresholds"] == contract["position_cut"]["thresholds"]
        and protocol["role_sets"] == {k: list(v) for k, v in ROLE_SETS.items()},
        {"depth": protocol["depth"], "sets": protocol["role_sets"]})
    expected_dependencies = {
        "parent_protocol": sha(PARENT / "protocol/preregistration.json"),
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
    add(checks, "cell_balance", all(sum(x["attribute"] == a and x["surface"] == s for x in manifest) == 12
        for a in ("temperature", "texture", "origin", "condition", "category", "priority")
        for s in ("registry_prose", "registry_ledger")), "12 per attribute-surface")
    material = {x["pair_key"]: x for x in rows(MATERIAL)}
    structure = True
    for item in manifest:
        pair = material[item["case_key"]]
        state0, state1 = pair["states"]
        expected_sets = {name: sorted({p for role in roles for p in state1["positions"][role]})
                         for name, roles in ROLE_SETS.items()}
        structure &= (
            item["state0"] == state0 and item["state1"] == state1
            and item["position_sets"] == expected_sets
            and len(state0["ids"]) == len(state1["ids"])
            and state0["positions"] == state1["positions"]
        )
    add(checks, "manifest_semantic_alignment", structure, "state0/state1 role positions aligned")
    add(checks, "arms_budget_stops", protocol["arms"] == list(ARMS) and protocol["formal_run_budget"] == 1
        and protocol["failure_authorization"] == "close_c035_at_registered_cut_boundary"
        and len(protocol["hard_stops"]) == 5, {"arms": protocol["arms"], "stops": protocol["hard_stops"]})
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
    write(PRE, checks, "pre_model", "run_phase1315_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    margins, answers = z["gold_minus_max_nongold_margin"], z["gold_correct"]
    add(checks, "array_schema", set(z.files) == {"gold_minus_max_nongold_margin", "gold_correct"}
        and margins.shape == answers.shape == (144, 6), {name: list(z[name].shape) for name in z.files})
    meta = load(META)
    add(checks, "raw_hashes", meta["array_sha256"] == sha(A) and meta["manifest_sha256"] == sha(M),
        {"array": sha(A), "manifest": sha(M)})
    expected_metadata = [{k: x[k] for k in ("case_key", "partition", "profile_index", "attribute", "surface")}
                         for x in rows(M)]
    add(checks, "metadata_alignment", meta["case_metadata"] == expected_metadata, len(meta["case_metadata"]))
    result = recompute(margins, answers, meta["case_metadata"], protocol["thresholds"])
    summary = load(S)
    add(checks, "independent_metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "independent_partitions", summary["partitions"] == result["partitions"], result["partitions"])
    add(checks, "independent_gates", summary["gates"] == result["gates"]
        and summary["all_gates_passed"] == result["all_gates_passed"], result["gates"])
    authorization = "phase1316_typed_rescue_only" if result["all_gates_passed"] else "close_c035_at_registered_cut_boundary"
    final = load(F)
    add(checks, "verdict_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"]
        and final["c035_closed"] == (not result["all_gates_passed"]), final)
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
