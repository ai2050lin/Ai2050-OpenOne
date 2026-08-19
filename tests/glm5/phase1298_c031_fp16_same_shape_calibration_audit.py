#!/usr/bin/env python3
"""Independent pre/post audit for Phase 1298 C031 calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1298_c031_fp16_same_shape_calibration"
PARENT = T / "result/phase1297_c031_event_interval_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_calibration_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/calibration_arrays.npz"
META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/calibration_summary.json"
TOLERANCE = OUT / "protocol/frozen_empirical_tolerance.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1298_c031_fp16_same_shape_calibration.py"
SCRIPT = Path(__file__).resolve()

EXPECTED = {
    "case_count_min": 96, "finite_fraction_min": 1.0,
    "exact_duplicate_relative_max": 1e-6, "same_batch_prefix_relative_max": 0.0025,
    "cross_composition_prefix_relative_max": 0.005, "derived_tolerance_multiplier": 4.0,
    "derived_tolerance_floor": 1e-6, "derived_tolerance_cap": 0.01,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while c := f.read(1024 * 1024): h.update(c)
    return h.hexdigest()


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def add(c: list[dict[str, Any]], n: str, p: bool, d: Any) -> None:
    c.append({"name": n, "passed": bool(p), "detail": d})


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    c: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(c, "digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(c, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(c, "parent_authorized", load(PARENT / "analysis/final.json")["authorization"] == "phase1298_numerical_calibration_only" and load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"], "parent")
    add(c, "thresholds", protocol["thresholds"] == EXPECTED, protocol["thresholds"])
    manifest = read_jsonl(MANIFEST)
    add(c, "manifest_hash_count", protocol["dependencies"]["manifest"] == sha(MANIFEST) and len(manifest) == 96, len(manifest))
    add(c, "manifest_unique", len({x["calibration_id"] for x in manifest}) == 96, len(manifest))
    add(c, "prefix_identity_frozen", all(x["prefix_identity_through_record_value"] for x in manifest), "all")
    add(c, "roles_depths", protocol["roles"] == ["record_slot0_entity", "record_slot0_value"] and protocol["depths"] == list(range(37)), [protocol["roles"], len(protocol["depths"])])
    add(c, "fixed_shape", protocol["execution"]["global_fixed_sequence_length"] and protocol["execution"]["fixed_batch_size"] == 12 and protocol["execution"]["explicit_position_ids"], protocol["execution"])
    add(c, "single_run", protocol["formal_run_budget"] == 1 and protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization", [protocol["formal_run_budget"], protocol["model"]])
    return c


def write(path: Path, checks: list[dict[str, Any]], stage: str, auth: str) -> None:
    passed = all(x["passed"] for x in checks)
    doc = {"phase": 1298, "campaign": "C031", "audit_stage": stage, "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False, "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks), "all_checks_passed": passed, "authorization": auth if passed else "none", "protocol_digest": load(PROTOCOL)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": doc["passed_count"], "total": doc["total_count"], "authorization": doc["authorization"]}))
    if not passed: raise SystemExit(1)


def pre() -> None:
    p = load(PROTOCOL); c = base(p)
    add(c, "formal_outputs_absent", not any(x.exists() for x in (ARRAYS, META, SUMMARY, TOLERANCE, FINAL, COMPLETE)), "clear")
    write(PRE, c, "pre_model", "run_phase1298_once")


def post() -> None:
    p = load(PROTOCOL); c = base(p)
    a = np.load(ARRAYS, allow_pickle=False)
    exact, prefix, cross = a["exact_duplicate"], a["same_batch_prefix"], a["cross_composition"]
    add(c, "array_shapes", exact.shape == prefix.shape == cross.shape == (96, 37, 2), [exact.shape, prefix.shape, cross.shape])
    add(c, "array_finite", all(np.isfinite(x).all() for x in (exact, prefix, cross)), "finite")
    maxima = {"exact_duplicate_relative_max": float(exact.max()), "same_batch_prefix_relative_max": float(prefix.max()), "cross_composition_prefix_relative_max": float(cross.max())}
    summary, final, tol, meta = load(SUMMARY), load(FINAL), load(TOLERANCE), load(META)
    add(c, "maxima_recompute", summary["maxima"] == maxima, maxima)
    max_noise = max(maxima.values())
    tau = max(EXPECTED["derived_tolerance_floor"], min(EXPECTED["derived_tolerance_cap"], EXPECTED["derived_tolerance_multiplier"] * max_noise))
    add(c, "tolerance_recompute", abs(tol["tau"] - tau) < 1e-12 and tol["source_array_sha256"] == sha(ARRAYS), {"tau": tau})
    gates = {"case_count": True, "finite": True, "exact_duplicate": maxima["exact_duplicate_relative_max"] <= EXPECTED["exact_duplicate_relative_max"], "same_batch_prefix": maxima["same_batch_prefix_relative_max"] <= EXPECTED["same_batch_prefix_relative_max"], "cross_composition": maxima["cross_composition_prefix_relative_max"] <= EXPECTED["cross_composition_prefix_relative_max"], "derived_tolerance_below_cap": tau < EXPECTED["derived_tolerance_cap"]}
    add(c, "gates_recompute", summary["gates"] == gates and summary["all_gates_passed"] == all(gates.values()), gates)
    auth = "phase1299_qwen3_behavior_only" if all(gates.values()) else "close_c031_as_numerically_unqualified"
    add(c, "authorization", final["authorization"] == auth and final["all_gates_passed"] == all(gates.values()), auth)
    qa = meta["model_audit"]
    add(c, "fp16_no_quant", qa["has_fp16_parameters"] and not qa["has_quantized_modules"], qa)
    add(c, "single_completion", load(COMPLETE)["formal_runs_consumed"] == 1, load(COMPLETE))
    write(POST, c, "post_model", auth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("preaudit", "postaudit")); args = parser.parse_args()
    pre() if args.stage == "preaudit" else post()
