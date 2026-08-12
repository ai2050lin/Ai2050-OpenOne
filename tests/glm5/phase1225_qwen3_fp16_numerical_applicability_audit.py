#!/usr/bin/env python3
"""Independent pre-model and result audit for Phase 1225."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1225_qwen3_fp16_numerical_applicability as p1225


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def fraction(values: list[bool]) -> float:
    return float(sum(bool(value) for value in values) / len(values)) if values else float("nan")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit_document(stage: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "phase": p1225.PHASE,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(bool(row["passed"]) for row in checks),
    }
    result["audit_digest"] = digest(result)
    return result


def preaudit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = read_json(p1225.PROTOCOL_PATH)
    manifest = read_jsonl(p1225.MANIFEST_PATH)
    source_final = read_json(p1225.SOURCE_FINAL)
    source_audit = read_json(p1225.SOURCE_AUDIT)
    add(checks, "source_final", source_final.get("final_digest") == p1225.EXPECTED_SOURCE_FINAL_DIGEST, source_final.get("final_digest"))
    add(
        checks,
        "source_audit",
        source_audit.get("all_checks_passed") and source_audit.get("audit_digest") == p1225.EXPECTED_SOURCE_AUDIT_DIGEST,
        source_audit.get("audit_digest"),
    )
    paths = {
        "script": p1225.SCRIPT,
        "audit_script": p1225.AUDIT_SCRIPT,
        "phase1224_final": p1225.SOURCE_FINAL,
        "phase1224_result_audit": p1225.SOURCE_AUDIT,
        "phase1224_protocol": p1225.SOURCE_PROTOCOL,
        "phase1224_manifest": p1225.SOURCE_MANIFEST,
        "phase1224_records": p1225.SOURCE_RECORDS,
        "phase1223_states": p1225.SOURCE_STATES,
    }
    hash_ok = all(protocol["source_hashes"].get(key) == file_sha256(path) for key, path in paths.items())
    add(checks, "source_hashes", hash_ok, protocol["source_hashes"])
    payload = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", protocol.get("protocol_digest") == digest(payload), protocol.get("protocol_digest"))
    add(checks, "manifest_digest", protocol["material"]["manifest_digest"] == digest(manifest), protocol["material"]["manifest_digest"])
    add(checks, "cardinality", len(manifest) == 160 and len({row["pair_id"] for row in manifest}) == 160, len(manifest))
    add(
        checks,
        "row_digests",
        all(row["row_digest"] == digest({key: value for key, value in row.items() if key != "row_digest"}) for row in manifest),
        "all rows",
    )
    split_counts = {split: sum(row["split"] == split for row in manifest) for split in p1225.SPLITS}
    add(checks, "split_balance", all(value == 40 for value in split_counts.values()), split_counts)
    geometry_ok = all(
        1 <= int(row["continuation_length"]) <= p1225.MAX_CONTINUATION
        and int(row["generation_boundary"]) >= 0
        and float(row["target_shift_abs"]) >= 1.0
        and len(row["candidates"]) == 4
        for row in manifest
    )
    add(checks, "geometry_and_denominator", geometry_ok, "all rows")
    add(checks, "typed_contracts", tuple(protocol["contracts"]) == ("C0", "C1", "C2", "C3"), protocol["contracts"])
    add(
        checks,
        "variant_registry",
        tuple(protocol["reference_variants"]["exact"]) == p1225.EXACT_VARIANTS
        and tuple(protocol["reference_variants"]["shape"]) == p1225.SHAPE_VARIANTS,
        protocol["reference_variants"],
    )
    add(
        checks,
        "envelope_rule",
        protocol["envelope"]["multiplier"] == p1225.ENVELOPE_MULTIPLIER
        and protocol["envelope"]["epsilon"] == p1225.EPSILON
        and protocol["envelope"]["functional_caps"] == p1225.FUNCTIONAL_CAPS,
        protocol["envelope"],
    )
    model_ok = (
        protocol["model"]["name"] == "qwen3"
        and protocol["model"]["precision"] == "float16"
        and protocol["model"]["quantization"] == "none"
        and protocol["model"]["attention_backend"] == "eager"
        and protocol["model"]["loads"] == 2
    )
    add(checks, "model_domain", model_ok, protocol["model"])
    claim_ok = (
        protocol["claim_boundary"]["instrument_only"]
        and protocol["claim_boundary"]["not_language_mechanism"]
        and not protocol["authorization"]["qwen_mechanism_scan"]
        and not protocol["authorization"]["head_or_neuron_search"]
        and not protocol["authorization"]["threshold_relaxation_after_run"]
    )
    add(checks, "claim_and_action_scope", claim_ok, {"claim": protocol["claim_boundary"], "authorization": protocol["authorization"]})
    no_outputs = not any(
        path.exists()
        for path in (
            p1225.REFERENCE_RECORD_PATH,
            p1225.REFERENCE_ARRAY_PATH,
            p1225.REFERENCE_SUMMARY_PATH,
            p1225.RELOAD_RECORD_PATH,
            p1225.RELOAD_SUMMARY_PATH,
            p1225.FINAL_PATH,
        )
    )
    add(checks, "no_model_output_before_preaudit", no_outputs, no_outputs)
    result = audit_document("pre", checks)
    write_json(p1225.PREAUDIT_PATH, result)
    return result


def envelope(values: list[float]) -> float:
    return float(p1225.ENVELOPE_MULTIPLIER * max(values, default=0.0) + p1225.EPSILON)


def independent_evaluate(reference: list[dict[str, Any]], reload: list[dict[str, Any]]) -> dict[str, Any]:
    disc_ref = [row for row in reference if row["split"] == "discovery"]
    hold_ref = [row for row in reference if row["split"] in p1225.HOLDOUT_SPLITS]
    disc_reload = [row for row in reload if row["split"] == "discovery"]
    hold_reload = [row for row in reload if row["split"] in p1225.HOLDOUT_SPLITS]
    caps = p1225.FUNCTIONAL_CAPS

    c0: dict[str, Any] = {}
    c0_pass = True
    for variant in p1225.EXACT_VARIANTS:
        rows = [row["variants"][variant] for row in reference]
        item = {
            "hidden_max_abs": max(row["hidden_max_abs"] for row in rows),
            "logit_max_abs": max(row["logit_max_abs"] for row in rows),
            "score_max_abs": max(row["score_max_abs"] for row in rows),
            "top1_min": min(row["top1_agreement"] for row in rows),
        }
        item["passed"] = (
            item["hidden_max_abs"] <= caps["exact_abs"]
            and item["logit_max_abs"] <= caps["exact_abs"]
            and item["score_max_abs"] <= caps["exact_abs"]
            and item["top1_min"] >= caps["top1_agreement"]
        )
        c0[variant] = item
        c0_pass = c0_pass and item["passed"]

    c1: dict[str, Any] = {}
    c1_pass = True
    for variant in p1225.SHAPE_VARIANTS:
        disc = [row["variants"][variant] for row in disc_ref]
        hold = [row["variants"][variant] for row in hold_ref]
        names = ["hidden_relative_rms", "logit_max_abs", "probability_max_abs"]
        if variant != "prompt_only":
            names += ["margin_drift_ratio", "score_drift_ratio"]
        metrics: dict[str, Any] = {}
        passed = True
        for name in names:
            disc_values = [float(row[name]) for row in disc]
            hold_values = [float(row[name]) for row in hold]
            limit = envelope(disc_values)
            observed = max(hold_values)
            gate = observed <= limit and observed <= float(caps[name])
            metrics[name] = {
                "discovery_max": max(disc_values), "holdout_max": observed,
                "envelope": limit, "functional_cap": float(caps[name]), "passed": gate,
            }
            passed = passed and gate
        top1 = min(float(row["top1_agreement"]) for row in hold)
        metrics["top1"] = {"holdout_min": top1, "passed": top1 >= caps["top1_agreement"]}
        passed = passed and metrics["top1"]["passed"]
        c1[variant] = {"metrics": metrics, "passed": passed}
        c1_pass = c1_pass and passed

    c2: dict[str, Any] = {}
    c2_pass = True
    for name in ("hidden_relative_rms", "logit_max_abs", "probability_max_abs", "margin_drift_ratio", "score_drift_ratio"):
        disc_values = [float(row["cross_load"][name]) for row in disc_reload]
        hold_values = [float(row["cross_load"][name]) for row in hold_reload]
        limit = envelope(disc_values)
        observed = max(hold_values)
        gate = observed <= limit and observed <= float(caps[name])
        c2[name] = {
            "discovery_max": max(disc_values), "holdout_max": observed,
            "envelope": limit, "functional_cap": float(caps[name]), "passed": gate,
        }
        c2_pass = c2_pass and gate
    top1 = min(float(row["cross_load"]["top1_agreement"]) for row in hold_reload)
    c2["top1"] = {"holdout_min": top1, "passed": top1 >= caps["top1_agreement"]}
    c2_pass = c2_pass and c2["top1"]["passed"]

    live = [row["conditions"]["live"] for row in reload]
    zero = [row["conditions"]["zero"] for row in reload]
    stored_disc = [row["conditions"]["stored"] for row in disc_reload]
    stored_hold = [row["conditions"]["stored"] for row in hold_reload]
    stored_limit = envelope([float(row["score_drift_ratio"]) for row in stored_disc])
    c3 = {
        "live_score_error_max": max(row["score_max_abs_vs_donor"] for row in live),
        "live_completion_median": median([row["completion"] for row in live]),
        "live_write_error_max": max(row["write_max_abs"] for row in live),
        "zero_score_error_max": max(row["score_max_abs_vs_recipient"] for row in zero),
        "zero_completion_abs_max": max(abs(row["completion"]) for row in zero),
        "zero_write_error_max": max(row["write_max_abs"] for row in zero),
        "stored_holdout_completion_median": median([row["completion"] for row in stored_hold]),
        "stored_holdout_positive_fraction": fraction([row["completion"] > 0 for row in stored_hold]),
        "stored_holdout_score_ratio_max": max(row["score_drift_ratio"] for row in stored_hold),
        "stored_score_ratio_envelope": stored_limit,
        "stored_holdout_top1_min": min(row["top1_agreement_vs_donor"] for row in stored_hold),
    }
    c3_pass = (
        c3["live_score_error_max"] <= caps["exact_abs"]
        and c3["live_completion_median"] >= 0.999
        and c3["live_write_error_max"] <= caps["exact_abs"]
        and c3["zero_score_error_max"] <= caps["exact_abs"]
        and c3["zero_completion_abs_max"] <= caps["exact_abs"]
        and c3["zero_write_error_max"] <= caps["exact_abs"]
        and c3["stored_holdout_completion_median"] >= caps["stored_completion_median"]
        and c3["stored_holdout_positive_fraction"] >= caps["stored_positive_fraction"]
        and c3["stored_holdout_score_ratio_max"] <= stored_limit
        and c3["stored_holdout_score_ratio_max"] <= caps["score_drift_ratio"]
        and c3["stored_holdout_top1_min"] >= caps["top1_agreement"]
    )
    c3["passed"] = c3_pass
    finite = all(
        row["reference"]["finite"] and all(value["finite"] for value in row["variants"].values())
        for row in reference
    ) and all(
        row["live_donor"]["finite"] and row["live_recipient"]["finite"]
        and all(value["bundle"]["finite"] for value in row["conditions"].values())
        for row in reload
    )
    contracts = {"C0": c0_pass, "C1": c1_pass, "C2": c2_pass, "C3": c3_pass}
    return {
        "finite": finite,
        "contracts": contracts,
        "details": {"C0": c0, "C1": c1, "C2": c2, "C3": c3},
        "passed": bool(finite and all(contracts.values())),
    }


def result_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    pre = read_json(p1225.PREAUDIT_PATH)
    protocol = read_json(p1225.PROTOCOL_PATH)
    manifest = read_jsonl(p1225.MANIFEST_PATH)
    reference = read_jsonl(p1225.REFERENCE_RECORD_PATH)
    reload = read_jsonl(p1225.RELOAD_RECORD_PATH)
    ref_summary = read_json(p1225.REFERENCE_SUMMARY_PATH)
    reload_summary = read_json(p1225.RELOAD_SUMMARY_PATH)
    final = read_json(p1225.FINAL_PATH)
    add(checks, "preaudit_passed", pre.get("all_checks_passed"), pre.get("audit_digest"))
    add(
        checks,
        "cardinality_and_ids",
        len(reference) == len(reload) == len(manifest) == 160
        and {row["pair_id"] for row in reference} == {row["pair_id"] for row in reload} == {row["pair_id"] for row in manifest},
        {"reference": len(reference), "reload": len(reload)},
    )
    add(checks, "reference_digest", ref_summary["record_digest"] == digest(reference), ref_summary["record_digest"])
    add(checks, "reload_digest", reload_summary["record_digest"] == digest(reload), reload_summary["record_digest"])
    add(
        checks,
        "row_digests",
        all(row["record_digest"] == digest({key: value for key, value in row.items() if key != "record_digest"}) for row in reference + reload),
        "all rows",
    )
    add(checks, "array_hash", ref_summary["array_sha256"] == file_sha256(p1225.REFERENCE_ARRAY_PATH), ref_summary["array_sha256"])
    with np.load(p1225.REFERENCE_ARRAY_PATH, allow_pickle=False) as arrays:
        array_ok = (
            arrays["hidden"].shape[:3] == (160, 4, p1225.MAX_CONTINUATION)
            and arrays["hidden"].dtype == np.float16
            and arrays["boundary_logits"].shape[0] == 160
            and arrays["boundary_logits"].dtype == np.float16
            and np.isfinite(arrays["hidden"]).all()
            and np.isfinite(arrays["boundary_logits"]).all()
        )
        array_detail = {"hidden": list(arrays["hidden"].shape), "boundary_logits": list(arrays["boundary_logits"].shape)}
    add(checks, "array_schema", array_ok, array_detail)
    precision_ok = all(
        summary["precision_audit"]["parameter_dtypes"] == {"float16": 4022468096}
        and not summary["precision_audit"]["has_bf16_parameters"]
        and not summary["precision_audit"]["has_quantized_modules"]
        and summary["attention_backend"] == "eager"
        for summary in (ref_summary, reload_summary)
    )
    add(checks, "two_load_precision_domain", precision_ok, {"reference": ref_summary["precision_audit"], "reload": reload_summary["precision_audit"]})

    formula_ok = True
    for row in reference:
        for variant, metrics in row["variants"].items():
            formula_ok = formula_ok and 0.0 <= metrics["top1_agreement"] <= 1.0
            if variant != "prompt_only":
                formula_ok = formula_ok and math.isclose(
                    metrics["score_drift_ratio"], metrics["score_max_abs"] / row["target_shift_abs"], rel_tol=1e-9, abs_tol=1e-12
                )
    for row in reload:
        donor = row["live_donor"]["fixed_margin"]
        recipient = row["live_recipient"]["fixed_margin"]
        target = donor - recipient
        for condition in row["conditions"].values():
            expected = (condition["bundle"]["fixed_margin"] - recipient) / target
            formula_ok = formula_ok and math.isclose(condition["completion"], expected, rel_tol=1e-9, abs_tol=1e-9)
    add(checks, "formula_recomputation", formula_ok, formula_ok)

    recomputed = independent_evaluate(reference, reload)
    add(checks, "independent_contract_recomputation", final["result"] == recomputed, recomputed["contracts"])
    final_payload = {key: value for key, value in final.items() if key != "final_digest"}
    add(checks, "final_digest", final["final_digest"] == digest(final_payload), final["final_digest"])
    status_ok = final["status"] == ("numerical_domain_confirmed" if recomputed["passed"] else "numerical_domain_not_confirmed")
    add(checks, "status", status_ok, final["status"])
    action_ok = (
        final["authorization"]["automatic_execution"] == recomputed["passed"]
        and not final["authorization"]["qwen_mechanism_scan"]
        and (final["authorization"]["next_experiment"] is not None) == recomputed["passed"]
    )
    add(checks, "action_scope", action_ok, final["authorization"])
    claim_ok = final["k_item"]["identifier"] == "K202" and final["new_mathematics_required"] is False
    add(checks, "claim_scope", claim_ok, final["k_item"])
    result = audit_document("result", checks)
    write_json(p1225.RESULT_AUDIT_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "result"))
    args = parser.parse_args()
    result = preaudit() if args.stage == "pre" else result_audit()
    print(canonical_json({
        "stage": result["stage"],
        "passed": result["all_checks_passed"],
        "checks": f"{result['passed_count']}/{result['check_count']}",
        "audit_digest": result["audit_digest"],
    }))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
