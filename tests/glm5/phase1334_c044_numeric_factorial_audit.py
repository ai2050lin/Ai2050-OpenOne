#!/usr/bin/env python3
"""Independent pre/post audit for Phase1334."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402

PARENT = T / "result/phase1333_c044_relational_measurement_contract"
OUT = T / "result/phase1334_c044_numeric_factorial"
MODELS = ("qwen3", "glm4", "deepseek7b")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def comparison(rows, left_key, right_key):
    absolute, common, margin_abs, margin_norm, ranks = [], [], [], [], []
    for row in rows:
        left, right = row[left_key], row[right_key]
        delta = [right[index] - left[index] for index in range(2)]
        absolute.extend(abs(value) for value in delta)
        common.append(abs(sum(delta) / 2))
        ml, mr = left[0] - left[1], right[0] - right[1]
        drift = abs(mr - ml)
        margin_abs.append(drift); margin_norm.append(drift / (1 + abs(ml)))
        ranks.append((left[0] > left[1]) == (right[0] > right[1]))
    return {"rank_agreement": sum(ranks) / len(ranks), "absolute_score_drift_max": max(absolute),
            "common_drift_p95": quantile(common, .95), "absolute_margin_drift_p95": quantile(margin_abs, .95),
            "normalized_margin_drift_p95": quantile(margin_norm, .95),
            "normalized_margin_drift_max": max(margin_norm)}


def preaudit() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    parent = core.load(PARENT / "protocol/preregistration.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    checks = {"parent": parent["authorization"] == "run_phase1334_c044_numeric_factorial" and parent_audit["all_checks_passed"],
              "manifest_hash": core.digest(frozen) == manifest["manifest_sha256"],
              "source_hashes": core.sha(T / "phase1334_c044_numeric_factorial.py") == manifest["script_sha256"]
                               and core.sha(Path(__file__).resolve()) == manifest["auditor_sha256"]
                               and core.sha(T / "phase1332_bf16_utils.py") == manifest["util_sha256"],
              "parent_hash": core.sha(PARENT / "protocol/preregistration.json") == manifest["parent_protocol_sha256"],
              "order": manifest["model_order"] == list(MODELS), "cases": len(manifest["case_ids"]) == 48,
              "cohorts": all(len(values) == 6 and all(len(group) == 8 for group in values)
                             for values in manifest["cohorts_by_model"].values()),
              "position_ids": manifest["explicit_position_ids"] and manifest["padding_side"] == "right",
              "no_results": not any((OUT / f"raw/{model}_factorial.jsonl").exists() for model in MODELS)}
    result = {"phase": 1334, "stage": "pre_model", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "run_models_in_frozen_order" if all(checks.values()) else "none"}
    core.save(OUT / "audit/independent_preaudit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


def postaudit() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    final = core.load(OUT / "analysis/final.json")
    threshold = manifest["gate"]
    checks = {}; qualified = []; metrics_all = {}
    for model in MODELS:
        rows = core.rows(OUT / f"raw/{model}_factorial.jsonl")
        summary = core.load(OUT / f"analysis/{model}_summary.json")
        runtime = core.load(OUT / f"runtime/{model}.json")
        scores = [value for row in rows for key in manifest["conditions"] for value in row[key]]
        shape = comparison(rows, "solo_fixed_width", "replicated_batch8")
        composition = comparison(rows, "replicated_batch8", "cohort_batch8")
        repeat = max(abs(a-b) for row in rows for a,b in zip(row["cohort_batch8"],row["cohort_batch8_repeat"]))
        metrics = {"finite_fraction": sum(math.isfinite(value) for value in scores)/len(scores),
                   "shape": shape, "composition": composition, "repeat_max_abs_score_diff": repeat,
                   "case_count": len(rows)}
        gates = {"finite_fraction": metrics["finite_fraction"] >= threshold["finite_fraction_min"],
                 "shape_rank_agreement": shape["rank_agreement"] >= threshold["shape_rank_agreement_min"],
                 "composition_rank_agreement": composition["rank_agreement"] >= threshold["composition_rank_agreement_min"],
                 "shape_margin_p95": shape["normalized_margin_drift_p95"] <= threshold["shape_normalized_margin_drift_p95_max"],
                 "composition_margin_p95": composition["normalized_margin_drift_p95"] <= threshold["composition_normalized_margin_drift_p95_max"],
                 "shape_margin_max": shape["normalized_margin_drift_max"] <= threshold["shape_normalized_margin_drift_max"],
                 "composition_margin_max": composition["normalized_margin_drift_max"] <= threshold["composition_normalized_margin_drift_max"],
                 "repeat": repeat <= threshold["repeat_max_abs_score_diff_max"], "case_count": len(rows)==48}
        q = all(gates.values())
        if q: qualified.append(model)
        metrics_all[model] = {"metrics":metrics,"gates":gates,"qualified":q}
        checks[f"{model}_raw"] = len(rows)==48 and len({row["case_id"] for row in rows})==48
        checks[f"{model}_summary"] = summary["metrics"]==metrics and summary["gates"]==gates and summary["qualified"]==q
        qa=runtime["quantization_audit"]
        checks[f"{model}_runtime"] = qa["has_bf16_parameters"] and not qa["has_quantized_modules"]
        checks[f"{model}_hash"] = core.sha(OUT/f"analysis/{model}_summary.json")==final["model_summary_sha256"][model]
    passed=len(qualified)>=threshold["minimum_authorized_models"]
    checks["final_models"]=final["qualified_models"]==qualified
    checks["final_branch"]=(final["all_gates_passed"]==passed and final["authorization"]==
                             ("run_phase1335_c044_multi_interface_behavior" if passed else "close_c044_numeric_factorial"))
    result={"phase":1334,"campaign":"C044","checks":checks,"independent_metrics":metrics_all,
            "independently_qualified_models":qualified,"passed":sum(checks.values()),"total":len(checks),
            "all_checks_passed":all(checks.values()),"authorization":final["authorization"] if all(checks.values()) else "none"}
    core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("pre","post"),required=True);args=parser.parse_args()
    preaudit() if args.stage=="pre" else postaudit()
