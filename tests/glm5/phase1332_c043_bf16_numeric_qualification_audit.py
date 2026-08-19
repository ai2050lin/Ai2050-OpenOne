#!/usr/bin/env python3
"""Independent pre/post audit for Phase1332."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PARENT = T / "result/phase1331_c043_native_relational_contract"
OUT = T / "result/phase1332_c043_bf16_numeric_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preaudit() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    parent = load(PARENT / "protocol/preregistration.json")
    parent_audit = load(PARENT / "audit/independent_final_audit.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    checks = {
        "parent": parent["authorization"] == "run_phase1332_bf16_numeric_qualification" and parent_audit["all_checks_passed"],
        "manifest_hash": hashlib.sha256(canonical(frozen).encode()).hexdigest() == manifest["manifest_sha256"],
        "source_hashes": (
            sha(T / "phase1332_c043_bf16_numeric_qualification.py") == manifest["script_sha256"]
            and sha(Path(__file__).resolve()) == manifest["auditor_sha256"]
            and sha(T / "phase1332_bf16_utils.py") == manifest["util_sha256"]
        ),
        "parent_hash": sha(PARENT / "protocol/preregistration.json") == manifest["parent_protocol_sha256"],
        "model_order": manifest["model_order"] == list(MODELS),
        "bf16": manifest["precision"] == "bfloat16-no-quantization",
        "runs": manifest["runs"] == [
            {"name": "single", "batch_size": 1},
            {"name": "batch", "batch_size": 8},
            {"name": "batch_repeat", "batch_size": 8},
        ],
        "sentinels": manifest["sentinel_case_ids"] == parent["numeric"]["sentinel_case_ids"],
        "gate": manifest["gate"] == parent["numeric"]["gate"],
        "no_results": not any((OUT / f"raw/{model}_scores.jsonl").exists() for model in MODELS),
    }
    result = {
        "phase": 1332,
        "stage": "pre_model",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "run_models_in_frozen_order" if all(checks.values()) else "none",
    }
    save(OUT / "audit/independent_preaudit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def postaudit() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    final = load(OUT / "analysis/final.json")
    thresholds = manifest["gate"]
    checks: dict[str, bool] = {}
    qualified_models = []
    independent_metrics = {}
    for model in MODELS:
        raw = rows(OUT / f"raw/{model}_scores.jsonl")
        summary = load(OUT / f"analysis/{model}_summary.json")
        runtime = load(OUT / f"runtime/{model}.json")
        values = [value for row in raw for key in ("single_scores", "batch_scores", "batch_repeat_scores") for value in row[key]]
        finite = sum(math.isfinite(value) for value in values) / len(values)
        rank = sum((row["single_scores"][0] > row["single_scores"][1]) == (row["batch_scores"][0] > row["batch_scores"][1]) for row in raw) / len(raw)
        batch_diff = max(abs(a - b) for row in raw for a, b in zip(row["single_scores"], row["batch_scores"]))
        repeat_diff = max(abs(a - b) for row in raw for a, b in zip(row["batch_scores"], row["batch_repeat_scores"]))
        metrics = {
            "finite_fraction": finite,
            "batch_rank_agreement": rank,
            "batch_max_abs_score_diff": batch_diff,
            "repeat_max_abs_score_diff": repeat_diff,
            "sentinel_case_count": len(raw),
        }
        gates = {
            "finite_fraction": finite >= thresholds["finite_fraction_min"],
            "batch_rank_agreement": rank >= thresholds["batch_rank_agreement_min"],
            "batch_max_abs_score_diff": batch_diff <= thresholds["batch_max_abs_score_diff_max"],
            "repeat_max_abs_score_diff": repeat_diff <= thresholds["repeat_max_abs_score_diff_max"],
            "sentinel_case_count": len(raw) == thresholds["sentinel_case_count"],
        }
        qualified = all(gates.values())
        if qualified:
            qualified_models.append(model)
        independent_metrics[model] = {"metrics": metrics, "gates": gates, "qualified": qualified}
        checks[f"{model}_raw"] = len(raw) == thresholds["sentinel_case_count"] and len({row["case_id"] for row in raw}) == len(raw)
        checks[f"{model}_summary"] = summary["metrics"] == metrics and summary["gates"] == gates and summary["qualified"] == qualified
        qa = runtime["quantization_audit"]
        checks[f"{model}_runtime"] = qa["has_bf16_parameters"] and not qa["has_quantized_modules"]
        checks[f"{model}_hashes"] = (
            sha(OUT / f"raw/{model}_scores.jsonl") == summary["raw_sha256"]
            and sha(OUT / f"analysis/{model}_summary.json") == final["model_summary_sha256"][model]
        )
    passed = len(qualified_models) >= manifest["minimum_models_for_behavior"]
    checks["final_models"] = final["qualified_models"] == qualified_models
    checks["final_branch"] = (
        final["all_gates_passed"] == passed
        and final["authorization"] == ("run_phase1333_bf16_behavior" if passed else "close_c043_numeric_ineligible")
    )
    result = {
        "phase": 1332,
        "campaign": "C043",
        "stage": "post_model",
        "checks": checks,
        "independent_metrics": independent_metrics,
        "independently_qualified_models": qualified_models,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": final["authorization"] if all(checks.values()) else "none",
    }
    save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "post"))
    args = parser.parse_args()
    preaudit() if args.stage == "pre" else postaudit()
