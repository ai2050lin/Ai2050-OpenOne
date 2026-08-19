#!/usr/bin/env python3
"""Independent pre/post audit for Phase1330."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PARENT = T / "result/phase1329_c042_relational_ecology_contract"
OUT = T / "result/phase1330_c042_sequential_behavior"
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
    checks = {"parent": parent["authorization"] == "run_phase1330_sequential_behavior" and parent_audit["all_checks_passed"],
              "manifest_hash": hashlib.sha256(canonical(frozen).encode()).hexdigest() == manifest["manifest_sha256"],
              "source_hashes": sha(T / "phase1330_c042_sequential_behavior.py") == manifest["script_sha256"]
                               and sha(Path(__file__).resolve()) == manifest["auditor_sha256"],
              "parent_hash": sha(PARENT / "protocol/preregistration.json") == manifest["parent_protocol_sha256"],
              "order": manifest["model_order"] == list(MODELS), "fp16": manifest["precision"] == "fp16-no-quantization",
              "score": manifest["score"].startswith("mean log probability"),
              "no_results": not any((OUT / f"raw/{model}_scores.jsonl").exists() for model in MODELS)}
    result = {"phase": 1330, "stage": "pre_model", "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()), "authorization": "run_models_in_frozen_order" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_preaudit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def postaudit() -> None:
    protocol = load(PARENT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    checks: dict[str, bool] = {}
    independently_qualified = []
    metrics = {}
    for model in MODELS:
        source = rows(PARENT / "material/frozen_behavior_cases.jsonl")
        raw = rows(OUT / f"raw/{model}_scores.jsonl")
        summary = load(OUT / f"analysis/{model}_summary.json")
        runtime = load(OUT / f"runtime/{model}.json")
        finite = sum(all(math.isfinite(v) for v in row["candidate_scores"]) for row in raw) / len(raw)
        accuracy = sum(row["correct"] for row in raw) / len(raw)
        partitions = {name: sum(row["correct"] for row in raw if row["partition"] == name)
                            / sum(row["partition"] == name for row in raw) for name in ("discovery", "confirmation", "holdout")}
        surfaces = {name: sum(row["correct"] for row in raw if row["surface"] == name)
                          / sum(row["surface"] == name for row in raw) for name in ("reference_family", "vocabulary_kind")}
        groups = defaultdict(list)
        for row in raw:
            groups[(row["semantic_set"], row["surface"])].append(row["correct"])
        pair_success = sum(len(values) == 2 and all(values) for values in groups.values()) / len(groups)
        mean_margin = sum(row["margin"] for row in raw) / len(raw)
        gates = {"finite_fraction": finite >= protocol["behavior_gate"]["finite_fraction_min"],
                 "candidate_accuracy": accuracy >= protocol["behavior_gate"]["candidate_accuracy_min"],
                 "partition_accuracy": min(partitions.values()) >= protocol["behavior_gate"]["partition_accuracy_min"],
                 "surface_accuracy": min(surfaces.values()) >= protocol["behavior_gate"]["surface_accuracy_min"],
                 "order_pair_success": pair_success >= protocol["behavior_gate"]["order_pair_success_min"],
                 "mean_correct_margin": mean_margin >= protocol["behavior_gate"]["mean_correct_margin_min"]}
        qualified = all(gates.values())
        if qualified:
            independently_qualified.append(model)
        metrics[model] = {"finite_fraction": finite, "candidate_accuracy": accuracy,
                          "partition_accuracy": partitions, "surface_accuracy": surfaces,
                          "order_pair_success": pair_success, "mean_correct_margin": mean_margin, "gates": gates}
        checks[f"{model}_counts"] = len(source) == len(raw) == 576 and len(groups) == 288
        checks[f"{model}_summary"] = (
            summary["qualified"] == qualified and summary["gates"] == gates
            and abs(summary["metrics"]["candidate_accuracy"] - accuracy) < 1e-12
            and abs(summary["metrics"]["mean_correct_margin"] - mean_margin) < 1e-12)
        checks[f"{model}_runtime"] = (not runtime["quantization_audit"]["has_quantized_modules"]
                                      and runtime["quantization_audit"]["has_fp16_parameters"])
        checks[f"{model}_hashes"] = sha(OUT / f"analysis/{model}_summary.json") == final["model_summary_sha256"][model]
    pass_gate = len(independently_qualified) >= protocol["behavior_gate"]["minimum_authorized_models"]
    checks["final_qualification"] = final["qualified_models"] == independently_qualified
    checks["final_branch"] = final["all_gates_passed"] == pass_gate and final["authorization"] == (
        "run_phase1331_relation_kernels" if pass_gate else "close_c042_before_hidden_states")
    output = {"phase": 1330, "campaign": "C042", "stage": "post_model", "checks": checks, "metrics": metrics,
              "independently_qualified_models": independently_qualified, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()), "authorization": final["authorization"] if all(checks.values()) else "none"}
    save(OUT / "audit/independent_final_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "post"))
    args = parser.parse_args()
    preaudit() if args.stage == "pre" else postaudit()
