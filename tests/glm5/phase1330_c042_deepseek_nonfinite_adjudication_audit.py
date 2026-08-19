#!/usr/bin/env python3
"""Independent final audit of Phase1330 including the DeepSeek numerical failure."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PARENT = T / "result/phase1329_c042_relational_ecology_contract"
OUT = T / "result/phase1330_c042_sequential_behavior"


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


def recompute(model: str, protocol) -> dict:
    raw = rows(OUT / f"raw/{model}_scores.jsonl")
    finite = sum(all(math.isfinite(v) for v in row["candidate_scores"]) for row in raw) / len(raw)
    accuracy = sum(row["correct"] for row in raw) / len(raw)
    partitions = {name: sum(row["correct"] for row in raw if row["partition"] == name)
                        / sum(row["partition"] == name for row in raw) for name in ("discovery", "confirmation", "holdout")}
    surfaces = {name: sum(row["correct"] for row in raw if row["surface"] == name)
                      / sum(row["surface"] == name for row in raw) for name in ("reference_family", "vocabulary_kind")}
    groups = defaultdict(list)
    for row in raw:
        groups[(row["semantic_set"], row["surface"])].append(row["correct"])
    pair = sum(len(values) == 2 and all(values) for values in groups.values()) / len(groups)
    margin = sum(row["margin"] for row in raw) / len(raw)
    th = protocol["behavior_gate"]
    gates = {"finite_fraction": finite >= th["finite_fraction_min"],
             "candidate_accuracy": accuracy >= th["candidate_accuracy_min"],
             "partition_accuracy": min(partitions.values()) >= th["partition_accuracy_min"],
             "surface_accuracy": min(surfaces.values()) >= th["surface_accuracy_min"],
             "order_pair_success": pair >= th["order_pair_success_min"],
             "mean_correct_margin": margin >= th["mean_correct_margin_min"]}
    return {"finite_fraction": finite, "candidate_accuracy": accuracy, "partition_accuracy": partitions,
            "surface_accuracy": surfaces, "order_pair_success": pair, "mean_correct_margin": margin,
            "gates": gates, "qualified": all(gates.values()), "raw_count": len(raw)}


def run() -> None:
    protocol = load(PARENT / "protocol/preregistration.json")
    pre = load(OUT / "audit/independent_preaudit.json")
    final = load(OUT / "analysis/final.json")
    failure = load(OUT / "runtime/deepseek7b_failure.json")
    qwen, glm = recompute("qwen3", protocol), recompute("glm4", protocol)
    checks = {
        "preaudit": pre["all_checks_passed"] is True,
        "qwen_recomputed": qwen["raw_count"] == 576 and qwen["qualified"] is True,
        "glm_recomputed": glm["raw_count"] == 576 and glm["qualified"] is False,
        "normal_runtime_fp16": all(not load(OUT / f"runtime/{model}.json")["quantization_audit"]["has_quantized_modules"]
                                    and load(OUT / f"runtime/{model}.json")["quantization_audit"]["has_fp16_parameters"]
                                    for model in ("qwen3", "glm4")),
        "deepseek_empty_raw": (OUT / "raw/deepseek7b_scores.jsonl").exists()
                              and (OUT / "raw/deepseek7b_scores.jsonl").stat().st_size == 0,
        "deepseek_nonfinite_gate": failure["qualified"] is False and failure["rerun_authorized"] is False
                                   and failure["finite_fraction_upper_bound"] < 1.0,
        "adjudication_hashes": sha(T / "phase1330_c042_deepseek_nonfinite_adjudication.py") == failure["adjudicator_sha256"]
                               and sha(Path(__file__).resolve()) == failure["auditor_sha256"],
        "frozen_order": final["model_order"] == ["qwen3", "glm4", "deepseek7b"],
        "qualified_count": final["qualified_models"] == ["qwen3"] and final["qualified_model_count"] == 1,
        "formal_stop": final["all_gates_passed"] is False and final["authorization"] == "close_c042_before_hidden_states",
        "no_hidden_outputs": not (OUT / "field").exists() and not (OUT / "hidden").exists(),
    }
    output = {"phase": 1330, "campaign": "C042", "audit_type": "final_with_numerical_failure",
              "checks": checks, "recomputed": {"qwen3": qwen, "glm4": glm,
                                                "deepseek7b": {"finite_fraction_upper_bound": failure["finite_fraction_upper_bound"],
                                                               "qualified": False}},
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "close_c042_before_hidden_states" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_final_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
