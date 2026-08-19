#!/usr/bin/env python3
"""Independent pre/post audit for Phase1338."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PARENT = T / "result/phase1337_c046_polarity_deconfounded_relation_contract"
OUT = T / "result/phase1338_c046_deconfounded_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def by(records, key, values):
    return {str(value): sum(row["correct"] for row in records if row[key] == value)
            / sum(row[key] == value for row in records) for value in values}


def recompute(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["semantic_key"]].append(row["correct"])
    return {
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "partition": by(records, "partition", ("discovery", "confirmation", "holdout")),
        "surface": by(records, "surface", ("noun_class", "dictionary_relation", "category_claim")),
        "family": by(records, "target_family", ("mammal", "gemstone", "vehicle", "vegetable")),
        "codebook": by(records, "codebook", ("standard", "reversed")),
        "truth": by(records, "truth", (True, False)),
        "truth_codebook": {f"{truth}:{codebook}": sum(row["correct"] for row in records
            if row["truth"] == truth and row["codebook"] == codebook) /
            sum(row["truth"] == truth and row["codebook"] == codebook for row in records)
            for truth in (True, False) for codebook in ("standard", "reversed")},
        "semantic_pair_success": sum(len(values) == 2 and all(values) for values in groups.values()) / len(groups),
        "median_margin": median(row["margin"] for row in records), "case_count": len(records),
    }


def close(left, right, tolerance=1e-10):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=tolerance)
    return left == right


def preaudit():
    parent = load(PARENT / "analysis/final.json")
    pa = load(PARENT / "audit/independent_final_audit.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    checks = {
        "parent": parent.get("authorization") == "run_phase1338_c046_deconfounded_behavior" and pa.get("all_checks_passed"),
        "hash": digest(frozen) == manifest["manifest_sha256"],
        "order": manifest["model_order"] == list(MODELS),
        "batch": manifest["batch_size"] == 8 and manifest["padding_side"] == "right" and manifest["explicit_position_ids"],
        "groups": all(len(manifest["executor_groups"][model]["cohort_a"]) == 6
                      and len(manifest["executor_groups"][model]["cohort_permuted"]) == 6 for model in MODELS),
        "no_results": not any((OUT / f"analysis/{model}_summary.json").exists() for model in MODELS),
    }
    result = {"phase": 1338, "stage": "pre_model", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "run_models_in_frozen_order" if all(checks.values()) else "deny_models"}
    path = OUT / "audit/independent_preaudit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def postaudit():
    protocol = load(PARENT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    checks = {}
    independently_qualified = []
    for model in MODELS:
        summary = load(OUT / f"analysis/{model}_summary.json")
        executor = rows(OUT / f"raw/{model}_executor.jsonl")
        behavior = rows(OUT / f"raw/{model}_behavior.jsonl")
        scores = [value for row in executor for key in ("cohort_a", "cohort_permuted", "cohort_a_repeat") for value in row[key]]
        perm = max(abs(a - b) for row in executor for a, b in zip(row["cohort_a"], row["cohort_permuted"]))
        repeat = max(abs(a - b) for row in executor for a, b in zip(row["cohort_a"], row["cohort_a_repeat"]))
        rank = sum((row["cohort_a"][0] > row["cohort_a"][1]) ==
                   (row["cohort_permuted"][0] > row["cohort_permuted"][1]) for row in executor) / len(executor)
        em = {"finite_fraction": sum(math.isfinite(value) for value in scores) / len(scores),
              "permuted_rank_agreement": rank, "permuted_max_abs_score_diff": perm,
              "repeat_max_abs_score_diff": repeat, "case_count": len(executor)}
        bm = recompute(behavior)
        bg = protocol["behavior_gate"]
        qualified = (em["finite_fraction"] == 1 and rank == 1 and perm <= 1e-6 and repeat <= 1e-6
                     and bm["accuracy"] >= bg["accuracy_min"]
                     and min(bm["partition"].values()) >= bg["partition_min"]
                     and min(bm["surface"].values()) >= bg["surface_min"]
                     and min(bm["family"].values()) >= bg["family_min"]
                     and min(bm["codebook"].values()) >= bg["codebook_min"]
                     and min(bm["truth"].values()) >= bg["truth_min"]
                     and min(bm["truth_codebook"].values()) >= bg["truth_codebook_cell_min"]
                     and bm["semantic_pair_success"] >= bg["semantic_pair_success_min"]
                     and bm["median_margin"] >= bg["median_margin_min"])
        if qualified:
            independently_qualified.append(model)
        checks[f"{model}_executor"] = close(em, summary["executor_metrics"])
        checks[f"{model}_behavior_count"] = len(behavior) == 1152
        checks[f"{model}_behavior_metrics"] = close(bm, summary["behavior_metrics"])
        checks[f"{model}_qualified"] = summary["qualified"] == qualified
        checks[f"{model}_hashes"] = (sha(OUT / f"raw/{model}_executor.jsonl") == summary["executor_raw_sha256"]
                                      and sha(OUT / f"raw/{model}_behavior.jsonl") == summary["behavior_raw_sha256"]
                                      and sha(OUT / f"runtime/{model}.json") == summary["runtime_sha256"])
    expected_branch = "run_phase1339_c046_full_relation_field" if len(independently_qualified) >= 2 else "close_c046_behavior"
    checks["final_models"] = final["qualified_models"] == independently_qualified
    checks["final_branch"] = final["authorization"] == expected_branch
    result = {"phase": 1338, "campaign": "C046", "checks": checks,
              "independently_qualified_models": independently_qualified,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": expected_branch if all(checks.values()) else "deny_phase1339"}
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre", "post"), required=True)
    args = parser.parse_args()
    preaudit() if args.stage == "pre" else postaudit()
