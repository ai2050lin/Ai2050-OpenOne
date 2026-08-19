#!/usr/bin/env python3
"""Independent audit for Phase1337 C046 contract."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1337_c046_polarity_deconfounded_relation_contract"
PARENT = T / "result/phase1336_c045_standard_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    import hashlib
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def main() -> None:
    protocol = load(OUT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    material = rows(OUT / "material/frozen_behavior_cases.jsonl")
    graph = load(OUT / "material/frozen_concept_graph.json")["concepts"]
    material_audit = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    timeless = {key: value for key, value in protocol.items() if key != "contract_sha256" and key != "authorization"}
    groups = defaultdict(list)
    for row in material:
        groups[row["semantic_key"]].append(row)
    checks = {
        "parent": load(PARENT / "analysis/final.json").get("authorization") == "close_c045_standard_behavior"
                  and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"),
        "contract_hash": digest(timeless) == protocol["contract_sha256"],
        "final": final.get("authorization") == "run_phase1338_c046_deconfounded_behavior"
                 and final.get("contract_sha256") == protocol["contract_sha256"],
        "material_hash": protocol["material"]["case_count"] == len(material) == 1152,
        "graph": len(graph) == 48 and len({row["word"] for row in graph}) == 48,
        "partitions": Counter(row["partition"] for row in material) == {"discovery": 384, "confirmation": 384, "holdout": 384},
        "surfaces": all(value == 384 for value in Counter(row["surface"] for row in material).values()),
        "codebooks": Counter(row["codebook"] for row in material) == {"standard": 576, "reversed": 576},
        "truth": Counter(row["truth"] for row in material) == {True: 288, False: 864},
        "gold": Counter(row["gold_value"] for row in material) == {"yes": 576, "no": 576},
        "semantic_pairs": len(groups) == 576 and all(len(group) == 2
                           and {row["codebook"] for row in group} == {"standard", "reversed"}
                           and {row["gold_value"] for row in group} == {"yes", "no"} for group in groups.values()),
        "truth_definition": all(row["truth"] == (row["target_family"] == row["tested_family"]) for row in material),
        "zero_models": protocol["zero_models"] == {"always_yes": .5, "always_no": .5,
                         "semantic_truth_ignore_codebook": .5, "codebook_assume_true": .25,
                         "codebook_assume_false": .75},
        "material_audit": material_audit.get("all_checks_passed") and material_audit.get("passed") == material_audit.get("total"),
        "behavior_before_hidden": protocol["behavior_gate"]["minimum_authorized_models"] == 2
                                  and protocol["hidden_numeric_gate"]["minimum_authorized_models"] == 2,
        "stop": "do not change" in protocol["stop_rule"].lower()
                and protocol["parameter_boundary"].startswith("No natural-model"),
    }
    for model in MODELS:
        compiled = rows(OUT / f"compiled/{model}_behavior.jsonl")
        checks[f"{model}_compiled"] = len(compiled) == len(material) and all(
            left["case_id"] == right["case_id"] for left, right in zip(material, compiled))
        checks[f"{model}_tokens"] = all(all(len(value) == 1 for value in row["candidate_ids"]) for row in compiled)
        checks[f"{model}_spans"] = all(row["target_span"] and row["tested_family_span"]
                                       and max(row["target_span"] + row["tested_family_span"]) < row["boundary_position"]
                                       for row in compiled)
    result = {"phase": 1337, "campaign": "C046", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "run_phase1338_c046_deconfounded_behavior" if all(checks.values()) else "deny_phase1338"}
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
