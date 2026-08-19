#!/usr/bin/env python3
"""Independent audit of Phase1333 C044 preregistration and material."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402

OUT = T / "result/phase1333_c044_relational_measurement_contract"
PARENT = T / "result/phase1332_c043_bf16_numeric_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("bird", "garment", "musical_instrument", "furniture")


def audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    parent = core.load(PARENT / "analysis/final.json")
    material_audit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    graph = core.load(OUT / "material/frozen_concept_graph.json")["concepts"]
    behavior = core.rows(OUT / "material/frozen_behavior_cases.jsonl")
    contexts = core.rows(OUT / "material/frozen_context_cases.jsonl")
    binary = [row for row in behavior if row["interface"] == "binary"]
    choice = [row for row in behavior if row["interface"] == "choice"]
    generation = [row for row in behavior if row["interface"] == "generation"]
    frozen = {key: value for key, value in protocol.items() if key not in {"contract_sha256", "authorization"}}
    previous = set()
    for path in (
        T / "result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json",
        T / "result/phase1331_c043_native_relational_contract/material/frozen_concept_graph.json",
    ):
        previous.update(row["word"] for row in core.load(path)["concepts"])
    checks = {
        "parent_terminal": parent["authorization"] == "close_c043_numeric_ineligible",
        "contract_hash": core.digest(frozen) == protocol["contract_sha256"],
        "source_hashes": core.sha(T / "phase1333_c044_relational_measurement_contract.py") == protocol["script_sha256"]
                         and core.sha(Path(__file__).resolve()) == protocol["auditor_sha256"],
        "material_hashes": (
            core.sha(OUT / "material/frozen_concept_graph.json") == protocol["material"]["graph_sha256"]
            and core.sha(OUT / "material/frozen_behavior_cases.jsonl") == protocol["material"]["behavior_sha256"]
            and core.sha(OUT / "material/frozen_context_cases.jsonl") == protocol["material"]["context_sha256"]
        ),
        "fresh": not ({row["word"] for row in graph} & previous),
        "graph": len(graph) == 48 and all(
            sum(row["partition"] == partition and row["family"] == family for row in graph) == 4
            for partition in PARTITIONS for family in FAMILIES
        ),
        "counts": len(binary) == 288 and len(choice) == 96 and len(generation) == 96 and len(contexts) == 144,
        "binary_balance": Counter(row["gold_value"] for row in binary) == {"yes": 144, "no": 144},
        "choice_balance": Counter(row["gold_position"] for row in choice) == {0: 24, 1: 24, 2: 24, 3: 24},
        "generation_balance": Counter(row["gold_value"] for row in generation) == {
            "bird": 24, "garment": 24, "musical instrument": 24, "furniture": 24,
        },
        "material_audit": material_audit["all_checks_passed"],
        "numeric_cases": len(protocol["numeric"]["case_ids"]) == 48 and len(set(protocol["numeric"]["case_ids"])) == 48,
        "hidden_after_behavior": protocol["numeric"]["failure"].startswith("fewer than two models closes C044 before behavior and hidden"),
        "multi_interface": set(row["interface"] for row in behavior) == {"binary", "choice", "generation"},
        "parameter_boundary": protocol["parameter_boundary"].startswith("No natural-model single-parameter"),
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1334_c044_numeric_factorial",
    }
    for model in MODELS:
        compiled_b = core.rows(OUT / f"compiled/{model}_behavior.jsonl")
        compiled_h = core.rows(OUT / f"compiled/{model}_context.jsonl")
        checks[f"{model}_compiled"] = len(compiled_b) == 480 and len(compiled_h) == 144
        checks[f"{model}_binary_single_token"] = all(
            all(len(candidate) == 1 for candidate in row["candidate_ids"])
            for row in compiled_b if row["interface"] == "binary"
        )
        checks[f"{model}_spans"] = all(
            row["target_span"] and max(row["target_span"]) < row["boundary_position"] < len(row["prompt_ids"])
            for row in compiled_h
        )
    result = {"phase": 1333, "campaign": "C044", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": protocol["authorization"] if all(checks.values()) else "none"}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
