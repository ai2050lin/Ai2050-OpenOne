#!/usr/bin/env python3
"""Independent audit for Phase1331 C043."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1331_c043_native_relational_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


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


def run() -> None:
    protocol, final = load(OUT / "protocol/preregistration.json"), load(OUT / "analysis/final.json")
    machine = load(OUT / "audit/tokenizer_semantic_zero_model_audit.json")
    graph = load(OUT / "material/frozen_concept_graph.json")["concepts"]
    behavior, context = rows(OUT / "material/frozen_behavior_cases.jsonl"), rows(OUT / "material/frozen_context_cases.jsonl")
    naturalness = load(OUT / "material/pre_model_semantic_naturalness_review.json")
    parent = load(T / "result/phase1330_c042_sequential_behavior/audit/independent_final_audit.json")
    pairs, label_truth, target_truth, surface_truth = defaultdict(list), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in behavior:
        pairs[row["pair_key"]].append(row)
        label_truth[row["tested_family"]][row["gold_value"]] += 1
        target_truth[row["target"]][row["gold_value"]] += 1
        surface_truth[row["surface"]][row["gold_value"]] += 1
    timeless = {key: value for key, value in protocol.items()
                if key not in {"contract_sha256", "script_sha256", "auditor_sha256", "core_sha256", "created_at_utc"}}
    compiled_ok = all(len(rows(OUT / f"compiled/{model}_behavior.jsonl")) == 288
                      and len(rows(OUT / f"compiled/{model}_context.jsonl")) == 144 for model in MODELS)
    previous = load(T / "result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json")["concepts"]
    checks = {
        "parent_terminal": parent["all_checks_passed"] is True and parent["authorization"] == "close_c042_before_hidden_states",
        "contract_hash": digest(timeless) == protocol["contract_sha256"],
        "source_hashes": sha(T / "phase1331_c043_native_relational_contract.py") == protocol["script_sha256"]
                         and sha(Path(__file__).resolve()) == protocol["auditor_sha256"]
                         and sha(T / "phase1331_relational_measurement_core.py") == protocol["core_sha256"],
        "material_hashes": sha(OUT / "material/frozen_concept_graph.json") == protocol["material"]["graph_sha256"]
                           and sha(OUT / "material/frozen_behavior_cases.jsonl") == protocol["material"]["behavior_sha256"]
                           and sha(OUT / "material/frozen_context_cases.jsonl") == protocol["material"]["context_sha256"],
        "counts": len(graph) == 48 and len(behavior) == 288 and len(context) == 144 and len(pairs) == 144,
        "fresh_targets": not ({row["word"] for row in graph} & {row["word"] for row in previous}),
        "graph_balance": Counter(row["partition"] for row in graph) == Counter({"discovery": 16, "confirmation": 16, "holdout": 16})
                         and Counter(row["family"] for row in graph) == Counter({"fruit": 12, "animal": 12, "tool": 12, "vehicle": 12}),
        "paired_truth": all(len(values) == 2 and {row["gold_value"] for row in values} == {"yes", "no"} for values in pairs.values()),
        "identity_balance": all(value == Counter({"yes": 36, "no": 36}) for value in label_truth.values())
                            and all(value == Counter({"yes": 3, "no": 3}) for value in target_truth.values())
                            and all(value == Counter({"yes": 48, "no": 48}) for value in surface_truth.values()),
        "zero_models": set(machine["zero_models"].values()) == {0.5},
        "compiled": compiled_ok and machine["all_candidate_lengths_matched"] and machine["all_spans_nonempty"],
        "numeric_sentinels": len(machine["numeric_sentinel_case_ids"]) == 24
                             and machine["numeric_sentinel_case_ids"] == protocol["numeric"]["sentinel_case_ids"],
        "semantic_scope": naturalness["semantic_uniqueness_rate"] == 1.0 and naturalness["independent_human_review"] is False
                          and "parameter mechanism" in naturalness["unauthorized_claims"],
        "bf16_frozen": all(model["dtype"] == "bfloat16" and model["quantization"] == "none" for model in protocol["models"]),
        "parameter_boundary": protocol["parameter_boundary"].startswith("No natural-model single-parameter"),
        "final": final["all_gates_passed"] is True and final["authorization"] == "run_phase1332_bf16_numeric_qualification",
    }
    output = {"phase": 1331, "campaign": "C043", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "run_phase1332_bf16_numeric_qualification" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_final_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
