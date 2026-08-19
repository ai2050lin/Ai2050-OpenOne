#!/usr/bin/env python3
"""Independent audit for Phase1329 C042."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1329_c042_relational_ecology_contract"
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
    machine = load(OUT / "audit/tokenizer_zero_model_audit.json")
    graph = load(OUT / "material/frozen_concept_graph.json")["concepts"]
    behavior, context = rows(OUT / "material/frozen_behavior_cases.jsonl"), rows(OUT / "material/frozen_context_cases.jsonl")
    naturalness = load(OUT / "material/pre_model_semantic_naturalness_review.json")
    parent = load(T / "result/phase1328_c041_balanced_noun_relation_contract/audit/independent_failure_audit.json")
    sets, identities = defaultdict(list), defaultdict(lambda: [0, 0])
    for row in behavior:
        sets[row["semantic_set"]].append(row)
        for i, candidate in enumerate(row["candidates"]):
            identities[candidate][0] += int(i == row["gold_position"])
            identities[candidate][1] += 1
    identity_rates = {word: right / total for word, (right, total) in identities.items()}
    timeless = {key: value for key, value in protocol.items()
                if key not in {"contract_sha256", "script_sha256", "auditor_sha256", "created_at_utc"}}
    checks = {
        "parent_terminal": parent["all_checks_passed"] is True and parent["authorization"] == "close_c041_and_permit_fresh_non_scaffold_contract",
        "contract_hash": digest(timeless) == protocol["contract_sha256"],
        "source_hashes": sha(T / "phase1329_c042_relational_ecology_contract.py") == protocol["script_sha256"]
                         and sha(Path(__file__).resolve()) == protocol["auditor_sha256"],
        "material_hashes": sha(OUT / "material/frozen_concept_graph.json") == protocol["material"]["graph_sha256"]
                           and sha(OUT / "material/frozen_behavior_cases.jsonl") == protocol["material"]["behavior_sha256"]
                           and sha(OUT / "material/frozen_context_cases.jsonl") == protocol["material"]["context_sha256"],
        "counts": len(graph) == 48 and len(behavior) == 576 and len(context) == 144 and len(sets) == 144,
        "partitions": Counter(row["partition"] for row in graph) == Counter({"discovery": 16, "confirmation": 16, "holdout": 16}),
        "families": Counter(row["family"] for row in graph) == Counter({"fruit": 12, "animal": 12, "tool": 12, "vehicle": 12}),
        "exact_sets": len(sets) == 144 and all(len(values) == 4 and Counter(r["gold_position"] for r in values) == Counter({0: 2, 1: 2})
                                                     and Counter(r["surface"] for r in values) == Counter({"reference_family": 2, "vocabulary_kind": 2})
                                                     for values in sets.values()),
        "anchors_disjoint": not ({row["word"] for row in graph} & set(identities)) and len(identities) == 8,
        "identity_exact": set(identity_rates.values()) == {0.5} and machine["zero_models"]["candidate_identity_majority"] == 0.5,
        "other_zero_models": machine["zero_models"]["candidate_position"] == 0.5
                             and machine["zero_models"]["lexicographic"] <= 0.55
                             and machine["zero_models"]["target_char_bigram_overlap"] <= 0.60
                             and max(machine["zero_models"]["per_model_shorter_token"].values()) <= 0.51,
        "compiled": all(len(rows(OUT / f"compiled/{model}_behavior.jsonl")) == 576
                        and len(rows(OUT / f"compiled/{model}_context.jsonl")) == 144 for model in MODELS)
                    and machine["candidate_lengths_matched"] and machine["context_spans_nonempty"],
        "naturalness_scope": naturalness["semantic_uniqueness_rate"] == 1.0 and naturalness["independent_human_review"] is False
                             and "complete knowledge ecology" in naturalness["unauthorized_claims"],
        "frozen_branch": protocol["behavior_gate"]["minimum_authorized_models"] == 2
                         and protocol["relation_gate"]["minimum_authorized_models"] == 2
                         and protocol["branch"]["behavior_fail"] == "close C042 without hidden states",
        "final": final["all_gates_passed"] is True and final["authorization"] == "run_phase1330_sequential_behavior",
    }
    output = {"phase": 1329, "campaign": "C042", "checks": checks, "identity_rates": identity_rates,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
              "authorization": "run_phase1330_sequential_behavior" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_final_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
