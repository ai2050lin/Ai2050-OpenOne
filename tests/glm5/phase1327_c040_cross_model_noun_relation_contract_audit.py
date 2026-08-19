#!/usr/bin/env python3
"""Independent audit for Phase1327 C040 preregistration."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1327_c040_cross_model_noun_relation_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("reference_family", "vocabulary_kind")


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


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    protocol, final = load(PROTOCOL), load(FINAL)
    behavior = rows(OUT / "material/frozen_behavior_cases.jsonl")
    contexts = rows(OUT / "material/frozen_context_cases.jsonl")
    graph = load(OUT / "material/frozen_concept_graph.json")["concepts"]
    naturalness = load(OUT / "material/pre_model_semantic_naturalness_review.json")
    machine = load(OUT / "audit/tokenizer_zero_model_audit.json")
    sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        sets[row["semantic_set"]].append(row)
    exact_sets = all(
        len(values) == 4 and Counter(r["gold_position"] for r in values) == Counter({0: 2, 1: 2})
        and Counter(r["surface"] for r in values) == Counter({s: 2 for s in SURFACES})
        for values in sets.values()
    )
    candidate_semantics = all(
        row["gold_value"] in row["candidates"] and row["candidates"][row["gold_position"]] == row["gold_value"]
        and row["candidates"][0] != row["candidates"][1]
        for row in behavior
    )
    compiled_ok = True
    for model in MODELS:
        compiled_behavior = rows(OUT / f"compiled/{model}_behavior.jsonl")
        compiled_context = rows(OUT / f"compiled/{model}_context.jsonl")
        compiled_ok &= (
            len(compiled_behavior) == 576 and len(compiled_context) == 144
            and all(row["prompt_ids"] and len(row["candidate_ids"]) == 2
                    and len(row["candidate_ids"][0]) == len(row["candidate_ids"][1]) > 0
                    for row in compiled_behavior)
            and all(row["text_ids"] and row["word_positions"]
                    and max(row["word_positions"]) < len(row["text_ids"])
                    and row["boundary_position"] == len(row["text_ids"]) - 1 for row in compiled_context)
        )
    timeless = {key: value for key, value in protocol.items()
                if key not in {"contract_sha256", "script_sha256", "auditor_sha256", "created_at_utc"}}
    checks = {
        "parent_terminal": protocol["parent"] == {
            "final_authorization": "close_c039_at_descriptive_composition_boundary",
            "audit": True, "erratum_authorization_change": "none"},
        "contract_hash": digest(timeless) == protocol["contract_sha256"],
        "self_hashes": sha(T / "phase1327_c040_cross_model_noun_relation_contract.py") == protocol["script_sha256"]
                       and sha(Path(__file__).resolve()) == protocol["auditor_sha256"],
        "material_hashes": (
            sha(OUT / "material/frozen_concept_graph.json") == protocol["material"]["concept_graph_sha256"]
            and sha(OUT / "material/frozen_behavior_cases.jsonl") == protocol["material"]["behavior_sha256"]
            and sha(OUT / "material/frozen_context_cases.jsonl") == protocol["material"]["context_sha256"]
            and sha(OUT / "material/pre_model_semantic_naturalness_review.json") == protocol["material"]["naturalness_sha256"]
        ),
        "concept_graph": len(graph) == 48 and Counter(r["family"] for r in graph) == Counter({f: 12 for f in ("fruit", "animal", "tool", "vehicle")})
                         and Counter(r["partition"] for r in graph) == Counter({p: 16 for p in PARTITIONS}),
        "behavior_counts": len(behavior) == 576 and len(sets) == 144
                           and Counter(r["partition"] for r in behavior) == Counter({p: 192 for p in PARTITIONS})
                           and Counter(r["surface"] for r in behavior) == Counter({s: 288 for s in SURFACES}),
        "mirrored_sets": exact_sets and Counter(r["gold_position"] for r in behavior) == Counter({0: 288, 1: 288}),
        "candidate_semantics": candidate_semantics,
        "context_counts": len(contexts) == 144 and len({r["case_id"] for r in contexts}) == 144,
        "compiled": bool(compiled_ok),
        "zero_models": (
            machine["zero_models"]["candidate_position"] == 0.5
            and machine["zero_models"]["lexicographic"] <= 0.55
            and machine["zero_models"]["candidate_identity_majority"] <= 0.55
            and machine["zero_models"]["target_char_bigram_overlap"] <= 0.60
            and max(machine["zero_models"]["per_model_shorter_token"].values()) <= 0.51
            and machine["all_candidate_token_lengths_matched"]
        ),
        "semantic_naturalness_scope": naturalness["semantic_uniqueness_rate"] == 1.0
                                      and naturalness["independent_human_review"] is False
                                      and "complete lexical knowledge graph" in naturalness["unauthorized_claims"],
        "frozen_models_and_gates": [m["name"] for m in protocol["models"]] == list(MODELS)
                                   and protocol["behavior_gate"]["minimum_authorized_models"] == 2
                                   and protocol["relation_gate"]["minimum_authorized_models"] == 2,
        "coordinate_free_scope": "not shared physical coordinates" in protocol["research_object"]
                                 and "never raw cross-model coordinates" in protocol["relation_camera"]["comparison"],
        "stop_rules": len(protocol["stop_rules"]) == 4 and "No hidden states" in protocol["stop_rules"][0],
        "final": final["all_gates_passed"] is True
                 and final["authorization"] == "run_phase1328_sequential_behavior",
    }
    output = {"phase": 1327, "campaign": "C040", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()), "final_sha256": sha(FINAL)}
    save(AUDIT, output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
