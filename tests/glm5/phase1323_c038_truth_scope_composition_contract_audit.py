#!/usr/bin/env python3
"""Independent zero-model audit for Phase1323; never imports the builder."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1323_c038_truth_scope_composition_contract"
PARENT = T / "result/phase1322_c037_isomorphic_full_state_field"
P = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_truth_scope_cases.jsonl"
PAIRS = OUT / "material/frozen_truth_scope_pairs.jsonl"
NATURAL = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CAL = OUT / "analysis/known_truth_composition_calibration.json"
F = OUT / "analysis/final.json"
MAIN = T / "phase1323_c038_truth_scope_composition_contract.py"
SELF = Path(__file__).resolve()
POST = OUT / "audit/independent_final_audit.json"
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = (
    "active_single", "active_outer_context_true", "active_outer_context_false",
    "active_inner_context_true", "active_inner_context_false", "wrong_scope", "lexical_null", "self_repeat",
)
ACTIVE = set(PANELS[:5])


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


def majority_accuracy(source: list[dict[str, Any]], key: str) -> float:
    groups: dict[Any, Counter[str]] = defaultdict(Counter)
    for row in source:
        groups[row[key]][row["gold_value"]] += 1
    return sum(max(values.values()) for values in groups.values()) / len(source)


def add(checks: dict[str, bool], name: str, value: bool) -> None:
    checks[name] = bool(value)


def main() -> None:
    protocol, source, pairs = load(P), rows(SOURCE), rows(PAIRS)
    natural, machine, calibration, final = load(NATURAL), load(MACHINE), load(CAL), load(F)
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    checks: dict[str, bool] = {}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)})
    add(checks, "parent_terminal", load(PARENT / "analysis/final.json").get("authorization")
        == "close_c037_at_isomorphic_field_boundary"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed", False))
    add(checks, "dependency_hashes", protocol["dependencies"] == {
        "c037_protocol": sha(PARENT / "protocol/preregistration.json"),
        "c037_final": sha(PARENT / "analysis/final.json"),
        "c037_audit": sha(PARENT / "audit/independent_final_audit.json"),
    })
    add(checks, "material_hashes", protocol["material"]["source_sha256"] == sha(SOURCE)
        and protocol["material"]["pairs_sha256"] == sha(PAIRS)
        and protocol["semantic_naturalness"]["sha256"] == sha(NATURAL)
        and protocol["known_truth"]["sha256"] == sha(CAL))
    add(checks, "counts", len(source) == 2304 and len(pairs) == 1152)
    add(checks, "gold_balance", Counter(row["gold_value"] for row in source) == Counter({"yes": 1152, "no": 1152}))
    add(checks, "partition_balance", Counter(row["partition"] for row in source) == Counter({p: 768 for p in PARTITIONS}))
    add(checks, "surface_balance", Counter(row["surface"] for row in source) == Counter({s: 1152 for s in SURFACES}))
    add(checks, "panel_balance", Counter(row["panel"] for row in source) == Counter({p: 288 for p in PANELS}))
    add(checks, "pair_states", all(len(pair["states"]) == 2 and {x["state_index"] for x in pair["states"]} == {0, 1} for pair in pairs))
    add(checks, "semantic_program", all(
        (pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"]) == (pair["panel"] in ACTIVE)
        for pair in pairs
    ))
    add(checks, "compiled_boundaries", all(row["true_boundary"] == len(row["ids"]) - 1
        and row["positions"]["assistant_boundary"] == [len(row["ids"]) - 1] for row in source))
    add(checks, "role_spans", all(row["positions"]["query_entity"] and row["positions"]["query_property"]
        and row["positions"]["query_end"] and row["positions"]["assistant_boundary"] for row in source))
    add(checks, "equal_pair_lengths", all(len(pair["states"][0]["ids"]) == len(pair["states"][1]["ids"]) for pair in pairs))
    add(checks, "shortcut_controls", majority_accuracy(source, "gold_position") <= 0.51
        and majority_accuracy(source, "surface") <= 0.51 and majority_accuracy(source, "active_truth") <= 0.60
        and machine["candidate_position_accuracy"] == majority_accuracy(source, "gold_position"))
    add(checks, "naturalness_machine_scope", natural["grammatical_template_rate"] == 1.0
        and natural["balanced_quotes_rate"] == 1.0 and natural["double_space_rate"] == 0.0
        and natural["semantic_uniqueness_rate"] == 1.0 and natural["answer_uniqueness_rate"] == 1.0
        and natural["independent_human_review"] is False)
    add(checks, "known_truth", calibration["double_false_is_identity"] is True
        and calibration["surface_twins"] is True and calibration["outer_inner_twins"] is True
        and calibration["malformed_scope_must_abstain"] is True)
    add(checks, "behavior_before_hidden", protocol["behavior"]["hidden_states_read"] is False
        and protocol["behavior"]["success_authorization"] == "phase1325_composition_field_only")
    add(checks, "field_and_causal_frozen", protocol["field"]["sketch_seed"] == 1325
        and protocol["causal"]["block_depth"] == 14 and protocol["causal"]["rescue_depth"] == 15
        and len(protocol["causal"]["arms"]) == 8)
    add(checks, "hard_stops", len(protocol["hard_stops"]) == 5
        and protocol["hard_stops"][-1].startswith("C038 closes"))
    add(checks, "claim_scope", "not all negation" in protocol["claim_scope"])
    add(checks, "final", final["all_gates_passed"] is True
        and final["authorization"] == "phase1324_qwen3_behavior_only")
    add(checks, "no_model", protocol["model_weights_loaded"] is False)
    passed = all(checks.values())
    value = {
        "phase": 1323, "campaign": "C038", "checks": checks, "passed": sum(checks.values()),
        "total": len(checks), "all_checks_passed": passed,
        "authorization": "run_phase1324_preregistration" if passed else "none",
        "claim_boundary": "Audits a controlled truth-scope composition kernel, not a complete language ontology.",
    }
    POST.parent.mkdir(parents=True, exist_ok=True)
    POST.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"passed": value["passed"], "total": value["total"], "authorization": value["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
