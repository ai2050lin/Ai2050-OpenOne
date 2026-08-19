#!/usr/bin/env python3
"""Independent Phase1324 audit. This file never imports the C039 builder."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1324_c039_exact_truth_scope_contract"
PARENT = T / "result/phase1323_c038_truth_scope_composition_contract"
P = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_truth_scope_cases.jsonl"
PAIRS = OUT / "material/frozen_truth_scope_pairs.jsonl"
NATURAL = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CAL = OUT / "analysis/known_truth_composition_calibration.json"
BALANCE = OUT / "analysis/exact_stratified_balance.json"
F = OUT / "analysis/final.json"
MAIN = T / "phase1324_c039_exact_truth_scope_contract.py"
SELF = Path(__file__).resolve()
POST = OUT / "audit/independent_final_audit.json"
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = ("active_single", "active_outer_context_true", "active_outer_context_false",
          "active_inner_context_true", "active_inner_context_false", "wrong_scope", "lexical_null", "self_repeat")
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


def add(checks: dict[str, bool], name: str, value: bool) -> None:
    checks[name] = bool(value)


def main() -> None:
    protocol, source, pairs = load(P), rows(SOURCE), rows(PAIRS)
    natural, machine, calibration, balance, final = load(NATURAL), load(MACHINE), load(CAL), load(BALANCE), load(F)
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    checks: dict[str, bool] = {}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"] == final["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)})
    parent_final, parent_audit = load(PARENT / "analysis/final.json"), load(PARENT / "audit/independent_final_audit.json")
    parent_failed = {name for name, passed in parent_audit["checks"].items() if not passed}
    add(checks, "c038_terminal", parent_final["authorization"] == "stop_c038_before_model"
        and parent_final["all_gates_passed"] is False and parent_audit["checks"]["no_model"] is True
        and parent_failed == {"shortcut_controls", "final"})
    add(checks, "dependency_hashes", protocol["dependencies"]["c038_protocol"] == sha(PARENT / "protocol/preregistration.json")
        and protocol["dependencies"]["c038_final"] == sha(PARENT / "analysis/final.json")
        and protocol["dependencies"]["c038_audit"] == sha(PARENT / "audit/independent_final_audit.json"))
    add(checks, "artifact_hashes", protocol["material"]["source_sha256"] == sha(SOURCE)
        and protocol["material"]["pairs_sha256"] == sha(PAIRS)
        and protocol["semantic_naturalness"]["sha256"] == sha(NATURAL)
        and protocol["known_truth"]["sha256"] == sha(CAL)
        and protocol["zero_models"]["balance_sha256"] == sha(BALANCE))
    add(checks, "counts", len(source) == 2304 and len(pairs) == 1152)
    add(checks, "global_balance", Counter(row["gold_value"] for row in source) == Counter({"yes": 1152, "no": 1152}))
    add(checks, "partition_balance", Counter(row["partition"] for row in source) == Counter({p: 768 for p in PARTITIONS}))
    add(checks, "surface_balance", Counter(row["surface"] for row in source) == Counter({s: 1152 for s in SURFACES}))
    add(checks, "panel_balance", Counter(row["panel"] for row in source) == Counter({p: 288 for p in PANELS}))
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        strata[(row["partition"], row["surface"], row["panel"], row["property"])].append(row)
    exact = len(strata) == 288 and all(
        len(values) == 8
        and Counter(row["gold_position"] for row in values) == Counter({0: 4, 1: 4})
        and Counter(row["gold_value"] for row in values) == Counter({"yes": 4, "no": 4})
        and Counter(tuple(row["candidates"]) for row in values) == Counter({("yes", "no"): 4, ("no", "yes"): 4})
        for values in strata.values()
    )
    add(checks, "exact_stratified_quota", exact and balance["all_strata_exact"]
        and balance["candidate_position_accuracy"] == 0.5 and balance["failure_count"] == 0)
    add(checks, "pair_semantics", all(len(pair["states"]) == 2
        and ((pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"]) == (pair["panel"] in ACTIVE))
        and all(state["gold_position"] == pair["candidates"].index(state["gold_value"]) for state in pair["states"])
        for pair in pairs))
    add(checks, "compiled_roles", all(row["true_boundary"] == len(row["ids"]) - 1
        and row["positions"]["assistant_boundary"] == [len(row["ids"]) - 1]
        and row["positions"]["query_entity"] and row["positions"]["query_property"] and row["positions"]["query_end"]
        for row in source))
    add(checks, "equal_pair_lengths", all(len(pair["states"][0]["ids"]) == len(pair["states"][1]["ids"]) for pair in pairs))
    add(checks, "zero_models", machine["candidate_position_accuracy"] == 0.5
        and machine["surface_only_accuracy"] <= 0.51 and machine["active_word_only_accuracy"] <= 0.60)
    add(checks, "machine_naturalness_scope", natural["grammatical_template_rate"] == 1.0
        and natural["semantic_uniqueness_rate"] == 1.0 and natural["answer_uniqueness_rate"] == 1.0
        and natural["independent_human_review"] is False and natural["authorized_claim"] == "controlled metalinguistic truth-scope kernel")
    add(checks, "known_truth", calibration["double_false_is_identity"] is True
        and calibration["surface_twins"] is True and calibration["outer_inner_twins"] is True)
    add(checks, "behavior_before_hidden", protocol["behavior"]["hidden_states_read"] is False
        and protocol["behavior"]["success_authorization"] == "phase1326_c039_composition_field_only")
    add(checks, "downstream_frozen", protocol["field"]["sketch_seed"] == 1326
        and protocol["causal"]["block_depth"] == 14 and protocol["causal"]["rescue_depth"] == 15
        and len(protocol["causal"]["arms"]) == 8)
    add(checks, "no_overclaim", "does not presuppose" in protocol["claim_scope"]
        and "typed conditional transition" in protocol["theory_competition"])
    add(checks, "hard_stops", len(protocol["hard_stops"]) == 5 and "no same-contract retry" in protocol["hard_stops"][-1])
    add(checks, "final", final["all_gates_passed"] is True
        and final["authorization"] == "phase1325_c039_qwen3_behavior_only"
        and final["exact_candidate_position_accuracy"] == 0.5)
    add(checks, "no_model", protocol["model_weights_loaded"] is False)
    passed = all(checks.values())
    result = {"phase": 1324, "campaign": "C039", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": passed,
              "authorization": "phase1325_c039_qwen3_behavior_only" if passed else "none",
              "claim_boundary": "Exact balance qualifies a controlled-English test; it does not establish natural-language external validity."}
    POST.parent.mkdir(parents=True, exist_ok=True)
    POST.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"passed": result["passed"], "total": result["total"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
