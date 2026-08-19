#!/usr/bin/env python3
"""Phase1324: freeze C039 with an exactly balanced truth-scope contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1323_c038_truth_scope_composition_contract as scaffold  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE, CAMPAIGN = 1324, "C039"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1324_c039_exact_truth_scope_contract_audit.py"
PARENT = T / "result/phase1323_c038_truth_scope_composition_contract"
OUT = T / "result/phase1324_c039_exact_truth_scope_contract"
SOURCE = OUT / "material/frozen_truth_scope_cases.jsonl"
PAIRS = OUT / "material/frozen_truth_scope_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_composition_calibration.json"
BALANCE = OUT / "analysis/exact_stratified_balance.json"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"

PARTITIONS = scaffold.PARTITIONS
SURFACES = scaffold.SURFACES
PANELS = scaffold.PANELS
ACTIVE_PANELS = scaffold.ACTIVE_PANELS
PROFILES = scaffold.PROFILES
PROPERTY_BANKS = {
    "discovery": ("composed", "aware", "tolerant", "sincere", "upbeat", "reserved"),
    "confirmation": ("tranquil", "observant", "forgiving", "candid", "sociable", "discreet"),
    "holdout": ("serene", "vigilant", "considerate", "frank", "optimistic", "restrained"),
}
NAME_CANDIDATES = tuple("""
Aaron Abel Abram Adam Aidan Albert Alec Alfred Alvin Andrew Anthony Arnold Arthur August Austin Barry Benjamin Blake
Bradley Brian Caleb Calvin Cameron Carl Charles Clark Claude Clayton Colin Curtis Dale Daniel Darren David Dean Dennis
Derek Douglas Duncan Earl Edward Eric Ernest Ethan Eugene Ezra Frederick Garrett George Harold Henry Howard Ian Jack
Jacob James Jason Jeffrey Jeremy John Joseph Kenneth Kevin Kyle Lawrence Louis Lucas Luke Marcus Mark Martin Matthew
Michael Nathan Patrick Paul Peter Raymond Richard Robert Roger Russell Samuel Scott Sean Stephen Stuart Taylor Thomas
Timothy Todd Wayne William Wyatt Alan Allen Bruce Chad Craig Donald Francis Gerald Gilbert Gordon Hector Horace Irving
Jerome Keith Lloyd Marvin Maurice Neil Norman Randall Roy Stanley Trevor Vincent Walter Wesley
""".split())

BEHAVIOR_TH = dict(scaffold.BEHAVIOR_TH)
FIELD_TH = dict(scaffold.FIELD_TH)
CAUSAL_TH = dict(scaffold.CAUSAL_TH)


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


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def majority_accuracy(source: list[dict[str, Any]], key: str) -> float:
    groups: dict[Any, Counter[str]] = defaultdict(Counter)
    for row in source:
        groups[row[key]][row["gold_value"]] += 1
    return sum(max(values.values()) for values in groups.values()) / len(source)


def c038_is_terminal() -> tuple[bool, dict[str, Any]]:
    final = load(PARENT / "analysis/final.json")
    audit = load(PARENT / "audit/independent_final_audit.json")
    expected_failed_checks = {name for name, passed in audit["checks"].items() if not passed}
    terminal = (
        final.get("authorization") == "stop_c038_before_model"
        and final.get("all_gates_passed") is False
        and audit["checks"].get("no_model") is True
        and expected_failed_checks == {"shortcut_controls", "final"}
    )
    return terminal, {"final": final, "audit": audit, "expected_failed_checks": sorted(expected_failed_checks)}


def exact_candidate_order(profile: int, prop_index: int, surface: str, panel: str) -> list[str]:
    cell_parity = (prop_index + SURFACES.index(surface) + PANELS.index(panel)) % 2
    swap = bool((profile // 2) ^ cell_parity)
    return ["no", "yes"] if swap else ["yes", "no"]


def enforce_exact_quota(source: list[dict[str, Any]], pairs: list[dict[str, Any]], tokenizer: Any) -> None:
    token_map = {word: tokenizer.encode(word, add_special_tokens=False)[0] for word in ("yes", "no")}
    pair_orders: dict[str, tuple[list[str], list[int]]] = {}
    for pair in pairs:
        candidates = exact_candidate_order(
            pair["profile_index"], pair["property_index"], pair["surface"], pair["panel"]
        )
        candidate_ids = [int(token_map[word]) for word in candidates]
        pair["candidates"] = candidates
        for state in pair["states"]:
            state["candidate_ids"] = candidate_ids
            state["gold_position"] = candidates.index(state["gold_value"])
        pair_orders[pair["pair_key"]] = (candidates, candidate_ids)
    for row in source:
        candidates, candidate_ids = pair_orders[row["pair_key"]]
        row["candidates"] = candidates
        row["candidate_ids"] = candidate_ids
        row["gold_position"] = candidates.index(row["gold_value"])


def balance_audit(source: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        strata[(row["partition"], row["surface"], row["panel"], row["property"])].append(row)
    failures: list[dict[str, Any]] = []
    for key, values in sorted(strata.items()):
        gold_positions = Counter(row["gold_position"] for row in values)
        orders = Counter(tuple(row["candidates"]) for row in values)
        gold_values = Counter(row["gold_value"] for row in values)
        if gold_positions != Counter({0: 4, 1: 4}) or orders != Counter({("yes", "no"): 4, ("no", "yes"): 4}) \
                or gold_values != Counter({"yes": 4, "no": 4}):
            failures.append({"stratum": key, "gold_positions": dict(gold_positions),
                             "orders": {str(k): v for k, v in orders.items()}, "gold_values": dict(gold_values)})
    return {
        "stratum_definition": ["partition", "surface", "panel", "property"],
        "stratum_count": len(strata), "cases_per_stratum": sorted({len(v) for v in strata.values()}),
        "required_gold_position_counts": {"0": 4, "1": 4},
        "required_candidate_order_counts": {"yes_no": 4, "no_yes": 4},
        "required_gold_value_counts": {"yes": 4, "no": 4},
        "failure_count": len(failures), "failures": failures,
        "candidate_position_accuracy": majority_accuracy(source, "gold_position"),
        "all_strata_exact": not failures,
    }


def build(force: bool) -> None:
    terminal, terminal_evidence = c038_is_terminal()
    if not terminal:
        raise RuntimeError("C038 terminal evidence is inconsistent")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    scaffold.OUT = OUT
    scaffold.PROPERTY_BANKS = PROPERTY_BANKS
    scaffold.NAME_CANDIDATES = NAME_CANDIDATES
    source, pairs = scaffold.build_material(tokenizer)
    enforce_exact_quota(source, pairs, tokenizer)
    machine, naturalness, calibration = scaffold.audits(source, pairs)
    balance = balance_audit(source)
    naturalness.update({
        "review_status": "machine_only_controlled_language",
        "independent_human_review_required_for": "natural-language external-validity claims only",
        "authorized_claim": "controlled metalinguistic truth-scope kernel",
        "unauthorized_claims": ["natural-language scope comprehension", "all negation", "cross-lingual logic"],
    })

    write_rows(SOURCE, source)
    write_rows(PAIRS, pairs)
    save(MACHINE, machine)
    save(NATURALNESS, naturalness)
    save(CALIBRATION, calibration)
    save(BALANCE, balance)
    all_pass = (
        len(source) == 2304 and len(pairs) == 1152 and balance["all_strata_exact"]
        and balance["candidate_position_accuracy"] == 0.5
        and Counter(row["gold_value"] for row in source) == Counter({"yes": 1152, "no": 1152})
        and machine["surface_only_accuracy"] <= 0.51 and machine["active_word_only_accuracy"] <= 0.60
        and machine["all_boundaries_compiled"] and machine["all_required_roles_present"]
        and machine["pair_lengths_equal"] and machine["semantic_program_exact"]
        and naturalness["grammatical_template_rate"] == 1.0 and naturalness["balanced_quotes_rate"] == 1.0
        and naturalness["double_space_rate"] == 0.0 and naturalness["semantic_uniqueness_rate"] == 1.0
        and calibration["double_false_is_identity"] and calibration["surface_twins"]
        and calibration["outer_inner_twins"]
    )

    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "schema_version": "phase1324.c039.exact_truth_scope_contract.v1",
        "research_object": "controlled typed proposition truth composition with inner/outer role interventions",
        "language_types": {
            "Attr": "Entity x Property -> Proposition", "Truth": "Bool x Proposition -> Proposition",
            "laws": ["Truth(true,P)=P", "Truth(false,P)=not P", "Truth(false,Truth(false,P))=P"],
        },
        "material": {
            "source_sha256": sha(SOURCE), "pairs_sha256": sha(PAIRS), "source_count": len(source),
            "pair_count": len(pairs), "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
            "panels": list(PANELS), "properties": PROPERTY_BANKS, "fresh_from_c038": True,
        },
        "partitions": {"discovery": "field prototypes only", "confirmation": "frozen confirmation",
                       "holdout": "final untouched holdout"},
        "model": "qwen3-4b-fp16-cuda-no-quantization", "models_excluded": ["glm4", "deepseek7b"],
        "zero_models": {
            "constant_label_max": 0.5, "candidate_position_exact": 0.5, "surface_only_max": 0.51,
            "active_word_only_max": 0.60, "controls": ["wrong_scope", "lexical_null", "self_repeat", "malformed_scope"],
            "balance_sha256": sha(BALANCE),
        },
        "semantic_naturalness": {
            "sha256": sha(NATURALNESS), "semantic_unique_min": 1.0, "answer_unique_min": 1.0,
            "machine_grammatical_min": 1.0, "independent_human_review": False,
            "claim_boundary": "controlled metalinguistic English only",
        },
        "known_truth": {"sha256": sha(CALIBRATION), "all_thresholds": 1.0},
        "behavior": {
            "thresholds": BEHAVIOR_TH, "hidden_states_read": False,
            "success_authorization": "phase1326_c039_composition_field_only",
            "failure_authorization": "close_c039_without_hidden",
        },
        "field": {
            "thresholds": FIELD_TH, "sketch_seed": 1326, "sketch_dim": 64,
            "roles": ["proposition_entity", "proposition_property", "active_operator", "context_operator",
                      "query_entity", "query_property", "query_end", "assistant_boundary"],
            "prototype_rule": "discovery outer-role parity prototypes classify confirmation/holdout inner-role and vice versa; no fitted alignment",
            "success_authorization": "phase1327_c039_composition_causal_only",
            "failure_authorization": "close_c039_at_descriptive_composition_boundary",
        },
        "causal": {
            "thresholds": CAUSAL_TH, "block_depth": 14, "rescue_depth": 15,
            "roles": ["proposition_entity", "proposition_property", "active_operator", "context_operator",
                      "query_entity", "query_property", "query_end"],
            "arms": ["baseline", "block", "self", "correct_parity", "wrong_parity", "wrong_role", "lexical_null", "random"],
            "success_authorization": "close_c039_with_typed_composition_causal_evidence",
            "failure_authorization": "close_c039_without_typed_composition_causal_evidence",
        },
        "hard_stops": [
            "No model before independent Phase1324 audit", "No hidden state before behavior qualification",
            "No attention/MLP/probe discovery", "No post-unblind object, material, split, model, metric, threshold, layer, role, or arm change",
            "C039 closes at first failed gate or after the causal phase; no same-contract retry",
        ],
        "claim_scope": "Tests a controlled metalinguistic truth-composition kernel; it does not presuppose a linguistic manifold, a scope tree in attention, or a categorical neural homomorphism.",
        "theory_competition": ["typed conditional transition", "response-equivalence state", "fixed direction/subspace", "task shortcut"],
        "dependencies": {
            "c038_main": sha(T / "phase1323_c038_truth_scope_composition_contract.py"),
            "c038_protocol": sha(PARENT / "protocol/preregistration.json"),
            "c038_final": sha(PARENT / "analysis/final.json"),
            "c038_audit": sha(PARENT / "audit/independent_final_audit.json"),
            "c038_terminal_digest": digest(terminal_evidence),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    authorization = "phase1325_c039_qwen3_behavior_only" if all_pass else "stop_c039_before_model"
    save(FINAL, {
        "phase": PHASE, "campaign": CAMPAIGN,
        "verdict": "contract_qualified" if all_pass else "contract_failed",
        "all_gates_passed": all_pass, "authorization": authorization,
        "exact_candidate_position_accuracy": balance["candidate_position_accuracy"],
        "exact_balance_failure_count": balance["failure_count"], "protocol_digest": digest(timeless),
    })
    print(canonical({"pairs": len(pairs), "cases": len(source), "passed": all_pass,
                     "candidate_position": balance["candidate_position_accuracy"], "authorization": authorization}))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
