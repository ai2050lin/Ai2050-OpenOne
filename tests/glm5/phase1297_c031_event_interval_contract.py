#!/usr/bin/env python3
"""Phase 1297: freeze C031 event-interval inverse-lookup contract.

C030 is closed.  C031 retains only its externally grounded inverse-lookup
function, replaces all entities and values, and preregisters a numerical
camera calibration before any semantic hidden-state measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1292_c029_object_attribute_convergence_contract as scaffold  # noqa: E402
import phase1294_c030_grounded_lookup_contract as c030  # noqa: E402


PHASE = 1297
CAMPAIGN = "C031"
SCRIPT = Path(__file__).resolve()
AUDITOR = TEST_ROOT / "phase1297_c031_event_interval_contract_audit.py"
OUT = TEST_ROOT / "result/phase1297_c031_event_interval_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_event_interval_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
INDEPENDENT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

C029_MACHINE = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract/audit/tokenizer_semantic_program_audit.json"
C030_MACHINE = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract/audit/tokenizer_semantic_program_audit.json"
C030_FINAL = TEST_ROOT / "result/phase1296_c030_multievent_response_path/analysis/final.json"
C030_AUDIT = TEST_ROOT / "result/phase1296_c030_multievent_response_path/audit/independent_final_audit.json"

VALUE_BANKS = {
    "discovery": {
        "color": ("copper", "lavender", "silver"),
        "material": ("acrylic", "cotton", "steel"),
        "location": ("foyer", "pantry", "workroom"),
        "size": ("narrow", "average", "towering"),
        "shape": ("circular", "square", "curved"),
        "status": ("accepted", "paused", "rejected"),
    },
    "confirmation": {
        "color": ("golden", "gray", "pink"),
        "material": ("plastic", "velvet", "glass"),
        "location": ("lounge", "depot", "cellar"),
        "size": ("short", "regular", "immense"),
        "shape": ("round", "flat", "pointed"),
        "status": ("authorized", "stalled", "cancelled"),
    },
    "holdout": {
        "color": ("orange", "white", "black"),
        "material": ("paper", "rubber", "stone"),
        "location": ("office", "warehouse", "hallway"),
        "size": ("slight", "medium", "vast"),
        "shape": ("angular", "rounded", "folded"),
        "status": ("released", "postponed", "suspended"),
    },
}

NAME_CANDIDATES = tuple("""
Aaron Abel Ada Adam Alec Alex Alexa Alexander Alexis Amy Andre Arthur Ashton Autumn Bailey Bella Benjamin Bernard Beth Bobby Brian Bruce Calvin Carlos Carol Carter Cassandra Chad Chelsea Chloe Claire Clarence Claudia Clayton Cole Courtney Damian Dana Daniel Danielle Darren Dawn Dean Dennis Devin Edward Elizabeth Ella Ellen Emma Eric Esther Eva Evan Fernando Finn Frederick Geoffrey George Georgia Gerald Gina Graham Gwen Hayden Henry Hunter Iris Isaiah Jackie Jacob James Jason Jean Joanna John Joseph Joy Juan Karen Kate Kirk Lance Linda Lisa Lloyd Lucy Luke Madison Manuel Marc Marie Marina Martin Mary Max Maya Michael Molly Nancy Naomi Nora Owen Paul Peggy Penelope Phoebe Quentin Rachel Regina Riley Rita Robert Roger Roland Rosemary Roy Ryan Sabrina Sally Sarah Scott Selena Shane Shawn Simon Sonia Sophie Stanley Steven Sylvia Tanya Theodore Tiffany Tina Tobias Tony Travis Trisha Tyler Vicki Vivian Wade Wallace Wanda Wesley Wyatt Xavier Yvonne Zoe Alvin Amos Ann Arnold Brooke Byron Celeste Damon Della Doreen Duncan Edmund Eliza Ellis Elsa Emery Ernest Ethel Etta Garrett Gemma Gilbert Greta Griffin Guy Harley Homer Hope Ines Ingrid Iona Irma Jack Jane Jasper Jeanette Jonas Josef Josie Jude June Kent Lara Laurel Leon Lila Lina Lionel Lola Loretta Louisa Lydia Malcolm Mara Marcia Marco Marjorie Matilda Maureen Milo Miriam Morris Nadia Nelson Noel Odette Olga Omar Opal Orlando Otto Pearl Petra Preston Quinn Raquel Raul Reid Rhonda Rodney Rosa Ross Rowan Ruben Ruth Saul Serena Sidney Silas Simone Solomon Sonya Suzanne Tabitha Ted Tessa Theo Theresa Toby Trent Troy Vera Vernon Viola Violet Wilbur Willis Yolanda Zane
""".split())

CALIBRATION_THRESHOLDS = {
    "case_count_min": 96,
    "finite_fraction_min": 1.0,
    "exact_duplicate_relative_max": 1e-6,
    "same_batch_prefix_relative_max": 0.0025,
    "cross_composition_prefix_relative_max": 0.005,
    "derived_tolerance_multiplier": 4.0,
    "derived_tolerance_floor": 1e-6,
    "derived_tolerance_cap": 0.01,
}

HIDDEN_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.99,
    "discovery_active_relative_median_min": 0.001,
    "discovery_active_to_max_control_ratio_min": 1.20,
    "discovery_active_over_controls_fraction_min": 0.75,
    "discovery_adjacent_depths_min": 2,
    "transfer_active_relative_median_min": 0.001,
    "transfer_active_to_max_control_ratio_min": 1.10,
    "transfer_active_over_controls_fraction_min": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    tmp.replace(path)


def old_names() -> set[str]:
    return set(load(C029_MACHINE)["token_audit"]["selected_names"]) | set(load(C030_MACHINE)["token_audit"]["selected_names"])


def old_values() -> set[str]:
    return {
        value
        for banks in (scaffold.VALUE_BANKS, c030.VALUE_BANKS)
        for partition in banks.values()
        for values in partition.values()
        for value in values
    }


def select_names(tokenizer: Any) -> tuple[str, ...]:
    prior = old_names()
    eligible = [
        name for name in dict.fromkeys(NAME_CANDIDATES)
        if name not in prior and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    needed = len(scaffold.PARTITIONS) * scaffold.PROFILES_PER_PARTITION * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} disjoint one-token names; need {needed}")
    return tuple(eligible[:needed])


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = select_names(tokenizer)
    original = (scaffold.VALUE_BANKS, scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause)
    try:
        scaffold.VALUE_BANKS = VALUE_BANKS
        scaffold.NAME_POOL = names
        scaffold.record_clause = c030.corrected_record
        scaffold.query_clause = c030.corrected_query
        rows, token_audit = scaffold.build_cases(tokenizer)
    finally:
        scaffold.VALUE_BANKS, scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause = original
    for row in rows:
        row["schema_version"] = "phase1297.c031.case.v1"
        row["case_id"] = "c031-" + digest({"group": row["group_id"], "state": row["binding_state"], "prompt": row["candidate_prompt"]})[:20]
    token_audit.update({
        "selected_names": list(names),
        "c029_c030_name_overlap": sorted(old_names() & set(names)),
        "eligible_disjoint_name_count": sum(
            name not in old_names() and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
            for name in dict.fromkeys(NAME_CANDIDATES)
        ),
    })
    return rows, token_audit


def naturalness_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    review = c030.grammar_review(rows)
    review.update({
        "reviewed_before_any_c031_weight_load": True,
        "semantic_uniqueness_recomputed_for_all_cases": all(
            sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1
            for row in rows
        ),
        "new_lexical_material": True,
        "limitation": "Controlled English and semantic uniqueness passed deterministic review; no independent human rating panel was available.",
    })
    return review


def build(force: bool) -> None:
    if load(C030_FINAL).get("authorization") != "close_c030_without_path_claim":
        raise RuntimeError("C030 is not frozen closed")
    if not load(C030_AUDIT).get("all_checks_passed"):
        raise RuntimeError("C030 final audit did not pass")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)

    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True)
    rows, token_audit = build_material(tokenizer)
    write_jsonl(MATERIAL, rows)
    review = naturalness_review(rows)
    save(NATURALNESS, review)
    program = scaffold.program_audit(rows)
    values = {value for partition in VALUE_BANKS.values() for values in partition.values() for value in values}
    machine = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_audit": token_audit,
        "program_audit": program,
        "prior_value_overlap": sorted(old_values() & values),
        "semantic_unique": review["semantic_uniqueness_recomputed_for_all_cases"],
        "naturalness_passed": review["all_checks_passed"],
        "all_machine_checks_passed": bool(
            review["all_checks_passed"] and review["semantic_uniqueness_recomputed_for_all_cases"]
            and not token_audit["c029_c030_name_overlap"] and not (old_values() & values)
            and token_audit["all_candidates_single_token"]
            and program["shortcut_ceiling"] <= scaffold.THRESHOLDS["shortcut_program_accuracy_max"]
            and program["active_same_bag_different_gold_pairs"] == program["active_pair_count"]
        ),
    }
    save(MACHINE, machine)
    save(ENVIRONMENT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "model_weights_loaded": False,
        "tokenizer_only": True,
    })

    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "experiment_id": "EXP-C031-WP00-001",
        "schema_version": "phase1297.c031.preregistration.v1",
        "purpose": "resolve C030 event-timing and numerical-camera ambiguities without reopening C030",
        "adjudication": {
            "accepted": [
                "C030 behavior object is strong within its frozen controlled-English interface",
                "answer-boundary residual magnitude band at depths 25-26 is repeatable and active-specific",
                "C030 did not close a multi-event path",
            ],
            "corrected_overclaims": [
                "answer-boundary magnitude does not identify content, route, or causal component",
                "four sampled tokens do not establish absence of gradual transport or one-shot convergence",
                "the record-value discrepancy was observed but not causally attributed to padding or FP16 kernels",
                "C030 inverse lookup does not prove general relation understanding",
            ],
        },
        "construct": {
            "world_state": "finite explicit map Entity x Attribute -> Value",
            "query": "unique inverse lookup (Attribute, Value) -> Entity",
            "type_signature": "(WorldState, Attribute, Value) -> Entity",
            "operation_requested_from_model": False,
            "gold_source": "explicit mapping, independently recomputed",
        },
        "material": {
            "partitions": list(scaffold.PARTITIONS),
            "profiles_per_partition": scaffold.PROFILES_PER_PARTITION,
            "attributes": list(scaffold.ATTRIBUTES),
            "panels": list(scaffold.PANELS),
            "surfaces": list(scaffold.SURFACES),
            "candidate_orders": list(scaffold.CANDIDATE_ORDERS),
            "binding_states": list(scaffold.BINDING_STATES),
            "case_count": len(rows),
            "independent_profile_count": 24,
            "typed_query_count": 144,
            "candidate_sequences": len(rows) * 3,
            "generation_cases": 1536,
            "c029_c030_entity_overlap": token_audit["c029_c030_name_overlap"],
            "c029_c030_value_overlap": sorted(old_values() & values),
            "material_sha256": sha(MATERIAL),
            "naturalness_sha256": sha(NATURALNESS),
        },
        "model": {
            "model_id": "qwen3-4b-fp16-cuda-no-quantization",
            "formal_behavior_runs": 1,
            "formal_calibration_runs": 1,
            "formal_hidden_runs": 1,
            "other_models_authorized": False,
            "native_chat_template": True,
            "enable_thinking": False,
        },
        "zero_models": program,
        "behavior_thresholds": scaffold.THRESHOLDS,
        "calibration_thresholds": CALIBRATION_THRESHOLDS,
        "hidden_thresholds": HIDDEN_THRESHOLDS,
        "event_registry": {
            "descriptive_events": ["record_slot0_entity", "record_slot0_value", "query_clause_end", "user_answer_cue_end", "assistant_answer_boundary"],
            "primary_transfer_events": ["user_answer_cue_end", "assistant_answer_boundary"],
            "query_clause_end_is_required_for_description_not_for_path_gate": True,
            "depths": list(range(37)),
            "discovery_selection": "earliest adjacent depth pair independently passing each primary event",
            "confirmation_and_holdout": "frozen event-specific discovery depths; no reselection",
        },
        "numerical_tolerance_rule": "tau=max(floor,min(cap,multiplier*max observed calibration noise)); freeze tau before semantic hidden run",
        "failure_and_stop_branches": {
            "phase1297_audit_pass": "authorize_phase1298_numerical_calibration_only",
            "phase1298_fail": "close_c031_as_numerically_unqualified",
            "phase1298_pass": "freeze_empirical_tolerance_and_authorize_phase1299_behavior_only",
            "phase1299_fail": "close_c031_without_hidden",
            "phase1299_pass": "authorize_phase1300_event_sequence_hidden_only",
            "phase1300_fail": "close_c031_without_path_claim",
            "phase1300_pass": "authorize_phase1301_preregistered_causal_transfer_and_rescue",
            "phase1301_fail": "close_c031_with_bounded_descriptive_path_only",
            "phase1301_pass": "complete_c031_qwen_single_model_closure",
        },
        "freeze_rules": [
            "No Qwen3 weight may load before this contract and independent audit pass.",
            "No semantic hidden state may be measured before calibration and behavior gates pass.",
            "No object, material, split, model, zero model, threshold, event, parser, or stop branch may change after creation.",
            "After unblinding only the preregistered branch may run; every branch failure closes C031.",
            "No threshold relaxation, event reselection, prompt repair, seed rerun, or other-model vote is allowed.",
        ],
        "dependencies": {
            "c030_final": sha(C030_FINAL),
            "c030_audit": sha(C030_AUDIT),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    frozen = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)}
    save(PROTOCOL, frozen)
    print(canonical({"phase": PHASE, "cases": len(rows), "issues": len(review["issues"]), "shortcut": program["shortcut_ceiling"], "digest": frozen["protocol_digest"]}))


def finalize() -> None:
    protocol = load(PROTOCOL)
    audit = load(INDEPENDENT)
    if not audit.get("all_checks_passed"):
        raise RuntimeError("independent audit failed")
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "c031_event_interval_contract_frozen_and_independently_audited",
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": protocol["material"]["material_sha256"],
        "model_weights_loaded": False,
        "audit_passed": True,
        "authorization": "phase1298_numerical_calibration_only",
    }
    save(FINAL, final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force) if args.command == "build" else finalize()


if __name__ == "__main__":
    main()
