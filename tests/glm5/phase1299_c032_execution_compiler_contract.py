#!/usr/bin/env python3
"""Phase 1299: freeze the full C032 execution-compiler campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1292_c029_object_attribute_convergence_contract as scaffold  # noqa: E402
import phase1294_c030_grounded_lookup_contract as grammar  # noqa: E402

PHASE = 1299
CAMPAIGN = "C032"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1299_c032_execution_compiler_contract_audit.py"
PARENT = T / "result/phase1298_c031_fp16_same_shape_calibration"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
SOURCE_MATERIAL = T / "result/phase1297_c031_event_interval_contract/material/frozen_event_interval_cases.jsonl"
SOURCE_NATURALNESS = T / "result/phase1297_c031_event_interval_contract/material/pre_model_semantic_naturalness_review.json"
SOURCE_MACHINE = T / "result/phase1297_c031_event_interval_contract/audit/tokenizer_semantic_program_audit.json"
OUT = T / "result/phase1299_c032_execution_compiler_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_inverse_lookup_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/semantic_program_reaudit.json"
INDEPENDENT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

COMPILER_ARMS = ("left_global_baseline", "right_padding", "record_event_aligned", "equalized_suffix")
COMPILER_CANDIDATES = ("right_padding", "record_event_aligned", "equalized_suffix")
COMPILER_PRIORITY = COMPILER_CANDIDATES
NUMERIC_THRESHOLDS = {
    "case_count_min": 96,
    "finite_fraction_min": 1.0,
    "exact_duplicate_relative_max": 1e-6,
    "same_prefix_relative_max": 1e-6,
    "cross_composition_relative_max": 1e-6,
    "candidate_compilers_passing_min": 2,
    "tau_multiplier": 4.0,
    "tau_floor": 1e-7,
    "tau_cap": 1e-4,
}
HIDDEN_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.99,
    "discovery_norm_median_min": 0.001,
    "discovery_norm_ratio_min": 1.20,
    "discovery_norm_win_fraction_min": 0.75,
    "discovery_identity_positive_fraction_min": 0.75,
    "discovery_identity_ratio_min": 1.15,
    "transfer_norm_median_min": 0.001,
    "transfer_norm_ratio_min": 1.10,
    "transfer_norm_win_fraction_min": 0.70,
    "transfer_identity_positive_fraction_min": 0.70,
    "transfer_identity_ratio_min": 1.05,
    "adjacent_depths_min": 2,
}
CAUSAL_THRESHOLDS = {
    "correct_donor_signed_gain_median_min": 0.5,
    "correct_over_wrong_donor_ratio_min": 1.25,
    "correct_over_matched_null_ratio_min": 1.25,
    "pairwise_correct_donor_win_fraction_min": 0.75,
    "confirmation_holdout_each_min": 0.70,
    "natural_behavior_retention_min": 0.99,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while c := f.read(1024 * 1024): h.update(c)
    return h.hexdigest()


def load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reaudit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = all(sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1 for row in rows)
    gold = all([e for e, f in row["assignments"].items() if f[row["attribute"]] == row["target_value"]] == [row["gold_candidate"]] for row in rows)
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: pairs[row["group_id"]].append(row)
    active = all(p[0]["gold_candidate"] != p[1]["gold_candidate"] for key, p in pairs.items() if "|active|" in key)
    null = all(p[0]["gold_candidate"] == p[1]["gold_candidate"] for key, p in pairs.items() if "|matched_null|" in key)
    review = grammar.grammar_review(rows)
    prior_machine = load(SOURCE_MACHINE)
    return {
        "case_count": len(rows), "semantic_unique": semantic, "gold_recomputes": gold,
        "active_changes_gold": active, "matched_null_preserves_gold": null,
        "naturalness_issue_count": len(review["issues"]), "naturalness_passed": review["all_checks_passed"],
        "single_token_candidates": prior_machine["token_audit"]["all_candidates_single_token"],
        "shortcut_ceiling": prior_machine["program_audit"]["shortcut_ceiling"],
        "all_checks_passed": semantic and gold and active and null and review["all_checks_passed"] and prior_machine["token_audit"]["all_candidates_single_token"] and prior_machine["program_audit"]["shortcut_ceiling"] <= 0.70,
    }


def build(force: bool) -> None:
    if load(PARENT_FINAL).get("authorization") != "close_c031_as_numerically_unqualified" or not load(PARENT_AUDIT).get("all_checks_passed"):
        raise RuntimeError("C031 is not audit-closed")
    if OUT.exists() and not force: raise RuntimeError(f"{OUT} exists")
    if OUT.exists(): shutil.rmtree(OUT)
    MATERIAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_MATERIAL, MATERIAL)
    shutil.copyfile(SOURCE_NATURALNESS, NATURALNESS)
    rows = read_jsonl(MATERIAL)
    audit = reaudit(rows)
    save(MACHINE, {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), **audit})
    zero = scaffold.program_audit(rows)
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "experiment_id": "EXP-C032-WP00-001",
        "schema_version": "phase1299.c032.preregistration.v1",
        "purpose": "select a reproducible physical execution compiler, then test one fixed inverse-lookup object through behavior, event identity, and causal rescue",
        "adjudication": {
            "accepted": ["C031 stopped correctly before semantic measurement", "logical prefix equivalence, physical layout equivalence, and functional equivalence require separate ledgers", "the inverse-lookup behavior object remains eligible under a new contract"],
            "corrected": ["Phase1298 maximum noise was not universal: the median was zero", "physical padding alignment is a candidate cause rather than an established unique cause", "Phase1298 alone does not invalidate the C030 answer-boundary band", "C023-C028 failures do not prove natural language is globally non-algebraic"],
        },
        "construct": {"world_state": "finite explicit map Entity x Attribute -> Value", "query": "unique inverse lookup (Attribute, Value) -> Entity", "type_signature": "(WorldState, Attribute, Value) -> Entity", "gold_source": "explicit mapping"},
        "material": {"source": "byte-identical C031 material not previously behavior-unblinded", "case_count": len(rows), "partitions": ["discovery", "confirmation", "holdout"], "profiles_per_partition": 8, "attributes": list(scaffold.ATTRIBUTES), "panels": list(scaffold.PANELS), "surfaces": list(scaffold.SURFACES), "candidate_orders": list(scaffold.CANDIDATE_ORDERS), "binding_states": list(scaffold.BINDING_STATES), "material_sha256": sha(MATERIAL), "naturalness_sha256": sha(NATURALNESS), "semantic_reaudit_sha256": sha(MACHINE), "limitation": "controlled-English audit without an independent human rating panel"},
        "model": {"id": "qwen3-4b", "dtype": "FP16", "device": "CUDA", "quantization": False, "other_models_authorized": False, "one_formal_run_per_model_phase": True},
        "zero_models": zero,
        "execution_compiler": {"arms": list(COMPILER_ARMS), "candidate_arms": list(COMPILER_CANDIDATES), "selection_priority": list(COMPILER_PRIORITY), "left_global_is_non_authorizing_baseline": True, "numeric_thresholds": NUMERIC_THRESHOLDS, "selection_rule": "at least two candidate arms pass; choose the first passing arm in frozen priority; freeze its tau before behavior"},
        "behavior": {"thresholds": scaffold.THRESHOLDS, "candidate_cases": 6912, "list_free_generation_cases": 1536, "hidden_states_forbidden": True},
        "hidden": {"events": ["record_slot0_entity", "record_slot0_value", "query_clause_end", "user_answer_cue_end", "assistant_answer_boundary"], "primary_events": ["user_answer_cue_end", "assistant_answer_boundary"], "depths": list(range(37)), "measurements": ["normalized residual response magnitude", "signed candidate-identity logit-lens response"], "identity_contrast": "active-state1 gold logit minus active-state0 gold logit, evaluated for every panel using the paired active identities", "selection": "earliest adjacent discovery depths jointly passing norm and identity gates per primary event", "transfer": "frozen depths on confirmation and holdout", "thresholds": HIDDEN_THRESHOLDS},
        "causal": {"intervention": "patch state1 donor residual into state0 at each frozen primary event/depth", "controls": ["matched-null donor", "wrong-entity donor", "wrong-attribute donor", "neutral no-patch"], "readouts": ["signed candidate margin gain", "answer identity", "natural behavior retention"], "thresholds": CAUSAL_THRESHOLDS, "discovery_forbidden": True, "confirmation_and_holdout_only": True},
        "branches": {"phase1299_pass": "authorize_phase1300_compiler_competition_only", "phase1300_fail": "close_c032_without_semantic_run", "phase1300_pass": "freeze_runtime_and_authorize_phase1301_behavior_only", "phase1301_fail": "close_c032_without_hidden", "phase1301_pass": "authorize_phase1302_event_identity_hidden_only", "phase1302_fail": "close_c032_without_causal_claim", "phase1302_pass": "authorize_phase1303_frozen_causal_rescue", "phase1303_fail": "close_c032_with_descriptive_event_candidate_only", "phase1303_pass": "complete_c032_qwen_single_model_mechanism_closure"},
        "freeze_rules": ["No C032 model weight loads before this contract and independent audit pass.", "No behavior runs before compiler qualification; no hidden states before behavior qualification.", "No object, material, split, model, zero model, compiler arm, priority, threshold, event, parser, control, or branch may change after freeze.", "After unblinding only the registered branch may run; every failure ends C032.", "No threshold relaxation, arm deletion, event reselection, prompt repair, rerun, or other-model vote."],
        "dependencies": {"c031_final": sha(PARENT_FINAL), "c031_audit": sha(PARENT_AUDIT), "source_material": sha(SOURCE_MATERIAL)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    print(canonical({"cases": len(rows), "reaudit": audit, "digest": protocol["protocol_digest"]}))


def finalize() -> None:
    p, a = load(PROTOCOL), load(INDEPENDENT)
    if not a.get("all_checks_passed"): raise RuntimeError("independent audit failed")
    final = {"phase": PHASE, "campaign": CAMPAIGN, "verdict": "c032_full_campaign_contract_frozen", "protocol_digest": p["protocol_digest"], "audit_passed": True, "model_weights_loaded": False, "authorization": "phase1300_compiler_competition_only"}
    save(FINAL, final); print(canonical(final))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=("build", "finalize")); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    build(args.force) if args.command == "build" else finalize()
