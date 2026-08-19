#!/usr/bin/env python3
"""Phase1304: freeze C033 role-typed answer-aggregator causal contract."""
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
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1292_c029_object_attribute_convergence_contract as scaffold  # noqa: E402
import phase1294_c030_grounded_lookup_contract as c030  # noqa: E402
import phase1297_c031_event_interval_contract as c031  # noqa: E402

PHASE = 1304
CAMPAIGN = "C033"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1304_c033_role_typed_causal_graph_contract_audit.py"
OUT = T / "result/phase1304_c033_role_typed_causal_graph_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENV = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_role_typed_lookup_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

C032_FINAL = T / "result/phase1303_c032_frozen_causal_rescue/analysis/final.json"
C032_AUDIT = T / "result/phase1303_c032_frozen_causal_rescue/audit/independent_final_audit.json"
C032_RUNTIME = T / "result/phase1300_c032_execution_compiler_competition/protocol/frozen_runtime.json"

VALUE_BANKS = {
    "discovery": {
        "color": ("red", "blue", "green"),
        "material": ("wood", "iron", "nylon"),
        "location": ("kitchen", "library", "bedroom"),
        "size": ("small", "large", "huge"),
        "shape": ("oval", "star", "cross"),
        "status": ("active", "inactive", "open"),
    },
    "confirmation": {
        "color": ("yellow", "purple", "brown"),
        "material": ("linen", "clay", "quartz"),
        "location": ("cabinet", "storeroom", "laboratory"),
        "size": ("long", "wide", "thick"),
        "shape": ("heart", "cube", "cone"),
        "status": ("closed", "ready", "waiting"),
    },
    "holdout": {
        "color": ("cream", "coral", "plum"),
        "material": ("foam", "cork", "plaster"),
        "location": ("closet", "shed", "attic"),
        "size": ("thin", "tall", "low"),
        "shape": ("disk", "prism", "pyramid"),
        "status": ("complete", "incomplete", "archived"),
    },
}

BEHAVIOR_TH = dict(scaffold.THRESHOLDS)
HIDDEN_TH = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.99,
    "active_norm_median_min": 0.001,
    "active_to_max_control_ratio_min": 1.10,
    "active_over_controls_fraction_min": 0.70,
    "identity_positive_fraction_min": 0.70,
    "identity_to_max_control_ratio_min": 1.05,
}
SWAP_TH = {
    "finite_fraction_min": 1.0,
    "direction_partition_accuracy_min": 0.75,
    "signed_margin_gain_median_min": 0.5,
    "correct_over_matched_null_ratio_min": 1.25,
    "pairwise_correct_win_fraction_min": 0.75,
    "natural_retention_min": 0.99,
}
RESCUE_TH = {
    "finite_fraction_min": 1.0,
    "baseline_accuracy_min": 0.99,
    "blocked_target_identity_accuracy_max": 0.30,
    "cross_surface_rescue_accuracy_min": 0.70,
    "cross_surface_rescue_recovery_fraction_median_min": 0.70,
    "cross_surface_over_null_margin_ratio_min": 1.25,
    "pairwise_rescue_win_fraction_min": 0.75,
    "natural_retention_min": 0.99,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def prior_name_sets() -> dict[str, set[str]]:
    paths = {
        "c029": T / "result/phase1292_c029_object_attribute_convergence_contract/audit/tokenizer_semantic_program_audit.json",
        "c030": T / "result/phase1294_c030_grounded_lookup_contract/audit/tokenizer_semantic_program_audit.json",
        "c031_c032": T / "result/phase1297_c031_event_interval_contract/audit/tokenizer_semantic_program_audit.json",
    }
    return {name: set(load(path)["token_audit"]["selected_names"]) for name, path in paths.items()}


def prior_values() -> set[str]:
    return {
        value
        for banks in (scaffold.VALUE_BANKS, c030.VALUE_BANKS, c031.VALUE_BANKS)
        for partition in banks.values()
        for values in partition.values()
        for value in values
    }


def select_names(tokenizer: Any) -> tuple[str, ...]:
    used = set().union(*prior_name_sets().values())
    eligible = [
        name
        for name in dict.fromkeys(c031.NAME_CANDIDATES)
        if name not in used and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    needed = len(scaffold.PARTITIONS) * scaffold.PROFILES_PER_PARTITION * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} unused single-token names; need {needed}")
    return tuple(eligible[:needed])


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = select_names(tokenizer)
    original = (scaffold.VALUE_BANKS, scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause)
    try:
        scaffold.VALUE_BANKS = VALUE_BANKS
        scaffold.NAME_POOL = names
        scaffold.record_clause = c030.corrected_record
        scaffold.query_clause = c030.corrected_query
        material, token_audit = scaffold.build_cases(tokenizer)
    finally:
        scaffold.VALUE_BANKS, scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause = original
    for row in material:
        row["schema_version"] = "phase1304.c033.case.v1"
        row["case_id"] = "c033-" + digest(
            {"group": row["group_id"], "state": row["binding_state"], "prompt": row["candidate_prompt"]}
        )[:20]
    used = set().union(*prior_name_sets().values())
    token_audit.update(
        {
            "selected_names": list(names),
            "prior_name_overlap": sorted(used & set(names)),
            "eligible_unused_single_token_count": sum(
                name not in used and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
                for name in dict.fromkeys(c031.NAME_CANDIDATES)
            ),
        }
    )
    return material, token_audit


def build(force: bool) -> None:
    if load(C032_FINAL).get("authorization") != "close_c032_with_descriptive_path_only":
        raise RuntimeError("C032 is not closed on its frozen failure branch")
    if not load(C032_AUDIT).get("all_checks_passed"):
        raise RuntimeError("C032 final audit did not pass")
    runtime = load(C032_RUNTIME)
    if runtime.get("selected_runtime") != "right_padding" or runtime.get("tau") != 1e-7:
        raise RuntimeError("frozen execution compiler unavailable")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} exists")
    if OUT.exists():
        shutil.rmtree(OUT)

    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    material, token_audit = build_material(tokenizer)
    write_rows(MATERIAL, material)
    review = c030.grammar_review(material)
    review.update(
        {
            "reviewed_before_any_c033_weight_load": True,
            "semantic_uniqueness_recomputed_for_all_cases": all(
                sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1
                for row in material
            ),
            "new_entities_and_values": True,
            "limitation": "Controlled English passed deterministic grammar and semantic review; no independent human panel was available.",
        }
    )
    save(NATURALNESS, review)
    program = scaffold.program_audit(material)
    values = {value for partition in VALUE_BANKS.values() for values in partition.values() for value in values}
    old_values = prior_values()
    machine = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_audit": token_audit,
        "program_audit": program,
        "prior_value_overlap": sorted(old_values & values),
        "semantic_unique": review["semantic_uniqueness_recomputed_for_all_cases"],
        "naturalness_passed": review["all_checks_passed"],
        "all_machine_checks_passed": bool(
            review["all_checks_passed"]
            and review["semantic_uniqueness_recomputed_for_all_cases"]
            and not token_audit["prior_name_overlap"]
            and not (old_values & values)
            and token_audit["all_candidates_single_token"]
            and program["shortcut_ceiling"] <= BEHAVIOR_TH["shortcut_program_accuracy_max"]
            and program["active_same_bag_different_gold_pairs"] == program["active_pair_count"]
        ),
    }
    save(MACHINE, machine)
    save(
        ENV,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "tokenizer_only": True,
            "model_weights_loaded": False,
        },
    )

    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1304.c033.role_typed_contract.v1",
        "purpose": "test whether the Qwen3 answer-boundary depth-26 state is bidirectionally sufficient and supports cross-surface block-rescue on independent material",
        "adjudication": {
            "accepted": [
                "C032 established a repeatable answer-boundary response and a strong local full-residual substitution effect",
                "C032 formal four-cell gate failed and remains closed",
                "readability, causal modulation, sufficiency, contrast-necessity, rescue, and minimality require separate claims",
            ],
            "corrected_overclaims": [
                "answer-boundary depth 25-26 is an aggregator candidate, not a demonstrated bottleneck",
                "100 percent substitution at depth 26 is local sufficiency, not natural necessity or a pure semantic variable",
                "user-cue depth 29-30 and answer-boundary depth 25-26 do not form one serial path in the Transformer DAG",
                "C032 did not show that question understanding and answer formation are separate mechanisms",
                "physical-layout attribution is limited to the frozen Qwen3 FP16 runtime and materials",
            ],
        },
        "construct": {
            "world_state": "finite explicit map Entity x Attribute -> Value",
            "query": "unique inverse lookup (Attribute, Value) -> Entity",
            "type_signature": "(WorldState, Attribute, Value) -> Entity",
            "requested_operation": "direct lookup only; no explicit transformation instruction",
            "gold_source": "explicit mapping independently recomputed",
        },
        "role_typed_graph": {
            "node_type": "(token event, residual depth)",
            "edge_rule": "(p,l)->(q,m) only if p<=q and l<m",
            "aggregator_candidate": {"event": "assistant_answer_boundary", "depth": 26},
            "block_site": {"event": "assistant_answer_boundary", "depth": 25},
            "rescue_site": {"event": "assistant_answer_boundary", "depth": 26},
            "user_cue_role": "not tested in C033; depth29 cannot mediate answer-boundary depth25-26",
            "source_and_mediator_claims": "out of scope",
        },
        "material": {
            "partitions": list(scaffold.PARTITIONS),
            "profiles_per_partition": scaffold.PROFILES_PER_PARTITION,
            "attributes": list(scaffold.ATTRIBUTES),
            "panels": list(scaffold.PANELS),
            "surfaces": list(scaffold.SURFACES),
            "candidate_orders": list(scaffold.CANDIDATE_ORDERS),
            "binding_states": list(scaffold.BINDING_STATES),
            "case_count": len(material),
            "independent_profile_count": 24,
            "entity_count": len(token_audit["selected_names"]),
            "prior_entity_overlap": token_audit["prior_name_overlap"],
            "prior_value_overlap": sorted(old_values & values),
            "material_sha256": sha(MATERIAL),
            "naturalness_sha256": sha(NATURALNESS),
        },
        "model": {
            "model_id": "qwen3-4b-fp16-cuda-no-quantization",
            "compiler": "right_padding",
            "formal_runs_per_model_phase": 1,
            "other_models_authorized": False,
            "native_chat_template": True,
            "enable_thinking": False,
        },
        "zero_models": program,
        "behavior_thresholds": BEHAVIOR_TH,
        "hidden": {
            "event": "assistant_answer_boundary",
            "depths": [25, 26],
            "measurements": ["normalized paired residual response", "paired identity logit-lens response"],
            "controls": ["matched_null", "surface_only", "semantic_neighbor"],
            "thresholds": HIDDEN_TH,
            "selection_forbidden": True,
        },
        "bidirectional_swap": {
            "event": "assistant_answer_boundary",
            "depth": 26,
            "directions": ["state0_to_state1", "state1_to_state0"],
            "controls": ["matched_null", "wrong_entity", "wrong_attribute", "neutral", "self_patch"],
            "partitions": ["confirmation", "holdout"],
            "thresholds": SWAP_TH,
            "claim_scope": "bidirectional local sufficiency and paired-contrast dependence; not universal natural necessity",
        },
        "cross_surface_block_rescue": {
            "target_surfaces": ["catalog_prose", "inventory_ledger"],
            "block": "replace target active-state1 answer-boundary depth25 residual with same-target active-state0 residual",
            "rescue": "at depth26 add the opposite-surface active state1-minus-state0 residual difference",
            "controls": ["block_only", "matched_null_cross_surface_delta", "wrong_attribute_cross_surface_delta", "no_block_self_retention"],
            "partitions": ["confirmation", "holdout"],
            "thresholds": RESCUE_TH,
            "claim_scope": "independent across surface carrier, but not across entity world or model",
        },
        "branches": {
            "phase1304_audit_pass": "authorize_phase1305_behavior_only",
            "phase1305_fail": "close_c033_without_hidden",
            "phase1305_pass": "authorize_phase1306_frozen_hidden_only",
            "phase1306_fail": "close_c033_without_causal",
            "phase1306_pass": "authorize_phase1307_bidirectional_swap_only",
            "phase1307_fail": "close_c033_without_rescue",
            "phase1307_pass": "authorize_phase1308_cross_surface_block_rescue_only",
            "phase1308_pass": "close_c033_with_cross_surface_rescue_candidate",
            "phase1308_fail": "close_c033_at_rescue_boundary",
        },
        "freeze_rules": [
            "No Qwen3 weights before Phase1304 independent audit passes.",
            "No hidden state before Phase1305 behavior gate passes.",
            "No causal intervention before Phase1306 frozen hidden gate passes.",
            "No user-cue event, layer scan, head scan, MLP scan, or component selection in C033.",
            "After unblinding, only the registered branch may run; no threshold, material, parser, donor, event, depth, or partition change.",
            "Any failed gate closes C033 at that evidential level.",
        ],
        "dependencies": {
            "c032_final": sha(C032_FINAL),
            "c032_audit": sha(C032_AUDIT),
            "runtime": sha(C032_RUNTIME),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    protocol = {
        **timeless,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": digest(timeless),
    }
    save(PROTOCOL, protocol)
    save(
        FINAL,
        {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "verdict": "pending_independent_zero_model_audit",
            "authorization": "none",
            "protocol_digest": protocol["protocol_digest"],
            "model_weights_loaded": False,
        },
    )
    print(canonical({"cases": len(material), "names": len(token_audit["selected_names"]), "digest": protocol["protocol_digest"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force)
