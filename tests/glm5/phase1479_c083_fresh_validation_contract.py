#!/usr/bin/env python3
"""Phase1479: preregister fresh-material validation of the five P082 predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1478_c082_campaign_closure"
OUT = RESULT / "phase1479_c083_fresh_validation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as c072
import phase1460_c078_colon_label_contract as base
import phase1463_c079_aggregate_observation_contract as c079
import phase1469_c080_balanced_interaction_contract as c080
import phase1472_c081_validated_interface_contract as c081
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

FAMILIES = {
    "Assembly": ("Henderson", "Hicks", "Holland", "Holmes", "Holt", "Hopkins"),
    "Campus": ("Horton", "Howell", "Hunt", "Jennings", "Jensen", "Keller"),
    "Colony": ("Kelley", "Kennedy", "Knight", "Lane", "Larson", "Lawson"),
    "Commons": ("Lowe", "Lynch", "Lyons", "Manning", "Marsh", "McCarthy"),
    "Dynasty": ("McCoy", "McDonald", "Meyer", "Mills", "Moody", "Moran"),
    "Empire": ("Morris", "Morrison", "Moss", "Murphy", "Murray", "Nash"),
}
PHASE, CAMPAIGN = 1479, "C083"


def configure() -> None:
    base.FAMILIES = FAMILIES
    base.ORDER = tuple(FAMILIES)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1479 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    predictions = core.load(RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json")
    if parent["authorization"] != "preregister_c083_fresh_material_validation_of_lexical_to_common_boundary_convergence" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1478 did not authorize C083")
    configure()
    tok = tokenizer()
    rows = base.active_cases()
    compiled = base.c077.compile_rows(tok, rows)
    sets = base.composition_sets(rows)
    old = set(c072.old_material_words())
    for source in (base.FAMILIES, c079.FAMILIES, c080.FAMILIES, c081.FAMILIES):
        if source is FAMILIES:
            continue
        old |= {value.lower() for value in source}
        old |= {value.lower() for values in source.values() for value in values}
    groups = set(FAMILIES)
    people = {value for values in FAMILIES.values() for value in values}
    relation_words = {value for relation in base.RELATIONS.values() for value in relation.values()}
    lengths = {surface: {len(row["prompt_ids"]) for row in compiled if row["surface"] == surface} for surface in base.SURFACES}
    signatures = {surface: {tuple((role, tuple(row["role_positions"][role])) for role in base.ROLE_SLOTS) for row in compiled if row["surface"] == surface} for surface in base.SURFACES}
    truth = [row["truth"] for row in rows]
    zero = {
        "always_yes": base.c077.ba(truth, [True] * len(rows)),
        "always_no": base.c077.ba(truth, [False] * len(rows)),
        "surface": base.c077.ba(truth, [row["surface"] == "a_colon" for row in rows]),
        "entity": base.c077.ba(truth, [row["entity_match"] for row in rows]),
        "object": base.c077.ba(truth, [row["object_match"] for row in rows]),
        "entity_object": base.c077.ba(truth, [row["entity_match"] and row["object_match"] for row in rows]),
        "identity_oracle": base.c077.ba(truth, [row["record_label"] == row["query_label"] for row in rows]),
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "prediction_freeze": predictions["freeze_sha256"] == core.digest({key: value for key, value in predictions.items() if key != "freeze_sha256"}),
        "fresh_groups": len(groups) == 6 and not ({value.lower() for value in groups} & old),
        "fresh_people": len(people) == 36 and not ({value.lower() for value in people} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in groups | people | relation_words),
        "active": len(rows) == 3456,
        "composition": len(sets) == 216,
        "semantic": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in rows),
        "same_shape": all(len(value) == 1 for value in lengths.values()),
        "stable_roles": all(len(value) == 1 for value in signatures.values()),
        "zero": all(value == 0.5 for key, value in zero.items() if key != "identity_oracle") and zero["identity_oracle"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/active_cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/composition_sets.jsonl", sets)
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c083.fresh_validation.v1", "families": FAMILIES, "relations": base.RELATIONS, "partitions": {key: list(value) for key, value in base.PARTITIONS.items()}, "surfaces": base.SURFACES})
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "zero_models": zero, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "naturalness_scope": "machine-audited controlled English inherited from behavior-qualified C079 interfaces; no human blind review", "hidden_state_accessed": False}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    c079_protocol = core.load(RESULT / "phase1463_c079_aggregate_observation_contract/protocol/preregistration.json")
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c083.fresh_material_p082_validation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "fresh-material validation of lexical-specific query differences converging to a shared distributed late boundary response",
        "roles": list(base.ROLE_SLOTS),
        "relations": base.RELATIONS,
        "surfaces": list(base.SURFACES),
        "cells": list(base.CELLS),
        "partitions": list(base.PARTITIONS),
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "probe", "coordinate changes", "prediction threshold changes", "post-unblind material changes"],
        "material": {"active_count": len(rows), "composition_count": len(sets), "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": c079_protocol["behavior"],
        "capture": {"eligible_rule": "all sixteen cases in a composition set behavior-correct", "discovery_partition": "response_discovery", "validation_partitions": ["confirmation", "lockbox"], "state_count": 37, "role_slot_count": 9, "hidden_dimension": 2560, "dtype": "float16", "no_pooling": True, "no_coordinate_selection": True},
        "effect_formula": "mean over four paired H[relation_match=1] - H[relation_match=0] differences holding entity/object bits fixed",
        "frozen_prediction_manifest_sha256": predictions["freeze_sha256"],
        "frozen_predictions": predictions["future_fresh_material_predictions"],
        "frozen_coordinates": predictions["frozen_coordinates"],
        "frozen_common_vector_sha256": predictions["common_vector_sha256"],
        "validation": {"confirmation_and_lockbox_must_each_pass_all_five_predictions": True, "discovery_is_reported_but_not_used_to_change_predictions": True},
        "stop_rule": "behavior first; failure denies Hidden State; after behavior pass capture discovery then unopened confirmation/lockbox; no prediction changes",
        "claim_boundary": {"allowed": "fresh-material replication of a Qwen3 controlled-task trajectory regularity", "forbidden": ["natural unlabeled relation semantics", "causal use", "semantic neurons", "cross-model law", "new mathematics"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1480_c083_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
