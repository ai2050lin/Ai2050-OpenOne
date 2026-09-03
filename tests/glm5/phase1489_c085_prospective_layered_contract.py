#!/usr/bin/env python3
"""Phase1489: freeze C085 fresh-material layered replication contract."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1488_c084_layered_observation_major_stage_closure"
PREDICTIONS = RESULT / "phase1487_c084_joint_synthesis_and_prediction_freeze/frozen/future_prediction_manifest.json"
OUT = RESULT / "phase1489_c085_prospective_layered_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as c072
import phase1460_c078_colon_label_contract as base
import phase1463_c079_aggregate_observation_contract as c079
import phase1469_c080_balanced_interaction_contract as c080
import phase1472_c081_validated_interface_contract as c081
import phase1479_c083_fresh_validation_contract as c083
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1489, "C085"
FAMILIES = {
    "Forum": ("Norris", "Owens", "Palmer", "Payne", "Pearson", "Perry"),
    "Coalition": ("Peters", "Pierce", "Porter", "Powell", "Pratt", "Price"),
    "Community": ("Ramsey", "Reed", "Reeves", "Rhodes", "Rice", "Richards"),
    "Federation": ("Robbins", "Roberts", "Rodgers", "Ross", "Rowe", "Sanders"),
    "Consortium": ("Schmidt", "Shaw", "Sims", "Snyder", "Steele", "Stevens"),
    "Chamber": ("Walker", "Sutton", "Tucker", "Tate", "Walsh", "Terry"),
}


def configure() -> None:
    base.FAMILIES = FAMILIES
    base.ORDER = tuple(FAMILIES)


def prior_words() -> set[str]:
    old = set(c072.old_material_words())
    for source in (base.FAMILIES, c079.FAMILIES, c080.FAMILIES, c081.FAMILIES, c083.FAMILIES):
        if source is FAMILIES:
            continue
        old |= {value.lower() for value in source}
        old |= {value.lower() for values in source.values() for value in values}
    return old


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1489 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    predictions = core.load(PREDICTIONS)
    if parent["authorization"] != "preregister_c085_prospective_layered_replication_and_diagnostic_capture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1488 did not authorize C085")
    configure()
    tok = tokenizer()
    rows = base.active_cases()
    compiled = base.c077.compile_rows(tok, rows)
    sets = base.composition_sets(rows)
    groups = set(FAMILIES)
    people = {value for values in FAMILIES.values() for value in values}
    relation_words = {value for relation in base.RELATIONS.values() for value in relation.values()}
    old = prior_words()
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
        "active": len(rows) == 3456 and Counter(row["surface"] for row in rows) == {surface: 1728 for surface in base.SURFACES},
        "composition": len(sets) == 216 and Counter(row["partition"] for row in sets) == {name: 72 for name in base.PARTITIONS},
        "truth_balance": Counter(truth) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in rows),
        "same_shape": all(len(value) == 1 for value in lengths.values()),
        "stable_roles": all(len(value) == 1 for value in signatures.values()),
        "role_singletons": all(all(len(row["role_positions"][role]) == 1 for role in base.ROLE_SLOTS) for row in compiled),
        "machine_naturalness": all(row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in rows),
        "zero_models": all(value == 0.5 for key, value in zero.items() if key != "identity_oracle") and zero["identity_oracle"] == 1.0,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/active_cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/composition_sets.jsonl", sets)
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c085.fresh_layered.v1", "families": FAMILIES, "relations": base.RELATIONS, "partitions": {key: list(value) for key, value in base.PARTITIONS.items()}, "surfaces": base.SURFACES})
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "zero_models": zero, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "semantic_scope": "explicit relation-label identity with orthogonal entity/object nuisances", "naturalness_scope": "machine-audited controlled English; no independent human blind review", "hidden_state_accessed": False}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    p084 = {row["id"]: row for row in predictions["predictions"]}
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c085.prospective_layered_replication.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "prospective fresh-material test of P084 with behavior success/mixed/failed strata captured before internal unblinding",
        "roles": list(base.ROLE_SLOTS), "relations": list(base.RELATIONS), "surfaces": list(base.SURFACES), "cells": list(base.CELLS), "partitions": list(base.PARTITIONS),
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "coordinate changes", "post-unblind material changes", "post-unblind threshold changes"],
        "material": {"active_count": 3456, "composition_count": 216, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior_strata": {"unit": "frozen sixteen-case composition set", "success": "16/16 correct", "mixed": "1-15/16 correct", "failed": "0/16 correct", "role": "stratification rather than campaign-wide hard stop"},
        "capture": {"scope": "all 3456 cases regardless of behavior", "state_count": 37, "role_slot_count": 9, "hidden_dimension": 2560, "dtype": "float16", "raw_format": "numpy NPY memmap plus JSONL index", "no_pooling": True, "no_coordinate_selection": True},
        "factorial": {"effects": ["relation", "entity", "object", "relation_entity", "relation_object", "entity_object", "relation_entity_object"], "effect_formula": "complete 2x2x2 orthogonal contrast over every frozen composition set"},
        "p084": {"freeze_sha256": predictions["freeze_sha256"], "predictions": p084, "formal_evidence_stratum": "success", "minimum_sets_per_relation_split_surface": 3, "missing_success_panel": "M2 not tested; never imputed as pass", "mixed_failed_role": "diagnostic divergence only"},
        "route": ["phase1490 behavior and strata", "phase1491 all-case field capture", "phase1492 stratified factorial atlas", "phase1493 prospective P084 adjudication", "phase1494 stratum diagnostics", "phase1495 closure"],
        "stop_rule": "only integrity, nonfinite execution, or unauthorized contract mutation stops the campaign; missing behavior strata are registered M2 and remaining authorized branches continue",
        "claim_boundary": {"allowed": "fresh-material replication or failure of a Qwen3 controlled explicit-label trajectory regularity", "forbidden": ["natural relation semantics", "causal mechanism", "semantic neurons", "cross-model law", "new mathematics"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1490_c085_behavior_stratification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/adjudication_of_uploaded_analyses.json", {
        "retain": ["C084 was retrospective deep mining with no new model run", "state0 anti-cosine is mainly six-way contrast geometry", "the 17 coordinates are a threshold slice", "P084 requires fresh prospective evaluation"],
        "correct": ["state22 is the first frozen threshold crossing, not onset of information", "C084 views are complementary analyses of one dataset, not independent replications", "98-99 percent is factorial coefficient energy, not output variance or causal share", "RDC and new mathematics remain unconfirmed"],
    })
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "prediction_freeze_sha256": predictions["freeze_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
