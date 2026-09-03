#!/usr/bin/env python3
"""Phase1472: preregister the one-shot C081 validated-interface rescue."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as c072
import phase1460_c078_colon_label_contract as c078
import phase1463_c079_aggregate_observation_contract as c079
import phase1469_c080_balanced_interaction_contract as base
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1472, "C081"
PARENT = TESTS / "result/phase1471_c080_behavior_gate_closure"
OUT = TESTS / "result/phase1472_c081_validated_interface_contract"
FAMILIES = {
    "Foundation": ("Adams", "Barnes", "Bell", "Brooks", "Butler", "Campbell"),
    "Institute": ("Chambers", "Chapman", "Coleman", "Collins", "Cook", "Crawford"),
    "League": ("Cruz", "Daniels", "Dawson", "Ferguson", "Fisher", "Fleming"),
    "Network": ("Ford", "Foster", "Freeman", "Gardner", "Gibbs", "Gibson"),
    "Republic": ("Gill", "Greene", "Hale", "Hall", "Hamilton", "Hammond"),
    "Trust": ("Hansen", "Hardy", "Harris", "Hart", "Hawkins", "Hayes"),
}
SURFACES = {
    "a_validated": (
        "First relation label: {record_label}. First clause: {record_target} saw {record_object}. "
        "Second relation label: {query_label}. Second clause: {query_target} may see {query_object}. "
        "Are the two relation labels identical? Answer only yes or no."
    ),
    "b_validated": (
        "Recorded label: {record_label}. Recorded fact: {record_target} saw {record_object}. "
        "Queried label: {query_label}. Queried possibility: {query_target} can see {query_object}. "
        "Are the recorded and queried labels identical? Answer only yes or no."
    ),
}


def configure() -> None:
    base.FAMILIES = FAMILIES
    base.ORDER = tuple(FAMILIES)
    base.EXPLICIT_SURFACES = SURFACES


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1472 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c081_historically_validated_interface_rescue_on_fresh_material" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1471 did not authorize C081")
    configure()
    tok = tokenizer()
    rows = base.active_cases("explicit")
    compiled = base.compile_rows(tok, rows, base.EXPLICIT_SYSTEM, base.EXPLICIT_ROLES)
    sets = base.interaction_sets(rows, "explicit")
    branch_checks = base.branch_checks(tok, rows, compiled, sets, SURFACES, base.EXPLICIT_ROLES)
    old = set(c072.old_material_words())
    for source in (c078.FAMILIES, c079.FAMILIES):
        old |= {value.lower() for value in source}
        old |= {value.lower() for values in source.values() for value in values}
    c080_graph = core.load(TESTS / "result/phase1469_c080_balanced_interaction_contract/material/frozen_language_graph.json")
    old |= {value.lower() for value in c080_graph["families"]}
    old |= {value.lower() for values in c080_graph["families"].values() for value in values}
    groups = set(FAMILIES)
    people = {value for values in FAMILIES.values() for value in values}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "one_shot_rescue": parent["rescue_limit"].startswith("one fresh-material rescue"),
        "fresh_groups": len(groups) == 6 and not ({value.lower() for value in groups} & old),
        "fresh_people": len(people) == 36 and not ({value.lower() for value in people} & old),
        "singletons": all(len(tok.encode(" " + value, add_special_tokens=False)) == 1 for value in groups | people | {"saw", "see"} | {item["label"] for item in base.RELATIONS.values()}),
        "historical_surface_a": SURFACES["a_validated"].startswith("First relation label:") and "Are the two relation labels identical?" in SURFACES["a_validated"],
        "historical_surface_b": SURFACES["b_validated"].startswith("Recorded label:") and "Are the recorded and queried labels identical?" in SURFACES["b_validated"],
        "branch": all(branch_checks.values()),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/active_cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/interaction_sets.jsonl", sets)
    core.save(OUT / "material/frozen_language_graph.json", {
        "schema": "c081.validated_interface_rescue.v1",
        "families": FAMILIES,
        "relations": base.RELATIONS,
        "pair_ids": list(base.PAIR_IDS),
        "partitions": {key: list(value) for key, value in base.PARTITIONS.items()},
        "surfaces": SURFACES,
    })
    preaudit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "branch_checks": branch_checks,
        "zero_models": base.zero_models(rows),
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "semantic_scope": "explicit relation-label equality on one-shot fresh material using historically qualified interface families",
        "naturalness_scope": "machine-audited controlled English; no human blind review",
        "hidden_state_accessed": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    c080 = core.load(TESTS / "result/phase1469_c080_balanced_interaction_contract/protocol/preregistration.json")
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c081.one_shot_validated_interface_balanced_interaction.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": c080["research_object"],
        "interaction_formula": c080["interaction_formula"],
        "off_diagonal_null_formula": c080["off_diagonal_null_formula"],
        "allowed_observables": c080["allowed_observables"],
        "forbidden": c080["forbidden"],
        "roles": list(base.EXPLICIT_ROLES),
        "surfaces": list(SURFACES),
        "partitions": list(base.PARTITIONS),
        "pair_ids": list(base.PAIR_IDS),
        "nuisance_cells": list(base.NUISANCE_CELLS),
        "pair_corners": list(base.PAIR_CORNERS),
        "material": {
            "active_count": len(rows),
            "interaction_set_count": len(sets),
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
            "sets_sha256": core.sha(OUT / "material/interaction_sets.jsonl"),
            "human_naturalness_lock": False,
        },
        "behavior": c080["branches"]["explicit"]["behavior"],
        "capture": {
            **c080["capture"],
            "role_slot_count": len(base.EXPLICIT_ROLES),
        },
        "observation": c080["explicit_observation"],
        "stop_rules": [
            "behavior failure permanently closes the explicit-label equality-interaction route",
            "only behavior-perfect 32-case sets may enter Hidden-State capture",
            "discovery freezes full-vector candidates before confirmation or lockbox access",
            "candidate failures close candidates individually; no threshold or material changes after unblinding",
        ],
        "claim_boundary": {
            "allowed": "full-dimensional equality-interaction trajectory regularities in behavior-correct Qwen3 controlled-label cases",
            "forbidden": ["natural-verb relation mechanism", "semantic neurons", "causal use", "cross-model invariance", "attention/MLP/parameter claims", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1473_c081_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
