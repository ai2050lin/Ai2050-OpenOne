#!/usr/bin/env python3
"""Independent audit for Phase1445 C074 preregistration."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1445, "C074"
OUT = TESTS / "result/phase1445_c074_directional_domain_contract"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def ba(truths: list[bool], predictions: list[bool]) -> float:
    positive = [pred for truth, pred in zip(truths, predictions) if truth]
    negative = [pred for truth, pred in zip(truths, predictions) if not truth]
    return 0.5 * (sum(positive) / len(positive) + sum(not value for value in negative) / len(negative))


def main() -> None:
    active = core.rows(OUT / "material/active_cases.jsonl")
    composition = core.rows(OUT / "material/composition_sets.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3_active.jsonl")
    graph = core.load(OUT / "material/frozen_concept_graph.json")
    preaudit = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    truths = [row["truth"] for row in active]
    zero = {
        "person": ba(truths, [row["record_target"] == row["query_target"] for row in active]),
        "group": ba(truths, [row["record_family"] == row["query_family"] for row in active]),
        "exact": ba(truths, [row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] for row in active]),
        "surface": ba(truths, [row["surface"] == protocol["surfaces"][0] for row in active]),
    }
    routes = protocol["routes"]
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "families": len(graph["families"]) == 6 and all(len(values) == 12 for values in graph["families"].values()),
        "active": len(active) == 5760 and Counter(row["surface"] for row in active) == {surface: 1440 for surface in protocol["surfaces"]},
        "truth": Counter(truths) == {True: 1440, False: 4320},
        "semantic": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "naturalness": all(row["prompt"].count("?") == 1 and row["prompt"].endswith("yes or no.") for row in active),
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {"response_discovery": 24, "confirmation": 24, "lockbox": 24},
        "compiled": len(compiled) == 5760 and all(len(row["role_positions"]) == 5 for row in compiled),
        "role_singletons": all(len({row["role_positions"][role][0] for role in ROLES}) == 4 for row in compiled),
        "routes": len(routes) == 16 and {(value["source"], value["target"]) for value in routes.values()} == {(source, target) for source in protocol["surfaces"] for target in protocol["surfaces"]},
        "route_metadata": all(value["same_surface"] == (value["source"] == value["target"]) for value in routes.values()),
        "zero_models": abs(zero["person"] - 5 / 6) < 1e-12 and abs(zero["group"] - 5 / 6) < 1e-12 and zero["exact"] == 1.0 and zero["surface"] == 0.5,
        "hashes": protocol["material"]["active_sha256"] == core.sha(OUT / "material/active_cases.jsonl") and protocol["material"]["composition_sha256"] == core.sha(OUT / "material/composition_sets.jsonl"),
        "fixed_object": protocol["state_index"] == 16 and protocol["model"] == "qwen3-bfloat16-cuda-no-quantization",
        "identity_only": protocol["camera"]["arms"] == protocol["domain"]["arms"] == ["self", "correct_identity", "wrong_identity"],
        "independent_edges": protocol["domain"]["edge_classes"]["robust"].startswith("confirmation and lockbox"),
        "forbidden": all(term in protocol["forbidden"] for term in ("attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "layer search", "coordinate search", "semantic permutation competition")),
        "hidden_not_accessed": preaudit["checks"]["hidden_not_accessed"],
        "authorization": final["authorization"] == "run_phase1446_c074_behavior" and final["contract_sha256"] == protocol["contract_sha256"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
