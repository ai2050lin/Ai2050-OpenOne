#!/usr/bin/env python3
"""C180: identifiable reachable-target choice across eight natural relation families."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C179 = RESULT / "phase1713_c179_visible_codebook_natural_ecology"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1712_c178_natural_knowledge_ecology as base

PHASE, CAMPAIGN = 1714, "C180"


def make_case(family, phrase, unit, nodes, surface, order):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    if order == 1:
        options, gold = f"(A) {d} (B) {f}", 0
    else:
        options, gold = f"(A) {f} (B) {d}", 1
    if surface == 1:
        prompt = f"Facts: {facts} Following only the stated '{phrase}' links, which target is reachable from {a}? {options}. Reply with only A or B."
    else:
        prompt = f"Registry notes: {facts} Start at {a} and follow only '{phrase}' arrows. Which registered target can be reached? {options}. Reply with only A or B."
    return {"case_id": "", "family": family, "phrase": phrase, "unit": unit, "unit_id": f"c180-{family}-{unit:02d}", "partition": base.partition(unit), "truth": 1, "surface": surface, "codebook": order, "gold_position": gold, "prompt": prompt, "semantic_edges": edges, "nodes": list(nodes), "intended": d, "alternative": f, "role_values": {"primary": a, "secondary": d, "relation": phrase, "context": b, "query": a}}


def material():
    cases = []
    for family, (phrase, units) in base.RELATIONS.items():
        for unit, nodes in enumerate(units):
            for surface, order in itertools.product((1, -1), repeat=2):
                row = make_case(family, phrase, unit, nodes, surface, order)
                row["case_id"] = f"c180-{len(cases):04d}"
                cases.append(row)
    return cases


def configure():
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    base.OUT = OUT
    base.material = material
    base.make_case = make_case


def contract():
    configure()
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C179 / "audit/independent_final_audit.json")
    cases = material()
    compiled = base.compile_rows(graph_base.tokenizer(), cases)
    cells = {(r["family"], r["unit"], r["surface"], r["codebook"]) for r in cases}
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C180_reachable_target_choice_ecology_new_contract",
        "cases": len(cases) == 192,
        "families": len(base.RELATIONS) == 8,
        "cells": len(cells) == 192,
        "partitions": all(sum(r["partition"] == p for r in cases) == 64 for p in base.PARTITIONS),
        "candidate_balance": float(np.mean([r["gold_position"] == 0 for r in cases])) == 0.5,
        "unique_reachable": all(r["intended"] != r["alternative"] and (r["nodes"][2], r["nodes"][3]) in r["semantic_edges"] and all(r["alternative"] not in edge for edge in r["semantic_edges"][:3]) for r in cases),
        "roles": all(set(r["role_positions"]) == set(base.ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) < base.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "reachable_target_choice_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "object": "choose the unique reachable concrete target from one three-edge chain and one disconnected distractor edge",
        "families": list(base.RELATIONS),
        "cases": len(cases),
        "behavior_gates": {"global_min": 0.80, "family_partition_min": 0.75, "all_frozen_anchors_correct": True},
        "zero_models": {"always_A": 0.5, "always_B": 0.5, "always_reachable": "not applicable because both options name concrete targets"},
        "hidden_policy": "no HiddenState capture or perturbation before behavior eligibility lock",
        "eligible_capture": "embedding, 36 block outputs, final norm; six roles for eligible rows; all tokens for 24 frozen anchors",
        "response": "q24 source-role locked 64 coordinates to q25 six-role full field, symmetric finite difference",
        "source_roles": ["primary", "query", "relation"],
        "claim_boundary": "explicit registry path selection across natural relation phrases; not unrestricted world knowledge",
        "forbidden": ["attention", "MLP", "weights", "PCA", "hidden capture before behavior lock"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C180_behavior_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(r["prompt_ids"]) for r in compiled)}, indent=2))


def behavior():
    base.behavior()
    rows = core.rows(OUT / "raw/behavior_index.jsonl")
    logits = np.load(OUT / "raw/behavior_logits.float32.npy", mmap_mode="r")
    previous = core.load(OUT / "analysis/behavior_run.json")
    checks = {"rows": len(rows) == 192 and list(logits.shape) == [192, 2], "finite": bool(np.isfinite(logits).all()), "bf16": previous["checks"]["bf16"]}
    protocol = core.load(OUT / "protocol/preregistration.json")
    protocol["adapter_correction"] = "C178 fixed count 384 replaced by the already frozen C180 contract count 192 before behavior reveal"
    protocol["producer_sha256"] = core.sha(Path(__file__))
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": previous["runtime"], "adapter_correction": "C178 fixed count 384 replaced by C180 contract count 192"})
    core.save(OUT / "audit/internal_behavior_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"corrected_checks": checks}, indent=2))


def analyze():
    base.analyze()
    path = OUT / "analysis/natural_ecology_atlas.json"
    report = core.load(path)
    report["next_authorization"] = "run_C181_cross_model_functional_eligibility_then_C182_synthesis_heatmap"
    core.save(path, report)


def main():
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "behavior": behavior, "lock": base.lock, "hidden": base.hidden, "analyze": analyze, "close": base.close}[args.command]()


if __name__ == "__main__":
    main()
