#!/usr/bin/env python3
"""C179: visible-codebook repair of C178, with the same semantic object and gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1713_c179_visible_codebook_natural_ecology"
C178 = RESULT / "phase1712_c178_natural_knowledge_ecology"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1712_c178_natural_knowledge_ecology as base

PHASE, CAMPAIGN = 1713, "C179"


def make_case(family, phrase, unit, nodes, truth, surface, codebook):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)] if truth == 1 else [(a, b), (c, b), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    options = "(A) yes (B) no" if codebook == 1 else "(A) no (B) yes"
    if surface == 1:
        prompt = f"Facts: {facts} Following only the stated '{phrase}' links, can {a} reach {d}? {options}. Reply with only A or B."
    else:
        prompt = f"Registry notes: {facts} Starting at {a} and following only '{phrase}' arrows, is {d} reachable? {options}. Reply with only A or B."
    yes_position = 0 if codebook == 1 else 1
    gold = yes_position if truth == 1 else 1 - yes_position
    return {"case_id": "", "family": family, "phrase": phrase, "unit": unit, "unit_id": f"c179-{family}-{unit:02d}", "partition": base.partition(unit), "truth": truth, "surface": surface, "codebook": codebook, "gold_position": gold, "prompt": prompt, "semantic_edges": edges, "nodes": list(nodes), "role_values": {"primary": a, "secondary": d, "relation": phrase, "context": b, "query": a}}


def configure():
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    base.OUT = OUT
    base.make_case = make_case


def contract():
    configure()
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C178 / "audit/independent_final_audit.json")
    cases = base.material()
    compiled = base.compile_rows(graph_base.tokenizer(), cases)
    cells = {(r["family"], r["unit"], r["truth"], r["surface"], r["codebook"]) for r in cases}
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C179_visible_codebook_repair_same_object",
        "cases": len(cases) == 384,
        "families": len(base.RELATIONS) == 8,
        "cells": len(cells) == 384,
        "partitions": all(sum(r["partition"] == p for r in cases) == 128 for p in base.PARTITIONS),
        "candidate_balance": float(np.mean([r["gold_position"] == 0 for r in cases])) == 0.5,
        "truth_balance": float(np.mean([r["truth"] == 1 for r in cases])) == 0.5,
        "visible_codebook": all(("(A) no (B) yes" in r["prompt"]) == (r["codebook"] == -1) for r in cases),
        "roles": all(set(r["role_positions"]) == set(base.ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) < base.WIDTH,
        "semantic_uniqueness": all((r["truth"] == 1) == ((r["nodes"][1], r["nodes"][2]) in r["semantic_edges"]) for r in cases),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "visible_codebook_natural_ecology_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "repair": "candidate meanings are visibly reversed when codebook=-1; semantic graphs, vocabulary, partitions and gates unchanged",
        "families": list(base.RELATIONS),
        "cases": len(cases),
        "behavior_gates": {"global_min": 0.80, "family_partition_min": 0.75, "all_frozen_anchors_correct": True},
        "hidden_policy": "no HiddenState capture or perturbation before behavior eligibility lock",
        "eligible_capture": "embedding, 36 block outputs, final norm; six roles for eligible rows; all tokens for frozen anchors",
        "response": "q24 source-role locked 64 coordinates to q25 six-role full field, symmetric finite difference",
        "source_roles": ["primary", "query", "relation"],
        "naturalness": "human-readable controlled registry English; machine-audited only",
        "claim_boundary": "explicit multi-hop language ecology, not unrestricted world knowledge or spontaneous generation",
        "forbidden": ["attention", "MLP", "weights", "PCA", "hidden capture before behavior lock"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C179_behavior_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(r["prompt_ids"]) for r in compiled)}, indent=2))


def main():
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "behavior": base.behavior, "lock": base.lock, "hidden": base.hidden, "analyze": base.analyze, "close": base.close}[args.command]()


if __name__ == "__main__":
    main()
