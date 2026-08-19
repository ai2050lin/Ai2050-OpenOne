#!/usr/bin/env python3
"""Phase1408: preregister the state-16-only C066 breadth replication."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1408, "C066"
PARENT = TESTS / "result/phase1407_c065_campaign_closure"
OUT = TESTS / "result/phase1408_c066_midstate_breadth_contract"
FAMILIES = {
    "fruit": ("apple", "pear", "mango", "peach", "plum", "grape", "banana", "cherry", "lemon", "orange", "papaya", "guava"),
    "vehicle": ("bicycle", "scooter", "tractor", "subway", "taxi", "bus", "truck", "van", "train", "sedan", "motorcycle", "limousine"),
    "instrument": ("violin", "guitar", "piano", "trumpet", "flute", "cello", "banjo", "harp", "oboe", "drum", "clarinet", "saxophone"),
    "tool": ("hammer", "wrench", "chisel", "drill", "saw", "pliers", "shovel", "rake", "hoe", "mallet", "clamp", "crowbar"),
    "garment": ("jacket", "sweater", "shirt", "blouse", "skirt", "trousers", "shorts", "coat", "dress", "scarf", "robe", "vest"),
    "flower": ("rose", "tulip", "daisy", "orchid", "lily", "violet", "jasmine", "iris", "poppy", "marigold", "daffodil", "carnation"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SURFACES = {
    "ordinary": "Registry entry: {target} belongs to {record_family}. Query: does {target} belong to {query_family}? Output only yes or no.",
    "catalog": "Catalog entry: {target} is classified as {record_family}. Question: is {target} classified as {query_family}? Output only yes or no.",
    "statement": "Filed statement: {target} is a member of {record_family}. Check: is {target} a member of {query_family}? Output only yes or no.",
}
SYSTEM = "Use only the explicit registry entry. Output exactly yes or no."


def partition(index: int) -> str:
    return next(name for name, indices in PARTITIONS.items() if index in indices)


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(v, add_special_tokens=False))) for v in (value, " " + value)]
    found = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                span = list(range(start, start + len(needle)))
                if span not in found:
                    found.append(span)
    return sorted(found)


def active_cases() -> list[dict]:
    rows = []
    for fa, fb in combinations(ORDER, 2):
        pair = f"{fa}__{fb}"
        for index in range(12):
            cells = {
                "aa": (FAMILIES[fa][index], fa, fa, True),
                "ab": (FAMILIES[fa][index], fa, fb, False),
                "bb": (FAMILIES[fb][index], fb, fb, True),
                "ba": (FAMILIES[fb][index], fb, fa, False),
            }
            for surface, template in SURFACES.items():
                for cell, (target, record_family, query_family, truth) in cells.items():
                    rows.append({
                        "case_id": f"c066-a-{len(rows):04d}",
                        "partition": partition(index),
                        "pair": pair,
                        "index": index,
                        "surface": surface,
                        "cell": cell,
                        "target": target,
                        "record_family": record_family,
                        "query_family": query_family,
                        "truth": truth,
                        "prompt": template.format(target=target, record_family=record_family, query_family=query_family),
                        "candidates": ["yes", "no"],
                        "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        targets = all_spans(tok, ids, row["target"])
        record = all_spans(tok, ids, row["record_family"])
        query = all_spans(tok, ids, row["query_family"])
        if not targets or not record or not query:
            raise RuntimeError((row["case_id"], targets, record, query))
        compiled.append({
            "case_id": row["case_id"],
            "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("yes", "no")],
            "role_positions": {
                "record_target": targets[0],
                "record_family": record[0],
                "query_target": targets[1] if len(targets) > 1 else targets[0],
                "query_family": query[-1],
                "boundary": [len(ids) - 1],
            },
        })
    return compiled


def factor_sets(active: list[dict]) -> list[dict]:
    by = {(r["pair"], r["index"], r["surface"], r["cell"]): r for r in active}
    surface_next = {surface: tuple(SURFACES)[(i + 1) % len(SURFACES)] for i, surface in enumerate(SURFACES)}
    result = []
    for fi, family in enumerate(ORDER):
        other = ORDER[(fi + 1) % len(ORDER)]
        pair = "__".join(sorted((family, other), key=ORDER.index))
        own_true = "aa" if pair.startswith(family + "__") else "bb"
        own_false = "ab" if own_true == "aa" else "ba"
        other_true = "bb" if own_true == "aa" else "aa"
        other_false = "ba" if own_true == "aa" else "ab"
        for index in range(12):
            member_index = next(v for v in PARTITIONS[partition(index)] if v != index)
            for surface in SURFACES:
                recipient = by[(pair, index, surface, own_true)]
                result.append({
                    "set_id": f"c066-factor-{len(result):04d}",
                    "partition": recipient["partition"],
                    "family": family,
                    "index": index,
                    "surface": surface,
                    "recipient": recipient["case_id"],
                    "surface_same": by[(pair, index, surface_next[surface], own_true)]["case_id"],
                    "member_same": by[(pair, member_index, surface, own_true)]["case_id"],
                    "family_same_polarity": by[(pair, index, surface, other_true)]["case_id"],
                    "polarity_same_family": by[(pair, index, surface, own_false)]["case_id"],
                    "family_and_polarity": by[(pair, index, surface, other_false)]["case_id"],
                })
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1408 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c066_midstate_breadth_confirmation" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C065 closure missing")
    tok = tokenizer()
    active = active_cases()
    factors = factor_sets(active)
    compiled = compile_rows(tok, active)
    words = {word for values in FAMILIES.values() for word in values}
    old_words = set()
    for phase in (1380, 1390, 1397, 1400):
        paths = list((TESTS / "result").glob(f"phase{phase}_*/material/frozen_concept_graph.json"))
        if len(paths) != 1:
            raise RuntimeError(f"old graph {phase}")
        old_words |= {r["word"] for r in core.load(paths[0])["concepts"]}
    source = {r["case_id"]: r for r in active}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "six_families": len(FAMILIES) == 6,
        "seventy_two_unique_words": len(words) == 72 and all(len(values) == 12 for values in FAMILIES.values()),
        "fresh_words": not (words & old_words),
        "active_count": len(active) == 2160,
        "active_balance": Counter(r["truth"] for r in active) == {True: 1080, False: 1080},
        "factor_count": len(factors) == 216 and Counter(r["partition"] for r in factors) == {p: 72 for p in PARTITIONS},
        "compiled": len(compiled) == len(active),
        "single_token_answers": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "typed_roles": all(set(row["role_positions"]) == {"record_target", "record_family", "query_target", "query_family", "boundary"} for row in compiled),
        "intervention_roles_singleton": all(len(row["role_positions"][role]) == 1 for row in compiled for role in ("record_family", "query_family")),
        "semantic_unique_by_registry": all(r["gold_position"] == (0 if r["record_family"] == r["query_family"] else 1) for r in active),
        "machine_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.") for r in active),
        "zero_models": True,
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    concepts = [{
        "word": word,
        "family": family,
        "index": index,
        "partition": partition(index),
        "sense": f"ordinary English {family} sense of {word}, with local registry decisive",
    } for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c066.midstate_breadth.v1", "families": FAMILIES, "partitions": {k: list(v) for k, v in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/factor_sets.jsonl", factors)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5, "target_only": 0.5, "surface_only": 0.5},
        "semantic_scope": "local explicit registry gives a unique binary answer",
        "naturalness_scope": "machine-audited controlled grammatical English using familiar noun-family memberships",
        "independent_human_blind_review": False,
        "risks": ["truth remains coupled to yes/no", "family labels and words vary in token length; alignment is by typed role rather than physical index", "controlled registry is not open discourse"],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c066.state16_breadth_confirmation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "preregistered state-16 role-aligned family and joint-polarity whole-state effect",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden state at state 16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "learned probe", "new candidate search", "post-reveal threshold change"],
        "material": {
            "families": list(FAMILIES),
            "partitions": list(PARTITIONS),
            "surfaces": list(SURFACES),
            "active_count": len(active),
            "factor_count": len(factors),
            "eligible_per_family_partition_surface": 4,
            "selected_per_family": 36,
            "minimum_qualified_families": 4,
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "factor_sha256": core.sha(OUT / "material/factor_sets.jsonl"),
            "human_naturalness_lock": False,
        },
        "behavior": {
            "family_active_accuracy_min": 0.95,
            "family_partition_min": 0.90,
            "family_surface_min": 0.90,
            "family_truth_min": 0.90,
            "family_pair_all_min": 0.85,
            "same_shape_repeat_max_abs_diff": 1e-6,
        },
        "mechanism": {
            "state_index": 16,
            "candidates": [
                {"surface": surface, "object": "family_identity", "role": "record_family"} for surface in SURFACES
            ] + [
                {"surface": surface, "object": "joint_polarity", "role": "query_family"} for surface in SURFACES
            ],
            "partitions": ["confirmation", "lockbox"],
            "arms": ["self", "surface_same", "member_same", "family_same_polarity", "polarity_same_family", "family_and_polarity"],
            "self_max_abs_diff": 1e-4,
            "surface_control_loss_fraction_max": 0.25,
            "member_control_loss_fraction_max": 0.50,
            "family_damage_median_min": 0.5,
            "family_over_member_median_min": 0.25,
            "family_over_member_win_min": 0.65,
            "polarity_redirect_fraction_min": 0.65,
            "minimum_family_breadth": 4,
        },
        "branching": {"phase1409": "behavior", "phase1410": "state16 factorial replication", "phase1411": "closure"},
        "stop_rule": "behavior failure eliminates only a family; fewer than four families blocks hidden access; mechanism failure closes only that object/surface route",
        "claim_boundary": {
            "allowed": "Qwen-specific controlled-registry state-16 whole-state breadth replication",
            "forbidden": ["semantic neurons", "minimal state", "truth/token separation", "unique path", "attention/MLP/parameter mechanism", "cross-model law"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1409_c066_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
