#!/usr/bin/env python3
"""Phase1417: preregister C068 bidirectional four-role state composition."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1417, "C068"
PARENT = TESTS / "result/phase1416_c067_campaign_closure"
OUT = TESTS / "result/phase1417_c068_four_role_composition_contract"
FAMILIES = {
    "organ": ("heart", "lung", "liver", "kidney", "brain", "stomach", "eye", "ear", "bladder", "intestine", "thyroid", "skin"),
    "month": ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
    "shape": ("circle", "square", "triangle", "rectangle", "oval", "cube", "sphere", "cone", "cylinder", "prism", "pyramid", "star"),
    "color": ("red", "blue", "green", "yellow", "black", "white", "teal", "purple", "pink", "brown", "gray", "cyan"),
    "number": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"),
    "city": ("London", "Paris", "Tokyo", "Rome", "Cairo", "Lima", "Berlin", "Madrid", "Seoul", "Nairobi", "Havana", "Oslo"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SURFACES = {
    "ordinary": "Registry entry: {record_target} belongs to {record_family}. Query: does {query_target} belong to {query_family}? Output only yes or no.",
    "catalog": "Catalog entry: {record_target} is classified as {record_family}. Question: is {query_target} classified as {query_family}? Output only yes or no.",
    "statement": "Filed statement: {record_target} is a member of {record_family}. Check: is {query_target} a member of {query_family}? Output only yes or no.",
}
SYSTEM = "Use only the explicit registry entry. Answer yes only when both queried labels exactly match that entry. Output exactly yes or no."
CELL_NAMES = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")


def partition(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


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
            aw, bw = FAMILIES[fa][index], FAMILIES[fb][index]
            cells = {
                "aa": (aw, fa, aw, fa, True),
                "ab": (aw, fa, aw, fb, False),
                "ac": (aw, fa, bw, fa, False),
                "ad": (aw, fa, bw, fb, False),
                "bb": (bw, fb, bw, fb, True),
                "ba": (bw, fb, bw, fa, False),
                "bc": (bw, fb, aw, fb, False),
                "bd": (bw, fb, aw, fa, False),
            }
            for surface, template in SURFACES.items():
                for cell, (record_target, record_family, query_target, query_family, truth) in cells.items():
                    rows.append({
                        "case_id": f"c068-a-{len(rows):04d}",
                        "partition": partition(index),
                        "pair": pair,
                        "index": index,
                        "surface": surface,
                        "cell": cell,
                        "record_target": record_target,
                        "record_family": record_family,
                        "query_target": query_target,
                        "query_family": query_family,
                        "truth": truth,
                        "prompt": template.format(record_target=record_target, record_family=record_family, query_target=query_target, query_family=query_family),
                        "candidates": ["yes", "no"],
                        "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        rt = all_spans(tok, ids, row["record_target"])
        rf = all_spans(tok, ids, row["record_family"])
        qt = all_spans(tok, ids, row["query_target"])
        qf = all_spans(tok, ids, row["query_family"])
        if not rt or not rf or not qt or not qf:
            raise RuntimeError((row["case_id"], rt, rf, qt, qf))
        record_target = rt[0]
        query_target = qt[-1]
        record_family = rf[0]
        query_family = qf[-1]
        compiled.append({
            "case_id": row["case_id"],
            "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("yes", "no")],
            "role_positions": {
                "record_target": record_target,
                "record_family": record_family,
                "query_target": query_target,
                "query_family": query_family,
                "boundary": [len(ids) - 1],
            },
        })
    return compiled


def first_by_signature(rows: list[dict]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (row["record_target"], row["record_family"], row["query_target"], row["query_family"], row["surface"])
        result.setdefault(key, row)
    return result


def composition_sets(active: list[dict]) -> list[dict]:
    by = first_by_signature(active)
    result = []
    for fi, family in enumerate(ORDER):
        g = ORDER[(fi + 1) % len(ORDER)]
        h = ORDER[(fi + 2) % len(ORDER)]
        for index in range(12):
            fw, gw, hw = FAMILIES[family][index], FAMILIES[g][index], FAMILIES[h][index]
            member_index = next(v for v in PARTITIONS[partition(index)] if v != index)
            fm = FAMILIES[family][member_index]
            result.append({
                "set_id": f"c068-compose-{len(result):04d}",
                "partition": partition(index),
                "family": family,
                "g_family": g,
                "h_family": h,
                "index": index,
                "surface": "catalog",
                "true_recipient": by[(fw, family, fw, family, "catalog")]["case_id"],
                "false_recipient": by[(fw, family, fw, g, "catalog")]["case_id"],
                "true_surface": by[(fw, family, fw, family, "ordinary")]["case_id"],
                "false_surface": by[(fw, family, fw, g, "ordinary")]["case_id"],
                "true_member": by[(fm, family, fm, family, "catalog")]["case_id"],
                "false_member": by[(fm, family, fm, g, "catalog")]["case_id"],
                "g_true": by[(gw, g, gw, g, "catalog")]["case_id"],
                "h_true": by[(hw, h, hw, h, "catalog")]["case_id"],
                "g_false_h": by[(gw, g, gw, h, "catalog")]["case_id"],
                "h_false_g": by[(hw, h, hw, g, "catalog")]["case_id"],
                "cross_gh": by[(gw, g, hw, h, "catalog")]["case_id"],
                "cross_hg": by[(hw, h, gw, g, "catalog")]["case_id"],
            })
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1417 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c068_distributed_four_role_composition" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C067 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    source = {row["case_id"]: row for row in active}
    words = {word for values in FAMILIES.values() for word in values}
    old_words = set()
    for path in (TESTS / "result").glob("phase*/material/frozen_concept_graph.json"):
        old_words |= {row["word"] for row in core.load(path)["concepts"]}
    intervention_roles = ("record_target", "record_family", "query_target", "query_family")
    donor_truth = {
        "true_recipient": True, "false_recipient": False,
        "true_surface": True, "false_surface": False,
        "true_member": True, "false_member": False,
        "g_true": True, "h_true": True,
        "g_false_h": False, "h_false_g": False,
        "cross_gh": False, "cross_hg": False,
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "six_families": len(FAMILIES) == 6,
        "seventy_two_words": len(words) == 72 and all(len(values) == 12 for values in FAMILIES.values()),
        "fresh_words": not (words & old_words),
        "active_count": len(active) == 4320,
        "cell_balance": Counter(row["cell"] for row in active) == {cell: 540 for cell in CELL_NAMES},
        "truth_counts": Counter(row["truth"] for row in active) == {True: 1080, False: 3240},
        "composition_count": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[key]]["truth"] == expected for row in composition for key, expected in donor_truth.items()),
        "compiled": len(compiled) == len(active),
        "single_token_answers": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "typed_roles": all(set(row["role_positions"]) == {*intervention_roles, "boundary"} for row in compiled),
        "quartet_singleton": all(len(row["role_positions"][role]) == 1 for row in compiled for role in intervention_roles),
        "semantic_unique": all(row["gold_position"] == (0 if row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"] else 1) for row in active),
        "machine_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.") for row in active),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    concepts = [{"word": word, "family": family, "index": index, "partition": partition(index), "sense": f"controlled English {family} sense of {word}; explicit registry is decisive"} for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c068.four_role_composition.v1", "families": FAMILIES, "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/composition_sets.jsonl", composition)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": {
            "always_yes_balanced_accuracy": 0.5,
            "always_no_balanced_accuracy": 0.5,
            "physical_first_balanced_accuracy": 0.5,
            "surface_only_balanced_accuracy": 0.5,
            "lexical_exact_match": "construct-equivalent rule, registered as a positive symbolic baseline rather than a null",
        },
        "semantic_scope": "closed-world equality of both explicit target and family labels uniquely fixes yes/no",
        "naturalness_scope": "machine-audited controlled English; cross-target negatives are valid only under the explicit closed-world instruction",
        "independent_human_blind_review": False,
        "risks": ["explicit registry rather than open discourse", "raw truth prevalence is 1:3", "whole states remain mixed variables", "all quartet spans are constrained to one token"],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    arms = ["self", "surface", "member", "matched_g", "matched_h", "mismatched_gh", "mismatched_hg", "natural_false_gh", "natural_false_hg"]
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c068.state16_bidirectional_four_role_composition.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "catalog state-16 four-role tuple composition in true-to-mismatch and false-to-match directions",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state 16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "subset search", "candidate search", "post-reveal changes"],
        "material": {
            "families": list(FAMILIES),
            "partitions": list(PARTITIONS),
            "surfaces": list(SURFACES),
            "mechanism_surface": "catalog",
            "active_count": len(active),
            "composition_count": len(composition),
            "minimum_qualified_families": 4,
            "selected_per_family_partition": 4,
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"),
            "human_naturalness_lock": False,
        },
        "behavior": {
            "family_active_accuracy_min": 0.95,
            "family_balanced_accuracy_min": 0.95,
            "family_partition_min": 0.90,
            "family_surface_min": 0.90,
            "family_truth_min": 0.90,
            "family_cell_min": 0.90,
            "family_set_all_min": 0.85,
            "same_shape_repeat_max_abs_diff": 1e-6,
        },
        "camera": {"known_truth_systems": 256, "qwen_discovery_sets": 24, "self_quartet_max_abs_diff": 1e-4, "state_index": 16, "roles": list(intervention_roles)},
        "mechanism": {
            "state_index": 16,
            "surface": "catalog",
            "directions": ["true_recipient", "false_recipient"],
            "arms": arms,
            "self_max_abs_diff": 1e-4,
            "control_sign_fraction_min": 0.90,
            "control_relative_deviation_median_max": 0.25,
            "matched_positive_true_min": 0.80,
            "mismatched_negative_true_min": 0.65,
            "natural_false_negative_true_min": 0.65,
            "matched_positive_false_min": 0.65,
            "mismatched_negative_false_min": 0.80,
            "natural_false_negative_false_min": 0.80,
            "matched_rescue_gain_false_median_min": 0.5,
            "interaction_median_min": 0.5,
            "interaction_win_fraction_min": 0.65,
            "minimum_family_breadth": 4,
        },
        "evidence_levels": {
            "graded": "controls plus positive diagonal interaction contrast in both directions and holdouts",
            "discrete": "controls plus matched-positive and mismatched/natural-false-negative sign gates in both directions and holdouts",
            "strong": "graded and discrete both pass with minimum family breadth",
        },
        "branching": {"phase1418": "behavior", "phase1419": "four-role camera", "phase1420": "bidirectional quartet composition", "phase1421": "closure"},
        "stop_rule": "behavior failure eliminates only a family; fewer than four families blocks hidden; quartet route runs once with frozen arms; no layer or subset fallback inside C068",
        "claim_boundary": {"allowed": "Qwen controlled-registry catalog state-16 bidirectional quartet response evidence", "forbidden": ["relation manifold", "semantic comparator", "minimal or necessary state", "attention/MLP/parameter mechanism", "cross-model/open-language law"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1418_c068_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
