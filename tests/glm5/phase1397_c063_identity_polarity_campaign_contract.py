#!/usr/bin/env python3
"""Phase1397: freeze C063 family-identity versus answer/polarity campaign."""
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

PHASE, CAMPAIGN = 1397, "C063"
PARENT = TESTS / "result/phase1396_c062_campaign_closure"
OUT = TESTS / "result/phase1397_c063_identity_polarity_campaign_contract"

FAMILIES = {
    "animal": ("horse", "rabbit", "monkey", "deer", "bear", "leopard", "squirrel", "dog", "cat", "elephant", "dolphin", "whale"),
    "building": ("palace", "mansion", "temple", "theater", "inn", "motel", "pub", "mall", "gym", "depot", "embassy", "arena"),
    "profession": ("doctor", "lawyer", "engineer", "teacher", "nurse", "dentist", "farmer", "pilot", "actor", "chef", "plumber", "accountant"),
    "country": ("Norway", "Sweden", "Denmark", "Poland", "Greece", "Turkey", "Chile", "Peru", "Kenya", "Nigeria", "Thailand", "Vietnam"),
}
PARTITIONS = {
    "response_discovery": range(0, 4),
    "confirmation": range(4, 8),
    "lockbox": range(8, 12),
}
SURFACES = {
    "ordinary": "Registry entry: {target} belongs to {record_family}. Query: does {target} belong to {query_family}? Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
    "catalog": "Catalog entry: {target} is classified as {record_family}. Question: is {target} classified as {query_family}? Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
    "statement": "Filed statement: {target} is a member of {record_family}. Check: is {target} a member of {query_family}? Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
}
STATUS_SURFACES = {
    "ordinary": "Registry entry: {target} belongs to {record_family}. Reference category: {query_family}. Query: is the record active? Record status: {status}. Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
    "catalog": "Catalog entry: {target} is classified as {record_family}. Reference category: {query_family}. Question: is the record active? Record status: {status}. Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
    "statement": "Filed statement: {target} is a member of {record_family}. Reference category: {query_family}. Check: is the record active? Record status: {status}. Response key: {key}. With alpha, use yes for true and no for false; with beta, use right for true and wrong for false. Output one word.",
}
SYSTEM = "Use only the explicit registry entry and response key. Output exactly one permitted word."
KEYS = ("alpha", "beta")
CANDIDATES = ("yes", "no", "right", "wrong")
FAMILY_ORDER = tuple(FAMILIES)


def gold(key: str, truth: bool) -> int:
    return (0 if truth else 1) if key == "alpha" else (2 if truth else 3)


def partition_for(index: int) -> str:
    return next(name for name, values in PARTITIONS.items() if index in values)


def active_cases() -> list[dict]:
    rows = []
    for fa, fb in combinations(FAMILY_ORDER, 2):
        pair = f"{fa}__{fb}"
        for index in range(12):
            values = {
                "aa": (FAMILIES[fa][index], fa, fa, True),
                "ab": (FAMILIES[fa][index], fa, fb, False),
                "bb": (FAMILIES[fb][index], fb, fb, True),
                "ba": (FAMILIES[fb][index], fb, fa, False),
            }
            for surface, template in SURFACES.items():
                for key in KEYS:
                    for cell, (target, record_family, query_family, truth) in values.items():
                        rows.append({
                            "case_id": f"c063-a-{len(rows):05d}",
                            "partition": partition_for(index),
                            "pair": pair,
                            "index": index,
                            "surface": surface,
                            "key": key,
                            "cell": cell,
                            "target": target,
                            "record_family": record_family,
                            "query_family": query_family,
                            "truth": truth,
                            "prompt": template.format(target=target, record_family=record_family, query_family=query_family, key=key),
                            "candidates": list(CANDIDATES),
                            "gold_position": gold(key, truth),
                        })
    return rows


def status_cases() -> list[dict]:
    rows = []
    for family_index, family in enumerate(FAMILY_ORDER):
        query_family = FAMILY_ORDER[(family_index + 1) % len(FAMILY_ORDER)]
        for index, target in enumerate(FAMILIES[family]):
            for surface, template in STATUS_SURFACES.items():
                for key in KEYS:
                    for status, truth in (("active", True), ("inactive", False)):
                        rows.append({
                            "case_id": f"c063-s-{len(rows):05d}",
                            "partition": partition_for(index),
                            "index": index,
                            "surface": surface,
                            "key": key,
                            "target": target,
                            "record_family": family,
                            "query_family": query_family,
                            "status": status,
                            "truth": truth,
                            "prompt": template.format(target=target, record_family=family, query_family=query_family, status=status, key=key),
                            "candidates": list(CANDIDATES),
                            "gold_position": gold(key, truth),
                        })
    return rows


def occurrences(ids: list[int], needles: list[list[int]]) -> list[list[int]]:
    found = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                span = list(range(start, start + len(needle)))
                if span not in found:
                    found.append(span)
    return sorted(found)


def spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [[int(x) for x in tok.encode(form, add_special_tokens=False)] for form in (value, " " + value)]
    return occurrences(ids, needles)


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        target_spans = spans(tok, ids, row["target"])
        record_spans = spans(tok, ids, row["record_family"])
        query_spans = spans(tok, ids, row["query_family"])
        key_spans = spans(tok, ids, row["key"])
        if not target_spans or not record_spans or not query_spans or not key_spans:
            raise RuntimeError((row["case_id"], target_spans, record_spans, query_spans, key_spans))
        compiled.append({
            "case_id": row["case_id"],
            "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(v, add_special_tokens=False)] for v in CANDIDATES],
            "role_positions": {
                "record_target": target_spans[0],
                "record_family": record_spans[0],
                "query_target": target_spans[1] if len(target_spans) > 1 else target_spans[0],
                "query_family": query_spans[-1],
                "response_key": key_spans[0],
                "boundary": [len(ids) - 1],
            },
        })
    return compiled


def factor_sets(active: list[dict], status: list[dict]) -> list[dict]:
    by = {(r["pair"], r["index"], r["surface"], r["key"], r["cell"]): r for r in active}
    status_by = {(r["record_family"], r["index"], r["surface"], r["key"], r["status"]): r for r in status}
    result = []
    next_surface = {name: tuple(SURFACES)[(i + 1) % len(SURFACES)] for i, name in enumerate(SURFACES)}
    for family_index, family in enumerate(FAMILY_ORDER):
        other = FAMILY_ORDER[(family_index + 1) % len(FAMILY_ORDER)]
        pair = "__".join(sorted((family, other), key=FAMILY_ORDER.index))
        self_cell = "aa" if pair.startswith(family + "__") else "bb"
        false_cell = "ab" if self_cell == "aa" else "ba"
        other_true = "bb" if self_cell == "aa" else "aa"
        other_false = "ba" if self_cell == "aa" else "ab"
        for index in range(12):
            member_index = next(v for v in PARTITIONS[partition_for(index)] if v != index)
            for surface in SURFACES:
                for key in KEYS:
                    other_key = "beta" if key == "alpha" else "alpha"
                    recipient = by[(pair, index, surface, key, self_cell)]
                    result.append({
                        "set_id": f"c063-factor-{len(result):04d}",
                        "partition": recipient["partition"],
                        "family": family,
                        "index": index,
                        "surface": surface,
                        "key": key,
                        "recipient": recipient["case_id"],
                        "surface_same": by[(pair, index, next_surface[surface], key, self_cell)]["case_id"],
                        "member_same": by[(pair, member_index, surface, key, self_cell)]["case_id"],
                        "family_only": by[(pair, index, surface, key, other_true)]["case_id"],
                        "answer_only": by[(pair, index, surface, other_key, self_cell)]["case_id"],
                        "family_and_answer": by[(pair, index, surface, other_key, other_true)]["case_id"],
                        "polarity_only": by[(pair, index, surface, key, false_cell)]["case_id"],
                        "family_and_polarity": by[(pair, index, surface, key, other_false)]["case_id"],
                        "status_null": status_by[(family, index, surface, key, "active")]["case_id"],
                    })
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1397 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if not parent_audit["all_checks_passed"] or parent["next_required_action"] != "new C063 contract must separate family identity from answer polarity at the late checkpoint and test natural-state necessity; do not extend C062":
        raise RuntimeError("C062 closure does not authorize the C063 object")

    tok = tokenizer()
    active, status = active_cases(), status_cases()
    factors = factor_sets(active, status)
    compiled_active, compiled_status = compile_rows(tok, active), compile_rows(tok, status)
    all_compiled = compiled_active + compiled_status
    old_graphs = [
        TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract/material/frozen_concept_graph.json",
        TESTS / "result/phase1390_c062_route_factorized_field_campaign_contract/material/frozen_concept_graph.json",
    ]
    old_words = {r["word"] for path in old_graphs for r in core.load(path)["concepts"]}
    words = {word for values in FAMILIES.values() for word in values}
    active_gold = Counter(r["gold_position"] for r in active)
    status_gold = Counter(r["gold_position"] for r in status)
    layout = defaultdict(set)
    for row, comp in zip(active, compiled_active):
        layout[row["surface"]].add((len(comp["prompt_ids"]), tuple((k, tuple(v)) for k, v in comp["role_positions"].items())))
    factor_keys = {"recipient", "surface_same", "member_same", "family_only", "answer_only", "family_and_answer", "polarity_only", "family_and_polarity", "status_null"}
    active_ids = {r["case_id"] for r in active}
    status_ids = {r["case_id"] for r in status}
    checks = {
        "parent_closed_audited": bool(parent_audit["all_checks_passed"]),
        "four_balanced_families": len(FAMILIES) == 4 and all(len(v) == 12 for v in FAMILIES.values()),
        "fresh_vs_c060_c062": not (words & old_words),
        "semantic_inventory_unique": len(words) == 48 and all(word and family for family, values in FAMILIES.items() for word in values),
        "active_count_balance": len(active) == 1728 and Counter(r["truth"] for r in active) == {True: 864, False: 864},
        "active_answer_balance": active_gold == {0: 432, 1: 432, 2: 432, 3: 432},
        "status_count_balance": len(status) == 576 and Counter(r["truth"] for r in status) == {True: 288, False: 288},
        "status_answer_balance": status_gold == {0: 144, 1: 144, 2: 144, 3: 144},
        "factor_count": len(factors) == 288,
        "factor_schema": all(factor_keys <= set(r) for r in factors),
        "factor_references_exist": all(r[k] in active_ids if k != "status_null" else r[k] in status_ids for r in factors for k in factor_keys),
        "compiled_counts": len(compiled_active) == len(active) and len(compiled_status) == len(status),
        "candidate_single_token": all(len(v) == 1 for r in all_compiled for v in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"record_target", "record_family", "query_target", "query_family", "response_key", "boundary"} for r in all_compiled),
        "active_surface_layout_exact": all(len(values) == 1 for values in layout.values()),
        "machine_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("Output one word.") for r in active + status),
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})

    concepts = [{"word": word, "family": family, "index": index, "partition": partition_for(index),
                 "sense": f"ordinary concrete {family} sense of {word}",
                 "adjudication": f"frozen as one unambiguous {family} member in this controlled registry"}
                for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c063.identity_polarity.v1", "families": FAMILIES,
              "partitions": {k: list(v) for k, v in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/factor_sets.jsonl", factors)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)

    preaudit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": {
            "always_yes": 0.25,
            "always_no": 0.25,
            "always_right": 0.25,
            "always_wrong": 0.25,
            "truth_without_key_upper_bound": 0.5,
            "key_without_truth_upper_bound": 0.5,
            "family_without_query_upper_bound": 0.5,
        },
        "semantic_scope": "explicit registry truth with ordinary family labels; proper capitalization disambiguates Turkey",
        "naturalness_scope": "controlled grammatical English with deterministic machine audit",
        "independent_human_blind_review": False,
        "disclosed_risks": ["controlled registry is not open natural generation", "building terms can denote institutions as well as structures"],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c063.identity_polarity_natural_field.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "natural full-state separation of family identity, semantic polarity, and answer-token identity",
        "allowed_observables": ["input token embeddings", "all layers/all positions full-dimensional hidden states", "logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP", "SAE", "learned probe", "post-reveal material/layer/role/threshold/donor replacement"],
        "material": {
            "families": list(FAMILIES), "partitions": list(PARTITIONS), "surfaces": list(SURFACES), "keys": list(KEYS),
            "active_count": len(active), "status_count": len(status), "factor_set_count": len(factors),
            "eligible_per_family_partition_surface_key": 2, "selected_per_family_partition": 12,
            "selected_per_family": 36, "minimum_qualified_families": 3,
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
            "factor_sha256": core.sha(OUT / "material/factor_sets.jsonl"),
            "human_naturalness_lock": False,
        },
        "behavior": {
            "family_active_accuracy_min": 0.90, "family_partition_min": 0.85, "family_surface_min": 0.85,
            "family_key_min": 0.90, "family_truth_min": 0.85, "family_quartet_all_min": 0.75,
            "status_accuracy_min": 0.95, "same_shape_repeat_max_abs_diff": 1e-6,
            "route_rule": "qualify or eliminate each family independently; require three qualified families",
        },
        "camera": {
            "known_truth_systems": 256, "qwen_cases": 24, "four_class_logit_identity_max_abs_diff": 1e-5,
            "all_state_identity_relative_l2_max": 1e-6, "role_swap_exact": True, "multi_checkpoint_exact": True,
        },
        "observation": {
            "partition": "response_discovery", "all_hidden_state_indices": list(range(37)), "all_physical_positions": True,
            "factor_donors": ["surface_same", "member_same", "family_only", "answer_only", "family_and_answer", "polarity_only", "family_and_polarity", "status_null"],
            "roles": ["record_target", "record_family", "query_target", "query_family", "response_key", "boundary"],
            "windows": [[1, 15], [16, 24], [25, 36]], "top_candidates_per_object_window": 1,
            "candidate_source_is_discovery_only": True,
        },
        "factorial_swap": {
            "partitions": ["confirmation", "lockbox"], "self_max_abs_diff": 1e-4,
            "control_loss_fraction_max": 0.25, "family_damage_median_min": 0.5,
            "family_over_answer_damage_median_min": 0.25, "family_over_answer_win_min": 0.65,
            "answer_redirect_fraction_min": 0.65, "polarity_redirect_fraction_min": 0.65,
            "minimum_family_breadth": 2,
        },
        "branching": {
            "phase1398": "factorized behavior", "phase1399": "known-truth and Qwen state-swap camera",
            "phase1400": "natural discovery field and frozen candidates", "phase1401": "holdout factorial natural-state swaps",
            "phase1402": "campaign closure",
        },
        "claim_boundary": {
            "allowed": "Qwen-specific controlled-registry natural-state identity/polarity/answer separation",
            "forbidden": ["semantic neurons", "minimal circuit", "unique serial natural path", "attention/MLP/parameter mechanism", "cross-model or open-language law"],
        },
        "stop_rule": "failure eliminates only its frozen family or candidate route; behavior breadth or camera failure blocks dependent hidden access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1398_c063_factorized_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
              "all_gates_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
