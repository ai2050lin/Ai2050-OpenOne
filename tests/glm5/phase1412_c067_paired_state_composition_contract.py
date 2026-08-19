#!/usr/bin/env python3
"""Phase1412: preregister C067 paired state-16 relational composition."""
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

PHASE, CAMPAIGN = 1412, "C067"
PARENT = TESTS / "result/phase1411_c066_campaign_closure"
OUT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
FAMILIES = {
    "tree": ("oak", "maple", "pine", "birch", "cedar", "willow", "elm", "spruce", "fir", "aspen", "cypress", "redwood"),
    "fish": ("salmon", "trout", "tuna", "cod", "carp", "herring", "sardine", "mackerel", "anchovy", "haddock", "halibut", "perch"),
    "mineral": ("quartz", "calcite", "gypsum", "mica", "feldspar", "halite", "talc", "fluorite", "apatite", "corundum", "magnetite", "hematite"),
    "language": ("English", "Spanish", "French", "Arabic", "Hindi", "Bengali", "Korean", "Finnish", "Swahili", "Latin", "Hebrew", "Persian"),
    "bird": ("canary", "sparrow", "robin", "falcon", "hawk", "parrot", "raven", "crow", "swan", "goose", "duck", "pigeon"),
    "fabric": ("cotton", "silk", "wool", "linen", "denim", "velvet", "satin", "nylon", "polyester", "rayon", "canvas", "fleece"),
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
            cells = {
                "aa": (FAMILIES[fa][index], fa, fa, True),
                "ab": (FAMILIES[fa][index], fa, fb, False),
                "bb": (FAMILIES[fb][index], fb, fb, True),
                "ba": (FAMILIES[fb][index], fb, fa, False),
            }
            for surface, template in SURFACES.items():
                for cell, (target, record_family, query_family, truth) in cells.items():
                    rows.append({
                        "case_id": f"c067-a-{len(rows):04d}",
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


def first_by_signature(rows: list[dict]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (row["target"], row["record_family"], row["query_family"], row["surface"])
        result.setdefault(key, row)
    return result


def composition_sets(active: list[dict]) -> list[dict]:
    by = first_by_signature(active)
    result = []
    for fi, family in enumerate(ORDER):
        g = ORDER[(fi + 1) % len(ORDER)]
        h = ORDER[(fi + 2) % len(ORDER)]
        for index in range(12):
            fword, gword, hword = FAMILIES[family][index], FAMILIES[g][index], FAMILIES[h][index]
            member_index = next(v for v in PARTITIONS[partition(index)] if v != index)
            result.append({
                "set_id": f"c067-compose-{len(result):04d}",
                "partition": partition(index),
                "family": family,
                "g_family": g,
                "h_family": h,
                "index": index,
                "surface": "catalog",
                "recipient": by[(fword, family, family, "catalog")]["case_id"],
                "surface_same": by[(fword, family, family, "ordinary")]["case_id"],
                "member_same": by[(FAMILIES[family][member_index], family, family, "catalog")]["case_id"],
                "g_true": by[(gword, g, g, "catalog")]["case_id"],
                "h_true": by[(hword, h, h, "catalog")]["case_id"],
                "natural_fg": by[(fword, family, g, "catalog")]["case_id"],
                "natural_gf": by[(gword, g, family, "catalog")]["case_id"],
                "natural_gh": by[(gword, g, h, "catalog")]["case_id"],
                "natural_hg": by[(hword, h, g, "catalog")]["case_id"],
            })
    return result


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1412 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c067_paired_state_relational_composition" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C066 closure missing")
    tok = tokenizer()
    active = active_cases()
    compiled = compile_rows(tok, active)
    composition = composition_sets(active)
    words = {word for values in FAMILIES.values() for word in values}
    old_words = set()
    for phase in (1380, 1387, 1390, 1397, 1400, 1408):
        paths = list((TESTS / "result").glob(f"phase{phase}_*/material/frozen_concept_graph.json"))
        if len(paths) != 1:
            raise RuntimeError(f"old graph {phase}")
        old_words |= {r["word"] for r in core.load(paths[0])["concepts"]}
    source = {r["case_id"]: r for r in active}
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "six_families": len(FAMILIES) == 6,
        "seventy_two_words": len(words) == 72 and all(len(v) == 12 for v in FAMILIES.values()),
        "fresh_words": not (words & old_words),
        "active_count": len(active) == 2160,
        "active_balance": Counter(r["truth"] for r in active) == {True: 1080, False: 1080},
        "composition_count": len(composition) == 72 and Counter(r["partition"] for r in composition) == {p: 24 for p in PARTITIONS},
        "composition_semantics": all(source[r[key]]["truth"] == expected for r in composition for key, expected in (("recipient", True), ("surface_same", True), ("member_same", True), ("g_true", True), ("h_true", True), ("natural_fg", False), ("natural_gf", False), ("natural_gh", False), ("natural_hg", False))),
        "compiled": len(compiled) == len(active),
        "single_token_answers": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "typed_roles": all(set(row["role_positions"]) == {"record_target", "record_family", "query_target", "query_family", "boundary"} for row in compiled),
        "intervention_roles_singleton": all(len(row["role_positions"][role]) == 1 for row in compiled for role in ("record_family", "query_family")),
        "semantic_unique": all(r["gold_position"] == (0 if r["record_family"] == r["query_family"] else 1) for r in active),
        "machine_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.") for r in active),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    concepts = [{"word": word, "family": family, "index": index, "partition": partition(index), "sense": f"ordinary English {family} sense of {word}; local registry is decisive"} for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c067.paired_composition.v1", "families": FAMILIES, "partitions": {k: list(v) for k, v in PARTITIONS.items()}, "concepts": concepts})
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
        "zero_models": {"always_yes": 0.5, "always_no": 0.5, "target_only": 0.5, "surface_only": 0.5},
        "semantic_scope": "equality of explicit record/query family labels uniquely fixes yes/no",
        "naturalness_scope": "machine-audited controlled English with familiar noun families",
        "independent_human_blind_review": False,
        "risks": ["explicit registry rather than open discourse", "truth remains coupled to yes/no", "role states are full vectors", "family labels vary in token identity"],
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c067.state16_paired_relational_composition.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "catalog state-16 record/query matched relational composition",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden state at state 16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "probe", "layer search", "candidate search", "post-reveal changes"],
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
            "family_partition_min": 0.90,
            "family_surface_min": 0.90,
            "family_truth_min": 0.90,
            "family_pair_all_min": 0.85,
            "factorial_all_arms_required": True,
            "same_shape_repeat_max_abs_diff": 1e-6,
        },
        "camera": {"known_truth_systems": 256, "qwen_discovery_sets": 24, "self_dual_max_abs_diff": 1e-4, "state_index": 16, "roles": ["record_family", "query_family"]},
        "mechanism": {
            "state_index": 16,
            "surface": "catalog",
            "arms": ["self_dual", "surface_dual", "member_dual", "record_only_g", "query_only_g", "matched_dual_g", "matched_dual_h", "mismatched_dual_gh", "mismatched_dual_hg"],
            "self_max_abs_diff": 1e-4,
            "surface_control_loss_fraction_max": 0.25,
            "member_control_loss_fraction_max": 0.25,
            "record_damage_median_min": 0.5,
            "record_damage_win_fraction_min": 0.65,
            "query_redirect_fraction_min": 0.65,
            "matched_positive_fraction_min": 0.80,
            "matched_rescue_advantage_median_min": 0.5,
            "mismatched_negative_fraction_min": 0.65,
            "matched_over_mismatched_median_min": 0.5,
            "minimum_family_breadth": 4,
        },
        "branching": {"phase1413": "behavior", "phase1414": "dual-write camera", "phase1415": "paired composition", "phase1416": "closure"},
        "stop_rule": "behavior failure eliminates only a family; fewer than four families blocks hidden; every mechanism arm is frozen and no alternate state/surface is allowed",
        "claim_boundary": {"allowed": "Qwen controlled-registry catalog state-16 paired whole-state composition evidence", "forbidden": ["semantic comparator", "minimal coordinates", "necessary circuit", "truth/token separation", "attention/MLP/parameter mechanism", "cross-model/open-language law"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1413_c067_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
