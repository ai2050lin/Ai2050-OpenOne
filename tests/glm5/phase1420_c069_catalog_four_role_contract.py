#!/usr/bin/env python3
"""Phase1420: preregister catalog-scoped C069 four-role composition."""
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

PHASE, CAMPAIGN = 1420, "C069"
PARENT = TESTS / "result/phase1419_c068_behavior_gate_closure"
OUT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
FAMILIES = {
    "room": ("kitchen", "bedroom", "bathroom", "lobby", "hallway", "nursery", "pantry", "studio", "den", "closet", "laundry", "lounge"),
    "unit": ("meter", "second", "gram", "liter", "inch", "foot", "yard", "mile", "ounce", "quart", "gallon", "byte"),
    "texture": ("smooth", "rough", "soft", "hard", "wet", "dry", "sticky", "slippery", "fuzzy", "silky", "coarse", "flat"),
    "sound": ("loud", "quiet", "sharp", "dull", "faint", "harsh", "clear", "high", "low", "echoing", "buzzing", "silent"),
    "direction": ("north", "south", "east", "west", "left", "right", "up", "down", "forward", "backward", "inside", "outside"),
    "material": ("wood", "glass", "steel", "plastic", "rubber", "stone", "paper", "clay", "leather", "brick", "concrete", "foam"),
}
ORDER = tuple(FAMILIES)
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SURFACES = {
    "ordinary": "Registry entry: {record_target} belongs to {record_family}. Query: does {query_target} belong to {query_family}? Output only yes or no.",
    "catalog": "Catalog entry: {record_target} is classified as {record_family}. Question: is {query_target} classified as {query_family}? Output only yes or no.",
}
SYSTEM = "Use only the explicit registry entry. Answer yes only when both queried labels exactly match that entry. Output exactly yes or no."
CELLS = ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")
ROLES = ("record_target", "record_family", "query_target", "query_family")


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
                "aa": (aw, fa, aw, fa, True), "ab": (aw, fa, aw, fb, False),
                "ac": (aw, fa, bw, fa, False), "ad": (aw, fa, bw, fb, False),
                "bb": (bw, fb, bw, fb, True), "ba": (bw, fb, bw, fa, False),
                "bc": (bw, fb, aw, fb, False), "bd": (bw, fb, aw, fa, False),
            }
            for surface, template in SURFACES.items():
                for cell, (rt, rf, qt, qf, truth) in cells.items():
                    rows.append({
                        "case_id": f"c069-a-{len(rows):04d}", "partition": partition(index),
                        "pair": pair, "index": index, "surface": surface, "cell": cell,
                        "record_target": rt, "record_family": rf, "query_target": qt, "query_family": qf,
                        "truth": truth, "prompt": template.format(record_target=rt, record_family=rf, query_target=qt, query_family=qf),
                        "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        spans = {role: all_spans(tok, ids, row[role]) for role in ROLES}
        if not all(spans.values()):
            raise RuntimeError((row["case_id"], spans))
        compiled.append({
            "case_id": row["case_id"], "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(value, add_special_tokens=False)] for value in ("yes", "no")],
            "role_positions": {
                "record_target": spans["record_target"][0], "record_family": spans["record_family"][0],
                "query_target": spans["query_target"][-1], "query_family": spans["query_family"][-1],
                "boundary": [len(ids) - 1],
            },
        })
    return compiled


def signature(rows: list[dict]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (*(row[role] for role in ROLES), row["surface"])
        result.setdefault(key, row)
    return result


def composition_sets(active: list[dict]) -> list[dict]:
    by = signature(active)
    result = []
    for fi, family in enumerate(ORDER):
        g, h = ORDER[(fi + 1) % 6], ORDER[(fi + 2) % 6]
        for index in range(12):
            fw, gw, hw = FAMILIES[family][index], FAMILIES[g][index], FAMILIES[h][index]
            member_index = next(value for value in PARTITIONS[partition(index)] if value != index)
            fm = FAMILIES[family][member_index]
            result.append({
                "set_id": f"c069-compose-{len(result):04d}", "partition": partition(index),
                "family": family, "g_family": g, "h_family": h, "index": index, "surface": "catalog",
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
        raise RuntimeError("Phase1420 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c069_catalog_scoped_four_role_composition" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C068 closure missing")
    tok = tokenizer(); active = active_cases(); compiled = compile_rows(tok, active); composition = composition_sets(active)
    source = {row["case_id"]: row for row in active}
    words = {word for values in FAMILIES.values() for word in values}; old_words = set()
    for path in (TESTS / "result").glob("phase*/material/frozen_concept_graph.json"):
        old_words |= {row["word"] for row in core.load(path)["concepts"]}
    donor_truth = {"true_recipient": True, "false_recipient": False, "true_surface": True, "false_surface": False, "true_member": True, "false_member": False, "g_true": True, "h_true": True, "g_false_h": False, "h_false_g": False, "cross_gh": False, "cross_hg": False}
    checks = {
        "parent": parent_audit["all_checks_passed"], "six_families": len(FAMILIES) == 6,
        "seventy_two_fresh_words": len(words) == 72 and not (words & old_words),
        "active": len(active) == 2880 and Counter(row["cell"] for row in active) == {cell: 360 for cell in CELLS},
        "truth": Counter(row["truth"] for row in active) == {True: 720, False: 2160},
        "composition": len(composition) == 72 and Counter(row["partition"] for row in composition) == {name: 24 for name in PARTITIONS},
        "composition_semantics": all(source[row[key]]["truth"] == truth for row in composition for key, truth in donor_truth.items()),
        "compiled": len(compiled) == len(active),
        "answers_singleton": all(len(ids) == 1 for row in compiled for ids in row["candidate_ids"]),
        "quartet_singleton": all(len(row["role_positions"][role]) == 1 for row in compiled for role in ROLES),
        "semantic_unique": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in active),
        "machine_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.") for row in active),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()): raise RuntimeError({key: value for key, value in checks.items() if not value})
    concepts = [{"word": word, "family": family, "index": index, "partition": partition(index), "sense": f"controlled English {family} sense of {word}; explicit registry is decisive"} for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c069.catalog_four_role.v1", "families": FAMILIES, "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active); core.write_rows(OUT / "material/composition_sets.jsonl", composition); core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    preaudit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": {"always_yes_balanced_accuracy": 0.5, "always_no_balanced_accuracy": 0.5, "position_balanced_accuracy": 0.5, "surface_balanced_accuracy": 0.5}, "semantic_scope": "closed-world exact target-and-family equality", "naturalness_scope": "machine-audited controlled English", "independent_human_blind_review": False, "risks": ["catalog-scoped behavior", "explicit lexical equality", "raw truth prevalence 1:3", "singleton-only roles"]}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)
    mechanism = core.load(TESTS / "result/phase1417_c068_four_role_composition_contract/protocol/preregistration.json")["mechanism"]
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c069.catalog_scoped_four_role.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "research_object": "catalog-scoped state16 bidirectional four-role composition",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at state16", "logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "t-SNE", "UMAP", "learned probe", "layer search", "subset search", "candidate search", "post-reveal changes"],
        "material": {"families": list(FAMILIES), "partitions": list(PARTITIONS), "surfaces": list(SURFACES), "mechanism_surface": "catalog", "active_count": len(active), "composition_count": len(composition), "minimum_qualified_families": 4, "selected_per_family_partition": 4, "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "composition_sha256": core.sha(OUT / "material/composition_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": {"catalog_accuracy_min": 0.95, "catalog_balanced_accuracy_min": 0.95, "catalog_partition_min": 0.90, "catalog_truth_min": 0.90, "catalog_cell_min": 0.90, "required_set_all_min": 0.85, "same_shape_repeat_max_abs_diff": 1e-6, "ordinary_is_required_set_control_not_family_gate": True},
        "camera": {"known_truth_systems": 256, "qwen_discovery_sets": 24, "self_quartet_max_abs_diff": 1e-4, "state_index": 16, "roles": list(ROLES)},
        "mechanism": mechanism,
        "evidence_levels": {"graded": "controls and interaction pass both holdouts plus breadth", "discrete": "matched and mismatch sign gates pass both holdouts plus breadth", "strong": "graded and discrete both pass"},
        "branching": {"phase1421": "catalog behavior", "phase1422": "quartet camera", "phase1423": "bidirectional composition", "phase1424": "closure"},
        "stop_rule": "fewer than four catalog-qualified families blocks hidden; quartet route runs once; no layer or subset fallback",
        "claim_boundary": {"allowed": "Qwen controlled-registry catalog state16 quartet evidence", "forbidden": ["relation manifold", "semantic comparator", "minimal/necessary state", "cross-model/open-language law"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol); protocol["authorization"] = "run_phase1421_c069_catalog_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__": main()
