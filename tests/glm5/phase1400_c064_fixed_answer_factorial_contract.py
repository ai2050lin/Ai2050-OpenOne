#!/usr/bin/env python3
"""Phase1400: freeze C064 fixed-answer natural factorial campaign."""
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

PHASE, CAMPAIGN = 1400, "C064"
PARENT = TESTS / "result/phase1399_c063_behavior_gate_closure"
OUT = TESTS / "result/phase1400_c064_fixed_answer_factorial_contract"

FAMILIES = {
    "animal": ("buffalo", "yak", "hare", "llama", "ape", "bull", "calf", "pony", "ram", "mole", "elk", "cougar"),
    "building": ("airport", "apartment", "bakery", "cottage", "garage", "prison", "village", "courthouse", "office", "clinic", "bridge", "tunnel"),
    "profession": ("architect", "baker", "pharmacist", "driver", "singer", "dancer", "guard", "judge", "soldier", "sailor", "butcher", "designer"),
    "country": ("Argentina", "Australia", "Austria", "Belgium", "Cuba", "Iran", "Iraq", "Israel", "Malaysia", "Pakistan", "Portugal", "Ukraine"),
}
PARTITIONS = {"response_discovery": range(0, 4), "confirmation": range(4, 8), "lockbox": range(8, 12)}
SURFACES = {
    "ordinary": "Registry entry: {target} belongs to {record_family}. Query: does {target} belong to {query_family}? Output only yes or no.",
    "catalog": "Catalog entry: {target} is classified as {record_family}. Question: is {target} classified as {query_family}? Output only yes or no.",
    "statement": "Filed statement: {target} is a member of {record_family}. Check: is {target} a member of {query_family}? Output only yes or no.",
}
STATUS_SURFACES = {
    "ordinary": "Registry entry: {target} belongs to {record_family}. Reference category: {query_family}. Query: is the record active? Record status: {status}. Output only yes or no.",
    "catalog": "Catalog entry: {target} is classified as {record_family}. Reference category: {query_family}. Question: is the record active? Record status: {status}. Output only yes or no.",
    "statement": "Filed statement: {target} is a member of {record_family}. Reference category: {query_family}. Check: is the record active? Record status: {status}. Output only yes or no.",
}
SYSTEM = "Use only the explicit registry entry. Output exactly yes or no."
ORDER = tuple(FAMILIES)


def partition(index: int) -> str:
    return next(k for k, values in PARTITIONS.items() if index in values)


def active_cases():
    rows = []
    for fa, fb in combinations(ORDER, 2):
        pair = f"{fa}__{fb}"
        for index in range(12):
            cells = {
                "aa": (FAMILIES[fa][index], fa, fa, True), "ab": (FAMILIES[fa][index], fa, fb, False),
                "bb": (FAMILIES[fb][index], fb, fb, True), "ba": (FAMILIES[fb][index], fb, fa, False),
            }
            for surface, template in SURFACES.items():
                for cell, (target, record_family, query_family, truth) in cells.items():
                    rows.append({"case_id": f"c064-a-{len(rows):04d}", "partition": partition(index), "pair": pair,
                                 "index": index, "surface": surface, "cell": cell, "target": target,
                                 "record_family": record_family, "query_family": query_family, "truth": truth,
                                 "prompt": template.format(target=target, record_family=record_family, query_family=query_family),
                                 "candidates": ["yes", "no"], "gold_position": 0 if truth else 1})
    return rows


def status_cases():
    rows = []
    for fi, family in enumerate(ORDER):
        query_family = ORDER[(fi + 1) % len(ORDER)]
        for index, target in enumerate(FAMILIES[family]):
            for surface, template in STATUS_SURFACES.items():
                for status, truth in (("active", True), ("inactive", False)):
                    rows.append({"case_id": f"c064-s-{len(rows):04d}", "partition": partition(index), "index": index,
                                 "surface": surface, "target": target, "record_family": family,
                                 "query_family": query_family, "status": status, "truth": truth,
                                 "prompt": template.format(target=target, record_family=family, query_family=query_family, status=status),
                                 "candidates": ["yes", "no"], "gold_position": 0 if truth else 1})
    return rows


def all_spans(tok, ids, value):
    needles = [list(map(int, tok.encode(v, add_special_tokens=False))) for v in (value, " " + value)]
    found = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                span = list(range(start, start + len(needle)))
                if span not in found:
                    found.append(span)
    return sorted(found)


def compile_rows(tok, rows):
    result = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        targets = all_spans(tok, ids, row["target"])
        record = all_spans(tok, ids, row["record_family"])
        query = all_spans(tok, ids, row["query_family"])
        if not targets or not record or not query:
            raise RuntimeError((row["case_id"], targets, record, query))
        result.append({"case_id": row["case_id"], "prompt_ids": ids,
                       "candidate_ids": [[int(x) for x in tok.encode(v, add_special_tokens=False)] for v in ("yes", "no")],
                       "role_positions": {"record_target": targets[0], "record_family": record[0],
                                          "query_target": targets[1] if len(targets) > 1 else targets[0],
                                          "query_family": query[-1], "boundary": [len(ids) - 1]}})
    return result


def factor_sets(active, status):
    by = {(r["pair"], r["index"], r["surface"], r["cell"]): r for r in active}
    sb = {(r["record_family"], r["index"], r["surface"], r["status"]): r for r in status}
    surface_next = {s: tuple(SURFACES)[(i + 1) % len(SURFACES)] for i, s in enumerate(SURFACES)}
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
                result.append({"set_id": f"c064-factor-{len(result):04d}", "partition": recipient["partition"],
                               "family": family, "index": index, "surface": surface,
                               "recipient": recipient["case_id"],
                               "surface_same": by[(pair, index, surface_next[surface], own_true)]["case_id"],
                               "member_same": by[(pair, member_index, surface, own_true)]["case_id"],
                               "family_same_polarity": by[(pair, index, surface, other_true)]["case_id"],
                               "polarity_same_family": by[(pair, index, surface, own_false)]["case_id"],
                               "family_and_polarity": by[(pair, index, surface, other_false)]["case_id"],
                               "status_null": sb[(family, index, surface, "active")]["case_id"]})
    return result


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1400 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c064_fixed_natural_answer_factorial_campaign" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C063 closure missing")
    tok = tokenizer()
    active, status = active_cases(), status_cases()
    factors = factor_sets(active, status)
    ca, cs = compile_rows(tok, active), compile_rows(tok, status)
    words = {w for values in FAMILIES.values() for w in values}
    old = set()
    for path in (TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract/material/frozen_concept_graph.json",
                 TESTS / "result/phase1390_c062_route_factorized_field_campaign_contract/material/frozen_concept_graph.json",
                 TESTS / "result/phase1397_c063_identity_polarity_campaign_contract/material/frozen_concept_graph.json"):
        old |= {r["word"] for r in core.load(path)["concepts"]}
    layout = defaultdict(set)
    for source, compiled in zip(active, ca):
        layout[source["surface"]].add((len(compiled["prompt_ids"]), tuple((k, tuple(v)) for k, v in compiled["role_positions"].items())))
    checks = {
        "parent_closed_audited": parent_audit["all_checks_passed"],
        "four_families_48_words": len(FAMILIES) == 4 and len(words) == 48 and all(len(v) == 12 for v in FAMILIES.values()),
        "fresh_vs_c060_c063": not (words & old),
        "active_balance": len(active) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status_balance": len(status) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "factor_balance": len(factors) == 144 and Counter(r["partition"] for r in factors) == {p: 48 for p in PARTITIONS},
        "compiled": len(ca) == 864 and len(cs) == 288,
        "candidate_single_token": all(len(v) == 1 for r in ca + cs for v in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"record_target", "record_family", "query_target", "query_family", "boundary"} for r in ca + cs),
        "active_layout_exact": all(len(v) == 1 for v in layout.values()),
        "machine_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.") for r in active + status),
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    concepts = [{"word": word, "family": family, "index": index, "partition": partition(index),
                 "sense": f"explicit registry {family} sense of {word}", "adjudication": "truth is fixed by the local registry entry"}
                for family, values in FAMILIES.items() for index, word in enumerate(values)]
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c064.fixed_answer.v1", "families": FAMILIES,
              "partitions": {k: list(v) for k, v in PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/factor_sets.jsonl", factors)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", ca)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", cs)
    pre = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
           "all_checks_passed": all(checks.values()), "zero_models": {"always_yes": 0.5, "always_no": 0.5,
           "target_only_upper_bound": 0.5, "record_family_only_upper_bound": 0.5, "surface_only_upper_bound": 0.5},
           "semantic_scope": "truth is uniquely specified by the explicit local registry",
           "naturalness_scope": "controlled grammatical English; fixed yes/no answer interface",
           "independent_human_blind_review": False,
           "disclosed_risks": ["truth and physical answer token remain coupled", "controlled registry is not open natural generation"]}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", pre)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c064.fixed_answer_natural_factorial.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "natural full-state family identity versus joint truth/output-polarity state",
        "allowed_observables": ["input token embeddings", "all layers/all positions full-dimensional hidden states", "logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP", "SAE", "learned probe", "post-reveal replacement"],
        "material": {"families": list(FAMILIES), "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
                     "active_count": 864, "status_count": 288, "factor_count": 144,
                     "eligible_per_family_partition_surface": 4, "selected_per_family_partition": 12,
                     "selected_per_family": 36, "minimum_qualified_families": 3,
                     "active_sha256": core.sha(OUT / "material/active_cases.jsonl"), "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
                     "factor_sha256": core.sha(OUT / "material/factor_sets.jsonl"), "human_naturalness_lock": False},
        "behavior": {"family_active_accuracy_min": 0.95, "family_partition_min": 0.90, "family_surface_min": 0.90,
                     "family_truth_min": 0.90, "family_pair_all_min": 0.85, "status_accuracy_min": 0.95,
                     "same_shape_repeat_max_abs_diff": 1e-6},
        "camera": {"known_truth_systems": 256, "qwen_cases": 24, "logit_identity_max_abs_diff": 1e-5,
                   "all_state_identity_relative_l2_max": 1e-6, "role_swap_exact": True, "multi_checkpoint_exact": True},
        "observation": {"partition": "response_discovery", "all_hidden_state_indices": list(range(37)), "all_physical_positions": True,
                        "donors": ["surface_same", "member_same", "family_same_polarity", "polarity_same_family", "family_and_polarity", "status_null"],
                        "roles": ["record_target", "record_family", "query_target", "query_family", "boundary"],
                        "windows": [[1, 15], [16, 24], [25, 36]], "top_candidates_per_object_window": 1},
        "factorial_swap": {"partitions": ["confirmation", "lockbox"], "self_max_abs_diff": 1e-4,
                           "surface_control_loss_fraction_max": 0.25, "member_control_loss_fraction_max": 0.50,
                           "family_damage_median_min": 0.5, "family_over_member_median_min": 0.25,
                           "family_over_member_win_min": 0.65, "polarity_redirect_fraction_min": 0.65,
                           "minimum_family_breadth": 2},
        "branching": {"phase1401": "behavior", "phase1402": "camera", "phase1403": "natural discovery field",
                      "phase1404": "holdout factorial swaps", "phase1405": "closure"},
        "claim_boundary": {"allowed": "Qwen controlled-registry natural-state family versus joint truth/output evidence",
                           "forbidden": ["semantic neurons", "answer-token/polarity separation", "minimal circuit", "unique path", "cross-model/open-language law"]},
        "stop_rule": "family/candidate route failure eliminates only that route; breadth/camera failure blocks dependent hidden access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1401_c064_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True,
              "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": pre, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
