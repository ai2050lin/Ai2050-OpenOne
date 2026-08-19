#!/usr/bin/env python3
"""Independent zero-model audit for Phase1313; never imports the main script."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PHASE = 1313
CAMPAIGN = "C035"
OUT = T / "result/phase1313_c035_semantic_position_cut_contract"
P = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_new_world_cases.jsonl"
MATERIAL = OUT / "material/frozen_position_cut_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_position_cut_calibration.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"
PARENT = T / "result/phase1312_c034_upstream_selective_rescue"
MAIN = T / "phase1313_c035_semantic_position_cut_contract.py"
SCRIPT = Path(__file__).resolve()
ATTRS = ("temperature", "texture", "origin", "condition", "category", "priority")
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("registry_prose", "registry_ledger")
PANELS = ("active", "matched_null")


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def minimal_sets(damaging: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    result = []
    for cut in sorted(damaging, key=lambda x: (len(x), x)):
        if not any(set(smaller).issubset(cut) for smaller in result):
            result.append(cut)
    return result


def recompute_camera() -> dict[str, float]:
    sites = tuple(range(6))
    cuts = [cut for size in range(1, 7) for cut in itertools.combinations(sites, size)]
    hits = Counter()
    totals = Counter()
    twins = []
    for split in ("discovery", "confirmation"):
        for morphology in ("single_required", "serial_required_pair", "redundant_pair", "readable_bypass", "response_twin"):
            for replicate in range(32):
                order = sorted(sites, key=lambda site: digest([split, morphology, replicate, site]))
                a, b = order[:2]
                if morphology == "single_required":
                    damaging = [cut for cut in cuts if a in cut]
                    expected = "single_required"
                elif morphology == "serial_required_pair":
                    damaging = [cut for cut in cuts if a in cut or b in cut]
                    expected = "serial_required_pair"
                elif morphology == "redundant_pair":
                    damaging = [cut for cut in cuts if a in cut and b in cut]
                    expected = "redundant_pair"
                else:
                    damaging = []
                    expected = "readable_nonessential_or_unregistered_bypass"
                mins = minimal_sets(damaging)
                if not mins:
                    predicted = "readable_nonessential_or_unregistered_bypass"
                elif len(mins) == 1 and len(mins[0]) == 1:
                    predicted = "single_required"
                elif len(mins) > 1 and all(len(x) == 1 for x in mins):
                    predicted = "serial_required_pair"
                elif len(mins) == 1 and len(mins[0]) > 1:
                    predicted = "redundant_pair"
                else:
                    predicted = "abstain_unregistered_morphology"
                hits[expected] += int(predicted == expected)
                totals[expected] += 1
                if morphology == "response_twin":
                    twins.append(predicted == "readable_nonessential_or_unregistered_bypass")
    result = {key: hits[key] / totals[key] for key in sorted(totals)}
    result["response_twin_origin_abstention_fraction"] = sum(twins) / len(twins)
    return result


def main() -> None:
    protocol = load(P)
    source = rows(SOURCE)
    material = rows(MATERIAL)
    machine = load(MACHINE)
    naturalness = load(NATURALNESS)
    calibration = load(CALIBRATION)
    final = load(FINAL)
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)},
        protocol["source_hashes"])
    add(checks, "parent_closed_audited",
        load(PARENT / "analysis/final.json").get("authorization") == "close_c034_at_upstream_rescue_boundary"
        and load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"), "C034")
    add(checks, "artifact_hashes", protocol["material"]["source_sha256"] == sha(SOURCE)
        and protocol["material"]["pair_sha256"] == sha(MATERIAL)
        and protocol["material"]["naturalness_sha256"] == sha(NATURALNESS)
        and protocol["known_truth_camera"]["calibration_sha256"] == sha(CALIBRATION), "all frozen")
    add(checks, "source_count", len(source) == 1728, len(source))
    add(checks, "pair_count", len(material) == 432, len(material))
    add(checks, "source_partition_balance", Counter(x["partition"] for x in source) == Counter({p: 576 for p in PARTITIONS}),
        dict(Counter(x["partition"] for x in source)))
    add(checks, "pair_partition_balance", Counter(x["partition"] for x in material) == Counter({p: 144 for p in PARTITIONS}),
        dict(Counter(x["partition"] for x in material)))
    add(checks, "pair_panel_balance", Counter(x["panel"] for x in material) == Counter({p: 216 for p in PANELS}),
        dict(Counter(x["panel"] for x in material)))
    add(checks, "attribute_surface_balance",
        all(sum(x["attribute"] == a and x["surface"] == s for x in material) == 36 for a in ATTRS for s in SURFACES),
        "36 pairs per attribute-surface cell")
    semantic_unique = all(
        sum(fields[x["attribute"]] == x["target_value"] for fields in x["assignments"].values()) == 1
        for x in source
    )
    add(checks, "semantic_uniqueness", semantic_unique, len(source))
    add(checks, "naturalness", naturalness["all_checks_passed"] and naturalness["semantic_uniqueness"], naturalness)
    ta = machine["token_audit"]
    add(checks, "fresh_lexicon", not ta["prior_name_overlap"] and not ta["prior_value_overlap"],
        {"names": ta["prior_name_overlap"], "values": ta["prior_value_overlap"]})
    add(checks, "token_contract", ta["all_candidates_single_token"] and ta["all_values_single_token"]
        and ta["same_shape_and_site_alignment_within_pairs"], ta["site_count_ranges"])
    pair_structure = all(
        len(x["states"]) == 2 and len(x["identity_positions"]) == 2
        and all(set(state["positions"]) == {"query_attribute", "query_value", "query_end", "answer_boundary",
                                                     "record_entities", "record_queried_values"} for state in x["states"])
        and x["states"][0]["positions"] == x["states"][1]["positions"]
        and len(x["states"][0]["ids"]) == len(x["states"][1]["ids"])
        for x in material
    )
    add(checks, "position_pair_structure", pair_structure, "aligned role sets")
    recomputed = recompute_camera()
    reported = dict(calibration["class_accuracy"])
    reported["response_twin_origin_abstention_fraction"] = calibration["response_twin_origin_abstention_fraction"]
    add(checks, "independent_cut_camera", reported == recomputed and all(v == 1.0 for v in recomputed.values()), recomputed)
    add(checks, "independent_typed_camera", calibration["typed_multi_readout_accuracy"] == 1.0
        and calibration["single_target_generic_typed_collision_fraction"] == 1.0, calibration["typed_multi_readout_accuracy"])
    add(checks, "machine_gate", machine["all_machine_checks_passed"], machine["program_audit"])
    expected_sets = {
        "query_end_only": ["query_end"],
        "query_bundle": ["query_attribute", "query_value", "query_end"],
        "record_bundle": ["record_entities", "record_queried_values"],
        "full_registered": ["query_attribute", "query_value", "query_end", "record_entities", "record_queried_values"],
    }
    add(checks, "frozen_cut_sets", protocol["position_cut"]["depth"] == 14
        and protocol["position_cut"]["sets"] == expected_sets, protocol["position_cut"])
    add(checks, "strict_branches_and_stops", protocol["branches"]["phase1315_fail"] == "close_c035_at_registered_cut_boundary"
        and protocol["branches"]["phase1316_any_verdict"] == "close_c035"
        and len(protocol["hard_stops"]) == 6, protocol["branches"])
    authorization = "phase1314_qwen3_behavior_only" if all(x["passed"] for x in checks) else "none"
    add(checks, "final_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == (authorization != "none"), final)
    passed = all(x["passed"] for x in checks)
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit_stage": "independent_zero_model_final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
        "all_checks_passed": passed, "authorization": "phase1314_qwen3_behavior_only" if passed else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"passed": result["passed_count"], "total": result["total_count"],
                     "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
