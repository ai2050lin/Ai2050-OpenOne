#!/usr/bin/env python3
"""C292: independently audit C277-C291 and close the extended campaign."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core = common.core
OUT = common.RESULT / "phase1826_c292_joint_response_campaign_independent_audit"
C290 = common.RESULT / "phase1824_c290_training_supported_causal_strata"
C291 = common.RESULT / "phase1825_c291_training_qualified_joint_word_causal"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c289_joint_response_campaign_atlas.json"
CANDIDATES = ("relation_query", "primary_relation_query", "all_six_roles")


def final_path(campaign: int) -> Path:
    if campaign <= 289:
        return common.OUTS[f"C{campaign}"] / "analysis/final.json"
    return (C290 if campaign == 290 else C291) / "analysis/final.json"


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1826,
        "campaign": "C292",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independent_audit_frozen",
        "scope": "C277-C291 final files, C280 raw coordinate counts, C291 raw samples, and C289 heatmap asset",
        "rule": "Recompute primary arithmetic without importing producer reports as conclusions.",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    finals = {f"C{i}": core.load(final_path(i)) for i in range(277, 292)}
    phases = [int(finals[f"C{i}"]["phase"]) for i in range(277, 292)]

    c280 = finals["C280"]["headline"]
    counts = np.load(common.OUTS["C280"] / "analysis/coordinate_correct_union_counts.uint32.npy", mmap_mode="r")
    recomputed = {}
    prediction_match = True
    for fi, family in enumerate(common.FAMILIES):
        recomputed[family] = {}
        final_family = next(row for row in c280["families"] if row["family"] == family)
        for ci, candidate in enumerate(CANDIDATES):
            correct = int(np.asarray(counts[fi, :, ci, 0], np.uint64).sum())
            union = int(np.asarray(counts[fi, :, ci, 1], np.uint64).sum())
            score = float(correct / max(union, 1))
            reported = float(final_family["candidates"][candidate]["completion"]["signed_jaccard"])
            prediction_match &= abs(score - reported) < 1e-12
            recomputed[family][candidate] = {
                "correct": correct,
                "union": union,
                "signed_jaccard": score,
                "reported": reported,
            }

    c291_samples = core.rows(C291 / "raw/sample_results.jsonl")
    c291_recomputed = {}
    causal_match = True
    for family in finals["C291"]["headline"]["eligible_families"]:
        rows = [row for row in c291_samples if row["family"] == family]
        values = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("deletion_drop", "correct_recovery_error", "correct_minus_best_wrong")
        }
        reported = next(row for row in finals["C291"]["headline"]["families"] if row["family"] == family)
        causal_match &= all(abs(values[key] - float(reported[key])) < 1e-12 for key in values)
        c291_recomputed[family] = {"recomputed": values, "reported": reported}

    asset = core.load(ASSET)
    asset_lengths = np.asarray([len(row["values"]) for row in asset["rows"]], np.int32)
    c290_passing = finals["C290"]["headline"]["passing_families"]
    c291_tested = finals["C291"]["headline"]["eligible_families"]
    checks = {
        "all_finals_closed": all(bool(row["all_checks_passed"]) for row in finals.values()),
        "phase_continuity": phases == list(range(1811, 1826)),
        "c280_raw_shape": list(counts.shape) == [6, 36, 3, 2, 2560],
        "c280_raw_scores_match_report": bool(prediction_match),
        "c280_all_candidates_all_families": all(row["families_passing"] == 6 and row["broad_gate_passed"] for row in c280["candidate_summary"]),
        "c281_all_candidates_all_families": all(row["families_passing"] == 6 and row["broad_gate_passed"] for row in finals["C281"]["headline"]["candidate_summary"]),
        "c285_preserved_as_no_test": "no_test" in finals["C285"]["headline"]["status"] and finals["C285"]["headline"]["eligible_coordinates_total"] == 1,
        "c290_branchwise_qualification": c290_passing == ["translation", "nested_attitude"],
        "c291_exact_authorized_branches": c291_tested == c290_passing and set(row["family"] for row in c291_samples) == set(c290_passing),
        "c291_aggregates_match_raw": bool(causal_match),
        "c291_no_causal_family_passed": finals["C291"]["headline"]["causal_families_passing_count"] == 0,
        "c288_three_pair_topology": len(finals["C288"]["headline"]["pairs"]) == 3 and all(row["pair_gate_passed"] for row in finals["C288"]["headline"]["pairs"]),
        "heatmap_schema": asset["schema"] == "c289_joint_response_campaign_atlas.v1",
        "heatmap_all_coordinates": len(asset["dimensions"]) == 2560 and bool(np.all(asset_lengths == 2560)),
        "heatmap_sources": {row["source"] for row in asset["rows"]} == {
            "c280_joint_word_prediction", "c282_c284_factorial_interaction", "c278_fifth_material_edit_response"
        },
    }
    audit = {
        "phase": 1826,
        "campaign": "C292",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "recomputed_c280": recomputed,
        "recomputed_c291": c291_recomputed,
        "source_sha256": {f"C{i}": core.sha(final_path(i)) for i in range(277, 292)},
        "heatmap_sha256": core.sha(ASSET),
    }
    core.save(OUT / "audit/independent_audit.json", audit)
    report = {
        "phase": 1826,
        "campaign": "C292",
        "status": "extended_campaign_independently_audited",
        "observational_result": "C280-C281 passed all six families for all three frozen role-event words: a finite signed-event automaton predicts unseen one-step and autonomous checkpoint trajectories.",
        "composition_result": "C282-C284 found broad factorial residuals, but these are descriptive interaction fields rather than identified composition operators.",
        "causal_result": "C285 was a no-test. C290 repaired local qualification for translation and nested attitude, but C291 deletion/rescue failed both registered causal gates; the tested same-coordinate source-role write interface is not supported.",
        "cross_model_result": "C288 found significant anonymous role-topology similarity for all three model pairs, without physical coordinate alignment or causal equivalence.",
        "theory_result": "Forward predictability, local controllability, and cross-model functional topology are distinct evidence axes. Existing finite-state, conditional dynamical-system and causal-intervention mathematics express the current evidence; the new-foundation mathematics gate remains closed.",
        "strict_conclusion": "The durable object is a partial, role- and checkpoint-conditioned response automaton over distributed physical coordinates. Its causal realization remains unresolved and is not the registered same-coordinate source-role coalition.",
        "next_stage": "Observe amplitude-conditioned and coordinate-coupled transitions without forcing same-coordinate transport; pre-register causal targets only after a training-supported intervention compiler exists.",
    }
    core.save(OUT / "analysis/summary.json", report)
    final_checks = {
        "independent_audit": audit["all_checks_passed"],
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": 1826,
        "campaign": "C292",
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": report,
        "next_authorization": report["next_stage"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
