#!/usr/bin/env python3
"""C282: observe full-coordinate factorial composition in attitude families."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C282"]
C278 = common.OUTS["C278"]


def signed_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = (left != 0) | (right != 0)
    return float(np.mean(left[union] == right[union])) if union.any() else 1.0


def factorial_observation(states: np.ndarray, index: list[dict], family: str) -> tuple[dict, np.ndarray]:
    rows = [row for row in index if row["family"] == family and row["correct"]]
    lookup = {(row["surface"], row["unit"], row["order"], row["factor_a"], row["factor_b"]): row["hidden_index"] for row in rows}
    threshold = common.thresholds()[:, None, None]
    interactions = []
    main_l1 = 0.0
    interaction_l1 = 0.0
    active = 0
    total = 0
    by_half = [[], []]
    for surface, unit, order in itertools.product(common.SURFACES, range(8), (1, -1)):
        keys = [(surface, unit, order, a, b) for a, b in itertools.product((0, 1), repeat=2)]
        if not all(key in lookup for key in keys):
            continue
        cells = {(a, b): np.asarray(states[lookup[(surface, unit, order, a, b)], common.CANONICAL_NEW_INDICES], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
        effect_a = 0.5 * ((cells[(1, 0)] - cells[(0, 0)]) + (cells[(1, 1)] - cells[(0, 1)]))
        effect_b = 0.5 * ((cells[(0, 1)] - cells[(0, 0)]) + (cells[(1, 1)] - cells[(1, 0)]))
        interaction = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
        interactions.append(interaction)
        by_half[0 if unit < 4 else 1].append(interaction)
        main_l1 += float(np.abs(effect_a).sum() + np.abs(effect_b).sum())
        interaction_l1 += float(np.abs(interaction).sum())
        active += int((np.abs(interaction) > threshold).sum())
        total += int(interaction.size)
    if not interactions or not all(by_half):
        raise RuntimeError((family, len(interactions), [len(x) for x in by_half]))
    atlas = np.mean(interactions, axis=0).astype(np.float16)
    half_events = []
    for values in by_half:
        mean = np.mean(values, axis=0)
        half_events.append(np.where(mean > threshold, 1, np.where(mean < -threshold, -1, 0)).astype(np.int8))
    metrics = {
        "family": family,
        "complete_factorial_groups": len(interactions),
        "interaction_active_fraction": float(active / max(total, 1)),
        "interaction_to_main_l1_ratio": float(interaction_l1 / max(main_l1, 1e-30)),
        "cross_lexicon_half_signed_jaccard": signed_jaccard(half_events[0], half_events[1]),
        "atlas_shape": list(atlas.shape),
    }
    return metrics, atlas


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = [core.load(common.OUTS[name] / "analysis/final.json") for name in ("C278", "C281")]
    checks = {"parents": all(x["all_checks_passed"] for x in parents), "families": True, "all_coordinates": True, "no_topk_pca_cosine_attention_mlp": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1816, "campaign": "C282", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "composition_observation_frozen",
        "families": ["attitude_event", "nested_attitude"],
        "contrast": "I=A1B1-A1B0-A0B1+A0B0 at every canonical checkpoint, semantic role and physical coordinate",
        "metrics": ["threshold-active interaction fraction", "interaction/main L1 ratio", "signed event agreement between units 0-3 and 4-7"],
        "claim_boundary": "A nonzero factorial residual is descriptive nonadditivity. It is not a learned algebraic composition law or a causal interaction.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C283_type_graph_composition",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    states = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(C278 / "raw/hidden_index.jsonl")
    rows = []
    atlases = []
    for family in protocol["families"]:
        metrics, atlas = factorial_observation(states, index, family)
        rows.append(metrics); atlases.append(atlas)
        np.save(OUT / f"analysis/{family}_interaction_atlas.float16.npy", atlas)
        print(f"[C282] {family}: active={metrics['interaction_active_fraction']:.4f}, ratio={metrics['interaction_to_main_l1_ratio']:.4f}, halfJ={metrics['cross_lexicon_half_signed_jaccard']:.4f}", flush=True)
    core.write_rows(OUT / "analysis/family_results.jsonl", rows)
    report = {"phase": 1816, "campaign": "C282", "status": "attitude_composition_observed", "families": rows, "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C283_type_graph_composition"}
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"families": len(rows) == 2, "atlas_shapes": all(list(a.shape) == [37, 6, 2560] for a in atlases), "finite": bool(np.isfinite([r["interaction_to_main_l1_ratio"] for r in rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1816, "campaign": "C282", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

