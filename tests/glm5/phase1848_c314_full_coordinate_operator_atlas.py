#!/usr/bin/env python3
"""C314: synthesize all-coordinate operator passports and freeze causal strata."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def main() -> None:
    parents = {campaign: common.core.load(common.OUTS[campaign] / "analysis/final.json") for campaign in ("C311", "C312", "C313")}
    checks = {"all_parents": all(value["all_checks_passed"] for value in parents.values()), "all_coordinate_archives": True, "training_only_selector": True}
    protocol = {
        "status": "full_coordinate_atlas_frozen",
        "atlas_channels": ["train_interaction_mean", "train_interaction_std", "sixth_interaction_mean", "sixth_interaction_std", "family_abs_error", "global_abs_error", "family_minus_global_abs_error", "train_signal_to_dispersion"],
        "causal_stratum_selector": "for each family choose q in 1..35 maximizing training-only mean(abs(interaction_mean))/(mean(interaction_std)+1e-6), averaged over all six roles and all 2560 coordinates",
        "forbidden_selector_inputs": ["sixth-material outcomes", "candidate logits", "causal outcomes", "Top-K coordinates"],
        "claim_boundary": "The atlas is a coordinate-complete descriptive passport. The selected checkpoint is a frozen intervention address, not a discovered semantic locus.",
    }
    out = common.prepare("C314", protocol, checks)
    atlas = np.zeros((6, 8, 37, 6, 2560), np.float32)
    strata = []
    sixth_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    sixth_index = common.core.rows(common.SIXTH_INDEX)
    all_train = {}
    for family in common.FAMILIES:
        values = []
        for state_path, index_path, _material in common.TRAIN_ROLE_SOURCES:
            arrays, _groups = common.factorial_arrays(np.load(state_path, mmap_mode="r"), common.core.rows(index_path), family)
            values.append(arrays["interaction"])
        all_train[family] = np.concatenate(values, axis=0)
    global_mean = np.concatenate(list(all_train.values()), axis=0).mean(axis=0)
    for family_i, family in enumerate(common.FAMILIES):
        train = all_train[family]
        test, groups = common.factorial_arrays(sixth_states, sixth_index, family)
        truth = test["interaction"]
        train_mean = train.mean(axis=0)
        train_std = train.std(axis=0)
        test_mean = truth.mean(axis=0)
        test_std = truth.std(axis=0)
        family_error = np.mean(np.abs(truth - train_mean[None, ...]), axis=0)
        global_error = np.mean(np.abs(truth - global_mean[None, ...]), axis=0)
        score_field = np.abs(train_mean) / (train_std + 1e-6)
        atlas[family_i] = np.stack((train_mean, train_std, test_mean, test_std, family_error, global_error, global_error - family_error, score_field), axis=0)
        q_scores = score_field.mean(axis=(1, 2))
        q = int(1 + np.argmax(q_scores[1:36]))
        strata.append({"family": family, "q": q, "checkpoint": common.CANONICAL_CHECKPOINTS[q], "training_score": float(q_scores[q]), "groups_train": len(train), "groups_sixth": len(groups), "roles": list(common.ROLES), "coordinates": common.DIM})
        print(f"[C314] {family}: q={q} score={q_scores[q]:.6f}", flush=True)
    np.save(out / "analysis/operator_passports.float32.npy", atlas)
    common.core.save(out / "protocol/selected_causal_strata.json", {"selector": protocol["causal_stratum_selector"], "strata": strata})
    q_role_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        for q, checkpoint in enumerate(common.CANONICAL_CHECKPOINTS):
            for role_i, role in enumerate(common.ROLES):
                q_role_rows.append({"family": family, "q": q, "checkpoint": checkpoint, "role": role, "train_abs_mean": float(np.mean(np.abs(atlas[family_i, 0, q, role_i]))), "specificity_advantage": float(np.mean(atlas[family_i, 6, q, role_i])), "signal_to_dispersion": float(np.mean(atlas[family_i, 7, q, role_i]))})
    common.core.write_rows(out / "analysis/checkpoint_role_summary.jsonl", q_role_rows)
    headline = {"status": "full_coordinate_operator_atlas_closed", "selected_causal_strata": strata, "source_results": {campaign: {"breadth_gate_passed": value["headline"].get("breadth_gate_passed")} for campaign, value in parents.items()}, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C314", headline, {"atlas_shape": list(atlas.shape) == [6, 8, 37, 6, 2560], "strata": len(strata) == 6, "q_range": all(1 <= row["q"] <= 35 for row in strata), "summary_rows": len(q_role_rows) == 6 * 37 * 6, "finite": bool(np.isfinite(atlas).all())}, "C315_frozen_stratum_causal_test")


if __name__ == "__main__":
    main()
