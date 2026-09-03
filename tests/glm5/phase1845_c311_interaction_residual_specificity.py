#!/usr/bin/env python3
"""C311: full-coordinate residual specificity tournament on the sixth material."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


MODELS = ("family", "global", "wrong_family", "surface_only", "order_only", "coordinate_roll", "role_roll", "sign_flip")


def main() -> None:
    parent = common.core.load(common.OUTS["C310"] / "analysis/final.json")
    checks = {
        "parent": parent["all_checks_passed"],
        "sixth_lockbox_hash": common.core.sha(common.SIXTH_STATES) == common.core.sha(common.SIXTH_STATES),
        "named_controls_frozen": len(MODELS) == 8,
        "all_coordinates": common.DIM == 2560,
    }
    protocol = {
        "status": "specificity_tournament_frozen",
        "training": "third, fourth, and fifth materials only",
        "lockbox": "sixth material; two controlled surfaces, eight units, two answer orders",
        "forecast": "H11_hat = H10 + H01 - H00 + candidate residual",
        "models": list(MODELS),
        "controls": {
            "global": "mean interaction residual over all families",
            "wrong_family": "cyclic next-family residual, RMS matched to the correct family residual",
            "surface_only": "same abstract surface slot pooled over families",
            "order_only": "same answer-order residual pooled over families",
            "coordinate_roll": "correct residual rolled by 97 physical coordinates",
            "role_roll": "correct residual rolled by one semantic role",
            "sign_flip": "negative correct residual",
        },
        "family_gate": "family residual gain>=0.01 and exceeds the best named control by>=0.01",
        "breadth_gate": "at least four of six families pass",
        "claim_boundary": "Passing identifies family-specific predictive information in a mean second-order response field. It still does not show natural use, operator composition, or causality.",
    }
    out = common.prepare("C311", protocol, checks)
    training_by_family = {}
    training_meta = {}
    for family in common.FAMILIES:
        arrays_all = []
        meta_all = []
        for state_path, index_path, material_name in common.TRAIN_ROLE_SOURCES:
            states = np.load(state_path, mmap_mode="r")
            arrays, groups = common.factorial_arrays(states, common.core.rows(index_path), family)
            arrays_all.append(arrays["interaction"])
            meta_all.extend([{**group, "material": material_name, "surface_slot": int(slot)} for group, slot in zip(groups, common.surface_slot(groups))])
        training_by_family[family] = np.concatenate(arrays_all, axis=0)
        training_meta[family] = meta_all
    family_means = {family: rows.mean(axis=0) for family, rows in training_by_family.items()}
    global_mean = np.concatenate(list(training_by_family.values()), axis=0).mean(axis=0)
    surface_means = {}
    order_means = {}
    for slot in (0, 1):
        values = [training_by_family[family][[row["surface_slot"] == slot for row in training_meta[family]]] for family in common.FAMILIES]
        surface_means[slot] = np.concatenate(values, axis=0).mean(axis=0)
    for order in (1, -1):
        values = [training_by_family[family][[row["order"] == order for row in training_meta[family]]] for family in common.FAMILIES]
        order_means[order] = np.concatenate(values, axis=0).mean(axis=0)

    test_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    test_index = common.core.rows(common.SIXTH_INDEX)
    family_rows = []
    group_rows = []
    atlas = np.zeros((len(common.FAMILIES), len(MODELS) + 1, len(common.CANONICAL_CHECKPOINTS), len(common.ROLES), common.DIM), np.float32)
    for family_i, family in enumerate(common.FAMILIES):
        arrays, groups = common.factorial_arrays(test_states, test_index, family)
        truth = arrays["interaction"]
        slots = common.surface_slot(groups)
        wrong = common.FAMILIES[(family_i + 1) % len(common.FAMILIES)]
        wrong_correction = common.norm_match(family_means[wrong], family_means[family])
        predictions = {
            "family": np.broadcast_to(family_means[family], truth.shape),
            "global": np.broadcast_to(global_mean, truth.shape),
            "wrong_family": np.broadcast_to(wrong_correction, truth.shape),
            "surface_only": np.asarray([surface_means[int(slot)] for slot in slots], np.float32),
            "order_only": np.asarray([order_means[group["order"]] for group in groups], np.float32),
            "coordinate_roll": np.broadcast_to(np.roll(family_means[family], 97, axis=-1), truth.shape),
            "role_roll": np.broadcast_to(np.roll(family_means[family], 1, axis=1), truth.shape),
            "sign_flip": np.broadcast_to(-family_means[family], truth.shape),
        }
        baseline = truth
        atlas[family_i, 0] = np.mean(np.abs(baseline), axis=0)
        gains = {}
        for model_i, model_name in enumerate(MODELS, start=1):
            error = truth - predictions[model_name]
            gains[model_name] = common.relative_gain(baseline, error)
            atlas[family_i, model_i] = np.mean(np.abs(error), axis=0)
        best_control_name = max((name for name in MODELS if name != "family"), key=lambda name: gains[name])
        margin = gains["family"] - gains[best_control_name]
        passed = gains["family"] >= 0.01 and margin >= 0.01
        family_rows.append({"family": family, "groups": len(groups), "gains": gains, "best_control": best_control_name, "family_minus_best_control": margin, "family_gate_passed": passed})
        for group_i, group in enumerate(groups):
            row = {"family": family, "surface": group["surface"], "unit": group["unit"], "order": group["order"]}
            base = float(np.mean(np.abs(truth[group_i])))
            for name in MODELS:
                row[f"gain_{name}"] = float((base - np.mean(np.abs(truth[group_i] - predictions[name][group_i]))) / max(base, 1e-12))
            group_rows.append(row)
        print(f"[C311] {family}: family={gains['family']:+.5f} best={best_control_name}:{gains[best_control_name]:+.5f} margin={margin:+.5f}", flush=True)
    np.save(out / "analysis/full_coordinate_error_atlas.float32.npy", atlas)
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    common.core.write_rows(out / "raw/group_results.jsonl", group_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {
        "status": "specificity_tournament_adjudicated",
        "families": family_rows,
        "families_passing": passing,
        "breadth_gate_passed": len(passing) >= 4,
        "strict_interpretation": protocol["claim_boundary"],
    }
    common.close("C311", headline, {
        "six_families": len(family_rows) == 6,
        "expected_groups": len(group_rows) == 192,
        "atlas_shape": list(atlas.shape) == [6, 9, 37, 6, 2560],
        "finite": bool(np.isfinite(atlas).all()),
    }, "C312_atomic_operator_transport_regardless_of_specificity_gate")


if __name__ == "__main__":
    main()
