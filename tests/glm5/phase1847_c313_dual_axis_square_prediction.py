#!/usr/bin/env python3
"""C313: test finite-difference semantic/depth squares without claiming a functor."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


OBJECTS = (("A_after_B", "a1"), ("B_after_A", "b1"), ("interaction", "interaction"))


def increments(values: np.ndarray) -> np.ndarray:
    return values[:, 1:] - values[:, :-1]


def main() -> None:
    parent = common.core.load(common.OUTS["C312"] / "analysis/final.json")
    checks = {"parent_complete": parent["all_checks_passed"], "finite_difference_square_only": True, "no_posthoc_checkpoint": True}
    protocol = {
        "status": "dual_axis_square_frozen",
        "object": "For each semantic finite difference Delta_o(q), predict the depth increment Delta_o(q+1)-Delta_o(q) on sixth-material states.",
        "square_defect": "[H(oG,q+1)-H(G,q+1)] - [H(oG,q)-H(G,q)]",
        "models": ["family training mean", "global all-family mean", "cyclic wrong-family RMS-matched mean", "zero increment"],
        "family_gate": "family mean gains>=0.01 over zero for all three objects and exceeds both named semantic controls on at least two objects",
        "claim_boundary": "This is an empirical finite-difference square on observed endpoints. It is not proof of a commuting diagram, functor, or globally defined semantic operator.",
    }
    out = common.prepare("C313", protocol, checks)
    train_means = {family: {} for family in common.FAMILIES}
    pooled = {name: [] for name, _key in OBJECTS}
    for family in common.FAMILIES:
        for name, key in OBJECTS:
            values = []
            for state_path, index_path, _material in common.TRAIN_ROLE_SOURCES:
                arrays, _groups = common.factorial_arrays(np.load(state_path, mmap_mode="r"), common.core.rows(index_path), family)
                values.append(increments(arrays[key]))
            merged = np.concatenate(values, axis=0)
            train_means[family][name] = merged.mean(axis=0)
            pooled[name].append(merged)
    global_means = {name: np.concatenate(values, axis=0).mean(axis=0) for name, values in pooled.items()}
    test_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    test_index = common.core.rows(common.SIXTH_INDEX)
    atlas = np.zeros((6, 3, 4, 36, 6, 2560), np.float32)
    family_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        arrays, groups = common.factorial_arrays(test_states, test_index, family)
        wrong = common.FAMILIES[(family_i + 1) % 6]
        object_rows = []
        superiority = 0
        all_positive = True
        for object_i, (name, key) in enumerate(OBJECTS):
            truth = increments(arrays[key])
            correct = train_means[family][name]
            predictions = {
                "family": correct,
                "global": global_means[name],
                "wrong_family": common.norm_match(train_means[wrong][name], correct),
            }
            atlas[family_i, object_i, 0] = np.mean(np.abs(truth), axis=0)
            gains = {}
            for model_i, (model, prediction) in enumerate(predictions.items(), start=1):
                error = truth - prediction[None, ...]
                atlas[family_i, object_i, model_i] = np.mean(np.abs(error), axis=0)
                gains[model] = common.relative_gain(truth, error)
            all_positive = all_positive and gains["family"] >= 0.01
            superiority += int(gains["family"] > max(gains["global"], gains["wrong_family"]))
            object_rows.append({"object": name, "groups": len(groups), "gains": gains})
        passed = all_positive and superiority >= 2
        family_rows.append({"family": family, "objects": object_rows, "semantic_control_wins": superiority, "family_gate_passed": passed})
        print(f"[C313] {family}: wins={superiority}/3 positive={all_positive} pass={passed}", flush=True)
    np.save(out / "analysis/dual_axis_full_coordinate_errors.float32.npy", atlas)
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {"status": "dual_axis_square_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C313", headline, {"six_families": len(family_rows) == 6, "three_objects": all(len(row["objects"]) == 3 for row in family_rows), "atlas_shape": list(atlas.shape) == [6, 3, 4, 36, 6, 2560], "finite": bool(np.isfinite(atlas).all())}, "C314_full_coordinate_operator_atlas")


if __name__ == "__main__":
    main()
