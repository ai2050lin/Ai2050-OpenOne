#!/usr/bin/env python3
"""C317: all-coordinate multi-role conditional response grammar."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


SOURCE_SETS = (("relation",), ("relation", "query"), ("primary", "relation", "query"))


def event(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0)).astype(np.int8)


def code(values: np.ndarray, roles: tuple[str, ...], threshold: float) -> np.ndarray:
    result = np.zeros((values.shape[0], values.shape[-1]), np.int16)
    for power, role in enumerate(roles):
        result += (event(values[:, common.ROLES.index(role)], threshold) + 1).astype(np.int16) * (3 ** power)
    return result


def predict_conditional(train_code: np.ndarray, train_target: np.ndarray, test_code: np.ndarray, states: int, support_min: int = 4) -> np.ndarray:
    prediction = np.zeros(test_code.shape, np.float32)
    for state in range(states):
        mask = train_code == state
        count = mask.sum(axis=0)
        mean = np.divide((train_target * mask).sum(axis=0), np.maximum(count, 1), dtype=np.float32)
        eligible = count >= support_min
        prediction += (test_code == state) * eligible[None, :] * mean[None, :]
    return prediction


def gain(truth: np.ndarray, prediction: np.ndarray) -> float:
    return common.relative_gain(truth, truth - prediction)


def main() -> None:
    parent = common.core.load(common.OUTS["C316"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "source_sets_nested": True, "all_coordinates": True, "all_q_destination_roles": True}
    protocol = {
        "status": "multisource_response_grammar_frozen",
        "source_sets": [list(value) for value in SOURCE_SETS],
        "rule": "For every physical coordinate, fit the mean next-checkpoint destination interaction residual conditional on the ternary sign word of one, two, or three source roles.",
        "support_min": 4,
        "selection": "q, destination role, and source-set size selected by leave-one-material-out gain on third/fourth/fifth only; sixth is lockbox",
        "gate": "selected sixth-material gain>=0.01 and positive in at least four families",
        "claim_boundary": "This is a same-coordinate conditional prediction grammar over role states. It is not a unique causal edge map and does not exclude cross-coordinate or nonlinear dependence.",
    }
    out = common.prepare("C317", protocol, checks)
    thresholds = common.old.thresholds()
    test_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    test_index = common.core.rows(common.SIXTH_INDEX)
    score_atlas = np.full((6, 3, 36, 6, 2), np.nan, np.float32)
    selected_rows = []
    family_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        train_parts = []
        for state_path, index_path, material in common.TRAIN_ROLE_SOURCES:
            arrays, _groups = common.factorial_arrays(np.load(state_path, mmap_mode="r"), common.core.rows(index_path), family)
            train_parts.append((arrays["interaction"], material))
        test_arrays, test_groups = common.factorial_arrays(test_states, test_index, family)
        test = test_arrays["interaction"]
        for source_i, roles in enumerate(SOURCE_SETS):
            states_count = 3 ** len(roles)
            for q in range(36):
                for destination in range(6):
                    cv_values = []
                    for holdout_i in range(3):
                        train = np.concatenate([rows for i, (rows, _name) in enumerate(train_parts) if i != holdout_i], axis=0)
                        holdout = train_parts[holdout_i][0]
                        train_code = code(train[:, q], roles, float(thresholds[q]))
                        holdout_code = code(holdout[:, q], roles, float(thresholds[q]))
                        prediction = predict_conditional(train_code, train[:, q + 1, destination], holdout_code, states_count)
                        cv_values.append(gain(holdout[:, q + 1, destination], prediction))
                    train = np.concatenate([rows for rows, _name in train_parts], axis=0)
                    prediction = predict_conditional(code(train[:, q], roles, float(thresholds[q])), train[:, q + 1, destination], code(test[:, q], roles, float(thresholds[q])), states_count)
                    score_atlas[family_i, source_i, q, destination] = [float(np.mean(cv_values)), gain(test[:, q + 1, destination], prediction)]
        train_scores = score_atlas[family_i, :, :, :, 0]
        best = np.unravel_index(int(np.nanargmax(train_scores)), train_scores.shape)
        source_i, q, destination = [int(value) for value in best]
        lockbox_gain = float(score_atlas[family_i, source_i, q, destination, 1])
        selected = {"family": family, "source_roles": list(SOURCE_SETS[source_i]), "source_count": source_i + 1, "q": q, "destination_role": common.ROLES[destination], "destination_index": destination, "training_cv_gain": float(train_scores[best]), "sixth_lockbox_gain": lockbox_gain, "sixth_groups": len(test_groups), "family_gate_passed": lockbox_gain >= 0.01}
        selected_rows.append(selected)
        family_rows.append(selected)
        print(f"[C317] {family}: roles={selected['source_roles']} q={q} dest={selected['destination_role']} cv={selected['training_cv_gain']:+.4f} test={lockbox_gain:+.4f}", flush=True)
    np.save(out / "analysis/all_coordinate_response_grammar_scores.float32.npy", score_atlas)
    common.core.save(out / "protocol/selected_strata.json", {"source_sets": [list(value) for value in SOURCE_SETS], "selected": selected_rows})
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {"status": "multisource_response_grammar_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C317", headline, {"six_families": len(family_rows) == 6, "atlas_shape": list(score_atlas.shape) == [6, 3, 36, 6, 2], "finite": bool(np.isfinite(score_atlas).all())}, "C318_distributed_intervention_contract")


if __name__ == "__main__":
    main()
