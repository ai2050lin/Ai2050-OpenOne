#!/usr/bin/env python3
"""C330: compare model-native response topologies without aligning coordinates."""
from __future__ import annotations

import itertools

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64).ravel()
    b = right.astype(np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> None:
    model_campaigns = {"qwen3": "C327", "glm4": "C328", "deepseek7b": "C329"}
    finals = {model: common.core.load(common.OUTS[campaign] / "analysis/final.json") for model, campaign in model_campaigns.items()}
    checks = {"all_models_closed": all(value["all_checks_passed"] for value in finals.values()), "native_axes_unaligned": True, "three_models": len(finals) == 3}
    protocol = {
        "status": "cross_model_response_comparison_frozen",
        "participants": "models passing their frozen confirmation behavior gate",
        "topology": "family x five relative checkpoints x six role energies from each model's family interaction mean; normalize within family/checkpoint",
        "null": "all 720 role permutations for each model pair",
        "causal_signature": "per-family correct residual deletion movement and selectivity over wrong-family/coordinate/role controls",
        "convergence_gate": "at least two participants; every pair centered topology cosine>=0.30 with exact upper p<=0.05; each participant has positive composition gain and positive causal selectivity in at least four families",
        "claim_boundary": "Even a pass is convergent anonymous response topology plus coarse causal response. Functional bisimulation would additionally require mutually predictive typed state transitions and is not established here.",
    }
    out = common.prepare("C330", protocol, checks)
    participants = [model for model, final in finals.items() if final["headline"]["behavior_eligible"]]
    topologies = {}
    model_rows = []
    for model in participants:
        campaign = model_campaigns[model]
        interactions = np.load(common.OUTS[campaign] / "analysis/family_interaction_means.float32.npy", mmap_mode="r")
        nq = interactions.shape[1]
        checkpoints = sorted(set(int(round(fraction * (nq - 1))) for fraction in (0, 0.25, 0.5, 0.75, 1.0)))
        energy = np.mean(np.abs(np.asarray(interactions[:, checkpoints], np.float32)), axis=-1)
        energy /= np.maximum(energy.sum(axis=-1, keepdims=True), 1e-12)
        topologies[model] = energy
        composition = finals[model]["headline"]["composition_prediction"]
        causal = finals[model]["headline"]["causal_response"]
        positive_composition = sum(row["relative_mae_gain"] > 0 for row in composition)
        positive_causal = sum((row["mean_correct_minus_best_wrong"] or -np.inf) > 0 for row in causal)
        model_rows.append({"model": model, "relative_checkpoints": checkpoints, "positive_composition_families": positive_composition, "positive_causal_selectivity_families": positive_causal, "model_gate_passed": positive_composition >= 4 and positive_causal >= 4})
    permutations = list(itertools.permutations(range(6)))
    pair_rows = []
    for left_i, left_name in enumerate(participants):
        for right_name in participants[left_i + 1:]:
            left = topologies[left_name] - topologies[left_name].mean(axis=-1, keepdims=True)
            right = topologies[right_name] - topologies[right_name].mean(axis=-1, keepdims=True)
            observed = cosine(left, right)
            null = np.asarray([cosine(left, right[:, :, permutation]) for permutation in permutations], np.float64)
            pair_rows.append({"models": [left_name, right_name], "centered_cosine": observed, "null_median": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null))), "role_permutations": len(null)})
    convergence = len(participants) >= 2 and bool(pair_rows) and all(row["centered_cosine"] >= 0.30 and row["exact_upper_p"] <= 0.05 for row in pair_rows) and all(row["model_gate_passed"] for row in model_rows)
    np.save(out / "analysis/model_native_response_topologies.float32.npy", np.asarray([topologies[model] for model in participants], np.float32))
    common.core.write_rows(out / "analysis/model_results.jsonl", model_rows)
    common.core.write_rows(out / "analysis/pair_tests.jsonl", pair_rows)
    headline = {"status": "cross_model_response_adjudicated", "participants": participants, "models": model_rows, "pair_tests": pair_rows, "convergent_response_gate_passed": convergence, "functional_bisimulation_established": False, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C330", headline, {"participant_subset": set(participants) <= set(common.MODELS), "pair_count": len(pair_rows) == len(participants) * (len(participants) - 1) // 2, "topology_count": len(topologies) == len(participants), "finite": common.finite_dict(headline)}, "C331_C335_knowledge_graph_branch")


if __name__ == "__main__":
    main()
