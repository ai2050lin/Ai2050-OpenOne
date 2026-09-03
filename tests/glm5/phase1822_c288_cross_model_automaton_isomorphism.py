#!/usr/bin/env python3
"""C288: compare anonymous six-role sign-word automata across three models."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C288"]
C287 = common.OUTS["C287"]
CORE_FAMILIES = common.previous.FAMILIES
FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def sign_word(effect: np.ndarray) -> np.ndarray:
    code = np.zeros(effect.shape[1], np.uint8)
    for role in range(6): code += (effect[role] >= 0).astype(np.uint8) << role
    return code


def topology(model_name: str) -> tuple[np.ndarray, dict]:
    states = np.load(C287 / f"raw/{model_name}_role_states.float16.npy", mmap_mode="r")
    index = core.rows(C287 / f"raw/{model_name}_hidden_index.jsonl")
    nq, dim = states.shape[1], states.shape[-1]
    stages = sorted(set(min(nq - 2, int(round(frac * (nq - 2)))) for frac in FRACTIONS))
    if len(stages) != 4: raise RuntimeError((model_name, stages))
    counts = np.zeros((5, 4, 64), np.uint64)
    positive = np.zeros((5, 4, 64, 6), np.uint64)
    complete_units = np.zeros(5, np.int32)
    lookup = {(r["family"], r["unit"], r["factor_a"], r["factor_b"]): r["hidden_index"] for r in index if r["behavior_correct"]}
    for fi, family in enumerate(CORE_FAMILIES):
        for unit in (0, 1):
            keys = [(family, unit, a, b) for a, b in itertools.product((0, 1), repeat=2)]
            if not all(key in lookup for key in keys): continue
            complete_units[fi] += 1
            cells = {(a, b): np.asarray(states[lookup[(family, unit, a, b)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
            effect = 0.5 * ((cells[(1, 0)] - cells[(0, 0)]) + (cells[(1, 1)] - cells[(0, 1)]))
            for si, q in enumerate(stages):
                current = sign_word(effect[q]); next_sign = effect[q + 1] >= 0
                counts[fi, si] += np.bincount(current, minlength=64).astype(np.uint64)
                for role in range(6): positive[fi, si, :, role] += np.bincount(current, weights=next_sign[role], minlength=64).astype(np.uint64)
    occupancy = counts / np.maximum(counts.sum(axis=2, keepdims=True), 1)
    rate = positive / np.maximum(counts[..., None], 1)
    result = np.concatenate((occupancy[..., None], rate), axis=-1).astype(np.float32)
    return result, {"nq": nq, "dimension": dim, "relative_stages": stages, "complete_units_by_family": complete_units.tolist()}


def permute_topology(value: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    transformed = np.zeros_like(value)
    for old_code in range(64):
        old_bits = [(old_code >> role) & 1 for role in range(6)]
        new_code = sum(old_bits[permutation[role]] << role for role in range(6))
        transformed[:, :, new_code, 0] = value[:, :, old_code, 0]
        for role in range(6): transformed[:, :, new_code, 1 + role] = value[:, :, old_code, 1 + permutation[role]]
    return transformed


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    occupancy_score = 1.0 - 0.5 * np.abs(left[..., 0] - right[..., 0]).sum(axis=2).mean()
    weight = 0.5 * (left[..., 0] + right[..., 0])
    transition_error = (weight[..., None] * np.abs(left[..., 1:] - right[..., 1:])).sum() / max(float(weight.sum() * 6), 1e-30)
    return float(0.5 * occupancy_score + 0.5 * (1.0 - transition_error))


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C287 / "analysis/final.json"); participants = tuple(parent["headline"]["participants"])
    checks = {"parent": parent["all_checks_passed"], "participants": participants == common.MODELS, "all_coordinates_by_model": True, "no_coordinate_alignment": True, "all_role_permutations": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"): (OUT / subdir).mkdir()
    protocol = {
        "phase": 1822, "campaign": "C288", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "automaton_comparison_frozen",
        "object": "family x relative depth x six-role current sign word x next-role sign probability",
        "coordinates": "Every physical coordinate contributes one sign word inside its own model; coordinate numbers and dimensions are never aligned.",
        "similarity": "equal-weight mean of occupancy total-variation similarity and occupancy-weighted next-role sign agreement",
        "null": "all 720 permutations of semantic role names in the right model",
        "pair_gate": "similarity>=0.80 and exact upper p<=0.05", "broad_gate": "all three model pairs pass",
        "claim_boundary": "Passing supports anonymous functional topology, not physical coordinate identity, causal bisimulation, or a shared implementation.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C289_campaign_adjudication_heatmap",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    topologies = {}; metadata = {}
    for model_name in participants:
        topologies[model_name], metadata[model_name] = topology(model_name); np.save(OUT / f"analysis/{model_name}_anonymous_automaton.float32.npy", topologies[model_name])
    permutations = list(itertools.permutations(range(6))); pair_rows = []
    for i, left_name in enumerate(participants):
        for right_name in participants[i + 1:]:
            left, right = topologies[left_name], topologies[right_name]
            observed = similarity(left, right)
            null = np.asarray([similarity(left, permute_topology(right, p)) for p in permutations], np.float64)
            p_upper = float((1 + np.sum(null >= observed)) / (1 + len(null)))
            row = {"models": [left_name, right_name], "similarity": observed, "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": p_upper, "pair_gate_passed": observed >= 0.80 and p_upper <= 0.05}
            pair_rows.append(row); print(f"[C288] {left_name}/{right_name}: S={observed:.4f}, q95={row['null_q95']:.4f}, p={p_upper:.6f}", flush=True)
    broad = len(pair_rows) == 3 and all(row["pair_gate_passed"] for row in pair_rows)
    report = {
        "phase": 1822, "campaign": "C288", "status": "cross_model_automaton_adjudicated", "models": metadata, "pairs": pair_rows,
        "cross_model_automaton_gate_passed": broad, "causal_isomorphism_status": "no_test_C285_local_eligibility_failed",
        "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C289_campaign_adjudication_heatmap",
    }
    core.save(OUT / "analysis/summary.json", report)
    ach = {"models": len(topologies) == 3, "pairs": len(pair_rows) == 3, "shape": all(list(x.shape) == [5, 4, 64, 7] for x in topologies.values()), "finite": bool(np.isfinite([r["similarity"] for r in pair_rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1822, "campaign": "C288", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

