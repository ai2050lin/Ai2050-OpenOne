#!/usr/bin/env python3
"""Phase1596 / C106: freeze nested coordinate coalitions inside replicated causal role states."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1591_c104_frozen_candidate_validation as validation
import phase1592_c104_upstream_role_intervention as intervention

PHASE = 1596
CAMPAIGN = "C106"
FAMILIES = ("attribute_binding", "agent_patient")
NESTED_K = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C106 already exists: {OUT}")
    c104_final = core.load(C104 / "analysis/final.json")
    c104_audit = core.load(C104 / "audit/independent_final_audit.json")
    if not c104_audit["all_checks_passed"] or sorted(c104_final["c104_corrected_causal"]["fully_controlled_families"]) != ["agent_patient", "attribute_binding"]:
        raise RuntimeError("C106 parent authorization missing")
    contract = core.load(C104 / "protocol/preregistration.json")
    predictions = {row["family"]: row for row in contract["predictions"]}
    source = np.load(ROOT / contract["barcode_path"], mmap_mode="r")
    coeff = np.load(C104 / "raw/qwen3_breadth_three_effect_coefficients.float32.npy", mmap_mode="r")
    units = core.rows(C104 / "raw/qwen3_breadth_three_effect_index.jsonl")
    rankings = {}
    discovery_rows = []
    for family in FAMILIES:
        family_index = validation.FAMILIES.index(family)
        prediction = predictions[family]
        target = validation.partition_vector(coeff, units, family, "response_discovery", 0,
                                             int(prediction["state"]), int(prediction["role_index"]))
        source_values = np.asarray(source[family_index], dtype=np.float64)
        same_sign = np.sign(source_values) == np.sign(target)
        stable_floor = np.where(same_sign, np.minimum(np.abs(source_values), np.abs(target)), -np.minimum(np.abs(source_values), np.abs(target)))
        rank = np.lexsort((np.arange(2560), -np.abs(source_values * target), -stable_floor)).astype(int).tolist()
        rankings[family] = rank
        for k in NESTED_K:
            coordinates = rank[:k]
            discovery_rows.append({
                "family": family, "role": prediction["role"], "state": prediction["state"], "k": k,
                "source_discovery_cosine": cosine(source_values[coordinates], target[coordinates]),
                "same_sign_fraction": float(np.mean(np.sign(source_values[coordinates]) == np.sign(target[coordinates]))),
                "source_norm": float(np.linalg.norm(source_values[coordinates])), "target_norm": float(np.linalg.norm(target[coordinates])),
            })
    discovery_path = OUT / "analysis/discovery_nested_support_observation.jsonl"
    core.write_rows(discovery_path, discovery_rows)
    pairs = intervention.build_pairs(core.rows(C104 / "compiled/qwen3.jsonl"), list(FAMILIES), predictions)
    manifest = [{
        "pair_id": row["pair_id"], "unit_id": row["unit_id"], "family": row["family"], "partition": row["partition"],
        "code": row["code"], "role": row["role"], "span_length": row["span_length"],
        "recipient_case_id": row["recipient"]["case_id"], "donor_case_id": row["donor"]["case_id"],
        "same_truth_donor_case_id": row["same_truth_donor"]["case_id"],
    } for row in pairs]
    manifest_path = OUT / "protocol/pair_manifest.jsonl"
    core.write_rows(manifest_path, manifest)
    parent_protocol = core.load(C104 / "protocol/upstream_intervention_protocol.json")
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "nested_minimal_coordinate_coalition_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "activation coordinates inside the frozen upstream role-state candidates for attribute binding and agent-patient",
        "families": list(FAMILIES),
        "predictions": [predictions[family] for family in FAMILIES],
        "ranking_rule": "same-sign source/discovery coordinates first by min absolute magnitude; tie-break by absolute product then coordinate id",
        "rankings": rankings,
        "nested_k": list(NESTED_K),
        "selection_partition": "response_discovery only",
        "formal_partitions": ["confirmation", "lockbox"],
        "code_strata": ["standard", "reversed"],
        "modes": list(intervention.MODES),
        "coordinate_permutations": {family: parent_protocol["coordinate_permutations"][family] for family in FAMILIES},
        "pairs": len(pairs),
        "pair_manifest_sha256": core.sha(manifest_path),
        "discovery_observation_sha256": core.sha(discovery_path),
        "parent_final_sha256": core.sha(C104 / "analysis/final.json"),
        "producer_sha256": core.sha(Path(__file__)),
        "candidate_order": ["yes", "no"],
        "readout": "candidate[0]-candidate[1] = Yes-minus-No; positive gain moves false recipient toward true donor",
        "adjudication": "for each K report family x partition x code; minimal K requires positive correct median and superiority to all three controls in all four formal cells",
        "typed_missingness": "none; K=2560 is the preregistered whole-role positive control",
        "claim_boundary": "activation-coordinate coalition sufficiency only; no weight, attention, MLP, cross-model, sparse-neuron or natural-language-universal claim",
        "authorization": "execute_phase1597_c106_nested_coordinate_interventions",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    checks = {
        "parent": c104_audit["all_checks_passed"],
        "families": list(FAMILIES) == ["attribute_binding", "agent_patient"],
        "rankings": all(sorted(rankings[family]) == list(range(2560)) for family in FAMILIES),
        "nested": list(NESTED_K) == sorted(set(NESTED_K)) and NESTED_K[-1] == 2560,
        "discovery": len(discovery_rows) == len(FAMILIES) * len(NESTED_K),
        "pairs": len(pairs) == len(manifest) == 96,
        "partitions": {row["partition"] for row in manifest} == {"confirmation", "lockbox"},
        "candidate_order": protocol["candidate_order"] == ["yes", "no"],
        "authorization": protocol["authorization"] == "execute_phase1597_c106_nested_coordinate_interventions",
    }
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()),
             "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
