#!/usr/bin/env python3
"""Phase1615 / C112: freeze exact-energy value permutations and a role transport lattice."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
SOURCE = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
C111 = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    closure = core.load(C111 / "analysis/closure.json")
    if not closure["next_authorization"].startswith("C112 frozen batch intervention"):
        raise RuntimeError("C112 authorization missing")
    source_protocol = core.load(SOURCE / "protocol/preregistration.json")
    rng = np.random.default_rng(1615)
    movement_permutations = {}
    for family, k in (("attribute_binding", 256), ("agent_patient", 128)):
        movement_permutations[family] = [rng.permutation(k).astype(int).tolist() for _ in range(8)]
    roles = source_protocol["roles"]
    role_coalitions = {
        "query_plus_record": ["query_anchor", "focus_record"],
        "query_plus_query_focus": ["query_anchor", "query_focus"],
        "record_to_query_path": ["focus_record", "focus_post", "query_focus", "query_anchor"],
        "all_registered_roles": roles,
    }
    sources = {
        "c110_protocol": SOURCE / "protocol/preregistration.json",
        "c110_compiled": SOURCE / "compiled/qwen3.jsonl",
        "c110_capture": SOURCE / "analysis/capture_summary.json",
        "c110_field": SOURCE / "analysis/field_prediction_adjudication.json",
        "c111_closure": C111 / "analysis/closure.json",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "phase": 1615,
        "campaign": "C112",
        "created_at_utc": now(),
        "status": "value_identity_role_lattice_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "test physical coordinate assignment of the frozen truth movement and relation-conditioned single/multi-role output leverage",
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_hashes": {name: core.sha(path) for name, path in sources.items()},
        "families": source_protocol["families"],
        "partitions": source_protocol["partitions"],
        "pairs": 192,
        "state": 19,
        "coordinates": 2560,
        "supports": source_protocol["supports"],
        "movement_permutations": movement_permutations,
        "single_roles": roles,
        "role_coalitions": role_coalitions,
        "modes": ["frozen_support"] + [f"movement_permutation_{index}" for index in range(8)] + [f"single_{role}" for role in roles] + [f"coalition_{name}" for name in role_coalitions],
        "numeric": {"movement_permutation_actual_l2_relative_tolerance": 0.02, "fixed_width": 224, "batch_size": 8},
        "frozen_predictions": {
            "attribute_exact_value_assignment_candidate": "frozen-support median raw truth gain exceeds the median of eight movement-permutation medians in each of four cells",
            "agent_focus_record_candidate": "single focus_record median raw truth gain is positive in each of four cells",
            "agent_record_path_candidate": "record_to_query_path median raw truth gain exceeds single query_anchor in each of four cells",
        },
        "adjudication": "report all cells and routes; a failed candidate retires only that candidate and does not stop the batch",
        "claim_boundary": "activation transport only; no attention, MLP, weight, minimality, necessity, or universal semantic-neuron claim",
        "authorization": "run_phase1616_c112_cuda_batch_interventions",
    }
    protocol = OUT / "protocol/preregistration.json"
    core.save(protocol, contract)
    checks = {
        "sources": all(Path(contract["source_paths"][name]).exists() and core.sha(Path(contract["source_paths"][name])) == digest for name, digest in contract["source_hashes"].items()),
        "pairs": contract["pairs"] == 192,
        "permutations": all(len(values) == 8 and all(sorted(permutation) == list(range(256 if family == "attribute_binding" else 128)) for permutation in values) for family, values in movement_permutations.items()),
        "roles": len(contract["single_roles"]) == 7 and len(contract["role_coalitions"]) == 4,
        "modes": len(contract["modes"]) == 20 and len(set(contract["modes"])) == 20,
        "no_hard_stop": "does not stop the batch" in contract["adjudication"],
        "boundary": "no attention" in contract["claim_boundary"],
        "authorization": contract["authorization"] == "run_phase1616_c112_cuda_batch_interventions",
    }
    report = {"phase": 1615, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "protocol_sha256": core.sha(protocol)}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
