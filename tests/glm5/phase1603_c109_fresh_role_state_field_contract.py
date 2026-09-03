#!/usr/bin/env python3
"""Phase1603 / C109: freeze an observation-first fresh role-state field atlas."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase1600_c108_fresh_coordinate_causality"
OUT = RESULT / "phase1603_c109_fresh_role_state_field_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE = 1603
CAMPAIGN = "C109"
STATES = 37
DIM = 2560
WIDTH = 224
BATCH_SIZE = 8
ROLES = (
    "focus_pre",
    "focus_record",
    "focus_post",
    "query_focus",
    "query_anchor",
    "code_instruction",
    "boundary",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C109 already exists: {OUT}")
    closure = core.load(SOURCE / "analysis/closure.json")
    source_audit = core.load(SOURCE / "audit/independent_closure_audit.json")
    source_protocol = core.load(SOURCE / "protocol/preregistration.json")
    rows = core.rows(SOURCE / "compiled/qwen3.jsonl")
    if not source_audit["all_checks_passed"] or not closure["next_authorization"].startswith("C109 observation-first"):
        raise RuntimeError("C109 authorization missing")
    tok = graph_base.tokenizer()
    occurrences = []
    role_counts = Counter()
    disjoint = True
    for row_index, row in enumerate(rows):
        occupied = []
        for role in ROLES:
            positions = [int(value) for value in row["role_positions"][role]]
            role_counts[role] += len(positions)
            occupied.extend(positions)
            for subtoken, position in enumerate(positions):
                token_id = int(row["prompt_ids"][position])
                occurrences.append({
                    "occurrence_index": len(occurrences),
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "family": row["family"],
                    "partition": row["partition"],
                    "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"],
                    "code": row["code"],
                    "role": role,
                    "subtoken": subtoken,
                    "span_length": len(positions),
                    "token_position": position,
                    "token_id": token_id,
                    "token_text": tok.convert_ids_to_tokens([token_id])[0],
                })
        disjoint = disjoint and len(occupied) == len(set(occupied))

    rankings = source_protocol["rankings"]
    supports = {
        "attribute_binding_k256": rankings["attribute_binding"][:256],
        "agent_patient_k128": rankings["agent_patient"][:128],
        "attribute_wrong_agent_k256": rankings["agent_patient"][:256],
        "agent_wrong_attribute_k128": rankings["attribute_binding"][:128],
    }
    overlaps = {
        "attribute_k256_vs_agent_k256": len(set(supports["attribute_binding_k256"]) & set(supports["attribute_wrong_agent_k256"])),
        "agent_k128_vs_attribute_k128": len(set(supports["agent_patient_k128"]) & set(supports["agent_wrong_attribute_k128"])),
    }
    expected_data_bytes = STATES * len(occurrences) * DIM * 2
    checks = {
        "authorization": source_audit["all_checks_passed"],
        "source_identity": closure["status"] == "fresh_coordinate_causality_stage_closed",
        "rows": len(rows) == 384,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in rows),
        "occurrence_index": all(row["occurrence_index"] == i for i, row in enumerate(occurrences)),
        "physical_role_disjointness": disjoint,
        "coordinates": DIM == 2560 and STATES == 37,
        "width": max(len(row["prompt_ids"]) for row in rows) < WIDTH,
        "balanced_source": Counter((row["family"], row["partition"]) for row in rows) == {
            (family, partition): 96
            for family in source_protocol["families"]
            for partition in source_protocol["partitions"]
        },
        "support_cardinality": all(len(values) == len(set(values)) for values in supports.values()),
        "support_bounds": all(0 <= value < DIM for values in supports.values() for value in values),
        "archive_size": expected_data_bytes == 1_873_182_720,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    contract = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "observation_first_full_role_state_field_contract_frozen",
        "source": {
            "campaign": "C108",
            "cases": 384,
            "material_digest": source_protocol["material_digest"],
            "compiled_sha256": core.sha(SOURCE / "compiled/qwen3.jsonl"),
            "exposure_status": "already exposed in C108; C109 is an observation atlas, not a fresh confirmation",
        },
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "embedding-to-HiddenState activation field over every physical subtoken in seven registered functional roles",
        "roles": list(ROLES),
        "states": STATES,
        "state_semantics": "state0 embedding; state1-state36 Hidden States",
        "activation_coordinates": DIM,
        "occurrences": len(occurrences),
        "role_occurrence_counts": dict(role_counts),
        "archive": {
            "path": "raw/qwen3_role_subtoken_all_states.uint16.npy",
            "shape": [STATES, len(occurrences), DIM],
            "dtype": "uint16 exact BF16 bit patterns",
            "expected_data_bytes": expected_data_bytes,
            "fixed_width": WIDTH,
            "batch_size": BATCH_SIZE,
        },
        "supports": supports,
        "support_overlap_counts": overlaps,
        "frozen_observations": [
            "exact BF16 archive; no PCA, projection, coordinate pooling, attention, MLP, or weight inspection",
            "derive a role-span mean only after preserving every physical subtoken",
            "compute exact balanced truth Walsh coefficient within each unit, role, and state",
            "report every state trajectory, all 2560 coordinates, frozen-support energy, same-K wrong-family energy, overlap, and per-pair patch energy",
            "separate raw Yes-minus-No response from code-aligned task response",
        ],
        "formulae": {
            "unit_truth_field": "B_u(r,s)=1/16 sum_{t,p,d,c} t H_u(t,p,d,c;r,s)",
            "partition_field": "B_f,q(r,s)=mean_{u in family f, partition q} B_u(r,s)",
            "support_energy": "rho_S=||B_S||_2^2/||B||_2^2",
            "pair_write_energy": "E_S(D,R)=sum_{j in S}(H_D[j]-H_R[j])^2",
        },
        "completion_rule": "numeric archive valid and every frozen descriptive table emitted; there is no post-hoc mechanism pass gate",
        "typed_missingness": {
            "fresh_confirmation": "missing: materials were already exposed by C108",
            "human_naturalness": "missing: no independent blind human rating",
            "causal_extension": "missing: C109 is observation-only",
            "cross_model": "missing: Qwen3 only",
        },
        "claim_boundary": "activation-field atlas for one controlled-English Qwen task; coordinates are activations, not parameters, neurons, semantic atoms, or a universal code",
        "manifest_sha256": core.sha(OUT / "protocol/role_occurrence_manifest.jsonl"),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1604_c109_qwen_role_state_capture",
    }
    core.save(OUT / "protocol/preregistration.json", contract)
    audit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "occurrences": len(occurrences),
        "role_occurrence_counts": dict(role_counts),
        "expected_data_bytes": expected_data_bytes,
        "support_overlap_counts": overlaps,
        "authorization": contract["authorization"],
    }
    core.save(OUT / "audit/pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
