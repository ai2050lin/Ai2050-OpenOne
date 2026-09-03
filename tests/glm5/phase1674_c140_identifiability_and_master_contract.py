#!/usr/bin/env python3
"""C140: repair identifiability/accounting and freeze the C141-C148 campaign."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1674_c140_identifiability_and_master_contract"
C135 = RESULT / "phase1669_c135_all_token_coordinate_transmission"
C136 = RESULT / "phase1670_c136_chinese_pattern_composition_field"
C138 = RESULT / "phase1672_c138_prospective_cross_model_topology"
C139 = RESULT / "phase1673_c139_campaign_synthesis_and_heatmap"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1674, "C140"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    raw_path = C135 / "raw/qwen3_all_token_all_checkpoint.bf16.npy"
    raw = np.load(raw_path, mmap_mode="r")
    shape = list(raw.shape)
    element_count = int(raw.size)
    expected_count = 38 * 1164 * 2560
    arithmetic = {
        "shape": shape,
        "dtype": str(raw.dtype),
        "element_count": element_count,
        "expected_element_count": expected_count,
        "array_nbytes": int(raw.nbytes),
        "file_bytes": raw_path.stat().st_size,
        "sha256": core.sha(raw_path),
    }
    ledger = {
        "A": {
            "behavior": "measured-fail",
            "discovery_prediction": "not-tested",
            "new_trajectory_prediction": "not-tested",
            "wrong_route_control": "not-tested",
            "causal": "not-applicable",
        },
        "B": {
            "behavior": "measured-pass",
            "discovery_prediction": "measured-pass",
            "new_trajectory_prediction": "measured-fail",
            "wrong_route_control": "measured-pass",
            "causal": "not-applicable",
        },
        "C": {
            "behavior": "measured-pass",
            "discovery_prediction": "measured-pass",
            "new_trajectory_prediction": "not-tested",
            "wrong_route_control": "not-tested",
            "causal": "not-applicable",
        },
        "D": {
            "behavior": "measured-fail",
            "discovery_prediction": "not-tested",
            "new_trajectory_prediction": "not-tested",
            "wrong_route_control": "not-tested",
            "causal": "not-applicable",
        },
        "E": {
            "behavior": "measured-fail-cross-model",
            "qwen_internal_observation": "measured-pass",
            "glm4_internal_observation": "not-tested",
            "deepseek7b_internal_observation": "not-tested",
            "cross_model_topology": "not-tested",
            "causal": "not-applicable",
        },
    }
    scaling = {
        "definition": "M_S = 2^(-k) sum_epsilon product(epsilon_j) Z(epsilon)",
        "one_factor": "half difference, (Z_plus-Z_minus)/2",
        "two_factor": "quarter Walsh interaction, (Z_pp-Z_pm-Z_mp+Z_mm)/4",
        "three_factor": "one-eighth third-order Mobius/Walsh coefficient",
        "warning": "raw full differences and normalized coefficients may share direction but not norm or energy",
    }
    adequacy = {
        "C135": {
            "independent_units_per_partition": 3,
            "units_per_length_stratum": "1-2",
            "judgment": "feasibility-only; cannot identify a dense 2560x2560 law",
        },
        "C136": {
            "independent_units_per_partition": 8,
            "paired_truth_contrasts_per_partition": 128,
            "judgment": "adequate for response replication, limited for high-capacity transition identification",
        },
        "C138": {
            "independent_units_per_partition": 16,
            "paired_truth_contrasts_per_partition": 64,
            "judgment": "adequate for within-Qwen response topology, not cross-model identification",
        },
        "C141_target": {
            "arms": 5,
            "units_per_arm": 8,
            "cases_per_unit": 32,
            "total_cases": 1280,
            "semantic_factor_cells_per_unit": 8,
            "surface_cells": 2,
            "output_code_cells": 2,
            "judgment": "large observation panel; model capacity must still be bounded by held-out prediction",
        },
    }
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "C141_C148_master_contract_frozen",
        "research_object": "typed, context-conditioned, cross-token/cross-coordinate HiddenState response and transition laws",
        "evidence_books": {
            "structure_observation": "retained after behavior failure, explicitly split into correct/error trajectories",
            "functional_use": "requires behavior-qualified cells and prospective replication",
            "causal": "requires held-out trajectory prediction plus matched wrong-route controls before perturbation",
        },
        "scaling": scaling,
        "status_vocabulary": ["measured-pass", "measured-fail", "not-tested", "not-applicable"],
        "stages": {
            "C141": "five-arm full-token/full-coordinate Qwen3 atlas: event, type graph, discourse, translation, comparison",
            "C142": "coordinate-level Mobius effects and output-code-separated pattern fields",
            "C143": "held-out residual-increment transition competition: zero, shrink, diagonal, role-mixing, linear kernel, quadratic kernel",
            "C144": "language/activation dual graph and prospective low-order composition reconstruction",
            "C145": "correct-versus-error knowledge-depth trajectory atlas",
            "C146": "sequential Qwen3/GLM4/DeepSeek7B behavior-interface sweep",
            "C147": "relative-depth/role-topology comparison for every behavior-qualified model",
            "C148": "local symmetric HiddenState perturbation only if C143 and matched controls authorize it; otherwise audited not-tested closure",
        },
        "C141_design": {
            "arms": ["event_composition", "type_graph", "discourse", "translation", "comparison"],
            "units_per_arm": 8,
            "partitions": {"discovery": 4, "confirmation": 4},
            "factors": 3,
            "surfaces": 2,
            "output_codebooks": 2,
            "cases_per_arm": 256,
            "total_cases": 1280,
            "checkpoints": "embedding + every post-block pre-final-norm state + final norm",
            "coordinates": "all physical activation coordinates",
            "tokens": "all actual tokens plus a six-role aligned field",
            "behavior_policy": "never blocks structural capture; correct/error status remains typed",
        },
        "C143_gate": {
            "confirmation_relative_error_ratio_vs_zero_max": 0.90,
            "confirmation_cosine_min": 0.60,
            "wrong_role_error_margin_min": 0.05,
            "wrong_coordinate_error_margin_min": 0.05,
            "rollout_relative_error_ratio_vs_zero_max": 0.95,
        },
        "C146_gate": {
            "common_interface_global_min": 0.80,
            "truth_min": 0.75,
            "models_required": 2,
        },
        "causal_authorization": "C143 held-out gate AND explicit direction candidate AND matched wrong-role/wrong-coordinate/wrong-checkpoint controls",
        "model_order": ["qwen3", "glm4", "deepseek7b"],
        "model_runtime": "local BF16 CUDA, nonquantized, strictly sequential",
        "forbidden": ["attention inspection", "MLP inspection", "weight inspection", "post-unblind threshold changes", "calling activation coordinates weight parameters"],
        "claim_boundary": "observation-first system identification; no semantic neuron, unique circuit, universal operator, topology phase transition, or new mathematics is presupposed",
        "source_paths": {
            "C135_raw": str(raw_path),
            "C136_final": str(C136 / "audit/independent_closure_audit.json"),
            "C138_final": str(C138 / "audit/independent_closure_audit.json"),
            "C139_final": str(C139 / "audit/independent_closure_audit.json"),
        },
        "source_hashes": {
            "C135_raw": core.sha(raw_path),
            "C136_final": core.sha(C136 / "audit/independent_closure_audit.json"),
            "C138_final": core.sha(C138 / "audit/independent_closure_audit.json"),
            "C139_final": core.sha(C139 / "audit/independent_closure_audit.json"),
        },
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "start_C141_multifamily_atlas",
    }
    checks = {
        "shape": shape == [38, 1164, 2560],
        "element_count": element_count == expected_count == 113233920,
        "dtype": str(raw.dtype) == "uint16",
        "hash_matches_C135": arithmetic["sha256"] == core.load(C135 / "analysis/capture.json")["sha256"],
        "source_audits": all(core.load(path)["all_checks_passed"] for path in (
            C136 / "audit/independent_closure_audit.json",
            C138 / "audit/independent_closure_audit.json",
            C139 / "audit/independent_closure_audit.json",
        )),
        "status_taxonomy": set(protocol["status_vocabulary"]) == {"measured-pass", "measured-fail", "not-tested", "not-applicable"},
        "five_arms": len(protocol["C141_design"]["arms"]) == 5,
        "continuation_policy": "never blocks" in protocol["C141_design"]["behavior_policy"],
        "causal_last": "C143" in protocol["causal_authorization"],
    }
    OUT.mkdir(parents=True)
    core.save(OUT / "analysis/c135_archive_audit.json", arithmetic)
    core.save(OUT / "analysis/c133_c139_typed_ledger.json", ledger)
    core.save(OUT / "analysis/sample_adequacy.json", adequacy)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": protocol["authorization"],
    })
    print(json.dumps({"arithmetic": arithmetic, "ledger": ledger, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
