#!/usr/bin/env python3
"""C202: adjudicate C194-C201 and publish a coordinate-complete Qwen3 atlas."""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1736_c202_campaign_theory_adjudication"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c202_signed_operator_campaign.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN, DIM = 1736, "C202", 2560
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
UPSTREAM = {
    "C194": RESULT / "phase1728_c194_signed_operator_campaign_contract",
    "C195": RESULT / "phase1729_c195_signed_role_checkpoint_trajectory",
    "C196": RESULT / "phase1730_c196_multidose_orthogonal_identification",
    "C197": RESULT / "phase1731_c197_structure_model_tournament",
    "C198": RESULT / "phase1732_c198_broad_natural_program_trajectory",
    "C199": RESULT / "phase1733_c199_unseen_composition_prediction",
    "C200": RESULT / "phase1734_c200_natural_deletion_rescue_adjudication",
    "C201": RESULT / "phase1735_c201_cross_model_functional_topology",
}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / den) if den > 1e-30 else 0.0


def topology_diagnostic(c201: dict) -> dict:
    profiles = {
        name: np.asarray(value["transition_role_energy"], dtype=np.float64)
        for name, value in c201["topologies"].items()
    }
    pairs = []
    for left_name, right_name in itertools.combinations(profiles, 2):
        left, right = profiles[left_name], profiles[right_name]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        observed = cosine(left_centered.ravel(), right_centered.ravel())
        null = []
        for permutation in itertools.permutations(range(len(ROLES))):
            null.append(cosine(left_centered.ravel(), right_centered[:, permutation].ravel()))
        null_array = np.asarray(null, dtype=np.float64)
        p_upper = float((1 + np.count_nonzero(null_array >= observed - 1e-12)) / (1 + len(null_array)))
        pairs.append({
            "models": [left_name, right_name],
            "raw_cosine": cosine(left.ravel(), right.ravel()),
            "centered_cosine": observed,
            "permutation_null_median": float(np.median(null_array)),
            "permutation_null_q95": float(np.quantile(null_array, 0.95)),
            "exact_upper_p": p_upper,
            "role_permutations": int(len(null_array)),
        })
    passed = all(row["centered_cosine"] >= 0.30 and row["exact_upper_p"] <= 0.05 for row in pairs)
    return {
        "status": "exploratory_posthoc_diagnostic",
        "reason": "C201 registered raw cosine but omitted a role-permutation null; this diagnostic cannot retroactively repair that gate.",
        "pairs": pairs,
        "exploratory_nontrivial_alignment": passed,
    }


def add_row(rows: list[dict], *, kind: str, label: str, values: np.ndarray, **meta) -> None:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((label, vector.shape, bool(np.isfinite(vector).all())))
    rows.append({"kind": kind, "label": label, **meta, "values": vector.tolist()})


def build_asset(adjudication: dict) -> dict:
    rows: list[dict] = []
    c195_asset = core.load(ROOT / "frontend/public/vis_data/research_kernel/c195_signed_operator_trajectory.json")
    for row in c195_asset["rows"]:
        keep_baseline = row["kind"] == "baseline_state"
        keep_response = (
            row["kind"] == "signed_response"
            and row.get("program") == "direct_target"
            and row.get("phrase_variant") == 0
            and row.get("role") in ("relation", "boundary")
        )
        if keep_baseline or keep_response:
            add_row(rows, kind=f"C195_{row['kind']}", label=f"C195/{row['label']}", values=row["values"], **{k: v for k, v in row.items() if k not in {"kind", "label", "values"}})

    c198 = UPSTREAM["C198"]
    index = core.rows(c198 / "raw/hidden_index.jsonl")
    baseline = np.load(c198 / "raw/natural_baseline_states.float16.npy", mmap_mode="r")
    response = np.load(c198 / "raw/natural_signed_trajectory.float16.npy", mmap_mode="r")
    programs = sorted({row["program"] for row in index})
    state_names = ("embedding", "q23", "q24", "q25")
    for program in programs:
        selected = [row["anchor_index"] for row in index if row["program"] == program]
        for state_i, state in enumerate(state_names):
            for role_i, role in ((2, "relation"), (5, "boundary")):
                add_row(
                    rows,
                    kind="C198_natural_baseline",
                    label=f"C198/{program}/{state}/{role}",
                    values=np.asarray(baseline[selected, state_i, role_i], dtype=np.float32).mean(axis=0),
                    program=program,
                    state=state,
                    role=role,
                )
        for response_i, state in enumerate(("q24_response", "q25_response")):
            for role_i, role in ((2, "relation"), (5, "boundary")):
                add_row(
                    rows,
                    kind="C198_natural_signed_response",
                    label=f"C198/{program}/{state}/{role}",
                    values=np.asarray(response[selected, :, response_i, role_i], dtype=np.float32).mean(axis=(0, 1)),
                    program=program,
                    state=state,
                    role=role,
                )

    score = np.max(np.abs(np.asarray([row["values"] for row in rows], dtype=np.float32)), axis=0)
    defaults = np.argsort(-score, kind="stable")[:16].astype(int).tolist()
    payload = {
        "schema": "c202_signed_operator_campaign.v1",
        "result_type": "signed_operator_campaign_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "qwen3-4b",
        "title": "C194-C202 Signed HiddenState Operator Campaign",
        "dimensions": list(range(DIM)),
        "default_coordinates": defaults,
        "rows": rows,
        "summary": {
            "c195_sign_persistence": adjudication["evidence_ledger"]["C195"]["weighted_sign_persistence_median"],
            "c197_primary_gate": adjudication["evidence_ledger"]["C197"]["primary_gate_passed"],
            "c199_primary_gate": adjudication["evidence_ledger"]["C199"]["primary_gate_passed"],
            "c200_causal_tested": adjudication["evidence_ledger"]["C200"]["natural_deletion_rescue_tested"],
            "c201_registered_gate_reclassified": True,
            "c201_exploratory_alignment": adjudication["c201_topology_reaudit"]["exploratory_nontrivial_alignment"],
        },
        "coordinate_semantics": "All 2560 columns are Qwen3-4B embedding or HiddenState activation coordinates. They are not weight parameters, neurons, attention heads, or MLP units.",
        "claim_boundary": "C195 establishes a signed local trajectory observation. C196-C199 did not validate a reusable predictor, C200 therefore did not run causal deletion/rescue, and C201 raw topology cosine is reclassified as descriptive because its registered analysis omitted a role-permutation null.",
    }
    core.save(ASSET, payload)
    return payload


def run() -> None:
    if OUT.exists():
        raise RuntimeError(f"result already exists: {OUT}")
    OUT.mkdir(parents=True)
    audits = {name: core.load(path / "audit/independent_final_audit.json") for name, path in UPSTREAM.items()}
    upstream_checks = {name: bool(value["all_checks_passed"]) for name, value in audits.items()}
    if not all(upstream_checks.values()):
        raise RuntimeError(upstream_checks)

    c195 = core.load(UPSTREAM["C195"] / "analysis/final.json")["headline"]
    c196 = core.load(UPSTREAM["C196"] / "analysis/final.json")["headline"]
    c197 = core.load(UPSTREAM["C197"] / "analysis/final.json")["headline"]
    c198 = core.load(UPSTREAM["C198"] / "analysis/final.json")["headline"]
    c199 = core.load(UPSTREAM["C199"] / "analysis/final.json")["headline"]
    c200 = core.load(UPSTREAM["C200"] / "analysis/final.json")["headline"]
    c201 = core.load(UPSTREAM["C201"] / "analysis/cross_model_topology.json")
    topology = topology_diagnostic(c201)

    evidence = {
        "C195": {"status": "observed", "weighted_sign_persistence_median": c195["weighted_sign_persistence_median"], "q25_over_q24_gain_median": c195["q25_over_q24_gain_median"]},
        "C196": {"status": "registered_gate_failed", "linear_superposition_gate_passed": c196["linear_superposition_gate_passed"], "dose_rows": c196["dose_rows"]},
        "C197": {"status": "registered_gate_failed", "winner": c197["winner"], "primary_gate_passed": c197["primary_gate_passed"], "confirmation_improvement": c197["winner_confirmation_identity_improvement"], "joint_improvement": c197["winner_joint_identity_improvement"]},
        "C198": {"status": "behavior_and_trajectory_observed", "behavior_accuracy": c198["behavior"]["global_accuracy"], "programs": len(c198["behavior"]["eligible_programs"]), "external_gain_passed": c198["graph_role_coordinate_gain_external"]["passed"], "external_gain_improvement": c198["graph_role_coordinate_gain_external"]["improvement"]},
        "C199": {"status": "registered_gate_failed", "winner": c199["winner"], "primary_gate_passed": c199["primary_gate_passed"], "identity_improvement": c199["identity_improvement"], "semantic_composition_tested": c199["semantic_composition_tested"]},
        "C200": {"status": "typed_not_tested", "natural_deletion_rescue_tested": c200["natural_deletion_rescue_tested"], "reason": c200["reason"]},
        "C201": {"status": "behavior_confirmed_topology_descriptive", "holdout_accuracy": c201["behavior"]["holdout_accuracy"], "registered_raw_median_similarity": c201["median_pair_similarity"], "registered_gate_mechanical_value": c201["topology_gate_passed"]},
    }
    math_upgrade = {
        "stable_cross_program_predictive_object": False,
        "unseen_composition_law": False,
        "typed_causal_edit_and_rescue": False,
        "nontrivial_cross_model_invariant": bool(topology["exploratory_nontrivial_alignment"]),
    }
    adjudication = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "campaign_closed_with_local_observation_and_predictive_failure",
        "upstream_independent_audits": upstream_checks,
        "evidence_ledger": evidence,
        "c201_topology_reaudit": topology,
        "theory_adjudication": {
            "stable_theory_name": "Conditional Output-Field Closure Theory",
            "organization_principle": "reuse-difference-conditioning (RDC)",
            "supported_update": "A q23 perturbation has a reproducible signed q24-to-q25 response, while simple diagonal/gain descriptions do not predict it well across joint stimuli, natural programs, or unseen composite holdouts.",
            "not_supported": ["fixed semantic coordinate", "coordinatewise gear train", "language-level composition algebra", "natural necessity or rescue", "shared cross-model coordinate code"],
            "new_foundational_mathematics_required": False,
            "reason": "Existing linear algebra, finite differences, typed conditional systems, causal inference, and graph descriptions express every current result; the missing item is a stable predictive empirical object.",
        },
        "new_mathematics_upgrade_gate": {"checks": math_upgrade, "passed_count": int(sum(math_upgrade.values())), "required": 4, "gate_passed": all(math_upgrade.values())},
        "headline": "Signed local response is real, but no tested simple operator predicts the full field across registered breadth; natural causal closure remains untested, and C201 cross-model topology is descriptive pending a preregistered nontrivial null.",
        "next_authorization": "C203_precision_calibrated_nonlinear_response_ecology_campaign",
    }
    core.save(OUT / "analysis/adjudication.json", adjudication)
    asset = build_asset(adjudication)
    asset_meta = {"path": str(ASSET.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(ASSET), "bytes": ASSET.stat().st_size, "rows": len(asset["rows"]), "dimensions": len(asset["dimensions"]), "schema": asset["schema"]}
    checks = {
        "upstream": all(upstream_checks.values()),
        "ledger": set(evidence) == set(UPSTREAM) - {"C194"},
        "asset_shape": len(asset["dimensions"]) == DIM and all(len(row["values"]) == DIM for row in asset["rows"]),
        "finite": bool(np.isfinite(np.asarray([row["values"] for row in asset["rows"]], dtype=np.float32)).all()),
        "causal_boundary": not evidence["C200"]["natural_deletion_rescue_tested"],
        "c201_reclassified": topology["status"] == "exploratory_posthoc_diagnostic",
    }
    core.save(OUT / "analysis/public_asset.json", asset_meta)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": adjudication["created_at_utc"],
        "object": "read-only adjudication of frozen C194-C201 results plus coordinate-complete public atlas",
        "posthoc_diagnostic": "C201 centered role topology and exact 6! role-permutation null; exploratory only",
        "forbidden": ["retroactive gate repair", "attention", "MLP", "weight attribution", "semantic composition claim", "causal claim"],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": adjudication, "asset": asset_meta, "next_authorization": adjudication["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    run()
