#!/usr/bin/env python3
"""Phase1215: typed mechanism-hypothesis elimination engine.

This is a zero-model evidence-consolidation phase.  It compiles K1-K192 and
the typed Phase1214 boundary into independent, non-exclusive mechanism
claims.  Strong forms can be closed by decisive contradictions, while weak
survivors remain constrained or explicitly unidentifiable.  The experiment
selector operates on unresolved predictions and measurement prerequisites;
it does not pretend that overlapping mechanism claims form a probabilistic,
mutually exclusive hypothesis space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1215_mechanism_hypothesis_elimination_engine_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1215_mechanism_hypothesis_elimination_engine"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
EVIDENCE_PATH = OUT_ROOT / "analysis/evidence_bundles.json"
REGISTRY_PATH = OUT_ROOT / "analysis/hypothesis_registry.json"
MATRIX_PATH = OUT_ROOT / "analysis/falsification_matrix.json"
FAILURE_LEDGER_PATH = OUT_ROOT / "analysis/failure_type_ledger.json"
SELECTOR_PATH = OUT_ROOT / "analysis/experiment_selector.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
PHASE1214_FINAL = (
    ROOT
    / "tests/glm5/result/phase1214_functional_event_formation_dynamics/analysis/final.json"
)
MEMO_PATH = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 1215


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def k_range(start: int, end: int) -> list[str]:
    return [f"K{index}" for index in range(start, end + 1)]


def evidence_bundles() -> list[dict[str, Any]]:
    specifications = [
        ("E01", 1, 7, "output_control_timing", "Output competition, control fields, reuse, and causal ordering."),
        ("E02", 8, 10, "conditional_content_ledgers", "Conditional fingerprints exist, but representation, control, and execution require separate ledgers."),
        ("E03", 11, 15, "reused_functional_topology", "Shared shells and coarse phases repeat more reliably than fine identity coordinates."),
        ("E04", 16, 20, "addressing_and_controlled_causality", "Addresses, source states, attention, MLP, and residual all contribute under bounded protocols."),
        ("E05", 21, 29, "coordinate_and_geometry_boundaries", "Differences, cosine, Gram geometry, and single-position vectors do not by themselves identify execution."),
        ("E06", 30, 36, "interface_and_cross_model_boundaries", "Additive interfaces and physical cross-model coordinates fail; numerical qualification and claim scope are prerequisites."),
        ("E07", 37, 50, "exact_key_registry", "Exact-key lookup is real, but lookup does not imply payload transport or behavioral necessity; the registry is closed."),
        ("E08", 51, 65, "natural_semantic_behavior", "Natural semantic modulation has narrow candidates, strong interface dependence, and no stable shared direction."),
        ("E09", 66, 69, "numerical_measurement", "Finite-precision qualification and first-failure localization precede mechanism interpretation."),
        ("E10", 70, 77, "temporal_binding", "Temporal binding and transport candidates exist, but no universal depth or project-specific full-residual sufficiency."),
        ("E11", 78, 83, "first_divergence_and_donor_controls", "Aligned positions matter, while wrong donors and batch drift defeat naive donor identity claims."),
        ("E12", 84, 87, "known_truth_identification", "Matched contrasts identify known payloads, but scale-specific identities do not automatically transfer."),
        ("E13", 88, 99, "learned_composition_boundaries", "Separate factor availability and auxiliary supervision do not guarantee joint binding or unseen composition."),
        ("E14", 100, 105, "mechanism_morphology_camera", "Continuous cameras recover known functional classes, expose identifiability limits, and fail to guarantee free-network validity."),
        ("E15", 106, 108, "gauge_and_factor_quotients", "Gauge-safe factor quotients recover coarse functional classes, not hidden architecture names."),
        ("E16", 109, 114, "causal_use_and_hyperedges", "Use, redundancy, pure interactions, and conditional gates require typed causal interventions."),
        ("E17", 115, 119, "free_transformer_factor_use", "Endpoint causal-use profiles can transfer narrowly; identity claims require shell subtraction, permutation controls, and abstention."),
        ("E18", 120, 135, "intervention_response_compression", "Low-order probes predict many joint interventions but contain exceptions and do not imply compositional generalization."),
        ("E19", 136, 149, "generalization_and_formation", "Training constraints underdetermine extrapolation; formation bifurcates and early static cameras do not predict broadly."),
        ("E20", 150, 160, "relation_closure_and_training_regime", "Relation cameras and optimizer forks expose terminal structure, time decoupling, and norm-path alternatives."),
        ("E21", 161, 171, "implementation_truth_and_typed_evidence", "Implementation truth is a response spectrum; type-safe evidence and endpoint rescue do not yield early identity."),
        ("E22", 172, 182, "local_training_dynamics_and_rescue", "Local quotient transitions are causal and predictable; long-horizon identity and sparse rescue remain distributed and conditional."),
        ("E23", 183, 183, "conditional_trace_mapper", "A conditional causal trace mapper succeeds on known truth and explicitly abstains on observational twins."),
        ("E24", 184, 190, "object_attribute_natural_transfer", "Qwen3 has a bounded behavior-hidden-causal chain; cross-model and selective natural intervention gates remain unclosed."),
        ("E25", 191, 192, "identifiability_and_functional_target", "Semantic targets may be unidentifiable from public states; functional target quotients require typed minimal observations."),
    ]
    bundles = []
    for identifier, start, end, domain, claim in specifications:
        bundles.append(
            {
                "id": identifier,
                "k_refs": k_range(start, end),
                "domain": domain,
                "claim": claim,
                "evidence_type": "mixed_typed_bundle",
                "closure_policy": "Only prediction-level decisive links may close a strong form.",
            }
        )
    bundles.append(
        {
            "id": "E26",
            "k_refs": [],
            "phase_refs": ["Phase1214"],
            "domain": "formation_dynamics_boundary",
            "claim": (
                "Across 600 frozen checkpoints B1E0 was not observed; raw events occurred in three "
                "behavior-right-censored runs; breadth failed in both splits. These are protocol-typed "
                "boundaries, not a universal formation law."
            ),
            "evidence_type": "audited_boundary_without_new_k",
            "closure_policy": "Cannot close a mechanism class by itself because rule and calibration clocks were mixed.",
        }
    )
    return bundles


def prediction(
    identifier: str,
    statement: str,
    verdict: str,
    strength: str,
    fatal: bool,
    evidence: list[str],
    bundles: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "statement": statement,
        "verdict": verdict,
        "strength": strength,
        "fatal_if_decisively_contradicted": fatal,
        "evidence_refs": evidence,
        "evidence_bundles": bundles,
        "reason": reason,
    }


def hypothesis_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "H01_FIXED_COORDINATE_STRONG",
            "name": "One stable physical coordinate or direction encodes each content identity",
            "form": "strong",
            "domain": "cross_surface_cross_model_language",
            "predictions": [
                prediction("P1", "Identity directions survive material, template, language, and model changes.", "contradicted", "decisive", True, ["K25", "K26", "K32"], ["E05", "E06"], "Repeated physical-direction failures directly violate the strong invariant."),
                prediction("P2", "The same coordinate is selectively necessary and transportable.", "contradicted", "decisive", True, ["K29", "K30", "K77", "K187"], ["E05", "E06", "E10", "E24"], "Single-position and purified-direction necessity did not generalize."),
            ],
            "reopen_conditions": ["A new observation dimension proves the old contrasts non-identifying.", "A protocol or numerical defect invalidates the decisive tests."],
        },
        {
            "id": "H02_STATIC_GEOMETRY_EXECUTION_STRONG",
            "name": "Static similarity or Gram geometry directly identifies causal execution",
            "form": "strong",
            "domain": "representation_to_execution",
            "predictions": [
                prediction("P1", "High cosine or Gram identity uniquely predicts execution identity.", "contradicted", "decisive", True, ["K22", "K23", "K27", "K103", "K110", "K152"], ["E05", "E14", "E16", "E20"], "Geometry can be stable while use differs, and implementation labels can remain unidentifiable."),
            ],
            "reopen_conditions": ["A richer, preregistered geometry is shown to identify intervention response beyond matched controls."],
        },
        {
            "id": "H03_SINGLE_SITE_CIRCUIT_STRONG",
            "name": "One neuron, head, layer, or position is the universal necessary and sufficient mechanism",
            "form": "strong",
            "domain": "cross_material_execution",
            "predictions": [
                prediction("P1", "A single site has stable selective necessity and sufficiency.", "contradicted", "decisive", True, ["K28", "K29", "K112", "K178", "K179", "K181", "K182", "K187"], ["E05", "E16", "E22", "E24"], "Redundancy and distributed rescue defeat the universal single-site claim."),
            ],
            "reopen_conditions": ["A new task is explicitly scoped to a single known bottleneck and passes matched necessity controls."],
        },
        {
            "id": "H04_ATTENTION_WEIGHT_MECHANISM_STRONG",
            "name": "Attention weight or key lookup alone is the content execution mechanism",
            "form": "strong",
            "domain": "attention_routing",
            "predictions": [
                prediction("P1", "A stronger exact-key lookup entails stronger payload transport and behavior.", "contradicted", "decisive", True, ["K43", "K44", "K47", "K48", "K49"], ["E07"], "Lookup, payload, output, and necessity were empirically separated."),
            ],
            "reopen_conditions": ["A typed A-times-V and output-necessity chain closes on independent material."],
        },
        {
            "id": "H05_ADDITIVE_COMPOSITION_STRONG",
            "name": "Reusable pattern vectors compose by fixed addition and guarantee unseen combinations",
            "form": "strong",
            "domain": "compositional_generalization",
            "predictions": [
                prediction("P1", "Separately learned factors combine additively into unseen joint behavior.", "contradicted", "decisive", True, ["K28", "K88", "K89", "K90", "K94", "K99", "K134", "K136", "K137"], ["E05", "E13", "E18", "E19"], "Readable factors and low-order response compression repeatedly failed to guarantee unseen composition."),
            ],
            "reopen_conditions": ["A new algebra specifies a non-additive operation and makes a distinct preregistered prediction."],
        },
        {
            "id": "H06_CROSS_MODEL_PHYSICAL_INVARIANCE_STRONG",
            "name": "Models share the same physical layer, head, or direction for a mechanism",
            "form": "strong",
            "domain": "cross_model_physical_coordinates",
            "predictions": [
                prediction("P1", "Fine physical coordinates survive architecture and scale changes.", "contradicted", "decisive", True, ["K26", "K32", "K45", "K64", "K73", "K87", "K184"], ["E05", "E06", "E07", "E08", "E10", "E12", "E24"], "Functional repetition did not preserve fine physical coordinates or even behavior qualification across models."),
            ],
            "reopen_conditions": ["An explicit cross-model gauge map is learned on discovery data and predicts confirmation interventions."],
        },
        {
            "id": "H07_UNIQUE_IMPLEMENTATION_STRONG",
            "name": "Behavior and task uniquely determine an internal implementation",
            "form": "strong",
            "domain": "implementation_identity",
            "predictions": [
                prediction("P1", "Behaviorally equivalent systems have one recoverable physical implementation identity.", "contradicted", "decisive", True, ["K102", "K103", "K108", "K139", "K141", "K161", "K162", "K164", "K171", "K174"], ["E14", "E15", "E19", "E21", "E22"], "Known-truth twins, underdetermined extensions, and free trajectories expose implementation multiplicity."),
            ],
            "reopen_conditions": ["The claim is weakened to an explicitly typed quotient relative to an intervention family."],
        },
        {
            "id": "H08_READABILITY_IMPLIES_USE_STRONG",
            "name": "A decodable feature is necessarily used by the network",
            "form": "strong",
            "domain": "readout_to_causality",
            "predictions": [
                prediction("P1", "Stable readout entails causal use and necessity.", "contradicted", "decisive", True, ["K3", "K44", "K92", "K110", "K152", "K187"], ["E01", "E07", "E13", "E16", "E20", "E24"], "Readability, use, necessity, and output were repeatedly separated."),
            ],
            "reopen_conditions": ["No reopen as a universal implication; only task-scoped causal claims are admissible."],
        },
        {
            "id": "H09_MEMORIZATION_ONLY_STRONG_NULL",
            "name": "All successful behavior is lookup or memorization with no reusable causal computation",
            "form": "strong_null",
            "domain": "controlled_and_free_transformers",
            "predictions": [
                prediction("P1", "No structure should repeat across held-out material or causal forks.", "contradicted", "decisive", True, ["K6", "K11", "K13", "K19", "K120", "K131", "K173", "K175", "K176", "K185", "K186", "K189"], ["E01", "E03", "E04", "E18", "E22", "E24"], "Multiple controlled and free-network structures repeat across held-out panels and interventions."),
            ],
            "reopen_conditions": ["May remain a local explanation for a failed task, never as the universal null."],
        },
        {
            "id": "H10_CARRIER_ONLY_STRONG_NULL",
            "name": "Every apparent content mechanism is entirely a surface or carrier shortcut",
            "form": "strong_null",
            "domain": "controlled_content_effects",
            "predictions": [
                prediction("P1", "Matched carrier controls remove every content-specific causal effect.", "contradicted", "decisive", True, ["K8", "K19", "K58", "K70", "K72", "K86", "K115", "K185", "K186"], ["E02", "E04", "E08", "E10", "E12", "E17", "E24"], "Bounded content-specific effects survive several matched controls, although broad natural claims remain open."),
            ],
            "reopen_conditions": ["Carrier-only remains admissible for any new protocol whose matched-null advantage fails."],
        },
        {
            "id": "H11_CONDITIONAL_ROUTING",
            "name": "Queries and context condition addresses and route computation through shared machinery",
            "form": "bounded_weak",
            "domain": "controlled_routing_and_bounded_language",
            "predictions": [
                prediction("P1", "Causal differences appear only after the selector or query becomes available.", "supported", "decisive", True, ["K5", "K16", "K17", "K46"], ["E01", "E04", "E07"], "Causal order and address controls support query-conditioned routing."),
                prediction("P2", "Routing effects survive matched carrier controls and predict downstream use.", "supported", "bounded", True, ["K19", "K115", "K173", "K175", "K176", "K185", "K186"], ["E04", "E17", "E22", "E24"], "Controlled causal transfer supports a bounded routing account."),
                prediction("P3", "Semantic routing survives paraphrase, natural material, and independent models.", "untested", "none", True, ["K31", "K50", "K59", "K60", "K64", "K184", "K190"], ["E06", "E07", "E08", "E24"], "Current natural and cross-model gates do not close this prediction."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H12_RDC_CONDITIONAL_FIELD",
            "name": "Shared processing shells are reused while small differences are context-conditioned",
            "form": "bounded_weak",
            "domain": "functional_topology",
            "predictions": [
                prediction("P1", "Shared task and output shells repeat more strongly than fine identity coordinates.", "supported", "decisive", True, ["K6", "K7", "K11", "K12", "K13", "K14"], ["E01", "E03"], "The shared-shell and convergent-topology evidence is broad."),
                prediction("P2", "Condition-specific differences exist but rotate with material and interface.", "supported", "decisive", True, ["K8", "K9", "K24", "K25", "K26"], ["E02", "E05"], "Small conditional fingerprints coexist with coordinate non-conservation."),
                prediction("P3", "Reuse is globally minimal or resource-optimal.", "unidentifiable", "none", False, ["K35"], ["E06"], "No architecture or resource counterfactual identifies minimality."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H13_REDUNDANT_CAUSAL_HYPERGRAPH",
            "name": "Execution is organized by redundant and gated causal hyperedges",
            "form": "bounded_weak",
            "domain": "intervention_response",
            "predictions": [
                prediction("P1", "Single deletions can be weak while joint interventions expose use.", "supported", "decisive", True, ["K111", "K112", "K120", "K126", "K127", "K181", "K182"], ["E16", "E18", "E22"], "Known-truth and free-network response surfaces show redundancy and joint leverage."),
                prediction("P2", "Time-resolved necessity and rescue recover a stable minimal hypergraph.", "untested", "none", True, ["K178", "K179", "K187", "K190"], ["E22", "E24"], "Rescue and natural necessity remain bounded or failed under current operators."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H14_FUNCTIONAL_QUOTIENT",
            "name": "Mechanism identity is an equivalence class under typed functional interventions",
            "form": "bounded_weak",
            "domain": "typed_intervention_families",
            "predictions": [
                prediction("P1", "Physical states may differ while typed response spectra remain equivalent.", "supported", "decisive", True, ["K101", "K102", "K103", "K108", "K161", "K162", "K164", "K172"], ["E14", "E15", "E21", "E22"], "Known-truth and free systems establish quotient-relative mechanism identity."),
                prediction("P2", "Quotient events predict held-out interventions better than physical labels.", "supported", "decisive", True, ["K115", "K120", "K173", "K175", "K176", "K177", "K183"], ["E17", "E18", "E22", "E23"], "Several held-out response predictions pass within typed domains."),
                prediction("P3", "A selective quotient intervention exists for natural semantic targets.", "untested", "none", True, ["K190", "K191", "K192"], ["E24", "E25"], "The public state contract can be non-identifying and the natural Qwen intervention failed selectivity controls."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H15_DYNAMIC_TRAJECTORY",
            "name": "Mechanism information resides in local transitions and trajectories rather than fixed terminal coordinates",
            "form": "bounded_weak",
            "domain": "training_dynamics",
            "predictions": [
                prediction("P1", "Matched local update directions produce repeatable held-out functional transitions.", "supported", "decisive", True, ["K173", "K175", "K176", "K177"], ["E22"], "Same-parent and calibration-to-evaluation transitions provide direct support."),
                prediction("P2", "A fixed early physical identity persists through long training.", "contradicted", "decisive", False, ["K149", "K156", "K171", "K174"], ["E19", "E20", "E21", "E22"], "The weak trajectory account does not require fixed identity; long-horizon identity is explicitly rejected."),
                prediction("P3", "Pre-rule trajectory features add predictive value beyond loss, accuracy, confidence, norms, and compute.", "untested", "none", True, ["K144", "K149", "K156", "Phase1214"], ["E19", "E20", "E26"], "The required three-clock incremental test has not been run."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H16_FORMATION_BIFURCATION",
            "name": "Task, initialization, data, and optimization select among multiple formation paths",
            "form": "bounded_weak",
            "domain": "training_formation",
            "predictions": [
                prediction("P1", "Formation outcomes vary systematically with task, seed, architecture, and optimizer path.", "supported", "decisive", True, ["K122", "K139", "K141", "K143", "K145", "K157", "K158", "K160", "Phase1214"], ["E18", "E19", "E20", "E26"], "Multiple controlled studies show formation and implementation bifurcation."),
                prediction("P2", "A stable cross-task formation law predicts which trajectories form before behavior.", "untested", "none", True, ["K149", "K156", "Phase1214"], ["E19", "E20", "E26"], "Existing early cameras and Phase1214 breadth did not establish such a law."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H17_PATTERN_COMPOSITION_ALGEBRA",
            "name": "Natural language patterns obey a reusable composition algebra",
            "form": "open_program",
            "domain": "natural_language_general",
            "predictions": [
                prediction("P1", "An operation learned on some pattern families predicts unseen semantic compositions.", "unidentifiable", "none", True, ["K99", "K134", "K136", "K137", "K183", "K190", "K191"], ["E13", "E18", "E19", "E23", "E24", "E25"], "Known-truth calibration exists, but natural semantic targets and unseen composition are not jointly identified."),
            ],
            "reopen_conditions": [],
        },
        {
            "id": "H18_GLOBAL_EFFICIENCY_OPTIMALITY",
            "name": "The learned code is globally minimal, energy-optimal, or brain-isomorphic",
            "form": "open_program",
            "domain": "resource_and_architecture_counterfactuals",
            "predictions": [
                prediction("P1", "Resource-matched alternatives are dominated by the observed implementation.", "unidentifiable", "none", True, ["K24", "K35", "K160"], ["E05", "E06", "E20"], "No intervention over resource, architecture, and training objective identifies global optimality."),
            ],
            "reopen_conditions": [],
        },
    ]


def failure_types() -> list[dict[str, Any]]:
    return [
        {"id": "F01_BEHAVIOR_OBJECT_NOT_FORMED", "meaning": "The behavior required for a mechanism query did not form broadly enough."},
        {"id": "F02_NUMERICAL_INVALID", "meaning": "Non-finite or precision-dependent execution invalidates the scientific object."},
        {"id": "F03_TARGET_UNIDENTIFIABLE", "meaning": "The observation contract cannot distinguish the target from an admissible twin."},
        {"id": "F04_INTERVENTION_NONSELECTIVE", "meaning": "The intervention moves carrier, protocol, or unrelated state with the target."},
        {"id": "F05_MECHANISM_PREDICTION_CONTRADICTED", "meaning": "A qualified object directly violates a preregistered necessary prediction."},
        {"id": "F06_EXTERNAL_VALIDITY_FAILED", "meaning": "A calibrated method fails on an independently qualified target domain."},
        {"id": "F07_RIGHT_CENSORED", "meaning": "The event is not observed inside the frozen horizon; absence is not established."},
        {"id": "F08_MEASUREMENT_NOT_CALIBRATED", "meaning": "The observation or clock lacks known-truth construct validation."},
        {"id": "F09_AUDIT_TYPE_MISMATCH", "meaning": "Scientific values are correct or unknown, but evidence serialization or audit types disagree."},
        {"id": "F10_PROTOCOL_SCOPE_EXCEEDED", "meaning": "A claim exceeds the domain authorized by the frozen contract."},
    ]


CURRENT_CAPABILITIES = {
    "typed_evidence_compiler",
    "known_truth_causal_use_camera",
    "free_transformer_local_transition",
    "right_censoring_ledger",
}

OPEN_MEASUREMENT_GAPS = {
    "three_clock_construct",
    "readability_use_separation_over_time",
    "censoring_clock_validation",
}


def candidate_experiments() -> list[dict[str, Any]]:
    return [
        {
            "id": "T01_KNOWN_TRUTH_THREE_CLOCK_ZOO",
            "title": "Known-truth rule-calibration-interface clock zoo",
            "domain": "known_truth",
            "prerequisites": [],
            "closes": sorted(OPEN_MEASUREMENT_GAPS),
            "targets": [],
            "cost_units": 1,
            "auto_executable": True,
            "reason": "Phase1214 mixed rule and calibration; scientific formation tests are blocked until the clocks are construct-valid.",
        },
        {
            "id": "T02_FACTORIAL_FREE_FORMATION",
            "title": "Task-by-vocabulary-by-seed free-Transformer three-clock formation panel",
            "domain": "free_transformer",
            "prerequisites": sorted(OPEN_MEASUREMENT_GAPS),
            "closes": ["factor_orthogonality", "continuous_response_trajectory"],
            "targets": ["H11_CONDITIONAL_ROUTING.P3", "H12_RDC_CONDITIONAL_FIELD.P3", "H15_DYNAMIC_TRAJECTORY.P3", "H16_FORMATION_BIFURCATION.P2"],
            "cost_units": 6,
            "auto_executable": False,
            "reason": "Separates task, vocabulary, seed, rule, confidence, and interface after camera calibration.",
        },
        {
            "id": "T03_PRE_RULE_INCREMENTAL_PREDICTION",
            "title": "Pre-rule incremental formation prediction against scalar baselines",
            "domain": "free_transformer",
            "prerequisites": ["factor_orthogonality", "continuous_response_trajectory"],
            "closes": ["pre_rule_incremental_prediction"],
            "targets": ["H15_DYNAMIC_TRAJECTORY.P3", "H16_FORMATION_BIFURCATION.P2"],
            "cost_units": 4,
            "auto_executable": False,
            "reason": "Tests whether mechanism events predict formation beyond accuracy, loss, confidence, norms, and compute.",
        },
        {
            "id": "T04_TIME_RESOLVED_REDUNDANCY",
            "title": "Time-resolved single and joint necessity-rescue response surface",
            "domain": "free_transformer",
            "prerequisites": ["continuous_response_trajectory"],
            "closes": ["time_resolved_causal_hypergraph"],
            "targets": ["H13_REDUNDANT_CAUSAL_HYPERGRAPH.P2", "H14_FUNCTIONAL_QUOTIENT.P3"],
            "cost_units": 7,
            "auto_executable": False,
            "reason": "Distinguishes distributed use from a merely readable quotient event.",
        },
        {
            "id": "T05_NATURAL_QWEN_TYPED_QUOTIENT",
            "title": "Natural Qwen typed functional-quotient intervention",
            "domain": "pretrained_language",
            "prerequisites": ["pre_rule_incremental_prediction", "time_resolved_causal_hypergraph"],
            "closes": ["natural_semantic_selectivity"],
            "targets": ["H11_CONDITIONAL_ROUTING.P3", "H14_FUNCTIONAL_QUOTIENT.P3", "H17_PATTERN_COMPOSITION_ALGEBRA.P1"],
            "cost_units": 12,
            "auto_executable": False,
            "reason": "Natural transfer is last because target identifiability and intervention selectivity remain open.",
        },
        {
            "id": "T06_CROSS_MODEL_GAUGE_TRANSFER",
            "title": "Cross-model learned-gauge functional transfer",
            "domain": "cross_model_language",
            "prerequisites": ["natural_semantic_selectivity"],
            "closes": ["cross_model_functional_invariance"],
            "targets": ["H11_CONDITIONAL_ROUTING.P3", "H14_FUNCTIONAL_QUOTIENT.P3"],
            "cost_units": 20,
            "auto_executable": False,
            "reason": "Physical invariance is closed; only a learned functional gauge is scientifically admissible.",
        },
    ]


def validate_static_registry(
    bundles: list[dict[str, Any]], hypotheses: list[dict[str, Any]], experiments: list[dict[str, Any]]
) -> None:
    seen_k: list[int] = []
    bundle_ids = {bundle["id"] for bundle in bundles}
    for bundle in bundles:
        for reference in bundle.get("k_refs", []):
            if not reference.startswith("K") or not reference[1:].isdigit():
                raise RuntimeError(f"invalid K reference: {reference}")
            seen_k.append(int(reference[1:]))
    if sorted(seen_k) != list(range(1, 193)) or len(seen_k) != len(set(seen_k)):
        raise RuntimeError("K1-K192 must be covered exactly once by evidence bundles")
    hypothesis_ids = {hypothesis["id"] for hypothesis in hypotheses}
    if len(hypothesis_ids) != len(hypotheses):
        raise RuntimeError("duplicate hypothesis id")
    valid_verdicts = {"supported", "contradicted", "unidentifiable", "untested"}
    valid_strengths = {"decisive", "bounded", "none"}
    prediction_ids: set[str] = set()
    for hypothesis in hypotheses:
        for item in hypothesis["predictions"]:
            full_id = f"{hypothesis['id']}.{item['id']}"
            if full_id in prediction_ids:
                raise RuntimeError(f"duplicate prediction id: {full_id}")
            prediction_ids.add(full_id)
            if item["verdict"] not in valid_verdicts or item["strength"] not in valid_strengths:
                raise RuntimeError(f"invalid prediction adjudication: {full_id}")
            if not set(item["evidence_bundles"]).issubset(bundle_ids):
                raise RuntimeError(f"unknown evidence bundle in {full_id}")
    experiment_ids = {experiment["id"] for experiment in experiments}
    if len(experiment_ids) != len(experiments):
        raise RuntimeError("duplicate experiment id")
    for experiment in experiments:
        unknown_targets = set(experiment["targets"]) - prediction_ids
        if unknown_targets:
            raise RuntimeError(f"unknown experiment targets: {sorted(unknown_targets)}")


def adjudicate_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    predictions = hypothesis["predictions"]
    fatal_decisive = [
        item
        for item in predictions
        if item["fatal_if_decisively_contradicted"]
        and item["verdict"] == "contradicted"
        and item["strength"] == "decisive"
    ]
    supported = [item for item in predictions if item["verdict"] == "supported"]
    unidentifiable = [item for item in predictions if item["verdict"] == "unidentifiable"]
    untested = [item for item in predictions if item["verdict"] == "untested"]
    bounded_contradictions = [
        item
        for item in predictions
        if item["verdict"] == "contradicted" and item not in fatal_decisive
    ]
    if fatal_decisive:
        status = "CLOSED_STRONG_FORM"
    elif supported:
        status = "ACTIVE_CONSTRAINED"
    elif unidentifiable:
        status = "UNIDENTIFIABLE"
    else:
        status = "OPEN_UNTESTED"
    return {
        **hypothesis,
        "status": status,
        "fatal_decisive_contradictions": [item["id"] for item in fatal_decisive],
        "supported_predictions": [item["id"] for item in supported],
        "bounded_contradictions": [item["id"] for item in bounded_contradictions],
        "unresolved_predictions": [item["id"] for item in unidentifiable + untested],
    }


def compiler_fixtures() -> list[dict[str, Any]]:
    cases = [
        ("fatal_decisive", [prediction("P", "x", "contradicted", "decisive", True, [], [], "x")], "CLOSED_STRONG_FORM"),
        ("nonfatal_decisive", [prediction("P", "x", "contradicted", "decisive", False, [], [], "x")], "OPEN_UNTESTED"),
        ("supported", [prediction("P", "x", "supported", "bounded", True, [], [], "x")], "ACTIVE_CONSTRAINED"),
        ("unidentifiable", [prediction("P", "x", "unidentifiable", "none", True, [], [], "x")], "UNIDENTIFIABLE"),
        ("untested", [prediction("P", "x", "untested", "none", True, [], [], "x")], "OPEN_UNTESTED"),
        ("support_plus_unresolved", [prediction("P1", "x", "supported", "decisive", True, [], [], "x"), prediction("P2", "x", "untested", "none", True, [], [], "x")], "ACTIVE_CONSTRAINED"),
    ]
    rows = []
    for identifier, predictions, expected in cases:
        observed = adjudicate_hypothesis(
            {
                "id": identifier,
                "name": identifier,
                "form": "fixture",
                "domain": "fixture",
                "predictions": predictions,
                "reopen_conditions": [],
            }
        )["status"]
        rows.append({"id": identifier, "expected": expected, "observed": observed, "pass": observed == expected})
    return rows


def score_experiments(
    experiments: list[dict[str, Any]], registry: list[dict[str, Any]]
) -> dict[str, Any]:
    unresolved = {
        f"{hypothesis['id']}.{prediction_item['id']}"
        for hypothesis in registry
        if hypothesis["status"] != "CLOSED_STRONG_FORM"
        for prediction_item in hypothesis["predictions"]
        if prediction_item["verdict"] in {"untested", "unidentifiable"}
    }
    scored = []
    for experiment in experiments:
        prerequisites = set(experiment["prerequisites"])
        eligible = prerequisites.issubset(CURRENT_CAPABILITIES)
        prerequisite_gain = len(set(experiment["closes"]) & OPEN_MEASUREMENT_GAPS)
        open_prediction_targets = sorted(set(experiment["targets"]) & unresolved)
        guaranteed_resolution_count = len(open_prediction_targets)
        # Lexicographic score: close blocking measurement contracts first, then
        # resolve more open predictions, then prefer lower declared cost.
        rank_key = [
            1 if eligible and prerequisite_gain > 0 else 0,
            prerequisite_gain if eligible else -1,
            guaranteed_resolution_count if eligible else -1,
            -int(experiment["cost_units"]),
        ]
        scored.append(
            {
                **experiment,
                "eligible": eligible,
                "missing_prerequisites": sorted(prerequisites - CURRENT_CAPABILITIES),
                "prerequisite_gain": prerequisite_gain,
                "open_prediction_targets": open_prediction_targets,
                "guaranteed_resolution_count": guaranteed_resolution_count,
                "rank_key": rank_key,
            }
        )
    eligible_rows = [row for row in scored if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: tuple(row["rank_key"])) if eligible_rows else None
    return {
        "selection_rule": (
            "Nonexclusive claim lattice: first close blocking measurement prerequisites, then maximize "
            "the count of unresolved typed predictions addressed, then minimize declared cost."
        ),
        "probabilistic_information_gain_used": False,
        "reason_probability_is_withheld": (
            "Mechanism claims can coexist and no credible prior over complete mechanism worlds is available."
        ),
        "current_capabilities": sorted(CURRENT_CAPABILITIES),
        "open_measurement_gaps": sorted(OPEN_MEASUREMENT_GAPS),
        "experiments": sorted(scored, key=lambda row: tuple(row["rank_key"]), reverse=True),
        "selected_experiment": selected["id"] if selected else None,
        "selected_auto_executable": bool(selected and selected["auto_executable"]),
    }


def build_protocol() -> dict[str, Any]:
    bundles = evidence_bundles()
    hypotheses = hypothesis_registry()
    experiments = candidate_experiments()
    validate_static_registry(bundles, hypotheses, experiments)
    upstream = read_json(PHASE1214_FINAL)
    return {
        "phase": PHASE,
        "schema_version": "phase1215.mechanism_elimination.protocol.v1",
        "created_at": utc_now(),
        "purpose": "compile K1-K192 into a typed, nonexclusive mechanism-claim lattice and select the next discriminating experiment",
        "scope": {
            "new_model_run": False,
            "new_neural_data": False,
            "new_empirical_k_item": False,
            "reinterprets_k1_k192": False,
            "scientific_adjudications_are_expert_coded": True,
        },
        "logic": {
            "hypotheses_mutually_exclusive": False,
            "probabilistic_priors_available": False,
            "entropy_claim_forbidden": True,
            "closure_rule": "fatal necessary prediction contradicted by decisive evidence",
            "support_rule": "compatibility never equals exclusive confirmation",
            "unidentifiable_rule": "lack of an identifying observation or intervention produces explicit abstention",
        },
        "source_hashes": {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)},
        "upstream": {
            "phase1214_final_sha256": file_sha256(PHASE1214_FINAL),
            "phase1214_final_digest": upstream["final_digest"],
            "phase1214_auto_continue": upstream["auto_continue"],
            "memo_sha256_before_phase1215": file_sha256(MEMO_PATH),
        },
        "evidence_bundles": bundles,
        "hypotheses": hypotheses,
        "failure_types": failure_types(),
        "candidate_experiments": experiments,
        "exclusions": [
            "This compiler does not discover hypotheses automatically.",
            "Audit validates compilation consistency, not the truth of expert semantic links.",
            "Registry-relative counts are not posterior probabilities or ontology-complete version-space sizes.",
            "Phase1214 B1E0=0 is not registered as a universal law.",
            "No natural-language, cross-model, or new-mathematics claim is authorized.",
        ],
    }


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}:
        raise RuntimeError("source changed after preregistration")
    upstream = read_json(PHASE1214_FINAL)
    if protocol["upstream"]["phase1214_final_sha256"] != file_sha256(PHASE1214_FINAL):
        raise RuntimeError("Phase1214 upstream changed")
    if protocol["upstream"]["phase1214_final_digest"] != upstream["final_digest"]:
        raise RuntimeError("Phase1214 final digest changed")
    validate_static_registry(protocol["evidence_bundles"], protocol["hypotheses"], protocol["candidate_experiments"])
    return protocol


def preregister() -> None:
    if PROTOCOL_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("Phase1215 protocol or outcomes already exist")
    protocol = build_protocol()
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"phase": PHASE, "protocol_digest": protocol["protocol_digest"]}))


def compile_registry() -> None:
    if SUMMARY_PATH.exists():
        raise RuntimeError("Phase1215 outcomes already exist")
    protocol = verify_protocol()
    registry = [adjudicate_hypothesis(item) for item in protocol["hypotheses"]]
    fixtures = compiler_fixtures()
    if not all(row["pass"] for row in fixtures):
        raise RuntimeError("compiler known-truth fixtures failed")

    status_counts: dict[str, int] = {}
    for hypothesis in registry:
        status_counts[hypothesis["status"]] = status_counts.get(hypothesis["status"], 0) + 1
    eliminated = [row["id"] for row in registry if row["status"] == "CLOSED_STRONG_FORM"]
    active = [row["id"] for row in registry if row["status"] == "ACTIVE_CONSTRAINED"]
    unidentifiable = [row["id"] for row in registry if row["status"] == "UNIDENTIFIABLE"]
    open_untested = [row["id"] for row in registry if row["status"] == "OPEN_UNTESTED"]
    version_space = active + unidentifiable + open_untested

    matrix_rows = []
    for hypothesis in registry:
        for item in hypothesis["predictions"]:
            matrix_rows.append(
                {
                    "hypothesis": hypothesis["id"],
                    "hypothesis_status": hypothesis["status"],
                    "prediction": item["id"],
                    "statement": item["statement"],
                    "verdict": item["verdict"],
                    "strength": item["strength"],
                    "fatal": item["fatal_if_decisively_contradicted"],
                    "evidence_refs": item["evidence_refs"],
                    "evidence_bundles": item["evidence_bundles"],
                }
            )
    selector = score_experiments(protocol["candidate_experiments"], registry)
    phase1214_failure = {
        "phase": 1214,
        "primary_failure_types": ["F01_BEHAVIOR_OBJECT_NOT_FORMED", "F07_RIGHT_CENSORED"],
        "measurement_warning": "F08_MEASUREMENT_NOT_CALIBRATED",
        "not_classified_as": "F05_MECHANISM_PREDICTION_CONTRADICTED",
        "reason": (
            "Behavior breadth failed and 14 runs were right-censored; B mixed top-1 rule and 0.95 calibration, "
            "so the event hypothesis itself was not cleanly contradicted."
        ),
    }
    failure_ledger = {"types": protocol["failure_types"], "phase1214": phase1214_failure}
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "typed_nonexclusive_hypothesis_lattice_compiled",
        "protocol_digest": protocol["protocol_digest"],
        "compiler_fixtures": fixtures,
        "hypothesis_count": len(registry),
        "status_counts": status_counts,
        "eliminated_strong_forms": eliminated,
        "active_constrained": active,
        "unidentifiable": unidentifiable,
        "open_untested": open_untested,
        "registry_relative_version_space": version_space,
        "registry_relative_version_space_size": len(version_space),
        "ontology_complete_version_space_claimed": False,
        "probability_or_entropy_claimed": False,
        "selected_next_experiment": selector["selected_experiment"],
        "auto_continue_candidate": selector["selected_auto_executable"],
        "new_k_item": False,
    }
    write_json(EVIDENCE_PATH, {"bundles": protocol["evidence_bundles"]})
    write_json(REGISTRY_PATH, {"hypotheses": registry})
    write_json(MATRIX_PATH, {"rows": matrix_rows})
    write_json(FAILURE_LEDGER_PATH, failure_ledger)
    write_json(SELECTOR_PATH, selector)
    write_json(SUMMARY_PATH, summary)
    print(canonical_json(summary))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    selector = read_json(SELECTOR_PATH)
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["status"],
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "summary": summary,
        "claims": {
            "methodological_upgrade": "confirmed",
            "strong_form_elimination_is_registry_relative": True,
            "survivors_are_confirmed_mechanisms": False,
            "phase1214_formation_law": "not_confirmed",
            "new_mathematics_required": "not_supported",
        },
        "authorized_next": {
            "experiment": selector["selected_experiment"],
            "scope": "known-truth measurement calibration only",
            "automatic_execution": selector["selected_auto_executable"],
            "pretrained_model_run": False,
            "natural_language_claim": False,
        },
        "new_k_item": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "compile", "finalize"))
    command = parser.parse_args().command
    {"preregister": preregister, "compile": compile_registry, "finalize": finalize}[command]()


if __name__ == "__main__":
    main()
