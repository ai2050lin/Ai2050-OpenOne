"""Phase1264: exact finite-population risk certificate calibration.

This campaign replaces C013's two-panel point threshold with three explicitly
separate objects:

1. exact risk over an exhaustively enumerated, registered finite population;
2. a finite selection-sample point estimate;
3. a simultaneous bounded-loss certificate with explicit abstention.

The target is a risk-certified predictive type, not generator identity and not
an epsilon quotient.  Approximate closeness is treated as a typed neighborhood
because it need not be transitive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "tests/glm5/phase1263_c013_predictive_equivalence_independent_rescue_calibration.py"
BASE_SPEC = importlib.util.spec_from_file_location("phase1263_c013_base", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("cannot load the frozen C013 numerical primitives")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)


OUT = ROOT / "tests/glm5/result/phase1264_c014_population_risk_certificate_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC = OUT / "material/public_systems.jsonl"
TRUTH = OUT / "material/private_population_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/system_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1264_c014_population_risk_certificate_calibration_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C014-WP01-001.json"
PROBE = ROOT / "tests/glm5_temp/phase1264_c014_population_risk_certificate_probe.json"

PHASE = 1264
CAMPAIGN = "C014"
CONTRACT_ID = "EXP-C014-WP01-001"
DIMENSION = base.DIMENSION
LATENT_RANK = base.LATENT_RANK
OUTPUT_DIMENSION = base.OUTPUT_DIMENSION
RFF_WIDTH = base.RFF_WIDTH
TASKS = 4
REPLICATES = 6
REGISTRY_SPLITS = ("calibration", "structural_holdout")
SOURCE_PROFILES = (
    "additive_exact",
    "linear_exact",
    "quadratic_exact",
    "rff_exact",
    "smooth_linear_alias",
    "pass_boundary_gradient",
    "earlier_boundary_gradient",
    "hidden_collision",
    "unsupported_high_frequency",
)
CANDIDATES = tuple(base.CANDIDATES)
SYSTEM_COUNT = len(REGISTRY_SPLITS) * len(SOURCE_PROFILES) * TASKS * REPLICATES

DISCOVERY_COUNT = 1024
ORACLE_UNIVERSE_COUNT = 8192
SELECTION_DRAWS = 16384
DONOR_COUNT = 768
CONFIRMATION_COUNT = 1024
POPULATION_PASS_MAX = 0.050
POPULATION_EARLIER_FAIL_MIN = 0.150
GLOBAL_ERROR_BUDGET = 0.01
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * SYSTEM_COUNT * len(CANDIDATES) / GLOBAL_ERROR_BUDGET)
    / (2.0 * SELECTION_DRAWS)
)
ROBUST_MARGIN_MULTIPLIER = 2.0
PASS_BOUNDARY_TARGETS = (0.0300, 0.0450, 0.0498, 0.0502, 0.0650, 0.1100)
EARLIER_BOUNDARY_TARGETS = (0.0900, 0.1350, 0.1498, 0.1502, 0.1900, 0.2600)

THRESHOLDS = {
    "population_pass_max": POPULATION_PASS_MAX,
    "population_earlier_fail_min": POPULATION_EARLIER_FAIL_MIN,
    "global_error_budget": GLOBAL_ERROR_BUDGET,
    "certificate_radius": CERTIFICATE_RADIUS,
    "robust_margin_multiplier": ROBUST_MARGIN_MULTIPLIER,
    "certificate_false_authorizations_max": 0,
    "rescue_authorization_upper_max": 0.025,
    "robust_coverage_min": 0.95,
    "ambiguous_abstention_min": 1.0,
    "split_robust_coverage_min": 0.90,
    "minimum_robust_class_count": 20,
    "boundary_profile_count_min": 80,
    "confirmation_a1_error_max": 0.09,
    "confirmation_j_error_max": 0.09,
    "identity_accuracy_min": 0.98,
    "nuisance_sensitivity_max": 0.03,
    "block_remaining_max": 1.0e-10,
    "dual_donor_rescue_error_max": 0.12,
    "wrong_rescue_false_target_max": 0.02,
    "donor_source_gap_min": 1.0e-5,
    "algebraic_replay_error_max": 1.0e-10,
    "manifold_p95_max": 0.22,
    "partial_cut_remaining_min": 0.02,
    "control_error_floor_min": 0.20,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_system_id(split: str, profile: str, task: int, replicate: int) -> str:
    payload = f"1264|{split}|{profile}|{task}|{replicate}".encode("utf-8")
    return "P" + hashlib.sha256(payload).hexdigest()[:15]


def seed_for(split: str, profile: str, task: int, replicate: int) -> int:
    payload = f"seed|1264|{split}|{profile}|{task}|{replicate}".encode("utf-8")
    return 1_264_000_000 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 500_000_000


@dataclass
class CertifiedSystem(base.FunctionalSystem):
    amplitude: float = 1.0

    def interaction(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a0: torch.Tensor,
        B: torch.Tensor,
        sign: torch.Tensor,
    ) -> torch.Tensor:
        x_latent = torch.cat((a, b), dim=1)
        linear = x_latent @ self.linear
        outer = torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)
        quadratic = outer @ self.quadratic
        x = torch.cat((a0, B), dim=1)
        omega, phase = base.global_rff(self.device)
        angle = x @ omega + phase
        phi = math.sqrt(1.0 / RFF_WIDTH) * torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
        nonlinear = phi @ self.rff_weight
        profile = self.source_family
        if profile == "additive_exact":
            coords = torch.zeros_like(linear)
        elif profile == "linear_exact":
            coords = linear
        elif profile == "quadratic_exact":
            coords = 1.8 * quadratic
        elif profile == "rff_exact":
            coords = 1.8 * nonlinear
        elif profile == "smooth_linear_alias":
            coords = linear + 0.008 * nonlinear
        elif profile == "pass_boundary_gradient":
            coords = linear + self.amplitude * nonlinear
        elif profile == "earlier_boundary_gradient":
            coords = 1.8 * quadratic + self.amplitude * nonlinear
        elif profile == "hidden_collision":
            coords = 0.72 * sign[:, None] * self.collision_vector[None, :]
        elif profile == "unsupported_high_frequency":
            high = torch.stack(
                (
                    torch.sin(7.0 * a[:, 0] + 3.0 * b[:, 1]),
                    torch.cos(6.0 * b[:, 0] - 2.0 * a[:, 1]),
                    torch.sin(5.0 * (a[:, 0] + b[:, 0])),
                ),
                dim=1,
            )
            coords = high @ self.unsupported_weight
        else:
            raise ValueError(profile)
        return coords @ self.j_basis.T


def bounded_loss_vector(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    error = torch.sum((predicted - truth) ** 2, dim=1)
    scale = torch.sum(truth**2, dim=1) + 0.25
    return torch.clamp(error / scale, min=0.0, max=1.0)


def fit_models(data: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {name: base.fit_candidate(name, data) for name in CANDIDATES}


def population_losses(
    models: dict[str, dict[str, Any]], universe: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        name: bounded_loss_vector(base.predict(model, universe["A0"], universe["B"]), universe["A1"])
        for name, model in models.items()
    }


def mean_risks(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(values.mean().item()) for name, values in losses.items()}


def select_point(risks: dict[str, float]) -> tuple[str, str]:
    passing = [name for name in CANDIDATES if risks[name] <= POPULATION_PASS_MAX]
    if not passing:
        return "abstain", "out_of_library"
    candidate = passing[0]
    index = CANDIDATES.index(candidate)
    if any(risks[name] < POPULATION_EARLIER_FAIL_MIN for name in CANDIDATES[:index]):
        return "abstain", "equivalence_overlap"
    return candidate, "population_margin_rule"


def select_certificate(bounds: dict[str, dict[str, float]]) -> tuple[str, str]:
    passing = [name for name in CANDIDATES if bounds[name]["upper"] <= POPULATION_PASS_MAX]
    if not passing:
        return "abstain", "pass_not_certified"
    candidate = passing[0]
    index = CANDIDATES.index(candidate)
    if any(bounds[name]["lower"] < POPULATION_EARLIER_FAIL_MIN for name in CANDIDATES[:index]):
        return "abstain", "earlier_failure_not_certified"
    return candidate, "simultaneously_certified"


def boundary_target(profile: str, replicate: int) -> tuple[str, float] | None:
    if profile == "pass_boundary_gradient":
        return "linear", PASS_BOUNDARY_TARGETS[replicate]
    if profile == "earlier_boundary_gradient":
        return "quadratic", EARLIER_BOUNDARY_TARGETS[replicate]
    return None


def exact_candidate_risk(system: CertifiedSystem, candidate: str) -> float:
    discovery = system.make_partition("discovery", DISCOVERY_COUNT)
    universe = system.make_partition("reference_a", ORACLE_UNIVERSE_COUNT)
    model = base.fit_candidate(candidate, discovery)
    return float(
        bounded_loss_vector(base.predict(model, universe["A0"], universe["B"]), universe["A1"]).mean().item()
    )


def tune_amplitude(profile: str, seed: int, task: int, device: torch.device, target: float) -> float:
    candidate = "linear" if profile == "pass_boundary_gradient" else "quadratic"
    low = 0.0
    high = 0.25
    for _ in range(8):
        risk = exact_candidate_risk(CertifiedSystem(profile, seed, task, device, high), candidate)
        if risk >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError(f"could not bracket boundary target {profile} {target}")
    for _ in range(22):
        middle = 0.5 * (low + high)
        risk = exact_candidate_risk(CertifiedSystem(profile, seed, task, device, middle), candidate)
        if risk < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def oracle_truth(system: CertifiedSystem) -> dict[str, Any]:
    discovery = system.make_partition("discovery", DISCOVERY_COUNT)
    universe = system.make_partition("reference_a", ORACLE_UNIVERSE_COUNT)
    models = fit_models(discovery)
    risks = mean_risks(population_losses(models, universe))
    exact_class, reason = select_point(risks)
    robust = False
    if exact_class != "abstain":
        index = CANDIDATES.index(exact_class)
        margin = ROBUST_MARGIN_MULTIPLIER * CERTIFICATE_RADIUS
        robust = (
            risks[exact_class] <= POPULATION_PASS_MAX - margin
            and all(risks[name] >= POPULATION_EARLIER_FAIL_MIN + margin for name in CANDIDATES[:index])
        )
    pass_distance = min(abs(value - POPULATION_PASS_MAX) for value in risks.values())
    fail_distance = min(abs(value - POPULATION_EARLIER_FAIL_MIN) for value in risks.values())
    return {
        "exact_class": exact_class,
        "truth_reason": reason,
        "population_risks": risks,
        "robust_actionable": robust,
        "nearest_pass_boundary": pass_distance,
        "nearest_earlier_fail_boundary": fail_distance,
    }


def make_system_rows(device: torch.device, probe: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    splits = REGISTRY_SPLITS[:1] if probe else REGISTRY_SPLITS
    replicates = range(2) if probe else range(REPLICATES)
    for split in splits:
        for profile in SOURCE_PROFILES:
            for task in range(TASKS):
                for replicate in replicates:
                    seed = seed_for(split, profile, task, replicate)
                    target = boundary_target(profile, replicate)
                    amplitude = 1.0
                    if target is not None:
                        amplitude = tune_amplitude(profile, seed, task, device, target[1])
                    system = CertifiedSystem(profile, seed, task, device, amplitude)
                    record = oracle_truth(system)
                    system_id = opaque_system_id(split, profile, task, replicate)
                    public.append(
                        {
                            "system_id": system_id,
                            "registry_split": split,
                            "task_id": task,
                            "replicate": replicate,
                            "state_dimension": DIMENSION,
                            "oracle_universe_count": ORACLE_UNIVERSE_COUNT,
                            "selection_draws": SELECTION_DRAWS,
                        }
                    )
                    truth.append(
                        {
                            "system_id": system_id,
                            "source_profile": profile,
                            "seed": seed,
                            "amplitude": amplitude,
                            "boundary_target": None if target is None else {"candidate": target[0], "risk": target[1]},
                            **record,
                        }
                    )
    return public, truth


def protocol_payload(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = {name: sum(row["exact_class"] == name for row in truth) for name in (*CANDIDATES, "abstain")}
    robust_counts = {name: sum(row["exact_class"] == name and row["robust_actionable"] for row in truth) for name in CANDIDATES}
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1264.c014.population_risk_certificate.v1",
        "claim_type": "known_truth_finite_population_risk_certificate_calibration",
        "question": "Can a simultaneous bounded-loss certificate attain zero false authorization and nontrivial robust coverage against exact finite-population risk?",
        "systems": len(public),
        "registered_finite_population": ORACLE_UNIVERSE_COUNT,
        "selection_draws_with_replacement": SELECTION_DRAWS,
        "discovery_cases": DISCOVERY_COUNT,
        "donor_cases_per_world": DONOR_COUNT,
        "confirmation_cases": CONFIRMATION_COUNT,
        "candidate_order": list(CANDIDATES) + ["abstain"],
        "class_counts": class_counts,
        "robust_class_counts": robust_counts,
        "thresholds": THRESHOLDS,
        "pass_boundary_targets": list(PASS_BOUNDARY_TARGETS),
        "earlier_boundary_targets": list(EARLIER_BOUNDARY_TARGETS),
        "certificate": {
            "loss_range": [0.0, 1.0],
            "bound": "Hoeffding plus union bound across all registered systems and four candidates",
            "radius": CERTIFICATE_RADIUS,
            "global_error_budget": GLOBAL_ERROR_BUDGET,
            "authorization": "candidate upper <= pass and every earlier candidate lower >= fail",
        },
        "truth_definition": "exact enumeration of the registered finite universe for models fitted only on discovery",
        "estimand": "risk-certified predictive type; approximate neighborhoods are not asserted to be equivalence relations",
        "baselines": ["point_estimator", "simultaneous_interval_certificate", "always_abstain"],
        "rescue_provenance": {
            "block": "remove the full constructed target channel",
            "donor_world_a": "fit selected type on donor_discovery",
            "donor_world_b": "fit selected type on an independently seeded reference_train partition",
            "partial_cut_control": "remove either coordinate fragment while retaining the complement",
            "algebraic_replay": "positive leakage sentinel only",
            "authorization": "rescue runs only when the selected type has certified upper risk <= 0.025",
        },
        "hard_stops": [
            "Any certificate false authorization closes C014 WP01.",
            "Always-abstain is invalid because robust coverage must be nontrivial.",
            "No boundary target, threshold, draw count, profile, or seed changes after preregistration.",
            "Failure blocks free-Transformer and pretrained-model execution.",
            "Pass authorizes only one separately frozen free-Transformer external-validity phase.",
        ],
        "forbidden_claims": [
            "epsilon quotient transitivity",
            "continuous-distribution population truth",
            "free-network mechanism",
            "natural-language mechanism",
            "qwen3",
            "new mathematics",
        ],
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "c013_numerical_dependency": file_sha256(BASE_PATH),
            "contract": file_sha256(CONTRACT),
        },
        "public_digest": digest(public),
        "truth_digest": digest(truth),
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "CUDA float64 deterministic finite-population algebra",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError(f"protocol already exists: {PROTOCOL}")
    if not torch.cuda.is_available():
        raise RuntimeError("preregistration requires CUDA")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and frozen contract must exist before preregistration")
    public, truth = make_system_rows(torch.device("cuda"))
    write_jsonl(PUBLIC, public)
    write_jsonl(TRUTH, truth)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(public, truth))
    print(canonical_json({"status": "preregistered", "systems": len(public), "class_counts": read_json(PROTOCOL)["class_counts"], "robust_counts": read_json(PROTOCOL)["robust_class_counts"]}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    public = read_jsonl(PUBLIC)
    truth = read_jsonl(TRUTH)
    expected = protocol_payload(public, truth)
    if protocol["protocol_digest"] != expected["protocol_digest"] or protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("protocol or source drift")
    if protocol["public_digest"] != digest(public) or protocol["truth_digest"] != digest(truth):
        raise RuntimeError("material digest drift")
    return protocol, public, truth


def selection_indices(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 8_888_811)
    return torch.randint(0, ORACLE_UNIVERSE_COUNT, (SELECTION_DRAWS,), generator=generator, device=device)


def confidence_bounds(sample_risks: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "point": risk,
            "lower": max(0.0, risk - CERTIFICATE_RADIUS),
            "upper": min(1.0, risk + CERTIFICATE_RADIUS),
        }
        for name, risk in sample_risks.items()
    }


def dual_donor_confirmation(
    system: CertifiedSystem,
    model: dict[str, Any],
    donor_a: dict[str, Any],
    donor_b: dict[str, Any],
    data: dict[str, torch.Tensor],
) -> dict[str, float]:
    metrics_a = base.evaluate_confirmation(system, model, donor_a, data)
    flipped = {**data, "donor_noise": -data["donor_noise"]}
    metrics_b = base.evaluate_confirmation(system, model, donor_b, flipped)
    mask = torch.zeros((DIMENSION,), dtype=torch.float64, device=data["A1"].device)
    mask[: DIMENSION // 2] = 1.0
    effect_a = (data["A1"] * mask[None, :]) @ system.advance.T
    effect_b = (data["A1"] * (1.0 - mask)[None, :]) @ system.advance.T
    partial_a = data["M_H11"] - effect_a
    partial_b = data["M_H11"] - effect_b
    partial_a_output = torch.tanh(partial_a @ system.output_weight.T)
    partial_b_output = torch.tanh(partial_b @ system.output_weight.T)
    target_scale = torch.linalg.vector_norm(data["Y_target"] - data["Y_base"]).clamp_min(1.0e-12)
    return {
        "a1_relative_error": metrics_a["a1_relative_error"],
        "j_relative_error": metrics_a["j_relative_error"],
        "correct_identity_accuracy": metrics_a["correct_identity_accuracy"],
        "wrong_identity_accuracy": metrics_a["wrong_identity_accuracy"],
        "nuisance_sensitivity": metrics_a["nuisance_sensitivity"],
        "block_remaining": metrics_a["block_remaining"],
        "donor_a_rescue_error": metrics_a["independent_rescue_error"],
        "donor_b_rescue_error": metrics_b["independent_rescue_error"],
        "donor_a_identity": metrics_a["independent_rescue_identity"],
        "donor_b_identity": metrics_b["independent_rescue_identity"],
        "wrong_rescue_false_target": max(metrics_a["wrong_rescue_false_target"], metrics_b["wrong_rescue_false_target"]),
        "donor_a_source_gap": metrics_a["donor_source_gap"],
        "donor_b_source_gap": metrics_b["donor_source_gap"],
        "algebraic_replay_error": metrics_a["algebraic_replay_error"],
        "on_manifold_relative_distance_p95": max(metrics_a["on_manifold_relative_distance_p95"], metrics_b["on_manifold_relative_distance_p95"]),
        "partial_cut_a_remaining": float((torch.linalg.vector_norm(partial_a_output - data["Y_base"]) / target_scale).item()),
        "partial_cut_b_remaining": float((torch.linalg.vector_norm(partial_b_output - data["Y_base"]) / target_scale).item()),
    }


def confirmation_passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["a1_relative_error"] <= THRESHOLDS["confirmation_a1_error_max"]
        and metrics["j_relative_error"] <= THRESHOLDS["confirmation_j_error_max"]
        and metrics["correct_identity_accuracy"] >= THRESHOLDS["identity_accuracy_min"]
        and metrics["wrong_identity_accuracy"] >= THRESHOLDS["identity_accuracy_min"]
        and metrics["nuisance_sensitivity"] <= THRESHOLDS["nuisance_sensitivity_max"]
        and metrics["block_remaining"] <= THRESHOLDS["block_remaining_max"]
        and max(metrics["donor_a_rescue_error"], metrics["donor_b_rescue_error"]) <= THRESHOLDS["dual_donor_rescue_error_max"]
        and min(metrics["donor_a_identity"], metrics["donor_b_identity"]) >= THRESHOLDS["identity_accuracy_min"]
        and metrics["wrong_rescue_false_target"] <= THRESHOLDS["wrong_rescue_false_target_max"]
        and min(metrics["donor_a_source_gap"], metrics["donor_b_source_gap"]) >= THRESHOLDS["donor_source_gap_min"]
        and metrics["algebraic_replay_error"] <= THRESHOLDS["algebraic_replay_error_max"]
        and metrics["on_manifold_relative_distance_p95"] <= THRESHOLDS["manifold_p95_max"]
        and min(metrics["partial_cut_a_remaining"], metrics["partial_cut_b_remaining"]) >= THRESHOLDS["partial_cut_remaining_min"]
    )


def run_system(public: dict[str, Any], truth: dict[str, Any], device: torch.device, include_confirmation: bool = True) -> dict[str, Any]:
    system = CertifiedSystem(
        str(truth["source_profile"]),
        int(truth["seed"]),
        int(public["task_id"]),
        device,
        float(truth["amplitude"]),
    )
    discovery = system.make_partition("discovery", DISCOVERY_COUNT)
    universe = system.make_partition("reference_a", ORACLE_UNIVERSE_COUNT)
    models = fit_models(discovery)
    losses = population_losses(models, universe)
    indices = selection_indices(int(truth["seed"]), device)
    sample_risks = {name: float(values[indices].mean().item()) for name, values in losses.items()}
    point_class, point_reason = select_point(sample_risks)
    bounds = confidence_bounds(sample_risks)
    certificate_class, certificate_reason = select_certificate(bounds)
    rescue_authorized = (
        certificate_class != "abstain"
        and bounds[certificate_class]["upper"] <= THRESHOLDS["rescue_authorization_upper_max"]
    )
    confirmation = None
    confirmation_ok = True
    controls: dict[str, float] = {}
    if include_confirmation and rescue_authorized:
        donor_a_data = system.make_partition("donor_discovery", DONOR_COUNT)
        donor_b_data = system.make_partition("reference_train", DONOR_COUNT)
        donor_a = base.fit_candidate(certificate_class, donor_a_data)
        donor_b = base.fit_candidate(certificate_class, donor_b_data)
        confirmation_data = system.make_partition("confirmation", CONFIRMATION_COUNT)
        confirmation = dual_donor_confirmation(system, models[certificate_class], donor_a, donor_b, confirmation_data)
        confirmation_ok = confirmation_passes(confirmation)
        mean_prediction = discovery["A1"].mean(dim=0, keepdim=True).expand_as(universe["A1"])
        controls = {
            "mean_response_error": base.relative(mean_prediction, universe["A1"], universe["A1"]),
            "sign_flip_error": base.relative(-universe["A0"], universe["A1"], universe["A1"]),
        }
    return {
        "system_id": public["system_id"],
        "registry_split": public["registry_split"],
        "task_id": public["task_id"],
        "replicate": public["replicate"],
        "point_class": point_class,
        "point_reason": point_reason,
        "certificate_class": certificate_class,
        "certificate_reason": certificate_reason,
        "sample_risks": sample_risks,
        "confidence_bounds": bounds,
        "always_abstain_class": "abstain",
        "rescue_authorized": rescue_authorized,
        "confirmation": confirmation,
        "confirmation_passed": confirmation_ok,
        "controls": controls,
    }


def probe() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA")
    public, truth = make_system_rows(torch.device("cuda"), probe=True)
    rows = [run_system(p, t, torch.device("cuda"), include_confirmation=True) for p, t in zip(public, truth)]
    class_counts = {name: sum(t["exact_class"] == name for t in truth) for name in (*CANDIDATES, "abstain")}
    robust_counts = {name: sum(t["exact_class"] == name and t["robust_actionable"] for t in truth) for name in CANDIDATES}
    false_authorizations = sum(r["certificate_class"] != "abstain" and r["certificate_class"] != t["exact_class"] for r, t in zip(rows, truth))
    point_false = sum(r["point_class"] != "abstain" and r["point_class"] != t["exact_class"] for r, t in zip(rows, truth))
    target_errors = [
        abs(t["population_risks"][t["boundary_target"]["candidate"]] - t["boundary_target"]["risk"])
        for t in truth
        if t["boundary_target"] is not None
    ]
    profile_summary = {
        profile: {
            "exact": {name: sum(t["source_profile"] == profile and t["exact_class"] == name for t in truth) for name in (*CANDIDATES, "abstain")},
            "robust": sum(t["source_profile"] == profile and t["robust_actionable"] for t in truth),
            "certified": sum(t["source_profile"] == profile and r["certificate_class"] != "abstain" for r, t in zip(rows, truth)),
            "confirmation_failures": sum(
                t["source_profile"] == profile and not r["confirmation_passed"] for r, t in zip(rows, truth)
            ),
            "maximum_rescue_error": max(
                [
                    max(r["confirmation"]["donor_a_rescue_error"], r["confirmation"]["donor_b_rescue_error"])
                    for r, t in zip(rows, truth)
                    if t["source_profile"] == profile and r["confirmation"] is not None
                ]
                or [0.0]
            ),
        }
        for profile in SOURCE_PROFILES
    }
    result = {
        "created_at_utc": utc_now(),
        "systems": len(rows),
        "certificate_radius": CERTIFICATE_RADIUS,
        "class_counts": class_counts,
        "robust_counts": robust_counts,
        "certificate_false_authorizations": false_authorizations,
        "point_false_authorizations": point_false,
        "boundary_target_max_error": max(target_errors) if target_errors else None,
        "certificate_selected": sum(r["certificate_class"] != "abstain" for r in rows),
        "rescue_authorized": sum(r["rescue_authorized"] for r in rows),
        "confirmation_failures": sum(not r["confirmation_passed"] for r in rows),
        "minimum_partial_cut_remaining": min(
            min(r["confirmation"]["partial_cut_a_remaining"], r["confirmation"]["partial_cut_b_remaining"])
            for r in rows
            if r["confirmation"] is not None
        ),
        "maximum_dual_donor_rescue_error": max(
            max(r["confirmation"]["donor_a_rescue_error"], r["confirmation"]["donor_b_rescue_error"])
            for r in rows
            if r["confirmation"] is not None
        ),
        "profile_summary": profile_summary,
    }
    atomic_json(PROBE, result)
    print(canonical_json(result))


def run(device_name: str) -> None:
    protocol, public, truth = verify_protocol()
    if COMPLETE.exists():
        raise RuntimeError("formal run already completed")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass")
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    truth_by_id = {row["system_id"]: row for row in truth}
    started = time.perf_counter()
    rows = []
    for index, public_row in enumerate(public, start=1):
        rows.append(run_system(public_row, truth_by_id[public_row["system_id"]], device))
        if index % 24 == 0:
            print(canonical_json({"completed": index, "total": len(public)}), flush=True)
    elapsed = time.perf_counter() - started
    write_jsonl(RAW, rows)
    atomic_json(
        SUMMARY,
        {
            "phase": PHASE,
            "contract_id": CONTRACT_ID,
            "created_at_utc": utc_now(),
            "systems": len(rows),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "device": torch.cuda.get_device_name(0),
            "raw_digest": digest(rows),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "raw_digest": digest(rows)})


def analyze() -> None:
    protocol, public, truth_rows = verify_protocol()
    rows = read_jsonl(RAW)
    truth = {row["system_id"]: row for row in truth_rows}
    if len(rows) != len(public):
        raise RuntimeError("formal row count mismatch")
    false_rows = [r for r in rows if r["certificate_class"] != "abstain" and r["certificate_class"] != truth[r["system_id"]]["exact_class"]]
    point_false_rows = [r for r in rows if r["point_class"] != "abstain" and r["point_class"] != truth[r["system_id"]]["exact_class"]]
    robust_rows = [r for r in rows if truth[r["system_id"]]["robust_actionable"]]
    robust_correct = [r for r in robust_rows if r["certificate_class"] == truth[r["system_id"]]["exact_class"]]
    ambiguous_rows = [r for r in rows if truth[r["system_id"]]["exact_class"] == "abstain"]
    ambiguous_correct = [r for r in ambiguous_rows if r["certificate_class"] == "abstain"]
    boundary_rows = [r for r in rows if truth[r["system_id"]]["source_profile"] in {"pass_boundary_gradient", "earlier_boundary_gradient"}]
    certified_rows = [r for r in rows if r["certificate_class"] != "abstain"]
    rescue_rows = [r for r in rows if r["rescue_authorized"]]
    robust_coverage = len(robust_correct) / max(1, len(robust_rows))
    ambiguous_abstention = len(ambiguous_correct) / max(1, len(ambiguous_rows))
    class_counts = {
        name: sum(truth[r["system_id"]]["robust_actionable"] and truth[r["system_id"]]["exact_class"] == name for r in rows)
        for name in CANDIDATES
    }
    split_metrics = {}
    for split in REGISTRY_SPLITS:
        split_rows = [r for r in rows if r["registry_split"] == split]
        split_robust = [r for r in split_rows if truth[r["system_id"]]["robust_actionable"]]
        split_metrics[split] = {
            "false_authorizations": sum(r["certificate_class"] != "abstain" and r["certificate_class"] != truth[r["system_id"]]["exact_class"] for r in split_rows),
            "robust_coverage": sum(r["certificate_class"] == truth[r["system_id"]]["exact_class"] for r in split_robust) / max(1, len(split_robust)),
        }
    control_values = [value for row in rescue_rows for value in row["controls"].values()]
    gates = {
        "G-EXACT-FINITE-POPULATION": all(
            abs(
                mean_risks(
                    population_losses(
                        fit_models(CertifiedSystem(str(t["source_profile"]), int(t["seed"]), int(next(p["task_id"] for p in public if p["system_id"] == t["system_id"])), torch.device("cuda"), float(t["amplitude"])).make_partition("discovery", DISCOVERY_COUNT)),
                        CertifiedSystem(str(t["source_profile"]), int(t["seed"]), int(next(p["task_id"] for p in public if p["system_id"] == t["system_id"])), torch.device("cuda"), float(t["amplitude"])).make_partition("reference_a", ORACLE_UNIVERSE_COUNT),
                    )
                )[name]
                - t["population_risks"][name]
            ) <= 1.0e-12
            for t in truth_rows[:8]
            for name in CANDIDATES
        ),
        "G-ZERO-FALSE-AUTHORIZATION": len(false_rows) <= THRESHOLDS["certificate_false_authorizations_max"],
        "G-ROBUST-COVERAGE": robust_coverage >= THRESHOLDS["robust_coverage_min"],
        "G-AMBIGUOUS-ABSTENTION": ambiguous_abstention >= THRESHOLDS["ambiguous_abstention_min"],
        "G-NONTRIVIAL-VS-ALWAYS-ABSTAIN": len(robust_correct) > 0 and robust_coverage > 0.0,
        "G-BOUNDARY-GRADIENT": len(boundary_rows) >= THRESHOLDS["boundary_profile_count_min"],
        "G-ROBUST-CLASS-BREADTH": all(count >= THRESHOLDS["minimum_robust_class_count"] for count in class_counts.values()),
        "G-SPLIT-BREADTH": all(item["false_authorizations"] == 0 and item["robust_coverage"] >= THRESHOLDS["split_robust_coverage_min"] for item in split_metrics.values()),
        "G-DUAL-DONOR-RESCUE": bool(rescue_rows) and all(row["confirmation_passed"] for row in rescue_rows),
        "G-NEGATIVE-CONTROLS": bool(control_values) and min(control_values) >= THRESHOLDS["control_error_floor_min"],
    }
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "systems": len(rows),
        "cases": len(rows) * (DISCOVERY_COUNT + ORACLE_UNIVERSE_COUNT + SELECTION_DRAWS + 2 * DONOR_COUNT + CONFIRMATION_COUNT),
        "certificate_radius": CERTIFICATE_RADIUS,
        "exact_class_counts": protocol["class_counts"],
        "robust_class_counts": class_counts,
        "certificate_false_authorizations": len(false_rows),
        "point_false_authorizations": len(point_false_rows),
        "certificate_total_coverage": len(certified_rows) / len(rows),
        "certificate_robust_coverage": robust_coverage,
        "always_abstain_robust_coverage": 0.0,
        "ambiguous_abstention": ambiguous_abstention,
        "boundary_systems": len(boundary_rows),
        "rescue_authorized_systems": len(rescue_rows),
        "split_metrics": split_metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "authorization": {
            "free_transformer_population_certificate": all(gates.values()),
            "qwen3": False,
            "glm4": False,
            "ds7b": False,
        },
        "claim_boundary": "Exact only for the registered finite populations and bounded i.i.d. selection draws. Approximate neighborhoods are not quotient classes. No free network, pretrained model, or natural language is tested here.",
        "failed_system_ids": [row["system_id"] for row in false_rows],
        "point_false_system_ids": [row["system_id"] for row in point_false_rows],
        "raw_digest": digest(rows),
    }
    final["final_digest"] = digest(final)
    atomic_json(ANALYSIS, {"created_at_utc": utc_now(), "gates": gates, "failed_system_ids": final["failed_system_ids"]})
    atomic_json(FINAL, final)
    print(canonical_json({"passed": final["passed"], "gates": gates, "false_authorizations": len(false_rows), "robust_coverage": robust_coverage, "point_false_authorizations": len(point_false_rows)}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--audit", choices=("pre", "final"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    actions = sum(bool(value) for value in (args.probe, args.preregister, args.run, args.analyze, args.audit))
    if actions != 1:
        raise SystemExit("choose exactly one action")
    if args.probe:
        probe()
    elif args.preregister:
        preregister(args.force)
    elif args.run:
        run(args.device)
    elif args.analyze:
        analyze()
    else:
        run_auditor(str(args.audit))


if __name__ == "__main__":
    main()
