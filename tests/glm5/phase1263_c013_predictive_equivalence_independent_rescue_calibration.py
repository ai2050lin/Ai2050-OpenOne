"""Phase1263: known-truth calibration of predictive functional equivalence.

The target is the least-complex frozen candidate that predicts two large,
independent reference panels with a registered risk margin.  It is deliberately
not the name of the program that generated the data.  Ambiguous and
out-of-library systems are retained and must produce typed abstention.

The path test also separates three objects that earlier phases mixed together:
an oracle channel block, a rescue estimated from an independent donor fold,
and a subtract/add algebraic replay sentinel.  The replay must work, but it is
explicitly excluded from evidence for mediation.
"""

from __future__ import annotations

import argparse
import hashlib
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
OUT = ROOT / "tests/glm5/result/phase1263_c013_predictive_equivalence_independent_rescue_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC = OUT / "material/public_systems.jsonl"
TRUTH = OUT / "material/private_functional_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/system_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1263_c013_predictive_equivalence_independent_rescue_calibration_audit.py"
PROBE = ROOT / "tests/glm5_temp/phase1263_c013_predictive_equivalence_probe.json"

PHASE = 1263
CAMPAIGN = "C013"
CONTRACT_ID = "EXP-C013-WP01-001"
DIMENSION = 12
LATENT_RANK = 2
OUTPUT_DIMENSION = 6
TASKS = 4
REPLICATES = 5
REGISTRY_SPLITS = ("calibration", "structural_holdout")
SOURCE_FAMILIES = (
    "additive",
    "linear",
    "quadratic",
    "rff_nonlinear",
    "smooth_alias_linear",
    "boundary_overlap",
    "hidden_collision",
    "unsupported_high_frequency",
)
CANDIDATES = ("additive", "linear", "quadratic", "rff")
PARTITION_COUNTS = {
    "reference_train": 1024,
    "reference_a": 768,
    "reference_b": 768,
    "discovery": 512,
    "selection": 512,
    "donor_discovery": 512,
    "confirmation": 768,
}
RIDGE = 1.0e-7
RFF_WIDTH = 32
REFERENCE_PASS_MAX = 0.050
REFERENCE_EARLIER_FAIL_MIN = 0.120
THRESHOLDS = {
    "selection_pass_max": 0.050,
    "selection_earlier_fail_min": 0.120,
    "confirmation_a1_error_max": 0.085,
    "confirmation_j_error_max": 0.085,
    "wrong_identity_accuracy_min": 0.98,
    "nuisance_sensitivity_max": 0.025,
    "block_remaining_max": 1.0e-10,
    "independent_rescue_error_max": 0.10,
    "independent_rescue_identity_min": 0.98,
    "wrong_rescue_false_target_max": 0.02,
    "donor_source_gap_min": 1.0e-5,
    "algebraic_replay_error_max": 1.0e-10,
    "manifold_p95_max": 0.20,
    "control_error_floor_min": 0.20,
    "type_accuracy_min": 1.0,
    "abstention_accuracy_min": 1.0,
    "minimum_class_count": 20,
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


def opaque_system_id(split: str, family: str, task: int, replicate: int) -> str:
    payload = f"1263|{split}|{family}|{task}|{replicate}".encode("utf-8")
    return "E" + hashlib.sha256(payload).hexdigest()[:15]


def seed_for(split: str, family: str, task: int, replicate: int) -> int:
    payload = f"seed|1263|{split}|{family}|{task}|{replicate}".encode("utf-8")
    return 1_263_000_000 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 500_000_000


def orthogonal(seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(DIMENSION, DIMENSION))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return torch.tensor(q * signs, dtype=torch.float64, device=device)


def global_rff(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(12630077)
    omega = torch.tensor(rng.normal(scale=0.85, size=(2 * DIMENSION, RFF_WIDTH)), dtype=torch.float64, device=device)
    phase = torch.tensor(rng.uniform(0.0, 2.0 * math.pi, size=(RFF_WIDTH,)), dtype=torch.float64, device=device)
    return omega, phase


def ridge_fit(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = features.double()
    y = target.double()
    gram = x.T @ x
    scale = float(torch.trace(gram).item()) / max(1, gram.shape[0])
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=x.device)
    return torch.linalg.solve(gram + RIDGE * max(scale, 1.0) * eye, x.T @ y)


def support_projector(samples: torch.Tensor) -> torch.Tensor:
    _u, singular, vh = torch.linalg.svd(samples.double(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 1.0e-12:
        return torch.zeros((samples.shape[1], samples.shape[1]), dtype=torch.float64, device=samples.device)
    rank = min(LATENT_RANK, int(torch.sum(singular > singular[0] * 1.0e-7).item()))
    basis = vh[:rank].T.contiguous()
    return basis @ basis.T


def design(name: str, a0: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.cat((a0, b), dim=1)
    ones = torch.ones((x.shape[0], 1), dtype=torch.float64, device=x.device)
    linear = torch.cat((ones, x), dim=1)
    if name == "linear":
        return linear
    cross = torch.einsum("bi,bj->bij", a0, b).reshape(a0.shape[0], -1)
    quadratic = torch.cat((linear, cross), dim=1)
    if name == "quadratic":
        return quadratic
    if name == "rff":
        omega, phase = global_rff(x.device)
        angles = x @ omega + phase
        features = math.sqrt(1.0 / RFF_WIDTH) * torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return torch.cat((quadratic, features), dim=1)
    raise ValueError(name)


def fit_candidate(name: str, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    a_projector = support_projector(data["A0"])
    b_projector = support_projector(data["B"])
    projected_a = data["A0"] @ a_projector
    projected_b = data["B"] @ b_projector
    if name == "additive":
        return {"name": name, "a_projector": a_projector, "b_projector": b_projector}
    features = design(name, projected_a, projected_b)
    weights = ridge_fit(features, data["J"])
    return {"name": name, "weights": weights, "a_projector": a_projector, "b_projector": b_projector}


def predict(model: dict[str, Any], a0: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    projected_a = a0 @ model["a_projector"]
    projected_b = b @ model["b_projector"]
    if model["name"] == "additive":
        return projected_a
    return projected_a + design(model["name"], projected_a, projected_b) @ model["weights"]


def relative(predicted: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(predicted - truth) / torch.linalg.vector_norm(scale).clamp_min(1.0e-12)).item())


@dataclass
class FunctionalSystem:
    source_family: str
    seed: int
    task_id: int
    device: torch.device

    def __post_init__(self) -> None:
        self.q = orthogonal(self.seed + 101, self.device)
        self.advance = orthogonal(self.seed + 211, self.device)
        self.a_basis = self.q[:, :LATENT_RANK]
        self.b_basis = 0.35 * self.q[:, :LATENT_RANK] + math.sqrt(1.0 - 0.35**2) * self.q[:, LATENT_RANK:2 * LATENT_RANK]
        self.j_basis = self.q[:, 2 * LATENT_RANK:3 * LATENT_RANK]
        self.nuisance_basis = self.q[:, 3 * LATENT_RANK:4 * LATENT_RANK]
        rng = np.random.default_rng(self.seed + 307 * self.task_id)
        self.linear = torch.tensor(rng.normal(scale=0.34, size=(2 * LATENT_RANK, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.quadratic = torch.tensor(rng.normal(scale=0.42, size=(LATENT_RANK * LATENT_RANK, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.rff_weight = torch.tensor(rng.normal(scale=0.55, size=(2 * RFF_WIDTH, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.unsupported_weight = torch.tensor(rng.normal(scale=0.52, size=(3, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.collision_vector = torch.tensor(rng.normal(size=(LATENT_RANK,)), dtype=torch.float64, device=self.device)
        self.collision_vector /= torch.linalg.vector_norm(self.collision_vector).clamp_min(1.0e-12)
        output = torch.tensor(rng.normal(size=(OUTPUT_DIMENSION, DIMENSION)), dtype=torch.float64, device=self.device)
        self.output_weight = output / torch.linalg.vector_norm(output, dim=1, keepdim=True).clamp_min(1.0e-12)
        self.base = torch.tensor(rng.normal(scale=0.12, size=(DIMENSION,)), dtype=torch.float64, device=self.device)

    def interaction(self, a: torch.Tensor, b: torch.Tensor, a0: torch.Tensor, B: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        x_latent = torch.cat((a, b), dim=1)
        linear = x_latent @ self.linear
        outer = torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)
        quadratic = outer @ self.quadratic
        x = torch.cat((a0, B), dim=1)
        omega, phase = global_rff(self.device)
        angle = x @ omega + phase
        phi = math.sqrt(1.0 / RFF_WIDTH) * torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
        nonlinear = phi @ self.rff_weight
        if self.source_family == "additive":
            coords = torch.zeros_like(linear)
        elif self.source_family == "linear":
            coords = linear
        elif self.source_family == "quadratic":
            coords = quadratic
        elif self.source_family == "rff_nonlinear":
            coords = nonlinear
        elif self.source_family == "smooth_alias_linear":
            coords = linear + 0.012 * nonlinear
        elif self.source_family == "boundary_overlap":
            coords = linear + 0.115 * nonlinear
        elif self.source_family == "hidden_collision":
            coords = 0.72 * sign[:, None] * self.collision_vector[None, :]
        elif self.source_family == "unsupported_high_frequency":
            high = torch.stack((torch.sin(7.0 * a[:, 0] + 3.0 * b[:, 1]), torch.cos(6.0 * b[:, 0] - 2.0 * a[:, 1]), torch.sin(5.0 * (a[:, 0] + b[:, 0]))), dim=1)
            coords = high @ self.unsupported_weight
        else:
            raise ValueError(self.source_family)
        return coords @ self.j_basis.T

    def make_partition(self, partition: str, count: int) -> dict[str, torch.Tensor]:
        partition_ids = {name: index + 1 for index, name in enumerate(PARTITION_COUNTS)}
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 100_000 * partition_ids[partition])
        if self.source_family == "hidden_collision":
            if count % 2:
                raise ValueError("collision partition count must be even")
            half = count // 2
            a_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            b_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            wrong_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            a = torch.cat((a_half, a_half), dim=0)
            b = torch.cat((b_half, b_half), dim=0)
            wrong_a = torch.cat((wrong_half, wrong_half), dim=0)
            sign = torch.cat((torch.ones(half, device=self.device), -torch.ones(half, device=self.device))).double()
        else:
            a = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            b = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            wrong_a = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            sign = torch.where(torch.arange(count, device=self.device) % 2 == 0, 1.0, -1.0).double()
        A0 = a @ self.a_basis.T
        B = b @ self.b_basis.T
        wrong_A0 = wrong_a @ self.a_basis.T
        J = self.interaction(a, b, A0, B, sign)
        wrong_J = self.interaction(wrong_a, b, wrong_A0, B, -sign)
        A1 = A0 + J
        wrong_A1 = wrong_A0 + wrong_J
        base_noise = 0.015 * torch.randn((count, DIMENSION), generator=generator, dtype=torch.float64, device=self.device)
        nuisance = 0.035 * torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device) @ self.nuisance_basis.T
        donor_noise = 0.0025 * torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device) @ self.nuisance_basis.T
        H00 = self.base[None, :] + base_noise
        H01 = H00 + B
        H11 = H01 + A1
        Hwrong11 = H01 + wrong_A1
        M_H01 = H01 @ self.advance.T
        M_effect = A1 @ self.advance.T
        M_H11 = M_H01 + M_effect
        output_base = torch.tanh(M_H01 @ self.output_weight.T)
        output_target = torch.tanh(M_H11 @ self.output_weight.T)
        return {
            "A0": A0,
            "B": B,
            "J": J,
            "A1": A1,
            "wrong_A0": wrong_A0,
            "wrong_J": wrong_J,
            "wrong_A1": wrong_A1,
            "nuisance": nuisance,
            "donor_noise": donor_noise,
            "H01": H01,
            "H11": H11,
            "Hwrong11": Hwrong11,
            "M_H01": M_H01,
            "M_effect": M_effect,
            "M_H11": M_H11,
            "Y_base": output_base,
            "Y_target": output_target,
        }


def candidate_risks(system: FunctionalSystem, train: dict[str, torch.Tensor], panels: list[dict[str, torch.Tensor]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    models = {name: fit_candidate(name, train) for name in CANDIDATES}
    risks = {
        name: [relative(predict(model, panel["A0"], panel["B"]), panel["A1"], panel["A1"]) for panel in panels]
        for name, model in models.items()
    }
    return models, risks


def select_from_risks(risks: dict[str, list[float]], pass_max: float, earlier_fail_min: float) -> tuple[str, str]:
    maxima = {name: max(values) for name, values in risks.items()}
    passing = [name for name in CANDIDATES if maxima[name] <= pass_max]
    if not passing:
        return "abstain", "out_of_library"
    candidate = passing[0]
    index = CANDIDATES.index(candidate)
    if any(maxima[name] < earlier_fail_min for name in CANDIDATES[:index]):
        return "abstain", "equivalence_overlap"
    return candidate, "margin_separated"


def reference_truth(system: FunctionalSystem) -> dict[str, Any]:
    train = system.make_partition("reference_train", PARTITION_COUNTS["reference_train"])
    panel_a = system.make_partition("reference_a", PARTITION_COUNTS["reference_a"])
    panel_b = system.make_partition("reference_b", PARTITION_COUNTS["reference_b"])
    _models, risks = candidate_risks(system, train, [panel_a, panel_b])
    label, reason = select_from_risks(risks, REFERENCE_PASS_MAX, REFERENCE_EARLIER_FAIL_MIN)
    return {"predictive_class": label, "truth_reason": reason, "reference_risks": risks}


def make_system_rows(device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for split in REGISTRY_SPLITS:
        for family in SOURCE_FAMILIES:
            for task in range(TASKS):
                for replicate in range(REPLICATES):
                    system_id = opaque_system_id(split, family, task, replicate)
                    seed = seed_for(split, family, task, replicate)
                    system = FunctionalSystem(family, seed, task, device)
                    functional_truth = reference_truth(system)
                    public.append({
                        "system_id": system_id,
                        "registry_split": split,
                        "task_id": task,
                        "replicate": replicate,
                        "state_dimension": DIMENSION,
                        "partitions": PARTITION_COUNTS,
                    })
                    truth.append({
                        "system_id": system_id,
                        "source_family": family,
                        "seed": seed,
                        **functional_truth,
                    })
    return public, truth


def protocol_payload(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = {name: sum(row["predictive_class"] == name for row in truth) for name in (*CANDIDATES, "abstain")}
    reason_counts = {reason: sum(row["truth_reason"] == reason for row in truth) for reason in ("margin_separated", "equivalence_overlap", "out_of_library")}
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1263.c013.predictive_equivalence_independent_rescue.v1",
        "claim_type": "known_truth_predictive_functional_equivalence_calibration",
        "question": "Can a finite cross-fitted camera recover the least-complex margin-separated predictive class, abstain on functional ambiguity, and pass a non-tautological independent-donor rescue?",
        "systems": len(public),
        "cases_per_system": sum(PARTITION_COUNTS.values()),
        "registry_splits": list(REGISTRY_SPLITS),
        "source_families_hidden_until_adjudication": True,
        "predictive_class_truth": "least-complex candidate passing two independent 768-case reference panels at risk <= 0.050, with every earlier candidate at risk >= 0.120",
        "candidate_order": list(CANDIDATES) + ["abstain"],
        "candidate_nesting": ["additive", "linear", "quadratic", "rff", "abstain"],
        "class_counts": class_counts,
        "truth_reason_counts": reason_counts,
        "partitions": PARTITION_COUNTS,
        "thresholds": THRESHOLDS,
        "reference_thresholds": {"pass_max": REFERENCE_PASS_MAX, "earlier_fail_min": REFERENCE_EARLIER_FAIL_MIN},
        "rescue_provenance": {
            "block": "remove the oracle natural target channel from M_H11",
            "correct_rescue": "estimate the target effect with the same selected class fitted only on donor_discovery, then add independent donor nuisance",
            "wrong_rescue": "use wrong-identity A0 through the independent donor model",
            "algebraic_replay_sentinel": "add back the exact removed channel; expected to pass by identity and excluded from mediation evidence",
        },
        "gates": ["equivalence_class", "margin_separation", "typed_abstention", "cross_fit_confirmation", "independent_rescue", "wrong_identity", "negative_controls", "split_breadth"],
        "hard_stops": [
            "No source-family label is used by selection.",
            "No candidate, threshold, family, or risk margin changes after preregistration.",
            "Algebraic replay is a leakage sentinel and cannot support a mediation claim.",
            "Any conjunctive failure blocks free-Transformer external validity.",
            "A pass authorizes one separately frozen free-Transformer phase and never Qwen3 directly.",
        ],
        "forbidden_claims": ["generator recovery", "natural-language mechanism", "qwen3", "independent mediation in a free network", "new mathematics"],
        "source_hashes": {"main": file_sha256(Path(__file__).resolve()), "auditor": file_sha256(AUDITOR)},
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
        "precision": "CUDA float64 deterministic known-truth tensor algebra",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError(f"protocol already exists: {PROTOCOL}")
    if not torch.cuda.is_available():
        raise RuntimeError("preregistration material generation requires CUDA")
    public, truth = make_system_rows(torch.device("cuda"))
    write_jsonl(PUBLIC, public)
    write_jsonl(TRUTH, truth)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(public, truth))
    print(canonical_json({"status": "preregistered", "systems": len(public), "class_counts": read_json(PROTOCOL)["class_counts"]}))


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


def evaluate_confirmation(system: FunctionalSystem, model: dict[str, Any], donor_model: dict[str, Any], data: dict[str, torch.Tensor]) -> dict[str, float]:
    predicted = predict(model, data["A0"], data["B"])
    predicted_wrong = predict(model, data["wrong_A0"], data["B"])
    predicted_nuisance = predict(model, data["A0"] + data["nuisance"], data["B"])
    correct_distance = torch.linalg.vector_norm(predicted - data["A1"], dim=1)
    correct_to_wrong = torch.linalg.vector_norm(predicted - data["wrong_A1"], dim=1)
    wrong_distance = torch.linalg.vector_norm(predicted_wrong - data["wrong_A1"], dim=1)
    wrong_to_target = torch.linalg.vector_norm(predicted_wrong - data["A1"], dim=1)

    removed_effect = data["M_effect"]
    blocked = data["M_H11"] - removed_effect
    target_effect_scale = torch.linalg.vector_norm(data["Y_target"] - data["Y_base"]).clamp_min(1.0e-12)
    blocked_output = torch.tanh(blocked @ system.output_weight.T)
    donor_effect = predict(donor_model, data["A0"], data["B"]) @ system.advance.T + data["donor_noise"]
    rescued = blocked + donor_effect
    rescued_output = torch.tanh(rescued @ system.output_weight.T)
    wrong_effect = predict(donor_model, data["wrong_A0"], data["B"]) @ system.advance.T + data["donor_noise"]
    wrong_rescued = blocked + wrong_effect
    wrong_output = torch.tanh(wrong_rescued @ system.output_weight.T)
    replay = blocked + removed_effect
    replay_output = torch.tanh(replay @ system.output_weight.T)
    donor_gap = torch.linalg.vector_norm(donor_effect - removed_effect) / torch.linalg.vector_norm(removed_effect).clamp_min(1.0e-12)
    rescue_distance = torch.linalg.vector_norm(rescued - data["M_H11"], dim=1)
    rescue_to_wrong = torch.linalg.vector_norm(rescued - (data["M_H01"] + data["wrong_A1"] @ system.advance.T), dim=1)
    wrong_target_distance = torch.linalg.vector_norm(wrong_rescued - data["M_H11"], dim=1)
    wrong_identity_distance = torch.linalg.vector_norm(wrong_rescued - (data["M_H01"] + data["wrong_A1"] @ system.advance.T), dim=1)
    manifold_scale = torch.linalg.vector_norm(removed_effect, dim=1).clamp_min(1.0e-12)
    return {
        "a1_relative_error": relative(predicted, data["A1"], data["A1"]),
        "j_relative_error": relative(predicted - data["A0"], data["J"], data["A1"]),
        "correct_identity_accuracy": float((correct_distance < correct_to_wrong).double().mean().item()),
        "wrong_identity_accuracy": float((wrong_distance < wrong_to_target).double().mean().item()),
        "nuisance_sensitivity": relative(predicted_nuisance, predicted, data["A1"]),
        "block_remaining": float((torch.linalg.vector_norm(blocked_output - data["Y_base"]) / target_effect_scale).item()),
        "independent_rescue_error": float((torch.linalg.vector_norm(rescued_output - data["Y_target"]) / target_effect_scale).item()),
        "independent_rescue_identity": float((rescue_distance < rescue_to_wrong).double().mean().item()),
        "wrong_rescue_false_target": float((wrong_target_distance < wrong_identity_distance).double().mean().item()),
        "donor_source_gap": float(donor_gap.item()),
        "algebraic_replay_error": float((torch.linalg.vector_norm(replay_output - data["Y_target"]) / target_effect_scale).item()),
        "on_manifold_relative_distance_p95": float(torch.quantile(rescue_distance / manifold_scale, 0.95).item()),
    }


def confirmation_passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["a1_relative_error"] <= THRESHOLDS["confirmation_a1_error_max"]
        and metrics["j_relative_error"] <= THRESHOLDS["confirmation_j_error_max"]
        and metrics["correct_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["nuisance_sensitivity"] <= THRESHOLDS["nuisance_sensitivity_max"]
        and metrics["block_remaining"] <= THRESHOLDS["block_remaining_max"]
        and metrics["independent_rescue_error"] <= THRESHOLDS["independent_rescue_error_max"]
        and metrics["independent_rescue_identity"] >= THRESHOLDS["independent_rescue_identity_min"]
        and metrics["wrong_rescue_false_target"] <= THRESHOLDS["wrong_rescue_false_target_max"]
        and metrics["donor_source_gap"] >= THRESHOLDS["donor_source_gap_min"]
        and metrics["algebraic_replay_error"] <= THRESHOLDS["algebraic_replay_error_max"]
        and metrics["on_manifold_relative_distance_p95"] <= THRESHOLDS["manifold_p95_max"]
    )


def run_system(public: dict[str, Any], truth: dict[str, Any], device: torch.device) -> dict[str, Any]:
    system = FunctionalSystem(str(truth["source_family"]), int(truth["seed"]), int(public["task_id"]), device)
    discovery = system.make_partition("discovery", PARTITION_COUNTS["discovery"])
    selection = system.make_partition("selection", PARTITION_COUNTS["selection"])
    models, risks = candidate_risks(system, discovery, [selection])
    selected, selection_reason = select_from_risks(risks, THRESHOLDS["selection_pass_max"], THRESHOLDS["selection_earlier_fail_min"])
    confirmation = None
    passed = True
    controls: dict[str, float] = {}
    if selected != "abstain":
        donor_discovery = system.make_partition("donor_discovery", PARTITION_COUNTS["donor_discovery"])
        donor_model = fit_candidate(selected, donor_discovery)
        confirmation_data = system.make_partition("confirmation", PARTITION_COUNTS["confirmation"])
        confirmation = evaluate_confirmation(system, models[selected], donor_model, confirmation_data)
        passed = confirmation_passes(confirmation)
        mean_prediction = discovery["A1"].mean(dim=0, keepdim=True).expand_as(selection["A1"])
        permutation = torch.arange(discovery["A1"].shape[0] - 1, -1, -1, device=device)
        if selected == "additive":
            shuffled_error = relative(-selection["A0"], selection["A1"], selection["A1"])
        else:
            shuffled_data = {**discovery, "J": discovery["J"][permutation]}
            shuffled_model = fit_candidate(selected, shuffled_data)
            shuffled_error = relative(predict(shuffled_model, selection["A0"], selection["B"]), selection["A1"], selection["A1"])
        controls = {
            "mean_response_error": relative(mean_prediction, selection["A1"], selection["A1"]),
            "sign_flip_error": relative(-selection["A0"], selection["A1"], selection["A1"]),
            "label_permutation_error": shuffled_error,
        }
    return {
        "system_id": public["system_id"],
        "registry_split": public["registry_split"],
        "task_id": public["task_id"],
        "replicate": public["replicate"],
        "selected_class": selected,
        "selection_reason": selection_reason,
        "selection_risks": risks,
        "confirmation": confirmation,
        "confirmation_passed": passed,
        "controls": controls,
    }


def run(device_name: str) -> None:
    protocol, public, truth = verify_protocol()
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
        if index % 20 == 0:
            print(canonical_json({"completed": index, "total": len(public)}), flush=True)
    elapsed = time.perf_counter() - started
    write_jsonl(RAW, rows)
    atomic_json(SUMMARY, {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "systems": len(rows),
        "cases_per_system": sum(PARTITION_COUNTS.values()),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "raw_digest": digest(rows),
        "protocol_hash": file_sha256(PROTOCOL),
    })
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "raw_hash": file_sha256(RAW), "summary_hash": file_sha256(SUMMARY)})
    print(canonical_json({"status": "formal_run_complete", "systems": len(rows), "elapsed_seconds": elapsed, "class_counts": protocol["class_counts"]}))


def analyze() -> None:
    protocol, _public, truth = verify_protocol()
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    rows = read_jsonl(RAW)
    if digest(rows) != read_json(SUMMARY)["raw_digest"]:
        raise RuntimeError("raw digest mismatch")
    truth_by_id = {row["system_id"]: row for row in truth}
    type_accuracy = sum(row["selected_class"] == truth_by_id[row["system_id"]]["predictive_class"] for row in rows) / len(rows)
    abstain_rows = [row for row in rows if truth_by_id[row["system_id"]]["predictive_class"] == "abstain"]
    abstention_accuracy = sum(row["selected_class"] == "abstain" for row in abstain_rows) / len(abstain_rows)
    actionable = [row for row in rows if row["selected_class"] != "abstain"]
    class_counts = {name: sum(truth_by_id[row["system_id"]]["predictive_class"] == name for row in rows) for name in (*CANDIDATES, "abstain")}
    selected_counts = {name: sum(row["selected_class"] == name for row in rows) for name in (*CANDIDATES, "abstain")}
    by_split = {
        split: all(row["selected_class"] == truth_by_id[row["system_id"]]["predictive_class"] and row["confirmation_passed"] for row in rows if row["registry_split"] == split)
        for split in REGISTRY_SPLITS
    }
    control_floor = min((min(row["controls"].values()) for row in actionable), default=0.0)
    confirmation_metrics = [row["confirmation"] for row in actionable if row["confirmation"] is not None]
    actionable_classes = [name for name in CANDIDATES if class_counts[name] > 0]
    class_breadth = all(class_counts[name] >= THRESHOLDS["minimum_class_count"] for name in actionable_classes) and class_counts["abstain"] >= THRESHOLDS["minimum_class_count"]
    cross_source_alias = any(
        len({truth_by_id[row["system_id"]]["source_family"] for row in rows if truth_by_id[row["system_id"]]["predictive_class"] == name}) >= 2
        for name in CANDIDATES
    )
    gates = {
        "G-EQUIVALENCE-CLASS": type_accuracy >= THRESHOLDS["type_accuracy_min"],
        "G-ABSTENTION": bool(abstain_rows) and abstention_accuracy >= THRESHOLDS["abstention_accuracy_min"],
        "G-CLASS-BREADTH": class_breadth,
        "G-CROSS-SOURCE-ALIAS": cross_source_alias,
        "G-CONFIRMATION": bool(actionable) and all(row["confirmation_passed"] for row in actionable),
        "G-INDEPENDENT-RESCUE": bool(confirmation_metrics) and all(metric["independent_rescue_error"] <= THRESHOLDS["independent_rescue_error_max"] and metric["donor_source_gap"] >= THRESHOLDS["donor_source_gap_min"] for metric in confirmation_metrics),
        "G-ALGEBRAIC-SENTINEL": bool(confirmation_metrics) and all(metric["algebraic_replay_error"] <= THRESHOLDS["algebraic_replay_error_max"] for metric in confirmation_metrics),
        "G-WRONG-IDENTITY": bool(confirmation_metrics) and all(metric["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"] and metric["wrong_rescue_false_target"] <= THRESHOLDS["wrong_rescue_false_target_max"] for metric in confirmation_metrics),
        "G-CONTROLS": control_floor >= THRESHOLDS["control_error_floor_min"],
        "G-SPLIT-BREADTH": all(by_split.values()),
    }
    passed = all(gates.values())
    worst = None if not confirmation_metrics else {
        "a1_relative_error_max": max(item["a1_relative_error"] for item in confirmation_metrics),
        "j_relative_error_max": max(item["j_relative_error"] for item in confirmation_metrics),
        "independent_rescue_error_max": max(item["independent_rescue_error"] for item in confirmation_metrics),
        "donor_source_gap_min": min(item["donor_source_gap"] for item in confirmation_metrics),
        "wrong_rescue_false_target_max": max(item["wrong_rescue_false_target"] for item in confirmation_metrics),
        "manifold_p95_max": max(item["on_manifold_relative_distance_p95"] for item in confirmation_metrics),
        "algebraic_replay_error_max": max(item["algebraic_replay_error"] for item in confirmation_metrics),
    }
    adjudication = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": utc_now(),
        "systems": len(rows),
        "cases_per_system": sum(PARTITION_COUNTS.values()),
        "type_accuracy": type_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "class_counts": class_counts,
        "selected_counts": selected_counts,
        "split_pass": by_split,
        "control_error_floor": control_floor,
        "worst_confirmation": worst,
        "gates": gates,
        "passed": passed,
        "verdict": "predictive_equivalence_independent_rescue_calibrated" if passed else "predictive_equivalence_independent_rescue_not_calibrated",
        "claim_boundary": "Known-truth predictive functional equivalence and independently sourced donor rescue only; the algebraic replay sentinel carries zero mediation evidence; no free-network, pretrained-model, or language claim.",
        "authorization": {"free_transformer_predictive_equivalence": passed, "qwen3": False, "language_mechanism": False, "new_mathematics": False},
    }
    atomic_json(ANALYSIS, adjudication)
    final = {
        **adjudication,
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "environment": file_sha256(ENVIRONMENT),
            "public": file_sha256(PUBLIC),
            "truth": file_sha256(TRUTH),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "summary": file_sha256(SUMMARY),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": final["verdict"], "passed": passed, "type_accuracy": type_accuracy, "abstention_accuracy": abstention_accuracy, "gates": gates}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def probe() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA")
    device = torch.device("cuda")
    rows = []
    for family in SOURCE_FAMILIES:
        for task in range(3):
            for replicate in range(3):
                seed = seed_for("probe", family, task, replicate)
                system = FunctionalSystem(family, seed, task, device)
                truth = {"source_family": family, "seed": seed, **reference_truth(system)}
                public = {"system_id": f"probe-{family}-{task}-{replicate}", "registry_split": "probe", "task_id": task, "replicate": replicate}
                result = run_system(public, truth, device)
                rows.append({**result, "source_family": family, "truth_class": truth["predictive_class"], "truth_reason": truth["truth_reason"], "reference_risks": truth["reference_risks"]})
    atomic_json(PROBE, rows)
    print(canonical_json({
        "systems": len(rows),
        "truth_counts": {name: sum(row["truth_class"] == name for row in rows) for name in (*CANDIDATES, "abstain")},
        "selected_counts": {name: sum(row["selected_class"] == name for row in rows) for name in (*CANDIDATES, "abstain")},
        "mismatches": [row["system_id"] for row in rows if row["selected_class"] != row["truth_class"]],
        "confirmation_failures": [row["system_id"] for row in rows if not row["confirmation_passed"]],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze", "all"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "probe":
        probe()
    elif args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run(args.device)
    elif args.command == "analyze":
        analyze()
    else:
        preregister(args.force)
        run_auditor("pre")
        run(args.device)
        analyze()
        run_auditor("final")


if __name__ == "__main__":
    main()
