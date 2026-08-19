"""Phase1261: known-truth calibration of factorial response compilers.

The experiment replaces the C011 target-preserve/null-zero object with a
context-preserving target editor.  For each held-out factorial bundle it sees
the target effect at context 0 (A0) and the legitimate context effect (B), and
must predict the target effect at context 1 (A1 = A0 + J).  The edited state
H01 + A1_hat must reproduce H11 while preserving B and the interaction J.

Five frozen candidates compete in increasing complexity, followed by typed
abstention.  Hidden mechanism labels are used only during adjudication.
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
OUT = ROOT / "tests/glm5/result/phase1261_c012_factorial_response_compiler_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC = OUT / "material/public_systems.jsonl"
TRUTH = OUT / "material/private_mechanism_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/system_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1261_c012_factorial_response_compiler_calibration_audit.py"
PROBE = ROOT / "tests/glm5_temp/phase1261_c012_factorial_response_compiler_probe.json"

PHASE = 1261
CAMPAIGN = "C012"
CONTRACT_ID = "EXP-C012-WP01-001"
DIMENSION = 12
LATENT_RANK = 2
OUTPUT_DIMENSION = 6
CONTROLS = 4
TASKS = 4
REPLICATES = 6
REGISTRY_SPLITS = ("calibration", "family_holdout")
FAMILIES = (
    "additive_orthogonal",
    "additive_oblique",
    "near_collision_additive",
    "linear_factorial",
    "bilinear_factorial",
    "smooth_local",
    "hidden_collision",
)
EXPECTED_COMPILER = {
    "additive_orthogonal": "legacy_zero_null",
    "additive_oblique": "legacy_zero_null",
    "near_collision_additive": "additive_preserving",
    "linear_factorial": "linear_factorial",
    "bilinear_factorial": "quadratic_factorial",
    "smooth_local": "local_kernel",
    "hidden_collision": "abstain",
}
COMPILER_ORDER = (
    "legacy_zero_null",
    "additive_preserving",
    "linear_factorial",
    "quadratic_factorial",
    "local_kernel",
)
PARTITION_COUNTS = {"discovery": 384, "selection": 192, "confirmation": 384}
RIDGE = 1.0e-6
KERNEL_MIN_BANDWIDTH = 0.25
KERNEL_MEDIAN_FACTOR = 0.5
THRESHOLDS = {
    "selection_a1_relative_error_max": 0.10,
    "selection_j_error_fraction_max": 0.10,
    "selection_context_error_fraction_max": 0.10,
    "selection_output_relative_error_max": 0.10,
    "selection_gain_max": 12.0,
    "confirmation_a1_relative_error_max": 0.12,
    "confirmation_j_error_fraction_max": 0.12,
    "confirmation_context_error_fraction_max": 0.12,
    "confirmation_output_relative_error_max": 0.12,
    "wrong_identity_accuracy_min": 0.98,
    "wrong_false_target_rate_max": 0.02,
    "nuisance_sensitivity_max": 0.06,
    "path_handoff_relative_error_max": 0.12,
    "path_block_remaining_max": 0.12,
    "path_rescue_relative_error_max": 0.12,
    "on_manifold_relative_distance_p95_max": 0.20,
    "compiler_type_accuracy_min": 1.0,
    "abstention_accuracy_min": 1.0,
    "control_error_floor_min": 0.25,
    "effective_rank_expected": LATENT_RANK,
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
    payload = f"1261|{split}|{family}|{task}|{replicate}".encode("utf-8")
    return "F" + hashlib.sha256(payload).hexdigest()[:15]


def seed_for(split: str, family: str, task: int, replicate: int) -> int:
    payload = f"seed|1261|{split}|{family}|{task}|{replicate}".encode("utf-8")
    return 1_261_000_000 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 500_000_000


def make_system_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for split in REGISTRY_SPLITS:
        for family in FAMILIES:
            for task in range(TASKS):
                for replicate in range(REPLICATES):
                    system_id = opaque_system_id(split, family, task, replicate)
                    seed = seed_for(split, family, task, replicate)
                    public.append({
                        "system_id": system_id,
                        "registry_split": split,
                        "task_id": task,
                        "replicate": replicate,
                        "state_dimension": DIMENSION,
                        "latent_rank": LATENT_RANK,
                        "controls": CONTROLS,
                        "partitions": PARTITION_COUNTS,
                    })
                    truth.append({
                        "system_id": system_id,
                        "family": family,
                        "expected_compiler": EXPECTED_COMPILER[family],
                        "seed": seed,
                    })
    return public, truth


def protocol_payload(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1261.c012.factorial_response_compiler.protocol.v1",
        "claim_type": "known_truth_factorial_response_compiler_calibration",
        "question": "Can a frozen finite compiler competition predict A, preserve B, predict J, reject nuisance and wrong identity, close path mediation, and abstain when J is not identifiable?",
        "systems": len(public),
        "systems_per_family": len(public) // len(FAMILIES),
        "registry_splits": list(REGISTRY_SPLITS),
        "mechanism_families_hidden_until_adjudication": True,
        "compiler_order": list(COMPILER_ORDER) + ["abstain"],
        "compiler_definitions": {
            "legacy_zero_null": "oblique projector retaining discovery A0 span and nulling discovery B span",
            "additive_preserving": "project donor A0 onto its discovery span and add it to the contextual receiver",
            "linear_factorial": "A0 skip connection plus a ridge map from discovery A/B coordinates to J",
            "quadratic_factorial": "A0 skip connection plus a ridge map with all A-coordinate by B-coordinate products to J",
            "local_kernel": "A0 skip connection plus a Gaussian kernel ridge map to J with a frozen median-distance bandwidth rule",
            "abstain": "selected when no earlier candidate passes all selection gates",
        },
        "factorial_contract": {
            "A0": "H10-H00",
            "B": "H01-H00",
            "J": "H11-H10-H01+H00",
            "A1": "H11-H01=A0+J",
            "edit": "H01 plus predicted A1 must reproduce H11",
            "path": "the selected compiler type is fitted independently at each stage; handoff compares predicted effects after the known stage map",
        },
        "partitions": PARTITION_COUNTS,
        "thresholds": THRESHOLDS,
        "ridge": RIDGE,
        "kernel_min_bandwidth": KERNEL_MIN_BANDWIDTH,
        "kernel_median_factor": KERNEL_MEDIAN_FACTOR,
        "gates": [
            "compiler_type_recovery",
            "typed_abstention",
            "A1_prediction",
            "B_preservation",
            "J_prediction",
            "wrong_identity_rejection",
            "nuisance_null_rejection",
            "path_block_and_rescue",
            "on_manifold",
            "family_holdout_breadth",
            "negative_control_separation",
        ],
        "controls": [
            "mean_response",
            "sign_flipped_target",
            "label_permutation",
            "nuisance_only_donor",
            "wrong_identity_donor",
            "oracle_positive_sentinel",
        ],
        "hard_stops": [
            "Candidate order, feature widths, ridge, thresholds and families cannot change after preregistration.",
            "Selection sees no confirmation row and no private family label.",
            "Hidden-collision systems remain in the denominator and must abstain.",
            "A failure denies free-network external validity; no pretrained model is authorized by this phase.",
            "A pass authorizes one separately frozen free-Transformer factorial compiler phase only.",
        ],
        "forbidden_claims": [
            "natural-language mechanism",
            "free-network external validity",
            "qwen3",
            "unique physical algorithm",
            "global Euclidean semantic space",
            "new mathematics",
        ],
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
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
        "precision": "CUDA float64 deterministic known-truth tensor algebra",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError(f"protocol already exists: {PROTOCOL}")
    public, truth = make_system_rows()
    write_jsonl(PUBLIC, public)
    write_jsonl(TRUTH, truth)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(public, truth))
    print(canonical_json({"status": "preregistered", "systems": len(public), "cases_per_system": sum(PARTITION_COUNTS.values())}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    public = read_jsonl(PUBLIC)
    truth = read_jsonl(TRUTH)
    expected = protocol_payload(public, truth)
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol or source drift")
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source hash drift")
    if protocol["public_digest"] != digest(public) or protocol["truth_digest"] != digest(truth):
        raise RuntimeError("material digest drift")
    if protocol["thresholds"] != THRESHOLDS:
        raise RuntimeError("threshold drift")
    return protocol, public, truth


def orthogonal(seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(DIMENSION, DIMENSION))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return torch.tensor(q * signs, dtype=torch.float64, device=device)


def effective_basis(samples: torch.Tensor) -> tuple[torch.Tensor, list[float], int]:
    _u, singular, vh = torch.linalg.svd(samples.double(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 1.0e-12:
        return torch.zeros((samples.shape[1], 0), dtype=torch.float64, device=samples.device), [], 0
    rank = int(torch.sum(singular > singular[0] * 1.0e-6).item())
    return vh[:rank].T.contiguous(), [float(v) for v in singular.tolist()], rank


def ridge_fit(features: torch.Tensor, target: torch.Tensor, ridge: float = RIDGE) -> torch.Tensor:
    x = features.double()
    y = target.double()
    gram = x.T @ x
    scale = float(torch.trace(gram).item()) / max(1, gram.shape[0])
    penalty = ridge * max(scale, 1.0)
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=x.device)
    return torch.linalg.solve(gram + penalty * eye, x.T @ y)


def standardize_fit(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return (values - mean) / scale, mean, scale


def standardize_apply(values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (values - mean) / scale


@dataclass
class SystemTruth:
    family: str
    seed: int
    task_id: int
    device: torch.device

    def __post_init__(self) -> None:
        self.q = orthogonal(self.seed + 101, self.device)
        self.advance = orthogonal(self.seed + 211, self.device)
        self.a_basis = self.q[:, :LATENT_RANK]
        angle = {
            "additive_orthogonal": math.pi / 2,
            "additive_oblique": math.pi / 4,
            "near_collision_additive": math.radians(1.5),
        }.get(self.family, math.pi / 3)
        self.b_basis = math.cos(angle) * self.q[:, :LATENT_RANK] + math.sin(angle) * self.q[:, LATENT_RANK:2 * LATENT_RANK]
        self.j_basis = self.q[:, 2 * LATENT_RANK:3 * LATENT_RANK]
        self.nuisance_basis = self.q[:, 3 * LATENT_RANK:4 * LATENT_RANK]
        rng = np.random.default_rng(self.seed + 307 * self.task_id)
        self.linear_a = torch.tensor(rng.normal(scale=0.28, size=(LATENT_RANK, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.linear_b = torch.tensor(rng.normal(scale=0.32, size=(LATENT_RANK, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.bilinear = torch.tensor(rng.normal(scale=0.40, size=(LATENT_RANK * LATENT_RANK, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.smooth_centers = torch.tensor(rng.normal(size=(12, 2 * LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.smooth_width = 1.30
        self.smooth_weight = torch.tensor(rng.normal(scale=0.32, size=(12, LATENT_RANK)), dtype=torch.float64, device=self.device)
        self.collision_vector = torch.tensor(rng.normal(size=(LATENT_RANK,)), dtype=torch.float64, device=self.device)
        self.collision_vector = self.collision_vector / torch.linalg.vector_norm(self.collision_vector).clamp_min(1.0e-12)
        out = torch.tensor(rng.normal(size=(OUTPUT_DIMENSION, DIMENSION)), dtype=torch.float64, device=self.device)
        self.output_weight = out / torch.linalg.vector_norm(out, dim=1, keepdim=True).clamp_min(1.0e-12)
        self.base = torch.tensor(rng.normal(scale=0.15, size=(DIMENSION,)), dtype=torch.float64, device=self.device)

    def interaction(self, a: torch.Tensor, b: torch.Tensor, hidden_sign: torch.Tensor) -> torch.Tensor:
        if self.family in {"additive_orthogonal", "additive_oblique", "near_collision_additive"}:
            coords = torch.zeros_like(a)
        elif self.family == "linear_factorial":
            coords = a @ self.linear_a + b @ self.linear_b
        elif self.family == "bilinear_factorial":
            outer = torch.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)
            coords = outer @ self.bilinear
        elif self.family == "smooth_local":
            inputs = torch.cat((a, b), dim=1)
            squared = torch.cdist(inputs, self.smooth_centers).square()
            features = torch.exp(-0.5 * squared / (self.smooth_width * self.smooth_width))
            coords = features @ self.smooth_weight
        elif self.family == "hidden_collision":
            coords = 0.65 * hidden_sign[:, None] * self.collision_vector[None, :]
        else:
            raise ValueError(self.family)
        return coords @ self.j_basis.T

    def make_partition(self, partition: str, count: int) -> dict[str, torch.Tensor]:
        partition_id = {"discovery": 11, "selection": 23, "confirmation": 37}[partition]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 10_000 * partition_id)
        if self.family == "hidden_collision":
            if count % 2:
                raise ValueError("collision counts must be even")
            half = count // 2
            a_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            b_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            aw_half = torch.randn((half, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            a = torch.cat((a_half, a_half), dim=0)
            b = torch.cat((b_half, b_half), dim=0)
            aw = torch.cat((aw_half, aw_half), dim=0)
            hidden_sign = torch.cat((torch.ones(half, device=self.device), -torch.ones(half, device=self.device))).double()
        else:
            a = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            b = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            aw = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device)
            hidden_sign = torch.where(torch.arange(count, device=self.device) % 2 == 0, 1.0, -1.0).double()
        control = torch.arange(count, device=self.device).remainder(CONTROLS)
        base_noise = torch.randn((count, DIMENSION), generator=generator, dtype=torch.float64, device=self.device) * 0.02
        nuisance_coords = torch.randn((count, LATENT_RANK), generator=generator, dtype=torch.float64, device=self.device) * 0.08
        nuisance = nuisance_coords @ self.nuisance_basis.T
        A0 = a @ self.a_basis.T
        B = b @ self.b_basis.T
        wrong_A0 = aw @ self.a_basis.T
        J = self.interaction(a, b, hidden_sign)
        wrong_J = self.interaction(aw, b, -hidden_sign)
        A1 = A0 + J
        wrong_A1 = wrong_A0 + wrong_J
        H00 = self.base[None, :] + base_noise
        H10 = H00 + A0
        H01 = H00 + B
        H11 = H01 + A1
        Hwrong10 = H00 + wrong_A0
        Hwrong11 = H01 + wrong_A1
        H01_nuisance = H01 + nuisance
        H11_nuisance = H11 + nuisance
        values = {
            "A0": A0,
            "B": B,
            "J": J,
            "A1": A1,
            "wrong_A0": wrong_A0,
            "wrong_J": wrong_J,
            "wrong_A1": wrong_A1,
            "nuisance": nuisance,
            "H00": H00,
            "H10": H10,
            "H01": H01,
            "H11": H11,
            "Hwrong10": Hwrong10,
            "Hwrong11": Hwrong11,
            "H01_nuisance": H01_nuisance,
            "H11_nuisance": H11_nuisance,
            "control": control,
        }
        for key in ("H00", "H10", "H01", "H11", "Hwrong10", "Hwrong11", "H01_nuisance", "H11_nuisance"):
            values[f"M_{key}"] = values[key] @ self.advance.T
            values[f"Y_{key}"] = torch.tanh(values[f"M_{key}"] @ self.output_weight.T)
        values["M_A0"] = values["M_H10"] - values["M_H00"]
        values["M_B"] = values["M_H01"] - values["M_H00"]
        values["M_J"] = values["M_H11"] - values["M_H10"] - values["M_H01"] + values["M_H00"]
        values["M_A1"] = values["M_H11"] - values["M_H01"]
        values["M_wrong_A0"] = values["M_Hwrong10"] - values["M_H00"]
        values["M_wrong_A1"] = values["M_Hwrong11"] - values["M_H01"]
        return values


def fit_legacy(A: torch.Tensor, B: torch.Tensor) -> dict[str, Any]:
    a_basis, a_singular, a_rank = effective_basis(A)
    b_basis, b_singular, b_rank = effective_basis(B)
    combined = torch.cat((a_basis, b_basis), dim=1)
    selector = torch.cat((
        torch.eye(a_rank, dtype=torch.float64, device=A.device),
        torch.zeros((a_rank, b_rank), dtype=torch.float64, device=A.device),
    ), dim=1)
    operator = a_basis @ selector @ torch.linalg.pinv(combined, rtol=1.0e-8, atol=1.0e-10)
    cross = a_basis.T @ b_basis
    principal = torch.linalg.svdvals(cross)
    min_angle = math.degrees(math.acos(float(principal.max().clamp(-1.0, 1.0).item()))) if principal.numel() else 90.0
    return {
        "name": "legacy_zero_null",
        "operator": operator,
        "a_basis": a_basis,
        "a_singular": a_singular,
        "b_singular": b_singular,
        "a_rank": a_rank,
        "b_rank": b_rank,
        "combined_rank": int(torch.linalg.matrix_rank(combined, rtol=1.0e-8, atol=1.0e-10).item()),
        "minimum_principal_angle_deg": min_angle,
        "gain": float(torch.linalg.matrix_norm(operator, ord=2).item()),
    }


def fit_additive(A: torch.Tensor) -> dict[str, Any]:
    a_basis, singular, rank = effective_basis(A)
    return {"name": "additive_preserving", "a_basis": a_basis, "a_singular": singular, "a_rank": rank, "gain": 1.0}


def coordinates(A: torch.Tensor, B: torch.Tensor, model: dict[str, Any]) -> torch.Tensor:
    return torch.cat((A @ model["a_basis"], B @ model["b_basis"]), dim=1)


def linear_features(coords: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.ones((coords.shape[0], 1), dtype=torch.float64, device=coords.device), coords), dim=1)


def quadratic_features(coords: torch.Tensor, a_rank: int, b_rank: int) -> torch.Tensor:
    a = coords[:, :a_rank]
    b = coords[:, a_rank:a_rank + b_rank]
    outer = torch.einsum("bi,bj->bij", a, b).reshape(coords.shape[0], -1)
    return torch.cat((linear_features(coords), outer), dim=1)


def fit_regression(name: str, A: torch.Tensor, B: torch.Tensor, A1: torch.Tensor, seed: int) -> dict[str, Any]:
    a_basis, a_singular, a_rank = effective_basis(A)
    b_basis, b_singular, b_rank = effective_basis(B)
    base = {"name": name, "a_basis": a_basis, "b_basis": b_basis, "a_singular": a_singular, "b_singular": b_singular, "a_rank": a_rank, "b_rank": b_rank}
    coords = coordinates(A, B, base)
    standardized, mean, scale = standardize_fit(coords)
    base["coord_mean"], base["coord_scale"] = mean, scale
    additive = (A @ a_basis) @ a_basis.T
    interaction_target = A1 - additive
    if name == "linear_factorial":
        features = linear_features(standardized)
        base["weights"] = ridge_fit(features, interaction_target)
    elif name == "quadratic_factorial":
        features = quadratic_features(standardized, a_rank, b_rank)
        base["weights"] = ridge_fit(features, interaction_target)
    elif name == "local_kernel":
        squared = torch.cdist(standardized, standardized).square()
        nonzero = torch.sqrt(squared[squared > 1.0e-12])
        bandwidth = float((KERNEL_MEDIAN_FACTOR * torch.median(nonzero)).clamp_min(KERNEL_MIN_BANDWIDTH).item())
        kernel = torch.exp(-0.5 * squared / (bandwidth * bandwidth))
        penalty = RIDGE * max(float(torch.trace(kernel).item()) / max(1, kernel.shape[0]), 1.0)
        eye = torch.eye(kernel.shape[0], dtype=torch.float64, device=A.device)
        base.update({
            "train_coords": standardized,
            "bandwidth": bandwidth,
            "weights": torch.linalg.solve(kernel + penalty * eye, interaction_target),
        })
    else:
        raise ValueError(name)
    base["gain"] = float(torch.linalg.matrix_norm(base["weights"], ord=2).item())
    return base


def predict(model: dict[str, Any], A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    name = model["name"]
    if name == "legacy_zero_null":
        return A @ model["operator"].T
    if name == "additive_preserving":
        return (A @ model["a_basis"]) @ model["a_basis"].T
    coords = coordinates(A, B, model)
    standardized = standardize_apply(coords, model["coord_mean"], model["coord_scale"])
    if name == "linear_factorial":
        features = linear_features(standardized)
    elif name == "quadratic_factorial":
        features = quadratic_features(standardized, model["a_rank"], model["b_rank"])
    elif name == "local_kernel":
        squared = torch.cdist(standardized, model["train_coords"]).square()
        features = torch.exp(-0.5 * squared / (model["bandwidth"] * model["bandwidth"]))
    else:
        raise ValueError(name)
    additive = (A @ model["a_basis"]) @ model["a_basis"].T
    return additive + features @ model["weights"]


def relative(predicted: torch.Tensor, target: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(predicted - target) / torch.linalg.vector_norm(denominator).clamp_min(1.0e-12)).item())


def output_of(system: SystemTruth, state: torch.Tensor) -> torch.Tensor:
    mediator = state @ system.advance.T
    return torch.tanh(mediator @ system.output_weight.T)


def candidate_metrics(system: SystemTruth, model: dict[str, Any], data: dict[str, torch.Tensor]) -> dict[str, float]:
    predicted_A1 = predict(model, data["A0"], data["B"])
    predicted_H11 = data["H01"] + predicted_A1
    predicted_J = predicted_A1 - data["A0"]
    true_context_at_target = data["H11"] - data["H10"]
    predicted_context_at_target = predicted_H11 - data["H10"]
    predicted_output = output_of(system, predicted_H11)
    true_output = data["Y_H11"]
    output_reference = true_output - data["Y_H01"]
    if model["name"] == "legacy_zero_null":
        gain = float(model["gain"])
    elif model["name"] == "additive_preserving":
        gain = 1.0
    else:
        epsilon = 1.0e-4
        local_gains = []
        probe_count = min(32, data["A0"].shape[0])
        probe_a = data["A0"][:probe_count]
        probe_b = data["B"][:probe_count]
        probe_base = predict(model, probe_a, probe_b)
        for source in ("A", "B"):
            for coordinate in range(data["A0"].shape[1]):
                delta = torch.zeros_like(probe_a)
                delta[:, coordinate] = epsilon
                changed = predict(model, probe_a + delta, probe_b) if source == "A" else predict(model, probe_a, probe_b + delta)
                local_gains.append(float((torch.linalg.vector_norm(changed - probe_base) / torch.linalg.vector_norm(delta)).item()))
        gain = max(local_gains)
    return {
        "a1_relative_error": relative(predicted_A1, data["A1"], data["A1"]),
        "j_error_fraction": relative(predicted_J, data["J"], data["A1"]),
        "context_error_fraction": relative(predicted_context_at_target, true_context_at_target, data["A1"]),
        "edited_state_relative_error": relative(predicted_H11, data["H11"], data["A1"]),
        "output_relative_error": relative(predicted_output, true_output, output_reference),
        "gain": gain,
    }


def selection_passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["a1_relative_error"] <= THRESHOLDS["selection_a1_relative_error_max"]
        and metrics["j_error_fraction"] <= THRESHOLDS["selection_j_error_fraction_max"]
        and metrics["context_error_fraction"] <= THRESHOLDS["selection_context_error_fraction_max"]
        and metrics["output_relative_error"] <= THRESHOLDS["selection_output_relative_error_max"]
        and metrics["gain"] <= THRESHOLDS["selection_gain_max"]
    )


def serializable_model_summary(model: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "a_rank", "b_rank", "combined_rank", "minimum_principal_angle_deg", "bandwidth", "gain")
    return {key: model[key] for key in keys if key in model}


def fit_candidates(system: SystemTruth, discovery: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        fit_legacy(discovery["A0"], discovery["B"]),
        fit_additive(discovery["A0"]),
        fit_regression("linear_factorial", discovery["A0"], discovery["B"], discovery["A1"], system.seed),
        fit_regression("quadratic_factorial", discovery["A0"], discovery["B"], discovery["A1"], system.seed),
        fit_regression("local_kernel", discovery["A0"], discovery["B"], discovery["A1"], system.seed),
    ]


def fit_candidate_by_name(system: SystemTruth, discovery: dict[str, torch.Tensor], name: str, prefix: str = "") -> dict[str, Any]:
    a = discovery[f"{prefix}A0"]
    b = discovery[f"{prefix}B"]
    a1 = discovery[f"{prefix}A1"]
    if name == "legacy_zero_null":
        return fit_legacy(a, b)
    if name == "additive_preserving":
        return fit_additive(a)
    return fit_regression(name, a, b, a1, system.seed + (17_000 if prefix else 0))


def confirmation_metrics(system: SystemTruth, model: dict[str, Any], mediator_model: dict[str, Any], data: dict[str, torch.Tensor]) -> dict[str, Any]:
    base = candidate_metrics(system, model, data)
    predicted_A1 = predict(model, data["A0"], data["B"])
    predicted_H11 = data["H01"] + predicted_A1
    predicted_wrong_A1 = predict(model, data["wrong_A0"], data["B"])
    predicted_wrong_H11 = data["H01"] + predicted_wrong_A1
    correct_distance = torch.linalg.vector_norm(predicted_H11 - data["H11"], dim=1)
    correct_to_wrong = torch.linalg.vector_norm(predicted_H11 - data["Hwrong11"], dim=1)
    wrong_distance = torch.linalg.vector_norm(predicted_wrong_H11 - data["Hwrong11"], dim=1)
    wrong_to_target = torch.linalg.vector_norm(predicted_wrong_H11 - data["H11"], dim=1)
    nuisance_prediction = predict(model, data["A0"] + data["nuisance"], data["B"])
    nuisance_sensitivity = float((torch.linalg.vector_norm(nuisance_prediction - predicted_A1) / torch.linalg.vector_norm(data["A1"]).clamp_min(1.0e-12)).item())

    propagated = predicted_H11 @ system.advance.T
    mediator_prediction = predict(mediator_model, data["M_A0"], data["M_B"])
    direct_mediator = data["M_H01"] + mediator_prediction
    handoff_error = relative(propagated, data["M_H11"], data["M_A1"])
    blocked = propagated - mediator_prediction
    blocked_output = torch.tanh(blocked @ system.output_weight.T)
    base_output = data["Y_H01"]
    target_output = data["Y_H11"]
    target_output_effect = target_output - base_output
    block_remaining = float((torch.linalg.vector_norm(blocked_output - base_output) / torch.linalg.vector_norm(target_output_effect).clamp_min(1.0e-12)).item())
    rescued = blocked + mediator_prediction
    rescue_output = torch.tanh(rescued @ system.output_weight.T)
    rescue_error = relative(rescue_output, target_output, target_output_effect)
    wrong_mediator = predict(mediator_model, data["M_wrong_A0"], data["M_B"])
    wrong_rescued = blocked + wrong_mediator
    wrong_rescue_output = torch.tanh(wrong_rescued @ system.output_weight.T)
    true_wrong_output = data["Y_Hwrong11"]
    wrong_rescue_distance = torch.linalg.vector_norm(wrong_rescue_output - true_wrong_output, dim=1)
    wrong_rescue_target_distance = torch.linalg.vector_norm(wrong_rescue_output - target_output, dim=1)
    manifold_scale = torch.linalg.vector_norm(data["A1"], dim=1).mean().clamp_min(1.0e-12)
    normalized_manifold_distance = correct_distance / manifold_scale
    return {
        **base,
        "wrong_identity_accuracy": float((wrong_distance < wrong_to_target).double().mean().item()),
        "wrong_false_target_rate": float((wrong_to_target < wrong_distance).double().mean().item()),
        "correct_identity_accuracy": float((correct_distance < correct_to_wrong).double().mean().item()),
        "nuisance_sensitivity": nuisance_sensitivity,
        "path_handoff_relative_error": handoff_error,
        "path_direct_mediator_relative_error": relative(direct_mediator, data["M_H11"], data["M_A1"]),
        "path_block_remaining": block_remaining,
        "path_rescue_relative_error": rescue_error,
        "wrong_rescue_identity_accuracy": float((wrong_rescue_distance < wrong_rescue_target_distance).double().mean().item()),
        "on_manifold_relative_distance_p95": float(torch.quantile(normalized_manifold_distance, 0.95).item()),
        "on_manifold_relative_distance_max_diagnostic": float(normalized_manifold_distance.max().item()),
    }


def confirmation_passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["a1_relative_error"] <= THRESHOLDS["confirmation_a1_relative_error_max"]
        and metrics["j_error_fraction"] <= THRESHOLDS["confirmation_j_error_fraction_max"]
        and metrics["context_error_fraction"] <= THRESHOLDS["confirmation_context_error_fraction_max"]
        and metrics["output_relative_error"] <= THRESHOLDS["confirmation_output_relative_error_max"]
        and metrics["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["wrong_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"]
        and metrics["correct_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["nuisance_sensitivity"] <= THRESHOLDS["nuisance_sensitivity_max"]
        and metrics["path_handoff_relative_error"] <= THRESHOLDS["path_handoff_relative_error_max"]
        and metrics["path_direct_mediator_relative_error"] <= THRESHOLDS["path_handoff_relative_error_max"]
        and metrics["path_block_remaining"] <= THRESHOLDS["path_block_remaining_max"]
        and metrics["path_rescue_relative_error"] <= THRESHOLDS["path_rescue_relative_error_max"]
        and metrics["wrong_rescue_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["on_manifold_relative_distance_p95"] <= THRESHOLDS["on_manifold_relative_distance_p95_max"]
    )


def control_metrics(system: SystemTruth, selected: dict[str, Any] | None, discovery: dict[str, torch.Tensor], selection: dict[str, torch.Tensor]) -> dict[str, float]:
    mean_prediction = discovery["A1"].mean(dim=0, keepdim=True).expand_as(selection["A1"])
    mean_error = relative(mean_prediction, selection["A1"], selection["A1"])
    sign_error = relative(-selection["A0"], selection["A1"], selection["A1"])
    if selected is None:
        shuffled_error = mean_error
    elif selected["name"] in {"legacy_zero_null", "additive_preserving"}:
        shuffled_error = sign_error
    else:
        permutation = torch.arange(discovery["A1"].shape[0] - 1, -1, -1, device=system.device)
        shuffled = fit_regression(selected["name"], discovery["A0"], discovery["B"], discovery["A1"][permutation], system.seed + 991)
        shuffled_error = candidate_metrics(system, shuffled, selection)["a1_relative_error"]
    return {"mean_response_error": mean_error, "sign_flip_error": sign_error, "label_permutation_error": shuffled_error, "oracle_error": 0.0}


def run_system(public: dict[str, Any], truth: dict[str, Any], device: torch.device) -> dict[str, Any]:
    system = SystemTruth(str(truth["family"]), int(truth["seed"]), int(public["task_id"]), device)
    discovery = system.make_partition("discovery", PARTITION_COUNTS["discovery"])
    selection = system.make_partition("selection", PARTITION_COUNTS["selection"])
    confirmation = system.make_partition("confirmation", PARTITION_COUNTS["confirmation"])
    candidates = fit_candidates(system, discovery)
    selection_rows = []
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        metrics = candidate_metrics(system, candidate, selection)
        passed = selection_passes(metrics)
        selection_rows.append({"compiler": candidate["name"], "metrics": metrics, "passed": passed, "summary": serializable_model_summary(candidate)})
        if selected is None and passed:
            selected = candidate
    selected_name = selected["name"] if selected is not None else "abstain"
    mediator_model = fit_candidate_by_name(system, discovery, selected_name, "M_") if selected is not None else None
    confirmation_result = confirmation_metrics(system, selected, mediator_model, confirmation) if selected is not None else None
    controls = control_metrics(system, selected, discovery, selection)
    return {
        "system_id": public["system_id"],
        "registry_split": public["registry_split"],
        "task_id": public["task_id"],
        "replicate": public["replicate"],
        "selected_compiler": selected_name,
        "selection": selection_rows,
        "confirmation": confirmation_result,
        "confirmation_passed": confirmation_passes(confirmation_result) if confirmation_result is not None else True,
        "controls": controls,
    }


def run(device_name: str) -> None:
    protocol, public, truth = verify_protocol()
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass before formal run")
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
    print(canonical_json({"status": "formal_run_complete", "systems": len(rows), "elapsed_seconds": elapsed}))


def analyze() -> None:
    protocol, public, truth = verify_protocol()
    if not COMPLETE.exists():
        raise RuntimeError("formal run is incomplete")
    rows = read_jsonl(RAW)
    if digest(rows) != read_json(SUMMARY)["raw_digest"]:
        raise RuntimeError("raw digest mismatch")
    truth_by_id = {row["system_id"]: row for row in truth}
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in REGISTRY_SPLITS}
    for row in rows:
        private = truth_by_id[row["system_id"]]
        evaluated = {**row, "family": private["family"], "expected_compiler": private["expected_compiler"], "type_correct": row["selected_compiler"] == private["expected_compiler"]}
        by_family[private["family"]].append(evaluated)
        by_split[row["registry_split"]].append(evaluated)
    type_accuracy = sum(row["selected_compiler"] == truth_by_id[row["system_id"]]["expected_compiler"] for row in rows) / len(rows)
    abstain_rows = [row for row in rows if truth_by_id[row["system_id"]]["expected_compiler"] == "abstain"]
    abstention_accuracy = sum(row["selected_compiler"] == "abstain" for row in abstain_rows) / len(abstain_rows)
    actionable = [row for row in rows if row["selected_compiler"] != "abstain"]
    applicable_controls = [row["controls"] for row in actionable]
    family_summary = {}
    for family, family_rows in by_family.items():
        confirmations = [row["confirmation"] for row in family_rows if row["confirmation"] is not None]
        family_summary[family] = {
            "systems": len(family_rows),
            "expected_compiler": EXPECTED_COMPILER[family],
            "selected_counts": {name: sum(row["selected_compiler"] == name for row in family_rows) for name in (*COMPILER_ORDER, "abstain")},
            "type_accuracy": sum(row["type_correct"] for row in family_rows) / len(family_rows),
            "confirmation_pass_fraction": sum(row["confirmation_passed"] for row in family_rows) / len(family_rows),
            "worst_confirmation": None if not confirmations else {
                "a1_relative_error_max": max(row["a1_relative_error"] for row in confirmations),
                "j_error_fraction_max": max(row["j_error_fraction"] for row in confirmations),
                "context_error_fraction_max": max(row["context_error_fraction"] for row in confirmations),
                "output_relative_error_max": max(row["output_relative_error"] for row in confirmations),
                "wrong_identity_accuracy_min": min(row["wrong_identity_accuracy"] for row in confirmations),
                "nuisance_sensitivity_max": max(row["nuisance_sensitivity"] for row in confirmations),
                "path_block_remaining_max": max(row["path_block_remaining"] for row in confirmations),
                "path_rescue_relative_error_max": max(row["path_rescue_relative_error"] for row in confirmations),
            },
        }
    split_pass = {split: all(row["type_correct"] and row["confirmation_passed"] for row in split_rows) for split, split_rows in by_split.items()}
    control_floor = min(min(row["mean_response_error"], row["sign_flip_error"], row["label_permutation_error"]) for row in applicable_controls) if applicable_controls else 0.0
    gates = {
        "G-COMPILER-TYPE": type_accuracy >= THRESHOLDS["compiler_type_accuracy_min"],
        "G-ABSTENTION": abstention_accuracy >= THRESHOLDS["abstention_accuracy_min"],
        "G-A1-PREDICTION": bool(actionable) and all(row["confirmation"] is not None and row["confirmation"]["a1_relative_error"] <= THRESHOLDS["confirmation_a1_relative_error_max"] for row in actionable),
        "G-B-PRESERVATION": bool(actionable) and all(row["confirmation"]["context_error_fraction"] <= THRESHOLDS["confirmation_context_error_fraction_max"] for row in actionable),
        "G-J-PREDICTION": bool(actionable) and all(row["confirmation"]["j_error_fraction"] <= THRESHOLDS["confirmation_j_error_fraction_max"] for row in actionable),
        "G-WRONG": bool(actionable) and all(row["confirmation"]["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"] and row["confirmation"]["wrong_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"] for row in actionable),
        "G-NUISANCE": bool(actionable) and all(row["confirmation"]["nuisance_sensitivity"] <= THRESHOLDS["nuisance_sensitivity_max"] for row in actionable),
        "G-PATH": bool(actionable) and all(row["confirmation"]["path_handoff_relative_error"] <= THRESHOLDS["path_handoff_relative_error_max"] and row["confirmation"]["path_block_remaining"] <= THRESHOLDS["path_block_remaining_max"] and row["confirmation"]["path_rescue_relative_error"] <= THRESHOLDS["path_rescue_relative_error_max"] for row in actionable),
        "G-MANIFOLD": bool(actionable) and all(row["confirmation"]["on_manifold_relative_distance_p95"] <= THRESHOLDS["on_manifold_relative_distance_p95_max"] for row in actionable),
        "G-FAMILY-HOLDOUT": all(split_pass.values()),
        "G-CONTROLS": control_floor >= THRESHOLDS["control_error_floor_min"],
    }
    passed = all(gates.values())
    adjudication = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": utc_now(),
        "systems": len(rows),
        "type_accuracy": type_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "actionable_systems": len(actionable),
        "family_summary": family_summary,
        "split_pass": split_pass,
        "control_error_floor": control_floor,
        "gates": gates,
        "passed": passed,
        "verdict": "factorial_response_compiler_camera_calibrated" if passed else "factorial_response_compiler_camera_not_calibrated",
        "claim_boundary": "Known-truth factorial response compiler and path camera only; no free-network, pretrained-model or language claim.",
        "authorization": {"free_transformer_factorial_compiler": passed, "qwen3": False, "language_mechanism": False, "new_mathematics": False},
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
    print(canonical_json({"verdict": final["verdict"], "passed": passed, "type_accuracy": type_accuracy, "abstention_accuracy": abstention_accuracy}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def probe(device_name: str) -> None:
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA")
    device = torch.device("cuda")
    rows = []
    for index, family in enumerate(FAMILIES):
        for task_id in range(2):
            for replicate in range(3):
                system_id = f"probe-{family}-{task_id}-{replicate}"
                public = {"system_id": system_id, "registry_split": "probe", "task_id": task_id, "replicate": replicate}
                truth = {
                    "family": family,
                    "seed": 1_261_990_000 + index * 10_000 + task_id * 1_000 + replicate * 100,
                    "expected_compiler": EXPECTED_COMPILER[family],
                }
                rows.append({**run_system(public, truth, device), "family": family, "expected": EXPECTED_COMPILER[family]})
    atomic_json(PROBE, rows)
    print(canonical_json({
        "systems": len(rows),
        "type_correct": sum(row["selected_compiler"] == row["expected"] for row in rows),
        "confirmation_passed": sum(row["confirmation_passed"] for row in rows),
        "mismatches": [row["system_id"] for row in rows if row["selected_compiler"] != row["expected"]],
        "confirmation_failures": [row["system_id"] for row in rows if not row["confirmation_passed"]],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze", "all"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "probe":
        probe(args.device)
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
