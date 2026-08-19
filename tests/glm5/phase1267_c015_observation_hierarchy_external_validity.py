"""Phase1267: observation-hierarchy triage after the C014 external-validity failure.

Nine fresh 4/6/8-layer free Transformers are evaluated on the same exhaustive
cyclic-code universe with fresh sampled partitions.  Four cameras compete:
the frozen C014 RFF chart, a standardized delta chart, a full same-event state
chart, and an answer-boundary prefix-trajectory chart.  Every chart predicts
the factorial interaction J and reconstructs A1=A0+J.

Selection uses simultaneous finite-population certificates.  Exact population
risk is adjudication only.  Rescue requires an independent donor fit, a
separate causal-selection sentinel, wrong-identity rejection, and an untouched
confirmation partition.  The outcome distinguishes chart parameterization,
base-state information, trajectory information, and registered-hierarchy
insufficiency without loading a pretrained language model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266


PHASE = 1267
CAMPAIGN = "C015"
CONTRACT_ID = "EXP-C015-WP01-001"
OUT = ROOT / "tests/glm5/result/phase1267_c015_observation_hierarchy_external_validity"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1267_c015_observation_hierarchy_external_validity_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C015-WP01-001.json"
PHASE1266_FINAL = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate/analysis/final.json"
PHASE1266_COMPLETE = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate/raw/FORMAL_RUN_COMPLETE.json"
PHASE1266_AUDIT = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate/audit/independent_final_audit.json"
PHASE1266_ERRATUM = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate/audit/independent_final_audit_erratum.json"

ARCHITECTURES = p1266.ARCHITECTURES
REPLICATES = 3
MODEL_SEEDS = {
    "shallow4_r0": 1_267_401_001,
    "shallow4_r1": 1_267_401_101,
    "shallow4_r2": 1_267_401_201,
    "middle6_r0": 1_267_601_001,
    "middle6_r1": 1_267_601_101,
    "middle6_r2": 1_267_601_201,
    "deep8_r0": 1_267_801_001,
    "deep8_r1": 1_267_801_101,
    "deep8_r2": 1_267_801_201,
}
MATERIAL_SEEDS = {
    "discovery": 1_267_910_001,
    "donor": 1_267_920_001,
    "causal_selection": 1_267_925_001,
    "confirmation": 1_267_930_001,
}
PARTITION_COUNTS = {
    "oracle": 3456,
    "discovery": 1024,
    "donor": 1024,
    "causal_selection": 1024,
    "confirmation": 1024,
}
CAMERAS = ("legacy_delta", "enhanced_delta", "full_state", "prefix_trajectory")
COMPONENT_RANK = 8
OUTPUT_RANK = 16
RFF_WIDTH = 256
RIDGE = 1.0e-5
SELECTION_DRAWS = 32768
PASS_MAX = 0.050
GLOBAL_ERROR_BUDGET = 0.01
MAX_EVENTS = sum(config.layers for config in ARCHITECTURES.values()) * REPLICATES
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_EVENTS * len(CAMERAS) / GLOBAL_ERROR_BUDGET)
    / (2.0 * SELECTION_DRAWS)
)
ROBUST_MULTIPLIER = 2.0

THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "population_pass_max": PASS_MAX,
    "certificate_radius": CERTIFICATE_RADIUS,
    "certificate_false_authorizations_max": 0,
    "robust_coverage_min": 0.90,
    "rescue_authorization_upper_max": 0.025,
    "causal_state_relative_error_max": 0.30,
    "causal_output_cosine_min": 0.90,
    "causal_correct_accuracy_min": 0.95,
    "wrong_identity_accuracy_min": 0.95,
    "wrong_false_target_max": 0.05,
    "oracle_patch_accuracy_min": 0.999,
    "reverse_block_accuracy_min": 0.999,
    "breadth_models_min": 6,
    "breadth_per_depth_min": 2,
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


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def sample_worlds(partition: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        source = int(rng.integers(4))
        target = int(rng.choice([value for value in range(4) if value != source]))
        shift0 = int(rng.integers(4))
        shift1 = int(rng.choice([value for value in range(4) if value != shift0]))
        order = rng.permutation(4).astype(int).tolist()
        rows.append(
            p1266.make_factorial_world(
                source,
                target,
                shift0,
                shift1,
                order,
                partition,
                f"{partition[0]}{index:04d}",
            )
        )
    return rows


def make_material() -> list[dict[str, Any]]:
    rows = p1266.enumerate_oracle_worlds()
    for partition in ("discovery", "donor", "causal_selection", "confirmation"):
        rows.extend(sample_worlds(partition, PARTITION_COUNTS[partition], MATERIAL_SEEDS[partition]))
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predecessor = read_json(PHASE1266_FINAL)
    frozen_audit = read_json(PHASE1266_AUDIT)
    erratum = read_json(PHASE1266_ERRATUM)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1267.c015.observation_hierarchy.v1",
        "claim_type": "free_transformer_observation_hierarchy_external_validity",
        "question": "Which registered observation level first yields certified, causally replayable factorial responses across fresh free Transformers?",
        "phase1266_dependency": {
            "formal_complete": PHASE1266_COMPLETE.exists(),
            "passed": predecessor.get("passed"),
            "final_hash": file_sha256(PHASE1266_FINAL),
            "frozen_audit": {
                "passed_checks": frozen_audit.get("passed_checks"),
                "total_checks": frozen_audit.get("total_checks"),
                "hash": file_sha256(PHASE1266_AUDIT),
            },
            "scope_erratum_passed": erratum.get("passed"),
            "scope_erratum_hash": file_sha256(PHASE1266_ERRATUM),
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "partitions": PARTITION_COUNTS,
        "material_seeds": MATERIAL_SEEDS,
        "world_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "camera_order": list(CAMERAS),
        "camera_contract": {
            "legacy_delta": "Phase1266 rank-16 unstandardized RFF chart on (A0,B), predicts A1=A0+J",
            "enhanced_delta": "componentwise rank-8 standardized RFF chart on (A0,B), predicts A1=A0+J",
            "full_state": "same enhanced chart on (H00,A0,B) at one answer-boundary event",
            "prefix_trajectory": "same enhanced chart on (H00,A0,B) for all answer-boundary layers through the event",
            "component_rank": COMPONENT_RANK,
            "output_rank": OUTPUT_RANK,
            "rff_width": RFF_WIDTH,
            "rff_input_scale": 0.35,
            "fit": "discovery only",
            "truth": "exact bounded risk over all 3456 registered worlds",
            "selection": f"{SELECTION_DRAWS} uniform with-replacement oracle draws",
        },
        "decision_order": {
            "legacy_delta": "legacy_library_sufficient_under_revised_breadth",
            "enhanced_delta": "camera_parameterization_bottleneck",
            "full_state": "base_state_information_bottleneck",
            "prefix_trajectory": "prefix_trajectory_information_bottleneck",
            "none": "registered_hierarchy_insufficient",
        },
        "thresholds": THRESHOLDS,
        "causal_contract": {
            "selection": "rescue-authorized event plus exact-state patch and reverse-block on causal_selection; freeze earliest and latest per camera",
            "confirmation": "refit the same camera on donor and test correct/wrong identity on untouched confirmation",
        },
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 2.5},
        "hard_stops": [
            "All nine behavior-qualified or unqualified models remain in the denominator.",
            "Exact population risk is adjudication only and cannot select a camera or event.",
            "No confirmation result may change camera, layer, threshold, seed, rank, or architecture.",
            "Failure closes C015; success authorizes only design of a separate pretrained contract.",
            "No pretrained model is loaded automatically.",
        ],
        "structured_scope": {
            "task": "synthetic cyclic-code",
            "models": "small free same-executor Transformers",
            "natural_language": False,
            "unique_circuit": False,
            "pretrained": False,
        },
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "phase1266": file_sha256(ROOT / "tests/glm5/phase1266_c014_free_transformer_population_certificate.py"),
            "task": file_sha256(ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"),
            "executor": file_sha256(ROOT / "tests/glm5/phase1260_c011_free_transformer_selective_operator_mediation.py"),
        },
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
        "precision": "fp32 training/intervention; fp64 camera fit and certificate",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    predecessor = read_json(PHASE1266_FINAL)
    frozen_audit = read_json(PHASE1266_AUDIT)
    erratum = read_json(PHASE1266_ERRATUM)
    if predecessor.get("passed") is not False or not PHASE1266_COMPLETE.exists():
        raise RuntimeError("Phase1266 is not a completed negative predecessor")
    if frozen_audit.get("passed_checks") != 15 or frozen_audit.get("total_checks") != 16:
        raise RuntimeError("Phase1266 frozen audit ledger drift")
    if not erratum.get("passed"):
        raise RuntimeError("Phase1266 scope erratum did not pass")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist before preregistration")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "models": len(MODEL_SEEDS), "radius": CERTIFICATE_RADIUS}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest drift")
    if protocol["source_hashes"] != expected["source_hashes"] or protocol["thresholds"] != THRESHOLDS:
        raise RuntimeError("source or threshold drift")
    counts = {name: sum(row["partition"] == name for row in rows) for name in PARTITION_COUNTS}
    if counts != PARTITION_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    return protocol, rows


def partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def svd_basis(samples: torch.Tensor, rank: int) -> torch.Tensor:
    values = samples.double()
    _u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 1.0e-12:
        return torch.zeros((values.shape[1], 0), dtype=torch.float64, device=values.device)
    effective = min(rank, int(torch.sum(singular > singular[0] * 1.0e-7).item()))
    return vh[:effective].T.contiguous()


def raw_components(capture: dict[str, Any], layer: int, camera: str, wrong: bool = False) -> list[torch.Tensor]:
    states = capture["states"]
    target_panel = "hwrong10" if wrong else "h10"

    def at(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h00 = states["h00"][:, index]
        a0 = states[target_panel][:, index] - h00
        b = states["h01"][:, index] - h00
        return h00, a0, b

    if camera in ("legacy_delta", "enhanced_delta"):
        _h00, a0, b = at(layer)
        return [a0, b]
    if camera == "full_state":
        return list(at(layer))
    if camera == "prefix_trajectory":
        values: list[torch.Tensor] = []
        for index in range(layer + 1):
            values.extend(at(index))
        return values
    raise ValueError(camera)


def generic_features(z: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((z.shape[0], 1), dtype=torch.float64, device=z.device)
    phi = math.sqrt(1.0 / RFF_WIDTH) * torch.cat(
        (torch.sin(z @ omega + phase), torch.cos(z @ omega + phase)), dim=1
    )
    return torch.cat((ones, z, phi), dim=1)


def ridge_fit(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = features.double(), target.double()
    gram = x.T @ x
    scale = float(torch.trace(gram).item()) / max(1, gram.shape[0])
    regularizer = RIDGE * max(scale, 1.0)
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=x.device)
    return torch.linalg.solve(gram + regularizer * eye, x.T @ y)


def fit_camera(capture: dict[str, Any], layer: int, camera: str) -> dict[str, Any]:
    factorial = p1266.factorial_at_layer(capture, layer)
    if camera == "legacy_delta":
        candidate = p1266.fit_layer_candidates(factorial["A0"], factorial["B"], factorial["A1"])["rff"]
        return {"camera": camera, "candidate": candidate}
    components = raw_components(capture, layer, camera)
    bases = [svd_basis(value, COMPONENT_RANK) for value in components]
    z = torch.cat([value.double() @ basis for value, basis in zip(components, bases)], dim=1)
    mean = z.mean(dim=0)
    std = z.std(dim=0).clamp_min(1.0e-5)
    z = (z - mean) / std
    generator = torch.Generator(device=z.device)
    generator.manual_seed(1_267_070 + 101 * layer + 11 * CAMERAS.index(camera) + z.shape[1])
    omega = torch.randn((z.shape[1], RFF_WIDTH), generator=generator, dtype=torch.float64, device=z.device) * 0.35
    phase = torch.rand((RFF_WIDTH,), generator=generator, dtype=torch.float64, device=z.device) * (2.0 * math.pi)
    interaction = (factorial["A1"] - factorial["A0"]).double()
    output_basis = svd_basis(interaction, OUTPUT_RANK)
    weights = ridge_fit(generic_features(z, omega, phase), interaction @ output_basis)
    return {
        "camera": camera,
        "bases": bases,
        "mean": mean,
        "std": std,
        "omega": omega,
        "phase": phase,
        "output_basis": output_basis,
        "weights": weights,
    }


def predict_camera(model: dict[str, Any], capture: dict[str, Any], layer: int, wrong: bool = False) -> torch.Tensor:
    camera = model["camera"]
    components = raw_components(capture, layer, camera, wrong=wrong)
    a0 = components[0] if camera in ("legacy_delta", "enhanced_delta") else components[1] if camera == "full_state" else components[-2]
    b = components[1] if camera in ("legacy_delta", "enhanced_delta") else components[2] if camera == "full_state" else components[-1]
    if camera == "legacy_delta":
        return p1266.predict_candidate(model["candidate"], a0, b)
    z = torch.cat([value.double() @ basis for value, basis in zip(components, model["bases"])], dim=1)
    z = (z - model["mean"]) / model["std"]
    interaction = generic_features(z, model["omega"], model["phase"]) @ model["weights"] @ model["output_basis"].T
    return a0.double() + interaction


def selection_indices(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 7_777_127)
    return torch.randint(0, PARTITION_COUNTS["oracle"], (SELECTION_DRAWS,), generator=generator, device=device)


def confidence(point: float) -> dict[str, float]:
    return {
        "point": point,
        "lower": max(0.0, point - CERTIFICATE_RADIUS),
        "upper": min(1.0, point + CERTIFICATE_RADIUS),
    }


def exact_causal_sentinel(
    model,
    layer: int,
    capture: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    states = capture["states"]
    h01_ids = torch.tensor([row["h01_ids"] for row in rows], device=device)
    h11_ids = torch.tensor([row["h11_ids"] for row in rows], device=device)
    h01_state = states["h01"][:, layer]
    h11_state = states["h11"][:, layer]
    with torch.inference_mode():
        oracle_logits, _ = p1266.executor.explicit_residual_forward(
            model, h01_ids, actions={layer: lambda _current: h11_state}, capture=False
        )
        reverse_logits, _ = p1266.executor.explicit_residual_forward(
            model, h11_ids, actions={layer: lambda _current: h01_state}, capture=False
        )
    target = torch.tensor([row["answers"]["h11"] for row in rows], device=device)
    base = torch.tensor([row["answers"]["h01"] for row in rows], device=device)
    oracle_pred = torch.argmax(oracle_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    reverse_pred = torch.argmax(reverse_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    oracle_accuracy = float((oracle_pred == target).float().mean().item())
    reverse_accuracy = float((reverse_pred == base).float().mean().item())
    return {
        "oracle_patch_accuracy": oracle_accuracy,
        "reverse_block_accuracy": reverse_accuracy,
        "causally_admissible": (
            oracle_accuracy >= THRESHOLDS["oracle_patch_accuracy_min"]
            and reverse_accuracy >= THRESHOLDS["reverse_block_accuracy_min"]
        ),
    }


def camera_confirmation(
    model,
    layer: int,
    camera: str,
    donor_capture: dict[str, Any],
    confirmation_capture: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    donor_model = fit_camera(donor_capture, layer, camera)
    predicted = predict_camera(donor_model, confirmation_capture, layer, wrong=False)
    predicted_wrong = predict_camera(donor_model, confirmation_capture, layer, wrong=True)
    states = confirmation_capture["states"]
    h01_ids = torch.tensor([row["h01_ids"] for row in rows], device=device)
    h11_ids = torch.tensor([row["h11_ids"] for row in rows], device=device)
    h01_state = states["h01"][:, layer]
    h11_state = states["h11"][:, layer]
    with torch.inference_mode():
        correct_logits, _ = p1266.executor.explicit_residual_forward(
            model, h01_ids, actions={layer: lambda current: current + predicted.to(current.dtype)}, capture=False
        )
        wrong_logits, _ = p1266.executor.explicit_residual_forward(
            model, h01_ids, actions={layer: lambda current: current + predicted_wrong.to(current.dtype)}, capture=False
        )
        oracle_logits, _ = p1266.executor.explicit_residual_forward(
            model, h01_ids, actions={layer: lambda _current: h11_state}, capture=False
        )
        reverse_logits, _ = p1266.executor.explicit_residual_forward(
            model, h11_ids, actions={layer: lambda _current: h01_state}, capture=False
        )
    base_output = p1266.centered_last(confirmation_capture["logits"]["h01"])
    target_output = p1266.centered_last(confirmation_capture["logits"]["h11"])
    correct_response = p1266.executor.centered(correct_logits) - base_output
    target_response = target_output - base_output
    target_answers = torch.tensor([row["answers"]["h11"] for row in rows], device=device)
    wrong_answers = torch.tensor([row["answers"]["hwrong11"] for row in rows], device=device)
    base_answers = torch.tensor([row["answers"]["h01"] for row in rows], device=device)
    correct_pred = torch.argmax(correct_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    wrong_pred = torch.argmax(wrong_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    oracle_pred = torch.argmax(oracle_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    reverse_pred = torch.argmax(reverse_logits[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
    truth = p1266.factorial_at_layer(confirmation_capture, layer)["A1"].double()
    metrics = {
        "layer": layer,
        "camera": camera,
        "cases": len(rows),
        "state_relative_error": float(
            (torch.linalg.vector_norm(predicted - truth) / torch.linalg.vector_norm(truth).clamp_min(1.0e-12)).item()
        ),
        "correct_output": p1266.effect_metrics(correct_response, target_response),
        "correct_accuracy": float((correct_pred == target_answers).float().mean().item()),
        "wrong_identity_accuracy": float((wrong_pred == wrong_answers).float().mean().item()),
        "wrong_false_target": float((wrong_pred == target_answers).float().mean().item()),
        "oracle_patch_accuracy": float((oracle_pred == target_answers).float().mean().item()),
        "reverse_block_accuracy": float((reverse_pred == base_answers).float().mean().item()),
    }
    metrics["passed"] = (
        metrics["state_relative_error"] <= THRESHOLDS["causal_state_relative_error_max"]
        and metrics["correct_output"]["cosine"] >= THRESHOLDS["causal_output_cosine_min"]
        and metrics["correct_accuracy"] >= THRESHOLDS["causal_correct_accuracy_min"]
        and metrics["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["wrong_false_target"] <= THRESHOLDS["wrong_false_target_max"]
        and metrics["oracle_patch_accuracy"] >= THRESHOLDS["oracle_patch_accuracy_min"]
        and metrics["reverse_block_accuracy"] >= THRESHOLDS["reverse_block_accuracy_min"]
    )
    return metrics


def run_model(
    architecture: str,
    replicate: int,
    config,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    key = model_key(architecture, replicate)
    seed = MODEL_SEEDS[key]
    p1266.set_seed(seed)
    model, training = p1266.task_module.train_model(config, seed, device)
    captures = {
        partition: p1266.capture_partition(model, partition_rows(rows, partition), device)
        for partition in PARTITION_COUNTS
    }
    natural_min = min(value for capture in captures.values() for value in capture["accuracies"].values())
    executor_gap = max(capture["executor_gap"] for capture in captures.values())
    qualified = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], natural_min)
        >= THRESHOLDS["behavior_accuracy_min"]
        and executor_gap <= THRESHOLDS["executor_gap_max"]
    )
    base = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "training": training,
        "natural_accuracy_min": natural_min,
        "executor_gap": executor_gap,
        "behavior_qualified": qualified,
    }
    if not qualified:
        return {**base, "cameras": {}, "passed_cameras": [], "passed": False}

    selection = selection_indices(seed, device)
    ledgers: dict[str, list[dict[str, Any]]] = {camera: [] for camera in CAMERAS}
    for layer in range(config.layers):
        truth = p1266.factorial_at_layer(captures["oracle"], layer)["A1"]
        for camera in CAMERAS:
            fitted = fit_camera(captures["discovery"], layer, camera)
            predicted = predict_camera(fitted, captures["oracle"], layer)
            losses = p1266.bounded_loss_vector(predicted, truth)
            population_risk = float(losses.mean().item())
            sample_risk = float(losses[selection].mean().item())
            bounds = confidence(sample_risk)
            exact_pass = population_risk <= PASS_MAX
            certificate_pass = bounds["upper"] <= PASS_MAX
            robust = population_risk <= PASS_MAX - ROBUST_MULTIPLIER * CERTIFICATE_RADIUS
            ledgers[camera].append(
                {
                    "layer": layer,
                    "population_risk": population_risk,
                    "sample_risk": sample_risk,
                    "confidence": bounds,
                    "exact_pass": exact_pass,
                    "robust_actionable": robust,
                    "certificate_pass": certificate_pass,
                    "rescue_authorized": certificate_pass
                    and bounds["upper"] <= THRESHOLDS["rescue_authorization_upper_max"],
                }
            )

    causal_rows = partition_rows(rows, "causal_selection")
    candidate_layers = sorted(
        {
            event["layer"]
            for camera in CAMERAS
            for event in ledgers[camera]
            if event["rescue_authorized"]
        }
    )
    sentinels = {
        layer: exact_causal_sentinel(model, layer, captures["causal_selection"], causal_rows, device)
        for layer in candidate_layers
    }
    confirmation_rows = partition_rows(rows, "confirmation")
    camera_results = {}
    for camera in CAMERAS:
        events = ledgers[camera]
        eligible = [
            event["layer"]
            for event in events
            if event["rescue_authorized"] and sentinels[event["layer"]]["causally_admissible"]
        ]
        selected = []
        if eligible:
            selected.append(eligible[0])
            if eligible[-1] != eligible[0]:
                selected.append(eligible[-1])
        confirmations = [
            camera_confirmation(
                model,
                layer,
                camera,
                captures["donor"],
                captures["confirmation"],
                confirmation_rows,
                device,
            )
            for layer in selected
        ]
        robust_events = [event for event in events if event["robust_actionable"]]
        robust_covered = sum(event["certificate_pass"] for event in robust_events)
        false_authorizations = sum(event["certificate_pass"] and not event["exact_pass"] for event in events)
        point_false_authorizations = sum(event["sample_risk"] <= PASS_MAX and not event["exact_pass"] for event in events)
        camera_results[camera] = {
            "events": events,
            "false_authorizations": false_authorizations,
            "point_false_authorizations": point_false_authorizations,
            "robust_events": len(robust_events),
            "robust_coverage": robust_covered / max(1, len(robust_events)),
            "certified_events": sum(event["certificate_pass"] for event in events),
            "rescue_authorized_events": sum(event["rescue_authorized"] for event in events),
            "selected_events": selected,
            "confirmations": confirmations,
            "passed": bool(selected) and all(item["passed"] for item in confirmations),
        }
    passed_cameras = [camera for camera in CAMERAS if camera_results[camera]["passed"]]
    return {
        **base,
        "causal_sentinels": sentinels,
        "cameras": camera_results,
        "passed_cameras": passed_cameras,
        "passed": bool(passed_cameras),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in rows if row["behavior_qualified"]]
    camera_summaries = {}
    authorized = []
    for camera in CAMERAS:
        camera_rows = [row["cameras"][camera] for row in qualified]
        all_events = [event for value in camera_rows for event in value["events"]]
        robust_events = [event for event in all_events if event["robust_actionable"]]
        passed_models = [row for row in qualified if row["cameras"][camera]["passed"]]
        per_depth = {
            architecture: sum(row["architecture"] == architecture for row in passed_models)
            for architecture in ARCHITECTURES
        }
        false_authorizations = sum(value["false_authorizations"] for value in camera_rows)
        robust_coverage = sum(event["certificate_pass"] for event in robust_events) / max(1, len(robust_events))
        breadth = (
            len(passed_models) >= THRESHOLDS["breadth_models_min"]
            and all(count >= THRESHOLDS["breadth_per_depth_min"] for count in per_depth.values())
        )
        gates = {
            "zero_false_authorization": false_authorizations <= THRESHOLDS["certificate_false_authorizations_max"],
            "robust_coverage": bool(robust_events) and robust_coverage >= THRESHOLDS["robust_coverage_min"],
            "causal_breadth": breadth,
        }
        camera_summaries[camera] = {
            "events": len(all_events),
            "exact_pass_events": sum(event["exact_pass"] for event in all_events),
            "certified_events": sum(event["certificate_pass"] for event in all_events),
            "robust_events": len(robust_events),
            "robust_coverage": robust_coverage,
            "false_authorizations": false_authorizations,
            "point_false_authorizations": sum(value["point_false_authorizations"] for value in camera_rows),
            "rescue_authorized_events": sum(value["rescue_authorized_events"] for value in camera_rows),
            "passed_models": len(passed_models),
            "per_depth": per_depth,
            "gates": gates,
            "authorized": all(gates.values()),
        }
        if camera_summaries[camera]["authorized"]:
            authorized.append(camera)
    if authorized:
        first = next(camera for camera in CAMERAS if camera in authorized)
        decision = {
            "legacy_delta": "legacy_library_sufficient_under_revised_breadth",
            "enhanced_delta": "camera_parameterization_bottleneck",
            "full_state": "base_state_information_bottleneck",
            "prefix_trajectory": "prefix_trajectory_information_bottleneck",
        }[first]
    else:
        first = None
        decision = "registered_hierarchy_insufficient"
    gates = {
        "G-BEHAVIOR": len(qualified) == len(rows),
        "G-IDENTIFIABLE-CAMERA": first is not None,
        "G-NO-PRETRAINED": True,
    }
    return {
        "models": len(rows),
        "qualified": len(qualified),
        "camera_summaries": camera_summaries,
        "authorized_cameras": authorized,
        "minimal_authorized_camera": first,
        "decision": decision,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(device_name: str) -> None:
    if COMPLETE.exists():
        raise RuntimeError("formal run already completed")
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass")
    protocol, rows = verify_protocol()
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    results = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            result = run_model(architecture, replicate, config, rows, device)
            results.append(result)
            write_jsonl(MODELS, results)
            print(canonical_json({"completed": len(results), "total": len(MODEL_SEEDS), "model": result["model_key"], "passed_cameras": result["passed_cameras"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "models": len(results),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "device": torch.cuda.get_device_name(0),
        "models_hash": file_sha256(MODELS),
        "run_digest": digest(results),
        "protocol_digest": protocol["protocol_digest"],
        "pretrained_model_loaded": False,
    }
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "run_digest": summary["run_digest"], "models_hash": summary["models_hash"]})


def analyze() -> None:
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    protocol, _rows = verify_protocol()
    results = read_jsonl(MODELS)
    summary = summarize(results)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        **summary,
        "authorization": {
            "new_pretrained_contract_design": summary["passed"],
            "automatic_pretrained_run": False,
            "qwen3": False,
            "glm4": False,
            "ds7b": False,
        },
        "structured_scope": protocol["structured_scope"],
        "protocol_digest": protocol["protocol_digest"],
        "models_hash": file_sha256(MODELS),
        "run_digest": digest(results),
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"decision": summary["decision"], "minimal_camera": summary["minimal_authorized_camera"], "passed": summary["passed"], "authorized": summary["authorized_cameras"]}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--force", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--device", default="cuda")
    sub.add_parser("analyze")
    audit = sub.add_parser("audit")
    audit.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        run(args.device)
    elif args.command == "analyze":
        analyze()
    elif args.command == "audit":
        run_auditor(args.mode)


if __name__ == "__main__":
    main()
