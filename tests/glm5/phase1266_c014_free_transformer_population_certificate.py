"""Phase1266: shape-safe free-Transformer external validity of C014.

Six freely trained same-executor Transformers are evaluated on an exhaustive
finite universe of source-change x later-shift-change factorial worlds.  Every
answer-boundary layer event is treated as a separate response object.  A
compressed additive/linear/quadratic/RFF library predicts A1 from A0 and B.

Exact finite-universe risk is used only for adjudication.  Selection uses the
frozen simultaneous interval rule.  High-fidelity certified events undergo an
independent-donor causal patch, wrong-identity patch, exact-state positive
sentinel and reverse-state block on separately sampled worlds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
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
from typing import Any, Callable, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1251_c004_causal_slice_competition as task_module
import phase1255_c008_same_executor_edge_external_validity as same_executor
import phase1260_c011_free_transformer_selective_operator_mediation as executor
from phase1146_learned_composition_benchmark import ModelConfig
from phase1251_c004_causal_slice_competition import CANDIDATE_SLICE, build_sequence


PHASE = 1266
CAMPAIGN = "C014"
CONTRACT_ID = "EXP-C014-WP03-001"
OUT = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1266_c014_free_transformer_population_certificate_audit.py"
CONTRACT = ROOT / "research/ai2050_research_os/contracts/EXP-C014-WP03-001.json"
PHASE1264_FINAL = ROOT / "tests/glm5/result/phase1264_c014_population_risk_certificate_calibration/analysis/final.json"
PHASE1264_AUDIT = ROOT / "tests/glm5/result/phase1264_c014_population_risk_certificate_calibration/audit/independent_final_audit.json"
PHASE1265_INVALID = ROOT / "tests/glm5/result/phase1265_c014_free_transformer_population_certificate/analysis/INVALID_ENGINEERING_RUN.json"
PROBE = ROOT / "tests/glm5_temp/phase1266_c014_free_transformer_population_certificate_probe.json"

ARCHITECTURES = same_executor.ARCHITECTURES
REPLICATES = 2
MODEL_SEEDS = {
    "shallow4_r0": 1_266_401_001,
    "shallow4_r1": 1_266_401_101,
    "middle6_r0": 1_266_601_001,
    "middle6_r1": 1_266_601_101,
    "deep8_r0": 1_266_801_001,
    "deep8_r1": 1_266_801_101,
}
MATERIAL_SEEDS = {
    "discovery": 1_266_910_001,
    "donor": 1_266_920_001,
    "causal_selection": 1_266_925_001,
    "confirmation": 1_266_930_001,
}
PARTITION_COUNTS = {
    "oracle": 3456,
    "discovery": 1024,
    "donor": 1024,
    "causal_selection": 1024,
    "confirmation": 1024,
}
PANELS = ("h00", "h10", "h01", "h11", "hwrong10", "hwrong11")
CANDIDATES = ("additive", "linear", "quadratic", "rff")
COMPRESSED_RANK = 16
RFF_WIDTH = 256
RIDGE = 1.0e-6
SELECTION_DRAWS = 16384
POPULATION_PASS_MAX = 0.050
POPULATION_EARLIER_FAIL_MIN = 0.150
GLOBAL_ERROR_BUDGET = 0.01
MAX_REGISTERED_EVENTS = sum(config.layers for config in ARCHITECTURES.values()) * REPLICATES
CERTIFICATE_RADIUS = math.sqrt(
    math.log(2.0 * MAX_REGISTERED_EVENTS * len(CANDIDATES) / GLOBAL_ERROR_BUDGET)
    / (2.0 * SELECTION_DRAWS)
)
ROBUST_MARGIN_MULTIPLIER = 2.0

THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "population_pass_max": POPULATION_PASS_MAX,
    "population_earlier_fail_min": POPULATION_EARLIER_FAIL_MIN,
    "certificate_radius": CERTIFICATE_RADIUS,
    "certificate_false_authorizations_max": 0,
    "robust_coverage_min": 0.90,
    "ambiguous_abstention_min": 1.0,
    "rescue_authorization_upper_max": 0.025,
    "minimum_rescue_events_per_model": 1,
    "minimum_certified_classes": 1,
    "causal_state_relative_error_max": 0.30,
    "causal_output_cosine_min": 0.90,
    "causal_correct_accuracy_min": 0.95,
    "wrong_identity_accuracy_min": 0.95,
    "wrong_false_target_max": 0.05,
    "oracle_patch_accuracy_min": 0.999,
    "reverse_block_accuracy_min": 0.999,
    "breadth_models_min": 4,
    "breadth_per_depth_min": 1,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def make_factorial_world(
    c0: int,
    target: int,
    shift0: int,
    shift1: int,
    order: list[int],
    partition: str,
    row_id: str,
) -> dict[str, Any]:
    remaining = [value for value in range(4) if value not in (c0, target)]
    c1, wrong = remaining
    panel_specs = {
        "h00": ([c0, c1], shift0),
        "h10": ([target, c1], shift0),
        "h01": ([c0, c1], shift1),
        "h11": ([target, c1], shift1),
        "hwrong10": ([wrong, c1], shift0),
        "hwrong11": ([wrong, c1], shift1),
    }
    ids = {name: build_sequence(1, codes, shift, order)[0] for name, (codes, shift) in panel_specs.items()}
    answers = {name: (codes[0] + shift) % 4 for name, (codes, shift) in panel_specs.items()}
    row = {
        "row_id": row_id,
        "partition": partition,
        "source_code": c0,
        "target_code": target,
        "wrong_code": wrong,
        "context_code": c1,
        "shift0": shift0,
        "shift1": shift1,
        "codebook_order": order,
        "answers": answers,
        **{f"{name}_ids": values for name, values in ids.items()},
    }
    row["row_digest"] = digest(row)
    return row


def enumerate_oracle_worlds() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    orders = [list(values) for values in itertools.permutations(range(4))]
    index = 0
    for c0 in range(4):
        for target in range(4):
            if target == c0:
                continue
            for shift0 in range(4):
                for shift1 in range(4):
                    if shift1 == shift0:
                        continue
                    for order in orders:
                        rows.append(make_factorial_world(c0, target, shift0, shift1, order, "oracle", f"o{index:04d}"))
                        index += 1
    if len(rows) != PARTITION_COUNTS["oracle"]:
        raise RuntimeError(f"oracle universe drift: {len(rows)}")
    return rows


def sample_worlds(partition: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        c0 = int(rng.integers(4))
        target = int(rng.choice([value for value in range(4) if value != c0]))
        shift0 = int(rng.integers(4))
        shift1 = int(rng.choice([value for value in range(4) if value != shift0]))
        order = rng.permutation(4).astype(int).tolist()
        rows.append(make_factorial_world(c0, target, shift0, shift1, order, partition, f"{partition[0]}{index:04d}"))
    return rows


def make_material() -> list[dict[str, Any]]:
    rows = enumerate_oracle_worlds()
    for partition in ("discovery", "donor", "causal_selection", "confirmation"):
        rows.extend(sample_worlds(partition, PARTITION_COUNTS[partition], MATERIAL_SEEDS[partition]))
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dependency = read_json(PHASE1264_FINAL)
    audit = read_json(PHASE1264_AUDIT)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1266.c014.shape_safe_free_transformer_population_certificate.v1",
        "claim_type": "free_transformer_response_type_certificate_external_validity",
        "question": "Do freely trained same-executor Transformers contain layerwise factorial response objects that the calibrated finite-population certificate can type and causally replay across depth?",
        "phase1264_dependency": {
            "passed": dependency.get("passed"),
            "final": file_sha256(PHASE1264_FINAL),
            "audit_passed": audit.get("all_checks_passed"),
            "audit": file_sha256(PHASE1264_AUDIT),
        },
        "phase1265_invalid_predecessor": {
            "status": read_json(PHASE1265_INVALID).get("status"),
            "record": file_sha256(PHASE1265_INVALID),
            "repair_scope": "dimension-adaptive RFF only; all scientific thresholds and resource ceilings unchanged",
        },
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "partitions": PARTITION_COUNTS,
        "panels": list(PANELS),
        "world_digest": digest([{"row_id": row["row_id"], "partition": row["partition"], "row_digest": row["row_digest"]} for row in rows]),
        "factorial_object": {
            "A0": "H10-H00 at one answer-boundary layer event",
            "B": "H01-H00 for a later shift/codebook change",
            "A1": "H11-H01 under the changed shift",
            "J": "A1-A0",
        },
        "candidate_order": list(CANDIDATES) + ["abstain"],
        "candidate_camera": {
            "input_compression": f"separate discovery SVD bases of rank at most {COMPRESSED_RANK} for A0 and B",
            "output_compression": f"discovery SVD basis of rank at most {COMPRESSED_RANK} for J",
            "classes": ["additive", "linear", "quadratic cross-features", f"RFF width {RFF_WIDTH}", "abstain"],
            "truth": "exact bounded risk over all 3456 registered worlds",
            "selection": f"{SELECTION_DRAWS} uniform with-replacement draws from the oracle universe",
            "radius": CERTIFICATE_RADIUS,
        },
        "thresholds": THRESHOLDS,
        "causal_confirmation": {
            "selection": "among rescue-authorized events, retain only layers where exact answer-state patch and reverse block pass on causal_selection; freeze earliest and latest",
            "correct": "fit the same selected class on an independent donor partition and add predicted A1 to H01",
            "wrong": "apply the donor compiler to a different source identity",
            "oracle_sentinel": "replace H01 layer state with exact H11 state",
            "reverse_block": "replace H11 layer state with exact H01 state",
        },
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 1.5},
        "hard_stops": [
            "Behavior-unqualified models and certificate abstentions remain in the denominator.",
            "Exact population risks cannot select a class or layer; selection uses sample bounds only.",
            "Confirmation cannot change class, layer, rank, threshold, seed or architecture.",
            "A pass is restricted to this synthetic cyclic-code task and small free Transformers.",
            "Failure closes C014 before pretrained models; pass authorizes only a new pretrained contract.",
        ],
        "forbidden_claims": ["natural language mechanism", "unique physical circuit", "Qwen3", "cross-model semantic invariant", "new mathematics"],
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "contract": file_sha256(CONTRACT),
            "task": file_sha256(ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"),
            "same_executor": file_sha256(ROOT / "tests/glm5/phase1255_c008_same_executor_edge_external_validity.py"),
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
        "precision": "fp32 same explicit executor; fp64 candidate fit and risk certificate",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    dependency = read_json(PHASE1264_FINAL)
    audit = read_json(PHASE1264_AUDIT)
    if not dependency.get("passed") or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1264 did not authorize free-network external validity")
    if not AUDITOR.exists() or not CONTRACT.exists():
        raise RuntimeError("auditor and contract must exist before preregistration")
    rows = make_material()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows), "partitions": PARTITION_COUNTS, "models": len(MODEL_SEEDS)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["protocol_digest"] != expected["protocol_digest"] or protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("protocol or source drift")
    counts = {name: sum(row["partition"] == name for row in rows) for name in PARTITION_COUNTS}
    if counts != PARTITION_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material row digest mismatch")
    return protocol, rows


def svd_basis(samples: torch.Tensor, rank: int = COMPRESSED_RANK) -> torch.Tensor:
    values = samples.double()
    _u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 1.0e-12:
        return torch.zeros((values.shape[1], 0), dtype=torch.float64, device=values.device)
    effective = min(rank, int(torch.sum(singular > singular[0] * 1.0e-7).item()))
    return vh[:effective].T.contiguous()


def rff_parameters(input_dimension: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1_266_077 + 97 * input_dimension)
    omega = torch.randn((input_dimension, RFF_WIDTH), generator=generator, dtype=torch.float64, device=device) * 0.8
    phase = torch.rand((RFF_WIDTH,), generator=generator, dtype=torch.float64, device=device) * (2.0 * math.pi)
    return omega, phase


def feature_matrix(name: str, za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    x = torch.cat((za, zb), dim=1)
    ones = torch.ones((x.shape[0], 1), dtype=torch.float64, device=x.device)
    linear = torch.cat((ones, x), dim=1)
    if name == "linear":
        return linear
    cross = torch.einsum("bi,bj->bij", za, zb).reshape(x.shape[0], -1)
    quadratic = torch.cat((linear, cross), dim=1)
    if name == "quadratic":
        return quadratic
    if name == "rff":
        omega, phase = rff_parameters(x.shape[1], x.device)
        phi = math.sqrt(1.0 / RFF_WIDTH) * torch.cat((torch.sin(x @ omega + phase), torch.cos(x @ omega + phase)), dim=1)
        return torch.cat((quadratic, phi), dim=1)
    raise ValueError(name)


def ridge_fit(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = features.double(), target.double()
    gram = x.T @ x
    scale = float(torch.trace(gram).item()) / max(1, gram.shape[0])
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=x.device)
    return torch.linalg.solve(gram + RIDGE * max(scale, 1.0) * eye, x.T @ y)


def fit_layer_candidates(a0: torch.Tensor, b: torch.Tensor, a1: torch.Tensor) -> dict[str, dict[str, Any]]:
    a_basis = svd_basis(a0)
    b_basis = svd_basis(b)
    j = a1 - a0
    j_basis = svd_basis(j)
    za, zb, zj = a0.double() @ a_basis, b.double() @ b_basis, j.double() @ j_basis
    models: dict[str, dict[str, Any]] = {"additive": {"name": "additive"}}
    for name in CANDIDATES[1:]:
        models[name] = {
            "name": name,
            "a_basis": a_basis,
            "b_basis": b_basis,
            "j_basis": j_basis,
            "weights": ridge_fit(feature_matrix(name, za, zb), zj),
        }
    return models


def predict_candidate(model: dict[str, Any], a0: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if model["name"] == "additive":
        return a0.double()
    za = a0.double() @ model["a_basis"]
    zb = b.double() @ model["b_basis"]
    zj = feature_matrix(model["name"], za, zb) @ model["weights"]
    return a0.double() + zj @ model["j_basis"].T


def bounded_loss_vector(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    error = torch.sum((predicted.double() - truth.double()) ** 2, dim=1)
    scale = torch.sum(truth.double() ** 2, dim=1) + 0.25
    return torch.clamp(error / scale, min=0.0, max=1.0)


def select_point(risks: dict[str, float]) -> tuple[str, str]:
    passing = [name for name in CANDIDATES if risks[name] <= POPULATION_PASS_MAX]
    if not passing:
        return "abstain", "out_of_library"
    candidate = passing[0]
    index = CANDIDATES.index(candidate)
    if any(risks[name] < POPULATION_EARLIER_FAIL_MIN for name in CANDIDATES[:index]):
        return "abstain", "equivalence_overlap"
    return candidate, "population_margin_rule"


def confidence_bounds(sample_risks: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "point": risk,
            "lower": max(0.0, risk - CERTIFICATE_RADIUS),
            "upper": min(1.0, risk + CERTIFICATE_RADIUS),
        }
        for name, risk in sample_risks.items()
    }


def select_certificate(bounds: dict[str, dict[str, float]]) -> tuple[str, str]:
    passing = [name for name in CANDIDATES if bounds[name]["upper"] <= POPULATION_PASS_MAX]
    if not passing:
        return "abstain", "pass_not_certified"
    candidate = passing[0]
    index = CANDIDATES.index(candidate)
    if any(bounds[name]["lower"] < POPULATION_EARLIER_FAIL_MIN for name in CANDIDATES[:index]):
        return "abstain", "earlier_failure_not_certified"
    return candidate, "simultaneously_certified"


def partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def capture_partition(model, rows: list[dict[str, Any]], device: torch.device, batch_size: int = 512) -> dict[str, Any]:
    state_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in PANELS}
    logit_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in PANELS}
    max_gap = 0.0
    accuracies: dict[str, list[torch.Tensor]] = {name: [] for name in PANELS}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for panel in PANELS:
                ids = torch.tensor([row[f"{panel}_ids"] for row in batch], device=device)
                native = model(ids)
                logits, states = executor.explicit_residual_forward(model, ids, capture=True)
                assert states is not None
                max_gap = max(max_gap, float(torch.max(torch.abs(native.float() - logits.float())).item()))
                expected = torch.tensor([row["answers"][panel] for row in batch], device=device)
                predicted = torch.argmax(logits[:, -1, CANDIDATE_SLICE], dim=-1)
                accuracies[panel].append((predicted == expected).float())
                state_chunks[panel].append(states)
                logit_chunks[panel].append(logits[:, -1, :])
    return {
        "states": {name: torch.cat(chunks, dim=0) for name, chunks in state_chunks.items()},
        "logits": {name: torch.cat(chunks, dim=0) for name, chunks in logit_chunks.items()},
        "executor_gap": max_gap,
        "accuracies": {name: float(torch.cat(chunks).mean().item()) for name, chunks in accuracies.items()},
    }


def factorial_at_layer(capture: dict[str, Any], layer: int) -> dict[str, torch.Tensor]:
    states = capture["states"]
    h00, h10, h01, h11 = (states[name][:, layer] for name in ("h00", "h10", "h01", "h11"))
    return {"A0": h10 - h00, "B": h01 - h00, "A1": h11 - h01}


def selection_indices(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 7_777_019)
    return torch.randint(0, PARTITION_COUNTS["oracle"], (SELECTION_DRAWS,), generator=generator, device=device)


def centered_last(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, CANDIDATE_SLICE].double()
    return values - values.mean(dim=-1, keepdim=True)


def effect_metrics(response: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p, t = response.reshape(-1), target.reshape(-1)
    pn = torch.linalg.vector_norm(p).clamp_min(1.0e-12)
    tn = torch.linalg.vector_norm(t).clamp_min(1.0e-12)
    return {
        "cosine": float((torch.dot(p, t) / (pn * tn)).item()),
        "relative_error": float((torch.linalg.vector_norm(p - t) / tn).item()),
    }


Action = Callable[[torch.Tensor], torch.Tensor]


def causal_confirmation(
    model,
    layer: int,
    selected_class: str,
    donor_capture: dict[str, Any],
    confirmation_capture: dict[str, Any],
    confirmation_rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    donor = factorial_at_layer(donor_capture, layer)
    donor_models = fit_layer_candidates(donor["A0"], donor["B"], donor["A1"])
    confirmation = factorial_at_layer(confirmation_capture, layer)
    states = confirmation_capture["states"]
    predicted = predict_candidate(donor_models[selected_class], confirmation["A0"], confirmation["B"])
    wrong_a0 = states["hwrong10"][:, layer] - states["h00"][:, layer]
    predicted_wrong = predict_candidate(donor_models[selected_class], wrong_a0, confirmation["B"])
    h01_ids = torch.tensor([row["h01_ids"] for row in confirmation_rows], device=device)
    h11_ids = torch.tensor([row["h11_ids"] for row in confirmation_rows], device=device)
    h01_state = states["h01"][:, layer]
    h11_state = states["h11"][:, layer]

    with torch.inference_mode():
        correct_logits, _ = executor.explicit_residual_forward(
            model,
            h01_ids,
            actions={layer: lambda current: current + predicted.to(current.dtype)},
            capture=False,
        )
        wrong_logits, _ = executor.explicit_residual_forward(
            model,
            h01_ids,
            actions={layer: lambda current: current + predicted_wrong.to(current.dtype)},
            capture=False,
        )
        oracle_logits, _ = executor.explicit_residual_forward(
            model,
            h01_ids,
            actions={layer: lambda _current: h11_state},
            capture=False,
        )
        reverse_logits, _ = executor.explicit_residual_forward(
            model,
            h11_ids,
            actions={layer: lambda _current: h01_state},
            capture=False,
        )

    base_output = centered_last(confirmation_capture["logits"]["h01"])
    target_output = centered_last(confirmation_capture["logits"]["h11"])
    correct_response = executor.centered(correct_logits) - base_output
    target_response = target_output - base_output
    target_answers = torch.tensor([row["answers"]["h11"] for row in confirmation_rows], device=device)
    wrong_answers = torch.tensor([row["answers"]["hwrong11"] for row in confirmation_rows], device=device)
    base_answers = torch.tensor([row["answers"]["h01"] for row in confirmation_rows], device=device)
    correct_pred = torch.argmax(correct_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    wrong_pred = torch.argmax(wrong_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    oracle_pred = torch.argmax(oracle_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    reverse_pred = torch.argmax(reverse_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    natural_a1 = confirmation["A1"].double()
    state_error = float(
        (torch.linalg.vector_norm(predicted - natural_a1) / torch.linalg.vector_norm(natural_a1).clamp_min(1.0e-12)).item()
    )
    metrics = {
        "layer": layer,
        "selected_class": selected_class,
        "cases": len(confirmation_rows),
        "state_relative_error": state_error,
        "correct_output": effect_metrics(correct_response, target_response),
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
    config: ModelConfig,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    key = model_key(architecture, replicate)
    seed = MODEL_SEEDS[key]
    set_seed(seed)
    model, training = task_module.train_model(config, seed, device)
    captures = {
        partition: capture_partition(model, partition_rows(rows, partition), device)
        for partition in PARTITION_COUNTS
    }
    natural_min = min(value for capture in captures.values() for value in capture["accuracies"].values())
    executor_gap = max(capture["executor_gap"] for capture in captures.values())
    behavior_qualified = (
        min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], natural_min)
        >= THRESHOLDS["behavior_accuracy_min"]
        and executor_gap <= THRESHOLDS["executor_gap_max"]
    )
    base_result = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "training": training,
        "natural_accuracy_min": natural_min,
        "executor_gap": executor_gap,
        "behavior_qualified": behavior_qualified,
    }
    if not behavior_qualified:
        return {**base_result, "event_ledger": [], "selected_events": [], "causal_confirmations": [], "passed": False}

    selection = selection_indices(seed, device)
    event_models: dict[int, dict[str, dict[str, Any]]] = {}
    event_ledger = []
    for layer in range(config.layers):
        discovery = factorial_at_layer(captures["discovery"], layer)
        oracle = factorial_at_layer(captures["oracle"], layer)
        models = fit_layer_candidates(discovery["A0"], discovery["B"], discovery["A1"])
        event_models[layer] = models
        losses = {
            name: bounded_loss_vector(predict_candidate(candidate, oracle["A0"], oracle["B"]), oracle["A1"])
            for name, candidate in models.items()
        }
        population_risks = {name: float(values.mean().item()) for name, values in losses.items()}
        sample_risks = {name: float(values[selection].mean().item()) for name, values in losses.items()}
        exact_class, exact_reason = select_point(population_risks)
        point_class, _point_reason = select_point(sample_risks)
        bounds = confidence_bounds(sample_risks)
        certificate_class, certificate_reason = select_certificate(bounds)
        robust = False
        if exact_class != "abstain":
            index = CANDIDATES.index(exact_class)
            margin = ROBUST_MARGIN_MULTIPLIER * CERTIFICATE_RADIUS
            robust = (
                population_risks[exact_class] <= POPULATION_PASS_MAX - margin
                and all(population_risks[name] >= POPULATION_EARLIER_FAIL_MIN + margin for name in CANDIDATES[:index])
            )
        rescue_authorized = (
            certificate_class != "abstain"
            and bounds[certificate_class]["upper"] <= THRESHOLDS["rescue_authorization_upper_max"]
        )
        event_ledger.append(
            {
                "layer": layer,
                "population_risks": population_risks,
                "exact_class": exact_class,
                "exact_reason": exact_reason,
                "robust_actionable": robust,
                "sample_risks": sample_risks,
                "point_class": point_class,
                "confidence_bounds": bounds,
                "certificate_class": certificate_class,
                "certificate_reason": certificate_reason,
                "rescue_authorized": rescue_authorized,
            }
        )

    false_authorizations = sum(
        event["certificate_class"] != "abstain" and event["certificate_class"] != event["exact_class"]
        for event in event_ledger
    )
    robust_events = [event for event in event_ledger if event["robust_actionable"]]
    robust_coverage = sum(event["certificate_class"] == event["exact_class"] for event in robust_events) / max(1, len(robust_events))
    ambiguous_events = [event for event in event_ledger if event["exact_class"] == "abstain"]
    ambiguous_abstention = sum(event["certificate_class"] == "abstain" for event in ambiguous_events) / max(1, len(ambiguous_events))
    causal_selection_rows = partition_rows(rows, "causal_selection")
    causal_selection = {}
    for event in event_ledger:
        if not event["rescue_authorized"]:
            continue
        layer = event["layer"]
        metrics = causal_confirmation(
            model,
            layer,
            event["certificate_class"],
            captures["donor"],
            captures["causal_selection"],
            causal_selection_rows,
            device,
        )
        causal_selection[layer] = {
            "oracle_patch_accuracy": metrics["oracle_patch_accuracy"],
            "reverse_block_accuracy": metrics["reverse_block_accuracy"],
            "answer_state_causally_admissible": (
                metrics["oracle_patch_accuracy"] >= THRESHOLDS["oracle_patch_accuracy_min"]
                and metrics["reverse_block_accuracy"] >= THRESHOLDS["reverse_block_accuracy_min"]
            ),
        }
        event["causal_selection_sentinel"] = causal_selection[layer]
    rescue_layers = [
        layer for layer, metrics in causal_selection.items() if metrics["answer_state_causally_admissible"]
    ]
    selected_layers = []
    if rescue_layers:
        selected_layers.append(rescue_layers[0])
        if rescue_layers[-1] != rescue_layers[0]:
            selected_layers.append(rescue_layers[-1])
    confirmation_rows = partition_rows(rows, "confirmation")
    causal = [
        causal_confirmation(
            model,
            layer,
            event_ledger[layer]["certificate_class"],
            captures["donor"],
            captures["confirmation"],
            confirmation_rows,
            device,
        )
        for layer in selected_layers
    ]
    passed = (
        false_authorizations == 0
        and robust_coverage >= THRESHOLDS["robust_coverage_min"]
        and ambiguous_abstention >= THRESHOLDS["ambiguous_abstention_min"]
        and len(selected_layers) >= THRESHOLDS["minimum_rescue_events_per_model"]
        and all(item["passed"] for item in causal)
    )
    return {
        **base_result,
        "event_ledger": event_ledger,
        "false_authorizations": false_authorizations,
        "robust_events": len(robust_events),
        "robust_coverage": robust_coverage,
        "ambiguous_events": len(ambiguous_events),
        "ambiguous_abstention": ambiguous_abstention,
        "selected_events": selected_layers,
        "causal_selection": causal_selection,
        "causal_confirmations": causal,
        "passed": passed,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in rows if row["behavior_qualified"]]
    passed = [row for row in rows if row["passed"]]
    all_events = [event for row in rows for event in row.get("event_ledger", [])]
    certified_classes = sorted({event["certificate_class"] for event in all_events if event["certificate_class"] != "abstain"})
    per_depth = {
        architecture: {
            "qualified": sum(row["architecture"] == architecture and row["behavior_qualified"] for row in rows),
            "passed": sum(row["architecture"] == architecture and row["passed"] for row in rows),
        }
        for architecture in ARCHITECTURES
    }
    breadth = (
        len(passed) >= THRESHOLDS["breadth_models_min"]
        and all(value["passed"] >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values())
    )
    gates = {
        "G-BEHAVIOR": len(qualified) == len(rows),
        "G-ZERO-FALSE-AUTHORIZATION": sum(row.get("false_authorizations", 0) for row in rows) == 0,
        "G-ROBUST-COVERAGE": bool(qualified) and all(row["robust_coverage"] >= THRESHOLDS["robust_coverage_min"] for row in qualified),
        "G-AMBIGUOUS-ABSTENTION": bool(qualified) and all(row["ambiguous_abstention"] >= THRESHOLDS["ambiguous_abstention_min"] for row in qualified),
        "G-NONTRIVIAL-EVENTS": bool(qualified) and all(len(row["selected_events"]) >= THRESHOLDS["minimum_rescue_events_per_model"] for row in qualified),
        "G-CLASS-DIVERSITY": (
            len(certified_classes) >= THRESHOLDS["minimum_certified_classes"]
            and any(name != "additive" for name in certified_classes)
        ),
        "G-INDEPENDENT-DONOR-CAUSAL": breadth and all(all(item["passed"] for item in row["causal_confirmations"]) for row in passed),
        "G-CROSS-DEPTH-BREADTH": breadth,
    }
    return {
        "models": len(rows),
        "qualified": len(qualified),
        "passed_models": len(passed),
        "per_depth": per_depth,
        "events": len(all_events),
        "certified_classes": certified_classes,
        "certificate_false_authorizations": sum(row.get("false_authorizations", 0) for row in rows),
        "point_false_authorizations": sum(
            event["point_class"] != "abstain" and event["point_class"] != event["exact_class"] for event in all_events
        ),
        "robust_events": sum(event["robust_actionable"] for event in all_events),
        "certified_events": sum(event["certificate_class"] != "abstain" for event in all_events),
        "rescue_authorized_events": sum(event["rescue_authorized"] for event in all_events),
        "gates": gates,
        "passed": all(gates.values()),
    }


def probe(device_name: str) -> None:
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA")
    rows = make_material()
    original = dict(MODEL_SEEDS)
    config = ModelConfig(layers=3, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22)
    key = "development3_r0"
    MODEL_SEEDS[key] = 1_266_301_001
    result = run_model("development3", 0, config, rows, torch.device("cuda"))
    MODEL_SEEDS.clear()
    MODEL_SEEDS.update(original)
    atomic_json(PROBE, result)
    print(
        canonical_json(
            {
                "behavior_qualified": result["behavior_qualified"],
                "natural_accuracy_min": result["natural_accuracy_min"],
                "false_authorizations": result.get("false_authorizations"),
                "robust_coverage": result.get("robust_coverage"),
                "event_classes": [event["certificate_class"] for event in result.get("event_ledger", [])],
                "selected_events": result.get("selected_events"),
                "causal": result.get("causal_confirmations"),
                "passed": result["passed"],
            }
        )
    )


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
            print(canonical_json({"completed": len(results), "total": len(MODEL_SEEDS), "model": result["model_key"], "passed": result["passed"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    write_jsonl(MODELS, results)
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
            "new_pretrained_contract": summary["passed"],
            "qwen3_automatic": False,
            "glm4": False,
            "ds7b": False,
        },
        "claim_boundary": "Free 4/6/8-layer same-executor Transformers on one exhaustive cyclic-code factorial universe. A pass concerns response-type certification and independent-donor causal sufficiency, not natural language or a unique circuit.",
        "protocol_digest": protocol["protocol_digest"],
        "models_hash": file_sha256(MODELS),
        "run_digest": digest(results),
    }
    final["final_digest"] = digest(final)
    atomic_json(ANALYSIS, {"created_at_utc": utc_now(), "gates": final["gates"], "per_depth": final["per_depth"]})
    atomic_json(FINAL, final)
    print(canonical_json({"passed": final["passed"], "gates": final["gates"], "per_depth": final["per_depth"], "certified_classes": final["certified_classes"]}))


def run_auditor(mode: str) -> None:
    subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze", "audit-pre", "audit-final"))
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
    elif args.command == "audit-pre":
        run_auditor("pre")
    else:
        run_auditor("final")


if __name__ == "__main__":
    main()
