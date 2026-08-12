#!/usr/bin/env python3
"""Calibrate a gauge-equivariant causal-use camera on matched systems.

Phase1155 recovered a coarse representation-dependency quotient, but a factor
can be represented without being used by the observed mediator.  This phase
constructs eight systems that share, within each replicate:

* exactly the same mediator states,
* exactly the same clean input-output function, and
* different downstream mediator-use masks for row, column, and context.

Factors not read from the mediator travel through an explicit raw-input bypass.
The bypass is a calibration scaffold, not a claim about natural networks.  A
matched donor differs from the receiver in exactly one semantic factor.  The
full donor-receiver state displacement transforms equivariantly under every
declared invertible linear gauge, so transporting that displacement tests
whether the downstream output actually uses that mediator factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import phase1153_blind_algorithm_coverage as coverage
import phase1155_learning_gauge_quotient_calibration as source


PHASE = 1156
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1156_causal_use_quotient_calibration"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1155_learning_gauge_quotient_calibration"
SPLITS = ("discovery", "confirmation")
N_ROWS = 4
N_COLS = 4
N_CONTEXTS = 2
N_SITES = 5
STATE_DIM = 16
TOTAL_DIM = N_SITES * STATE_DIM
FACTORS = ("row", "col", "context")
USE_MASKS = {
    "000_none": (0, 0, 0),
    "001_context": (0, 0, 1),
    "010_col": (0, 1, 0),
    "011_col_context": (0, 1, 1),
    "100_row": (1, 0, 0),
    "101_row_context": (1, 0, 1),
    "110_row_col": (1, 1, 0),
    "111_all": (1, 1, 1),
}
GAUGES = ("identity", "site_gl", "cross_site_orthogonal", "cross_site_gl")
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
ALGORITHMS = ("state_gram", "dependency_rank_quotient", "matched_transport_causal_use")
CANDIDATE = "matched_transport_causal_use"
LOGIT_SCALE = 20.0
GAUGE_CONDITION = 5.0
RANK_RELATIVE_TOLERANCE = 1e-6
RANK_ABSOLUTE_TOLERANCE = 1e-8
THRESHOLDS = {
    "clean_accuracy_min": 1.0,
    "clean_target_probability_min": 0.9999,
    "clean_function_abs_error_max": 1e-10,
    "finite_fraction_min": 1.0,
    "representation_mask_match_fraction_min": 1.0,
    "dependency_gauge_match_fraction_min": 1.0,
    "candidate_gauge_abs_error_max": 1e-10,
    "alpha_zero_tv_max": 1e-12,
    "matched_null_tv_max": 1e-12,
    "used_donor_probability_min": 0.9999,
    "unused_receiver_probability_min": 0.9999,
    "expected_hybrid_probability_min": 0.9999,
    "discovery_accuracy_min": 1.0,
    "discovery_min_label_accuracy_min": 1.0,
    "confirmation_accuracy_min": 1.0,
    "confirmation_min_label_accuracy_min": 1.0,
    "candidate_gauge_accuracy_gap_max": 0.0,
    "representation_control_accuracy_max": 0.125,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_for(split: str, replicate: int) -> int:
    return (115610 if split == "discovery" else 115690) + int(replicate) * 1009


def make_orthogonal(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def make_invertible(size: int, seed: int, condition: float = GAUGE_CONDITION) -> np.ndarray:
    left = make_orthogonal(size, seed)
    right = make_orthogonal(size, seed + 7919)
    singular = np.geomspace(condition ** -0.5, condition ** 0.5, size)
    return left @ np.diag(singular) @ right.T


def gauge_matrix(gauge: str, seed: int) -> np.ndarray:
    if gauge == "identity":
        return np.eye(TOTAL_DIM, dtype=np.float64)
    if gauge == "site_gl":
        value = np.zeros((TOTAL_DIM, TOTAL_DIM), dtype=np.float64)
        for site in range(N_SITES):
            start = site * STATE_DIM
            value[start : start + STATE_DIM, start : start + STATE_DIM] = make_invertible(
                STATE_DIM, seed + 1013 * site
            )
        return value
    if gauge == "cross_site_orthogonal":
        return np.kron(make_orthogonal(N_SITES, seed), np.eye(STATE_DIM, dtype=np.float64))
    if gauge == "cross_site_gl":
        return np.kron(make_invertible(N_SITES, seed), make_orthogonal(STATE_DIM, seed + 1543))
    raise ValueError(gauge)


def all_inputs() -> list[tuple[int, int, int]]:
    return [(row, col, context) for context in range(N_CONTEXTS) for row in range(N_ROWS) for col in range(N_COLS)]


def target_indices(inputs: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor([(context * N_ROWS + row) * N_COLS + col for row, col, context in inputs], dtype=torch.long, device=device)


def changed_inputs(inputs: list[tuple[int, int, int]], factor: str) -> list[tuple[int, int, int]]:
    changed = []
    for row, col, context in inputs:
        if factor == "row":
            changed.append(((row + 1) % N_ROWS, col, context))
        elif factor == "col":
            changed.append((row, (col + 1) % N_COLS, context))
        elif factor == "context":
            changed.append((row, col, 1 - context))
        else:
            raise ValueError(factor)
    return changed


class BaseRepresentation:
    """One fixed factorized representation reused by all eight use masks."""

    def __init__(self, seed: int, device: torch.device) -> None:
        self.seed = int(seed)
        self.device = device
        self.row_embedding = torch.tensor(make_orthogonal(STATE_DIM, seed + 11)[:N_ROWS], dtype=torch.float64, device=device)
        self.col_embedding = torch.tensor(make_orthogonal(STATE_DIM, seed + 23)[:N_COLS], dtype=torch.float64, device=device)
        self.context_embedding = torch.tensor(
            make_orthogonal(STATE_DIM, seed + 37)[:N_CONTEXTS], dtype=torch.float64, device=device
        )
        rng = np.random.default_rng(seed + 53)
        self.joint_nuisance = torch.tensor(
            rng.normal(scale=0.35, size=(N_CONTEXTS, N_ROWS, N_COLS, STATE_DIM)), dtype=torch.float64, device=device
        )
        self.row_nuisance = torch.tensor(rng.normal(scale=0.25, size=(N_ROWS, STATE_DIM)), dtype=torch.float64, device=device)
        self.col_nuisance = torch.tensor(rng.normal(scale=0.25, size=(N_COLS, STATE_DIM)), dtype=torch.float64, device=device)
        self.context_nuisance = torch.tensor(
            rng.normal(scale=0.25, size=(N_CONTEXTS, STATE_DIM)), dtype=torch.float64, device=device
        )

    def export(self) -> dict[str, np.ndarray]:
        return {
            "row_embedding": self.row_embedding.detach().cpu().numpy(),
            "col_embedding": self.col_embedding.detach().cpu().numpy(),
            "context_embedding": self.context_embedding.detach().cpu().numpy(),
            "joint_nuisance": self.joint_nuisance.detach().cpu().numpy(),
            "row_nuisance": self.row_nuisance.detach().cpu().numpy(),
            "col_nuisance": self.col_nuisance.detach().cpu().numpy(),
            "context_nuisance": self.context_nuisance.detach().cpu().numpy(),
        }

    def logical_states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        rows = torch.tensor([row for row, _col, _context in inputs], dtype=torch.long, device=self.device)
        cols = torch.tensor([col for _row, col, _context in inputs], dtype=torch.long, device=self.device)
        contexts = torch.tensor([context for _row, _col, context in inputs], dtype=torch.long, device=self.device)
        states = torch.zeros((len(inputs), N_SITES, STATE_DIM), dtype=torch.float64, device=self.device)
        states[:, 0] = self.row_embedding[rows]
        states[:, 1] = self.col_embedding[cols]
        states[:, 2] = self.context_embedding[contexts]
        states[:, 3] = self.joint_nuisance[contexts, rows, cols]
        states[:, 4] = self.row_nuisance[rows] + self.col_nuisance[cols] + self.context_nuisance[contexts]
        return states

    def decoded_factor_scores(self, logical: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "row": logical[:, 0] @ self.row_embedding.T,
            "col": logical[:, 1] @ self.col_embedding.T,
            "context": logical[:, 2] @ self.context_embedding.T,
        }


class GaugeOracle:
    def __init__(self, base: BaseRepresentation, gauge: str, gauge_seed: int) -> None:
        self.base = base
        self.device = base.device
        self.gauge = gauge
        self.matrix = torch.tensor(gauge_matrix(gauge, gauge_seed), dtype=torch.float64, device=self.device)
        self.inverse = torch.linalg.inv(self.matrix)

    def states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        logical = self.base.logical_states(inputs).reshape(len(inputs), -1)
        return (logical @ self.matrix.T).reshape(len(inputs), N_SITES, STATE_DIM)

    def decode(self, observed: torch.Tensor) -> torch.Tensor:
        flattened = observed.reshape(len(observed), -1)
        return (flattened @ self.inverse.T).reshape(len(observed), N_SITES, STATE_DIM)

    def output(
        self,
        observed: torch.Tensor,
        receivers: list[tuple[int, int, int]],
        use_mask: tuple[int, int, int],
    ) -> torch.Tensor:
        logical = self.decode(observed)
        mediated = self.base.decoded_factor_scores(logical)
        raw = {
            "row": torch.nn.functional.one_hot(
                torch.tensor([row for row, _col, _context in receivers], device=self.device), N_ROWS
            ).to(torch.float64),
            "col": torch.nn.functional.one_hot(
                torch.tensor([col for _row, col, _context in receivers], device=self.device), N_COLS
            ).to(torch.float64),
            "context": torch.nn.functional.one_hot(
                torch.tensor([context for _row, _col, context in receivers], device=self.device), N_CONTEXTS
            ).to(torch.float64),
        }
        probabilities = {}
        for index, factor in enumerate(FACTORS):
            scores = mediated[factor] if use_mask[index] else raw[factor]
            probabilities[factor] = torch.softmax(LOGIT_SCALE * scores, dim=1)
        joint = (
            probabilities["context"][:, :, None, None]
            * probabilities["row"][:, None, :, None]
            * probabilities["col"][:, None, None, :]
        )
        return joint.reshape(len(receivers), -1)


def factor_difference_matrices(states: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(states, dtype=np.float64).reshape(N_CONTEXTS, N_ROWS, N_COLS, -1)
    row = np.stack(
        [value[k, r, c] - value[k, 0, c] for k in range(N_CONTEXTS) for c in range(N_COLS) for r in range(1, N_ROWS)]
    )
    col = np.stack(
        [value[k, r, c] - value[k, r, 0] for k in range(N_CONTEXTS) for r in range(N_ROWS) for c in range(1, N_COLS)]
    )
    context = np.stack([value[1, r, c] - value[0, r, c] for r in range(N_ROWS) for c in range(N_COLS)])
    row_col = np.stack(
        [
            value[k, r, c] - value[k, r, 0] - value[k, 0, c] + value[k, 0, 0]
            for k in range(N_CONTEXTS)
            for r in range(1, N_ROWS)
            for c in range(1, N_COLS)
        ]
    )
    row_context = np.stack(
        [(value[1, r, c] - value[1, 0, c]) - (value[0, r, c] - value[0, 0, c]) for c in range(N_COLS) for r in range(1, N_ROWS)]
    )
    col_context = np.stack(
        [(value[1, r, c] - value[1, r, 0]) - (value[0, r, c] - value[0, r, 0]) for r in range(N_ROWS) for c in range(1, N_COLS)]
    )
    triple = np.stack(
        [
            (value[1, r, c] - value[1, r, 0] - value[1, 0, c] + value[1, 0, 0])
            - (value[0, r, c] - value[0, r, 0] - value[0, 0, c] + value[0, 0, 0])
            for r in range(1, N_ROWS)
            for c in range(1, N_COLS)
        ]
    )
    centered = value.reshape(-1, value.shape[-1]) - np.mean(value.reshape(-1, value.shape[-1]), axis=0, keepdims=True)
    return {
        "row": row,
        "col": col,
        "context": context,
        "row_col": row_col,
        "row_context": row_context,
        "col_context": col_context,
        "triple": triple,
        "centered": centered,
    }


def matrix_rank(matrix: np.ndarray, threshold: float) -> int:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return int(np.sum(singular > threshold))


def dependency_rank_feature(states: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    matrices = factor_difference_matrices(states.detach().cpu().numpy())
    first_order = np.concatenate([matrices["row"], matrices["col"], matrices["context"]], axis=0)
    singular = np.linalg.svd(first_order, compute_uv=False)
    scale = float(singular[0]) if len(singular) else 0.0
    threshold = max(RANK_ABSOLUTE_TOLERANCE, RANK_RELATIVE_TOLERANCE * scale)
    rank = {name: matrix_rank(matrix, threshold) for name, matrix in matrices.items()}
    rank["row_col_union"] = matrix_rank(np.concatenate([matrices["row"], matrices["col"]], axis=0), threshold)
    rank["row_context_union"] = matrix_rank(np.concatenate([matrices["row"], matrices["context"]], axis=0), threshold)
    rank["col_context_union"] = matrix_rank(np.concatenate([matrices["col"], matrices["context"]], axis=0), threshold)
    rank["all_first_order_union"] = matrix_rank(first_order, threshold)
    rank["row_col_intersection"] = rank["row"] + rank["col"] - rank["row_col_union"]
    rank["row_context_intersection"] = rank["row"] + rank["context"] - rank["row_context_union"]
    rank["col_context_intersection"] = rank["col"] + rank["context"] - rank["col_context_union"]
    names = (
        "row",
        "col",
        "context",
        "row_col",
        "row_context",
        "col_context",
        "triple",
        "centered",
        "row_col_union",
        "row_context_union",
        "col_context_union",
        "all_first_order_union",
        "row_col_intersection",
        "row_context_intersection",
        "col_context_intersection",
    )
    return np.asarray([rank[name] for name in names], dtype=np.float64), {
        "rank_threshold": threshold,
        "rank_scale": scale,
        "rank_names": list(names),
        "ranks": rank,
    }


def state_gram_feature(states: torch.Tensor) -> np.ndarray:
    value = states.detach().cpu().numpy().reshape(len(states), -1).astype(np.float64)
    value = value - np.mean(value, axis=0, keepdims=True)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    normalized = value / np.where(norms > 1e-12, norms, 1.0)
    gram = normalized @ normalized.T
    return gram[np.triu_indices(len(gram), k=1)]


def causal_use_feature(
    oracle: GaugeOracle,
    inputs: list[tuple[int, int, int]],
    states: torch.Tensor,
    use_mask: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = oracle.output(states, inputs, use_mask)
    receiver_targets = target_indices(inputs, oracle.device)
    batch = torch.arange(len(inputs), device=oracle.device)
    features: list[float] = []
    factors: dict[str, Any] = {}
    used_donor = []
    unused_receiver = []
    expected_hybrid = []
    null_tvs = []
    alpha_zero_tvs = []
    for factor_index, factor in enumerate(FACTORS):
        donors = changed_inputs(inputs, factor)
        donor_states = oracle.states(donors)
        delta = donor_states - states
        patched = oracle.output(states + delta, inputs, use_mask)
        alpha_zero = oracle.output(states + 0.0 * delta, inputs, use_mask)
        matched_null = oracle.output(states + (states - states), inputs, use_mask)
        donor_targets = target_indices(donors, oracle.device)
        donor_probability = float(torch.mean(patched[batch, donor_targets]).item())
        receiver_probability = float(torch.mean(patched[batch, receiver_targets]).item())
        total_variation = float(torch.mean(0.5 * torch.sum(torch.abs(patched - baseline), dim=1)).item())
        alpha_zero_tv = float(torch.mean(0.5 * torch.sum(torch.abs(alpha_zero - baseline), dim=1)).item())
        matched_null_tv = float(torch.mean(0.5 * torch.sum(torch.abs(matched_null - baseline), dim=1)).item())
        expected_targets = donor_targets if use_mask[factor_index] else receiver_targets
        expected_probability = float(torch.mean(patched[batch, expected_targets]).item())
        features.extend([donor_probability, receiver_probability, total_variation])
        factors[factor] = {
            "used": bool(use_mask[factor_index]),
            "donor_probability": donor_probability,
            "receiver_probability": receiver_probability,
            "total_variation": total_variation,
            "expected_hybrid_probability": expected_probability,
            "alpha_zero_tv": alpha_zero_tv,
            "matched_null_tv": matched_null_tv,
        }
        expected_hybrid.append(expected_probability)
        null_tvs.append(matched_null_tv)
        alpha_zero_tvs.append(alpha_zero_tv)
        if use_mask[factor_index]:
            used_donor.append(donor_probability)
        else:
            unused_receiver.append(receiver_probability)
    return np.asarray(features, dtype=np.float64), {
        "factors": factors,
        "used_donor_probability_min": min(used_donor) if used_donor else 1.0,
        "unused_receiver_probability_min": min(unused_receiver) if unused_receiver else 1.0,
        "expected_hybrid_probability_min": min(expected_hybrid),
        "alpha_zero_tv_max": max(alpha_zero_tvs),
        "matched_null_tv_max": max(null_tvs),
    }


def classification_metrics(predicted: list[str], truth: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    correct = [predicted[offset] == str(truth[index]["use_label"]) for offset, index in enumerate(indices)]
    labels = sorted({str(truth[index]["use_label"]) for index in indices})
    per_label = {}
    for label in labels:
        selected = [offset for offset, index in enumerate(indices) if str(truth[index]["use_label"]) == label]
        per_label[label] = float(np.mean([correct[offset] for offset in selected]))
    per_gauge = {}
    for gauge in GAUGES:
        selected = [offset for offset, index in enumerate(indices) if truth[index]["gauge"] == gauge]
        if selected:
            per_gauge[gauge] = float(np.mean([correct[offset] for offset in selected]))
    bit_accuracy = {}
    for bit_index, factor in enumerate(FACTORS):
        bit_accuracy[factor] = float(
            np.mean(
                [
                    int(predicted[offset][:3][bit_index]) == int(truth[index]["use_mask"][bit_index])
                    for offset, index in enumerate(indices)
                ]
            )
        )
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(per_label.values())),
        "per_label_accuracy": per_label,
        "per_gauge_accuracy": per_gauge,
        "gauge_accuracy_gap": float(max(per_gauge.values()) - min(per_gauge.values())) if per_gauge else 0.0,
        "factor_bit_accuracy": bit_accuracy,
        "count": len(indices),
    }


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to rewrite Phase1156 artifacts")
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks = {
        "source_dependency_quotient_confirmed": bool(source_final["coarse_dependency_quotient_confirmed"]),
        "source_audit_passed": bool(source_audit["all_checks_passed"]),
        "source_next_task_is_causal_use": "causal-use" in source_final["next_legal_task"],
        "same_representation_required_across_masks": True,
        "same_clean_function_required_across_masks": True,
        "matched_donor_changes_one_factor": True,
        "raw_bypass_is_explicit_calibration_scaffold": True,
        "candidate_predeclared": CANDIDATE == "matched_transport_causal_use",
        "confirmation_truth_forbidden_in_predict": True,
        "hyperedge_redundancy_gate_claims_forbidden": True,
        "free_transformer_and_pretrained_scan_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "matched factor-transport causal-use quotient calibration",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1155_digest": source_final["final_digest"],
        "source_phase1155_audit_digest": source_audit["audit_digest"],
        "factors": list(FACTORS),
        "use_masks": {label: list(mask) for label, mask in USE_MASKS.items()},
        "gauges": list(GAUGES),
        "replicates": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "algorithms": list(ALGORITHMS),
        "candidate": CANDIDATE,
        "logit_scale": LOGIT_SCALE,
        "gauge_condition": GAUGE_CONDITION,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "blind eight-way mediator-use-mask recovery under matched factor transport and four linear gauges",
        "negative_control_endpoint": "representation-only cameras remain at the balanced one-of-eight ceiling",
        "hard_stops": [
            "The raw-input bypass is a known-truth calibration scaffold and is not evidence for a bypass in a natural Transformer.",
            "A represented factor is upgraded to mediated causal use only under matched one-factor transport and matched null controls.",
            "This phase identifies three single-factor mediator-use bits only; it does not identify interaction hyperedges, redundancy, gates, or necessity.",
            "Passing this calibration does not authorize a free Transformer or pretrained-LLM scan.",
            "Confirmation predictions must be sealed before confirmation truth is read.",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    body = dict(protocol)
    protocol["protocol_digest"] = digest(body)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(checks.values()),
            "all_checks_passed": all(checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if digest(body) != stored or sha256_file(SCRIPT) != protocol["script_sha256"]:
        raise RuntimeError("Phase1156 frozen protocol mismatch")
    return protocol


def run_command(split: str) -> None:
    protocol = verify_protocol()
    if split == "confirmation":
        fit = read_json(OUT_ROOT / "analysis/fit.json")
        if not fit["confirmation_run_authorized"]:
            raise RuntimeError("confirmation run denied by discovery")
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    inputs = all_inputs()
    batch = torch.arange(len(inputs), device=device)
    receiver_targets = target_indices(inputs, device)
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in ALGORITHMS}
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    representation_manifest: list[dict[str, Any]] = []
    representations = out / "representations"
    representations.mkdir(parents=True, exist_ok=False)
    index = 0
    by_mask_rep_gauge: dict[tuple[str, int, str], int] = {}
    for replicate in range(REPLICATES):
        seed = seed_for(split, replicate)
        base = BaseRepresentation(seed, device)
        representation_id = digest({"phase": PHASE, "split": split, "replicate": replicate, "seed": seed})[:18]
        representation_path = representations / f"{representation_id}.npz"
        np.savez_compressed(representation_path, **base.export())
        representation_manifest.append(
            {
                "representation_id": representation_id,
                "replicate": replicate,
                "seed": seed,
                "representation_sha256": sha256_file(representation_path),
            }
        )
        for gauge in GAUGES:
            oracle = GaugeOracle(base, gauge, seed + 17011 + 313 * GAUGES.index(gauge))
            states = oracle.states(inputs)
            gram_feature = state_gram_feature(states)
            rank_feature, rank_diagnostics = dependency_rank_feature(states)
            reference = oracle.output(states, inputs, USE_MASKS["000_none"])
            for use_label, use_mask in USE_MASKS.items():
                clean = oracle.output(states, inputs, use_mask)
                clean_prediction = torch.argmax(clean, dim=1)
                clean_accuracy = float(torch.mean((clean_prediction == receiver_targets).to(torch.float64)).item())
                clean_target_probability = float(torch.min(clean[batch, receiver_targets]).item())
                clean_function_error = float(torch.max(torch.abs(clean - reference)).item())
                causal_feature, causal_diagnostics = causal_use_feature(oracle, inputs, states, use_mask)
                feature_rows["state_gram"].append(gram_feature.astype(np.float32))
                feature_rows["dependency_rank_quotient"].append(rank_feature.astype(np.float32))
                feature_rows[CANDIDATE].append(causal_feature.astype(np.float64))
                unit_id = digest(
                    {"phase": PHASE, "split": split, "representation_id": representation_id, "gauge": gauge, "use_label": use_label}
                )[:20]
                public.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "representation_id": representation_id,
                        "split": split,
                        "replicate": replicate,
                        "gauge": gauge,
                    }
                )
                truth.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "representation_id": representation_id,
                        "split": split,
                        "replicate": replicate,
                        "gauge": gauge,
                        "use_label": use_label,
                        "use_mask": list(use_mask),
                    }
                )
                diagnostics.append(
                    {
                        "index": index,
                        "unit_id": unit_id,
                        "clean_accuracy": clean_accuracy,
                        "clean_target_probability_min": clean_target_probability,
                        "clean_function_abs_error_max": clean_function_error,
                        "rank_diagnostics": rank_diagnostics,
                        "causal_diagnostics": causal_diagnostics,
                    }
                )
                by_mask_rep_gauge[(use_label, replicate, gauge)] = index
                index += 1
        del base
        torch.cuda.empty_cache()

    arrays = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public)
    write_jsonl(out / "sealed_truth.jsonl", truth)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)
    write_jsonl(out / "representation_manifest.jsonl", representation_manifest)

    representation_matches = []
    dependency_gauge_matches = []
    candidate_gauge_errors = []
    for replicate in range(REPLICATES):
        for gauge in GAUGES:
            reference_index = by_mask_rep_gauge[("000_none", replicate, gauge)]
            for use_label in USE_MASKS:
                current = by_mask_rep_gauge[(use_label, replicate, gauge)]
                representation_matches.append(
                    bool(
                        np.array_equal(arrays["state_gram"][reference_index], arrays["state_gram"][current])
                        and np.array_equal(
                            arrays["dependency_rank_quotient"][reference_index], arrays["dependency_rank_quotient"][current]
                        )
                    )
                )
        for use_label in USE_MASKS:
            identity = by_mask_rep_gauge[(use_label, replicate, "identity")]
            for gauge in GAUGES[1:]:
                current = by_mask_rep_gauge[(use_label, replicate, gauge)]
                dependency_gauge_matches.append(
                    bool(
                        np.array_equal(
                            arrays["dependency_rank_quotient"][identity], arrays["dependency_rank_quotient"][current]
                        )
                    )
                )
                candidate_gauge_errors.append(
                    float(np.max(np.abs(arrays[CANDIDATE][identity] - arrays[CANDIDATE][current])))
                )
    t = protocol["thresholds"]
    used_donor = [
        row["causal_diagnostics"]["used_donor_probability_min"]
        for row in diagnostics
        if any(truth[row["index"]]["use_mask"])
    ]
    unused_receiver = [
        row["causal_diagnostics"]["unused_receiver_probability_min"]
        for row in diagnostics
        if not all(truth[row["index"]]["use_mask"])
    ]
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "representation_count": len(representation_manifest),
        "unit_count": len(public),
        "clean_accuracy_min": min(row["clean_accuracy"] for row in diagnostics),
        "clean_target_probability_min": min(row["clean_target_probability_min"] for row in diagnostics),
        "clean_function_abs_error_max": max(row["clean_function_abs_error_max"] for row in diagnostics),
        "finite_fraction": float(np.mean([np.isfinite(array).mean() for array in arrays.values()])),
        "representation_mask_match_fraction": float(np.mean(representation_matches)),
        "dependency_gauge_match_fraction": float(np.mean(dependency_gauge_matches)),
        "candidate_gauge_abs_error_max": max(candidate_gauge_errors),
        "alpha_zero_tv_max": max(row["causal_diagnostics"]["alpha_zero_tv_max"] for row in diagnostics),
        "matched_null_tv_max": max(row["causal_diagnostics"]["matched_null_tv_max"] for row in diagnostics),
        "used_donor_probability_min": min(used_donor),
        "unused_receiver_probability_min": min(unused_receiver),
        "expected_hybrid_probability_min": min(
            row["causal_diagnostics"]["expected_hybrid_probability_min"] for row in diagnostics
        ),
        "feature_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "diagnostics_sha256": sha256_file(out / "diagnostics.jsonl"),
        "representation_manifest_sha256": sha256_file(out / "representation_manifest.jsonl"),
    }
    checks = {
        "clean_accuracy": summary["clean_accuracy_min"] >= t["clean_accuracy_min"],
        "clean_probability": summary["clean_target_probability_min"] >= t["clean_target_probability_min"],
        "same_clean_function": summary["clean_function_abs_error_max"] <= t["clean_function_abs_error_max"],
        "finite": summary["finite_fraction"] >= t["finite_fraction_min"],
        "same_representation_across_masks": summary["representation_mask_match_fraction"] >= t["representation_mask_match_fraction_min"],
        "dependency_gauge_invariance": summary["dependency_gauge_match_fraction"] >= t["dependency_gauge_match_fraction_min"],
        "candidate_gauge_equivariance": summary["candidate_gauge_abs_error_max"] <= t["candidate_gauge_abs_error_max"],
        "alpha_zero_null": summary["alpha_zero_tv_max"] <= t["alpha_zero_tv_max"],
        "matched_null": summary["matched_null_tv_max"] <= t["matched_null_tv_max"],
        "used_factor_transport": summary["used_donor_probability_min"] >= t["used_donor_probability_min"],
        "unused_factor_retention": summary["unused_receiver_probability_min"] >= t["unused_receiver_probability_min"],
        "expected_hybrid_target": summary["expected_hybrid_probability_min"] >= t["expected_hybrid_probability_min"],
    }
    summary["checks"] = checks
    summary["run_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def fit_command() -> None:
    protocol = verify_protocol()
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("discovery run gate failed")
    root = OUT_ROOT / "runs/discovery"
    truth = read_jsonl(root / "sealed_truth.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    fit_indices = [
        index for index, row in enumerate(truth) if int(row["replicate"]) in FIT_REPLICATES and row["gauge"] == "identity"
    ]
    validation_indices = [index for index, row in enumerate(truth) if int(row["replicate"]) in VALIDATION_REPLICATES]
    prototypes: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        labels, proto = coverage.build_prototypes(arrays[algorithm], truth, fit_indices, "use_label")
        prototypes[algorithm] = proto.astype(np.float64)
        metadata[algorithm] = {"labels": labels}
        prediction, _ = coverage.predict(arrays[algorithm][validation_indices], labels, proto)
        metrics[algorithm] = classification_metrics(prediction, truth, validation_indices)
    candidate = metrics[CANDIDATE]
    t = protocol["thresholds"]
    checks = {
        "candidate_accuracy": candidate["accuracy"] >= t["discovery_accuracy_min"],
        "candidate_min_label_accuracy": candidate["min_label_accuracy"] >= t["discovery_min_label_accuracy_min"],
        "candidate_gauge_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "state_control_at_ceiling": metrics["state_gram"]["accuracy"] <= t["representation_control_accuracy_max"],
        "rank_control_at_ceiling": metrics["dependency_rank_quotient"]["accuracy"] <= t["representation_control_accuracy_max"],
        "discovery_run_gate": bool(summary["run_gate_passed"]),
    }
    analysis = OUT_ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(analysis / "frozen_prototypes.npz", **prototypes)
    write_json(analysis / "prototype_labels.json", metadata)
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_count": len(fit_indices),
        "validation_count": len(validation_indices),
        "algorithm_metrics": metrics,
        "candidate_checks": checks,
        "candidate_qualified": all(checks.values()),
        "confirmation_run_authorized": all(checks.values()),
        "prototype_sha256": sha256_file(analysis / "frozen_prototypes.npz"),
        "labels_sha256": sha256_file(analysis / "prototype_labels.json"),
    }
    result["fit_digest"] = digest(result)
    write_json(analysis / "fit.json", result)
    print(canonical(result))


def predict_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if not fit["confirmation_run_authorized"]:
        raise RuntimeError("confirmation prediction denied")
    if (OUT_ROOT / "predictions").exists():
        raise RuntimeError("refusing to overwrite confirmation predictions")
    root = OUT_ROOT / "runs/confirmation"
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "feature_pack.npz") as pack:
        arrays = {name: np.asarray(pack[name]) for name in pack.files}
    metadata = read_json(OUT_ROOT / "analysis/prototype_labels.json")
    with np.load(OUT_ROOT / "analysis/frozen_prototypes.npz") as stored:
        prototypes = {name: np.asarray(stored[name]) for name in stored.files}
    rows = []
    for algorithm in ALGORITHMS:
        predicted, scores = coverage.predict(arrays[algorithm], metadata[algorithm]["labels"], prototypes[algorithm])
        for index, public_row in enumerate(public):
            if algorithm == ALGORITHMS[0]:
                rows.append(
                    {
                        "index": index,
                        "unit_id": public_row["unit_id"],
                        "replicate": public_row["replicate"],
                        "gauge": public_row["gauge"],
                        "algorithms": {},
                    }
                )
            rows[index]["algorithms"][algorithm] = {"prediction": predicted[index], "cosine": float(scores[index])}
    path = OUT_ROOT / "predictions/confirmation_predictions.jsonl"
    write_jsonl(path, rows)
    manifest = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_count": len(rows),
        "confirmation_truth_read": False,
        "prediction_sha256": sha256_file(path),
    }
    manifest["prediction_digest"] = digest(manifest)
    write_json(OUT_ROOT / "predictions/manifest.json", manifest)
    print(canonical(manifest))


def score_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    manifest = read_json(OUT_ROOT / "predictions/manifest.json")
    predictions = read_jsonl(OUT_ROOT / "predictions/confirmation_predictions.jsonl")
    truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    if len(predictions) != len(truth):
        raise RuntimeError("prediction/truth length mismatch")
    indices = list(range(len(truth)))
    metrics = {}
    for algorithm in ALGORITHMS:
        predicted = [row["algorithms"][algorithm]["prediction"] for row in predictions]
        metrics[algorithm] = classification_metrics(predicted, truth, indices)
    candidate = metrics[CANDIDATE]
    t = protocol["thresholds"]
    checks = {
        "candidate_accuracy": candidate["accuracy"] >= t["confirmation_accuracy_min"],
        "candidate_min_label_accuracy": candidate["min_label_accuracy"] >= t["confirmation_min_label_accuracy_min"],
        "candidate_gauge_gap": candidate["gauge_accuracy_gap"] <= t["candidate_gauge_accuracy_gap_max"],
        "state_control_at_ceiling": metrics["state_gram"]["accuracy"] <= t["representation_control_accuracy_max"],
        "rank_control_at_ceiling": metrics["dependency_rank_quotient"]["accuracy"] <= t["representation_control_accuracy_max"],
        "prediction_was_blind": manifest["confirmation_truth_read"] is False,
        "prediction_hash": sha256_file(OUT_ROOT / "predictions/confirmation_predictions.jsonl") == manifest["prediction_sha256"],
    }
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": manifest["prediction_digest"],
        "count": len(truth),
        "algorithm_metrics": metrics,
        "candidate_checks": checks,
        "candidate_confirmed": all(checks.values()),
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/score.json", result)
    print(canonical(result))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    score = read_json(OUT_ROOT / "analysis/score.json")
    discovery_ids = {row["unit_id"] for row in read_jsonl(OUT_ROOT / "runs/discovery/public_manifest.jsonl")}
    confirmation_ids = {row["unit_id"] for row in read_jsonl(OUT_ROOT / "runs/confirmation/public_manifest.jsonl")}
    overlap = len(discovery_ids & confirmation_ids)
    passed = bool(
        discovery["run_gate_passed"]
        and fit["candidate_qualified"]
        and confirmation["run_gate_passed"]
        and score["candidate_confirmed"]
        and overlap == 0
    )
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "fit_digest": fit["fit_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "score_digest": score["score_digest"],
        "split_overlap": overlap,
        "matched_transport_causal_use_camera_confirmed": passed,
        "representation_and_clean_behavior_matched_separation_confirmed": passed,
        "causal_use_on_independently_learned_networks_confirmed": False,
        "interaction_hyperedge_identification_confirmed": False,
        "redundancy_or_gate_identification_confirmed": False,
        "free_transformer_scan_authorized": False,
        "pretrained_model_scan_authorized": False,
        "outcome": "controlled_causal_use_quotient_confirmed" if passed else "controlled_causal_use_quotient_not_confirmed",
        "claim_boundary": (
            "The confirmed object, if passed, is an eight-way row/column/context mediator-use mask in an explicit known-truth bypass scaffold. "
            "It separates represented factors from factors used through the declared mediator under matched one-factor transport and bounded invertible linear gauges. "
            "It does not identify interaction hyperedges, redundancy, gates, necessity, nonlinear gauges, learned-network external validity, or a natural-language mechanism."
        ),
        "next_legal_task": (
            "Calibrate interaction-only and additive-versus-synergistic causal hyperedges with the same matched representation/behavior discipline before testing redundancy or learned systems."
        ),
        "auto_continue": passed,
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("--split", choices=SPLITS, required=True)
    sub.add_parser("fit")
    sub.add_parser("predict")
    sub.add_parser("score")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.split)
    elif args.command == "fit":
        fit_command()
    elif args.command == "predict":
        predict_command()
    elif args.command == "score":
        score_command()
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
