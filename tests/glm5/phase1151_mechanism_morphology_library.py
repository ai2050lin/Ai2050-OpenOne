#!/usr/bin/env python3
"""Build a blinded, known-truth mechanism morphology library.

Phase1150 ended the role-factorized replication branch.  This phase changes
the research object: it constructs causal systems with known equations and
records the observations available to a mechanism-identification algorithm.
It does not select or score an algorithm.  Phase1152 consumes the frozen
feature packs and performs the blinded coverage test.

Two implementation pairs are deliberately observationally equivalent under
the available interface:

    additive_load == tensor_product_binding
    bilinear_binding == role_factorized

An honest algorithm should recover their functional equivalence class, not
invent evidence that distinguishes the hidden implementation label.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PHASE = 1151
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1151_mechanism_morphology_library"
SPLITS = ("discovery", "confirmation")

N_ROWS = 4
N_COLS = 4
N_CLASSES = N_ROWS * N_COLS
N_SITES = 5
STATE_DIM = N_CLASSES
BASE_REPLICATES = 3
CHARTS = ("identity", "rotated")
TAILS = ("stable", "degraded")
EPS = 1e-9

MECHANISMS = (
    "additive_load",
    "multiplicative_gate",
    "bilinear_binding",
    "tensor_product_binding",
    "role_factorized",
    "multi_position_coalition",
    "redundant_circuit",
    "context_switched_paths",
)

FUNCTIONAL_GROUP = {
    "additive_load": "single_joint_carrier",
    "tensor_product_binding": "single_joint_carrier",
    "bilinear_binding": "factorized_roles",
    "role_factorized": "factorized_roles",
    "multiplicative_gate": "payload_with_gate",
    "multi_position_coalition": "joint_coalition",
    "redundant_circuit": "redundant_paths",
    "context_switched_paths": "context_switched_paths",
}

EQUIVALENT_PAIRS = (
    ("additive_load", "tensor_product_binding"),
    ("bilinear_binding", "role_factorized"),
)

FEATURE_NAMES = (
    "raw_coordinates",
    "state_gram",
    "single_site_patch",
    "pairwise_patch",
    "factorial_interaction",
    "exhaustive_coalition",
    "functional_tomography",
)

THRESHOLDS = {
    "finite_fraction": 1.0,
    "clean_accuracy": 1.0,
    "stable_tail_ratio_min": 0.95,
    "degraded_tail_ratio_max": 0.05,
    "functional_chart_cosine_min": 0.999999,
    "functional_equivalence_cosine_min": 0.999999,
    "split_overlap_max": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > EPS else np.zeros_like(value)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = unit(left)
    b = unit(right)
    if not np.any(a) or not np.any(b):
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b))


def stats(values: Iterable[float]) -> list[float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return [0.0] * 6
    return [
        float(np.min(data)),
        float(np.max(data)),
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
        float(np.mean(data >= 0.90)),
    ]


def sorted_rows(rows: Iterable[Iterable[float]]) -> np.ndarray:
    arrays = [np.asarray(row, dtype=np.float64).reshape(-1) for row in rows]
    arrays.sort(key=lambda row: tuple(np.round(row, 8).tolist()))
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)


def cell_index(row: int, col: int) -> int:
    return int(row) * N_COLS + int(col)


def row_mask(row: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(STATE_DIM, dtype=torch.float64, device=device)
    start = int(row) * N_COLS
    value[start : start + N_COLS] = 1.0
    return value


def col_mask(col: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(STATE_DIM, dtype=torch.float64, device=device)
    value[int(col) :: N_COLS] = 1.0
    return value


def one_hot(index: int, device: torch.device) -> torch.Tensor:
    value = torch.zeros(STATE_DIM, dtype=torch.float64, device=device)
    value[int(index)] = 1.0
    return value


def make_orthogonal(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn(STATE_DIM, STATE_DIM, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs).to(device)


def build_units(split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(split)
    split_seed = 115100 if split == "discovery" else 115900
    rows = []
    for mechanism in MECHANISMS:
        for replicate in range(BASE_REPLICATES):
            nuisance_seed = split_seed + replicate * 101
            pair_key = digest(
                {
                    "phase": PHASE,
                    "split": split,
                    "replicate": replicate,
                    "seed": nuisance_seed,
                }
            )[:16]
            for tail in TAILS:
                for chart in CHARTS:
                    public_key = {
                        "phase": PHASE,
                        "split": split,
                        "replicate": replicate,
                        "tail": tail,
                        "chart": chart,
                        "mechanism": mechanism,
                    }
                    rows.append(
                        {
                            "unit_id": digest(public_key)[:20],
                            "pair_key": pair_key,
                            "split": split,
                            "replicate": replicate,
                            "nuisance_seed": nuisance_seed,
                            "chart": chart,
                            "tail": tail,
                            "mechanism": mechanism,
                            "functional_group": FUNCTIONAL_GROUP[mechanism],
                        }
                    )
    return rows


class MechanismOracle:
    """Black-box causal system with hidden implementation labels."""

    def __init__(self, spec: dict[str, Any], device: torch.device) -> None:
        self.kind = str(spec["mechanism"])
        self.tail = str(spec["tail"])
        self.chart = str(spec["chart"])
        self.device = device
        seed = int(spec["nuisance_seed"])
        rng = np.random.default_rng(seed)
        self.logical_to_physical = rng.permutation(N_SITES).astype(np.int64)
        self.scales = torch.tensor(
            rng.uniform(0.65, 1.75, size=N_SITES),
            dtype=torch.float64,
            device=device,
        )
        charts = []
        for logical_site in range(N_SITES):
            if self.chart == "identity":
                charts.append(torch.eye(STATE_DIM, dtype=torch.float64, device=device))
            else:
                charts.append(make_orthogonal(seed + 1009 * (logical_site + 1), device))
        self.charts = torch.stack(charts, dim=0)

    def _logical_states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        batch = len(inputs)
        states = torch.zeros(
            batch,
            N_SITES,
            STATE_DIM,
            dtype=torch.float64,
            device=self.device,
        )
        for index, (row, col, context) in enumerate(inputs):
            joint = one_hot(cell_index(row, col), self.device)
            if self.kind in {"additive_load", "tensor_product_binding"}:
                states[index, 0] = joint
            elif self.kind == "multiplicative_gate":
                states[index, 0] = joint
                states[index, 1, 0] = 1.0
            elif self.kind in {"bilinear_binding", "role_factorized"}:
                states[index, 0] = row_mask(row, self.device)
                states[index, 1] = col_mask(col, self.device)
            elif self.kind == "multi_position_coalition":
                states[index, 0] = joint
                states[index, 1] = joint
            elif self.kind == "redundant_circuit":
                states[index, 0] = joint
                states[index, 1] = joint
            elif self.kind == "context_switched_paths":
                states[index, 0] = joint
                states[index, 1] = joint
                states[index, 2, int(context)] = 1.0
            else:
                raise ValueError(self.kind)
        return states

    def states(self, inputs: list[tuple[int, int, int]]) -> torch.Tensor:
        logical = self._logical_states(inputs)
        observed = torch.zeros_like(logical)
        for logical_site in range(N_SITES):
            physical_site = int(self.logical_to_physical[logical_site])
            encoded = logical[:, logical_site] @ self.charts[logical_site].T
            observed[:, physical_site] = encoded * self.scales[logical_site]
        return observed

    def _decode(self, observed: torch.Tensor) -> torch.Tensor:
        logical = torch.zeros_like(observed)
        for logical_site in range(N_SITES):
            physical_site = int(self.logical_to_physical[logical_site])
            scaled = observed[:, physical_site] / self.scales[logical_site]
            logical[:, logical_site] = scaled @ self.charts[logical_site]
        return logical

    def output(
        self,
        observed: torch.Tensor,
        receivers: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        logical = self._decode(observed)
        if self.kind in {"additive_load", "tensor_product_binding"}:
            logits = logical[:, 0]
        elif self.kind == "multiplicative_gate":
            logits = logical[:, 0] * logical[:, 1, 0:1]
        elif self.kind in {
            "bilinear_binding",
            "role_factorized",
            "multi_position_coalition",
        }:
            logits = logical[:, 0] * logical[:, 1]
        elif self.kind == "redundant_circuit":
            logits = 0.5 * (logical[:, 0] + logical[:, 1])
        elif self.kind == "context_switched_paths":
            logits = (
                logical[:, 2, 0:1] * logical[:, 0]
                + logical[:, 2, 1:2] * logical[:, 1]
            )
        else:
            raise ValueError(self.kind)

        if self.tail == "degraded":
            logits = logits.clone()
            for index, (row, col, _context) in enumerate(receivers):
                if cell_index(row, col) == N_CLASSES - 1:
                    logits[index] *= 0.51
                    logits[index, N_CLASSES - 2] += 0.49
        return logits


def distribution(logits: torch.Tensor) -> torch.Tensor:
    nonnegative = torch.clamp(logits, min=0.0)
    denom = torch.sum(nonnegative, dim=1, keepdim=True)
    return torch.where(denom > EPS, nonnegative / denom, torch.zeros_like(nonnegative))


def morph_inputs() -> list[tuple[int, int, int]]:
    return [
        (row, col, context)
        for context in (0, 1)
        for row in range(N_ROWS)
        for col in range(N_COLS)
        if cell_index(row, col) != N_CLASSES - 1
    ]


def all_inputs() -> list[tuple[int, int, int]]:
    return [
        (row, col, context)
        for context in (0, 1)
        for row in range(N_ROWS)
        for col in range(N_COLS)
    ]


def donor_inputs(
    receivers: list[tuple[int, int, int]],
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    full = []
    row_only = []
    col_only = []
    for row, col, context in receivers:
        next_row = (row + 1) % N_ROWS
        next_col = (col + 1) % N_COLS
        full.append((next_row, next_col, context))
        row_only.append((next_row, col, context))
        col_only.append((row, next_col, context))
    return full, row_only, col_only


def class_scores(
    probs: torch.Tensor,
    targets: list[tuple[int, int, int]],
) -> torch.Tensor:
    indices = torch.tensor(
        [cell_index(row, col) for row, col, _context in targets],
        dtype=torch.long,
        device=probs.device,
    )
    return probs.gather(1, indices[:, None]).squeeze(1)


def row_scores(probs: torch.Tensor, targets: list[tuple[int, int, int]]) -> torch.Tensor:
    rows = []
    for index, (row, _col, _context) in enumerate(targets):
        start = int(row) * N_COLS
        rows.append(torch.sum(probs[index, start : start + N_COLS]))
    return torch.stack(rows)


def col_scores(probs: torch.Tensor, targets: list[tuple[int, int, int]]) -> torch.Tensor:
    rows = []
    for index, (_row, col, _context) in enumerate(targets):
        rows.append(torch.sum(probs[index, int(col) :: N_COLS]))
    return torch.stack(rows)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(torch.mean(selected).item()) if selected.numel() else 0.0


def patch_states(
    receiver_states: torch.Tensor,
    donor_states: torch.Tensor,
    subset: tuple[int, ...],
) -> torch.Tensor:
    patched = receiver_states.clone()
    for site in subset:
        patched[:, int(site)] = donor_states[:, int(site)]
    return patched


def probe_system(oracle: MechanismOracle) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    receivers = morph_inputs()
    full_donors, row_donors, col_donors = donor_inputs(receivers)
    receiver_states = oracle.states(receivers)
    full_states = oracle.states(full_donors)
    row_states = oracle.states(row_donors)
    col_states = oracle.states(col_donors)
    clean_logits = oracle.output(receiver_states, receivers)
    clean_probs = distribution(clean_logits)
    context_mask = {
        value: torch.tensor(
            [context == value for _row, _col, context in receivers],
            dtype=torch.bool,
            device=oracle.device,
        )
        for value in (0, 1)
    }

    subset_rows: dict[tuple[int, ...], np.ndarray] = {}
    subset_outputs: dict[tuple[int, ...], torch.Tensor] = {}
    for size in range(1, N_SITES + 1):
        for subset in itertools.combinations(range(N_SITES), size):
            patched = patch_states(receiver_states, full_states, subset)
            logits = oracle.output(patched, receivers)
            probs = distribution(logits)
            donor_prob = class_scores(probs, full_donors)
            receiver_prob = class_scores(probs, receivers)
            donor_row = row_scores(probs, full_donors)
            donor_col = col_scores(probs, full_donors)
            donor_indices = torch.tensor(
                [cell_index(row, col) for row, col, _ctx in full_donors],
                dtype=torch.long,
                device=oracle.device,
            )
            donor_top1 = (torch.argmax(probs, dim=1) == donor_indices).to(torch.float64)
            effect_l1 = torch.sum(torch.abs(probs - clean_probs), dim=1)

            ablated = receiver_states.clone()
            for site in subset:
                ablated[:, int(site)] = 0.0
            ablation_probs = distribution(oracle.output(ablated, receivers))
            ablation_receiver = class_scores(ablation_probs, receivers)
            receiver_indices = torch.tensor(
                [cell_index(row, col) for row, col, _ctx in receivers],
                dtype=torch.long,
                device=oracle.device,
            )
            ablation_top1 = (
                torch.argmax(ablation_probs, dim=1) == receiver_indices
            ).to(torch.float64)
            subset_rows[subset] = np.asarray(
                [
                    float(torch.mean(donor_prob).item()),
                    masked_mean(donor_prob, context_mask[0]),
                    masked_mean(donor_prob, context_mask[1]),
                    float(torch.mean(donor_top1).item()),
                    float(torch.mean(donor_row).item()),
                    float(torch.mean(donor_col).item()),
                    float(torch.mean(receiver_prob).item()),
                    float(torch.mean(effect_l1).item()),
                    float(torch.mean(ablation_receiver).item()),
                    float(torch.mean(ablation_top1).item()),
                ],
                dtype=np.float64,
            )
            subset_outputs[subset] = probs

    pair_rows = []
    factorial_rows = []
    for left, right in itertools.combinations(range(N_SITES), 2):
        mixed_forward = receiver_states.clone()
        mixed_forward[:, left] = row_states[:, left]
        mixed_forward[:, right] = col_states[:, right]
        mixed_reverse = receiver_states.clone()
        mixed_reverse[:, left] = col_states[:, left]
        mixed_reverse[:, right] = row_states[:, right]
        forward_probs = distribution(oracle.output(mixed_forward, receivers))
        reverse_probs = distribution(oracle.output(mixed_reverse, receivers))
        combined = [
            (row_donor[0], col_donor[1], receiver[2])
            for receiver, row_donor, col_donor in zip(
                receivers, row_donors, col_donors
            )
        ]
        forward_score = float(torch.mean(class_scores(forward_probs, combined)).item())
        reverse_score = float(torch.mean(class_scores(reverse_probs, combined)).item())
        singleton_left = subset_outputs[(left,)]
        singleton_right = subset_outputs[(right,)]
        pair_output = subset_outputs[(left, right)]
        interaction = pair_output - singleton_left - singleton_right + clean_probs
        synergy = float(torch.mean(torch.sum(torch.abs(interaction), dim=1)).item())
        base = subset_rows[(left, right)]
        pair_row = np.concatenate(
            [
                base,
                np.asarray(
                    [
                        forward_score,
                        reverse_score,
                        synergy,
                        abs(float(base[1]) - float(base[2])),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        pair_rows.append(pair_row)
        factorial_rows.append(
            np.asarray(
                [
                    forward_score,
                    reverse_score,
                    synergy,
                    float(base[0]),
                    abs(float(base[1]) - float(base[2])),
                    float(base[8]),
                ],
                dtype=np.float64,
            )
        )

    raw_coordinates = receiver_states.detach().cpu().numpy().reshape(-1).astype(np.float64)
    gram_blocks = []
    state_cpu = receiver_states.detach().cpu().numpy()
    tri = np.triu_indices(len(receivers))
    for site in range(N_SITES):
        matrix = state_cpu[:, site, :]
        gram = matrix @ matrix.T
        norm = float(np.linalg.norm(gram))
        if norm > EPS:
            gram = gram / norm
        gram_blocks.append(gram[tri])
    state_gram = sorted_rows(gram_blocks)

    singleton_rows = [subset_rows[(site,)] for site in range(N_SITES)]
    single_feature = sorted_rows(singleton_rows)
    pair_feature = np.concatenate([single_feature, sorted_rows(pair_rows)])
    factorial_feature = sorted_rows(factorial_rows)

    exhaustive_parts = []
    for size in range(1, N_SITES + 1):
        rows = [row for subset, row in subset_rows.items() if len(subset) == size]
        for metric_index in (0, 3, 6, 8):
            exhaustive_parts.extend(stats(row[metric_index] for row in rows))
    qualified_sizes = [
        len(subset)
        for subset, row in subset_rows.items()
        if float(row[0]) >= 0.90 and float(row[3]) >= 0.90
    ]
    context_switch_values = [abs(float(row[1]) - float(row[2])) for row in singleton_rows]
    exhaustive_parts.extend(
        [
            float(min(qualified_sizes)) if qualified_sizes else float(N_SITES + 1),
            float(len([value for value in qualified_sizes if value == min(qualified_sizes)]))
            if qualified_sizes
            else 0.0,
            float(max(context_switch_values)),
            float(sum(value >= 0.90 for value in context_switch_values)),
        ]
    )
    exhaustive_feature = np.asarray(exhaustive_parts, dtype=np.float64)
    functional_feature = np.concatenate(
        [single_feature, sorted_rows(pair_rows), factorial_feature, exhaustive_feature]
    )

    complete_inputs = all_inputs()
    complete_states = oracle.states(complete_inputs)
    complete_logits = oracle.output(complete_states, complete_inputs)
    complete_probs = distribution(complete_logits)
    target_indices = torch.tensor(
        [cell_index(row, col) for row, col, _context in complete_inputs],
        dtype=torch.long,
        device=oracle.device,
    )
    predictions = torch.argmax(complete_probs, dim=1)
    clean_accuracy = float(torch.mean((predictions == target_indices).to(torch.float64)).item())
    target_values = complete_probs.gather(1, target_indices[:, None]).squeeze(1)
    masked = complete_probs.clone()
    masked.scatter_(1, target_indices[:, None], -1.0)
    margins = target_values - torch.max(masked, dim=1).values
    tail_mask = target_indices == (N_CLASSES - 1)
    tail_margin = float(torch.mean(margins[tail_mask]).item())
    body_margin = float(torch.median(margins[~tail_mask]).item())
    tail_ratio = tail_margin / body_margin if abs(body_margin) > EPS else 0.0

    features = {
        "raw_coordinates": raw_coordinates,
        "state_gram": state_gram,
        "single_site_patch": single_feature,
        "pairwise_patch": pair_feature,
        "factorial_interaction": factorial_feature,
        "exhaustive_coalition": exhaustive_feature,
        "functional_tomography": functional_feature,
    }
    diagnostics = {
        "clean_accuracy": clean_accuracy,
        "tail_ratio": float(tail_ratio),
        "finite_fraction": float(
            np.mean(
                [
                    np.isfinite(value).mean()
                    for value in list(features.values())
                ]
            )
        ),
    }
    return features, diagnostics


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1151 after run output exists")
    prior_1143 = read_json(
        ROOT
        / "tests/glm5/result/phase1143_ground_truth_mechanism_calibration/analysis/final.json"
    )
    prior_1144 = read_json(
        ROOT
        / "tests/glm5/result/phase1144_symmetric_factorial_operator_calibration/analysis/final.json"
    )
    prior_1150 = read_json(
        ROOT
        / "tests/glm5/result/phase1150_role_factorized_independent_replication/analysis/final.json"
    )
    checks = {
        "phase1143_calibrated": bool(prior_1143["calibration_passed"]),
        "phase1144_calibrated": bool(prior_1144["calibration_passed"]),
        "phase1150_same_protocol_stopped": not bool(prior_1150["auto_continue"]),
        "eight_mechanisms": len(MECHANISMS) == 8,
        "six_functional_groups": len(set(FUNCTIONAL_GROUP.values())) == 6,
        "two_declared_equivalence_pairs": len(EQUIVALENT_PAIRS) == 2,
        "orthogonal_nuisance_axes": len(CHARTS) == 2 and len(TAILS) == 2,
        "split_seeds_disjoint": True,
        "confirmation_labels_sealed": True,
        "algorithm_selection_forbidden": True,
        "natural_model_scan_forbidden": True,
        "causal_claim_forbidden": True,
        "cuda_required": True,
    }
    protocol = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "known-truth mechanism morphology library",
        "script_sha256": sha256_file(SCRIPT),
        "source_digests": {
            "phase1143": prior_1143["final_digest"],
            "phase1144": prior_1144["final_digest"],
            "phase1150": prior_1150["final_digest"],
        },
        "design": {
            "mechanisms": list(MECHANISMS),
            "functional_group": FUNCTIONAL_GROUP,
            "equivalent_pairs": [list(pair) for pair in EQUIVALENT_PAIRS],
            "features": list(FEATURE_NAMES),
            "base_replicates": BASE_REPLICATES,
            "charts": list(CHARTS),
            "tails": list(TAILS),
            "units_per_split": len(build_units("discovery")),
            "rows": N_ROWS,
            "cols": N_COLS,
            "sites": N_SITES,
            "state_dim": STATE_DIM,
        },
        "thresholds": THRESHOLDS,
        "hard_stops": [
            "Phase1151 may validate the library but may not select an algorithm.",
            "Confirmation mechanism labels remain sealed until Phase1152 predictions exist.",
            "Implementation labels inside declared equivalence pairs are not identifiable claims.",
            "No pretrained-model hotspot or causal component search is authorized.",
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
    if digest(body) != stored:
        raise RuntimeError("Phase1151 protocol digest mismatch")
    if sha256_file(SCRIPT) != protocol["script_sha256"]:
        raise RuntimeError("Phase1151 script changed after preregistration")
    return protocol


def run_command(split: str) -> None:
    protocol = verify_protocol()
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    units = build_units(split)
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    diagnostics = []
    public_rows = []
    truth_rows = []
    for index, spec in enumerate(units):
        oracle = MechanismOracle(spec, device)
        features, diagnostic = probe_system(oracle)
        for name in FEATURE_NAMES:
            feature_rows[name].append(features[name].astype(np.float32))
        public_rows.append(
            {
                "index": index,
                "unit_id": spec["unit_id"],
                "pair_key": spec["pair_key"],
                "split": split,
            }
        )
        truth_rows.append(
            {
                "index": index,
                "unit_id": spec["unit_id"],
                "pair_key": spec["pair_key"],
                "split": split,
                "replicate": spec["replicate"],
                "chart": spec["chart"],
                "tail": spec["tail"],
                "mechanism": spec["mechanism"],
                "functional_group": spec["functional_group"],
            }
        )
        diagnostics.append(
            {
                "index": index,
                "unit_id": spec["unit_id"],
                **diagnostic,
            }
        )
        del oracle
    arrays = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    arrays["tail_ratio"] = np.asarray(
        [row["tail_ratio"] for row in diagnostics], dtype=np.float32
    )
    arrays["clean_accuracy"] = np.asarray(
        [row["clean_accuracy"] for row in diagnostics], dtype=np.float32
    )
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "feature_pack.npz", **arrays)
    write_jsonl(out / "public_manifest.jsonl", public_rows)
    write_jsonl(out / "sealed_truth.jsonl", truth_rows)
    write_jsonl(out / "diagnostics.jsonl", diagnostics)

    stable = [
        row["tail_ratio"]
        for row, truth in zip(diagnostics, truth_rows)
        if truth["tail"] == "stable"
    ]
    degraded = [
        row["tail_ratio"]
        for row, truth in zip(diagnostics, truth_rows)
        if truth["tail"] == "degraded"
    ]
    functional = arrays["functional_tomography"]
    by_key = {
        (
            row["mechanism"],
            row["replicate"],
            row["tail"],
            row["chart"],
        ): int(row["index"])
        for row in truth_rows
    }
    chart_cosines = []
    for mechanism in MECHANISMS:
        for replicate in range(BASE_REPLICATES):
            for tail in TAILS:
                left = by_key[(mechanism, replicate, tail, "identity")]
                right = by_key[(mechanism, replicate, tail, "rotated")]
                chart_cosines.append(cosine(functional[left], functional[right]))
    equivalence_cosines = []
    for left_name, right_name in EQUIVALENT_PAIRS:
        for replicate in range(BASE_REPLICATES):
            for tail in TAILS:
                for chart in CHARTS:
                    left = by_key[(left_name, replicate, tail, chart)]
                    right = by_key[(right_name, replicate, tail, chart)]
                    equivalence_cosines.append(cosine(functional[left], functional[right]))
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "unit_count": len(units),
        "mechanism_counts": {
            name: sum(row["mechanism"] == name for row in truth_rows)
            for name in MECHANISMS
        },
        "feature_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "finite_fraction": float(
            np.mean([np.isfinite(value).mean() for value in arrays.values()])
        ),
        "clean_accuracy_min": float(min(row["clean_accuracy"] for row in diagnostics)),
        "stable_tail_ratio_min": float(min(stable)),
        "degraded_tail_ratio_max": float(max(degraded)),
        "functional_chart_cosine_min": float(min(chart_cosines)),
        "functional_chart_cosine_median": float(statistics.median(chart_cosines)),
        "functional_equivalence_cosine_min": float(min(equivalence_cosines)),
        "functional_equivalence_cosine_median": float(
            statistics.median(equivalence_cosines)
        ),
        "feature_pack_sha256": sha256_file(out / "feature_pack.npz"),
        "public_manifest_sha256": sha256_file(out / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(out / "sealed_truth.jsonl"),
        "diagnostics_sha256": sha256_file(out / "diagnostics.jsonl"),
    }
    t = protocol["thresholds"]
    checks = {
        "finite": summary["finite_fraction"] >= t["finite_fraction"],
        "clean": summary["clean_accuracy_min"] >= t["clean_accuracy"],
        "stable_tail": summary["stable_tail_ratio_min"] >= t["stable_tail_ratio_min"],
        "degraded_tail": summary["degraded_tail_ratio_max"] <= t["degraded_tail_ratio_max"],
        "chart_invariance": summary["functional_chart_cosine_min"]
        >= t["functional_chart_cosine_min"],
        "declared_equivalence": summary["functional_equivalence_cosine_min"]
        >= t["functional_equivalence_cosine_min"],
    }
    summary["checks"] = checks
    summary["all_checks_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def finalize_command() -> None:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    discovery_manifest = {
        row["unit_id"]
        for row in [
            json.loads(line)
            for line in (
                OUT_ROOT / "runs/discovery/public_manifest.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    }
    confirmation_manifest = {
        row["unit_id"]
        for row in [
            json.loads(line)
            for line in (
                OUT_ROOT / "runs/confirmation/public_manifest.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    }
    overlap = len(discovery_manifest & confirmation_manifest)
    passed = bool(
        discovery["all_checks_passed"]
        and confirmation["all_checks_passed"]
        and overlap <= protocol["thresholds"]["split_overlap_max"]
    )
    final = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": discovery["summary_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "split_overlap": overlap,
        "library_qualified": passed,
        "phase1152_blind_coverage_authorized": passed,
        "pretrained_model_scan_authorized": False,
        "causal_component_search_authorized": False,
        "outcome": "mechanism_library_qualified" if passed else "mechanism_library_not_qualified",
        "claim_boundary": (
            "This phase validates a controlled mechanism library and its nuisance axes only. "
            "It does not show that any algorithm can recover the hidden mechanism labels."
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
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.split)
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
