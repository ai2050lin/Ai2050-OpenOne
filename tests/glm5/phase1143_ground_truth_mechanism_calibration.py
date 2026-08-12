#!/usr/bin/env python3
"""Ground-truth calibration for causal mechanism-identification operators.

Phase1142 showed that absolute residual replacement has strong role-compatible
effects without stable item identity.  This phase does not inspect another
natural-model hotspot.  It builds controlled deep residual systems whose
shared, relation, address, and item-payload terms are known, then asks which
intervention operator recovers the known item mapping without inventing one in
payload-free controls.

The candidate operator is a matched double difference (MDD):

    payload_j = (A_j+ - A_j-) - (N_j+ - N_j-)
    patched_i<-j = A_i- + (N_i+ - N_i-) + payload_j

Here A is the active system and N is a matched payload-null system.  The
correct diagonal reconstructs A_i+ algebraically, while the full donor matrix
still tests whether the item-conditioned increment is replaceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PHASE = 1143
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1143_ground_truth_mechanism_calibration"
SCRIPT_PATH = Path(__file__).resolve()
SPLITS = ("discovery", "confirmation")
METHODS = (
    "absolute_state",
    "raw_counterfactual_difference",
    "matched_double_difference",
)
SCENARIOS = (
    "linear_payload",
    "linear_shared_only",
    "nonlinear_payload",
    "nonlinear_shared_only",
    "linear_permuted_payload",
)
CONTEXTS = (0, 1)
N_ITEMS = 32
N_RELATIONS = 4
ITEMS_PER_RELATION = N_ITEMS // N_RELATIONS
N_LAYERS = 12
SHARED_LAYER = 3
RELATION_LAYER = 5
PAYLOAD_LAYER = 7
EPSILON = 1e-7

CONFIGS = {
    "discovery": (
        {"config_id": "d96", "width": 96, "seed": 114301},
        {"config_id": "d128", "width": 128, "seed": 114302},
        {"config_id": "d160", "width": 160, "seed": 114303},
    ),
    "confirmation": (
        {"config_id": "c104", "width": 104, "seed": 114321},
        {"config_id": "c152", "width": 152, "seed": 114322},
        {"config_id": "c184", "width": 184, "seed": 114323},
    ),
}

THRESHOLDS = {
    "finite_fraction": 1.0,
    "positive_expected_top1_within_relation_fraction": 0.95,
    "positive_expected_advantage_same_relation_median": 0.50,
    "positive_expected_advantage_positive_fraction": 0.95,
    "positive_expected_endpoint_flip_fraction": 0.95,
    "negative_max_abs_same_relation_advantage": 1e-4,
    "negative_max_within_relation_spread": 1e-4,
    "permuted_expected_top1_within_relation_fraction": 0.95,
    "permuted_identity_top1_within_relation_fraction_max": 0.20,
    "payload_onset_layer_tolerance": 0,
    "mdd_diagonal_reconstruction_max_abs_error": 2e-4,
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else float("nan")


def relation_of(item: int) -> int:
    return item // ITEMS_PER_RELATION


def scenario_payload(scenario: str) -> bool:
    return scenario in {
        "linear_payload",
        "nonlinear_payload",
        "linear_permuted_payload",
    }


def scenario_nonlinear(scenario: str) -> bool:
    return scenario in {"nonlinear_payload", "nonlinear_shared_only"}


def derangement_within_relations(seed: int) -> list[int]:
    mapping = list(range(N_ITEMS))
    generator = np.random.default_rng(seed)
    for relation in range(N_RELATIONS):
        start = relation * ITEMS_PER_RELATION
        local = np.arange(ITEMS_PER_RELATION)
        shift = int(generator.integers(1, ITEMS_PER_RELATION))
        local = np.roll(local, shift)
        for index, value in enumerate(local.tolist()):
            mapping[start + index] = start + int(value)
    if any(index == value for index, value in enumerate(mapping)):
        raise RuntimeError("permutation is not a derangement")
    return mapping


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1143 after run output exists")
    checks = {
        "item_count_divisible_by_relations": N_ITEMS % N_RELATIONS == 0,
        "payload_after_relation": PAYLOAD_LAYER > RELATION_LAYER > SHARED_LAYER,
        "three_candidate_methods": len(METHODS) == 3,
        "positive_and_negative_linear_controls": {
            "linear_payload",
            "linear_shared_only",
        }.issubset(SCENARIOS),
        "positive_and_negative_nonlinear_controls": {
            "nonlinear_payload",
            "nonlinear_shared_only",
        }.issubset(SCENARIOS),
        "permuted_mapping_control": "linear_permuted_payload" in SCENARIOS,
        "independent_config_ids": not (
            {row["config_id"] for row in CONFIGS["discovery"]}
            & {row["config_id"] for row in CONFIGS["confirmation"]}
        ),
        "independent_seeds": not (
            {row["seed"] for row in CONFIGS["discovery"]}
            & {row["seed"] for row in CONFIGS["confirmation"]}
        ),
        "independent_widths": not (
            {row["width"] for row in CONFIGS["discovery"]}
            & {row["width"] for row in CONFIGS["confirmation"]}
        ),
        "all_widths_fit_basis": all(
            int(row["width"]) >= 1 + N_RELATIONS + N_ITEMS + N_ITEMS
            for split in SPLITS
            for row in CONFIGS[split]
        ),
        "confirmation_locked_before_discovery": True,
        "natural_model_scan_forbidden": True,
        "component_search_forbidden": True,
        "payload_claim_forbidden": True,
        "nonlinear_negative_required": True,
        "mapping_not_diagonal_required": True,
        "layer_onset_required": True,
        "raw_arrays_required_for_audit": True,
        "cuda_required": True,
    }
    prereg: dict[str, Any] = {
        "phase": PHASE,
        "title": "ground-truth mechanism-identification calibration",
        "created_at_utc": utc_now(),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "device_requirement": "cuda",
        "design": {
            "splits": list(SPLITS),
            "configs": CONFIGS,
            "methods": list(METHODS),
            "scenarios": list(SCENARIOS),
            "contexts": list(CONTEXTS),
            "n_items": N_ITEMS,
            "n_relations": N_RELATIONS,
            "n_layers": N_LAYERS,
            "shared_layer": SHARED_LAYER,
            "relation_layer": RELATION_LAYER,
            "payload_layer": PAYLOAD_LAYER,
            "state_terms": ["static_address", "shared_role", "relation_role", "item_payload"],
            "coordinate_systems": ["linear", "invertible_nonlinear"],
            "negative_controls": ["linear_shared_only", "nonlinear_shared_only"],
            "mapping_control": "relation-preserving deranged donor payload identity",
        },
        "candidate_operator": {
            "name": "matched_double_difference",
            "active_delta": "A_plus - A_minus",
            "null_delta": "N_plus - N_minus",
            "donor_increment": "active_delta - null_delta",
            "patched_state": "A_target_minus + null_delta_target + donor_increment",
        },
        "thresholds": THRESHOLDS,
        "gates": {
            "discovery": [
                "known payload recovered in linear and nonlinear systems",
                "no item identity in both shared-only controls",
                "deranged semantic donor mapping recovered instead of raw diagonal",
                "first selective layer equals the known payload insertion layer",
                "correct MDD donor reconstructs the counterfactual endpoint",
            ],
            "confirmation": "all discovery gates repeat in new seeds and unseen widths",
            "natural_transfer": "authorized only if both splits pass; authorization is protocol-only, not a mechanism claim",
        },
        "forbidden": [
            "inspect natural-model hidden states",
            "search heads or neurons",
            "relax thresholds after discovery",
            "drop nonlinear shared-only control",
            "treat controlled calibration as evidence about LLM semantics",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    body = dict(prereg)
    prereg["protocol_digest"] = digest(body)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "phase": PHASE,
            "check_count": len(checks),
            "passed_count": sum(bool(value) for value in checks.values()),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "protocol_digest": prereg["protocol_digest"],
        },
    )
    print(canonical({"protocol_digest": prereg["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    stored = str(prereg["protocol_digest"])
    body = dict(prereg)
    del body["protocol_digest"]
    if digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    if sha256_file(SCRIPT_PATH) != prereg["script_sha256"]:
        raise RuntimeError("primary script changed after preregistration")
    if not read_json(OUT_ROOT / "protocol/audit.json")["all_checks_passed"]:
        raise RuntimeError("protocol audit failed")
    return prereg


def householder(values: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    projection = torch.sum(values * vector, dim=-1, keepdim=True)
    return values - 2.0 * projection * vector


def encode(values: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    if not nonlinear:
        return values
    beta = 8.0
    return torch.sign(values) * torch.log1p(beta * torch.abs(values)) / beta


def decode(values: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    if not nonlinear:
        return values
    beta = 8.0
    return torch.sign(values) * torch.expm1(beta * torch.abs(values)) / beta


def simplex(count: int, device: torch.device) -> torch.Tensor:
    values = torch.eye(count, device=device, dtype=torch.float32)
    values = values - torch.full_like(values, 1.0 / count)
    values = values / torch.linalg.vector_norm(values, dim=1, keepdim=True)
    return values


class ControlledResidualSystem:
    def __init__(self, width: int, seed: int, context: int, device: torch.device) -> None:
        self.width = width
        self.seed = seed
        self.context = context
        self.device = device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        basis_count = 1 + N_RELATIONS + N_ITEMS + N_ITEMS
        raw = torch.randn(width, basis_count, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        basis = q.to(dtype=torch.float32, device=device).T.contiguous()
        cursor = 0
        self.shared = basis[cursor]
        cursor += 1
        relation_basis = basis[cursor : cursor + N_RELATIONS]
        cursor += N_RELATIONS
        payload_basis = basis[cursor : cursor + N_ITEMS]
        cursor += N_ITEMS
        self.address = basis[cursor : cursor + N_ITEMS]

        rel_coeff = simplex(N_RELATIONS, device)
        self.relations = rel_coeff @ relation_basis
        payload_rows = []
        payload_simplex = simplex(ITEMS_PER_RELATION, device)
        context_generator = torch.Generator(device="cpu")
        context_generator.manual_seed(seed + 1000 + context)
        for relation in range(N_RELATIONS):
            block = payload_basis[
                relation * ITEMS_PER_RELATION : (relation + 1) * ITEMS_PER_RELATION
            ]
            if context == 1:
                local_raw = torch.randn(
                    ITEMS_PER_RELATION,
                    ITEMS_PER_RELATION,
                    generator=context_generator,
                    dtype=torch.float64,
                )
                local_q, _ = torch.linalg.qr(local_raw)
                block = local_q.to(dtype=torch.float32, device=device) @ block
            payload_rows.append(payload_simplex @ block)
        self.payloads = torch.cat(payload_rows, dim=0)

        transform_generator = torch.Generator(device="cpu")
        transform_generator.manual_seed(seed + 2000)
        reflections = torch.randn(N_LAYERS, width, generator=transform_generator, dtype=torch.float64)
        reflections = reflections / torch.linalg.vector_norm(reflections, dim=1, keepdim=True)
        self.reflections = reflections.to(dtype=torch.float32, device=device)
        self.final_components = {
            "address": 1.20 * self.address,
            "shared": 0.80 * self.shared,
            "relation": 0.65 * self.relations,
            "payload": 1.00 * self.payloads,
        }
        self.injections = self._build_injections()
        self.initial_address = self._backward(self.final_components["address"], 0)
        self.readout = (
            0.90 * self.shared.unsqueeze(0)
            + 0.75 * self.relations[
                torch.tensor([relation_of(item) for item in range(N_ITEMS)], device=device)
            ]
            + 1.10 * self.payloads
            + 0.85 * self.address
        )
        self.bias = -torch.sum(self.readout * self.final_components["address"], dim=1)
        self.permutation = derangement_within_relations(seed + 3000)
        self.inverse_permutation = [0] * N_ITEMS
        for donor, payload_id in enumerate(self.permutation):
            self.inverse_permutation[payload_id] = donor

    def _backward(self, final_values: torch.Tensor, start_layer: int) -> torch.Tensor:
        result = final_values
        for layer in range(N_LAYERS - 1, start_layer - 1, -1):
            result = householder(result, self.reflections[layer])
        return result

    def _injection_at(self, final_values: torch.Tensor, layer: int) -> torch.Tensor:
        result = final_values
        for future in range(N_LAYERS - 1, layer, -1):
            result = householder(result, self.reflections[future])
        return result

    def _build_injections(self) -> dict[str, torch.Tensor]:
        return {
            "shared": self._injection_at(self.final_components["shared"], SHARED_LAYER),
            "relation": self._injection_at(self.final_components["relation"], RELATION_LAYER),
            "payload": self._injection_at(self.final_components["payload"], PAYLOAD_LAYER),
        }

    def states(
        self,
        sign: float,
        payload_enabled: bool,
        nonlinear: bool,
        donor_permutation: bool = False,
    ) -> torch.Tensor:
        latent = self.initial_address.clone()
        captured = []
        relation_index = torch.tensor(
            [relation_of(item) for item in range(N_ITEMS)],
            device=self.device,
            dtype=torch.long,
        )
        payload_index = torch.arange(N_ITEMS, device=self.device)
        if donor_permutation:
            payload_index = torch.tensor(self.permutation, device=self.device, dtype=torch.long)
        for layer in range(N_LAYERS):
            latent = householder(latent, self.reflections[layer])
            if layer == SHARED_LAYER:
                latent = latent + sign * self.injections["shared"].unsqueeze(0)
            if layer == RELATION_LAYER:
                latent = latent + sign * self.injections["relation"][relation_index]
            if layer == PAYLOAD_LAYER and payload_enabled:
                latent = latent + sign * self.injections["payload"][payload_index]
            captured.append(encode(latent, nonlinear))
        return torch.stack(captured, dim=1)

    def continue_from(
        self,
        observed: torch.Tensor,
        layer: int,
        payload_enabled: bool,
        nonlinear: bool,
    ) -> torch.Tensor:
        latent = decode(observed, nonlinear)
        target_relations = torch.tensor(
            [relation_of(item) for item in range(N_ITEMS)],
            device=self.device,
            dtype=torch.long,
        )
        for future in range(layer + 1, N_LAYERS):
            latent = householder(latent, self.reflections[future])
            if future == SHARED_LAYER:
                latent = latent - self.injections["shared"].view(1, 1, -1)
            if future == RELATION_LAYER:
                term = self.injections["relation"][target_relations]
                latent = latent - term[:, None, :]
            if future == PAYLOAD_LAYER and payload_enabled:
                term = self.injections["payload"]
                latent = latent - term[:, None, :]
        margins = torch.einsum("ijd,id->ij", latent, self.readout) + self.bias[:, None]
        return margins

    def natural_margins(
        self,
        states: torch.Tensor,
        nonlinear: bool,
    ) -> torch.Tensor:
        latent = decode(states[:, -1, :], nonlinear)
        return torch.sum(latent * self.readout, dim=1) + self.bias


def effect_matrix(
    system: ControlledResidualSystem,
    scenario: str,
    method: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload_enabled = scenario_payload(scenario)
    nonlinear = scenario_nonlinear(scenario)
    permuted = scenario == "linear_permuted_payload"
    active_minus = system.states(-1.0, payload_enabled, nonlinear, donor_permutation=False)
    active_plus_target = system.states(+1.0, payload_enabled, nonlinear, donor_permutation=False)
    active_plus_donor = system.states(+1.0, payload_enabled, nonlinear, donor_permutation=permuted)
    active_minus_donor = system.states(-1.0, payload_enabled, nonlinear, donor_permutation=permuted)
    null_minus = system.states(-1.0, False, nonlinear, donor_permutation=False)
    null_plus = system.states(+1.0, False, nonlinear, donor_permutation=False)

    target_base = active_minus[:, layer, :]
    if method == "absolute_state":
        patched = active_plus_donor[:, layer, :].unsqueeze(0).expand(N_ITEMS, -1, -1)
    elif method == "raw_counterfactual_difference":
        donor_delta = active_plus_donor[:, layer, :] - active_minus_donor[:, layer, :]
        patched = target_base[:, None, :] + donor_delta[None, :, :]
    elif method == "matched_double_difference":
        null_target_delta = null_plus[:, layer, :] - null_minus[:, layer, :]
        donor_increment = (
            active_plus_donor[:, layer, :]
            - active_minus_donor[:, layer, :]
            - (null_plus[:, layer, :] - null_minus[:, layer, :])
        )
        patched = (
            target_base[:, None, :]
            + null_target_delta[:, None, :]
            + donor_increment[None, :, :]
        )
    else:
        raise ValueError(method)

    endpoint = system.continue_from(patched, layer, payload_enabled, nonlinear)
    baseline = system.natural_margins(active_minus, nonlinear)
    target_positive = system.natural_margins(active_plus_target, nonlinear)
    effect = endpoint - baseline[:, None]
    return effect, endpoint, target_positive


def expected_mapping(system: ControlledResidualSystem, scenario: str) -> list[int] | None:
    if scenario in {"linear_shared_only", "nonlinear_shared_only"}:
        return None
    if scenario == "linear_permuted_payload":
        return list(system.inverse_permutation)
    return list(range(N_ITEMS))


def matrix_metrics(
    effect: np.ndarray,
    endpoint: np.ndarray,
    target_positive: np.ndarray,
    expected: list[int] | None,
) -> dict[str, Any]:
    finite = np.isfinite(effect) & np.isfinite(endpoint)
    result: dict[str, Any] = {
        "finite_fraction": float(np.mean(finite)),
        "within_relation_spread_max": 0.0,
        "same_relation_advantage_median": 0.0,
        "same_relation_advantage_positive_fraction": 0.0,
        "expected_top1_within_relation_fraction": 0.0,
        "identity_top1_within_relation_fraction": 0.0,
        "expected_endpoint_flip_fraction": 0.0,
        "expected_reconstruction_max_abs_error": None,
    }
    spreads = []
    identity_top1 = []
    for target in range(N_ITEMS):
        relation = relation_of(target)
        donors = list(
            range(relation * ITEMS_PER_RELATION, (relation + 1) * ITEMS_PER_RELATION)
        )
        values = effect[target, donors]
        spreads.append(float(np.max(values) - np.min(values)))
        own = donors.index(target)
        identity_top1.append(
            bool(values[own] > np.max(np.delete(values, own)) + EPSILON)
        )
    result["within_relation_spread_max"] = float(np.max(spreads))
    result["identity_top1_within_relation_fraction"] = float(np.mean(identity_top1))
    if expected is None:
        return result

    advantages = []
    top1 = []
    flips = []
    reconstruction = []
    for target, donor in enumerate(expected):
        relation = relation_of(target)
        donors = [
            candidate
            for candidate in range(
                relation * ITEMS_PER_RELATION,
                (relation + 1) * ITEMS_PER_RELATION,
            )
            if candidate != donor
        ]
        expected_value = float(effect[target, donor])
        other_values = [float(effect[target, candidate]) for candidate in donors]
        advantages.append(expected_value - median(other_values))
        top1.append(expected_value > max(other_values) + EPSILON)
        flips.append(float(endpoint[target, donor]) > 0.0)
        reconstruction.append(abs(float(endpoint[target, donor]) - float(target_positive[target])))
    result.update(
        {
            "same_relation_advantage_median": median(advantages),
            "same_relation_advantage_positive_fraction": float(
                np.mean(np.asarray(advantages) > 0.0)
            ),
            "expected_top1_within_relation_fraction": float(np.mean(top1)),
            "expected_endpoint_flip_fraction": float(np.mean(flips)),
            "expected_reconstruction_max_abs_error": float(np.max(reconstruction)),
        }
    )
    return result


def positive_cell_pass(metrics: dict[str, Any], threshold: dict[str, float]) -> bool:
    return bool(
        metrics["finite_fraction"] >= threshold["finite_fraction"]
        and metrics["expected_top1_within_relation_fraction"]
        >= threshold["positive_expected_top1_within_relation_fraction"]
        and metrics["same_relation_advantage_median"]
        >= threshold["positive_expected_advantage_same_relation_median"]
        and metrics["same_relation_advantage_positive_fraction"]
        >= threshold["positive_expected_advantage_positive_fraction"]
        and metrics["expected_endpoint_flip_fraction"]
        >= threshold["positive_expected_endpoint_flip_fraction"]
    )


def run_command(split: str) -> None:
    prereg = verify_protocol()
    if split not in SPLITS:
        raise ValueError(split)
    if split == "confirmation":
        selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
        if not selection["candidate_qualified"]:
            raise RuntimeError("confirmation is not authorized")
    run_dir = OUT_ROOT / "runs" / split
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing split: {split}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1143 requires CUDA")
    device = torch.device("cuda")
    configs = list(prereg["design"]["configs"][split])
    shape = (
        len(configs),
        len(CONTEXTS),
        len(SCENARIOS),
        len(METHODS),
        N_LAYERS,
        N_ITEMS,
        N_ITEMS,
    )
    effects = np.empty(shape, dtype=np.float32)
    endpoints = np.empty(shape, dtype=np.float32)
    positive_endpoints = np.empty(
        (len(configs), len(CONTEXTS), len(SCENARIOS), N_ITEMS), dtype=np.float32
    )
    rows: list[dict[str, Any]] = []
    permutations: dict[str, list[int]] = {}
    for config_index, config in enumerate(configs):
        for context_index, context in enumerate(CONTEXTS):
            system = ControlledResidualSystem(
                width=int(config["width"]),
                seed=int(config["seed"]),
                context=int(context),
                device=device,
            )
            permutations[f"{config['config_id']}.context{context}"] = system.permutation
            for scenario_index, scenario in enumerate(SCENARIOS):
                expected = expected_mapping(system, scenario)
                for method_index, method in enumerate(METHODS):
                    for layer in range(N_LAYERS):
                        effect, endpoint, target_positive = effect_matrix(
                            system, scenario, method, layer
                        )
                        effect_np = effect.detach().cpu().numpy().astype(np.float32)
                        endpoint_np = endpoint.detach().cpu().numpy().astype(np.float32)
                        target_np = target_positive.detach().cpu().numpy().astype(np.float32)
                        effects[
                            config_index,
                            context_index,
                            scenario_index,
                            method_index,
                            layer,
                        ] = effect_np
                        endpoints[
                            config_index,
                            context_index,
                            scenario_index,
                            method_index,
                            layer,
                        ] = endpoint_np
                        positive_endpoints[
                            config_index,
                            context_index,
                            scenario_index,
                        ] = target_np
                        metrics = matrix_metrics(effect_np, endpoint_np, target_np, expected)
                        rows.append(
                            {
                                "split": split,
                                "config_id": config["config_id"],
                                "width": int(config["width"]),
                                "seed": int(config["seed"]),
                                "context": int(context),
                                "scenario": scenario,
                                "method": method,
                                "layer": layer,
                                **metrics,
                            }
                        )
            del system
            torch.cuda.empty_cache()
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run_dir / "raw_matrices.npz",
        effects=effects,
        endpoints=endpoints,
        positive_endpoints=positive_endpoints,
    )
    write_jsonl(run_dir / "metrics.jsonl", rows)
    metadata = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "protocol_digest": prereg["protocol_digest"],
        "dimensions": {
            "configs": configs,
            "contexts": list(CONTEXTS),
            "scenarios": list(SCENARIOS),
            "methods": list(METHODS),
            "layers": list(range(N_LAYERS)),
            "items": list(range(N_ITEMS)),
        },
        "permutations": permutations,
        "raw_shape": list(shape),
        "metric_row_count": len(rows),
    }
    write_json(run_dir / "metadata.json", metadata)
    summary = summarize_split(rows, prereg["thresholds"])
    summary["raw_sha256"] = sha256_file(run_dir / "raw_matrices.npz")
    summary["metrics_sha256"] = sha256_file(run_dir / "metrics.jsonl")
    summary["metadata_sha256"] = sha256_file(run_dir / "metadata.json")
    summary["summary_digest"] = digest(summary)
    write_json(run_dir / "summary.json", summary)
    print(canonical(summary))


def summarize_split(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {
        (
            row["config_id"],
            int(row["context"]),
            row["scenario"],
            row["method"],
            int(row["layer"]),
        ): row
        for row in rows
    }
    config_ids = sorted({row["config_id"] for row in rows})
    cell_results = []
    for config_id in config_ids:
        for context in CONTEXTS:
            method = "matched_double_difference"
            onset_by_scenario = {}
            for scenario in ("linear_payload", "nonlinear_payload"):
                passing = [
                    layer
                    for layer in range(N_LAYERS)
                    if positive_cell_pass(
                        lookup[(config_id, context, scenario, method, layer)], thresholds
                    )
                ]
                onset_by_scenario[scenario] = min(passing) if passing else None
            late_positive = all(
                positive_cell_pass(
                    lookup[(config_id, context, scenario, method, N_LAYERS - 1)],
                    thresholds,
                )
                for scenario in ("linear_payload", "nonlinear_payload")
            )
            negative_rows = [
                lookup[(config_id, context, scenario, method, layer)]
                for scenario in ("linear_shared_only", "nonlinear_shared_only")
                for layer in range(N_LAYERS)
            ]
            negative_clean = all(
                abs(float(row["same_relation_advantage_median"]))
                <= thresholds["negative_max_abs_same_relation_advantage"]
                and float(row["within_relation_spread_max"])
                <= thresholds["negative_max_within_relation_spread"]
                for row in negative_rows
            )
            permuted = lookup[
                (config_id, context, "linear_permuted_payload", method, N_LAYERS - 1)
            ]
            permuted_pass = bool(
                permuted["expected_top1_within_relation_fraction"]
                >= thresholds["permuted_expected_top1_within_relation_fraction"]
                and permuted["identity_top1_within_relation_fraction"]
                <= thresholds["permuted_identity_top1_within_relation_fraction_max"]
            )
            reconstruction_rows = [
                lookup[(config_id, context, scenario, method, layer)]
                for scenario in ("linear_payload", "nonlinear_payload")
                for layer in range(PAYLOAD_LAYER, N_LAYERS)
            ]
            reconstruction_pass = all(
                row["expected_reconstruction_max_abs_error"] is not None
                and float(row["expected_reconstruction_max_abs_error"])
                <= thresholds["mdd_diagonal_reconstruction_max_abs_error"]
                for row in reconstruction_rows
            )
            onset_pass = all(
                onset == PAYLOAD_LAYER for onset in onset_by_scenario.values()
            )
            passed = bool(
                late_positive
                and negative_clean
                and permuted_pass
                and reconstruction_pass
                and onset_pass
            )
            cell_results.append(
                {
                    "config_id": config_id,
                    "context": context,
                    "payload_onset_by_scenario": onset_by_scenario,
                    "late_positive_pass": late_positive,
                    "negative_clean_pass": negative_clean,
                    "permuted_mapping_pass": permuted_pass,
                    "diagonal_reconstruction_pass": reconstruction_pass,
                    "cell_pass": passed,
                }
            )

    absolute_false_positive_rows = [
        row
        for row in rows
        if row["method"] == "absolute_state"
        and row["scenario"] in {"linear_shared_only", "nonlinear_shared_only"}
        and int(row["layer"]) == N_LAYERS - 1
    ]
    raw_false_positive_rows = [
        row
        for row in rows
        if row["method"] == "raw_counterfactual_difference"
        and row["scenario"] == "nonlinear_shared_only"
        and int(row["layer"]) == N_LAYERS - 1
    ]
    return {
        "phase": PHASE,
        "split": rows[0]["split"],
        "metric_row_count": len(rows),
        "cell_count": len(cell_results),
        "cell_pass_count": sum(bool(row["cell_pass"]) for row in cell_results),
        "all_cells_pass": all(bool(row["cell_pass"]) for row in cell_results),
        "cell_results": cell_results,
        "diagnostics": {
            "absolute_shared_only_identity_top1_median": median(
                row["identity_top1_within_relation_fraction"]
                for row in absolute_false_positive_rows
            ),
            "absolute_shared_only_advantage_median": median(
                row["same_relation_advantage_median"]
                for row in absolute_false_positive_rows
            ),
            "raw_nonlinear_shared_only_identity_top1_median": median(
                row["identity_top1_within_relation_fraction"]
                for row in raw_false_positive_rows
            ),
            "raw_nonlinear_shared_only_advantage_median": median(
                row["same_relation_advantage_median"]
                for row in raw_false_positive_rows
            ),
        },
    }


def selection_command() -> None:
    prereg = verify_protocol()
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    candidate_qualified = bool(summary["all_cells_pass"])
    selection = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": prereg["protocol_digest"],
        "candidate": "matched_double_difference",
        "candidate_qualified": candidate_qualified,
        "confirmation_authorized": candidate_qualified,
        "natural_model_authorized": False,
        "reason": (
            "ground-truth discovery calibration passed; independent confirmation required"
            if candidate_qualified
            else "ground-truth discovery calibration failed"
        ),
        "discovery_summary_digest": summary["summary_digest"],
    }
    body = dict(selection)
    selection["selection_digest"] = digest(body)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", selection)
    print(canonical(selection))


def finalize_command() -> None:
    prereg = verify_protocol()
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    confirmation_path = OUT_ROOT / "runs/confirmation/summary.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    calibration_passed = bool(
        selection["candidate_qualified"]
        and confirmation is not None
        and confirmation["all_cells_pass"]
    )
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "candidate": "matched_double_difference",
        "discovery_passed": bool(discovery["all_cells_pass"]),
        "confirmation_ran": confirmation is not None,
        "confirmation_passed": bool(confirmation and confirmation["all_cells_pass"]),
        "calibration_passed": calibration_passed,
        "outcome": (
            "controlled_calibration_confirmed"
            if calibration_passed
            else "controlled_calibration_not_confirmed"
        ),
        "natural_discovery_protocol_authorized": calibration_passed,
        "natural_hidden_scan_authorized": False,
        "component_search_authorized": False,
        "claim_scope": (
            "The MDD operator recovered a deliberately planted conditional payload and its onset in controlled systems, "
            "including an invertible nonlinear coordinate chart, while remaining null in payload-free controls. "
            "This calibrates an instrument family; it is not evidence that natural LLM states have the same decomposition."
        ),
        "discovery_summary_digest": discovery["summary_digest"],
        "confirmation_summary_digest": (
            confirmation["summary_digest"] if confirmation is not None else None
        ),
        "auto_continue": calibration_passed,
    }
    body = dict(final)
    final["final_digest"] = digest(body)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", choices=SPLITS, required=True)
    subparsers.add_parser("select")
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.split)
    elif args.command == "select":
        selection_command()
    elif args.command == "finalize":
        finalize_command()


if __name__ == "__main__":
    main()
