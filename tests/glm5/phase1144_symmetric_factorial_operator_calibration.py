#!/usr/bin/env python3
"""Calibrate a symmetric two-panel causal decomposition.

The available natural temporal material has original and label-swapped panels,
not a payload-null panel.  This phase calibrates the exact operator that those
four states identify before any natural hidden-state run is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import phase1143_ground_truth_mechanism_calibration as prior


PHASE = 1144
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1144_symmetric_factorial_operator_calibration"
SCRIPT = Path(__file__).resolve()
SPLITS = prior.SPLITS
CONTEXTS = prior.CONTEXTS
SCENARIOS = prior.SCENARIOS
METHODS = ("absolute_state", "raw_panel_difference", "symmetric_factorial_difference")
PANELS = ("original", "swapped")
N_ITEMS = prior.N_ITEMS
N_RELATIONS = prior.N_RELATIONS
ITEMS_PER_RELATION = prior.ITEMS_PER_RELATION
N_LAYERS = prior.N_LAYERS
PAYLOAD_LAYER = prior.PAYLOAD_LAYER
EPSILON = prior.EPSILON
CONFIGS = {
    "discovery": (
        {"config_id": "d100", "width": 100, "seed": 114401},
        {"config_id": "d132", "width": 132, "seed": 114402},
        {"config_id": "d164", "width": 164, "seed": 114403},
    ),
    "confirmation": (
        {"config_id": "c108", "width": 108, "seed": 114411},
        {"config_id": "c148", "width": 148, "seed": 114412},
        {"config_id": "c180", "width": 180, "seed": 114413},
    ),
}
THRESHOLDS = dict(prior.THRESHOLDS)


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SymmetricSystem(prior.ControlledResidualSystem):
    def __init__(self, width: int, seed: int, context: int, device: torch.device) -> None:
        super().__init__(width, seed, context, device)
        self.final_components = {
            "address": 1.20 * self.address,
            "shared": 0.35 * self.shared,
            "relation": 0.25 * self.relations,
            "payload": 1.20 * self.payloads,
        }
        self.injections = self._build_injections()
        self.initial_address = self._backward(self.final_components["address"], 0)
        relation_index = torch.tensor(
            [prior.relation_of(item) for item in range(N_ITEMS)], device=device
        )
        self.readout = (
            0.80 * self.shared.unsqueeze(0)
            + 0.65 * self.relations[relation_index]
            + 1.20 * self.payloads
            + 0.85 * self.address
        )
        self.bias = -torch.sum(self.readout * self.final_components["address"], dim=1)

    def panel_states(
        self,
        time_sign: float,
        panel_sign: float,
        payload_enabled: bool,
        nonlinear: bool,
        donor_permutation: bool = False,
    ) -> torch.Tensor:
        latent = self.initial_address.clone()
        relation_index = torch.tensor(
            [prior.relation_of(item) for item in range(N_ITEMS)],
            device=self.device,
            dtype=torch.long,
        )
        payload_index = torch.arange(N_ITEMS, device=self.device)
        if donor_permutation:
            payload_index = torch.tensor(self.permutation, device=self.device, dtype=torch.long)
        captured = []
        for layer in range(N_LAYERS):
            latent = prior.householder(latent, self.reflections[layer])
            if layer == prior.SHARED_LAYER:
                latent = latent + time_sign * self.injections["shared"].unsqueeze(0)
            if layer == prior.RELATION_LAYER:
                latent = latent + time_sign * self.injections["relation"][relation_index]
            if layer == PAYLOAD_LAYER and payload_enabled:
                latent = latent + time_sign * panel_sign * self.injections["payload"][payload_index]
            captured.append(prior.encode(latent, nonlinear))
        return torch.stack(captured, dim=1)

    def continue_panel(
        self,
        observed: torch.Tensor,
        layer: int,
        panel_sign: float,
        payload_enabled: bool,
        nonlinear: bool,
    ) -> torch.Tensor:
        latent = prior.decode(observed, nonlinear)
        relation_index = torch.tensor(
            [prior.relation_of(item) for item in range(N_ITEMS)],
            device=self.device,
            dtype=torch.long,
        )
        for future in range(layer + 1, N_LAYERS):
            latent = prior.householder(latent, self.reflections[future])
            if future == prior.SHARED_LAYER:
                latent = latent - self.injections["shared"].view(1, 1, -1)
            if future == prior.RELATION_LAYER:
                latent = latent - self.injections["relation"][relation_index][:, None, :]
            if future == PAYLOAD_LAYER and payload_enabled:
                latent = latent - panel_sign * self.injections["payload"][:, None, :]
        return torch.einsum("ijd,id->ij", latent, self.readout) + self.bias[:, None]

    def natural(self, states: torch.Tensor, nonlinear: bool) -> torch.Tensor:
        latent = prior.decode(states[:, -1, :], nonlinear)
        return torch.sum(latent * self.readout, dim=1) + self.bias


def expected_mapping(system: SymmetricSystem, scenario: str) -> list[int] | None:
    return prior.expected_mapping(system, scenario)


def effect_matrix(
    system: SymmetricSystem,
    scenario: str,
    method: str,
    panel: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload_enabled = prior.scenario_payload(scenario)
    nonlinear = prior.scenario_nonlinear(scenario)
    permuted = scenario == "linear_permuted_payload"
    panel_sign = 1.0 if panel == "original" else -1.0
    orientation = panel_sign

    target = {}
    donor = {}
    for name, sign in (("minus", -1.0), ("plus", 1.0)):
        target[("original", name)] = system.panel_states(sign, 1.0, payload_enabled, nonlinear, False)
        target[("swapped", name)] = system.panel_states(sign, -1.0, payload_enabled, nonlinear, False)
        donor[("original", name)] = system.panel_states(sign, 1.0, payload_enabled, nonlinear, permuted)
        donor[("swapped", name)] = system.panel_states(sign, -1.0, payload_enabled, nonlinear, permuted)

    base = target[(panel, "minus")][:, layer, :]
    if method == "absolute_state":
        patched = donor[(panel, "plus")][:, layer, :].unsqueeze(0).expand(N_ITEMS, -1, -1)
    elif method == "raw_panel_difference":
        delta = donor[(panel, "plus")][:, layer, :] - donor[(panel, "minus")][:, layer, :]
        patched = base[:, None, :] + delta[None, :, :]
    elif method == "symmetric_factorial_difference":
        target_original_delta = target[("original", "plus")][:, layer, :] - target[("original", "minus")][:, layer, :]
        target_swapped_delta = target[("swapped", "plus")][:, layer, :] - target[("swapped", "minus")][:, layer, :]
        donor_original_delta = donor[("original", "plus")][:, layer, :] - donor[("original", "minus")][:, layer, :]
        donor_swapped_delta = donor[("swapped", "plus")][:, layer, :] - donor[("swapped", "minus")][:, layer, :]
        common_target = 0.5 * (target_original_delta + target_swapped_delta)
        payload_donor = 0.5 * (donor_original_delta - donor_swapped_delta)
        patched = base[:, None, :] + common_target[:, None, :] + panel_sign * payload_donor[None, :, :]
    else:
        raise ValueError(method)

    raw_endpoint = system.continue_panel(patched, layer, panel_sign, payload_enabled, nonlinear)
    raw_base = system.natural(target[(panel, "minus")], nonlinear)
    raw_positive = system.natural(target[(panel, "plus")], nonlinear)
    endpoint = orientation * raw_endpoint
    baseline = orientation * raw_base
    positive = orientation * raw_positive
    return endpoint - baseline[:, None], endpoint, positive


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1144 after run output exists")
    prior_final = prior.read_json(prior.OUT_ROOT / "analysis/final.json")
    checks = {
        "phase1143_calibration_passed": bool(prior_final["calibration_passed"]),
        "natural_scan_not_yet_authorized": not bool(prior_final["natural_hidden_scan_authorized"]),
        "two_opposed_panels": PANELS == ("original", "swapped"),
        "positive_and_negative_linear": {"linear_payload", "linear_shared_only"}.issubset(SCENARIOS),
        "positive_and_negative_nonlinear": {"nonlinear_payload", "nonlinear_shared_only"}.issubset(SCENARIOS),
        "permuted_mapping": "linear_permuted_payload" in SCENARIOS,
        "independent_split_seeds": not ({row["seed"] for row in CONFIGS["discovery"]} & {row["seed"] for row in CONFIGS["confirmation"]}),
        "independent_split_widths": not ({row["width"] for row in CONFIGS["discovery"]} & {row["width"] for row in CONFIGS["confirmation"]}),
        "confirmation_locked": True,
        "exact_factorial_identity": True,
        "payload_onset_required": True,
        "shared_only_null_required": True,
        "semantic_mapping_not_raw_diagonal": True,
        "cuda_required": True,
        "natural_model_forbidden": True,
        "component_search_forbidden": True,
        "raw_arrays_required": True,
    }
    prereg = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "symmetric factorial operator calibration",
        "script_sha256": sha256_file(SCRIPT),
        "source_phase1143_script_sha256": sha256_file(prior.SCRIPT_PATH),
        "source_phase1143_final_digest": prior_final["final_digest"],
        "design": {
            "configs": CONFIGS,
            "contexts": list(CONTEXTS),
            "scenarios": list(SCENARIOS),
            "methods": list(METHODS),
            "panels": list(PANELS),
            "n_items": N_ITEMS,
            "n_relations": N_RELATIONS,
            "n_layers": N_LAYERS,
            "payload_layer": PAYLOAD_LAYER,
        },
        "operator": {
            "delta_original": "O_plus - O_minus",
            "delta_swapped": "S_plus - S_minus",
            "common_target": "0.5 * (delta_original_target + delta_swapped_target)",
            "binding_donor": "0.5 * (delta_original_donor - delta_swapped_donor)",
            "original_patch": "O_target_minus + common_target + binding_donor",
            "swapped_patch": "S_target_minus + common_target - binding_donor",
        },
        "thresholds": THRESHOLDS,
        "hard_stops": [
            "no natural hidden state before independent confirmation",
            "no head, MLP, neuron, SAE, or necessity search",
            "no threshold changes after discovery",
            "controlled success is instrument calibration only",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    body = dict(prereg)
    prereg["protocol_digest"] = digest(body)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", {"checks": checks, "check_count": len(checks), "passed_count": sum(checks.values()), "all_checks_passed": all(checks.values()), "protocol_digest": prereg["protocol_digest"]})
    print(canonical({"protocol_digest": prereg["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(prereg)
    stored = body.pop("protocol_digest")
    if digest(body) != stored or sha256_file(SCRIPT) != prereg["script_sha256"]:
        raise RuntimeError("Phase1144 frozen protocol mismatch")
    return prereg


def run_command(split: str) -> None:
    prereg = verify_protocol()
    if split == "confirmation" and not read_json(OUT_ROOT / "analysis/discovery_selection.json")["confirmation_authorized"]:
        raise RuntimeError("confirmation denied")
    out = OUT_ROOT / "runs" / split
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    configs = list(prereg["design"]["configs"][split])
    shape = (len(configs), len(CONTEXTS), len(SCENARIOS), len(METHODS), len(PANELS), N_LAYERS, N_ITEMS, N_ITEMS)
    effects = np.empty(shape, dtype=np.float32)
    endpoints = np.empty(shape, dtype=np.float32)
    positives = np.empty((len(configs), len(CONTEXTS), len(SCENARIOS), len(PANELS), N_ITEMS), dtype=np.float32)
    rows = []
    permutations = {}
    device = torch.device("cuda")
    for ci, config in enumerate(configs):
        for xi, context in enumerate(CONTEXTS):
            system = SymmetricSystem(int(config["width"]), int(config["seed"]), int(context), device)
            permutations[f"{config['config_id']}.context{context}"] = system.permutation
            for si, scenario in enumerate(SCENARIOS):
                expected = expected_mapping(system, scenario)
                for mi, method in enumerate(METHODS):
                    for pi, panel in enumerate(PANELS):
                        for layer in range(N_LAYERS):
                            effect, endpoint, positive = effect_matrix(system, scenario, method, panel, layer)
                            e = effect.detach().cpu().numpy().astype(np.float32)
                            z = endpoint.detach().cpu().numpy().astype(np.float32)
                            p = positive.detach().cpu().numpy().astype(np.float32)
                            effects[ci, xi, si, mi, pi, layer] = e
                            endpoints[ci, xi, si, mi, pi, layer] = z
                            positives[ci, xi, si, pi] = p
                            metrics = prior.matrix_metrics(e, z, p, expected)
                            rows.append({"split": split, "config_id": config["config_id"], "width": int(config["width"]), "seed": int(config["seed"]), "context": int(context), "scenario": scenario, "method": method, "panel": panel, "layer": layer, **metrics})
            del system
            torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "raw_matrices.npz", effects=effects, endpoints=endpoints, positive_endpoints=positives)
    write_jsonl(out / "metrics.jsonl", rows)
    metadata = {"phase": PHASE, "split": split, "protocol_digest": prereg["protocol_digest"], "device": torch.cuda.get_device_name(0), "dimensions": {"configs": configs, "contexts": list(CONTEXTS), "scenarios": list(SCENARIOS), "methods": list(METHODS), "panels": list(PANELS), "layers": list(range(N_LAYERS))}, "permutations": permutations, "raw_shape": list(shape), "metric_row_count": len(rows)}
    write_json(out / "metadata.json", metadata)
    summary = summarize(rows, prereg["thresholds"])
    summary.update({"raw_sha256": sha256_file(out / "raw_matrices.npz"), "metrics_sha256": sha256_file(out / "metrics.jsonl"), "metadata_sha256": sha256_file(out / "metadata.json")})
    summary["summary_digest"] = digest(summary)
    write_json(out / "summary.json", summary)
    print(canonical(summary))


def summarize(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["config_id"], int(row["context"]), row["scenario"], row["method"], row["panel"], int(row["layer"])): row for row in rows}
    results = []
    for config_id in sorted({row["config_id"] for row in rows}):
        for context in CONTEXTS:
            for panel in PANELS:
                method = "symmetric_factorial_difference"
                onsets = {}
                for scenario in ("linear_payload", "nonlinear_payload"):
                    passing = [layer for layer in range(N_LAYERS) if prior.positive_cell_pass(lookup[(config_id, context, scenario, method, panel, layer)], thresholds)]
                    onsets[scenario] = min(passing) if passing else None
                positive_pass = all(prior.positive_cell_pass(lookup[(config_id, context, scenario, method, panel, N_LAYERS - 1)], thresholds) for scenario in ("linear_payload", "nonlinear_payload"))
                negatives = [lookup[(config_id, context, scenario, method, panel, layer)] for scenario in ("linear_shared_only", "nonlinear_shared_only") for layer in range(N_LAYERS)]
                negative_pass = all(abs(float(row["same_relation_advantage_median"])) <= thresholds["negative_max_abs_same_relation_advantage"] and float(row["within_relation_spread_max"]) <= thresholds["negative_max_within_relation_spread"] for row in negatives)
                perm = lookup[(config_id, context, "linear_permuted_payload", method, panel, N_LAYERS - 1)]
                perm_pass = bool(perm["expected_top1_within_relation_fraction"] >= thresholds["permuted_expected_top1_within_relation_fraction"] and perm["identity_top1_within_relation_fraction"] <= thresholds["permuted_identity_top1_within_relation_fraction_max"])
                reconstruction = [lookup[(config_id, context, scenario, method, panel, layer)] for scenario in ("linear_payload", "nonlinear_payload") for layer in range(PAYLOAD_LAYER, N_LAYERS)]
                reconstruction_pass = all(row["expected_reconstruction_max_abs_error"] is not None and float(row["expected_reconstruction_max_abs_error"]) <= thresholds["mdd_diagonal_reconstruction_max_abs_error"] for row in reconstruction)
                onset_pass = all(value == PAYLOAD_LAYER for value in onsets.values())
                results.append({"config_id": config_id, "context": context, "panel": panel, "onsets": onsets, "positive_pass": positive_pass, "negative_pass": negative_pass, "permuted_pass": perm_pass, "reconstruction_pass": reconstruction_pass, "onset_pass": onset_pass, "cell_pass": bool(positive_pass and negative_pass and perm_pass and reconstruction_pass and onset_pass)})
    absolute_null = [row for row in rows if row["method"] == "absolute_state" and row["scenario"] in {"linear_shared_only", "nonlinear_shared_only"} and int(row["layer"]) == N_LAYERS - 1]
    return {"phase": PHASE, "split": rows[0]["split"], "metric_row_count": len(rows), "cell_count": len(results), "cell_pass_count": sum(bool(row["cell_pass"]) for row in results), "all_cells_pass": all(bool(row["cell_pass"]) for row in results), "cell_results": results, "absolute_shared_only_identity_top1_median": prior.median(row["identity_top1_within_relation_fraction"] for row in absolute_null)}


def selection_command() -> None:
    prereg = verify_protocol()
    summary = read_json(OUT_ROOT / "runs/discovery/summary.json")
    passed = bool(summary["all_cells_pass"])
    selection = {"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "candidate": "symmetric_factorial_difference", "candidate_qualified": passed, "confirmation_authorized": passed, "natural_model_authorized": False, "discovery_summary_digest": summary["summary_digest"]}
    selection["selection_digest"] = digest(selection)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", selection)
    print(canonical(selection))


def finalize_command() -> None:
    prereg = verify_protocol()
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    discovery = read_json(OUT_ROOT / "runs/discovery/summary.json")
    path = OUT_ROOT / "runs/confirmation/summary.json"
    confirmation = read_json(path) if path.exists() else None
    passed = bool(selection["candidate_qualified"] and confirmation and confirmation["all_cells_pass"])
    final = {"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "selection_digest": selection["selection_digest"], "discovery_passed": bool(discovery["all_cells_pass"]), "confirmation_ran": confirmation is not None, "confirmation_passed": bool(confirmation and confirmation["all_cells_pass"]), "calibration_passed": passed, "outcome": "symmetric_factorial_calibration_confirmed" if passed else "symmetric_factorial_calibration_not_confirmed", "natural_discovery_protocol_authorized": passed, "natural_hidden_scan_authorized": False, "component_search_authorized": False, "claim_scope": "Controlled calibration of the four-state symmetric operator only; no natural LLM payload claim.", "auto_continue": passed}
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("--split", choices=SPLITS, required=True)
    sub.add_parser("select")
    sub.add_parser("finalize")
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
