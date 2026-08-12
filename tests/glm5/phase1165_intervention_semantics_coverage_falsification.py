#!/usr/bin/env python3
"""One-shot falsification of the Phase1164 coverage law across patch semantics.

The frozen max-lower-pair estimator is calibrated independently for four
intervention operators.  Predictions are sealed before any new triple,
quadruple, or quintuple schedule is executed in newly trained networks.

This phase tests an intervention-response law.  It does not identify the
unpatched natural computation and does not claim that a network computes max.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1165_intervention_semantics_coverage_falsification_audit.py"
P1164_SCRIPT = ROOT / "tests/glm5/phase1164_max_lower_pair_coverage_confirmation.py"
P1164_AUDIT = ROOT / "tests/glm5/phase1164_max_lower_pair_coverage_confirmation_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1165_intervention_semantics_coverage_falsification"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1164_max_lower_pair_coverage_confirmation as p1164  # noqa: E402


p1163 = p1164.p1163
p1161 = p1164.p1161
source = p1164.source
PHASE = 1165
FACTORS = source.FACTORS
ROLES = source.ROLES
ARCHITECTURES = source.ARCHITECTURES
REPLICATES = 4
SEMANTICS = (
    "residual_state_replace",
    "residual_delta_add",
    "attention_output_replace",
    "mlp_output_replace",
)
ALGORITHMS = ("pairwise", "max_single", "max_pair")
HOLDOUT_SEED = 1165007
HOLDOUT_QUAD_COUNT = 256
HOLDOUT_QUINT_COUNT = 256
SURFACE_CHUNK_SIZE = 32
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "denominator_min": 1e-5,
    "null_abs_max": 1e-8,
    "transport_unit_q95_min": 0.05,
    "transport_unit_count_min": 18,
    "transport_median_q95_min": 0.10,
    "coverage_median_unit_mae_max": 0.03,
    "coverage_median_relative_mae_max": 0.15,
    "coverage_unit_mae_max": 0.05,
    "coverage_unit_relative_mae_max": 0.25,
    "coverage_unit_count_min": 18,
    "coverage_pairwise_advantage_min": 0.005,
    "coverage_max_single_advantage_min": 0.002,
    "coverage_schedule_abs_error_q95_max": 0.10,
}


def model_seed(architecture: str, replicate: int) -> int:
    return 1165100 + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def model_id(seed: int) -> str:
    return p1163.digest({"phase": PHASE, "seed": seed})[:16]


def sites() -> list[dict[str, Any]]:
    return p1163.sites()


def calibration_subsets() -> list[tuple[int, ...]]:
    return p1163.calibration_subsets()


def prior_used_subsets() -> set[tuple[int, ...]]:
    return (
        set(p1161.discovery_holdout_subsets())
        | set(p1161.confirmation_holdout_subsets())
        | set(p1163.registry_subsets())
        | set(p1164.broad_holdout_subsets())
        | set(p1164.stress_subsets())
    )


def broad_holdout_subsets() -> list[tuple[int, ...]]:
    old = prior_used_subsets()
    rng = np.random.default_rng(HOLDOUT_SEED)
    triples = [
        row for row in itertools.combinations(range(len(sites())), 3) if row not in old
    ]
    quads = [
        row for row in itertools.combinations(range(len(sites())), 4) if row not in old
    ]
    quints = [
        row for row in itertools.combinations(range(len(sites())), 5) if row not in old
    ]
    quad_indices = sorted(
        rng.choice(len(quads), size=HOLDOUT_QUAD_COUNT, replace=False).tolist()
    )
    quint_indices = sorted(
        rng.choice(len(quints), size=HOLDOUT_QUINT_COUNT, replace=False).tolist()
    )
    return triples + [quads[int(index)] for index in quad_indices] + [
        quints[int(index)] for index in quint_indices
    ]


def stress_subsets() -> list[tuple[int, ...]]:
    return [p1163.QUERY_CHAIN, p1163.A_STAR, tuple(range(len(sites())))]


def all_test_subsets() -> list[tuple[int, ...]]:
    return broad_holdout_subsets() + stress_subsets()


def clean_trace(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    hidden = model.embed(input_ids)
    states = [hidden]
    attention_outputs: list[torch.Tensor] = []
    mlp_outputs: list[torch.Tensor] = []
    for block in model.blocks:
        attention_output = block.attn(block.attn_norm(hidden))
        hidden = hidden + attention_output
        mlp_output = block.mlp(block.mlp_norm(hidden))
        hidden = hidden + mlp_output
        attention_outputs.append(attention_output)
        mlp_outputs.append(mlp_output)
        states.append(hidden)
    return model.lm_head(model.final_norm(hidden)), states, attention_outputs, mlp_outputs


def patch_selected_sites(
    target: torch.Tensor,
    source_tensor: torch.Tensor,
    receiver_tensor: torch.Tensor | None,
    chunk_subsets: list[tuple[int, ...]],
    layer_index: int,
    actual_by_depth: dict[float, int],
    role_positions: dict[str, torch.Tensor],
    additive: bool,
) -> torch.Tensor:
    batch_size = int(source_tensor.shape[0])
    batch_index = torch.arange(batch_size, device=target.device)
    site_rows = sites()
    touched = False
    output = target
    for local_index, subset in enumerate(chunk_subsets):
        layer_sites = [
            site_index
            for site_index in subset
            if actual_by_depth[float(site_rows[site_index]["depth"])] == layer_index
        ]
        if not layer_sites:
            continue
        if not touched:
            output = output.clone()
            touched = True
        flat_batch = batch_index + local_index * batch_size
        for site_index in layer_sites:
            role = str(site_rows[site_index]["role"])
            token_positions = role_positions[role]
            donor_value = source_tensor[batch_index, token_positions]
            if additive:
                if receiver_tensor is None:
                    raise RuntimeError("additive patch requires a receiver reference")
                receiver_value = receiver_tensor[batch_index, token_positions]
                output[flat_batch, token_positions] += donor_value - receiver_value
            else:
                output[flat_batch, token_positions] = donor_value
    return output


def intervention_factor_surfaces(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    factor: str,
    subsets: list[tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = next(model.parameters()).device
    (
        receiver_cpu,
        donor_cpu,
        _control_cpu,
        receiver_target_cpu,
        donor_target_cpu,
        positions_cpu,
    ) = source.scan_batch(lexicon, "confirmation", factor)
    receiver = receiver_cpu.to(device)
    donor = donor_cpu.to(device)
    receiver_targets = receiver_target_cpu.to(device)
    donor_targets = donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    candidates = source.answer_ids(lexicon, device)
    role_positions = {role: positions[:, ROLES.index(role)] for role in ROLES}
    actual_by_depth = {
        depth: source.actual_depth_index(config, depth) for depth in p1161.INTERIOR_DEPTHS
    }
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw, receiver_states, _receiver_attention, _receiver_mlp = clean_trace(
            model, receiver
        )
        donor_raw, donor_states, donor_attention, donor_mlp = clean_trace(model, donor)
    receiver_logits = source.candidate_logits(receiver_raw, candidates)
    donor_logits = source.candidate_logits(donor_raw, candidates)
    base_margin = source.target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = source.target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    denominator_min = float(torch.min(denominator).item())
    if denominator_min <= THRESHOLDS["denominator_min"]:
        raise RuntimeError(f"nonpositive denominator: {factor}/{denominator_min}")

    values: dict[str, list[np.ndarray]] = {semantic: [] for semantic in SEMANTICS}
    batch_size = int(receiver.shape[0])
    for start in range(0, len(subsets), SURFACE_CHUNK_SIZE):
        chunk = subsets[start : start + SURFACE_CHUNK_SIZE]
        schedule_count = len(chunk)
        for semantic in SEMANTICS:
            hidden = model.embed(receiver).repeat(schedule_count, 1, 1)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for layer_index, block in enumerate(model.blocks, start=1):
                    attention_output = block.attn(block.attn_norm(hidden))
                    if semantic == "attention_output_replace":
                        attention_output = patch_selected_sites(
                            attention_output,
                            donor_attention[layer_index - 1],
                            None,
                            chunk,
                            layer_index,
                            actual_by_depth,
                            role_positions,
                            additive=False,
                        )
                    hidden = hidden + attention_output
                    mlp_output = block.mlp(block.mlp_norm(hidden))
                    if semantic == "mlp_output_replace":
                        mlp_output = patch_selected_sites(
                            mlp_output,
                            donor_mlp[layer_index - 1],
                            None,
                            chunk,
                            layer_index,
                            actual_by_depth,
                            role_positions,
                            additive=False,
                        )
                    hidden = hidden + mlp_output
                    if semantic == "residual_state_replace":
                        hidden = patch_selected_sites(
                            hidden,
                            donor_states[layer_index],
                            None,
                            chunk,
                            layer_index,
                            actual_by_depth,
                            role_positions,
                            additive=False,
                        )
                    elif semantic == "residual_delta_add":
                        hidden = patch_selected_sites(
                            hidden,
                            donor_states[layer_index],
                            receiver_states[layer_index],
                            chunk,
                            layer_index,
                            actual_by_depth,
                            role_positions,
                            additive=True,
                        )
                patched_raw = model.lm_head(model.final_norm(hidden))
            patched_logits = source.candidate_logits(patched_raw, candidates)
            patched_margin = source.target_margin(
                patched_logits, donor_targets.repeat(schedule_count), receiver_targets.repeat(schedule_count)
            )
            effect = ((patched_margin - base_margin.repeat(schedule_count)) / denominator.repeat(schedule_count))
            case_matrix = effect.float().reshape(schedule_count, batch_size).cpu().numpy()
            values[semantic].extend(case_matrix.astype(np.float32))
    matrices = {semantic: np.stack(rows, axis=0) for semantic, rows in values.items()}
    return matrices, {
        "case_count": batch_size,
        "denominator_min": denominator_min,
        "denominator_median": float(torch.median(denominator).item()),
        "finite_fraction": float(
            np.mean([np.isfinite(matrix).mean() for matrix in matrices.values()])
        ),
    }


def checkpoint_payload(
    model: torch.nn.Module, config: Any, lexicon: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "lexicon": lexicon,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = source.ModelConfig(**payload["config"])
    model = source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def prior_artifacts() -> dict[str, Any]:
    return {
        "final": p1163.read_json(p1164.OUT_ROOT / "analysis/final.json"),
        "audit": p1163.read_json(p1164.OUT_ROOT / "audit/independent_audit.json"),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1165 artifacts")
    prior = prior_artifacts()
    broad = broad_holdout_subsets()
    cardinalities = {size: sum(len(row) == size for row in broad) for size in (3, 4, 5)}
    checks = {
        "phase1164_confirmed": prior["final"]["coverage_response_rule_confirmed"],
        "phase1164_audit_passed": prior["audit"]["all_checks_passed"],
        "phase1164_branch_closed": prior["final"]["branch_status"] == "closed_after_independent_confirmation",
        "semantic_count": len(SEMANTICS) == 4,
        "calibration_count": len(calibration_subsets()) == 121,
        "holdout_count": len(broad) == cardinalities[3] + HOLDOUT_QUAD_COUNT + HOLDOUT_QUINT_COUNT,
        "all_unused_triples": cardinalities[3] == 67,
        "quad_count": cardinalities[4] == HOLDOUT_QUAD_COUNT,
        "quint_count": cardinalities[5] == HOLDOUT_QUINT_COUNT,
        "holdout_unique": len(broad) == len(set(broad)),
        "holdout_disjoint_from_prior": not bool(set(broad).intersection(prior_used_subsets())),
        "predictions_before_outcomes": True,
        "one_shot_semantic_axis": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": "one-shot intervention-semantics coverage falsification",
        "source_digests": {
            "phase1164_final": prior["final"]["final_digest"],
            "phase1164_audit": prior["audit"]["audit_digest"],
        },
        "source_hashes": {
            "primary_script": p1163.sha256_file(SCRIPT),
            "audit_script": p1163.sha256_file(AUDIT_SCRIPT),
            "phase1164_script": p1163.sha256_file(P1164_SCRIPT),
            "phase1164_audit": p1163.sha256_file(P1164_AUDIT),
        },
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_count": len(ARCHITECTURES) * REPLICATES,
        "factors": list(FACTORS),
        "semantics": list(SEMANTICS),
        "semantic_definitions": {
            "residual_state_replace": "replace the selected post-block token state with its matched clean-donor state",
            "residual_delta_add": "add the matched clean donor-minus-clean receiver post-block state difference without erasing accumulated receiver effects",
            "attention_output_replace": "replace only the selected attention residual-branch output, then naturally recompute MLP and later blocks",
            "mlp_output_replace": "replace only the selected MLP residual-branch output, then naturally recompute later blocks",
        },
        "common_normalizer": "natural matched-donor answer-margin change",
        "calibration_subsets": [list(row) for row in calibration_subsets()],
        "broad_holdout_subsets": [list(row) for row in broad],
        "stress_subsets": [list(row) for row in stress_subsets()],
        "holdout_cardinality_counts": cardinalities,
        "algorithms": list(ALGORITHMS),
        "primary_algorithm": "max_pair",
        "primary_formula": "max response over contained null/single/pair calibration subsets, fitted separately per intervention semantic",
        "thresholds": THRESHOLDS,
        "primary_endpoint": "determine whether the frozen low-order upper-envelope law survives non-overwriting and component-output interventions with identifiable transport",
        "allowed_mode_decisions": [
            "coverage_confirmed",
            "coverage_rejected",
            "uninformative_transport",
        ],
        "allowed_global_decisions": [
            "cross_semantic_coverage_confirmed",
            "coverage_not_general_across_semantics",
            "alternative_semantics_inconclusive",
            "positive_control_failed",
        ],
        "hard_stops": [
            "Predictions for every semantic are sealed before any holdout intervention is run.",
            "Each alternative semantic must pass a response-transport gate before its coverage result is interpretable.",
            "Failure cannot be repaired by selecting sites, semantics, models, schedules, thresholds, or a new order statistic.",
            "Success describes intervention responses only; it does not identify the natural forward algorithm or a literal max operator.",
            "This semantic-comparison branch closes after this one test regardless of outcome.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = p1163.digest(protocol)
    p1163.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    p1163.write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(checks.values()),
            "all_checks_passed": all(checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(p1163.canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if p1163.digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1164_script", P1164_SCRIPT),
        ("phase1164_audit", P1164_AUDIT),
    ):
        if p1163.sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    if protocol["broad_holdout_subsets"] != [list(row) for row in broad_holdout_subsets()]:
        raise RuntimeError("holdout registry drift")
    return protocol


def run_calibration_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/calibration"
    if root.exists():
        raise RuntimeError("refusing to overwrite calibration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_rows = []
    truth_rows = []
    training_rows = []
    diagnostic_rows = []
    model_arrays = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(architecture, replicate)
            identifier = model_id(seed)
            lexicon = source.make_lexicon(seed + 18017)
            model, training = source.train_model(config, seed, lexicon, device)
            if not training["qualified"]:
                raise RuntimeError(f"behavior gate failed: {identifier}")
            factor_arrays = []
            factor_diagnostics = {}
            for factor in FACTORS:
                matrices, detail = intervention_factor_surfaces(
                    model, config, lexicon, factor, calibration_subsets()
                )
                factor_arrays.append(
                    np.stack(
                        [np.median(matrices[semantic], axis=1) for semantic in SEMANTICS],
                        axis=0,
                    ).astype(np.float32)
                )
                factor_diagnostics[factor] = detail
            model_arrays.append(np.stack(factor_arrays, axis=0))
            public_rows.append({"model_id": identifier, "factor_count": len(FACTORS)})
            truth_rows.append(
                {
                    "model_id": identifier,
                    "architecture": architecture,
                    "replicate": replicate,
                    "seed": seed,
                    "lexicon_digest": p1163.digest(lexicon),
                }
            )
            training_rows.append({"model_id": identifier, **training})
            diagnostic_rows.append({"model_id": identifier, "factor": factor_diagnostics})
            checkpoint = root / "checkpoints" / f"{identifier}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload(model, config, lexicon), checkpoint)
            del model
            torch.cuda.empty_cache()
    response = np.stack(model_arrays, axis=0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "calibration_responses.npz", response=response)
    p1163.write_jsonl(root / "public_manifest.jsonl", public_rows)
    p1163.write_jsonl(root / "sealed_truth.jsonl", truth_rows)
    p1163.write_jsonl(root / "training_metrics.jsonl", training_rows)
    p1163.write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(public_rows) == protocol["model_count"],
        "all_models_qualified": all(row["qualified"] for row in training_rows),
        "behavior_accuracy": min(row["accuracy"] for row in training_rows)
        >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": min(row["minimum_probability"] for row in training_rows)
        >= THRESHOLDS["behavior_min_probability_min"],
        "response_shape": response.shape
        == (protocol["model_count"], len(FACTORS), len(SEMANTICS), len(calibration_subsets())),
        "finite": bool(np.isfinite(response).all()),
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "null": float(np.max(np.abs(response[:, :, :, 0]))) <= THRESHOLDS["null_abs_max"],
        "architecture_hidden": all("architecture" not in row for row in public_rows),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "response_shape": list(response.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "denominator_min": denominator_min,
        "null_max_abs": float(np.max(np.abs(response[:, :, :, 0]))),
        "calibration_pack_sha256": p1163.sha256_file(root / "calibration_responses.npz"),
        "checks": checks,
        "calibration_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def max_lower_prediction(
    calibration: np.ndarray,
    targets: list[tuple[int, ...]],
    maximum_order: int,
) -> np.ndarray:
    lookup = {subset: index for index, subset in enumerate(calibration_subsets())}
    prediction = np.zeros(calibration.shape[:-1] + (len(targets),), dtype=np.float64)
    for target_index, target in enumerate(targets):
        target_set = set(target)
        lower_indices = [
            index
            for subset, index in lookup.items()
            if len(subset) <= maximum_order and set(subset).issubset(target_set)
        ]
        prediction[..., target_index] = np.max(calibration[..., lower_indices], axis=-1)
    return prediction


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    summary = p1163.read_json(OUT_ROOT / "runs/calibration/summary.json")
    if not summary["calibration_gate_passed"]:
        raise RuntimeError("calibration gate failed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout outcomes already exist")
    prediction_root = OUT_ROOT / "predictions"
    if prediction_root.exists():
        raise RuntimeError("refusing to overwrite predictions")
    with np.load(OUT_ROOT / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    targets = all_test_subsets()
    predictions: dict[str, np.ndarray] = {}
    pairwise = np.zeros(calibration.shape[:-1] + (len(targets),), dtype=np.float64)
    for model_index in range(calibration.shape[0]):
        for factor_index in range(calibration.shape[1]):
            for semantic_index in range(calibration.shape[2]):
                coefficient = p1161.fit_coefficients(
                    "pairwise",
                    calibration_subsets(),
                    calibration[model_index, factor_index, semantic_index],
                )
                pairwise[model_index, factor_index, semantic_index] = p1161.predict_values(
                    "pairwise", coefficient, targets
                )
    predictions["pairwise"] = pairwise.astype(np.float32)
    predictions["max_single"] = max_lower_prediction(calibration, targets, 1).astype(np.float32)
    predictions["max_pair"] = max_lower_prediction(calibration, targets, 2).astype(np.float32)
    prediction_root.mkdir(parents=True)
    np.savez_compressed(prediction_root / "sealed_predictions.npz", **predictions)
    metadata = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "algorithms": list(ALGORITHMS),
        "semantics": list(SEMANTICS),
        "primary_algorithm": "max_pair",
        "test_subset_ids": [p1163.subset_id(row) for row in targets],
        "broad_count": len(broad_holdout_subsets()),
        "stress_count": len(stress_subsets()),
        "holdout_outcomes_absent_at_sealing": True,
        "calibration_pack_sha256": summary["calibration_pack_sha256"],
        "prediction_pack_sha256": p1163.sha256_file(prediction_root / "sealed_predictions.npz"),
    }
    metadata["prediction_digest"] = p1163.digest(metadata)
    p1163.write_json(prediction_root / "metadata.json", metadata)
    print(p1163.canonical(metadata))


def run_holdout_command() -> None:
    protocol = verify_protocol()
    metadata = p1163.read_json(OUT_ROOT / "predictions/metadata.json")
    if not metadata["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid prediction seal")
    root = OUT_ROOT / "runs/holdout"
    if root.exists():
        raise RuntimeError("refusing to overwrite holdout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    calibration_root = OUT_ROOT / "runs/calibration"
    public = p1163.read_jsonl(calibration_root / "public_manifest.jsonl")
    truth = p1163.read_jsonl(calibration_root / "sealed_truth.jsonl")
    targets = all_test_subsets()
    arrays = []
    diagnostic_rows = []
    for public_row, truth_row in zip(public, truth, strict=True):
        if public_row["model_id"] != truth_row["model_id"]:
            raise RuntimeError("manifest order mismatch")
        model, config, lexicon = load_checkpoint(
            calibration_root / "checkpoints" / f"{public_row['model_id']}.pt", device
        )
        factor_arrays = []
        factor_diagnostics = {}
        for factor in FACTORS:
            matrices, detail = intervention_factor_surfaces(model, config, lexicon, factor, targets)
            factor_arrays.append(
                np.stack(
                    [np.median(matrices[semantic], axis=1) for semantic in SEMANTICS], axis=0
                ).astype(np.float32)
            )
            factor_diagnostics[factor] = detail
        arrays.append(np.stack(factor_arrays, axis=0))
        diagnostic_rows.append({"model_id": public_row["model_id"], "factor": factor_diagnostics})
        del model
        torch.cuda.empty_cache()
    observed = np.stack(arrays, axis=0)
    root.mkdir(parents=True)
    np.savez_compressed(root / "holdout_responses.npz", response=observed)
    p1163.write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(arrays) == protocol["model_count"],
        "response_shape": observed.shape
        == (protocol["model_count"], len(FACTORS), len(SEMANTICS), len(targets)),
        "finite": bool(np.isfinite(observed).all()),
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "prediction_integrity": p1163.sha256_file(OUT_ROOT / "predictions/sealed_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "prediction_precedes_holdout": metadata["created_at_utc"] < p1163.now(),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "response_shape": list(observed.shape),
        "denominator_min": denominator_min,
        "holdout_pack_sha256": p1163.sha256_file(root / "holdout_responses.npz"),
        "checks": checks,
        "holdout_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def semantic_metrics(
    semantic_index: int,
    predictions: dict[str, np.ndarray],
    observed: np.ndarray,
    broad_count: int,
) -> dict[str, Any]:
    truth = observed[:, :, semantic_index, :broad_count]
    response_q95 = np.quantile(np.abs(truth), 0.95, axis=2)
    transport_units = response_q95 >= THRESHOLDS["transport_unit_q95_min"]
    transport_checks = {
        "median_q95": float(np.median(response_q95)) >= THRESHOLDS["transport_median_q95_min"],
        "unit_count": int(np.sum(transport_units)) >= THRESHOLDS["transport_unit_count_min"],
    }
    algorithm_results = {}
    for algorithm in ALGORITHMS:
        prediction = predictions[algorithm][:, :, semantic_index, :broad_count]
        error = np.abs(prediction - truth)
        unit_mae = np.mean(error, axis=2)
        relative_mae = unit_mae / np.maximum(response_q95, THRESHOLDS["transport_unit_q95_min"])
        algorithm_results[algorithm] = {
            "median_unit_mae": float(np.median(unit_mae)),
            "mean_unit_mae": float(np.mean(unit_mae)),
            "median_unit_relative_mae": float(np.median(relative_mae)),
            "schedule_abs_error_q95": float(np.quantile(error, 0.95)),
            "schedule_abs_error_max": float(np.max(error)),
            "unit_mae": unit_mae.tolist(),
            "unit_relative_mae": relative_mae.tolist(),
        }
    max_pair_mae = np.asarray(algorithm_results["max_pair"]["unit_mae"])
    max_pair_relative = np.asarray(algorithm_results["max_pair"]["unit_relative_mae"])
    pairwise_mae = np.asarray(algorithm_results["pairwise"]["unit_mae"])
    max_single_mae = np.asarray(algorithm_results["max_single"]["unit_mae"])
    coverage_units = (max_pair_mae <= THRESHOLDS["coverage_unit_mae_max"]) & (
        max_pair_relative <= THRESHOLDS["coverage_unit_relative_mae_max"]
    )
    coverage_checks = {
        "median_unit_mae": algorithm_results["max_pair"]["median_unit_mae"]
        <= THRESHOLDS["coverage_median_unit_mae_max"],
        "median_relative_mae": algorithm_results["max_pair"]["median_unit_relative_mae"]
        <= THRESHOLDS["coverage_median_relative_mae_max"],
        "unit_count": int(np.sum(coverage_units)) >= THRESHOLDS["coverage_unit_count_min"],
        "beats_pairwise": float(np.median(pairwise_mae - max_pair_mae))
        >= THRESHOLDS["coverage_pairwise_advantage_min"],
        "beats_max_single": float(np.median(max_single_mae - max_pair_mae))
        >= THRESHOLDS["coverage_max_single_advantage_min"],
        "schedule_q95": algorithm_results["max_pair"]["schedule_abs_error_q95"]
        <= THRESHOLDS["coverage_schedule_abs_error_q95_max"],
    }
    transport_identified = all(transport_checks.values())
    coverage_passed = all(coverage_checks.values())
    if not transport_identified:
        decision = "uninformative_transport"
    elif coverage_passed:
        decision = "coverage_confirmed"
    else:
        decision = "coverage_rejected"
    return {
        "decision": decision,
        "transport_identified": transport_identified,
        "coverage_passed": coverage_passed,
        "response_q95_median": float(np.median(response_q95)),
        "response_q95_unit_count": int(np.sum(transport_units)),
        "transport_checks": transport_checks,
        "coverage_checks": coverage_checks,
        "coverage_unit_count": int(np.sum(coverage_units)),
        "max_pair_pairwise_advantage_median": float(np.median(pairwise_mae - max_pair_mae)),
        "max_pair_max_single_advantage_median": float(np.median(max_single_mae - max_pair_mae)),
        "algorithm_results": algorithm_results,
    }


def calculate_results(
    predictions: dict[str, np.ndarray], observed: np.ndarray
) -> dict[str, Any]:
    broad_count = len(broad_holdout_subsets())
    mode_results = {
        semantic: semantic_metrics(index, predictions, observed, broad_count)
        for index, semantic in enumerate(SEMANTICS)
    }
    positive = mode_results["residual_state_replace"]
    alternatives = [mode_results[semantic] for semantic in SEMANTICS[1:]]
    if positive["decision"] != "coverage_confirmed":
        decision = "positive_control_failed"
    elif all(row["decision"] == "coverage_confirmed" for row in alternatives):
        decision = "cross_semantic_coverage_confirmed"
    elif any(row["decision"] == "coverage_rejected" for row in alternatives):
        decision = "coverage_not_general_across_semantics"
    else:
        decision = "alternative_semantics_inconclusive"
    return {
        "decision": decision,
        "mode_results": mode_results,
        "broad_schedule_count": broad_count,
        "stress_schedule_count": len(stress_subsets()),
        "claim_scope": "one-shot comparison of a frozen low-order upper-envelope response estimator across four matched intervention operators in one deterministic micro-task",
        "non_implications": [
            "A response-equivalent max estimator is not a literal neural max operation.",
            "Intervention response prediction does not recover the unpatched natural computation graph.",
            "Component-output patching does not isolate individual attention heads or MLP neurons.",
            "Results on freely trained micro-Transformers do not establish a pretrained language-model mechanism.",
        ],
    }


def score_command() -> None:
    protocol = verify_protocol()
    metadata = p1163.read_json(OUT_ROOT / "predictions/metadata.json")
    holdout_summary = p1163.read_json(OUT_ROOT / "runs/holdout/summary.json")
    if not holdout_summary["holdout_gate_passed"]:
        raise RuntimeError("holdout gate failed")
    with np.load(OUT_ROOT / "predictions/sealed_predictions.npz") as pack:
        predictions = {algorithm: np.asarray(pack[algorithm], dtype=np.float64) for algorithm in ALGORITHMS}
    with np.load(OUT_ROOT / "runs/holdout/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    results = calculate_results(predictions, observed)
    integrity_checks = {
        "prediction_integrity": p1163.sha256_file(OUT_ROOT / "predictions/sealed_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "holdout_integrity": p1163.sha256_file(OUT_ROOT / "runs/holdout/holdout_responses.npz")
        == holdout_summary["holdout_pack_sha256"],
        "prediction_precedes_holdout": metadata["created_at_utc"] < holdout_summary["created_at_utc"],
        "one_shot_branch_closed": True,
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"integrity checks failed: {integrity_checks}")
    score = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "results": results,
        "integrity_checks": integrity_checks,
        "branch_status": "closed_after_one_shot_semantic_comparison",
    }
    score["score_digest"] = p1163.digest(score)
    p1163.write_json(OUT_ROOT / "analysis/score.json", score)
    compact = {
        semantic: {
            "decision": row["decision"],
            "response_q95_median": row["response_q95_median"],
            "max_pair_mae": row["algorithm_results"]["max_pair"]["median_unit_mae"],
            "relative_mae": row["algorithm_results"]["max_pair"]["median_unit_relative_mae"],
            "pairwise_advantage": row["max_pair_pairwise_advantage_median"],
        }
        for semantic, row in results["mode_results"].items()
    }
    print(
        p1163.canonical(
            {"decision": results["decision"], "mode_results": compact, "score_digest": score["score_digest"]}
        )
    )


def finalize_command() -> None:
    protocol = verify_protocol()
    score = p1163.read_json(OUT_ROOT / "analysis/score.json")
    results = score["results"]
    final = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": results["decision"],
        "mode_decisions": {
            semantic: row["decision"] for semantic, row in results["mode_results"].items()
        },
        "natural_mechanism_recovered": False,
        "branch_status": "closed_after_one_shot_semantic_comparison",
        "auto_continue": False,
        "auto_continue_reason": "The intervention-semantics axis has received its predeclared one-shot test. Any next phase must change task family or use a separately justified natural-computation/causal design, not add or tune patch operators.",
        "non_implications": results["non_implications"],
    }
    final["final_digest"] = p1163.digest(final)
    p1163.write_json(OUT_ROOT / "analysis/final.json", final)
    print(p1163.canonical(final))


def smoke_command() -> None:
    broad = broad_holdout_subsets()
    print(
        p1163.canonical(
            {
                "calibration_count": len(calibration_subsets()),
                "broad_holdout_count": len(broad),
                "cardinality_counts": {size: sum(len(row) == size for row in broad) for size in (3, 4, 5)},
                "holdout_disjoint_from_prior": not bool(set(broad).intersection(prior_used_subsets())),
                "semantic_count": len(SEMANTICS),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "protocol",
            "run-calibration",
            "seal-predictions",
            "run-holdout",
            "score",
            "finalize",
            "smoke",
        ),
    )
    args = parser.parse_args()
    {
        "protocol": protocol_command,
        "run-calibration": run_calibration_command,
        "seal-predictions": seal_predictions_command,
        "run-holdout": run_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }[args.command]()


if __name__ == "__main__":
    main()
