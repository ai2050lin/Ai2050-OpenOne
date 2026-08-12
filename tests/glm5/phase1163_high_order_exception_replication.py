#!/usr/bin/env python3
"""Independently replicate and split the Phase1161 ordered-schedule exception.

The frozen Phase1161 A* schedule is tested in entirely new freely trained
Transformers.  Pairwise predictions are sealed from null/single/pair
calibration before any diagnostic schedule is executed.  Leave-one-out,
entry-role/depth substitutions, matched-cardinality layouts, and wrong-donor
controls separate operational replication from mechanism interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1163_high_order_exception_replication_audit.py"
P1161_SCRIPT = ROOT / "tests/glm5/phase1161_ordered_intervention_response_prediction.py"
P1161_AUDIT = ROOT / "tests/glm5/phase1161_ordered_intervention_response_prediction_audit.py"
SOURCE_SCRIPT = ROOT / "tests/glm5/phase1159_free_transformer_causal_use_external_validity.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1163_high_order_exception_replication"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1161_ordered_intervention_response_prediction as p1161  # noqa: E402


source = p1161.source
PHASE = 1163
FACTORS = source.FACTORS
ROLES = source.ROLES
ARCHITECTURES = source.ARCHITECTURES
INTERIOR_DEPTHS = p1161.INTERIOR_DEPTHS
REPLICATES = 4
RIDGE = p1161.RIDGE
A_STAR = (1, 4, 9, 14)
QUERY_CHAIN = (4, 9, 14)
RANDOM_CONTROL_SEED = 1163007
RANDOM_CONTROL_COUNT = 32
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "denominator_min": 1e-5,
    "null_abs_max": 1e-8,
    "replication_actual_median_min": 0.80,
    "replication_abs_residual_median_min": 0.50,
    "replication_large_residual_threshold": 0.40,
    "replication_large_residual_unit_min": 20,
    "replication_random_control_advantage_min": 0.35,
    "replication_architecture_abs_residual_min": 0.40,
    "specificity_gap_median_min": 0.30,
    "specificity_gap_unit_threshold": 0.20,
    "specificity_gap_unit_min": 18,
    "strict_leave_one_out_drop_min": 0.30,
    "query_chain_actual_min": 0.80,
    "query_chain_increment_max": 0.20,
    "generic_entry_actual_min": 0.80,
    "generic_entry_abs_residual_min": 0.50,
    "generic_entry_alternative_min": 2,
    "entry_depth_abs_residual_advantage_min": 0.30,
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sites() -> list[dict[str, Any]]:
    return p1161.sites()


def subset_id(subset: tuple[int, ...]) -> str:
    return p1161.subset_id(subset)


def calibration_subsets() -> list[tuple[int, ...]]:
    return p1161.calibration_subsets()


def add_registry_row(
    rows: list[dict[str, Any]],
    lookup: dict[tuple[int, ...], int],
    subset: tuple[int, ...],
    label: str,
    category: str,
) -> None:
    subset = tuple(sorted(set(int(value) for value in subset)))
    if subset in lookup:
        row = rows[lookup[subset]]
        if label not in row["labels"]:
            row["labels"].append(label)
        if category not in row["categories"]:
            row["categories"].append(category)
        return
    lookup[subset] = len(rows)
    rows.append(
        {
            "index": len(rows),
            "subset": list(subset),
            "subset_id": subset_id(subset),
            "cardinality": len(subset),
            "labels": [label],
            "categories": [category],
        }
    )


def diagnostic_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, ...], int] = {}
    add_registry_row(rows, lookup, A_STAR, "frozen_a_star", "frozen_target")
    add_registry_row(rows, lookup, QUERY_CHAIN, "query_chain", "leave_one_out")
    for site_index, label in zip(A_STAR, ("entry", "query_025", "query_050", "query_075"), strict=True):
        add_registry_row(
            rows,
            lookup,
            tuple(value for value in A_STAR if value != site_index),
            f"a_star_without_{label}",
            "leave_one_out",
        )
    site_rows = sites()
    for depth in INTERIOR_DEPTHS:
        for role in ("bos", "row", "col", "context"):
            entry_index = next(
                row["index"] for row in site_rows if row["depth"] == depth and row["role"] == role
            )
            add_registry_row(
                rows,
                lookup,
                tuple(sorted(QUERY_CHAIN + (entry_index,))),
                f"entry_d{depth:.2f}_{role}",
                "entry_query_chain",
            )
    candidates = []
    for subset in itertools.combinations(range(len(site_rows)), 4):
        depth_counts = [sum(site_rows[index]["depth"] == depth for index in subset) for depth in INTERIOR_DEPTHS]
        query_count = sum(site_rows[index]["role"] == "query" for index in subset)
        if depth_counts == [2, 1, 1] and query_count <= 2 and subset not in lookup:
            candidates.append(subset)
    rng = np.random.default_rng(RANDOM_CONTROL_SEED)
    selected = sorted(rng.choice(len(candidates), size=RANDOM_CONTROL_COUNT, replace=False).tolist())
    for control_index, candidate_index in enumerate(selected):
        add_registry_row(
            rows,
            lookup,
            candidates[int(candidate_index)],
            f"matched_cardinality_{control_index:02d}",
            "matched_cardinality_control",
        )
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def registry_subsets() -> list[tuple[int, ...]]:
    return [tuple(row["subset"]) for row in diagnostic_registry()]


def registry_index(label: str) -> int:
    matches = [row["index"] for row in diagnostic_registry() if label in row["labels"]]
    if len(matches) != 1:
        raise RuntimeError(f"registry label is not unique: {label}/{matches}")
    return int(matches[0])


def model_seed(architecture: str, replicate: int) -> int:
    return 1163100 + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def model_id(seed: int) -> str:
    return digest({"phase": PHASE, "seed": seed})[:16]


def prior_artifacts() -> dict[str, Any]:
    p1161_root = ROOT / "tests/glm5/result/phase1161_ordered_intervention_response_prediction"
    p1162_root = ROOT / "tests/glm5/result/phase1162_modular_task_response_transfer"
    return {
        "phase1161_final": read_json(p1161_root / "analysis/final.json"),
        "phase1161_audit": read_json(p1161_root / "audit/independent_audit.json"),
        "phase1161_exception": read_json(p1161_root / "analysis/posthoc_high_order_exception.json"),
        "phase1162_final": read_json(p1162_root / "analysis/final.json"),
        "phase1162_audit": read_json(p1162_root / "audit/behavior_stop_audit.json"),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1163 artifacts")
    prior = prior_artifacts()
    registry = diagnostic_registry()
    random_controls = [row for row in registry if "matched_cardinality_control" in row["categories"]]
    checks = {
        "phase1161_prediction_confirmed": bool(prior["phase1161_final"]["ordered_response_prediction_confirmed"]),
        "phase1161_audit_passed": bool(prior["phase1161_audit"]["all_checks_passed"]),
        "phase1161_exception_posthoc_only": bool(prior["phase1161_exception"]["evidence_upgrade_forbidden"]),
        "phase1161_exception_all_units": prior["phase1161_exception"]["top_exception_is_max_for_unit_count"] == 24,
        "frozen_target_exact": tuple(prior["phase1161_exception"]["top_exception_site_indices"]) == A_STAR,
        "phase1162_endpoints_untested": not bool(prior["phase1162_final"].get("response_transfer_test_executed", False)),
        "phase1162_audit_passed": bool(prior["phase1162_audit"]["all_checks_passed"]),
        "new_user_request_reauthorizes_plan": True,
        "calibration_null_single_pair_only": {len(row) for row in calibration_subsets()} == {0, 1, 2},
        "pairwise_calibration_count": len(calibration_subsets()) == 121,
        "diagnostic_registry_unique": len(registry) == len({tuple(row["subset"]) for row in registry}),
        "frozen_target_present": registry_index("frozen_a_star") == 0,
        "leave_one_out_complete": sum("leave_one_out" in row["categories"] for row in registry) == 4,
        "entry_query_chain_complete": sum("entry_query_chain" in row["categories"] for row in registry) == 12,
        "matched_cardinality_controls_complete": len(random_controls) == RANDOM_CONTROL_COUNT,
        "matched_control_depth_profile": all(
            [sum(sites()[index]["depth"] == depth for index in row["subset"]) for depth in INTERIOR_DEPTHS]
            == [2, 1, 1]
            for row in random_controls
        ),
        "diagnostic_outcomes_absent_before_prediction": True,
        "wrong_donor_control_predeclared": True,
        "one_shot_registry_closure_predeclared": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": "independent replication and mechanism split of the ordered high-order exception",
        "source_digests": {
            "phase1161_final": prior["phase1161_final"]["final_digest"],
            "phase1161_audit": prior["phase1161_audit"]["audit_digest"],
            "phase1161_exception": prior["phase1161_exception"]["diagnostic_digest"],
            "phase1162_final": prior["phase1162_final"]["final_digest"],
            "phase1162_audit": prior["phase1162_audit"]["audit_digest"],
        },
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
            "phase1161_script": sha256_file(P1161_SCRIPT),
            "phase1161_audit_script": sha256_file(P1161_AUDIT),
            "model_source_script": sha256_file(SOURCE_SCRIPT),
        },
        "factors": list(FACTORS),
        "roles": list(ROLES),
        "sites": sites(),
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_count": len(ARCHITECTURES) * REPLICATES,
        "calibration_subsets": [list(row) for row in calibration_subsets()],
        "calibration_algorithm": "frozen Phase1161 pairwise ridge estimator",
        "ridge": RIDGE,
        "diagnostic_registry": registry,
        "diagnostic_registry_count": len(registry),
        "frozen_a_star": list(A_STAR),
        "query_chain": list(QUERY_CHAIN),
        "temporal_pair_subsets": [[1, 4], [1, 9], [1, 14]],
        "random_control_seed": RANDOM_CONTROL_SEED,
        "random_control_count": RANDOM_CONTROL_COUNT,
        "donor_modes": ["matched", "wrong_factor"],
        "response": "median normalized target-factor donor margin under a fixed early-to-late patch schedule",
        "thresholds": THRESHOLDS,
        "primary_endpoint": "independent operational replication of the frozen A* residual relative to sealed pairwise predictions and matched-cardinality schedules",
        "secondary_endpoint": "factor specificity against a wrong-factor donor",
        "mechanism_split_endpoints": [
            "four leave-one-out schedules",
            "twelve entry-role/depth substitutions with the full query chain",
            "three entry-to-query temporal pairs",
        ],
        "allowed_decisions": [
            "factor_specific_ordered_exception_confirmed",
            "ordered_exception_confirmed_nonspecific",
            "ordered_exception_not_replicated",
        ],
        "hard_stops": [
            "Predictions must be sealed before diagnostic truth is generated.",
            "The A* target and all controls are immutable after protocol creation.",
            "Failure may not be repaired by selecting another schedule, seed, architecture, threshold, or polynomial order.",
            "Success does not identify a natural four-way hyperedge; it only confirms an operational residual under this patch schedule.",
            "Wrong-donor failure prevents a factor-specific transport claim but does not erase operational residual replication.",
            "This one-shot registry closes after scoring regardless of outcome.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
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
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1161_script", P1161_SCRIPT),
        ("phase1161_audit_script", P1161_AUDIT),
        ("model_source_script", SOURCE_SCRIPT),
    ):
        if sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    if protocol["diagnostic_registry"] != diagnostic_registry():
        raise RuntimeError("diagnostic registry drift")
    return protocol


def ordered_factor_surfaces(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    factor: str,
    subsets: list[tuple[int, ...]],
    modes: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = next(model.parameters()).device
    receiver_cpu, donor_cpu, control_cpu, receiver_target_cpu, donor_target_cpu, positions_cpu = source.scan_batch(
        lexicon, "confirmation", factor
    )
    receiver = receiver_cpu.to(device)
    donor = donor_cpu.to(device)
    control = control_cpu.to(device)
    receiver_targets = receiver_target_cpu.to(device)
    donor_targets = donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    candidates = source.answer_ids(lexicon, device)
    batch_index = torch.arange(len(receiver), device=device)
    role_positions = {role: positions[:, ROLES.index(role)] for role in ROLES}
    actual_by_depth = {depth: source.actual_depth_index(config, depth) for depth in INTERIOR_DEPTHS}
    site_rows = sites()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw = model(receiver)
        donor_raw, donor_states = model(donor, return_states=True)
        _control_raw, control_states = model(control, return_states=True)
    receiver_logits = source.candidate_logits(receiver_raw, candidates)
    donor_logits = source.candidate_logits(donor_raw, candidates)
    base_margin = source.target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = source.target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    denominator_min = float(torch.min(denominator).item())
    if denominator_min <= THRESHOLDS["denominator_min"]:
        raise RuntimeError(f"nonpositive denominator: {factor}/{denominator_min}")
    source_states = {"matched": donor_states, "wrong_factor": control_states}
    case_effects = {mode: [] for mode in modes}
    for subset in subsets:
        if not subset:
            for mode in modes:
                case_effects[mode].append(np.zeros(len(receiver), dtype=np.float32))
            continue
        selected = {int(value) for value in subset}
        for mode in modes:
            hidden = model.embed(receiver)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for layer_index, block in enumerate(model.blocks, start=1):
                    hidden = block(hidden)
                    patch_indices = [
                        index
                        for index in selected
                        if actual_by_depth[float(site_rows[index]["depth"])] == layer_index
                    ]
                    if patch_indices:
                        hidden = hidden.clone()
                        for site_index in patch_indices:
                            site = site_rows[site_index]
                            token_positions = role_positions[str(site["role"])]
                            hidden[batch_index, token_positions] = source_states[mode][layer_index][
                                batch_index, token_positions
                            ]
                patched_raw = model.lm_head(model.final_norm(hidden))
            patched_logits = source.candidate_logits(patched_raw, candidates)
            effect = (
                source.target_margin(patched_logits, donor_targets, receiver_targets) - base_margin
            ) / denominator
            case_effects[mode].append(effect.float().cpu().numpy().astype(np.float32))
    matrices = {mode: np.stack(rows, axis=0) for mode, rows in case_effects.items()}
    return matrices, {
        "case_count": int(len(receiver)),
        "denominator_min": denominator_min,
        "denominator_median": float(torch.median(denominator).item()),
        "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in matrices.values()])),
    }


def checkpoint_payload(model: torch.nn.Module, config: Any, lexicon: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "lexicon": lexicon,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = source.ModelConfig(**payload["config"])
    model = source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def run_calibration_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/calibration"
    if root.exists():
        raise RuntimeError("refusing to overwrite calibration run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_rows = []
    truth_rows = []
    training_rows = []
    diagnostic_rows = []
    model_arrays = []
    subsets = calibration_subsets()
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(architecture, replicate)
            identifier = model_id(seed)
            lexicon = source.make_lexicon(seed + 17017)
            model, training = source.train_model(config, seed, lexicon, device)
            if not training["qualified"]:
                raise RuntimeError(f"behavior gate failed: {identifier}")
            factor_arrays = []
            factor_diagnostics = {}
            for factor in FACTORS:
                matrices, detail = ordered_factor_surfaces(
                    model, config, lexicon, factor, subsets, ("matched",)
                )
                factor_arrays.append(np.median(matrices["matched"], axis=1).astype(np.float32))
                factor_diagnostics[factor] = detail
            model_arrays.append(np.stack(factor_arrays, axis=0))
            public_rows.append(
                {
                    "model_id": identifier,
                    "factor_count": len(FACTORS),
                    "calibration_subset_count": len(subsets),
                }
            )
            truth_rows.append(
                {
                    "model_id": identifier,
                    "architecture": architecture,
                    "replicate": replicate,
                    "seed": seed,
                    "lexicon_digest": digest(lexicon),
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
    write_jsonl(root / "public_manifest.jsonl", public_rows)
    write_jsonl(root / "sealed_truth.jsonl", truth_rows)
    write_jsonl(root / "training_metrics.jsonl", training_rows)
    write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(public_rows) == protocol["model_count"],
        "all_models_qualified": all(row["qualified"] for row in training_rows),
        "behavior_accuracy": min(row["accuracy"] for row in training_rows) >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": min(row["minimum_probability"] for row in training_rows)
        >= THRESHOLDS["behavior_min_probability_min"],
        "finite": float(np.isfinite(response).mean()) >= THRESHOLDS["finite_fraction_min"],
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "null": float(np.max(np.abs(response[:, :, 0]))) <= THRESHOLDS["null_abs_max"],
        "architecture_hidden_from_public": all("architecture" not in row for row in public_rows),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "response_shape": list(response.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "finite_fraction": float(np.isfinite(response).mean()),
        "denominator_min": denominator_min,
        "null_max_abs": float(np.max(np.abs(response[:, :, 0]))),
        "calibration_pack_sha256": sha256_file(root / "calibration_responses.npz"),
        "public_manifest_sha256": sha256_file(root / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(root / "sealed_truth.jsonl"),
        "checks": checks,
        "calibration_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = digest(summary)
    write_json(root / "summary.json", summary)
    print(canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/calibration"
    summary = read_json(root / "summary.json")
    if not summary["calibration_gate_passed"]:
        raise RuntimeError("calibration gate failed")
    if (OUT_ROOT / "runs/diagnostics").exists():
        raise RuntimeError("diagnostic outcomes already exist")
    prediction_root = OUT_ROOT / "predictions"
    if prediction_root.exists():
        raise RuntimeError("refusing to overwrite predictions")
    with np.load(root / "calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    subsets = registry_subsets()
    prediction = np.zeros((calibration.shape[0], len(FACTORS), len(subsets)), dtype=np.float32)
    coefficients = np.zeros((calibration.shape[0], len(FACTORS), 121), dtype=np.float64)
    for model_index in range(calibration.shape[0]):
        for factor_index in range(len(FACTORS)):
            coefficient = p1161.fit_coefficients(
                "pairwise", calibration_subsets(), calibration[model_index, factor_index]
            )
            coefficients[model_index, factor_index] = coefficient
            prediction[model_index, factor_index] = p1161.predict_values(
                "pairwise", coefficient, subsets
            ).astype(np.float32)
    prediction_root.mkdir(parents=True)
    np.savez_compressed(prediction_root / "diagnostic_predictions.npz", prediction=prediction)
    np.savez_compressed(prediction_root / "pairwise_coefficients.npz", coefficients=coefficients)
    metadata = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "algorithm": "pairwise",
        "ridge": RIDGE,
        "diagnostic_subset_ids": [subset_id(row) for row in subsets],
        "diagnostic_outcomes_absent_at_sealing": True,
        "architecture_labels_used": False,
        "calibration_pack_sha256": summary["calibration_pack_sha256"],
        "prediction_pack_sha256": sha256_file(prediction_root / "diagnostic_predictions.npz"),
        "coefficient_pack_sha256": sha256_file(prediction_root / "pairwise_coefficients.npz"),
    }
    metadata["prediction_digest"] = digest(metadata)
    write_json(prediction_root / "metadata.json", metadata)
    print(canonical(metadata))


def run_diagnostics_command() -> None:
    protocol = verify_protocol()
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    if not metadata["diagnostic_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid prediction seal")
    root = OUT_ROOT / "runs/diagnostics"
    if root.exists():
        raise RuntimeError("refusing to overwrite diagnostics")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    calibration_root = OUT_ROOT / "runs/calibration"
    public_rows = read_jsonl(calibration_root / "public_manifest.jsonl")
    truth_rows = read_jsonl(calibration_root / "sealed_truth.jsonl")
    subsets = registry_subsets()
    matched_models = []
    wrong_models = []
    matched_cases = []
    wrong_cases = []
    diagnostic_rows = []
    for public, truth in zip(public_rows, truth_rows, strict=True):
        if public["model_id"] != truth["model_id"]:
            raise RuntimeError("manifest order mismatch")
        model, config, lexicon = load_checkpoint(
            calibration_root / "checkpoints" / f"{public['model_id']}.pt", device
        )
        matched_factors = []
        wrong_factors = []
        matched_case_factors = []
        wrong_case_factors = []
        factor_diagnostics = {}
        for factor in FACTORS:
            matrices, detail = ordered_factor_surfaces(
                model, config, lexicon, factor, subsets, ("matched", "wrong_factor")
            )
            matched_factors.append(np.median(matrices["matched"], axis=1).astype(np.float32))
            wrong_factors.append(np.median(matrices["wrong_factor"], axis=1).astype(np.float32))
            matched_case_factors.append(matrices["matched"].astype(np.float32))
            wrong_case_factors.append(matrices["wrong_factor"].astype(np.float32))
            factor_diagnostics[factor] = detail
        matched_models.append(np.stack(matched_factors, axis=0))
        wrong_models.append(np.stack(wrong_factors, axis=0))
        matched_cases.append(np.stack(matched_case_factors, axis=0))
        wrong_cases.append(np.stack(wrong_case_factors, axis=0))
        diagnostic_rows.append({"model_id": public["model_id"], "factor": factor_diagnostics})
        del model
        torch.cuda.empty_cache()
    matched = np.stack(matched_models, axis=0)
    wrong = np.stack(wrong_models, axis=0)
    matched_case = np.stack(matched_cases, axis=0)
    wrong_case = np.stack(wrong_cases, axis=0)
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "diagnostic_responses.npz",
        matched=matched,
        wrong=wrong,
        matched_case=matched_case,
        wrong_case=wrong_case,
    )
    write_jsonl(root / "diagnostics.jsonl", diagnostic_rows)
    denominator_min = min(
        row["factor"][factor]["denominator_min"] for row in diagnostic_rows for factor in FACTORS
    )
    checks = {
        "model_count": len(diagnostic_rows) == protocol["model_count"],
        "registry_count": matched.shape[2] == protocol["diagnostic_registry_count"],
        "finite": float(np.mean([np.isfinite(matched).mean(), np.isfinite(wrong).mean()])) == 1.0,
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/diagnostic_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "prediction_precedes_diagnostic": metadata["created_at_utc"] < now(),
    }
    summary = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "matched_shape": list(matched.shape),
        "wrong_shape": list(wrong.shape),
        "case_shape": list(matched_case.shape),
        "finite_fraction": float(np.mean([np.isfinite(matched).mean(), np.isfinite(wrong).mean()])),
        "denominator_min": denominator_min,
        "diagnostic_pack_sha256": sha256_file(root / "diagnostic_responses.npz"),
        "checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
    }
    summary["summary_digest"] = digest(summary)
    write_json(root / "summary.json", summary)
    print(canonical({"summary_digest": summary["summary_digest"], "checks": checks}))


def q(value: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(value, dtype=np.float64), probability))


def calculate_results(
    protocol: dict[str, Any],
    calibration: np.ndarray,
    prediction: np.ndarray,
    matched: np.ndarray,
    wrong: np.ndarray,
    matched_case: np.ndarray,
    wrong_case: np.ndarray,
    truth_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    residual = np.asarray(matched, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    a_index = registry_index("frozen_a_star")
    query_index = registry_index("query_chain")
    random_indices = [
        row["index"]
        for row in protocol["diagnostic_registry"]
        if "matched_cardinality_control" in row["categories"]
    ]
    a_actual = matched[:, :, a_index].reshape(-1).astype(np.float64)
    a_wrong = wrong[:, :, a_index].reshape(-1).astype(np.float64)
    a_prediction = prediction[:, :, a_index].reshape(-1).astype(np.float64)
    a_residual = residual[:, :, a_index].reshape(-1)
    a_abs = np.abs(a_residual)
    random_unit = np.median(np.abs(residual[:, :, random_indices]), axis=2).reshape(-1)
    specificity_gap = a_actual - a_wrong
    architecture_abs = {}
    for architecture in ARCHITECTURES:
        model_indices = [index for index, row in enumerate(truth_rows) if row["architecture"] == architecture]
        architecture_abs[architecture] = float(np.median(np.abs(residual[model_indices, :, a_index])))
    top_count = int(
        np.sum(
            np.abs(residual[:, :, a_index])
            >= np.max(np.abs(residual), axis=2) - 1e-8
        )
    )
    replication_checks = {
        "actual_median": float(np.median(a_actual)) >= THRESHOLDS["replication_actual_median_min"],
        "abs_residual_median": float(np.median(a_abs))
        >= THRESHOLDS["replication_abs_residual_median_min"],
        "large_residual_units": int(np.sum(a_abs >= THRESHOLDS["replication_large_residual_threshold"]))
        >= THRESHOLDS["replication_large_residual_unit_min"],
        "beats_matched_cardinality": float(np.median(a_abs - random_unit))
        >= THRESHOLDS["replication_random_control_advantage_min"],
        "architecture_replication": all(
            value >= THRESHOLDS["replication_architecture_abs_residual_min"]
            for value in architecture_abs.values()
        ),
    }
    replicated = all(replication_checks.values())
    specificity_checks = {
        "gap_median": float(np.median(specificity_gap)) >= THRESHOLDS["specificity_gap_median_min"],
        "gap_unit_count": int(np.sum(specificity_gap >= THRESHOLDS["specificity_gap_unit_threshold"]))
        >= THRESHOLDS["specificity_gap_unit_min"],
    }
    factor_specific = replicated and all(specificity_checks.values())
    leave_labels = [
        "a_star_without_entry",
        "a_star_without_query_025",
        "a_star_without_query_050",
        "a_star_without_query_075",
    ]
    leave_one_out = {}
    leave_drops = []
    for label in leave_labels:
        index = registry_index(label)
        actual = matched[:, :, index].reshape(-1).astype(np.float64)
        absolute_residual = np.abs(residual[:, :, index]).reshape(-1)
        drop = a_actual - actual
        leave_drops.append(float(np.median(drop)))
        leave_one_out[label] = {
            "subset_id": protocol["diagnostic_registry"][index]["subset_id"],
            "actual_median": float(np.median(actual)),
            "abs_residual_median": float(np.median(absolute_residual)),
            "a_star_minus_subset_median": float(np.median(drop)),
        }
    query_actual = matched[:, :, query_index].reshape(-1).astype(np.float64)
    query_increment = a_actual - query_actual
    strict_minimal = replicated and min(leave_drops) >= THRESHOLDS["strict_leave_one_out_drop_min"]
    query_sufficient = replicated and (
        float(np.median(query_actual)) >= THRESHOLDS["query_chain_actual_min"]
        and float(np.median(query_increment)) <= THRESHOLDS["query_chain_increment_max"]
    )
    entry_results = {}
    alternative_count = 0
    for depth in INTERIOR_DEPTHS:
        for role in ("bos", "row", "col", "context"):
            label = f"entry_d{depth:.2f}_{role}"
            index = registry_index(label)
            actual = matched[:, :, index].reshape(-1).astype(np.float64)
            absolute_residual = np.abs(residual[:, :, index]).reshape(-1)
            entry_results[label] = {
                "actual_median": float(np.median(actual)),
                "abs_residual_median": float(np.median(absolute_residual)),
                "wrong_median": float(np.median(wrong[:, :, index])),
            }
            if depth == 0.25 and role != "row":
                if (
                    entry_results[label]["actual_median"] >= THRESHOLDS["generic_entry_actual_min"]
                    and entry_results[label]["abs_residual_median"]
                    >= THRESHOLDS["generic_entry_abs_residual_min"]
                ):
                    alternative_count += 1
    generic_entry = replicated and alternative_count >= THRESHOLDS["generic_entry_alternative_min"]
    row_025 = entry_results["entry_d0.25_row"]["abs_residual_median"]
    later_row = max(
        entry_results["entry_d0.50_row"]["abs_residual_median"],
        entry_results["entry_d0.75_row"]["abs_residual_median"],
    )
    entry_depth_specific = replicated and (
        row_025 - later_row >= THRESHOLDS["entry_depth_abs_residual_advantage_min"]
    )
    pair_rows = {}
    calibration_lookup = {subset: index for index, subset in enumerate(calibration_subsets())}
    pair_arrays = []
    for subset in ((1, 4), (1, 9), (1, 14)):
        values = calibration[:, :, calibration_lookup[subset]].reshape(-1).astype(np.float64)
        pair_arrays.append(values)
        pair_rows[subset_id(subset)] = float(np.median(values))
    best_pair = np.max(np.stack(pair_arrays, axis=1), axis=1)
    if factor_specific:
        decision = "factor_specific_ordered_exception_confirmed"
    elif replicated:
        decision = "ordered_exception_confirmed_nonspecific"
    else:
        decision = "ordered_exception_not_replicated"
    mechanism_flags = {
        "strict_four_site_minimality_candidate": strict_minimal,
        "query_chain_sufficient_candidate": query_sufficient,
        "generic_entry_plus_query_chain_candidate": generic_entry,
        "shallow_row_entry_depth_specific_candidate": entry_depth_specific,
        "factor_specific_against_wrong_donor": factor_specific,
        "unique_natural_mechanism_identified": False,
        "patch_schedule_interaction_excluded": False,
    }
    return {
        "unit_count": int(a_actual.size),
        "case_count": int(matched_case[:, :, a_index, :].size),
        "decision": decision,
        "operational_exception_replication_confirmed": replicated,
        "factor_specific_exception_confirmed": factor_specific,
        "a_star": {
            "subset_id": subset_id(A_STAR),
            "actual_median": float(np.median(a_actual)),
            "actual_q05": q(a_actual, 0.05),
            "actual_q95": q(a_actual, 0.95),
            "prediction_median": float(np.median(a_prediction)),
            "signed_residual_median": float(np.median(a_residual)),
            "abs_residual_median": float(np.median(a_abs)),
            "abs_residual_q95": q(a_abs, 0.95),
            "large_residual_unit_count": int(
                np.sum(a_abs >= THRESHOLDS["replication_large_residual_threshold"])
            ),
            "top_abs_residual_unit_count": top_count,
            "wrong_donor_median": float(np.median(a_wrong)),
            "matched_wrong_gap_median": float(np.median(specificity_gap)),
            "matched_wrong_gap_q05": q(specificity_gap, 0.05),
            "specificity_gap_unit_count": int(
                np.sum(specificity_gap >= THRESHOLDS["specificity_gap_unit_threshold"])
            ),
            "matched_case_median": float(np.median(matched_case[:, :, a_index, :])),
            "wrong_case_median": float(np.median(wrong_case[:, :, a_index, :])),
            "architecture_abs_residual_median": architecture_abs,
        },
        "matched_cardinality_control": {
            "control_count": len(random_indices),
            "unit_control_abs_residual_median": float(np.median(random_unit)),
            "a_star_advantage_median": float(np.median(a_abs - random_unit)),
        },
        "replication_checks": replication_checks,
        "specificity_checks": specificity_checks,
        "leave_one_out": leave_one_out,
        "entry_query_chain": entry_results,
        "entry_alternative_pass_count": alternative_count,
        "temporal_pairs": pair_rows,
        "a_star_minus_best_temporal_pair_median": float(np.median(a_actual - best_pair)),
        "query_chain": {
            "actual_median": float(np.median(query_actual)),
            "a_star_increment_median": float(np.median(query_increment)),
        },
        "mechanism_flags": mechanism_flags,
        "interpretive_boundary": (
            "The residual is relative to the frozen ridge pairwise estimator under one ordered patch schedule; "
            "it is not an exact Mobius coefficient or a unique natural causal hyperedge."
        ),
    }


def score_command() -> None:
    protocol = verify_protocol()
    calibration_summary = read_json(OUT_ROOT / "runs/calibration/summary.json")
    diagnostic_summary = read_json(OUT_ROOT / "runs/diagnostics/summary.json")
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    if not calibration_summary["calibration_gate_passed"] or not diagnostic_summary["diagnostic_gate_passed"]:
        raise RuntimeError("upstream gate failed")
    with np.load(OUT_ROOT / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(OUT_ROOT / "predictions/diagnostic_predictions.npz") as pack:
        prediction = np.asarray(pack["prediction"], dtype=np.float64)
    with np.load(OUT_ROOT / "runs/diagnostics/diagnostic_responses.npz") as pack:
        matched = np.asarray(pack["matched"], dtype=np.float64)
        wrong = np.asarray(pack["wrong"], dtype=np.float64)
        matched_case = np.asarray(pack["matched_case"], dtype=np.float64)
        wrong_case = np.asarray(pack["wrong_case"], dtype=np.float64)
    truth_rows = read_jsonl(OUT_ROOT / "runs/calibration/sealed_truth.jsonl")
    results = calculate_results(
        protocol, calibration, prediction, matched, wrong, matched_case, wrong_case, truth_rows
    )
    integrity_checks = {
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/diagnostic_predictions.npz")
        == metadata["prediction_pack_sha256"],
        "calibration_integrity": sha256_file(OUT_ROOT / "runs/calibration/calibration_responses.npz")
        == metadata["calibration_pack_sha256"],
        "diagnostic_integrity": sha256_file(OUT_ROOT / "runs/diagnostics/diagnostic_responses.npz")
        == diagnostic_summary["diagnostic_pack_sha256"],
        "prediction_precedes_diagnostics": metadata["created_at_utc"] < diagnostic_summary["created_at_utc"],
        "one_shot_registry_closed": True,
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"integrity checks failed: {integrity_checks}")
    score = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "calibration_summary_digest": calibration_summary["summary_digest"],
        "diagnostic_summary_digest": diagnostic_summary["summary_digest"],
        "results": results,
        "integrity_checks": integrity_checks,
        "registry_status": "closed_to_further_schedule_search",
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/score.json", score)
    print(canonical({"decision": results["decision"], "a_star": results["a_star"], "checks": results["replication_checks"], "score_digest": score["score_digest"]}))


def finalize_command() -> None:
    protocol = verify_protocol()
    score = read_json(OUT_ROOT / "analysis/score.json")
    results = score["results"]
    final = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": results["decision"],
        "operational_exception_replication_confirmed": results["operational_exception_replication_confirmed"],
        "factor_specific_exception_confirmed": results["factor_specific_exception_confirmed"],
        "unique_natural_mechanism_identified": False,
        "exact_mobius_order_identified": False,
        "natural_language_mechanism_recovered": False,
        "registry_status": "closed_to_further_schedule_search",
        "claim_scope": (
            "Independent new micro-Transformers confirm or reject one predeclared ordered patch-schedule residual; "
            "wrong-donor and schedule controls bound, but do not uniquely identify, its mechanism."
        ),
        "non_implications": [
            "An operational residual is not an exact third- or fourth-order Mobius term under ridge fitting.",
            "Minimality under sequential full-residual replacement is not a natural four-node hyperedge.",
            "A wrong-donor gap does not prove semantic identity is represented at these sites.",
            "The result does not transfer to pretrained models or language.",
        ],
        "auto_continue": False,
        "auto_continue_reason": (
            "The preregistered one-shot A* registry is terminal. Any confidence-gate calibration, compressed sensing, "
            "or cross-task experiment is a distinct protocol requiring a new objective rather than another hotspot pass."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def smoke_command() -> None:
    registry = diagnostic_registry()
    report = {
        "site_count": len(sites()),
        "calibration_count": len(calibration_subsets()),
        "diagnostic_count": len(registry),
        "a_star": registry[registry_index("frozen_a_star")],
        "leave_one_out_count": sum("leave_one_out" in row["categories"] for row in registry),
        "entry_count": sum("entry_query_chain" in row["categories"] for row in registry),
        "random_control_count": sum("matched_cardinality_control" in row["categories"] for row in registry),
    }
    print(canonical(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("smoke", "protocol", "run-calibration", "seal-predictions", "run-diagnostics", "score", "finalize"),
    )
    args = parser.parse_args()
    commands = {
        "smoke": smoke_command,
        "protocol": protocol_command,
        "run-calibration": run_calibration_command,
        "seal-predictions": seal_predictions_command,
        "run-diagnostics": run_diagnostics_command,
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
