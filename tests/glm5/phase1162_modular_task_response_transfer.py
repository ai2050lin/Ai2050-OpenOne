#!/usr/bin/env python3
"""Independent modular-task transfer of Phase1161's frozen response estimator."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1162_modular_task_response_transfer_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1162_modular_task_response_transfer"
SOURCE1161 = ROOT / "tests/glm5/phase1161_ordered_intervention_response_prediction.py"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1159_free_transformer_causal_use_external_validity as model_source  # noqa: E402
import phase1161_ordered_intervention_response_prediction as prior  # noqa: E402


PHASE = 1162
FACTORS = prior.FACTORS
ROLES = prior.ROLES
ARCHITECTURES = prior.ARCHITECTURES
REPLICATES = prior.REPLICATES
ROWS = 4
COLS = 4
CONTEXTS = 2
OUTPUT_CLASSES = 8
VOCAB_SIZE = 48
SELECTED_ALGORITHM = "pairwise"
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "denominator_min": 1e-5,
    "global_median_mae_max": prior.THRESHOLDS["confirmation_median_mae_max"],
    "global_median_correlation_min": prior.THRESHOLDS["confirmation_median_correlation_min"],
    "global_unit_mae_max": prior.THRESHOLDS["confirmation_unit_mae_max"],
    "global_unit_correlation_min": prior.THRESHOLDS["confirmation_unit_correlation_min"],
    "global_unit_pass_min": prior.THRESHOLDS["confirmation_unit_pass_min"],
    "global_unit_total": prior.THRESHOLDS["confirmation_unit_total"],
    "architecture_median_mae_max": prior.THRESHOLDS["confirmation_architecture_median_mae_max"],
    "layout_mae_advantage_min": prior.THRESHOLDS["confirmation_layout_mae_advantage_min"],
    "stress_median_absolute_error_max": 0.20,
    "stress_each_subset_median_absolute_error_max": 0.25,
    "null_abs_max": 1e-8,
}
TRAINING = prior.source.TRAINING


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(architecture: str, replicate: int) -> int:
    return 1162100 + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def model_id(seed: int) -> str:
    return digest({"phase": PHASE, "seed": seed})[:16]


def make_lexicon(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(2, 44)).tolist()
    cursor = 0
    lexicon: dict[str, Any] = {"bos": 0, "query": 1}
    for role, count in (("row", ROWS), ("col", COLS), ("context", CONTEXTS), ("answer", OUTPUT_CLASSES)):
        lexicon[role] = ids[cursor : cursor + count]
        cursor += count
    return lexicon


def target_index(row: int, col: int, context: int) -> int:
    return (int(row) + int(col)) % 4 + 4 * int(context)


def encode(
    row: int,
    col: int,
    context: int,
    template_index: int,
    lexicon: dict[str, Any],
) -> tuple[list[int], dict[str, int]]:
    template = model_source.TEMPLATES[int(template_index)]
    values = {"row": int(row), "col": int(col), "context": int(context)}
    tokens = [int(lexicon["bos"])]
    positions = {"bos": 0, "query": 4}
    for position, role in enumerate(template, start=1):
        tokens.append(int(lexicon[role][values[role]]))
        positions[role] = position
    tokens.append(int(lexicon["query"]))
    return tokens, positions


def all_training_examples(lexicon: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for template_index in range(len(model_source.TEMPLATES)):
        for context in range(CONTEXTS):
            for row in range(ROWS):
                for col in range(COLS):
                    tokens, _ = encode(row, col, context, template_index, lexicon)
                    inputs.append(tokens)
                    targets.append(target_index(row, col, context))
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def answer_ids(lexicon: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(lexicon["answer"], dtype=torch.long, device=device)


def evaluate_behavior(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lexicon: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs.to(device))[:, -1].float().index_select(-1, answer_ids(lexicon, device))
    probabilities = torch.softmax(logits, dim=-1)
    predicted = torch.argmax(logits, dim=-1).cpu()
    target_probability = probabilities.gather(1, targets.to(device)[:, None]).squeeze(1)
    return {
        "case_count": int(len(targets)),
        "accuracy": float(torch.mean((predicted == targets).to(torch.float32)).item()),
        "minimum_probability": float(torch.min(target_probability).item()),
        "mean_probability": float(torch.mean(target_probability).item()),
        "finite_fraction": float(torch.isfinite(logits).to(torch.float32).mean().item()),
    }


def train_model(
    config: Any,
    seed: int,
    lexicon: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seed(seed)
    model = model_source.TinyCausalTransformer(config).to(device)
    inputs, targets = all_training_examples(lexicon)
    candidates = answer_ids(lexicon, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        weight_decay=float(TRAINING["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 31)
    consecutive = 0
    logs = []
    final_step = 0
    for step in range(1, int(TRAINING["max_steps"]) + 1):
        model.train()
        indices = torch.randint(0, len(inputs), (int(TRAINING["batch_size"]),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs[indices].to(device))[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), targets[indices].to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(TRAINING["gradient_clip_norm"]))
        optimizer.step()
        final_step = step
        if step % int(TRAINING["evaluation_interval"]) == 0:
            metrics = evaluate_behavior(model, inputs, targets, lexicon)
            qualified = (
                metrics["accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
                and metrics["minimum_probability"] >= THRESHOLDS["behavior_min_probability_min"]
                and metrics["finite_fraction"] == 1.0
            )
            consecutive = consecutive + 1 if qualified else 0
            logs.append({"step": step, "loss": float(loss.item()), "gradient_norm": float(gradient_norm), **metrics})
            if step >= int(TRAINING["minimum_steps"]) and consecutive >= int(TRAINING["required_consecutive_passes"]):
                break
    metrics = evaluate_behavior(model, inputs, targets, lexicon)
    metrics.update(
        {
            "steps": final_step,
            "consecutive_passes": consecutive,
            "qualified": bool(
                metrics["accuracy"] >= THRESHOLDS["behavior_accuracy_min"]
                and metrics["minimum_probability"] >= THRESHOLDS["behavior_min_probability_min"]
                and metrics["finite_fraction"] == 1.0
                and consecutive >= int(TRAINING["required_consecutive_passes"])
            ),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "logs": logs,
        }
    )
    return model, metrics


def changed_values(row: int, col: int, context: int, factor: str) -> tuple[int, int, int]:
    values = {"row": int(row), "col": int(col), "context": int(context)}
    modulus = {"row": ROWS, "col": COLS, "context": CONTEXTS}[factor]
    values[factor] = (values[factor] + 1) % modulus
    return values["row"], values["col"], values["context"]


def scan_batch(
    lexicon: dict[str, Any],
    factor: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    receivers = []
    donors = []
    receiver_targets = []
    donor_targets = []
    positions = []
    for template_index in (3, 4, 5):
        for context in range(CONTEXTS):
            for row in range(ROWS):
                for col in range(COLS):
                    donor_values = changed_values(row, col, context, factor)
                    receiver_tokens, receiver_positions = encode(row, col, context, template_index, lexicon)
                    donor_tokens, _ = encode(*donor_values, template_index, lexicon)
                    receivers.append(receiver_tokens)
                    donors.append(donor_tokens)
                    receiver_targets.append(target_index(row, col, context))
                    donor_targets.append(target_index(*donor_values))
                    positions.append([receiver_positions[role] for role in ROLES])
    return (
        torch.tensor(receivers, dtype=torch.long),
        torch.tensor(donors, dtype=torch.long),
        torch.tensor(receiver_targets, dtype=torch.long),
        torch.tensor(donor_targets, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


def target_margin(logits: torch.Tensor, donor_targets: torch.Tensor, receiver_targets: torch.Tensor) -> torch.Tensor:
    donor = logits.gather(1, donor_targets[:, None]).squeeze(1)
    receiver = logits.gather(1, receiver_targets[:, None]).squeeze(1)
    return donor - receiver


def stress_subsets() -> list[tuple[int, ...]]:
    site_rows = prior.sites()
    query_indices = tuple(row["index"] for row in site_rows if row["role"] == "query")
    entry_indices = {
        factor: next(
            row["index"]
            for row in site_rows
            if row["depth"] == 0.25 and row["role"] == factor
        )
        for factor in FACTORS
    }
    return [tuple(sorted((entry_indices[factor], *query_indices))) for factor in FACTORS]


def holdout_subsets() -> list[tuple[int, ...]]:
    excluded = set(prior.discovery_holdout_subsets()) | set(prior.confirmation_holdout_subsets())
    stress = stress_subsets()
    rng = np.random.default_rng(1162001)
    rows: list[tuple[int, ...]] = []
    for cardinality in (3, 4):
        population = [
            row
            for row in itertools.combinations(range(len(prior.sites())), cardinality)
            if row not in excluded and row not in stress
        ]
        chosen = rng.choice(len(population), size=64, replace=False)
        rows.extend(population[int(index)] for index in sorted(chosen.tolist()))
    rows.extend(stress)
    return rows


def ordered_surface(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    factor: str,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, float]]:
    device = next(model.parameters()).device
    receiver_cpu, donor_cpu, receiver_target_cpu, donor_target_cpu, positions_cpu = scan_batch(lexicon, factor)
    receiver = receiver_cpu.to(device)
    donor = donor_cpu.to(device)
    receiver_targets = receiver_target_cpu.to(device)
    donor_targets = donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    candidates = answer_ids(lexicon, device)
    batch_index = torch.arange(len(receiver), device=device)
    role_positions = {role: positions[:, ROLES.index(role)] for role in ROLES}
    actual_by_depth = {depth: model_source.actual_depth_index(config, depth) for depth in prior.INTERIOR_DEPTHS}
    site_rows = prior.sites()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw = model(receiver)
        donor_raw, donor_states = model(donor, return_states=True)
    receiver_logits = receiver_raw[:, -1].float().index_select(-1, candidates)
    donor_logits = donor_raw[:, -1].float().index_select(-1, candidates)
    base_margin = target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    if float(torch.min(denominator).item()) <= THRESHOLDS["denominator_min"]:
        raise RuntimeError(f"nonpositive denominator for {factor}")
    values = []
    for subset in subsets:
        if not subset:
            values.append(0.0)
            continue
        selected = set(subset)
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
                        role = str(site_rows[site_index]["role"])
                        token_positions = role_positions[role]
                        hidden[batch_index, token_positions] = donor_states[layer_index][batch_index, token_positions]
            patched_raw = model.lm_head(model.final_norm(hidden))
        patched_logits = patched_raw[:, -1].float().index_select(-1, candidates)
        effect = (target_margin(patched_logits, donor_targets, receiver_targets) - base_margin) / denominator
        values.append(float(torch.median(effect.float()).item()))
    array = np.asarray(values, dtype=np.float32)
    return array, {
        "case_count": int(len(receiver)),
        "denominator_min": float(torch.min(denominator).item()),
        "denominator_median": float(torch.median(denominator).item()),
        "finite_fraction": float(np.isfinite(array).mean()),
        "null_abs": float(abs(array[0])) if subsets and not subsets[0] else 0.0,
    }


def source_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = prior.OUT_ROOT
    return (
        read_json(root / "analysis/final.json"),
        read_json(root / "audit/independent_audit.json"),
        read_json(root / "analysis/posthoc_high_order_exception.json"),
    )


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1162 artifacts")
    final, audit, diagnostic = source_artifacts()
    calibration = prior.calibration_subsets()
    holdout = holdout_subsets()
    stress = stress_subsets()
    checks = {
        "phase1161_confirmed": bool(final["ordered_response_prediction_confirmed"]),
        "phase1161_next_phase_authorized": bool(final["next_phase_authorized"]),
        "phase1161_audit_passed": bool(audit["all_checks_passed"]),
        "phase1161_exception_non_upgrading": bool(diagnostic["evidence_upgrade_forbidden"]),
        "selected_algorithm_frozen_pairwise": final["selected_algorithm"] == SELECTED_ALGORITHM,
        "new_task_not_cartesian_identity": all(
            target_index(row, col, context) < OUTPUT_CLASSES
            for row in range(ROWS) for col in range(COLS) for context in range(CONTEXTS)
        ) and OUTPUT_CLASSES < ROWS * COLS * CONTEXTS,
        "calibration_unchanged": calibration == prior.calibration_subsets(),
        "holdout_triple_quad_only": {len(row) for row in holdout} == {3, 4},
        "random_holdout_new": not bool(
            (set(holdout) - set(stress)).intersection(
                prior.discovery_holdout_subsets() + prior.confirmation_holdout_subsets()
            )
        ),
        "three_stress_subsets_present": len(stress) == 3 and all(row in holdout for row in stress),
        "prediction_before_holdout": True,
        "architecture_labels_forbidden": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": "modular-sum task response transfer and high-order stress",
        "source_phase1161_final_digest": final["final_digest"],
        "source_phase1161_audit_digest": audit["audit_digest"],
        "source_phase1161_diagnostic_digest": diagnostic["diagnostic_digest"],
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
            "phase1161_script": sha256_file(SOURCE1161),
        },
        "task": {
            "name": "modular_sum",
            "formula": "((row + col) mod 4) + 4*context",
            "input_combinations": ROWS * COLS * CONTEXTS,
            "output_classes": OUTPUT_CLASSES,
            "training_contains_all_combinations": True,
        },
        "sites": prior.sites(),
        "calibration_subsets": [list(row) for row in calibration],
        "holdout_subsets": [list(row) for row in holdout],
        "stress_subsets": [list(row) for row in stress],
        "stress_subset_ids": [prior.subset_id(row) for row in stress],
        "selected_algorithm": SELECTED_ALGORITHM,
        "algorithms_for_baseline_audit": list(prior.ALGORITHMS),
        "ridge": prior.RIDGE,
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "thresholds": THRESHOLDS,
        "primary_endpoints": [
            "global held-out intervention prediction under the frozen Phase1161 estimator",
            "predeclared factor-entry plus all-query high-order stress error",
        ],
        "allowed_outputs": ["full_transfer", "global_only_partial_transfer", "abstain"],
        "hard_stops": [
            "The Phase1161 pairwise estimator, ridge, site system, and calibration masks are frozen.",
            "Predictions must be sealed before modular-task holdout outcomes are generated.",
            "Global and stress gates are reported separately; one cannot compensate for the other.",
            "No estimator upgrade, cubic term, stress-subset deletion, or threshold change is permitted.",
            "This is the only automatically authorized independent task-family extension from Phase1161.",
            "No natural-language, pretrained-model, graph, hyperedge, or stable-identity claim is authorized.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(OUT_ROOT / "protocol/audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "protocol_digest": protocol["protocol_digest"]})
    print(canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (("primary_script", SCRIPT), ("audit_script", AUDIT_SCRIPT), ("phase1161_script", SOURCE1161)):
        if sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    return protocol


def checkpoint_payload(model: torch.nn.Module, config: Any, lexicon: dict[str, Any]) -> dict[str, Any]:
    return {"config": asdict(config), "lexicon": lexicon, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = model_source.ModelConfig(**payload["config"])
    model = model_source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def collect_model(model: torch.nn.Module, config: Any, lexicon: dict[str, Any], subsets: list[tuple[int, ...]]) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.zeros((len(FACTORS), len(subsets)), dtype=np.float32)
    diagnostics = {"factor": {}}
    for factor_index, factor in enumerate(FACTORS):
        values, detail = ordered_surface(model, config, lexicon, factor, subsets)
        rows[factor_index] = values
        diagnostics["factor"][factor] = detail
    return rows, diagnostics


def write_run_summary(root: Path, name: str, arrays: np.ndarray, diagnostics: list[dict[str, Any]], training: list[dict[str, Any]], public: list[dict[str, Any]], truth: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    pack_path = root / name
    denominator_min = min(detail["denominator_min"] for row in diagnostics for detail in row["factor"].values())
    null_max = float(np.max(np.abs(arrays[:, :, 0]))) if arrays.shape[2] and name.startswith("calibration") else 0.0
    summary = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "pack_name": name,
        "model_count": len(public),
        "response_shape": list(arrays.shape),
        "behavior_accuracy_min": min(row["accuracy"] for row in training),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training),
        "finite_fraction": float(np.isfinite(arrays).mean()),
        "denominator_min": denominator_min,
        "null_max_abs": null_max,
        "effect_pack_sha256": sha256_file(pack_path),
    }
    checks = {
        "model_count": len(public) == len(ARCHITECTURES) * REPLICATES,
        "behavior_accuracy": summary["behavior_accuracy_min"] >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": summary["behavior_min_probability_min"] >= THRESHOLDS["behavior_min_probability_min"],
        "finite": summary["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        "null": null_max <= THRESHOLDS["null_abs_max"],
        "architecture_hidden": all("architecture" not in row for row in public),
    }
    summary["checks"] = checks
    summary["run_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    summary_path = root / ("calibration_summary.json" if name.startswith("calibration") else "holdout_summary.json")
    write_json(summary_path, summary)


def run_calibration_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/models"
    if root.exists():
        raise RuntimeError("refusing to overwrite model run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public = []
    truth = []
    training_rows = []
    diagnostics_rows = []
    arrays = []
    calibration = prior.calibration_subsets()
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(architecture, replicate)
            identifier = model_id(seed)
            lexicon = make_lexicon(seed + 17)
            model, training = train_model(config, seed, lexicon, device)
            if not training["qualified"]:
                raise RuntimeError(f"training failed: {identifier}")
            response, diagnostics = collect_model(model, config, lexicon, calibration)
            arrays.append(response)
            public.append({"model_id": identifier, "factor_count": len(FACTORS), "subset_count": len(calibration)})
            truth.append({"model_id": identifier, "architecture": architecture, "replicate": replicate, "seed": seed})
            training_rows.append({"model_id": identifier, **training})
            diagnostics_rows.append({"model_id": identifier, **diagnostics})
            checkpoint = root / "checkpoints" / f"{identifier}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload(model, config, lexicon), checkpoint)
            del model
            torch.cuda.empty_cache()
    stacked = np.stack(arrays)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "calibration_responses.npz", response=stacked)
    write_jsonl(root / "public_manifest.jsonl", public)
    write_jsonl(root / "sealed_truth.jsonl", truth)
    write_jsonl(root / "training_metrics.jsonl", training_rows)
    write_jsonl(root / "calibration_diagnostics.jsonl", diagnostics_rows)
    write_run_summary(root, "calibration_responses.npz", stacked, diagnostics_rows, training_rows, public, truth, protocol)
    print(canonical(read_json(root / "calibration_summary.json")))


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/models"
    summary = read_json(root / "calibration_summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("calibration gate failed")
    if (root / "holdout_responses.npz").exists():
        raise RuntimeError("holdout outcomes already exist")
    prediction_root = OUT_ROOT / "predictions"
    if prediction_root.exists():
        raise RuntimeError("refusing to overwrite predictions")
    with np.load(root / "calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    holdout = holdout_subsets()
    predictions = {name: np.zeros((8, len(FACTORS), len(holdout)), dtype=np.float32) for name in prior.ALGORITHMS}
    for model_index in range(8):
        for factor_index in range(len(FACTORS)):
            for algorithm in prior.ALGORITHMS:
                coefficients = prior.fit_coefficients(algorithm, prior.calibration_subsets(), calibration[model_index, factor_index])
                predictions[algorithm][model_index, factor_index] = prior.predict_values(algorithm, coefficients, holdout).astype(np.float32)
    prediction_root.mkdir(parents=True)
    np.savez_compressed(prediction_root / "predictions.npz", **predictions)
    metadata = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "selected_algorithm": SELECTED_ALGORITHM,
        "holdout_subset_ids": [prior.subset_id(row) for row in holdout],
        "stress_subset_ids": protocol["stress_subset_ids"],
        "holdout_outcomes_absent_at_sealing": True,
        "architecture_labels_used": False,
        "prediction_pack_sha256": sha256_file(prediction_root / "predictions.npz"),
        "calibration_pack_sha256": summary["effect_pack_sha256"],
    }
    metadata["prediction_digest"] = digest(metadata)
    write_json(prediction_root / "metadata.json", metadata)
    print(canonical(metadata))


def run_holdout_command() -> None:
    protocol = verify_protocol()
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    root = OUT_ROOT / "runs/models"
    if (root / "holdout_responses.npz").exists():
        raise RuntimeError("refusing to overwrite holdout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    public = read_jsonl(root / "public_manifest.jsonl")
    truth = read_jsonl(root / "sealed_truth.jsonl")
    training = read_jsonl(root / "training_metrics.jsonl")
    holdout = holdout_subsets()
    arrays = []
    diagnostics_rows = []
    for public_row, truth_row in zip(public, truth, strict=True):
        checkpoint = root / "checkpoints" / f"{public_row['model_id']}.pt"
        model, config, lexicon = load_checkpoint(checkpoint, device)
        response, diagnostics = collect_model(model, config, lexicon, holdout)
        arrays.append(response)
        diagnostics_rows.append({"model_id": public_row["model_id"], **diagnostics})
        del model
        torch.cuda.empty_cache()
    stacked = np.stack(arrays)
    np.savez_compressed(root / "holdout_responses.npz", response=stacked)
    write_jsonl(root / "holdout_diagnostics.jsonl", diagnostics_rows)
    write_run_summary(root, "holdout_responses.npz", stacked, diagnostics_rows, training, public, truth, protocol)
    summary = read_json(root / "holdout_summary.json")
    summary["prediction_digest"] = metadata["prediction_digest"]
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(root / "holdout_summary.json", summary)
    print(canonical(summary))


def score_command() -> None:
    protocol = verify_protocol()
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    root = OUT_ROOT / "runs/models"
    summary = read_json(root / "holdout_summary.json")
    if not summary["run_gate_passed"]:
        raise RuntimeError("holdout run gate failed")
    with np.load(OUT_ROOT / "predictions/predictions.npz") as pack:
        predictions = {name: np.asarray(pack[name], dtype=np.float64) for name in prior.ALGORITHMS}
    with np.load(root / "holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    truth = read_jsonl(root / "sealed_truth.jsonl")
    results: dict[str, Any] = {}
    for algorithm in prior.ALGORITHMS:
        units = []
        for model_index, truth_row in enumerate(truth):
            for factor_index, factor in enumerate(FACTORS):
                detail = prior.metrics(predictions[algorithm][model_index, factor_index], observed[model_index, factor_index])
                units.append({"model_index": model_index, "architecture": truth_row["architecture"], "factor": factor, **detail})
        results[algorithm] = {
            "unit_metrics": units,
            "median_mae": float(np.median([row["mae"] for row in units])),
            "median_correlation": float(np.median([row["correlation"] for row in units])),
            "unit_pass_count": int(sum(row["mae"] <= THRESHOLDS["global_unit_mae_max"] and row["correlation"] >= THRESHOLDS["global_unit_correlation_min"] for row in units)),
            "unit_count": len(units),
            "architecture_median_mae": {
                architecture: float(np.median([row["mae"] for row in units if row["architecture"] == architecture]))
                for architecture in ARCHITECTURES
            },
        }
    selected = results[SELECTED_ALGORITHM]
    layout_advantage = results["layout"]["median_mae"] - selected["median_mae"]
    global_checks = {
        "prediction_integrity": sha256_file(OUT_ROOT / "predictions/predictions.npz") == metadata["prediction_pack_sha256"],
        "median_mae": selected["median_mae"] <= THRESHOLDS["global_median_mae_max"],
        "median_correlation": selected["median_correlation"] >= THRESHOLDS["global_median_correlation_min"],
        "unit_pass": selected["unit_pass_count"] >= THRESHOLDS["global_unit_pass_min"],
        "unit_total": selected["unit_count"] == THRESHOLDS["global_unit_total"],
        "architecture_median_mae": all(value <= THRESHOLDS["architecture_median_mae_max"] for value in selected["architecture_median_mae"].values()),
        "beats_layout_baseline": layout_advantage >= THRESHOLDS["layout_mae_advantage_min"],
    }
    holdout = holdout_subsets()
    stress_indices = [holdout.index(row) for row in stress_subsets()]
    absolute_error = np.abs(predictions[SELECTED_ALGORITHM] - observed)
    stress_errors = absolute_error[:, :, stress_indices]
    stress_subset_medians = {
        prior.subset_id(holdout[index]): float(np.median(absolute_error[:, :, index]))
        for index in stress_indices
    }
    stress_checks = {
        "stress_median_absolute_error": float(np.median(stress_errors)) <= THRESHOLDS["stress_median_absolute_error_max"],
        "each_stress_subset": all(value <= THRESHOLDS["stress_each_subset_median_absolute_error_max"] for value in stress_subset_medians.values()),
    }
    score = {
        "phase": PHASE,
        "created_at_utc": now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": metadata["prediction_digest"],
        "selected_algorithm": SELECTED_ALGORITHM,
        "algorithm_results": results,
        "layout_mae_advantage": layout_advantage,
        "global_checks": global_checks,
        "global_transfer_passed": all(global_checks.values()),
        "stress_indices": stress_indices,
        "stress_subset_median_absolute_errors": stress_subset_medians,
        "stress_median_absolute_error": float(np.median(stress_errors)),
        "stress_checks": stress_checks,
        "high_order_stress_passed": all(stress_checks.values()),
        "full_independent_task_transfer_passed": all(global_checks.values()) and all(stress_checks.values()),
        "holdout_summary_digest": summary["summary_digest"],
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/score.json", score)
    print(canonical({key: score[key] for key in ("selected_algorithm", "layout_mae_advantage", "global_checks", "global_transfer_passed", "stress_subset_median_absolute_errors", "stress_checks", "high_order_stress_passed", "full_independent_task_transfer_passed", "score_digest")}))


def finalize_command() -> None:
    protocol = verify_protocol()
    score = read_json(OUT_ROOT / "analysis/score.json")
    if score["full_independent_task_transfer_passed"]:
        decision = "full_transfer"
    elif score["global_transfer_passed"]:
        decision = "global_only_partial_transfer"
    else:
        decision = "abstain"
    final = {
        "phase": PHASE,
        "created_at_utc": now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": decision,
        "global_transfer_passed": score["global_transfer_passed"],
        "high_order_stress_passed": score["high_order_stress_passed"],
        "full_independent_task_transfer_passed": score["full_independent_task_transfer_passed"],
        "causal_graph_recovered": False,
        "physical_hyperedges_recovered": False,
        "full_mechanism_recovery_complete": False,
        "claim_scope": "Frozen low-order response prediction on an independent modular-sum task, with a separate predeclared high-order stress gate.",
        "auto_continue": False,
        "auto_continue_reason": "The sole Phase1161-authorized independent task-family extension is complete; further estimator changes require a new user-level research decision.",
        "non_implications": [
            "Global transfer does not imply stress completeness.",
            "Stress failure does not negate all low-order predictive structure.",
            "The task still exposes all input combinations during training.",
            "The supplied depth-role coordinate system and matched donors prevent a fully blind claim.",
            "No natural-language or pretrained-model mechanism follows.",
        ],
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "run-calibration", "seal-predictions", "run-holdout", "score", "finalize"))
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "run-calibration": run_calibration_command,
        "seal-predictions": seal_predictions_command,
        "run-holdout": run_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
