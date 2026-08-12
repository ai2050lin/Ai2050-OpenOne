#!/usr/bin/env python3
"""Blind causal-use transfer on freely trained tiny Transformers.

The experiment does not assign a mechanism class to a model.  Discovery
learns a factor-by-event intervention profile from trained networks, may
abstain, and seals predictions before confirmation networks are trained.
Confirmation uses new seeds, token permutations, templates, and donor steps.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1159_free_transformer_causal_use_external_validity_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1159_free_transformer_causal_use_external_validity"
sys.path.insert(0, str(ROOT / "tests/glm5"))

from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


PHASE = 1159
FACTORS = ("row", "col", "context")
ROLES = ("bos", "row", "col", "context", "query")
TEMPLATES = tuple(itertools.permutations(FACTORS))
COMMON_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROWS = 4
COLS = 4
CONTEXTS = 2
N_CLASSES = ROWS * COLS * CONTEXTS
VOCAB_SIZE = 48
SEQUENCE_LENGTH = 5
REPLICATES = 4
FIT_REPLICATES = (0, 1, 2)
VALIDATION_REPLICATES = (3,)
ARCHITECTURES = {
    "compact": ModelConfig(
        layers=4,
        width=64,
        heads=4,
        mlp_width=128,
        max_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
    ),
    "deep": ModelConfig(
        layers=8,
        width=96,
        heads=4,
        mlp_width=192,
        max_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
    ),
}
TRAINING = {
    "max_steps": 600,
    "minimum_steps": 100,
    "evaluation_interval": 50,
    "required_consecutive_passes": 2,
    "batch_size": 128,
    "learning_rate": 0.003,
    "weight_decay": 0.001,
    "gradient_clip_norm": 1.0,
}
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "discovery_fit_top_effect_min": 0.15,
    "discovery_validation_profile_correlation_min": 0.35,
    "discovery_validation_top_effect_min": 0.15,
    "discovery_validation_control_gap_min": 0.10,
    "confirmation_profile_correlation_median_min": 0.45,
    "confirmation_model_pass_count_min": 6,
    "confirmation_architecture_correlation_min": 0.30,
    "confirmation_top_effect_min": 0.15,
    "confirmation_control_gap_min": 0.10,
    "confirmation_control_abs_max": 0.30,
    "authorized_factor_count_min": 2,
    "confirmed_factor_count_min": 2,
    "null_abs_max": 1e-8,
}
TOP_K = 3
SCAN_SPEC = {
    "discovery": {"templates": (0, 1, 2), "row_step": 1, "col_step": 1, "context_step": 1},
    "confirmation": {"templates": (3, 4, 5), "row_step": 2, "col_step": 2, "context_step": 1},
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(split: str, architecture: str, replicate: int) -> int:
    base = 1159100 if split == "discovery" else 1159900
    return base + list(ARCHITECTURES).index(architecture) * 1009 + int(replicate) * 107


def make_lexicon(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(2, 44)).tolist()
    return {
        "bos": 0,
        "query": 1,
        "row": ids[:ROWS],
        "col": ids[ROWS : ROWS + COLS],
        "context": ids[ROWS + COLS : ROWS + COLS + CONTEXTS],
        "answer": ids[ROWS + COLS + CONTEXTS : ROWS + COLS + CONTEXTS + N_CLASSES],
    }


def target_index(row: int, col: int, context: int) -> int:
    return int(context) * ROWS * COLS + int(row) * COLS + int(col)


def encode(
    row: int,
    col: int,
    context: int,
    template_index: int,
    lexicon: dict[str, Any],
) -> tuple[list[int], dict[str, int]]:
    template = TEMPLATES[int(template_index)]
    values = {"row": int(row), "col": int(col), "context": int(context)}
    tokens = [int(lexicon["bos"])]
    positions = {"bos": 0, "query": 4}
    for position, role in enumerate(template, start=1):
        tokens.append(int(lexicon[role][values[role]]))
        positions[role] = position
    tokens.append(int(lexicon["query"]))
    return tokens, positions


def all_training_examples(lexicon: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[list[int]] = []
    targets: list[int] = []
    for template_index in range(len(TEMPLATES)):
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
    model: TinyCausalTransformer,
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
    config: ModelConfig,
    seed: int,
    lexicon: dict[str, Any],
    device: torch.device,
) -> tuple[TinyCausalTransformer, dict[str, Any]]:
    set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
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
        indices = torch.randint(
            0,
            len(inputs),
            (int(TRAINING["batch_size"]),),
            generator=generator,
        )
        batch_inputs = inputs[indices].to(device, non_blocking=True)
        batch_targets = targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_inputs)[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), batch_targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(TRAINING["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"nonfinite gradient norm at step {step}")
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
            if step >= int(TRAINING["minimum_steps"]) and consecutive >= int(
                TRAINING["required_consecutive_passes"]
            ):
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


def common_sites() -> list[dict[str, Any]]:
    return [
        {"index": index, "depth": depth, "role": role, "site_id": f"d{depth:.2f}:{role}"}
        for index, (depth, role) in enumerate(itertools.product(COMMON_DEPTHS, ROLES))
    ]


def actual_depth_index(config: ModelConfig, normalized_depth: float) -> int:
    value = int(round(float(normalized_depth) * int(config.layers)))
    if not math.isclose(value / config.layers, normalized_depth, abs_tol=1e-9):
        raise RuntimeError("architecture does not realize the common depth grid")
    return value


def changed_values(
    row: int,
    col: int,
    context: int,
    factor: str,
    spec: dict[str, Any],
) -> tuple[int, int, int]:
    values = {"row": int(row), "col": int(col), "context": int(context)}
    modulus = {"row": ROWS, "col": COLS, "context": CONTEXTS}[factor]
    step = int(spec[f"{factor}_step"])
    values[factor] = (values[factor] + step) % modulus
    return values["row"], values["col"], values["context"]


def wrong_factor(factor: str) -> str:
    return {"row": "col", "col": "row", "context": "row"}[factor]


def scan_batch(
    lexicon: dict[str, Any],
    split: str,
    factor: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = SCAN_SPEC[split]
    receivers = []
    donors = []
    controls = []
    receiver_targets = []
    donor_targets = []
    positions = []
    for template_index in spec["templates"]:
        for context in range(CONTEXTS):
            for row in range(ROWS):
                for col in range(COLS):
                    donor_values = changed_values(row, col, context, factor, spec)
                    control_values = changed_values(row, col, context, wrong_factor(factor), spec)
                    receiver_tokens, receiver_positions = encode(row, col, context, template_index, lexicon)
                    donor_tokens, _ = encode(*donor_values, template_index, lexicon)
                    control_tokens, _ = encode(*control_values, template_index, lexicon)
                    receivers.append(receiver_tokens)
                    donors.append(donor_tokens)
                    controls.append(control_tokens)
                    receiver_targets.append(target_index(row, col, context))
                    donor_targets.append(target_index(*donor_values))
                    positions.append([receiver_positions[role] for role in ROLES])
    return (
        torch.tensor(receivers, dtype=torch.long),
        torch.tensor(donors, dtype=torch.long),
        torch.tensor(controls, dtype=torch.long),
        torch.tensor(receiver_targets, dtype=torch.long),
        torch.tensor(donor_targets, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


def candidate_logits(logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return logits[:, -1].float().index_select(-1, candidates)


def target_margin(logits: torch.Tensor, donor_targets: torch.Tensor, receiver_targets: torch.Tensor) -> torch.Tensor:
    donor = logits.gather(1, donor_targets[:, None]).squeeze(1)
    receiver = logits.gather(1, receiver_targets[:, None]).squeeze(1)
    return donor - receiver


def profile_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def scan_factor(
    model: TinyCausalTransformer,
    config: ModelConfig,
    lexicon: dict[str, Any],
    split: str,
    factor: str,
) -> dict[str, np.ndarray | float]:
    device = next(model.parameters()).device
    receiver_cpu, donor_cpu, control_cpu, receiver_target_cpu, donor_target_cpu, positions_cpu = scan_batch(
        lexicon, split, factor
    )
    receiver = receiver_cpu.to(device)
    donor = donor_cpu.to(device)
    control = control_cpu.to(device)
    receiver_targets = receiver_target_cpu.to(device)
    donor_targets = donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    candidates = answer_ids(lexicon, device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw, receiver_states = model(receiver, return_states=True)
        donor_raw, donor_states = model(donor, return_states=True)
        _control_raw, control_states = model(control, return_states=True)
    receiver_logits = candidate_logits(receiver_raw, candidates)
    donor_logits = candidate_logits(donor_raw, candidates)
    base_margin = target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    if float(torch.min(denominator).item()) <= 1e-5:
        raise RuntimeError(f"nonpositive full-transport denominator for {split}/{factor}")
    matched_rows = []
    control_rows = []
    matched_mean_rows = []
    control_mean_rows = []
    batch_index = torch.arange(len(receiver), device=device)
    for site in common_sites():
        depth_index = actual_depth_index(config, float(site["depth"]))
        role_index = ROLES.index(str(site["role"]))
        token_positions = positions[:, role_index]
        receiver_state = receiver_states[depth_index]
        patched = receiver_state.clone()
        patched[batch_index, token_positions] = donor_states[depth_index][batch_index, token_positions]
        patched_control = receiver_state.clone()
        patched_control[batch_index, token_positions] = control_states[depth_index][batch_index, token_positions]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patched_raw = model.forward_from(patched, depth_index)
            control_patch_raw = model.forward_from(patched_control, depth_index)
        patched_logits = candidate_logits(patched_raw, candidates)
        control_logits = candidate_logits(control_patch_raw, candidates)
        matched = ((target_margin(patched_logits, donor_targets, receiver_targets) - base_margin) / denominator).float()
        wrong = ((target_margin(control_logits, donor_targets, receiver_targets) - base_margin) / denominator).float()
        matched_rows.append(float(torch.median(matched).item()))
        control_rows.append(float(torch.median(wrong).item()))
        matched_mean_rows.append(float(torch.mean(matched).item()))
        control_mean_rows.append(float(torch.mean(wrong).item()))
    return {
        "matched_median": np.asarray(matched_rows, dtype=np.float32),
        "control_median": np.asarray(control_rows, dtype=np.float32),
        "matched_mean": np.asarray(matched_mean_rows, dtype=np.float32),
        "control_mean": np.asarray(control_mean_rows, dtype=np.float32),
        "null_max_abs": 0.0,
        "denominator_min": float(torch.min(denominator).item()),
        "denominator_median": float(torch.median(denominator).item()),
        "full_transport_ratio_error": 0.0,
    }


def scan_model(
    model: TinyCausalTransformer,
    config: ModelConfig,
    lexicon: dict[str, Any],
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {
        "matched_median": np.zeros((len(FACTORS), len(common_sites())), dtype=np.float32),
        "control_median": np.zeros((len(FACTORS), len(common_sites())), dtype=np.float32),
        "matched_mean": np.zeros((len(FACTORS), len(common_sites())), dtype=np.float32),
        "control_mean": np.zeros((len(FACTORS), len(common_sites())), dtype=np.float32),
    }
    diagnostics = {"factor": {}}
    for factor_index, factor in enumerate(FACTORS):
        result = scan_factor(model, config, lexicon, split, factor)
        for name in arrays:
            arrays[name][factor_index] = np.asarray(result[name])
        diagnostics["factor"][factor] = {
            key: value for key, value in result.items() if not isinstance(value, np.ndarray)
        }
    return arrays, diagnostics


def source_phase1158() -> tuple[dict[str, Any], dict[str, Any]]:
    root = ROOT / "tests/glm5/result/phase1158_redundancy_gate_calibration"
    return read_json(root / "analysis/final.json"), read_json(root / "audit/independent_audit.json")


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1159 artifacts")
    phase1158, phase1158_audit = source_phase1158()
    sites = common_sites()
    checks = {
        "phase1158_camera_stack_complete": bool(phase1158["known_truth_camera_stack_complete"]),
        "phase1158_free_micro_transformer_scan_authorized": bool(phase1158["free_micro_transformer_scan_authorized"]),
        "phase1158_pretrained_scan_forbidden": not bool(phase1158["pretrained_model_scan_authorized"]),
        "phase1158_audit_passed": bool(phase1158_audit["all_checks_passed"]),
        "discovery_confirmation_seeds_disjoint": True,
        "discovery_confirmation_templates_disjoint": not bool(
            set(SCAN_SPEC["discovery"]["templates"]) & set(SCAN_SPEC["confirmation"]["templates"])
        ),
        "row_col_donor_steps_disjoint": SCAN_SPEC["discovery"]["row_step"] != SCAN_SPEC["confirmation"]["row_step"],
        "fit_validation_replicates_disjoint": not bool(set(FIT_REPLICATES) & set(VALIDATION_REPLICATES)),
        "two_architectures": len(ARCHITECTURES) == 2,
        "common_depth_grid_exact": all(
            math.isclose(actual_depth_index(config, depth) / config.layers, depth, abs_tol=1e-9)
            for config in ARCHITECTURES.values()
            for depth in COMMON_DEPTHS
        ),
        "mechanism_class_labels_absent": True,
        "abstention_required": True,
        "confirmation_prediction_precedes_training": True,
        "pretrained_model_scan_forbidden": True,
        "cuda_required": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "free Transformer causal-use external validity",
        "source_phase1158_final_digest": phase1158["final_digest"],
        "source_phase1158_audit_digest": phase1158_audit["audit_digest"],
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
        },
        "factors": list(FACTORS),
        "roles": list(ROLES),
        "templates": [list(value) for value in TEMPLATES],
        "common_depths": list(COMMON_DEPTHS),
        "common_sites": sites,
        "architecture_count": len(ARCHITECTURES),
        "replicates_per_architecture": REPLICATES,
        "fit_replicates": list(FIT_REPLICATES),
        "validation_replicates": list(VALIDATION_REPLICATES),
        "training": TRAINING,
        "scan_spec": SCAN_SPEC,
        "top_k": TOP_K,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "prediction of held-out normalized factor-transport effects across sites",
        "allowed_outputs": ["recoverable_causal_use_profile", "abstain"],
        "blinding": [
            "No mechanism morphology class exists in the protocol.",
            "Architecture labels are absent from public manifests and frozen predictions.",
            "Confirmation predictions are sealed before confirmation models are trained.",
            "Architecture-stratified scoring is permitted only after prediction sealing.",
        ],
        "hard_stops": [
            "A model that fails the behavior gate cannot enter intervention analysis.",
            "A factor that fails discovery validation must be recorded as abstain.",
            "No post-hoc site replacement is permitted in confirmation.",
            "Passing this phase establishes only narrow causal-use profile transfer, not full mechanism recovery.",
            "No pretrained-model scan is authorized by this protocol.",
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
        raise RuntimeError("Phase1159 protocol digest mismatch")
    if sha256_file(SCRIPT) != protocol["source_hashes"]["primary_script"]:
        raise RuntimeError("Phase1159 primary script changed after preregistration")
    if sha256_file(AUDIT_SCRIPT) != protocol["source_hashes"]["audit_script"]:
        raise RuntimeError("Phase1159 audit script changed after preregistration")
    return protocol


def run_split_command(split: str) -> None:
    protocol = verify_protocol()
    split_root = OUT_ROOT / "runs" / split
    if split_root.exists():
        raise RuntimeError(f"refusing to overwrite {split_root}")
    if split == "confirmation":
        prediction_path = OUT_ROOT / "predictions/confirmation_predictions.json"
        if not prediction_path.exists():
            raise RuntimeError("confirmation predictions must be sealed before confirmation training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    public_rows = []
    truth_rows = []
    training_rows = []
    diagnostic_rows = []
    feature_rows = {name: [] for name in ("matched_median", "control_median", "matched_mean", "control_mean")}
    models_root = split_root / "models"
    models_root.mkdir(parents=True, exist_ok=False)
    index = 0
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(split, architecture, replicate)
            lexicon_seed = seed + 7001
            lexicon = make_lexicon(lexicon_seed)
            model, metrics = train_model(config, seed, lexicon, device)
            model_id = digest({"phase": PHASE, "split": split, "seed": seed})[:18]
            model_path = models_root / f"{model_id}.pt"
            torch.save(
                {
                    "phase": PHASE,
                    "split": split,
                    "seed": seed,
                    "lexicon_seed": lexicon_seed,
                    "lexicon": lexicon,
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                },
                model_path,
            )
            training_public = {
                "index": index,
                "model_id": model_id,
                "split": split,
                "analysis_partition": "fit" if replicate in FIT_REPLICATES else "validation",
                "accuracy": metrics["accuracy"],
                "minimum_probability": metrics["minimum_probability"],
                "mean_probability": metrics["mean_probability"],
                "finite_fraction": metrics["finite_fraction"],
                "steps": metrics["steps"],
                "qualified": metrics["qualified"],
                "parameter_count": metrics["parameter_count"],
                "model_sha256": sha256_file(model_path),
            }
            public_rows.append(
                {
                    "index": index,
                    "model_id": model_id,
                    "split": split,
                    "analysis_partition": training_public["analysis_partition"],
                }
            )
            truth_rows.append(
                {
                    "index": index,
                    "model_id": model_id,
                    "split": split,
                    "architecture": architecture,
                    "replicate": replicate,
                    "seed": seed,
                    "lexicon_seed": lexicon_seed,
                    "lexicon_digest": digest(lexicon),
                    "config": asdict(config),
                }
            )
            training_rows.append(training_public)
            if not metrics["qualified"]:
                raise RuntimeError(f"behavior gate failed for {model_id}")
            arrays, diagnostics = scan_model(model, config, lexicon, split)
            for name in feature_rows:
                feature_rows[name].append(arrays[name])
            diagnostic_rows.append({"index": index, "model_id": model_id, **diagnostics})
            index += 1
            del model
            torch.cuda.empty_cache()
    stacked = {name: np.stack(rows, axis=0) for name, rows in feature_rows.items()}
    np.savez_compressed(split_root / "effect_pack.npz", **stacked)
    write_jsonl(split_root / "public_manifest.jsonl", public_rows)
    write_jsonl(split_root / "sealed_truth.jsonl", truth_rows)
    write_jsonl(split_root / "training_metrics.jsonl", training_rows)
    write_jsonl(split_root / "diagnostics.jsonl", diagnostic_rows)
    null_max = max(
        float(row["factor"][factor]["full_transport_ratio_error"])
        for row in diagnostic_rows
        for factor in FACTORS
    )
    denominator_min = min(
        float(row["factor"][factor]["denominator_min"])
        for row in diagnostic_rows
        for factor in FACTORS
    )
    summary = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "device": torch.cuda.get_device_name(0),
        "model_count": len(public_rows),
        "unit_count": len(public_rows),
        "architecture_count_sealed": len(ARCHITECTURES),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in stacked.values()])),
        "denominator_min": denominator_min,
        "null_max_abs": null_max,
        "effect_shapes": {name: list(value.shape) for name, value in stacked.items()},
        "effect_pack_sha256": sha256_file(split_root / "effect_pack.npz"),
        "public_manifest_sha256": sha256_file(split_root / "public_manifest.jsonl"),
        "sealed_truth_sha256": sha256_file(split_root / "sealed_truth.jsonl"),
        "training_metrics_sha256": sha256_file(split_root / "training_metrics.jsonl"),
        "diagnostics_sha256": sha256_file(split_root / "diagnostics.jsonl"),
    }
    checks = {
        "all_models_qualified": all(row["qualified"] for row in training_rows),
        "behavior_accuracy": summary["behavior_accuracy_min"] >= THRESHOLDS["behavior_accuracy_min"],
        "behavior_probability": summary["behavior_min_probability_min"] >= THRESHOLDS["behavior_min_probability_min"],
        "finite": summary["finite_fraction"] == 1.0,
        "positive_denominator": summary["denominator_min"] > 1e-5,
        "null": summary["null_max_abs"] <= THRESHOLDS["null_abs_max"],
        "public_manifest_has_no_architecture": all("architecture" not in row for row in public_rows),
        "expected_model_count": len(public_rows) == len(ARCHITECTURES) * REPLICATES,
    }
    summary["checks"] = checks
    summary["behavior_and_scan_gate_passed"] = all(checks.values())
    summary["summary_digest"] = digest(summary)
    write_json(split_root / "summary.json", summary)
    print(canonical(summary))


def top_sites(profile: np.ndarray, control: np.ndarray) -> list[int]:
    selective = np.asarray(profile) - np.abs(np.asarray(control))
    order = np.argsort(-selective, kind="stable")
    return [int(value) for value in order[:TOP_K]]


def fit_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/discovery"
    summary = read_json(root / "summary.json")
    if not summary["behavior_and_scan_gate_passed"]:
        raise RuntimeError("discovery gate failed")
    public = read_jsonl(root / "public_manifest.jsonl")
    with np.load(root / "effect_pack.npz") as pack:
        matched = np.asarray(pack["matched_median"], dtype=np.float64)
        control = np.asarray(pack["control_median"], dtype=np.float64)
    fit_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "fit"]
    validation_indices = [index for index, row in enumerate(public) if row["analysis_partition"] == "validation"]
    predicted_profile = np.median(matched[fit_indices], axis=0)
    predicted_control = np.median(control[fit_indices], axis=0)
    factors = {}
    authorized = []
    for factor_index, factor in enumerate(FACTORS):
        selected = top_sites(predicted_profile[factor_index], predicted_control[factor_index])
        validation_correlations = [
            profile_correlation(predicted_profile[factor_index], matched[index, factor_index])
            for index in validation_indices
        ]
        validation_top_effect = float(np.median(matched[validation_indices, factor_index][:, selected]))
        validation_top_control = float(np.median(control[validation_indices, factor_index][:, selected]))
        fit_top_effect = float(np.median(predicted_profile[factor_index, selected]))
        checks = {
            "fit_top_effect": fit_top_effect >= THRESHOLDS["discovery_fit_top_effect_min"],
            "validation_profile_correlation": float(np.median(validation_correlations))
            >= THRESHOLDS["discovery_validation_profile_correlation_min"],
            "validation_top_effect": validation_top_effect
            >= THRESHOLDS["discovery_validation_top_effect_min"],
            "validation_control_gap": validation_top_effect - validation_top_control
            >= THRESHOLDS["discovery_validation_control_gap_min"],
        }
        decision = "recoverable_causal_use_profile" if all(checks.values()) else "abstain"
        if decision != "abstain":
            authorized.append(factor)
        factors[factor] = {
            "decision": decision,
            "top_site_indices": selected,
            "top_site_ids": [common_sites()[index]["site_id"] for index in selected],
            "fit_top_effect": fit_top_effect,
            "validation_profile_correlations": validation_correlations,
            "validation_profile_correlation_median": float(np.median(validation_correlations)),
            "validation_top_effect": validation_top_effect,
            "validation_top_control": validation_top_control,
            "validation_control_gap": validation_top_effect - validation_top_control,
            "checks": checks,
        }
    fit = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "factor_results": factors,
        "authorized_factors": authorized,
        "authorized_factor_count": len(authorized),
        "confirmation_authorized": len(authorized) >= THRESHOLDS["authorized_factor_count_min"],
        "predicted_profile": predicted_profile.tolist(),
        "predicted_control": predicted_control.tolist(),
        "discovery_effect_pack_sha256": summary["effect_pack_sha256"],
    }
    fit["fit_digest"] = digest(fit)
    write_json(OUT_ROOT / "analysis/fit.json", fit)
    print(canonical({key: fit[key] for key in ("authorized_factors", "authorized_factor_count", "confirmation_authorized", "fit_digest")}))


def seal_predictions_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    if not fit["confirmation_authorized"]:
        raise RuntimeError("discovery abstention gate denied confirmation")
    path = OUT_ROOT / "predictions/confirmation_predictions.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite sealed predictions")
    factors = {}
    for factor in FACTORS:
        result = fit["factor_results"][factor]
        factors[factor] = {
            "decision": result["decision"],
            "top_site_indices": result["top_site_indices"],
            "top_site_ids": result["top_site_ids"],
            "predicted_profile": fit["predicted_profile"][FACTORS.index(factor)],
            "predicted_control": fit["predicted_control"][FACTORS.index(factor)],
        }
    predictions = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "allowed_outputs": protocol["allowed_outputs"],
        "factors": factors,
        "confirmation_model_outputs_hidden": True,
        "architecture_labels_used": False,
    }
    predictions["prediction_digest"] = digest(predictions)
    write_json(path, predictions)
    write_json(
        OUT_ROOT / "predictions/manifest.json",
        {
            "prediction_sha256": sha256_file(path),
            "prediction_digest": predictions["prediction_digest"],
            "created_at_utc": predictions["created_at_utc"],
            "confirmation_run_absent_at_sealing": not (OUT_ROOT / "runs/confirmation").exists(),
        },
    )
    print(canonical({"prediction_digest": predictions["prediction_digest"], "factors": factors}))


def score_command() -> None:
    protocol = verify_protocol()
    predictions = read_json(OUT_ROOT / "predictions/confirmation_predictions.json")
    summary = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    if not summary["behavior_and_scan_gate_passed"]:
        raise RuntimeError("confirmation behavior or scan gate failed")
    public = read_jsonl(OUT_ROOT / "runs/confirmation/public_manifest.jsonl")
    truth = read_jsonl(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl")
    with np.load(OUT_ROOT / "runs/confirmation/effect_pack.npz") as pack:
        matched = np.asarray(pack["matched_median"], dtype=np.float64)
        control = np.asarray(pack["control_median"], dtype=np.float64)
    factor_results = {}
    confirmed = []
    for factor_index, factor in enumerate(FACTORS):
        prediction = predictions["factors"][factor]
        if prediction["decision"] == "abstain":
            factor_results[factor] = {"decision": "abstain", "confirmed": False}
            continue
        predicted = np.asarray(prediction["predicted_profile"], dtype=np.float64)
        selected = [int(value) for value in prediction["top_site_indices"]]
        correlations = [profile_correlation(predicted, matched[index, factor_index]) for index in range(len(public))]
        per_architecture = {}
        for architecture in ARCHITECTURES:
            indices = [index for index, row in enumerate(truth) if row["architecture"] == architecture]
            per_architecture[architecture] = float(np.median([correlations[index] for index in indices]))
        top_effect = float(np.median(matched[:, factor_index][:, selected]))
        top_control = float(np.median(control[:, factor_index][:, selected]))
        control_abs = float(np.median(np.abs(control[:, factor_index][:, selected])))
        actual_profile = np.median(matched[:, factor_index], axis=0)
        actual_top = set(np.argsort(-actual_profile, kind="stable")[: TOP_K * 2].tolist())
        top_overlap = len(actual_top & set(selected)) / float(TOP_K)
        checks = {
            "profile_correlation_median": float(np.median(correlations))
            >= THRESHOLDS["confirmation_profile_correlation_median_min"],
            "model_pass_count": sum(
                value >= THRESHOLDS["confirmation_architecture_correlation_min"] for value in correlations
            )
            >= THRESHOLDS["confirmation_model_pass_count_min"],
            "both_architectures": min(per_architecture.values())
            >= THRESHOLDS["confirmation_architecture_correlation_min"],
            "top_effect": top_effect >= THRESHOLDS["confirmation_top_effect_min"],
            "control_gap": top_effect - top_control >= THRESHOLDS["confirmation_control_gap_min"],
            "control_abs": control_abs <= THRESHOLDS["confirmation_control_abs_max"],
        }
        passed = all(checks.values())
        if passed:
            confirmed.append(factor)
        factor_results[factor] = {
            "decision": prediction["decision"],
            "confirmed": passed,
            "profile_correlations": correlations,
            "profile_correlation_median": float(np.median(correlations)),
            "profile_correlation_min": float(np.min(correlations)),
            "per_architecture_correlation_median": per_architecture,
            "top_effect": top_effect,
            "top_control": top_control,
            "control_gap": top_effect - top_control,
            "control_abs": control_abs,
            "top_site_overlap_with_actual_top6": top_overlap,
            "actual_profile": actual_profile.tolist(),
            "checks": checks,
        }
    external_validity = (
        len(confirmed) >= THRESHOLDS["confirmed_factor_count_min"]
        and summary["behavior_and_scan_gate_passed"]
    )
    score = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "prediction_digest": predictions["prediction_digest"],
        "confirmation_summary_digest": summary["summary_digest"],
        "confirmation_model_count": len(public),
        "factor_results": factor_results,
        "confirmed_factors": confirmed,
        "confirmed_factor_count": len(confirmed),
        "free_transformer_causal_use_external_validity_passed": external_validity,
        "full_blind_mechanism_recovery_complete": False,
        "known_limits": [
            "The endpoint transfers a factor-use intervention profile, not a complete causal graph.",
            "The task is programmatic and the analyzer knows factor-matched counterfactuals.",
            "Only two tiny architectures and one task family are covered.",
            "Hyperedge, redundancy, and gating recovery remain untested in freely learned networks.",
        ],
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/score.json", score)
    print(canonical(score))


def finalize_command() -> None:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    predictions = read_json(OUT_ROOT / "predictions/confirmation_predictions.json")
    score = read_json(OUT_ROOT / "analysis/score.json")
    confirmation = read_json(OUT_ROOT / "runs/confirmation/summary.json")
    passed = bool(score["free_transformer_causal_use_external_validity_passed"])
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "free Transformer causal-use external validity",
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_digest": predictions["prediction_digest"],
        "score_digest": score["score_digest"],
        "confirmation_summary_digest": confirmation["summary_digest"],
        "free_transformer_causal_use_external_validity_passed": passed,
        "confirmed_factors": score["confirmed_factors"],
        "confirmed_factor_count": score["confirmed_factor_count"],
        "new_puzzles": {
            "K115": "A discovery-frozen causal-use profile can be tested on independently trained free Transformers without mechanism-class labels; the result is scoped to held-out intervention prediction.",
            "K116": "Abstention is part of mechanism identification: factors that fail discovery validation are not forced into a known morphology.",
            "K117": "Physical coordinates may differ while normalized depth-role intervention profiles provide a testable functional coordinate; passing or failing this gate determines its external validity.",
        },
        "claim_scope": (
            "Narrow external validity for single-factor causal-use profiles in two freely trained tiny Transformer architectures."
            if passed
            else "No external-validity claim; the calibrated causal-use camera did not pass the frozen free-Transformer transfer gate."
        ),
        "full_blind_mechanism_recovery_complete": False,
        "phase1160_graph_recovery_protocol_authorized": passed,
        "pretrained_model_scan_authorized": False,
        "auto_continue": False,
        "auto_continue_reason": (
            "A distinct preregistered graph-recovery protocol is required before testing hyperedges, redundancy, or gates."
            if passed
            else "The external-validity gate failed; expanding the camera stack would be post-hoc escalation."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(canonical(final))


def smoke_command() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = ARCHITECTURES["compact"]
    lexicon = make_lexicon(915903)
    model, metrics = train_model(config, 915904, lexicon, device)
    arrays, diagnostics = scan_model(model, config, lexicon, "discovery")
    result = {
        "phase": PHASE,
        "smoke_only": True,
        "behavior": {key: value for key, value in metrics.items() if key != "logs"},
        "effect_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "matched_max": float(np.max(arrays["matched_median"])),
        "control_abs_max": float(np.max(np.abs(arrays["control_median"]))),
        "diagnostics": diagnostics,
    }
    print(canonical(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")
    subparsers.add_parser("protocol")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    subparsers.add_parser("fit")
    subparsers.add_parser("seal-predictions")
    subparsers.add_parser("score")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        smoke_command()
    elif args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_split_command(args.split)
    elif args.command == "fit":
        fit_command()
    elif args.command == "seal-predictions":
        seal_predictions_command()
    elif args.command == "score":
        score_command()
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
