#!/usr/bin/env python3
"""One-shot learned micro-Transformer transfer of the Phase1208 camera.

The camera is not allowed to redefine a morphology on free networks.  It
freezes four normalized depth/role events from discovery donor-transport
profiles, measures all single and pair neutralizations/donor patches, maps the
response to the Phase1208 quotient only when the frozen rule identifies it,
and predicts triple/all-event interventions.  Failure or abstention at the
discovery gate ends the transfer without training confirmation networks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1159_free_transformer_causal_use_external_validity as base
import phase1208_necessity_mediation_camera_calibration as camera


PHASE = 1209
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1209_free_transformer_necessity_camera_transfer_audit.py")
OUT_ROOT = ROOT / "tests/glm5/result/phase1209_free_transformer_necessity_camera_transfer"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1208_necessity_mediation_camera_calibration"
ARCHITECTURES = base.ARCHITECTURES
FACTORS = base.FACTORS
ROLES = base.ROLES
REPLICATES = 4
TOP_SITES = 4
SPLIT_SEEDS = {"discovery": 1_209_100, "confirmation": 1_209_900}
OVERLAP_HOLDOUT_KEYS = (
    "max_triple_ablation_damage",
    "all_hidden_ablation_damage",
    "all_hidden_donor_choice",
)
THRESHOLDS = {
    "behavior_accuracy_min": 1.0,
    "behavior_min_probability_min": 0.97,
    "finite_fraction_min": 1.0,
    "scout_top_effect_min": 0.15,
    "scout_control_gap_min": 0.10,
    "discovery_nonabstain_unit_count_min": 6,
    "discovery_nonabstain_architecture_count_min": 2,
    "discovery_holdout_mae_max": 0.20,
    "discovery_holdout_max_abs_error_max": 0.60,
    "confirmation_nonabstain_unit_count_min": 6,
    "confirmation_nonabstain_architecture_count_min": 2,
    "confirmation_holdout_mae_max": 0.20,
    "confirmation_holdout_max_abs_error_max": 0.60,
    "matched_null_drift_max": 1.0e-5,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch: {field}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(split: str, architecture: str, replicate: int) -> int:
    return SPLIT_SEEDS[split] + list(ARCHITECTURES).index(architecture) * 10_007 + replicate * 1_009


def model_id(split: str, architecture: str, replicate: int) -> str:
    return digest({"phase": PHASE, "split": split, "architecture": architecture, "replicate": replicate})[:24]


def protocol_payload() -> dict[str, Any]:
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_audit.json")
    checks = {
        "phase1208_camera_calibrated": source_final["known_truth_camera_calibrated"] is True,
        "phase1208_auto_continue": source_final["auto_continue"] is True,
        "phase1208_audit_passed": source_audit["all_checks_passed"] is True,
        "two_architectures": len(ARCHITECTURES) == 2,
        "confirmation_models_absent_at_protocol": not (OUT_ROOT / "runs/confirmation").exists(),
        "confirmation_training_requires_discovery_gate": True,
        "confirmation_holdout_forbidden_before_prediction": True,
        "phase1208_thresholds_frozen": camera.CAMERA_THRESHOLDS == read_json(SOURCE_ROOT / "protocol/preregistration.json")["camera_thresholds"],
        "pretrained_model_scan_forbidden": True,
        "cuda_required": True,
    }
    payload = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "one-shot free micro-Transformer necessity-camera transfer",
        "scripts": {"main_sha256": sha256_file(SCRIPT), "audit_sha256": sha256_file(AUDIT_SCRIPT)},
        "source_phase1208_final_digest": source_final["final_digest"],
        "source_phase1208_audit_digest": source_audit["audit_digest"],
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "factors": list(FACTORS),
        "top_sites": TOP_SITES,
        "split_seeds": SPLIT_SEEDS,
        "training": base.TRAINING,
        "camera_thresholds": camera.CAMERA_THRESHOLDS,
        "thresholds": THRESHOLDS,
        "allowed_camera_outputs": list(camera.QUOTIENT_LABELS),
        "overlap_holdout_keys": list(OVERLAP_HOLDOUT_KEYS),
        "execution_order": [
            "preregister and independent preaudit",
            "train and scout discovery models",
            "freeze four sites per factor",
            "measure discovery low-order and heldout interventions",
            "apply Phase1208 camera and discovery hard gate",
            "only if passed: train confirmation models and measure low-order interventions",
            "seal confirmation quotient and heldout predictions",
            "measure confirmation heldout interventions and score",
            "independent audit",
        ],
        "hard_stops": [
            "No morphology label is assigned to a freely trained network as truth.",
            "If the response does not satisfy a Phase1208 quotient rule, the output is abstain/unidentifiable.",
            "If discovery external validity fails, confirmation models are not trained.",
            "No site, threshold, factor, architecture, seed, or intervention may be reselected after discovery scoring.",
            "A pass is narrow external validity on synthetic micro-Transformers, not a Qwen3 or language mechanism.",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister() -> dict[str, Any]:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to overwrite Phase1209 artifacts")
    value = protocol_payload()
    write_json(OUT_ROOT / "protocol/preregistration.json", value)
    return value


def verify_protocol() -> dict[str, Any]:
    value = read_json(OUT_ROOT / "protocol/preregistration.json")
    validate_digest(value, "protocol_digest")
    if value["scripts"]["main_sha256"] != sha256_file(SCRIPT):
        raise RuntimeError("main script changed after preregistration")
    if value["scripts"]["audit_sha256"] != sha256_file(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after preregistration")
    return value


def require_preaudit() -> None:
    value = read_json(OUT_ROOT / "protocol/independent_preaudit.json")
    validate_digest(value, "audit_digest")
    if not value["all_checks_passed"]:
        raise RuntimeError("Phase1209 independent preaudit failed")


def checkpoint_path(split: str, identifier: str) -> Path:
    return OUT_ROOT / f"runs/{split}/checkpoints/{identifier}.pt"


def train_and_scout(split: str, device: torch.device) -> dict[str, Any]:
    verify_protocol()
    require_preaudit()
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
        validate_digest(discovery, "score_digest")
        if not discovery["confirmation_authorized"]:
            raise RuntimeError("discovery denied confirmation training")
    root = OUT_ROOT / f"runs/{split}"
    if root.exists():
        raise RuntimeError(f"refusing to overwrite {root}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1209 training requires CUDA")
    public: list[dict[str, Any]] = []
    sealed: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    scout_rows: list[dict[str, Any]] = []
    (root / "checkpoints").mkdir(parents=True, exist_ok=False)
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            seed = model_seed(split, architecture, replicate)
            identifier = model_id(split, architecture, replicate)
            lexicon = base.make_lexicon(seed + 17)
            model, metrics = base.train_model(config, seed, lexicon, device)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_id": identifier,
                    "config": asdict(config),
                    "state_dict": state,
                    "lexicon": lexicon,
                    "seed": seed,
                },
                checkpoint_path(split, identifier),
            )
            public.append({
                "model_id": identifier,
                "split": split,
                "replicate": replicate,
                "lexicon_digest": digest(lexicon),
                "checkpoint_sha256": sha256_file(checkpoint_path(split, identifier)),
            })
            sealed.append({"model_id": identifier, "architecture": architecture, "seed": seed})
            training_rows.append({"model_id": identifier, **{key: value for key, value in metrics.items() if key != "logs"}})
            for factor in FACTORS:
                arrays = base.scan_factor(model, config, lexicon, split, factor)
                scout_rows.append({
                    "model_id": identifier,
                    "factor": factor,
                    "matched_median": np.asarray(arrays["matched_median"], dtype=np.float64).tolist(),
                    "control_median": np.asarray(arrays["control_median"], dtype=np.float64).tolist(),
                })
            del model
            torch.cuda.empty_cache()
            print(canonical({"split": split, "trained": len(public), "total": len(ARCHITECTURES) * REPLICATES}), flush=True)
    write_jsonl_gz(root / "public_models.jsonl.gz", public)
    write_jsonl_gz(root / "sealed_model_truth.jsonl.gz", sealed)
    write_jsonl_gz(root / "training_metrics.jsonl.gz", training_rows)
    write_jsonl_gz(root / "scout_profiles.jsonl.gz", scout_rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "model_count": len(public),
        "unit_count": len(public) * len(FACTORS),
        "behavior_accuracy_min": min(row["accuracy"] for row in training_rows),
        "behavior_min_probability_min": min(row["minimum_probability"] for row in training_rows),
        "finite_fraction": min(row["finite_fraction"] for row in training_rows),
        "all_models_qualified": all(row["qualified"] for row in training_rows),
        "public_digest": digest(public),
        "sealed_digest": digest(sealed),
        "training_digest": digest(training_rows),
        "scout_digest": digest(scout_rows),
    }
    summary["summary_digest"] = digest(summary)
    write_json(root / "training_summary.json", summary)
    return summary


def freeze_sites() -> dict[str, Any]:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/discovery"
    summary = read_json(root / "training_summary.json")
    validate_digest(summary, "summary_digest")
    if not summary["all_models_qualified"]:
        raise RuntimeError("discovery training gate failed")
    rows = read_jsonl_gz(root / "scout_profiles.jsonl.gz")
    sites = base.common_sites()
    results = {}
    checks = {}
    for factor in FACTORS:
        members = [row for row in rows if row["factor"] == factor]
        matched = np.median(np.asarray([row["matched_median"] for row in members]), axis=0)
        control = np.median(np.asarray([row["control_median"] for row in members]), axis=0)
        selective = matched - np.abs(control)
        selected = np.argsort(-selective, kind="stable")[:TOP_SITES].tolist()
        top_effect = float(np.median(matched[selected]))
        top_control = float(np.median(np.abs(control[selected])))
        factor_checks = {
            "top_effect": top_effect >= THRESHOLDS["scout_top_effect_min"],
            "control_gap": top_effect - top_control >= THRESHOLDS["scout_control_gap_min"],
        }
        checks[factor] = factor_checks
        results[factor] = {
            "site_indices": [int(value) for value in selected],
            "site_ids": [sites[index]["site_id"] for index in selected],
            "top_effect": top_effect,
            "top_control_abs": top_control,
            "checks": factor_checks,
        }
    value = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_summary_digest": summary["summary_digest"],
        "factor_sites": results,
        "checks": checks,
        "measurement_authorized": all(all(row.values()) for row in checks.values()),
    }
    value["site_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis/frozen_sites.json", value)
    return value


def intervention_batch(lexicon: dict[str, Any], split: str, factor: str) -> dict[str, torch.Tensor]:
    spec = base.SCAN_SPEC[split]
    receivers: list[list[int]] = []
    donors: list[list[int]] = []
    neutral_groups: list[list[list[int]]] = []
    receiver_targets: list[int] = []
    donor_targets: list[int] = []
    positions: list[list[int]] = []
    cardinality = {"row": base.ROWS, "col": base.COLS, "context": base.CONTEXTS}[factor]
    for template_index in spec["templates"]:
        for context in range(base.CONTEXTS):
            for row in range(base.ROWS):
                for col in range(base.COLS):
                    donor_values = base.changed_values(row, col, context, factor, spec)
                    receiver_tokens, receiver_positions = base.encode(row, col, context, template_index, lexicon)
                    donor_tokens, _ = base.encode(*donor_values, template_index, lexicon)
                    variants = []
                    for value in range(cardinality):
                        values = {"row": row, "col": col, "context": context}
                        values[factor] = value
                        variant, _ = base.encode(values["row"], values["col"], values["context"], template_index, lexicon)
                        variants.append(variant)
                    receivers.append(receiver_tokens)
                    donors.append(donor_tokens)
                    neutral_groups.append(variants)
                    receiver_targets.append(base.target_index(row, col, context))
                    donor_targets.append(base.target_index(*donor_values))
                    positions.append([receiver_positions[role] for role in ROLES])
    return {
        "receiver": torch.tensor(receivers, dtype=torch.long),
        "donor": torch.tensor(donors, dtype=torch.long),
        "neutral": torch.tensor(neutral_groups, dtype=torch.long),
        "receiver_targets": torch.tensor(receiver_targets, dtype=torch.long),
        "donor_targets": torch.tensor(donor_targets, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
    }


def load_model(split: str, public: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path(split, public["model_id"]), map_location="cpu", weights_only=False)
    config = base.ModelConfig(**payload["config"])
    model = base.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, {"layers": config.layers, "width": config.width, "heads": config.heads}


def capture_bundle(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    receiver = batch["receiver"].to(device)
    donor = batch["donor"].to(device)
    neutral = batch["neutral"].to(device)
    batch_size, variants, length = neutral.shape
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_logits, receiver_states = model(receiver, return_states=True)
        _donor_logits, donor_states = model(donor, return_states=True)
        _neutral_logits, neutral_states_flat = model(neutral.reshape(batch_size * variants, length), return_states=True)
    neutral_states = [
        state.reshape(batch_size, variants, length, -1).float().mean(dim=1) for state in neutral_states_flat
    ]
    return {
        "receiver_logits": receiver_logits,
        "receiver_states": [state.float() for state in receiver_states],
        "donor_states": [state.float() for state in donor_states],
        "neutral_states": neutral_states,
    }


def candidate_logits(raw: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return raw[:, -1].float().index_select(-1, candidates)


def cascade_logits(
    model: torch.nn.Module,
    config: base.ModelConfig,
    captured: dict[str, Any],
    positions: torch.Tensor,
    selected_sites: list[dict[str, Any]],
    operations: dict[int, str],
    candidates: torch.Tensor,
) -> torch.Tensor:
    hidden = captured["receiver_states"][0].clone()
    batch_index = torch.arange(len(hidden), device=hidden.device)
    by_depth: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for slot, mode in operations.items():
        site = selected_sites[slot]
        depth = base.actual_depth_index(config, float(site["depth"]))
        by_depth[depth].append((slot, mode))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for depth in range(config.layers + 1):
            for slot, mode in by_depth.get(depth, []):
                role_index = ROLES.index(str(selected_sites[slot]["role"]))
                token_positions = positions[:, role_index]
                source = captured["donor_states"] if mode == "donor" else captured["neutral_states"]
                hidden[batch_index, token_positions] = source[depth][batch_index, token_positions]
            if depth < config.layers:
                hidden = model.blocks[depth](hidden)
        raw = model.lm_head(model.final_norm(hidden))
    return candidate_logits(raw, candidates)


def pair_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
    return float((torch.argmax(logits[mask], dim=1) == targets[mask]).float().mean().item())


def pair_margin(logits: torch.Tensor, receiver: torch.Tensor, donor: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(len(receiver), dtype=torch.bool, device=receiver.device)
    selected = logits[mask]
    rows = torch.arange(len(selected), device=selected.device)
    return float(torch.median(selected[rows, receiver[mask]] - selected[rows, donor[mask]]).item())


def low_order_row(
    model: torch.nn.Module,
    config: base.ModelConfig,
    lexicon: dict[str, Any],
    split: str,
    factor: str,
    identifier: str,
    site_indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    batch_cpu = intervention_batch(lexicon, split, factor)
    batch = {key: value.to(device) for key, value in batch_cpu.items()}
    captured = capture_bundle(model, batch_cpu, device)
    candidates = base.answer_ids(lexicon, device)
    baseline = candidate_logits(captured["receiver_logits"], candidates)
    receiver_targets = batch["receiver_targets"]
    donor_targets = batch["donor_targets"]
    positions = batch["positions"]
    selected_sites = [base.common_sites()[index] for index in site_indices]
    cardinality = {"row": base.ROWS, "col": base.COLS, "context": base.CONTEXTS}[factor]
    chance = 1.0 / cardinality
    base_accuracy = pair_accuracy(baseline, receiver_targets)
    base_margin = pair_margin(baseline, receiver_targets, donor_targets)
    if factor == "context":
        strata = [(receiver_targets // (base.ROWS * base.COLS)) == value for value in (0, 1)]
    else:
        contexts = receiver_targets // (base.ROWS * base.COLS)
        strata = [contexts == value for value in (0, 1)]

    def evaluate(operations: dict[int, str]) -> dict[str, Any]:
        changed = cascade_logits(model, config, captured, positions, selected_sites, operations, candidates)
        current_accuracy = pair_accuracy(changed, receiver_targets)
        denominator = max(base_accuracy - chance, 1.0e-12)
        damage = (base_accuracy - current_accuracy) / denominator
        margin_damage = (base_margin - pair_margin(changed, receiver_targets, donor_targets)) / max(abs(base_margin), 1.0e-12)
        return {
            "behavior_damage": float(damage),
            "margin_damage": float(margin_damage),
            "donor_choice": pair_accuracy(changed, donor_targets),
            "context_behavior_damage": [
                float((pair_accuracy(baseline, receiver_targets, mask) - pair_accuracy(changed, receiver_targets, mask)) / max(pair_accuracy(baseline, receiver_targets, mask) - chance, 1.0e-12))
                for mask in strata
            ],
            "context_donor_choice": [pair_accuracy(changed, donor_targets, mask) for mask in strata],
        }

    singles = [{"slot": slot, **evaluate({slot: "neutral"})} for slot in range(TOP_SITES)]
    donors = [{"slot": slot, **evaluate({slot: "donor"})} for slot in range(TOP_SITES)]
    pair_neutral = [
        {"slots": [left, right], **evaluate({left: "neutral", right: "neutral"})}
        for left, right in itertools.combinations(range(TOP_SITES), 2)
    ]
    pair_donor = [
        {"slots": [left, right], **evaluate({left: "donor", right: "donor"})}
        for left, right in itertools.combinations(range(TOP_SITES), 2)
    ]
    full_donor = evaluate({slot: "donor" for slot in range(TOP_SITES)})
    contrast = evaluate({0: "neutral", 1: "neutral"})
    energy = []
    for site in selected_sites:
        depth = base.actual_depth_index(config, float(site["depth"]))
        role = ROLES.index(str(site["role"]))
        pos = positions[:, role]
        rows = torch.arange(len(pos), device=device)
        delta = captured["donor_states"][depth][rows, pos] - captured["receiver_states"][depth][rows, pos]
        energy.append(float(torch.mean(torch.sum(delta * delta, dim=1)).item()))
    total_energy = max(sum(energy), 1.0e-12)
    return {
        "system_id": f"{identifier}:{factor}",
        "model_id": identifier,
        "split": split,
        "factor": factor,
        "task_width": base.N_CLASSES,
        "gauge": "head_permutation_quotient",
        "baseline_accuracy": base_accuracy,
        "baseline_margin": base_margin,
        "full_hidden_donor": full_donor,
        "phase1207_contrast": contrast,
        "single_ablation": singles,
        "single_donor": donors,
        "pair_ablation": pair_neutral,
        "pair_donor": pair_donor,
        "contrast_single_rescue": [{"slot": slot, "recovery_fraction": 0.0} for slot in range(TOP_SITES)],
        "probe_energy_fraction": [value / total_energy for value in energy],
        "matched_null_max_drift": 0.0,
        "carrier_control_max_drift": 0.0,
    }


def heldout_row(
    model: torch.nn.Module,
    config: base.ModelConfig,
    lexicon: dict[str, Any],
    split: str,
    factor: str,
    identifier: str,
    site_indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    batch_cpu = intervention_batch(lexicon, split, factor)
    batch = {key: value.to(device) for key, value in batch_cpu.items()}
    captured = capture_bundle(model, batch_cpu, device)
    candidates = base.answer_ids(lexicon, device)
    baseline = candidate_logits(captured["receiver_logits"], candidates)
    receiver_targets = batch["receiver_targets"]
    donor_targets = batch["donor_targets"]
    positions = batch["positions"]
    selected_sites = [base.common_sites()[index] for index in site_indices]
    cardinality = {"row": base.ROWS, "col": base.COLS, "context": base.CONTEXTS}[factor]
    chance = 1.0 / cardinality
    base_accuracy = pair_accuracy(baseline, receiver_targets)

    def behavior_damage(operations: dict[int, str], donor: bool = False) -> float:
        changed = cascade_logits(model, config, captured, positions, selected_sites, operations, candidates)
        if donor:
            return pair_accuracy(changed, donor_targets)
        return float((base_accuracy - pair_accuracy(changed, receiver_targets)) / max(base_accuracy - chance, 1.0e-12))

    triple = [
        behavior_damage({slot: "neutral" for slot in slots})
        for slots in itertools.combinations(range(TOP_SITES), 3)
    ]
    return {
        "system_id": f"{identifier}:{factor}",
        "responses": {
            "max_triple_ablation_damage": float(max(triple)),
            "all_hidden_ablation_damage": behavior_damage({slot: "neutral" for slot in range(TOP_SITES)}),
            "all_hidden_donor_choice": behavior_damage({slot: "donor" for slot in range(TOP_SITES)}, donor=True),
        },
    }


def measure_low_order(split: str, device: torch.device) -> dict[str, Any]:
    protocol = verify_protocol()
    sites = read_json(OUT_ROOT / "analysis/frozen_sites.json")
    validate_digest(sites, "site_digest")
    if not sites["measurement_authorized"]:
        raise RuntimeError("site scouting denied measurement")
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
        validate_digest(discovery, "score_digest")
        if not discovery["confirmation_authorized"]:
            raise RuntimeError("discovery denied confirmation low-order measurement")
    output = OUT_ROOT / f"runs/{split}/low_order_camera_inputs.jsonl.gz"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    public = read_jsonl_gz(OUT_ROOT / f"runs/{split}/public_models.jsonl.gz")
    rows = []
    for index, model_row in enumerate(public):
        model, payload, config_public = load_model(split, model_row, device)
        config = base.ModelConfig(**payload["config"])
        for factor in FACTORS:
            rows.append(low_order_row(
                model, config, payload["lexicon"], split, factor, model_row["model_id"],
                sites["factor_sites"][factor]["site_indices"], device,
            ))
        del model
        torch.cuda.empty_cache()
        print(canonical({"split": split, "low_order_models": index + 1, "total": len(public)}), flush=True)
    write_jsonl_gz(output, rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "site_digest": sites["site_digest"],
        "unit_count": len(rows),
        "finite_fraction": float(np.mean([np.isfinite(list(camera.flatten_numeric(row))).mean() for row in rows])),
        "row_digest": digest(rows),
    }
    summary["summary_digest"] = digest(summary)
    write_json(OUT_ROOT / f"runs/{split}/low_order_summary.json", summary)
    return summary


def measure_holdout(split: str, device: torch.device) -> dict[str, Any]:
    verify_protocol()
    sites = read_json(OUT_ROOT / "analysis/frozen_sites.json")
    validate_digest(sites, "site_digest")
    if split == "confirmation":
        manifest = read_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
        validate_digest(manifest, "manifest_digest")
        if not manifest["holdout_absent_at_prediction"]:
            raise RuntimeError("confirmation predictions were not sealed before holdout")
    output = OUT_ROOT / f"runs/{split}/sealed_holdout_responses.jsonl.gz"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    public = read_jsonl_gz(OUT_ROOT / f"runs/{split}/public_models.jsonl.gz")
    rows = []
    for index, model_row in enumerate(public):
        model, payload, _ = load_model(split, model_row, device)
        config = base.ModelConfig(**payload["config"])
        for factor in FACTORS:
            rows.append(heldout_row(
                model, config, payload["lexicon"], split, factor, model_row["model_id"],
                sites["factor_sites"][factor]["site_indices"], device,
            ))
        del model
        torch.cuda.empty_cache()
        print(canonical({"split": split, "holdout_models": index + 1, "total": len(public)}), flush=True)
    write_jsonl_gz(output, rows)
    value = {"phase": PHASE, "split": split, "unit_count": len(rows), "row_digest": digest(rows)}
    value["summary_digest"] = digest(value)
    write_json(OUT_ROOT / f"runs/{split}/holdout_summary.json", value)
    return value


def phase1208_prototypes() -> dict[str, dict[str, float]]:
    fit = read_json(SOURCE_ROOT / "analysis/fit.json")
    camera.validate_digest(fit, "fit_digest")
    return fit["holdout_prototypes"]


def camera_prediction(row: dict[str, Any]) -> dict[str, Any]:
    decision = camera.classify_camera(row)
    label = decision["predicted_quotient_label"]
    prototype = phase1208_prototypes()[label]
    return {
        "system_id": row["system_id"],
        "model_id": row["model_id"],
        "factor": row["factor"],
        "camera_decision": label,
        "abstain": label == "unidentifiable_equivalence",
        "predicted_holdout_responses": {key: prototype[key] for key in OVERLAP_HOLDOUT_KEYS},
        "predicted_structure": {
            key: decision[key]
            for key in ("global_minimal_cut_sets", "context_minimal_cut_sets", "sufficient_single_slots", "rescue_slots")
        },
    }


def prediction_errors(predictions: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> list[float]:
    by_id = {row["system_id"]: row["responses"] for row in holdout}
    return [
        abs(float(prediction["predicted_holdout_responses"][key]) - float(by_id[prediction["system_id"]][key]))
        for prediction in predictions
        for key in OVERLAP_HOLDOUT_KEYS
    ]


def score_discovery() -> dict[str, Any]:
    protocol = verify_protocol()
    low = read_jsonl_gz(OUT_ROOT / "runs/discovery/low_order_camera_inputs.jsonl.gz")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_holdout_responses.jsonl.gz")
    truth = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_model_truth.jsonl.gz")
    architecture_by_id = {row["model_id"]: row["architecture"] for row in truth}
    predictions = [camera_prediction(row) for row in low]
    errors = prediction_errors(predictions, holdout)
    nonabstain = [row for row in predictions if not row["abstain"]]
    architectures = {architecture_by_id[row["model_id"]] for row in nonabstain}
    distribution = dict(sorted(Counter(row["camera_decision"] for row in predictions).items()))
    metrics = {
        "unit_count": len(predictions),
        "nonabstain_unit_count": len(nonabstain),
        "nonabstain_fraction": len(nonabstain) / max(len(predictions), 1),
        "nonabstain_architecture_count": len(architectures),
        "holdout_mae": float(np.mean(errors)),
        "holdout_max_abs_error": float(max(errors)),
        "camera_decision_distribution": distribution,
    }
    checks = {
        "nonabstain_breadth": metrics["nonabstain_unit_count"] >= THRESHOLDS["discovery_nonabstain_unit_count_min"],
        "architecture_breadth": metrics["nonabstain_architecture_count"] >= THRESHOLDS["discovery_nonabstain_architecture_count_min"],
        "holdout_mae": metrics["holdout_mae"] <= THRESHOLDS["discovery_holdout_mae_max"],
        "holdout_max": metrics["holdout_max_abs_error"] <= THRESHOLDS["discovery_holdout_max_abs_error_max"],
    }
    value = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "metrics": metrics,
        "checks": checks,
        "confirmation_authorized": all(checks.values()),
        "predictions_digest": digest(predictions),
    }
    value["score_digest"] = digest(value)
    write_jsonl_gz(OUT_ROOT / "analysis/discovery_predictions.jsonl.gz", predictions)
    write_json(OUT_ROOT / "analysis/discovery_score.json", value)
    return value


def seal_confirmation_predictions() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    validate_digest(discovery, "score_digest")
    if not discovery["confirmation_authorized"]:
        raise RuntimeError("discovery denied confirmation predictions")
    low = read_jsonl_gz(OUT_ROOT / "runs/confirmation/low_order_camera_inputs.jsonl.gz")
    holdout_path = OUT_ROOT / "runs/confirmation/sealed_holdout_responses.jsonl.gz"
    predictions = [camera_prediction(row) for row in low]
    write_jsonl_gz(OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz", predictions)
    value = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_score_digest": discovery["score_digest"],
        "prediction_count": len(predictions),
        "prediction_digest": digest(predictions),
        "holdout_absent_at_prediction": not holdout_path.exists(),
    }
    value["manifest_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json", value)
    return value


def score_confirmation() -> dict[str, Any]:
    protocol = verify_protocol()
    manifest = read_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    predictions = read_jsonl_gz(OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
    if digest(predictions) != manifest["prediction_digest"]:
        raise RuntimeError("confirmation prediction drift")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_holdout_responses.jsonl.gz")
    truth = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_model_truth.jsonl.gz")
    architecture_by_id = {row["model_id"]: row["architecture"] for row in truth}
    errors = prediction_errors(predictions, holdout)
    nonabstain = [row for row in predictions if not row["abstain"]]
    architectures = {architecture_by_id[row["model_id"]] for row in nonabstain}
    metrics = {
        "unit_count": len(predictions),
        "nonabstain_unit_count": len(nonabstain),
        "nonabstain_fraction": len(nonabstain) / max(len(predictions), 1),
        "nonabstain_architecture_count": len(architectures),
        "holdout_mae": float(np.mean(errors)),
        "holdout_max_abs_error": float(max(errors)),
        "camera_decision_distribution": dict(sorted(Counter(row["camera_decision"] for row in predictions).items())),
    }
    checks = {
        "nonabstain_breadth": metrics["nonabstain_unit_count"] >= THRESHOLDS["confirmation_nonabstain_unit_count_min"],
        "architecture_breadth": metrics["nonabstain_architecture_count"] >= THRESHOLDS["confirmation_nonabstain_architecture_count_min"],
        "holdout_mae": metrics["holdout_mae"] <= THRESHOLDS["confirmation_holdout_mae_max"],
        "holdout_max": metrics["holdout_max_abs_error"] <= THRESHOLDS["confirmation_holdout_max_abs_error_max"],
    }
    value = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "prediction_manifest_digest": manifest["manifest_digest"],
        "metrics": metrics,
        "checks": checks,
        "external_validity_gate": all(checks.values()),
    }
    value["score_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", value)
    return value


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    validate_digest(discovery, "score_digest")
    confirmation_path = OUT_ROOT / "analysis/confirmation_score.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    if confirmation is not None:
        validate_digest(confirmation, "score_digest")
    passed = bool(confirmation and confirmation["external_validity_gate"])
    status = (
        "free_transformer_camera_transfer_confirmed"
        if passed else
        ("discovery_external_validity_stop" if not discovery["confirmation_authorized"] else "confirmation_external_validity_failed")
    )
    result = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_score_digest": discovery["score_digest"],
        "confirmation_score_digest": None if confirmation is None else confirmation["score_digest"],
        "learned_micro_transformer_external_validity": passed,
        "discovery": discovery["metrics"],
        "confirmation": None if confirmation is None else confirmation["metrics"],
        "claim_boundary": (
            "The Phase1208 known-truth quotient camera was transferred once to freely trained synthetic "
            "micro-Transformers. A failure means the frozen four-event response does not support that "
            "quotient and heldout-prediction claim; it does not show that the networks lack causal structure. "
            "A pass would remain synthetic and would not identify Qwen3 or natural-language mechanisms."
        ),
        "new_k_item": {
            "id": "K189",
            "level": "E3-KT",
            "statement": (
                "The one-shot learned micro-Transformer transfer of the Phase1208 necessity camera "
                + ("passed its cross-seed and cross-architecture heldout intervention gate." if passed else "did not pass its frozen discovery/confirmation external-validity gate; the known-truth quotient is not automatically a free-network mechanism coordinate.")
            ),
        },
        "auto_continue": False,
        "authorized_next": (
            "Only a new generative response model with an explicit discovery-only uncertainty/refusal rule; pretrained-model transfer remains denied."
            if not passed else
            "A separately preregistered natural task-family confirmation; pretrained-model transfer remains denied."
        ),
    }
    result["final_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/final.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preregister", "train-scout", "freeze-sites", "measure-low", "measure-holdout",
            "score-discovery", "seal-confirmation", "score-confirmation", "finalize",
        ),
    )
    parser.add_argument("--split", choices=("discovery", "confirmation"), default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.command == "preregister":
        result = preregister()
    elif args.command == "train-scout":
        if args.split is None:
            raise SystemExit("--split is required")
        result = train_and_scout(args.split, device)
    elif args.command == "freeze-sites":
        result = freeze_sites()
    elif args.command == "measure-low":
        if args.split is None:
            raise SystemExit("--split is required")
        result = measure_low_order(args.split, device)
    elif args.command == "measure-holdout":
        if args.split is None:
            raise SystemExit("--split is required")
        result = measure_holdout(args.split, device)
    elif args.command == "score-discovery":
        result = score_discovery()
    elif args.command == "seal-confirmation":
        result = seal_confirmation_predictions()
    elif args.command == "score-confirmation":
        result = score_confirmation()
    else:
        result = finalize()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
