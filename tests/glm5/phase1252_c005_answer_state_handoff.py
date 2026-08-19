#!/usr/bin/env python3
"""Phase1252: known-truth causal handoff into the answer state.

The experiment physically transplants or removes typed position groups at each
layer cut.  It asks when source and later-mapping counterfactuals become
sufficient at the answer token and when the remaining prefix can no longer
reconstruct them.  No observational camera is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer
from phase1251_c004_causal_slice_competition import (
    ANSWER_POSITION,
    CANDIDATE_SLICE,
    SOURCE_POSITIONS,
    build_sequence,
    centered_logits,
    train_model,
)


PHASE = 1252
CONTRACT_ID = "EXP-C005-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1252_c005_answer_state_handoff_audit.py"
PHASE1251_DEPENDENCY = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"
MODEL_DEPENDENCY = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
OUT = ROOT / "tests/glm5/result/phase1252_c005_answer_state_handoff"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_counterfactuals.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
CURVES = OUT / "raw/handoff_curves.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/handoff_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

ARCHITECTURES = {
    "shallow4": ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
    "middle6": ModelConfig(layers=6, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
    "deep8": ModelConfig(layers=8, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22),
}
REPLICATES = 2
MODEL_SEEDS = {
    "shallow4_r0": 1_252_401_001,
    "shallow4_r1": 1_252_401_101,
    "middle6_r0": 1_252_601_001,
    "middle6_r1": 1_252_601_101,
    "deep8_r0": 1_252_801_001,
    "deep8_r1": 1_252_801_101,
}
WORLD_SEED = 1_252_900_001
WORLD_COUNT = 64
CONDITIONS = ("source", "mapping", "joint")
POSITION_GROUPS = {
    "source": [4],
    "unqueried_source": [8],
    "mapping": list(range(11, 20)),
    "query": [20, 21],
    "answer": [22],
    "other": [0, 1, 2, 3, 5, 6, 7, 9, 10],
}
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "target_effect_norm_min": 5.0,
    "null_effect_fraction_max": 0.08,
    "endpoint_cosine_min": 0.9999,
    "endpoint_relative_error_max": 1.0e-4,
    "endpoint_remaining_fraction_max": 1.0e-4,
    "answer_onset_cosine_min": 0.90,
    "answer_onset_relative_error_max": 0.35,
    "answer_onset_projection_min": 0.65,
    "answer_lock_remaining_fraction_max": 0.20,
    "wrong_identity_cosine_max": 0.50,
    "identity_margin_min": 0.40,
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def make_worlds() -> list[dict[str, Any]]:
    rng = np.random.default_rng(WORLD_SEED)
    rows: list[dict[str, Any]] = []
    for group in range(WORLD_COUNT):
        codes = rng.choice(4, 2, replace=False).astype(int).tolist()
        alternatives = [value for value in range(4) if value not in codes]
        rng.shuffle(alternatives)
        target_code, wrong_code = int(alternatives[0]), int(alternatives[1])
        base_shift = int(rng.integers(4))
        shift_offsets = rng.choice(np.asarray([1, 2, 3]), 2, replace=False).astype(int).tolist()
        target_shift = (base_shift + shift_offsets[0]) % 4
        wrong_shift = (base_shift + shift_offsets[1]) % 4
        order = rng.permutation(4).astype(int).tolist()
        source_codes = list(codes)
        source_codes[0] = target_code
        wrong_source_codes = list(codes)
        wrong_source_codes[0] = wrong_code
        null_codes = list(codes)
        null_codes[1] = target_code
        base, positions = build_sequence(1, codes, base_shift, order)
        source, _ = build_sequence(1, source_codes, base_shift, order)
        mapping, _ = build_sequence(1, codes, target_shift, order)
        joint, _ = build_sequence(1, source_codes, target_shift, order)
        wrong_source, _ = build_sequence(1, wrong_source_codes, base_shift, order)
        wrong_mapping, _ = build_sequence(1, codes, wrong_shift, order)
        wrong_joint, _ = build_sequence(1, wrong_source_codes, wrong_shift, order)
        null, _ = build_sequence(1, null_codes, base_shift, order)
        row = {
            "row_id": f"g{group:03d}",
            "group": group,
            "codes": codes,
            "target_code": target_code,
            "wrong_code": wrong_code,
            "base_shift": base_shift,
            "target_shift": target_shift,
            "wrong_shift": wrong_shift,
            "codebook_order": order,
            "codebook_value_positions": {str(key): value for key, value in positions.items()},
            "base_ids": base,
            "source_ids": source,
            "mapping_ids": mapping,
            "joint_ids": joint,
            "wrong_source_ids": wrong_source,
            "wrong_mapping_ids": wrong_mapping,
            "wrong_joint_ids": wrong_joint,
            "null_ids": null,
            "answers": {
                "base": (codes[0] + base_shift) % 4,
                "source": (target_code + base_shift) % 4,
                "mapping": (codes[0] + target_shift) % 4,
                "joint": (target_code + target_shift) % 4,
                "wrong_source": (wrong_code + base_shift) % 4,
                "wrong_mapping": (codes[0] + wrong_shift) % 4,
                "wrong_joint": (wrong_code + wrong_shift) % 4,
                "null": (codes[0] + base_shift) % 4,
            },
        }
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "schema_version": "phase1252.c005.answer_handoff.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "known_truth_typed_state_handoff_causal_calibration",
        "question": "When do source and later-mapping counterfactuals become causally sufficient and necessary at the answer position?",
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "world_count": len(rows),
        "world_digest": digest([{key: row[key] for key in ("row_id", "row_digest")} for row in rows]),
        "conditions": list(CONDITIONS),
        "position_partition": POSITION_GROUPS,
        "interventions": {
            "rescue": "insert one typed position-group state from a counterfactual run into the base state at a fixed layer cut",
            "block": "restore one typed position-group state from base into the counterfactual state at a fixed layer cut",
            "origin": {"source": ["source"], "mapping": ["mapping"], "joint": ["source", "mapping"]},
            "wrong_identity": "at late and final cuts, transplant the corresponding answer state from a counterfactual with a different answer",
            "readout": "centered four-label logit response",
        },
        "metrics": {
            "rescue": ["cosine", "relative_error", "signed_projection", "norm_fraction"],
            "block": ["cosine", "relative_error", "signed_projection", "remaining_norm_fraction"],
            "answer_write_onset": "earliest layer meeting frozen rescue thresholds",
            "answer_lock_in": "earliest layer whose answer restoration to base leaves at most the frozen response fraction",
        },
        "thresholds": THRESHOLDS,
        "budgets": {"max_gpu_hours": 1.5, "max_formal_runs": 1, "max_adaptive_rounds": 0},
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDITOR),
            "phase1251_dependency": file_sha256(PHASE1251_DEPENDENCY),
            "model_dependency": file_sha256(MODEL_DEPENDENCY),
        },
        "hard_stops": [
            "Every behavior-unqualified seed remains in the breadth denominator and is not replaced.",
            "All layers and position groups are exhaustive and frozen; no hotspot selection occurs.",
            "Endpoint identities calibrate the intervention instrument but are not empirical mechanism discoveries.",
            "A handoff pass establishes a typed state-carrier transition in this synthetic task, not an attention edge, semantic circuit, or natural-language mechanism.",
            "Wrong-identity controls must remain separate from matched-null specificity.",
            "No Qwen3, GLM4, or DS7B run is authorized by this known-truth phase alone.",
        ],
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
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32_parameters_and_execution",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_worlds()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "worlds": len(rows)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source changed after preregistration")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if len(rows) != WORLD_COUNT:
        raise RuntimeError("world count drift")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("world digest mismatch")
    return protocol, rows


def response_metrics(response: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    response = response.float()
    full = full.float()
    full_norm = full.norm(dim=-1).clamp_min(1.0e-8)
    response_norm = response.norm(dim=-1)
    cosine = (response * full).sum(-1) / (response_norm * full_norm).clamp_min(1.0e-8)
    projection = (response * full).sum(-1) / full.pow(2).sum(-1).clamp_min(1.0e-8)
    relative_error = (response - full).norm(dim=-1) / full_norm
    return {
        "cosine_mean": float(cosine.mean().cpu()),
        "relative_error_mean": float(relative_error.mean().cpu()),
        "signed_projection_mean": float(projection.mean().cpu()),
        "norm_fraction_mean": float((response_norm / full_norm).mean().cpu()),
    }


def output_from_state(model: TinyCausalTransformer, state: torch.Tensor, layer: int) -> torch.Tensor:
    logits = model.forward_from(state, layer)
    return centered_logits(logits)


def positions_for(condition: str, group: str) -> list[int]:
    if group == "origin":
        return POSITION_GROUPS["source"] if condition == "source" else (
            POSITION_GROUPS["mapping"] if condition == "mapping" else POSITION_GROUPS["source"] + POSITION_GROUPS["mapping"]
        )
    return POSITION_GROUPS[group]


@torch.no_grad()
def causal_curves(model: TinyCausalTransformer, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    names = ("base", "source", "mapping", "joint", "wrong_source", "wrong_mapping", "wrong_joint", "null")
    inputs = {
        name: torch.tensor([row[f"{name}_ids"] for row in rows], device=device)
        for name in names
    }
    outputs: dict[str, torch.Tensor] = {}
    states: dict[str, list[torch.Tensor]] = {}
    for name in names:
        logits, hidden = model(inputs[name], return_states=True)
        outputs[name] = centered_logits(logits)
        states[name] = hidden
    full = {condition: outputs[condition] - outputs["base"] for condition in CONDITIONS}
    null = outputs["null"] - outputs["base"]
    full_norm = {condition: float(full[condition].norm(dim=-1).mean().cpu()) for condition in CONDITIONS}
    null_norm = float(null.norm(dim=-1).mean().cpu())
    curves: dict[str, Any] = {}
    wrong_names = {"source": "wrong_source", "mapping": "wrong_mapping", "joint": "wrong_joint"}
    for condition in CONDITIONS:
        condition_curve: list[dict[str, Any]] = []
        for layer in range(len(model.blocks) + 1):
            groups: dict[str, Any] = {}
            for group in (*POSITION_GROUPS.keys(), "origin"):
                positions = positions_for(condition, group)
                rescue_state = states["base"][layer].clone()
                rescue_state[:, positions] = states[condition][layer][:, positions]
                rescue_response = output_from_state(model, rescue_state, layer) - outputs["base"]
                blocked_state = states[condition][layer].clone()
                blocked_state[:, positions] = states["base"][layer][:, positions]
                blocked_response = output_from_state(model, blocked_state, layer) - outputs["base"]
                groups[group] = {
                    "rescue": response_metrics(rescue_response, full[condition]),
                    "block": response_metrics(blocked_response, full[condition]),
                }
            wrong_state = states["base"][layer].clone()
            wrong_state[:, ANSWER_POSITION] = states[wrong_names[condition]][layer][:, ANSWER_POSITION]
            wrong_response = output_from_state(model, wrong_state, layer) - outputs["base"]
            groups["wrong_answer_identity"] = {"rescue": response_metrics(wrong_response, full[condition])}
            condition_curve.append({"layer": layer, "relative_depth": layer / len(model.blocks), "groups": groups})
        curves[condition] = condition_curve
    return {
        "full_effect_norm": full_norm,
        "null_effect_norm": null_norm,
        "null_fraction_of_smallest_target": null_norm / max(min(full_norm.values()), 1.0e-9),
        "curves": curves,
    }


def find_write_onset(curve: list[dict[str, Any]]) -> int | None:
    for row in curve:
        metric = row["groups"]["answer"]["rescue"]
        if (
            metric["cosine_mean"] >= THRESHOLDS["answer_onset_cosine_min"]
            and metric["relative_error_mean"] <= THRESHOLDS["answer_onset_relative_error_max"]
            and metric["signed_projection_mean"] >= THRESHOLDS["answer_onset_projection_min"]
        ):
            return int(row["layer"])
    return None


def find_lock_in(curve: list[dict[str, Any]]) -> int | None:
    for row in curve:
        remaining = row["groups"]["answer"]["block"]["norm_fraction_mean"]
        if remaining <= THRESHOLDS["answer_lock_remaining_fraction_max"]:
            return int(row["layer"])
    return None


def endpoint_gate(curves: dict[str, Any], layers: int) -> bool:
    for condition in CONDITIONS:
        start = curves["curves"][condition][0]["groups"]["origin"]
        final = curves["curves"][condition][layers]["groups"]["answer"]
        if not (
            start["rescue"]["cosine_mean"] >= THRESHOLDS["endpoint_cosine_min"]
            and start["rescue"]["relative_error_mean"] <= THRESHOLDS["endpoint_relative_error_max"]
            and start["block"]["norm_fraction_mean"] <= THRESHOLDS["endpoint_remaining_fraction_max"]
            and final["rescue"]["cosine_mean"] >= THRESHOLDS["endpoint_cosine_min"]
            and final["rescue"]["relative_error_mean"] <= THRESHOLDS["endpoint_relative_error_max"]
            and final["block"]["norm_fraction_mean"] <= THRESHOLDS["endpoint_remaining_fraction_max"]
        ):
            return False
    return True


def execute_model(architecture: str, replicate: int, rows: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    key = model_key(architecture, replicate)
    model, behavior = train_model(ARCHITECTURES[architecture], MODEL_SEEDS[key], device)
    if min(behavior["accuracy_direct"], behavior["accuracy_code"]) < THRESHOLDS["behavior_accuracy_min"]:
        del model
        torch.cuda.empty_cache()
        return {
            "model_key": key,
            "architecture": architecture,
            "replicate": replicate,
            "behavior": behavior,
            "behavior_gate": False,
        }, {}
    curves = causal_curves(model, rows, device)
    layers = len(model.blocks)
    onsets = {condition: find_write_onset(curves["curves"][condition]) for condition in CONDITIONS}
    locks = {condition: find_lock_in(curves["curves"][condition]) for condition in CONDITIONS}
    identity: dict[str, Any] = {}
    late_layer = int(np.ceil(2 * layers / 3))
    identity_gate = True
    for condition in CONDITIONS:
        correct = curves["curves"][condition][late_layer]["groups"]["answer"]["rescue"]["cosine_mean"]
        wrong = curves["curves"][condition][late_layer]["groups"]["wrong_answer_identity"]["rescue"]["cosine_mean"]
        final_wrong = curves["curves"][condition][layers]["groups"]["wrong_answer_identity"]["rescue"]["cosine_mean"]
        identity[condition] = {
            "late_layer": late_layer,
            "correct_cosine": correct,
            "wrong_cosine": wrong,
            "margin": correct - wrong,
            "final_wrong_cosine": final_wrong,
        }
        identity_gate = identity_gate and correct - wrong >= THRESHOLDS["identity_margin_min"] and final_wrong <= THRESHOLDS["wrong_identity_cosine_max"]
    endpoints = endpoint_gate(curves, layers)
    specificity = bool(
        min(curves["full_effect_norm"].values()) >= THRESHOLDS["target_effect_norm_min"]
        and curves["null_fraction_of_smallest_target"] <= THRESHOLDS["null_effect_fraction_max"]
    )
    handoff = bool(
        endpoints
        and specificity
        and identity_gate
        and all(onsets[condition] is not None and 0 < onsets[condition] <= layers for condition in CONDITIONS)
        and all(locks[condition] is not None and onsets[condition] <= locks[condition] <= layers for condition in CONDITIONS)
    )
    summary = {
        "model_key": key,
        "architecture": architecture,
        "replicate": replicate,
        "behavior": behavior,
        "behavior_gate": True,
        "layers": layers,
        "endpoint_instrument_gate": endpoints,
        "specificity_gate": specificity,
        "identity_gate": bool(identity_gate),
        "answer_write_onset": onsets,
        "answer_lock_in": locks,
        "write_lock_gap": {condition: locks[condition] - onsets[condition] if locks[condition] is not None and onsets[condition] is not None else None for condition in CONDITIONS},
        "identity": identity,
        "full_effect_norm": curves["full_effect_norm"],
        "null_effect_norm": curves["null_effect_norm"],
        "null_fraction_of_smallest_target": curves["null_fraction_of_smallest_target"],
        "model_handoff_gate": handoff,
    }
    del model
    torch.cuda.empty_cache()
    return summary, {"model_key": key, **curves}


def formal_run() -> None:
    protocol, rows = verify_protocol()
    if not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("preaudit failed")
    if any(path.exists() for path in (RAW, CURVES, COMPLETE)):
        raise RuntimeError("one-shot formal output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    start = time.perf_counter()
    models: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for replicate in range(REPLICATES):
            summary, curves = execute_model(architecture, replicate, rows, torch.device("cuda"))
            models.append(summary)
            if curves:
                all_curves.append(curves)
            print(canonical_json({"status": "model_complete", "model": summary["model_key"], "handoff_gate": summary.get("model_handoff_gate")}), flush=True)
    elapsed = time.perf_counter() - start
    curves_payload = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "models": all_curves,
    }
    curves_payload["curves_digest"] = digest(curves_payload)
    atomic_json(CURVES, curves_payload)
    payload = {
        "phase": PHASE,
        "schema_version": "phase1252.c005.answer_handoff.run.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "models": models,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "curves_sha256": file_sha256(CURVES),
        "pretrained_model_loaded": False,
    }
    payload["run_digest"] = digest(payload)
    atomic_json(RAW, payload)
    marker = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": payload["run_digest"],
        "raw_sha256": file_sha256(RAW),
        "curves_sha256": file_sha256(CURVES),
        "status": "formal_run_complete",
    }
    marker["marker_digest"] = digest(marker)
    atomic_json(COMPLETE, marker)
    print(canonical_json({"status": "formal_complete", "models": len(models), "gpu_hours": payload["gpu_hours"]}))


def breadth(models: list[dict[str, Any]], field: str) -> tuple[bool, dict[str, int]]:
    per_depth = {
        architecture: sum(bool(row.get(field)) for row in models if row["architecture"] == architecture)
        for architecture in ARCHITECTURES
    }
    passed = sum(bool(row.get(field)) for row in models) >= THRESHOLDS["breadth_models_min"] and all(
        value >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values()
    )
    return bool(passed), per_depth


def analyze() -> None:
    protocol, _rows = verify_protocol()
    run = read_json(RAW)
    marker = read_json(COMPLETE)
    if marker["run_digest"] != run["run_digest"] or marker["raw_sha256"] != file_sha256(RAW):
        raise RuntimeError("completion marker mismatch")
    if run["curves_sha256"] != file_sha256(CURVES) or marker["curves_sha256"] != run["curves_sha256"]:
        raise RuntimeError("curve hash mismatch")
    models = run["models"]
    behavior_gate, behavior_depth = breadth(models, "behavior_gate")
    endpoint_gate_value, endpoint_depth = breadth(models, "endpoint_instrument_gate")
    specificity_gate_value, specificity_depth = breadth(models, "specificity_gate")
    identity_gate_value, identity_depth = breadth(models, "identity_gate")
    handoff_gate, handoff_depth = breadth(models, "model_handoff_gate")
    gates = {
        "G-BEHAVIOR-BREADTH": behavior_gate,
        "G-ENDPOINT-INSTRUMENT": endpoint_gate_value,
        "G-SPECIFICITY": specificity_gate_value,
        "G-IDENTITY": identity_gate_value,
        "G-HANDOFF-BREADTH": handoff_gate,
    }
    verdict = "known_truth_answer_state_handoff_confirmed" if all(gates.values()) else "known_truth_answer_state_handoff_not_confirmed"
    adjudication = {
        "phase": PHASE,
        "schema_version": "phase1252.c005.answer_handoff.adjudication.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_digest": run["run_digest"],
        "verdict": verdict,
        "gates": gates,
        "per_depth": {
            "behavior": behavior_depth,
            "endpoint": endpoint_depth,
            "specificity": specificity_depth,
            "identity": identity_depth,
            "handoff": handoff_depth,
        },
        "models": models,
        "authorization": {
            "fresh_qwen_single_model_typed_handoff_contract": bool(all(gates.values())),
            "semantic_mechanism_claim": False,
            "attention_edge_claim": False,
            "cross_model_claim": False,
        },
        "interpretation": [
            "The endpoint identities audit the state-transplant instrument and are not discoveries.",
            "A nonzero write-lock gap means the answer state can carry the response before the remaining prefix loses its capacity to reconstruct it.",
            "Wrong-identity answer states test response identity, while matched-null tests query specificity.",
            "A pass confirms a typed causal carrier handoff in free synthetic Transformers; it does not identify attention heads or natural-language semantics.",
        ],
    }
    adjudication["adjudication_digest"] = digest(adjudication)
    atomic_json(ANALYSIS, adjudication)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "gates": gates,
        "per_depth": adjudication["per_depth"],
        "authorization": adjudication["authorization"],
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "material": file_sha256(MATERIAL),
            "environment": file_sha256(ENVIRONMENT),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "curves": file_sha256(CURVES),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"status": "analyzed", "verdict": verdict, "gates": gates}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        formal_run()
    else:
        analyze()


if __name__ == "__main__":
    main()
