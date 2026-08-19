"""Phase1260: free-Transformer external validity of the Phase1259 camera.

Six freely trained same-executor Transformers (4/6/8 layers, two seeds each)
are tested.  Discovery fits target-vs-context response operators at every
answer-boundary residual event.  Selection freezes the operator type and an
early/late event pair.  Confirmation alone performs target, wrong and null
patches plus downstream block/correct-rescue/wrong-rescue tests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1251_c004_causal_slice_competition as task_module
import phase1254_c007_free_transformer_edge_external_validity as edge_base
import phase1255_c008_same_executor_edge_external_validity as same_executor
from phase1146_learned_composition_benchmark import ModelConfig
from phase1251_c004_causal_slice_competition import CANDIDATE_SLICE, build_sequence


PHASE = 1260
CAMPAIGN = "C011"
CONTRACT_ID = "EXP-C011-WP02-001"
OUT = ROOT / "tests/glm5/result/phase1260_c011_free_transformer_selective_operator_mediation"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1260_c011_free_transformer_selective_operator_mediation_audit.py"
PHASE1259_FINAL = ROOT / "tests/glm5/result/phase1259_c011_selective_operator_mediation_calibration/analysis/final.json"
PHASE1259_AUDIT = ROOT / "tests/glm5/result/phase1259_c011_selective_operator_mediation_calibration/audit/independent_final_audit.json"
PROBE = ROOT / "tests/glm5_temp/phase1260_c011_free_transformer_probe.json"

ARCHITECTURES = same_executor.ARCHITECTURES
REPLICATES = 2
MODEL_SEEDS = {
    "shallow4_r0": 1_260_401_001,
    "shallow4_r1": 1_260_401_101,
    "middle6_r0": 1_260_601_001,
    "middle6_r1": 1_260_601_101,
    "deep8_r0": 1_260_801_001,
    "deep8_r1": 1_260_801_101,
}
WORLD_SEED = 1_260_900_001
WORLD_COUNTS = {"discovery": 128, "selection": 128, "confirmation": 256}
THRESHOLDS = {
    "behavior_accuracy_min": 0.995,
    "executor_gap_max": 2.0e-4,
    "selection_target_relative_error_max": 0.55,
    "selection_null_leakage_max": 0.20,
    "selection_idempotence_error_max": 0.15,
    "minimum_selected_events": 2,
    "context_probe_selection_accuracy_min": 0.80,
    "correct_cosine_min": 0.85,
    "correct_relative_error_max": 0.65,
    "correct_accuracy_min": 0.90,
    "wrong_identity_accuracy_min": 0.90,
    "wrong_false_target_rate_max": 0.10,
    "null_effect_fraction_max": 0.15,
    "block_remaining_fraction_max": 0.40,
    "rescue_accuracy_min": 0.90,
    "wrong_rescue_identity_accuracy_min": 0.90,
    "context_probe_confirmation_accuracy_min": 0.80,
    "context_probe_retention_min": 0.90,
    "mediator_target_state_relative_error_max": 0.75,
    "mediator_target_nearest_fraction_min": 0.80,
    "breadth_models_min": 4,
    "breadth_per_depth_min": 1,
    "random_control_target_relative_error_min": 0.35,
    "full_state_null_leakage_min": 0.99,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_key(architecture: str, replicate: int) -> str:
    return f"{architecture}_r{replicate}"


def make_worlds(seed: int = WORLD_SEED, counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    counts = counts or WORLD_COUNTS
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group = 0
    for partition, count in counts.items():
        for _ in range(count):
            codes = rng.choice(4, 2, replace=False).astype(int).tolist()
            alternatives = [value for value in range(4) if value not in codes]
            rng.shuffle(alternatives)
            target_code, wrong_code = alternatives
            context_alt = target_code
            shift = int(rng.integers(4))
            order = rng.permutation(4).astype(int).tolist()
            panels = {
                "base": codes,
                "target": [target_code, codes[1]],
                "wrong": [wrong_code, codes[1]],
                "null": [codes[0], context_alt],
            }
            sequences = {name: build_sequence(1, values, shift, order)[0] for name, values in panels.items()}
            row = {
                "row_id": f"g{group:04d}",
                "group": group,
                "partition": partition,
                "shift": shift,
                "codes": codes,
                "target_code": target_code,
                "wrong_code": wrong_code,
                "context_alt": context_alt,
                "codebook_order": order,
                **{f"{name}_ids": sequence for name, sequence in sequences.items()},
                "answers": {
                    "base": (codes[0] + shift) % 4,
                    "target": (target_code + shift) % 4,
                    "wrong": (wrong_code + shift) % 4,
                    "null": (codes[0] + shift) % 4,
                    "context": (codes[1] + shift) % 4,
                    "context_alt": (context_alt + shift) % 4,
                },
            }
            row["row_digest"] = digest(row)
            rows.append(row)
            group += 1
    return rows


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1260.c011.free_transformer_selective_operator.protocol.v1",
        "claim_type": "free_transformer_selective_operator_mediation_external_validity",
        "question": "Does the calibrated global/conditioned/abstain response camera yield a held-out target-selective and path-mediated intervention in freely trained same-executor Transformers across depth?",
        "phase1259_dependency": {"final": file_sha256(PHASE1259_FINAL), "audit": file_sha256(PHASE1259_AUDIT)},
        "architectures": {name: vars(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "model_seeds": MODEL_SEEDS,
        "world_counts": WORLD_COUNTS,
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "row_digest")} for row in rows]),
        "executor_invariant": "training, natural evaluation and intervention all use the same explicit QK/OV/MLP arithmetic program",
        "camera": {
            "events": "answer-boundary residual after every complete Transformer block",
            "candidates": ["global_oblique_operator", "shift_conditioned_oblique_operator", "typed_abstention"],
            "fit": "discovery target deltas are identity constraints; discovery matched-null deltas are zero constraints",
            "selection": "per layer choose global, else conditioned, else abstain; freeze earliest and latest passing layers",
            "confirmation": "target/wrong/null upstream patch; downstream block; correct/wrong rescue; no reselection",
            "context_diagnostic": "discovery-fitted linear readout of the unqueried record; diagnostic preservation only, not natural-use proof",
        },
        "thresholds": THRESHOLDS,
        "gates": ["behavior", "camera_breadth", "target_rescue", "wrong_rejection", "matched_null", "path_mediation", "context_probe_preservation", "manifold_proximity", "controls"],
        "budgets": {"max_formal_runs": 1, "max_adaptive_rounds": 0, "max_gpu_hours": 1.5},
        "hard_stops": [
            "Behavior-unqualified models and camera abstentions remain in the breadth denominator and are never replaced.",
            "No confirmation row may choose operator type, event layers, threshold or model.",
            "The context readout is diagnostic decodability, not evidence that the network naturally uses that state.",
            "A pass is restricted to this cyclic-code synthetic task and free small Transformers.",
            "Failure blocks Qwen3 escalation; pass authorizes only a separately frozen natural-template contract, not an automatic language-mechanism claim.",
        ],
        "source_hashes": {
            "main": file_sha256(Path(__file__).resolve()),
            "auditor": file_sha256(AUDITOR),
            "same_executor": file_sha256(ROOT / "tests/glm5/phase1255_c008_same_executor_edge_external_validity.py"),
            "edge_executor": file_sha256(ROOT / "tests/glm5/phase1254_c007_free_transformer_edge_external_validity.py"),
            "task": file_sha256(ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"),
        },
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "fp32_same_explicit_executor_training_natural_intervention; fp64_operator_fit",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    dependency = read_json(PHASE1259_FINAL)
    audit = read_json(PHASE1259_AUDIT)
    if dependency.get("verdict") != "selective_operator_mediation_camera_calibrated" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1259 did not authorize WP02")
    rows = make_worlds()
    write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "worlds": len(rows), "models": len(MODEL_SEEDS)}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    expected = protocol_payload(rows)
    if protocol["protocol_digest"] != expected["protocol_digest"] or protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("protocol or source drift")
    if protocol["thresholds"] != THRESHOLDS:
        raise RuntimeError("threshold drift")
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        if digest(value) != stored:
            raise RuntimeError("material digest mismatch")
    counts = {name: sum(row["partition"] == name for row in rows) for name in WORLD_COUNTS}
    if counts != WORLD_COUNTS:
        raise RuntimeError(f"partition drift: {counts}")
    return protocol, rows


Action = Callable[[torch.Tensor], torch.Tensor]


def explicit_residual_forward(
    model,
    input_ids: torch.Tensor,
    actions: dict[int, Action] | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden = model.embed(input_ids)
    residuals: list[torch.Tensor] = []
    length = input_ids.shape[1]
    causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
    for layer_index, block in enumerate(model.blocks):
        normalized = block.attn_norm(hidden)
        batch, _, width = normalized.shape
        qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        weights = torch.softmax(scores.masked_fill(causal[None, None], float("-inf")), dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + block.attn.out(attended)
        hidden = hidden + block.mlp(block.mlp_norm(hidden))
        if actions and layer_index in actions:
            hidden = hidden.clone()
            hidden[:, -1, :] = actions[layer_index](hidden[:, -1, :])
        if capture:
            residuals.append(hidden[:, -1, :].detach().clone())
    logits = model.lm_head(model.final_norm(hidden))
    return logits, torch.stack(residuals, dim=1) if capture else None


def centered(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, -1, CANDIDATE_SLICE].double()
    return values - values.mean(dim=-1, keepdim=True)


def basis(samples: torch.Tensor) -> torch.Tensor:
    _u, singular, vh = torch.linalg.svd(samples.double(), full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 1.0e-12:
        return torch.zeros((samples.shape[1], 0), dtype=torch.float64, device=samples.device)
    rank = int(torch.sum(singular > singular[0] * 1.0e-6).item())
    return vh[:rank].T.contiguous()


def fit_operator(target: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    target_basis = basis(target)
    null_basis = basis(null)
    combined = torch.cat((target_basis, null_basis), dim=1)
    selector = torch.cat((
        torch.eye(target_basis.shape[1], dtype=torch.float64, device=target.device),
        torch.zeros((target_basis.shape[1], null_basis.shape[1]), dtype=torch.float64, device=target.device),
    ), dim=1)
    return target_basis @ selector @ torch.linalg.pinv(combined, rtol=1.0e-6, atol=1.0e-8)


def apply_operator(delta: torch.Tensor, operator: torch.Tensor | dict[int, torch.Tensor], condition: torch.Tensor) -> torch.Tensor:
    values = delta.double()
    if isinstance(operator, dict):
        stack = torch.stack([operator[int(value)] for value in condition.tolist()])
        return torch.bmm(stack, values.unsqueeze(-1)).squeeze(-1).to(delta.dtype)
    return (values @ operator.T).to(delta.dtype)


def operator_metrics(operator: torch.Tensor | dict[int, torch.Tensor], target: torch.Tensor, null: torch.Tensor, condition: torch.Tensor) -> dict[str, float]:
    predicted_target = apply_operator(target, operator, condition).double()
    predicted_null = apply_operator(null, operator, condition).double()
    target64, null64 = target.double(), null.double()
    if isinstance(operator, dict):
        idem = max(float((torch.linalg.vector_norm(value @ value - value) / torch.linalg.vector_norm(value).clamp_min(1.0e-12)).item()) for value in operator.values())
    else:
        idem = float((torch.linalg.vector_norm(operator @ operator - operator) / torch.linalg.vector_norm(operator).clamp_min(1.0e-12)).item())
    return {
        "target_relative_error": float((torch.linalg.vector_norm(predicted_target - target64) / torch.linalg.vector_norm(target64).clamp_min(1.0e-12)).item()),
        "null_leakage": float((torch.linalg.vector_norm(predicted_null) / torch.linalg.vector_norm(null64).clamp_min(1.0e-12)).item()),
        "idempotence_error": idem,
        "target_norm": float(torch.linalg.vector_norm(target64).item()),
        "null_norm": float(torch.linalg.vector_norm(null64).item()),
    }


def operator_passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["target_relative_error"] <= THRESHOLDS["selection_target_relative_error_max"]
        and metrics["null_leakage"] <= THRESHOLDS["selection_null_leakage_max"]
        and metrics["idempotence_error"] <= THRESHOLDS["selection_idempotence_error_max"]
    )


def fit_layer_cameras(states: dict[str, torch.Tensor], shifts: torch.Tensor, partitions: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    target_delta = states["target"] - states["base"]
    null_delta = states["null"] - states["base"]
    cameras = []
    for layer in range(target_delta.shape[1]):
        discovery = partitions["discovery"]
        selection = partitions["selection"]
        global_operator = fit_operator(target_delta[discovery, layer], null_delta[discovery, layer])
        conditioned = {}
        for shift in range(4):
            mask = discovery[shifts[discovery] == shift]
            conditioned[shift] = fit_operator(target_delta[mask, layer], null_delta[mask, layer])
        global_metrics = operator_metrics(global_operator, target_delta[selection, layer], null_delta[selection, layer], shifts[selection])
        conditioned_metrics = operator_metrics(conditioned, target_delta[selection, layer], null_delta[selection, layer], shifts[selection])
        if operator_passes(global_metrics):
            selected_type, selected = "global", global_operator
        elif operator_passes(conditioned_metrics):
            selected_type, selected = "conditioned", conditioned
        else:
            selected_type, selected = "abstain", None
        target_basis = basis(target_delta[discovery, layer])
        generator = torch.Generator(device=target_delta.device)
        generator.manual_seed(1_260_100 + layer)
        random = torch.randn((target_delta.shape[-1], max(1, target_basis.shape[1])), generator=generator, dtype=torch.float64, device=target_delta.device)
        random_basis, _ = torch.linalg.qr(random, mode="reduced")
        random_operator = random_basis @ random_basis.T
        controls = {
            "full_state": operator_metrics(torch.eye(target_delta.shape[-1], dtype=torch.float64, device=target_delta.device), target_delta[selection, layer], null_delta[selection, layer], shifts[selection]),
            "random": operator_metrics(random_operator, target_delta[selection, layer], null_delta[selection, layer], shifts[selection]),
        }
        cameras.append({"layer": layer, "selected_type": selected_type, "selected": selected, "global_metrics": global_metrics, "conditioned_metrics": conditioned_metrics, "controls": controls})
    return cameras


def fit_context_probe(states: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    x = states[indices].double()
    x = torch.cat((x, torch.ones((x.shape[0], 1), dtype=torch.float64, device=x.device)), dim=1)
    y = F.one_hot(labels[indices], num_classes=4).double()
    return torch.linalg.pinv(x, rtol=1.0e-6, atol=1.0e-8) @ y


def context_predict(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x = torch.cat((states.double(), torch.ones((states.shape[0], 1), dtype=torch.float64, device=states.device)), dim=1)
    return torch.argmax(x @ weights, dim=-1)


def output_metrics(response: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p, t = response.reshape(-1), target.reshape(-1)
    pn, tn = torch.linalg.vector_norm(p).clamp_min(1.0e-12), torch.linalg.vector_norm(t).clamp_min(1.0e-12)
    return {
        "cosine": float((torch.dot(p, t) / (pn * tn)).item()),
        "relative_error": float((torch.linalg.vector_norm(p - t) / tn).item()),
        "projection": float((torch.dot(p, t) / torch.dot(t, t).clamp_min(1.0e-12)).item()),
    }


def run_model(architecture: str, replicate: int, config: ModelConfig, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    key = model_key(architecture, replicate)
    seed = MODEL_SEEDS[key]
    set_seed(seed)
    model, training = task_module.train_model(config, seed, device)
    ids = {name: torch.tensor([row[f"{name}_ids"] for row in rows], device=device) for name in ("base", "target", "wrong", "null")}
    labels = {name: torch.tensor([row["answers"][name] for row in rows], device=device) for name in ("base", "target", "wrong", "null")}
    context_labels = torch.tensor([row["answers"]["context"] for row in rows], device=device)
    shifts = torch.tensor([row["shift"] for row in rows], device=device)
    partition_indices = {name: torch.tensor([index for index, row in enumerate(rows) if row["partition"] == name], device=device) for name in WORLD_COUNTS}
    natural_logits: dict[str, torch.Tensor] = {}
    states: dict[str, torch.Tensor] = {}
    executor_gaps = {}
    natural_accuracies = {}
    with torch.inference_mode():
        for panel in ids:
            native = model(ids[panel])
            explicit, residual = explicit_residual_forward(model, ids[panel], capture=True)
            assert residual is not None
            natural_logits[panel], states[panel] = explicit, residual
            executor_gaps[panel] = float(torch.max(torch.abs(native.float() - explicit.float())).item())
            natural_accuracies[panel] = float((torch.argmax(explicit[:, -1, CANDIDATE_SLICE], dim=-1) == labels[panel]).float().mean().item())
    behavior_qualified = min(training["accuracy_overall"], training["accuracy_direct"], training["accuracy_code"], *natural_accuracies.values()) >= THRESHOLDS["behavior_accuracy_min"] and max(executor_gaps.values()) <= THRESHOLDS["executor_gap_max"]
    base_result = {
        "model_key": key, "architecture": architecture, "replicate": replicate, "seed": seed,
        "training": training, "natural_accuracies": natural_accuracies, "executor_gaps": executor_gaps,
        "behavior_qualified": behavior_qualified,
    }
    if not behavior_qualified:
        return {**base_result, "selected_event_pair": None, "passed": False}

    cameras = fit_layer_cameras(states, shifts, partition_indices)
    passing = [camera for camera in cameras if camera["selected_type"] != "abstain"]
    if len(passing) < THRESHOLDS["minimum_selected_events"]:
        return {
            **base_result,
            "layer_cameras": [{key: value for key, value in camera.items() if key != "selected"} for camera in cameras],
            "selected_event_pair": None,
            "camera_abstained": True,
            "passed": False,
        }
    upstream_camera, mediator_camera = passing[0], passing[-1]
    if upstream_camera["layer"] >= mediator_camera["layer"]:
        return {**base_result, "selected_event_pair": None, "camera_abstained": True, "passed": False}
    upstream, mediator = upstream_camera["layer"], mediator_camera["layer"]
    op_up, op_med = upstream_camera["selected"], mediator_camera["selected"]
    assert op_up is not None and op_med is not None

    discovery = partition_indices["discovery"]
    selection = partition_indices["selection"]
    confirmation = partition_indices["confirmation"]
    context_train_states = torch.cat((states["base"][:, mediator], states["target"][:, mediator], states["wrong"][:, mediator]), dim=0)
    context_train_labels = context_labels.repeat(3)
    discovery_context = torch.cat((discovery, discovery + len(rows), discovery + 2 * len(rows)))
    context_weights = fit_context_probe(context_train_states, context_train_labels, discovery_context)
    context_selection_accuracy = float((context_predict(states["base"][selection, mediator], context_weights) == context_labels[selection]).float().mean().item())
    if context_selection_accuracy < THRESHOLDS["context_probe_selection_accuracy_min"]:
        return {
            **base_result,
            "layer_cameras": [{key: value for key, value in camera.items() if key != "selected"} for camera in cameras],
            "selected_event_pair": [upstream, mediator],
            "selected_camera_types": [upstream_camera["selected_type"], mediator_camera["selected_type"]],
            "context_probe_selection_accuracy": context_selection_accuracy,
            "context_probe_qualified": False,
            "passed": False,
        }

    idx = confirmation
    condition = shifts[idx]
    base_up, target_up, wrong_up, null_up = (states[name][idx, upstream] for name in ("base", "target", "wrong", "null"))
    base_med, target_med, wrong_med, null_med = (states[name][idx, mediator] for name in ("base", "target", "wrong", "null"))

    def evaluate(donor: str, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        donor_up = {"target": target_up, "wrong": wrong_up, "null": null_up}[donor]
        actions: dict[int, Action] = {
            upstream: lambda current: current + apply_operator(donor_up - base_up, op_up, condition),
        }
        if mode != "direct":
            def mediator_action(current: torch.Tensor) -> torch.Tensor:
                value = current + apply_operator(base_med - current, op_med, condition)
                if mode == "rescue":
                    value = value + apply_operator(target_med - value, op_med, condition)
                elif mode == "wrong_rescue":
                    value = value + apply_operator(wrong_med - value, op_med, condition)
                return value
            actions[mediator] = mediator_action
        logits, trace = explicit_residual_forward(model, ids["base"][idx], actions=actions, capture=True)
        assert trace is not None
        return logits, trace[:, mediator]

    with torch.inference_mode():
        correct_logits, correct_med = evaluate("target", "direct")
        wrong_logits, wrong_patch_med = evaluate("wrong", "direct")
        null_logits, null_patch_med = evaluate("null", "direct")
        block_logits, block_med = evaluate("target", "block")
        rescue_logits, rescue_med = evaluate("target", "rescue")
        wrong_rescue_logits, wrong_rescue_med = evaluate("target", "wrong_rescue")

    base_output = centered(natural_logits["base"][idx])
    target_effect = centered(natural_logits["target"][idx]) - base_output
    correct_response = centered(correct_logits) - base_output
    wrong_response = centered(wrong_logits) - base_output
    null_response = centered(null_logits) - base_output
    block_response = centered(block_logits) - base_output
    rescue_response = centered(rescue_logits) - base_output
    wrong_rescue_response = centered(wrong_rescue_logits) - base_output
    target_norm = torch.linalg.vector_norm(target_effect).clamp_min(1.0e-12)
    natural_target_state_effect = target_med - base_med
    state_norm = torch.linalg.vector_norm(natural_target_state_effect.double()).clamp_min(1.0e-12)
    candidate_states = torch.stack((base_med, target_med, wrong_med, null_med), dim=1)
    distances = torch.linalg.vector_norm(correct_med.unsqueeze(1) - candidate_states, dim=-1)
    nearest = torch.argmin(distances, dim=1)
    context_base_prediction = context_predict(base_med, context_weights)
    context_correct_prediction = context_predict(correct_med, context_weights)
    context_null_prediction = context_predict(null_patch_med, context_weights)
    context_rescue_prediction = context_predict(rescue_med, context_weights)
    correct_output = torch.argmax(correct_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    wrong_output = torch.argmax(wrong_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    block_output = torch.argmax(block_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    rescue_output = torch.argmax(rescue_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    wrong_rescue_output = torch.argmax(wrong_rescue_logits[:, -1, CANDIDATE_SLICE], dim=-1)
    metrics = {
        "cases": int(idx.numel()),
        "correct": output_metrics(correct_response, target_effect),
        "correct_accuracy": float((correct_output == labels["target"][idx]).float().mean().item()),
        "wrong_identity_accuracy": float((wrong_output == labels["wrong"][idx]).float().mean().item()),
        "wrong_false_target_rate": float((wrong_output == labels["target"][idx]).float().mean().item()),
        "wrong_response_cosine_to_target": output_metrics(wrong_response, target_effect)["cosine"],
        "null_effect_fraction": float((torch.linalg.vector_norm(null_response) / target_norm).item()),
        "block_remaining_fraction": float((torch.linalg.vector_norm(block_response) / target_norm).item()),
        "block_base_accuracy": float((block_output == labels["base"][idx]).float().mean().item()),
        "rescue": output_metrics(rescue_response, target_effect),
        "rescue_accuracy": float((rescue_output == labels["target"][idx]).float().mean().item()),
        "wrong_rescue_identity_accuracy": float((wrong_rescue_output == labels["wrong"][idx]).float().mean().item()),
        "wrong_rescue_false_target_rate": float((wrong_rescue_output == labels["target"][idx]).float().mean().item()),
        "mediator_target_state_relative_error": float((torch.linalg.vector_norm((correct_med - target_med).double()) / state_norm).item()),
        "mediator_target_nearest_fraction": float((nearest == 1).float().mean().item()),
        "context_probe_base_accuracy": float((context_base_prediction == context_labels[idx]).float().mean().item()),
        "context_probe_correct_accuracy": float((context_correct_prediction == context_labels[idx]).float().mean().item()),
        "context_probe_null_accuracy": float((context_null_prediction == context_labels[idx]).float().mean().item()),
        "context_probe_rescue_accuracy": float((context_rescue_prediction == context_labels[idx]).float().mean().item()),
        "context_probe_correct_retention": float((context_correct_prediction == context_base_prediction).float().mean().item()),
        "context_probe_null_retention": float((context_null_prediction == context_base_prediction).float().mean().item()),
        "context_probe_rescue_retention": float((context_rescue_prediction == context_base_prediction).float().mean().item()),
        "state_diagnostics": {
            "null_mediator_fraction": float((torch.linalg.vector_norm((null_patch_med - base_med).double()) / state_norm).item()),
            "block_mediator_fraction": float((torch.linalg.vector_norm((block_med - base_med).double()) / state_norm).item()),
            "rescue_mediator_relative_error": float((torch.linalg.vector_norm((rescue_med - target_med).double()) / state_norm).item()),
            "wrong_rescue_mediator_relative_error": float((torch.linalg.vector_norm((wrong_rescue_med - wrong_med).double()) / torch.linalg.vector_norm((wrong_med - base_med).double()).clamp_min(1.0e-12)).item()),
        },
    }
    context_min = min(metrics["context_probe_base_accuracy"], metrics["context_probe_correct_accuracy"], metrics["context_probe_null_accuracy"], metrics["context_probe_rescue_accuracy"])
    retention_min = min(metrics["context_probe_correct_retention"], metrics["context_probe_null_retention"], metrics["context_probe_rescue_retention"])
    passed = (
        metrics["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"]
        and metrics["correct"]["relative_error"] <= THRESHOLDS["correct_relative_error_max"]
        and metrics["correct_accuracy"] >= THRESHOLDS["correct_accuracy_min"]
        and metrics["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["wrong_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"]
        and metrics["null_effect_fraction"] <= THRESHOLDS["null_effect_fraction_max"]
        and metrics["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"]
        and metrics["rescue_accuracy"] >= THRESHOLDS["rescue_accuracy_min"]
        and metrics["wrong_rescue_identity_accuracy"] >= THRESHOLDS["wrong_rescue_identity_accuracy_min"]
        and context_min >= THRESHOLDS["context_probe_confirmation_accuracy_min"]
        and retention_min >= THRESHOLDS["context_probe_retention_min"]
        and metrics["mediator_target_state_relative_error"] <= THRESHOLDS["mediator_target_state_relative_error_max"]
        and metrics["mediator_target_nearest_fraction"] >= THRESHOLDS["mediator_target_nearest_fraction_min"]
    )
    return {
        **base_result,
        "layer_cameras": [{key: value for key, value in camera.items() if key != "selected"} for camera in cameras],
        "selected_event_pair": [upstream, mediator],
        "selected_relative_event_pair": [upstream / max(1, config.layers - 1), mediator / max(1, config.layers - 1)],
        "selected_camera_types": [upstream_camera["selected_type"], mediator_camera["selected_type"]],
        "context_probe_selection_accuracy": context_selection_accuracy,
        "context_probe_qualified": True,
        "confirmation": metrics,
        "controls": {
            "full_state_null_leakage_applicable": [
                camera["controls"]["full_state"]["null_leakage"]
                for camera in (upstream_camera, mediator_camera)
                if camera["controls"]["full_state"]["null_norm"] > 1.0e-8
            ],
            "random_target_relative_error_min": min(upstream_camera["controls"]["random"]["target_relative_error"], mediator_camera["controls"]["random"]["target_relative_error"]),
        },
        "passed": passed,
    }


def run(device_name: str) -> None:
    protocol, rows = verify_protocol()
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass")
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    results = []
    for architecture, config in ARCHITECTURES.items():
        for replicate in range(REPLICATES):
            result = run_model(architecture, replicate, config, rows, device)
            results.append(result)
            write_jsonl(MODELS, results)
            print(canonical_json({"completed": len(results), "total": len(MODEL_SEEDS), "model": result["model_key"], "behavior": result["behavior_qualified"], "pair": result.get("selected_event_pair"), "passed": result["passed"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    summary = {
        "phase": PHASE, "created_at_utc": utc_now(), "models": len(results),
        "elapsed_seconds": time.perf_counter() - started, "gpu_hours": (time.perf_counter() - started) / 3600.0,
        "run_digest": digest(results), "models_hash": file_sha256(MODELS), "protocol_digest": protocol["protocol_digest"],
    }
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "models_hash": file_sha256(MODELS), "summary_hash": file_sha256(SUMMARY)})
    print(canonical_json({"status": "formal_run_complete", "elapsed_seconds": summary["elapsed_seconds"]}))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in rows if row["behavior_qualified"]]
    actionable = [row for row in rows if row.get("selected_event_pair") is not None and row.get("context_probe_qualified")]
    passed = [row for row in rows if row["passed"]]
    per_depth = {
        architecture: {
            "models": sum(row["architecture"] == architecture for row in rows),
            "behavior_qualified": sum(row["architecture"] == architecture and row["behavior_qualified"] for row in rows),
            "camera_actionable": sum(row["architecture"] == architecture and row.get("selected_event_pair") is not None and row.get("context_probe_qualified") for row in rows),
            "passed": sum(row["architecture"] == architecture and row["passed"] for row in rows),
        }
        for architecture in ARCHITECTURES
    }
    breadth = len(passed) >= THRESHOLDS["breadth_models_min"] and all(value["passed"] >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values())
    controls = [row["controls"] for row in actionable]
    applicable_full_state = [value for control in controls for value in control["full_state_null_leakage_applicable"]]
    gates = {
        "G-BEHAVIOR": len(qualified) == len(rows),
        "G-CAMERA-BREADTH": len(actionable) >= THRESHOLDS["breadth_models_min"] and all(value["camera_actionable"] >= THRESHOLDS["breadth_per_depth_min"] for value in per_depth.values()),
        "G-TARGET-RESCUE": breadth and all(row["confirmation"]["correct"]["cosine"] >= THRESHOLDS["correct_cosine_min"] and row["confirmation"]["correct_accuracy"] >= THRESHOLDS["correct_accuracy_min"] for row in passed),
        "G-WRONG-REJECTION": breadth and all(row["confirmation"]["wrong_identity_accuracy"] >= THRESHOLDS["wrong_identity_accuracy_min"] for row in passed),
        "G-MATCHED-NULL": breadth and all(row["confirmation"]["null_effect_fraction"] <= THRESHOLDS["null_effect_fraction_max"] for row in passed),
        "G-PATH-MEDIATION": breadth and all(row["confirmation"]["block_remaining_fraction"] <= THRESHOLDS["block_remaining_fraction_max"] and row["confirmation"]["rescue_accuracy"] >= THRESHOLDS["rescue_accuracy_min"] for row in passed),
        "G-CONTEXT-PROBE": breadth and all(min(row["confirmation"]["context_probe_correct_retention"], row["confirmation"]["context_probe_null_retention"], row["confirmation"]["context_probe_rescue_retention"]) >= THRESHOLDS["context_probe_retention_min"] for row in passed),
        "G-MANIFOLD": breadth and all(row["confirmation"]["mediator_target_nearest_fraction"] >= THRESHOLDS["mediator_target_nearest_fraction_min"] for row in passed),
        "G-CONTROLS": bool(controls) and bool(applicable_full_state) and min(applicable_full_state) >= THRESHOLDS["full_state_null_leakage_min"] and min(value["random_target_relative_error_min"] for value in controls) >= THRESHOLDS["random_control_target_relative_error_min"],
        "G-BREADTH": breadth,
    }
    return {
        "models": len(rows), "behavior_qualified": len(qualified), "camera_actionable": len(actionable), "passed_models": len(passed),
        "per_depth": per_depth, "selected_pairs": {row["model_key"]: row.get("selected_event_pair") for row in rows},
        "selected_camera_types": {row["model_key"]: row.get("selected_camera_types") for row in rows},
        "gates": gates, "passed_all": all(gates.values()),
    }


def analyze() -> None:
    protocol, _rows = verify_protocol()
    if not COMPLETE.exists():
        raise RuntimeError("formal run incomplete")
    models = read_jsonl(MODELS)
    result = summarize(models)
    passed = result["passed_all"]
    adjudication = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": utc_now(),
        "verdict": "free_transformer_selective_operator_mediation_confirmed" if passed else "free_transformer_selective_operator_mediation_not_confirmed",
        "passed": passed, "summary": result,
        "claim_boundary": "Free same-executor small Transformers on one cyclic-code factorial task; context is diagnostic decodability; no language or pretrained-model mechanism claim.",
        "authorization": {"fresh_natural_template_contract": passed, "qwen3": False, "language_mechanism": False, "new_mathematics": False},
    }
    atomic_json(ANALYSIS, adjudication)
    final = {
        **adjudication,
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL), "environment": file_sha256(ENVIRONMENT), "material": file_sha256(MATERIAL),
            "preaudit": file_sha256(PREAUDIT), "models": file_sha256(MODELS), "summary": file_sha256(SUMMARY),
            "complete": file_sha256(COMPLETE), "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": final["verdict"], "summary": result}))


def run_auditor(mode: str) -> None:
    completed = subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"auditor failed: {mode}")


def probe(device_name: str) -> None:
    device = torch.device(device_name)
    config = ModelConfig(layers=4, width=96, heads=4, mlp_width=192, max_length=23, vocab_size=22)
    rows = make_worlds(seed=1_260_300_001, counts={"discovery": 64, "selection": 64, "confirmation": 128})
    old = MODEL_SEEDS["shallow4_r0"]
    MODEL_SEEDS["shallow4_r0"] = 1_260_301_001
    try:
        result = run_model("shallow4", 0, config, rows, device)
    finally:
        MODEL_SEEDS["shallow4_r0"] = old
    atomic_json(PROBE, result)
    print(canonical_json({"behavior": result["behavior_qualified"], "pair": result.get("selected_event_pair"), "context_probe": result.get("context_probe_selection_accuracy"), "passed": result["passed"], "confirmation": result.get("confirmation")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preregister", "preaudit", "run", "analyze", "audit", "probe", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "preregister":
        preregister(args.force)
    elif args.stage == "preaudit":
        run_auditor("pre")
    elif args.stage == "run":
        run(args.device)
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "audit":
        run_auditor("final")
    elif args.stage == "probe":
        probe(args.device)
    else:
        preregister(args.force)
        run_auditor("pre")
        run(args.device)
        analyze()
        run_auditor("final")


if __name__ == "__main__":
    main()
