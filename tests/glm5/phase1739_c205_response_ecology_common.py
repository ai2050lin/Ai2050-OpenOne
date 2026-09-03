#!/usr/bin/env python3
"""Shared HiddenState-only utilities for the C205-C215 response-ecology campaign."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1726_c192_multi_program_response_equivalence as c192

DIM = 2560
WIDTH = 96
ROLES = c192.ROLES
DOSES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
UNITS = (1, 2, 5, 6)

C198 = RESULT / "phase1732_c198_broad_natural_program_trajectory"
C204 = RESULT / "phase1738_c204_odd_nonlinear_dose_response"
C205 = RESULT / "phase1739_c205_full_sequence_response_campaign"
C206 = RESULT / "phase1740_c206_full_sequence_response_regimes"
C207 = RESULT / "phase1741_c207_single_joint_coupling_separation"
C208 = RESULT / "phase1742_c208_complete_orthogonal_block_prediction"
C209 = RESULT / "phase1743_c209_omitted_token_closure"
C210 = RESULT / "phase1744_c210_natural_edit_trajectory"
C211 = RESULT / "phase1745_c211_flagship_route_ledger"
C212 = RESULT / "phase1746_c212_true_factorial_composition"
C213 = RESULT / "phase1747_c213_qualified_deletion_rescue"
C214 = RESULT / "phase1748_c214_cross_model_functional_isomorphism"
C215 = RESULT / "phase1749_c215_campaign_synthesis_heatmap"


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def hadamard(size: int) -> np.ndarray:
    value = np.ones((1, 1), np.float32)
    while value.shape[0] < size:
        value = np.block([[value, value], [value, -value]])
    return value


def selected_anchors() -> list[dict]:
    rows = core.rows(C198 / "compiled/qwen3.jsonl")
    selected = [
        row for row in rows
        if row["surface"] == 0 and row["order"] == 1 and row["unit"] in UNITS
    ]
    return sorted(selected, key=lambda row: (row["program"], row["unit"], row["case_id"]))


def source_coordinates() -> list[int]:
    return [int(value) for value in core.load(C198 / "protocol/source_coordinates.json")["coordinates"]]


def behavior_by_case() -> dict[str, dict]:
    return {row["case_id"]: row for row in core.rows(C198 / "raw/behavior_index.jsonl")}


def epsilon_by_case() -> dict[str, float]:
    return {row["case_id"]: float(row["epsilon"]) for row in core.rows(C198 / "raw/hidden_index.jsonl")}


def fixed_batch(rows: list[dict], pad: int, device, width: int = WIDTH):
    return fixed_base.fixed_batch(rows, pad, device, width)


@torch.inference_mode()
def baseline_full(model, rows: list[dict], pad: int, device, width: int = WIDTH):
    """Return embedding/q23/q24/q25 full-token fields and final candidate logits."""
    ids, mask, positions, lengths = fixed_batch(rows, pad, device, width)
    base = model.model
    caught: dict[str, torch.Tensor] = {}
    hooks = [
        base.embed_tokens.register_forward_hook(lambda _m, _a, v: caught.__setitem__("embedding", tensor(v).detach())),
        base.layers[22].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q23", tensor(v).detach())),
        base.layers[23].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q24", tensor(v).detach())),
        base.layers[24].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q25", tensor(v).detach())),
    ]
    try:
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    fields = np.zeros((len(rows), 4, width, DIM), np.float16)
    for state_i, name in enumerate(("embedding", "q23", "q24", "q25")):
        fields[:, state_i] = caught[name].float().cpu().numpy().astype(np.float16)
    logits = np.empty((len(rows), 2), np.float32)
    for local, row in enumerate(rows):
        logits[local] = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
    del output, ids, mask, positions
    return fields, logits, lengths


@torch.inference_mode()
def patched_full(
    model,
    rows: list[dict],
    patterns: np.ndarray,
    doses: np.ndarray,
    epsilons: np.ndarray,
    sign: float,
    pad: int,
    device,
    width: int = WIDTH,
):
    """Patch q23 relation-role coordinates and return q24/q25 full-token fields."""
    if not (len(rows) == len(patterns) == len(doses) == len(epsilons)):
        raise ValueError("batch metadata length mismatch")
    coordinates = source_coordinates()
    ids, mask, positions, lengths = fixed_batch(rows, pad, device, width)
    base = model.model
    caught: dict[str, torch.Tensor] = {}
    actual_write = np.zeros((len(rows), len(coordinates)), np.float32)

    def patch(_module, _args, value):
        state = tensor(value)
        changed = state.clone()
        for local, row in enumerate(rows):
            requested = float(sign * doses[local] * epsilons[local] / np.sqrt(len(coordinates)))
            for source_i, coordinate in enumerate(coordinates):
                delta = float(patterns[local, source_i]) * requested
                positions_for_role = row["role_positions"]["relation"]
                before = changed[local, positions_for_role, coordinate].clone()
                changed[local, positions_for_role, coordinate] += delta
                after = changed[local, positions_for_role, coordinate]
                actual_write[local, source_i] = float((after - before).float().mean().cpu())
        return (changed,) + value[1:] if isinstance(value, tuple) else changed

    hooks = [
        base.layers[22].register_forward_hook(patch),
        base.layers[23].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q24", tensor(v).detach())),
        base.layers[24].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q25", tensor(v).detach())),
    ]
    try:
        model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    fields = np.zeros((len(rows), 2, width, DIM), np.float16)
    fields[:, 0] = caught["q24"].float().cpu().numpy().astype(np.float16)
    fields[:, 1] = caught["q25"].float().cpu().numpy().astype(np.float16)
    del ids, mask, positions
    return fields, actual_write, lengths


def role_means(fields: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Map [case,state,token,dim] fields to role-aligned [case,state,role,dim]."""
    result = np.empty((len(rows), fields.shape[1], len(ROLES), fields.shape[-1]), np.float32)
    for case_i, row in enumerate(rows):
        for role_i, role in enumerate(ROLES):
            role_positions = row["role_positions"][role]
            result[case_i, :, role_i] = np.asarray(
                fields[case_i][:, role_positions, :], dtype=np.float32
            ).mean(axis=1)
    return result


def nrmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error2 = np.square(np.asarray(prediction, np.float32) - np.asarray(truth, np.float32), dtype=np.float64).sum()
    truth2 = np.square(np.asarray(truth, np.float32), dtype=np.float64).sum()
    return float(np.sqrt(error2 / max(truth2, 1e-30)))


def weighted_sign(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    weight = np.minimum(np.abs(left), np.abs(right)).astype(np.float64)
    return float((weight * (np.signbit(left) == np.signbit(right))).sum() / max(weight.sum(), 1e-30))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-30))


def release(model) -> None:
    if model is not None:
        release_bf16(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
