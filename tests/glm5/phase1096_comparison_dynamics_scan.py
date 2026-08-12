#!/usr/bin/env python3
"""Collect Phase1096 full-depth signed representation/control/execution fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
from phase1065_multimode_response_atlas_scan import RoleCapture, event_definitions
import phase1096_comparison_dynamics_protocol as protocol


EPSILON = 1e-12


def pad_rows(rows: list[dict[str, Any]], pad_id: int, device):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)), dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor([
            int(row["role_positions"][role]) for role in protocol.CAPTURE_ROLES
        ], dtype=torch.long, device=device)
    return input_ids, attention_mask, lengths, positions


def projection_seed(model_name: str, d_model: int, replicate: int) -> int:
    material = (
        f"{protocol.SIGNED_PROJECTION_SEED}:{model_name}:"
        f"{d_model}:{replicate}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def projection_matrix(model_name: str, d_model: int, replicate: int, device):
    seed = projection_seed(model_name, d_model, replicate)
    rng = np.random.default_rng(seed)
    values = rng.integers(
        0, 2,
        size=(protocol.SIGNED_PROJECTION_DIM, d_model),
        dtype=np.int8,
    )
    values = (
        values.astype(np.float32) * 2.0 - 1.0
    ) / math.sqrt(protocol.SIGNED_PROJECTION_DIM)
    return torch.tensor(values, dtype=torch.float32, device=device), {
        "replicate": replicate,
        "seed": seed,
        "matrix_digest": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


def field_contrasts(values: torch.Tensor) -> torch.Tensor:
    """Return [field, role, d_model] factorial contrasts for one unit."""
    indices = {
        protocol.state_factors(state): index
        for index, state in enumerate(protocol.STATES)
    }

    def mean(*, panel=None, task=None, orientation=None, order=None):
        selected = []
        for factors, index in indices.items():
            p_value, t_value, o_value, c_value = factors
            if panel is not None and p_value != panel:
                continue
            if task is not None and t_value != task:
                continue
            if orientation is not None and o_value != orientation:
                continue
            if order is not None and c_value != order:
                continue
            selected.append(values[index])
        if not selected:
            raise RuntimeError("empty factorial selection")
        return torch.stack(selected).mean(dim=0)

    fields: dict[str, torch.Tensor] = {}
    for panel, prefix in (("relational", "relational"), ("role_lookup", "lookup")):
        representation = (
            mean(panel=panel, orientation=1)
            - mean(panel=panel, orientation=0)
        )
        control = mean(panel=panel, task="max") - mean(panel=panel, task="min")
        execution = 0.5 * (
            (mean(panel=panel, task="max", orientation=0)
             - mean(panel=panel, task="min", orientation=0))
            - (mean(panel=panel, task="max", orientation=1)
               - mean(panel=panel, task="min", orientation=1))
        )
        carrier = 0.5 * (
            (mean(panel=panel, orientation=1, order=1)
             - mean(panel=panel, orientation=0, order=1))
            - (mean(panel=panel, orientation=1, order=0)
               - mean(panel=panel, orientation=0, order=0))
        )
        fields[f"{prefix}_representation"] = representation
        fields[f"{prefix}_control"] = control
        fields[f"{prefix}_execution"] = execution
        fields[f"{prefix}_carrier"] = carrier
    fields["comparison_control"] = 0.5 * (
        fields["relational_control"] - fields["lookup_control"]
    )
    fields["comparison_execution"] = 0.5 * (
        fields["relational_execution"] - fields["lookup_execution"]
    )
    fields["comparison_carrier"] = 0.5 * (
        fields["relational_carrier"] - fields["lookup_carrier"]
    )
    return torch.stack([fields[name] for name in protocol.SIGNED_FIELDS])


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1096 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1096 hidden scan is not authorized")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "relation": str(row["relation"]),
            "surface": str(row["surface"]),
            "split": str(row["split"]),
            "template": int(row["template"]),
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit {unit_id}")
        units.append({"unit_id": unit_id, **unit_meta[unit_id], "states": grouped[unit_id]})

    started = time.time()
    model = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        event_keys = [(str(row["component"]), int(row["depth"])) for row in events]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        projections = []
        projection_meta = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            matrix, meta = projection_matrix(model_name, d_model, replicate, device)
            projections.append(matrix)
            projection_meta.append(meta)

        relation_index = {value: index for index, value in enumerate(protocol.RELATIONS)}
        surface_index = {value: index for index, value in enumerate(protocol.SURFACES)}
        split_index = {value: index for index, value in enumerate(protocol.SPLITS)}
        role_index = {value: index for index, value in enumerate(protocol.CAPTURE_ROLES)}
        field_index = {value: index for index, value in enumerate(protocol.SIGNED_FIELDS)}
        direction_shape = (
            len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.SPLITS),
            len(events), len(protocol.CAPTURE_ROLES), len(protocol.SIGNED_FIELDS),
            protocol.SIGNED_PROJECTION_REPLICATES, protocol.SIGNED_PROJECTION_DIM,
        )
        direction_sum = np.zeros(direction_shape, dtype=np.float32)
        direction_count = np.zeros(direction_shape[:-1], dtype=np.int32)
        relative_sum = np.zeros(direction_shape[:-2], dtype=np.float64)
        relative_count = np.zeros(direction_shape[:-2], dtype=np.int32)
        template_relative_sum = np.zeros(
            (len(protocol.RELATIONS), len(protocol.SURFACES), len(protocol.TEMPLATES),
             len(events), len(protocol.CAPTURE_ROLES), len(protocol.SIGNED_FIELDS)),
            dtype=np.float64,
        )
        template_relative_count = np.zeros_like(template_relative_sum, dtype=np.int32)
        projection_errors: list[list[float]] = [
            [] for _ in range(protocol.SIGNED_PROJECTION_REPLICATES)
        ]
        hidden_observations = nonfinite_hidden = 0
        candidate_total = candidate_finite = candidate_hit = 0
        identity_maximum = 0.0
        pre_task_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")
        capture = RoleCapture(model, layers)
        capture.register()
        state_order = list(protocol.STATES)
        with torch.inference_mode():
            for unit_number, unit in enumerate(units):
                state_rows = [unit["states"][state] for state in state_order]
                forward_rows = list(state_rows)
                identity_index = None
                if unit_number == 0:
                    forward_rows.append(state_rows[0])
                    identity_index = len(forward_rows) - 1
                input_ids, attention_mask, lengths, positions = pad_rows(
                    forward_rows, int(pad_id), device
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                capture.validate()
                batch_axis = torch.arange(
                    len(state_rows), dtype=torch.long, device=output.logits.device
                )
                last_positions = lengths[:len(state_rows)].to(output.logits.device) - 1
                logits = output.logits[batch_axis, last_positions, :].float()
                for slot, row in enumerate(state_rows):
                    scores = {
                        key: float(logits[slot, ids[0]].item())
                        for key, ids in row["candidate_first_token_ids"].items()
                    }
                    expected = str(row["expected_class"])
                    other = "e1" if expected == "e0" else "e0"
                    margin = scores[expected] - scores[other]
                    finite = all(math.isfinite(value) for value in scores.values()) and math.isfinite(margin)
                    candidate_total += 1
                    candidate_finite += int(finite)
                    candidate_hit += int(finite and margin > 0.0)

                ri = relation_index[unit["relation"]]
                si = surface_index[unit["surface"]]
                qi = split_index[unit["split"]]
                ti = int(unit["template"])
                for event_number, key in enumerate(event_keys):
                    captured = capture.values[key].float()
                    if identity_index is not None:
                        identity_maximum = max(
                            identity_maximum,
                            float((captured[0] - captured[identity_index]).abs().max().item()),
                        )
                    values = captured[:len(state_rows)]
                    fields = field_contrasts(values)
                    state_norm = torch.linalg.vector_norm(values, dim=-1).mean(dim=0)
                    field_norm = torch.linalg.vector_norm(fields, dim=-1)
                    relative = field_norm / torch.clamp(state_norm[None, :], min=EPSILON)
                    finite_vectors = torch.isfinite(fields).all(dim=-1) & (field_norm > EPSILON)
                    hidden_observations += int(finite_vectors.numel())
                    nonfinite_hidden += int((~torch.isfinite(fields).all(dim=-1)).sum().item())
                    normalized_fields = torch.zeros_like(fields)
                    normalized_fields[finite_vectors] = (
                        fields[finite_vectors] / field_norm[finite_vectors, None]
                    )
                    for replicate, matrix in enumerate(projections):
                        projected = torch.einsum("kd,frd->frk", matrix, normalized_fields)
                        projected_np = projected.cpu().numpy()
                        valid_np = finite_vectors.cpu().numpy()
                        for role_number in range(len(protocol.CAPTURE_ROLES)):
                            for field_number in range(len(protocol.SIGNED_FIELDS)):
                                if not valid_np[field_number, role_number]:
                                    continue
                                direction_sum[
                                    ri, si, qi, event_number, role_number,
                                    field_number, replicate,
                                ] += projected_np[field_number, role_number]
                                direction_count[
                                    ri, si, qi, event_number, role_number,
                                    field_number, replicate,
                                ] += 1
                        comparison_field = field_index["comparison_execution"]
                        answer_role = role_index["answer_boundary"]
                        if valid_np[comparison_field, answer_role]:
                            norm = float(np.linalg.norm(
                                projected_np[comparison_field, answer_role]
                            ))
                            if math.isfinite(norm):
                                projection_errors[replicate].append(abs(norm - 1.0))
                    relative_np = relative.cpu().numpy()
                    finite_relative = torch.isfinite(relative).cpu().numpy()
                    for role_number in range(len(protocol.CAPTURE_ROLES)):
                        for field_number in range(len(protocol.SIGNED_FIELDS)):
                            if not finite_relative[field_number, role_number]:
                                continue
                            value = float(relative_np[field_number, role_number])
                            relative_sum[ri, si, qi, event_number, role_number, field_number] += value
                            relative_count[ri, si, qi, event_number, role_number, field_number] += 1
                            template_relative_sum[ri, si, ti, event_number, role_number, field_number] += value
                            template_relative_count[ri, si, ti, event_number, role_number, field_number] += 1
                    for role in protocol.PRE_TASK_ROLES:
                        role_number = role_index[role]
                        for field in (
                            "relational_control", "lookup_control", "comparison_control",
                            "relational_execution", "lookup_execution", "comparison_execution",
                        ):
                            pre_task_maximum = max(
                                pre_task_maximum,
                                float(fields[field_index[field], role_number].abs().max().item()),
                            )
                    del captured, values, fields, normalized_fields, relative
                del output, logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available() and (unit_number + 1) % 8 == 0:
                    torch.cuda.empty_cache()
                completed = unit_number + 1
                if completed % 12 == 0 or completed == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "units_complete": completed,
                        "units_total": len(units),
                    }), flush=True)

        capture.close()
        capture = None
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "three_ledger_fields.npz",
            direction_sum=direction_sum,
            direction_count=direction_count,
            relative_sum=relative_sum,
            relative_count=relative_count,
            template_relative_sum=template_relative_sum,
            template_relative_count=template_relative_count,
        )
        projection_audit = {
            "replicates": [
                {
                    **projection_meta[index],
                    "observation_count": len(errors),
                    "mean_abs_norm_error": float(np.mean(errors)) if errors else None,
                    "median_abs_norm_error": percentile(errors, 50.0),
                    "p95_abs_norm_error": percentile(errors, 95.0),
                    "maximum_abs_norm_error": max(errors) if errors else None,
                }
                for index, errors in enumerate(projection_errors)
            ]
        }
        projection_audit["projection_audit_digest"] = protocol.digest(projection_audit)
        protocol.write_json(atlas_root / "projection_audit.json", projection_audit)
        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1096_model_atlas_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "behavior_formal": bool(
                authorization["models"][model_name]["model_behavior_passed"]
            ),
            "precision": precision,
            "placement": placement,
            "d_model": d_model,
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "relations": list(protocol.RELATIONS),
            "surfaces": list(protocol.SURFACES),
            "splits": list(protocol.SPLITS),
            "roles": list(protocol.CAPTURE_ROLES),
            "fields": list(protocol.SIGNED_FIELDS),
            "direction_axes": [
                "relation", "surface", "split", "event", "role", "field",
                "replicate", "projection",
            ],
            "candidate_count": candidate_total,
            "candidate_finite_fraction": candidate_finite / candidate_total,
            "candidate_accuracy": candidate_hit / candidate_total,
            "hidden_observation_count": hidden_observations,
            "nonfinite_hidden_count": nonfinite_hidden,
            "hidden_finite_fraction_lower_bound": (
                1.0 - nonfinite_hidden / hidden_observations
                if hidden_observations else 0.0
            ),
            "identity_maximum": identity_maximum,
            "pre_task_control_execution_maximum": pre_task_maximum,
            "projection_audit": projection_audit,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print({
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior_formal": summary["behavior_formal"],
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "pre_task_maximum": pre_task_maximum,
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        })
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
