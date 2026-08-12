#!/usr/bin/env python3
"""Collect Phase1086 middle-band signed fields with two frozen sketches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
from phase1065_multimode_response_atlas_scan import (
    RoleCapture as FullRoleCapture,
    event_definitions as full_event_definitions,
)
from phase1079_output_orthogonal_pattern_scan import delta_stats
import phase1086_signed_shared_field_protocol as protocol


EPSILON = 1e-12
UNIT_BATCH_SIZE = 1


def targeted_event_definitions(n_layers: int) -> list[dict[str, Any]]:
    rows = [
        dict(row) for row in full_event_definitions(n_layers)
        if (
            protocol.TARGET_RELATIVE_DEPTH_MIN
            <= float(row["relative_depth"])
            <= protocol.TARGET_RELATIVE_DEPTH_MAX
        )
    ]
    for index, row in enumerate(rows):
        row["event_index"] = index
    return rows


class MiddleBandRoleCapture(FullRoleCapture):
    def __init__(self, model, layers):
        super().__init__(model, layers)
        self.allowed = {
            (str(row["component"]), int(row["depth"]))
            for row in targeted_event_definitions(len(layers))
        }

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, 1):
            if ("residual", depth) in self.allowed:
                self.handles.append(layer.register_forward_hook(
                    self._hook("residual", depth)
                ))
            if ("attention_output", depth) in self.allowed:
                self.handles.append(layer.self_attn.register_forward_hook(
                    self._hook("attention_output", depth)
                ))
            if ("mlp_output", depth) in self.allowed:
                self.handles.append(layer.mlp.register_forward_hook(
                    self._hook("mlp_output", depth)
                ))

    def validate(self) -> None:
        missing = self.allowed - set(self.values)
        unexpected = set(self.values) - self.allowed
        repeated = {
            str(key): count for key, count in self.counts.items() if count != 1
        }
        if missing or unexpected or repeated:
            raise RuntimeError(
                f"capture drift missing={list(missing)[:5]} "
                f"unexpected={list(unexpected)[:5]} repeated={repeated}"
            )


def pad_rows(
    rows: list[dict[str, Any]], pad_id: int, device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor([
            int(row["role_positions"][role])
            for role in protocol.CAPTURE_ROLES
        ], dtype=torch.long, device=device)
    return input_ids, attention_mask, lengths, positions


def projection_seed(model_name: str, d_model: int, replicate: int) -> int:
    material = (
        f"{protocol.SIGNED_PROJECTION_SEED}:{model_name}:"
        f"{d_model}:{replicate}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def projection_matrix(
    model_name: str, d_model: int, replicate: int, device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    seed = projection_seed(model_name, d_model, replicate)
    rng = np.random.default_rng(seed)
    values = rng.integers(
        0, 2,
        size=(protocol.SIGNED_PROJECTION_DIM, d_model),
        dtype=np.int8,
    )
    values = (values.astype(np.float32) * 2.0 - 1.0) / math.sqrt(
        protocol.SIGNED_PROJECTION_DIM
    )
    digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        {"replicate": replicate, "seed": seed, "matrix_digest": digest},
    )


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1086 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1086 hidden scan is not authorized")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "family": str(row["family"]), "split": str(row["split"])
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit: {unit_id}")
        units.append({
            "unit_id": unit_id,
            **unit_meta[unit_id],
            "states": grouped[unit_id],
        })

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
        events = targeted_event_definitions(len(layers))
        event_keys = [
            (str(row["component"]), int(row["depth"])) for row in events
        ]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        projections = []
        projection_meta = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            matrix, meta = projection_matrix(
                model_name, d_model, replicate, device
            )
            projections.append(matrix)
            projection_meta.append(meta)

        template_ids = tuple(getattr(protocol, "TEMPLATE_IDS", (0, 1)))
        output_set_ids = tuple(getattr(protocol, "OUTPUT_SET_IDS", (0, 1)))
        template_axis = {
            value: index for index, value in enumerate(template_ids)
        }
        output_axis = {
            value: index for index, value in enumerate(output_set_ids)
        }
        family_index = {value: index for index, value in enumerate(protocol.FAMILIES)}
        split_index = {value: index for index, value in enumerate(protocol.SPLITS)}
        role_index = {value: index for index, value in enumerate(protocol.CAPTURE_ROLES)}
        field_index = {value: index for index, value in enumerate(protocol.SIGNED_FIELDS)}
        direction_shape = (
            len(protocol.FAMILIES), len(protocol.SPLITS), len(events),
            len(protocol.CAPTURE_ROLES), len(protocol.SIGNED_FIELDS),
            len(template_ids), len(output_set_ids),
            protocol.SIGNED_PROJECTION_REPLICATES,
            protocol.SIGNED_PROJECTION_DIM,
        )
        count_shape = direction_shape[:-1]
        direction_sum = np.zeros(direction_shape, dtype=np.float32)
        direction_count = np.zeros(count_shape, dtype=np.int32)
        relative_sum = np.zeros(count_shape[:-1], dtype=np.float64)
        relative_count = np.zeros(count_shape[:-1], dtype=np.int32)
        surface_relative_sum = np.zeros(
            (len(protocol.FAMILIES), len(protocol.SPLITS), len(events), len(protocol.CAPTURE_ROLES)),
            dtype=np.float64,
        )
        surface_relative_count = np.zeros_like(surface_relative_sum, dtype=np.int32)
        output_relative_sum = np.zeros_like(surface_relative_sum)
        output_relative_count = np.zeros_like(surface_relative_sum, dtype=np.int32)
        projection_errors: list[list[float]] = [
            [] for _ in range(protocol.SIGNED_PROJECTION_REPLICATES)
        ]
        candidate_total = candidate_finite = candidate_hit = 0
        nonfinite_hidden_count = 0
        hidden_observation_count = 0
        pre_query_max_abs = 0.0
        identity_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        capture = MiddleBandRoleCapture(model, layers)
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
                logits = output.logits
                final_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(logits.shape[0], device=logits.device)
                final_logits = logits[batch_axis, final_positions, :].float()
                del output, logits

                for local, state_name in enumerate(state_order):
                    row = unit["states"][state_name]
                    if row["panel"] != "active":
                        continue
                    values = final_logits[local]
                    scores = {}
                    for answer_class in ("a0", "a1"):
                        ids = torch.tensor(
                            row["candidate_first_token_ids"][answer_class],
                            dtype=torch.long,
                            device=values.device,
                        )
                        scores[answer_class] = float(values[ids].max().item())
                    expected = str(row["expected_class"])
                    other = "a1" if expected == "a0" else "a0"
                    finite = all(math.isfinite(value) for value in scores.values())
                    margin = scores[expected] - scores[other]
                    finite = finite and math.isfinite(margin)
                    candidate_total += 1
                    candidate_finite += int(finite)
                    candidate_hit += int(finite and margin > 0.0)

                fi = family_index[unit["family"]]
                si = split_index[unit["split"]]

                def state_tensor(
                    values: torch.Tensor,
                    template: int,
                    panel: str,
                    mapping: int,
                    query: int,
                    output_set: int,
                ) -> torch.Tensor:
                    name = (
                        f"t{template}_c{panel}_m{mapping}_q{query}_w{output_set}"
                    )
                    return values[state_order.index(name)]

                for ei, event_key in enumerate(event_keys):
                    values = capture.values[event_key].float()
                    if identity_index is not None:
                        identity_delta = values[identity_index] - values[0]
                        if torch.isfinite(identity_delta).all():
                            identity_maximum = max(
                                identity_maximum,
                                float(identity_delta.abs().max().item()),
                            )

                    pair_left = []
                    pair_right = []
                    pair_meta = []
                    for template in template_ids:
                        for output_set in output_set_ids:
                            custom_builder = getattr(
                                protocol, "signed_pair_records", None
                            )
                            if custom_builder is not None:
                                records = custom_builder(
                                    state_tensor, values, template, output_set
                                )
                                for field, left, right, subindex in records:
                                    pair_left.append(left)
                                    pair_right.append(right)
                                    pair_meta.append((
                                        field, template, output_set, subindex
                                    ))
                            else:
                                for mapping in (0, 1):
                                    true_query = mapping
                                    false_query = 1 - mapping
                                    active_true = state_tensor(
                                        values, template, "active", mapping,
                                        true_query, output_set,
                                    )
                                    active_false = state_tensor(
                                        values, template, "active", mapping,
                                        false_query, output_set,
                                    )
                                    null_true_position = state_tensor(
                                        values, template, "field_null", mapping,
                                        true_query, output_set,
                                    )
                                    null_false_position = state_tensor(
                                        values, template, "field_null", mapping,
                                        false_query, output_set,
                                    )
                                    field_pairs = {
                                        "active_truth": (active_false, active_true),
                                        "field_null": (
                                            null_false_position, null_true_position
                                        ),
                                        "content": (
                                            0.5 * (active_false + null_true_position),
                                            0.5 * (active_true + null_false_position),
                                        ),
                                    }
                                    for field, (left, right) in field_pairs.items():
                                        pair_left.append(left)
                                        pair_right.append(right)
                                        pair_meta.append((
                                            field, template, output_set, mapping
                                        ))

                    left = torch.stack(pair_left)
                    right = torch.stack(pair_right)
                    relative, magnitude_valid, direction, direction_valid = delta_stats(
                        right - left, left, right
                    )
                    projected = torch.stack([
                        torch.matmul(direction, matrix.T)
                        for matrix in projections
                    ])
                    relative_np = relative.cpu().numpy()
                    magnitude_valid_np = magnitude_valid.cpu().numpy()
                    direction_valid_np = direction_valid.cpu().numpy()
                    projected_np = projected.cpu().numpy()
                    for observation, (field, template, output_set, _mapping) in enumerate(pair_meta):
                        fj = field_index[field]
                        template_index = template_axis[template]
                        output_index = output_axis[output_set]
                        for ri in range(len(protocol.CAPTURE_ROLES)):
                            hidden_observation_count += 1
                            if magnitude_valid_np[observation, ri]:
                                relative_sum[
                                    fi, si, ei, ri, fj,
                                    template_index, output_index,
                                ] += float(
                                    relative_np[observation, ri]
                                )
                                relative_count[
                                    fi, si, ei, ri, fj,
                                    template_index, output_index,
                                ] += 1
                            else:
                                nonfinite_hidden_count += 1
                            if not direction_valid_np[observation, ri]:
                                continue
                            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                                direction_sum[
                                    fi, si, ei, ri, fj,
                                    template_index, output_index,
                                    replicate,
                                ] += projected_np[replicate, observation, ri]
                                direction_count[
                                    fi, si, ei, ri, fj,
                                    template_index, output_index,
                                    replicate,
                                ] += 1
                                if (
                                    field == "content"
                                    and ri == role_index["answer_boundary"]
                                ):
                                    norm = float(np.linalg.norm(
                                        projected_np[replicate, observation, ri]
                                    ))
                                    if math.isfinite(norm):
                                        projection_errors[replicate].append(abs(norm - 1.0))

                    # Surface and output controls remain relative magnitudes;
                    # their signed directions are not interpreted as content.
                    surface_pairs = []
                    output_pairs = []
                    for panel in protocol.PANELS:
                        for mapping in (0, 1):
                            for query in (0, 1):
                                for output_set in output_set_ids:
                                    if len(template_ids) < 2:
                                        continue
                                    surface_pairs.append((
                                        state_tensor(values, template_ids[0], panel, mapping, query, output_set),
                                        state_tensor(values, template_ids[1], panel, mapping, query, output_set),
                                    ))
                                for template in template_ids:
                                    if len(output_set_ids) < 2:
                                        continue
                                    output_pairs.append((
                                        state_tensor(values, template, panel, mapping, query, output_set_ids[0]),
                                        state_tensor(values, template, panel, mapping, query, output_set_ids[1]),
                                    ))
                    for pairs, sums, counts in (
                        (surface_pairs, surface_relative_sum, surface_relative_count),
                        (output_pairs, output_relative_sum, output_relative_count),
                    ):
                        if not pairs:
                            continue
                        control_left = torch.stack([pair[0] for pair in pairs])
                        control_right = torch.stack([pair[1] for pair in pairs])
                        control_relative, valid, _, _ = delta_stats(
                            control_right - control_left,
                            control_left,
                            control_right,
                        )
                        sums[fi, si, ei] += control_relative.sum(dim=0).cpu().numpy()
                        counts[fi, si, ei] += valid.sum(dim=0).cpu().numpy().astype(np.int32)

                    dossier_role = role_index["dossier_end"]
                    content_rows = [
                        index for index, meta in enumerate(pair_meta)
                        if meta[0] == "content"
                    ]
                    if content_rows:
                        pre_values = (right - left)[content_rows, dossier_role]
                        finite = torch.isfinite(pre_values)
                        maximum = torch.where(
                            finite, pre_values.abs(), torch.zeros_like(pre_values)
                        ).max()
                        pre_query_max_abs = max(
                            pre_query_max_abs, float(maximum.item())
                        )
                    del values, left, right, direction, projected

                del final_logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available():
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
            atlas_root / "signed_fields.npz",
            direction_sum=direction_sum,
            direction_count=direction_count,
            relative_sum=relative_sum,
            relative_count=relative_count,
            surface_relative_sum=surface_relative_sum,
            surface_relative_count=surface_relative_count,
            output_relative_sum=output_relative_sum,
            output_relative_count=output_relative_count,
        )
        projection_audit = {
            "replicates": [
                {
                    **projection_meta[replicate],
                    "observation_count": len(errors),
                    "mean_abs_norm_error": float(np.mean(errors)) if errors else None,
                    "median_abs_norm_error": percentile(errors, 50.0),
                    "p95_abs_norm_error": percentile(errors, 95.0),
                    "maximum_abs_norm_error": max(errors) if errors else None,
                }
                for replicate, errors in enumerate(projection_errors)
            ]
        }
        projection_audit["projection_audit_digest"] = protocol.digest(projection_audit)
        protocol.write_json(atlas_root / "projection_audit.json", projection_audit)
        elapsed = time.time() - started
        hidden_finite_fraction = (
            1.0 - nonfinite_hidden_count / hidden_observation_count
            if hidden_observation_count else 0.0
        )
        summary = {
            "schema_version": "phase1086_model_signed_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "d_model": d_model,
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "roles": list(protocol.CAPTURE_ROLES),
            "fields": list(protocol.SIGNED_FIELDS),
            "families": list(protocol.FAMILIES),
            "splits": list(protocol.SPLITS),
            "direction_axes": [
                "family", "split", "event", "role", "field",
                "template", "output_set", "replicate", "projection",
            ],
            "template_ids": list(template_ids),
            "output_set_ids": list(output_set_ids),
            "candidate_count": candidate_total,
            "candidate_finite_count": candidate_finite,
            "candidate_hit_count": candidate_hit,
            "candidate_finite_fraction": (
                candidate_finite / candidate_total if candidate_total else 0.0
            ),
            "candidate_accuracy": (
                candidate_hit / candidate_total if candidate_total else 0.0
            ),
            "hidden_observation_count": hidden_observation_count,
            "nonfinite_hidden_count": nonfinite_hidden_count,
            "hidden_finite_fraction_lower_bound": hidden_finite_fraction,
            "pre_query_global_max_abs": pre_query_max_abs,
            "identity_maximum": identity_maximum,
            "projection_audit": projection_audit,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print({
            "phase": protocol.PHASE,
            "model": model_name,
            "event_count": len(events),
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": hidden_finite_fraction,
            "pre_query_global_max_abs": pre_query_max_abs,
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        })
    finally:
        if capture is not None:
            capture.close()
        for name in ("projections",):
            if name in locals():
                del projections
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
