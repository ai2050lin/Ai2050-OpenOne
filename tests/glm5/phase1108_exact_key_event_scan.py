#!/usr/bin/env python3
"""Collect the Phase1108 signed exact-key routing event map."""

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
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
from phase1065_multimode_response_atlas_scan import event_definitions as full_event_definitions
import phase1108_exact_key_event_protocol as protocol


EPSILON = 1e-12
FIELDS = (
    "relation_exact_routing",
    "neutral_exact_routing",
    "relation_ordinal_routing",
    "neutral_ordinal_routing",
    "relation_exact_selector",
    "neutral_exact_selector",
    "relation_ordinal_selector",
    "neutral_ordinal_selector",
    "relation_lexical_address",
    "neutral_lexical_address",
    "relation_selector_address",
    "neutral_selector_address",
)


def sampled_event_definitions(layer_count: int) -> list[dict[str, Any]]:
    events = full_event_definitions(layer_count)
    selected = []
    seen = set()
    for component in protocol.COMPONENTS:
        candidates = [row for row in events if row["component"] == component]
        for fraction in protocol.DEPTH_FRACTIONS:
            row = min(
                candidates,
                key=lambda value: (
                    abs(float(value["relative_depth"]) - fraction),
                    int(value["depth"]),
                ),
            )
            key = (str(row["component"]), int(row["depth"]))
            if key in seen:
                continue
            seen.add(key)
            copied = dict(row)
            copied["event_index"] = len(selected)
            selected.append(copied)
    return selected


class SampledRoleCapture:
    def __init__(self, model, layers, events):
        self.model = model
        self.layers = layers
        self.allowed = {
            (str(row["component"]), int(row["depth"])) for row in events
        }
        self.positions: torch.Tensor | None = None
        self.values: dict[tuple[str, int], torch.Tensor] = {}
        self.counts: Counter = Counter()
        self.handles = []

    def _hook(self, component: str, depth: int):
        key = (component, depth)

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[key] = value[batch, positions, :].detach()
            self.counts[key] += 1
            return output

        return hook

    def register(self) -> None:
        if ("residual", 0) in self.allowed:
            self.handles.append(
                self.model.get_input_embeddings().register_forward_hook(
                    self._hook("residual", 0)
                )
            )
        for depth, layer in enumerate(self.layers, 1):
            if ("residual", depth) in self.allowed:
                self.handles.append(
                    layer.register_forward_hook(self._hook("residual", depth))
                )
            if ("attention_output", depth) in self.allowed:
                self.handles.append(
                    layer.self_attn.register_forward_hook(
                        self._hook("attention_output", depth)
                    )
                )
            if ("mlp_output", depth) in self.allowed:
                self.handles.append(
                    layer.mlp.register_forward_hook(self._hook("mlp_output", depth))
                )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = Counter()

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

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


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
        0,
        2,
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


def pad_rows(rows: list[dict[str, Any]], pad_id: int, device):
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
        positions[index] = torch.tensor(
            [int(row["role_positions"][role]) for role in protocol.CAPTURE_ROLES],
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, lengths, positions


def route_contrasts(
    values: torch.Tensor,
    state_order: list[str],
    regime: str,
    route: str,
) -> dict[str, torch.Tensor]:
    indices = {}
    for index, state in enumerate(state_order):
        factors = protocol.state_factors(state)
        if factors[0] == regime and factors[1] == route:
            indices[factors[2:]] = index

    def mean(*, congruence=None, target=None):
        selected = []
        for (c_value, q_value, _order, _orientation), index in indices.items():
            if congruence is not None and c_value != congruence:
                continue
            if target is not None and q_value != target:
                continue
            selected.append(values[index])
        if not selected:
            raise RuntimeError("empty Phase1108 factorial selection")
        return torch.stack(selected).mean(dim=0)

    conflict = mean(congruence="conflict", target=1) - mean(
        congruence="conflict", target=0
    )
    congruent = mean(congruence="congruent", target=1) - mean(
        congruence="congruent", target=0
    )
    return {
        "routing": 0.5 * (conflict - congruent),
        "selector": 0.5 * (conflict + congruent),
    }


def build_fields(
    values: torch.Tensor, state_order: list[str],
) -> dict[str, torch.Tensor]:
    base = {}
    for regime in protocol.LABEL_REGIMES:
        regime_prefix = "relation" if regime == "relation_label" else "neutral"
        for route in protocol.ROUTE_TYPES:
            contrasts = route_contrasts(values, state_order, regime, route)
            for kind, vector in contrasts.items():
                base[f"{regime_prefix}_{route}_{kind}"] = vector
    base["relation_lexical_address"] = (
        base["relation_exact_routing"] - base["relation_ordinal_routing"]
    )
    base["neutral_lexical_address"] = (
        base["neutral_exact_routing"] - base["neutral_ordinal_routing"]
    )
    base["relation_selector_address"] = (
        base["relation_exact_selector"] - base["relation_ordinal_selector"]
    )
    base["neutral_selector_address"] = (
        base["neutral_exact_selector"] - base["neutral_ordinal_selector"]
    )
    return {field: base[field] for field in FIELDS}


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1108 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1108 hidden scan is not authorized")
    if model_name not in authorization["authorized_models"]:
        raise RuntimeError(f"{model_name} is not behavior-authorized for Phase1108 hidden access")
    authorized_pairs = [
        pair for pair in authorization["cross_model_pairs"]
        if pair in authorization["models"][model_name]["passing_pairs"]
    ]
    rows = [
        row for row in protocol.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
        if row["relation_pair"] in authorized_pairs
    ]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "relation_pair": str(row["relation_pair"]),
            "surface": str(row["surface"]),
            "split": str(row["split"]),
            "template": int(row["template"]),
            "item_index": int(row["item_index"]),
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete Phase1108 unit {unit_id}")
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
        events = sampled_event_definitions(len(layers))
        event_keys = [(str(row["component"]), int(row["depth"])) for row in events]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        projections = []
        projection_meta = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            matrix, meta = projection_matrix(model_name, d_model, replicate, device)
            projections.append(matrix)
            projection_meta.append(meta)

        pair_index = {value: index for index, value in enumerate(authorized_pairs)}
        surface_index = {value: index for index, value in enumerate(protocol.SURFACES)}
        split_index = {value: index for index, value in enumerate(protocol.SPLITS)}
        role_index = {value: index for index, value in enumerate(protocol.CAPTURE_ROLES)}
        field_index = {value: index for index, value in enumerate(FIELDS)}
        shape = (
            len(authorized_pairs),
            len(protocol.SURFACES),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
            len(FIELDS),
            protocol.SIGNED_PROJECTION_REPLICATES,
            protocol.SIGNED_PROJECTION_DIM,
        )
        direction_sum = np.zeros(shape, dtype=np.float32)
        direction_count = np.zeros(shape[:-1], dtype=np.int32)
        relative_sum = np.zeros(shape[:-2], dtype=np.float64)
        relative_count = np.zeros(shape[:-2], dtype=np.int32)
        projection_norm_errors: list[list[float]] = [
            [] for _ in range(protocol.SIGNED_PROJECTION_REPLICATES)
        ]
        hidden_observations = nonfinite_hidden = 0
        candidate_total = candidate_finite = candidate_hit = 0
        identity_maximum = 0.0
        pre_query_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")
        capture = SampledRoleCapture(model, layers, events)
        capture.register()
        state_order = list(protocol.STATES)
        pre_query_roles = [
            role_index["fact1_end"], role_index["facts_end"]
        ]
        with torch.inference_mode():
            for unit_number, unit in enumerate(units):
                state_rows = [unit["states"][state] for state in state_order]
                forward_rows = list(state_rows)
                duplicate_index = None
                if unit_number == 0:
                    forward_rows.append(state_rows[0])
                    duplicate_index = len(forward_rows) - 1
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
                final_positions = lengths[:len(state_rows)].to(output.logits.device) - 1
                logits = output.logits[batch_axis, final_positions, :].float()
                for slot, row in enumerate(state_rows):
                    e0_id = int(row["candidate_first_token_ids"]["e0"][0])
                    e1_id = int(row["candidate_first_token_ids"]["e1"][0])
                    scores = {
                        "e0": float(logits[slot, e0_id].item()),
                        "e1": float(logits[slot, e1_id].item()),
                    }
                    expected = str(row["expected_class"])
                    other = "e1" if expected == "e0" else "e0"
                    margin = scores[expected] - scores[other]
                    finite = all(math.isfinite(value) for value in scores.values()) and math.isfinite(margin)
                    candidate_total += 1
                    candidate_finite += int(finite)
                    candidate_hit += int(finite and margin > 0.0)

                pi = pair_index[unit["relation_pair"]]
                si = surface_index[unit["surface"]]
                xi = split_index[unit["split"]]
                for ei, event_key in enumerate(event_keys):
                    captured = capture.values[event_key].float()
                    if duplicate_index is not None:
                        identity_delta = captured[duplicate_index] - captured[0]
                        if torch.isfinite(identity_delta).all():
                            identity_maximum = max(
                                identity_maximum,
                                float(identity_delta.abs().max().item()),
                            )
                    values = captured[:len(state_rows)]
                    fields = build_fields(values, state_order)
                    baseline = torch.linalg.vector_norm(values, dim=-1).mean(dim=0)
                    for field, vector in fields.items():
                        fj = field_index[field]
                        norms = torch.linalg.vector_norm(vector, dim=-1)
                        finite = torch.isfinite(vector).all(dim=-1) & torch.isfinite(norms)
                        hidden_observations += int(finite.numel())
                        nonfinite_hidden += int((~finite).sum().item())
                        for pre_role in pre_query_roles:
                            if finite[pre_role]:
                                pre_query_maximum = max(
                                    pre_query_maximum,
                                    float(vector[pre_role].abs().max().item()),
                                )
                        relative = norms / torch.clamp(baseline, min=EPSILON)
                        for ri in range(len(protocol.CAPTURE_ROLES)):
                            if not bool(finite[ri].item()) or float(norms[ri].item()) <= EPSILON:
                                continue
                            relative_sum[pi, si, xi, ei, ri, fj] += float(relative[ri].item())
                            relative_count[pi, si, xi, ei, ri, fj] += 1
                            for replicate, matrix in enumerate(projections):
                                projected = torch.matmul(vector[ri], matrix.T)
                                projected_norm = torch.linalg.vector_norm(projected)
                                if not torch.isfinite(projected_norm) or float(projected_norm.item()) <= EPSILON:
                                    continue
                                direction = projected / projected_norm
                                direction_sum[pi, si, xi, ei, ri, fj, replicate] += (
                                    direction.detach().cpu().numpy().astype(np.float32)
                                )
                                direction_count[pi, si, xi, ei, ri, fj, replicate] += 1
                                projection_norm_errors[replicate].append(
                                    abs(float(projected_norm.item()) / float(norms[ri].item()) - 1.0)
                                )
                    del captured, values, fields

                del output, logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = unit_number + 1
                if completed % 6 == 0 or completed == len(units):
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
            atlas_root / "signed_event_fields.npz",
            direction_sum=direction_sum,
            direction_count=direction_count,
            relative_sum=relative_sum,
            relative_count=relative_count,
        )
        projection_audit = {
            "replicates": [
                {
                    **projection_meta[replicate],
                    "observation_count": len(errors),
                    "median_relative_norm_error": percentile(errors, 50.0),
                    "p95_relative_norm_error": percentile(errors, 95.0),
                    "maximum_relative_norm_error": max(errors) if errors else None,
                }
                for replicate, errors in enumerate(projection_norm_errors)
            ]
        }
        projection_audit["projection_audit_digest"] = protocol.digest(projection_audit)
        protocol.write_json(atlas_root / "projection_audit.json", projection_audit)
        elapsed = time.time() - started
        hidden_finite_fraction = 1.0 - nonfinite_hidden / max(hidden_observations, 1)
        summary = {
            "schema_version": "phase1108_model_signed_event_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "behavior_authorization_digest": authorization["authorization_digest"],
            "precision": precision,
            "placement": placement,
            "d_model": d_model,
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "roles": list(protocol.CAPTURE_ROLES),
            "fields": list(FIELDS),
            "relation_pairs": authorized_pairs,
            "surfaces": list(protocol.SURFACES),
            "splits": list(protocol.SPLITS),
            "direction_axes": [
                "pair", "surface", "split", "event", "role", "field",
                "replicate", "projection",
            ],
            "unit_count": len(units),
            "candidate_total": candidate_total,
            "candidate_finite_fraction": candidate_finite / max(candidate_total, 1),
            "candidate_accuracy": candidate_hit / max(candidate_finite, 1),
            "hidden_observations": hidden_observations,
            "nonfinite_hidden": nonfinite_hidden,
            "hidden_finite_fraction": hidden_finite_fraction,
            "identity_maximum_error": identity_maximum,
            "pre_query_maximum_error": pre_query_maximum,
            "projection_audit": projection_audit,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "event_count": len(events),
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "hidden_finite_fraction": hidden_finite_fraction,
            "pre_query_maximum_error": pre_query_maximum,
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        }), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if "projections" in locals():
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
