#!/usr/bin/env python3
"""Collect exact signed relation-pair routing geometry for Phase1101."""

from __future__ import annotations

import argparse
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
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
from phase1065_multimode_response_atlas_scan import RoleCapture, event_definitions as full_event_definitions
import phase1101_relation_identity_routing_protocol as protocol


EPSILON = 1e-12


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


def load_superunits(model_name: str) -> list[dict[str, Any]]:
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    units: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit_id = str(row["unit_id"])
        units[unit_id][str(row["state"])] = row
        unit_meta[unit_id] = {
            "superunit_id": str(row["superunit_id"]),
            "relation_pair": str(row["relation_pair"]),
            "surface": str(row["surface"]),
            "split": str(row["split"]),
            "template": int(row["template"]),
            "item_index": int(row["item_index"]),
            "entity0": str(row["entity0"]),
            "entity1": str(row["entity1"]),
        }
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    grouped_meta: dict[str, dict[str, Any]] = {}
    for unit_id, states in units.items():
        if set(states) != set(protocol.STATES):
            raise RuntimeError(f"incomplete Phase1101 unit {unit_id}")
        meta = unit_meta[unit_id]
        grouped[meta["superunit_id"]][meta["relation_pair"]] = {
            "unit_id": unit_id,
            "states": states,
        }
        grouped_meta[meta["superunit_id"]] = {
            key: value for key, value in meta.items() if key != "relation_pair"
        }
    result = []
    for superunit_id in sorted(grouped):
        if set(grouped[superunit_id]) != set(protocol.RELATION_PAIRS):
            raise RuntimeError(f"incomplete Phase1101 pair set {superunit_id}")
        result.append({
            **grouped_meta[superunit_id],
            "superunit_id": superunit_id,
            "pairs": grouped[superunit_id],
        })
    return result


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


def route_contrasts(values: torch.Tensor, route_type: str) -> dict[str, torch.Tensor]:
    states = [state for state in protocol.STATES if protocol.state_factors(state)[0] == route_type]
    indices = {
        protocol.state_factors(state)[1:]: index for index, state in enumerate(states)
    }

    def mean(*, congruence=None, target=None, order=None, orientation=None):
        selected = []
        for factors, index in indices.items():
            c_value, q_value, o_value, b_value = factors
            if congruence is not None and c_value != congruence:
                continue
            if target is not None and q_value != target:
                continue
            if order is not None and o_value != order:
                continue
            if orientation is not None and b_value != orientation:
                continue
            selected.append(values[index])
        if not selected:
            raise RuntimeError("empty Phase1101 factorial selection")
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


def centered_pair_geometry(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert [pair,event,field,role,d] vectors into exact centered Gram geometry."""
    values = raw.astype(np.float64)
    mean = np.mean(values, axis=0)
    centered = values - mean[None, ...]
    total = np.mean(np.sum(values ** 2, axis=-1), axis=0)
    shared = np.sum(mean ** 2, axis=-1) / np.maximum(total, EPSILON)
    differential = np.mean(np.sum(centered ** 2, axis=-1), axis=0) / np.maximum(total, EPSILON)
    norms = np.linalg.norm(centered, axis=-1)
    normalized = np.zeros_like(centered)
    valid = np.isfinite(norms) & (norms > EPSILON)
    normalized[valid] = centered[valid] / norms[valid, None]
    gram = np.einsum("pefrd,qefrd->efrpq", normalized, normalized, optimize=True)
    pair_valid = np.einsum(
        "pefr,qefr->efrpq", valid.astype(np.int8), valid.astype(np.int8), optimize=True
    ) > 0
    gram[~pair_valid] = np.nan
    return (
        gram.astype(np.float32),
        shared.astype(np.float32),
        differential.astype(np.float32),
        np.moveaxis(norms.astype(np.float32), 0, -1),
    )


def centered_scalar_gram(raw: np.ndarray) -> np.ndarray:
    """Return [field,pair,pair] Gram for pairwise scalar output interactions."""
    values = raw.astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    # A scalar cannot carry a nontrivial within-cell 15-pair direction; this
    # signed outer product is saved only as an output-identity audit.
    norms = np.abs(centered)
    normalized = np.zeros_like(centered)
    valid = np.isfinite(norms) & (norms > EPSILON)
    normalized[valid] = centered[valid] / norms[valid]
    gram = np.einsum("pf,qf->fpq", normalized, normalized, optimize=True)
    pair_valid = np.einsum("pf,qf->fpq", valid.astype(np.int8), valid.astype(np.int8), optimize=True) > 0
    gram[~pair_valid] = np.nan
    return gram.astype(np.float32)


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1101 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1101 hidden scan is not authorized")
    superunits = load_superunits(model_name)
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
        role_index = {role: index for index, role in enumerate(protocol.CAPTURE_ROLES)}
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        pair_grams = []
        shared_energies = []
        differential_energies = []
        centered_norms = []
        output_grams = []
        index_records = []
        hidden_observations = nonfinite_hidden = 0
        candidate_total = candidate_finite = candidate_hit = 0
        identity_maximum = 0.0
        pre_query_maximum = 0.0
        capture = RoleCapture(model, layers)
        capture.register()
        with torch.inference_mode():
            for superunit_number, superunit in enumerate(superunits):
                all_pair_vectors = []
                all_pair_output = []
                for pair_number, relation_pair in enumerate(protocol.RELATION_PAIRS):
                    unit = superunit["pairs"][relation_pair]
                    route_hidden: dict[str, list[np.ndarray]] = {}
                    route_output: dict[str, dict[str, float]] = {}
                    for route_type in protocol.ROUTE_TYPES:
                        states = [
                            state for state in protocol.STATES
                            if protocol.state_factors(state)[0] == route_type
                        ]
                        state_rows = [unit["states"][state] for state in states]
                        forward_rows = list(state_rows)
                        duplicate_index = None
                        if superunit_number == 0 and pair_number == 0 and route_type == "semantic":
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
                        last_positions = lengths[:len(state_rows)].to(output.logits.device) - 1
                        logits = output.logits[batch_axis, last_positions, :].float()
                        native_margins = []
                        for slot, row in enumerate(state_rows):
                            e0_id = int(row["candidate_first_token_ids"]["e0"][0])
                            e1_id = int(row["candidate_first_token_ids"]["e1"][0])
                            e0_score = float(logits[slot, e0_id].item())
                            e1_score = float(logits[slot, e1_id].item())
                            native_margin = e0_score - e1_score
                            expected_margin = (
                                native_margin if row["expected_class"] == "e0" else -native_margin
                            )
                            finite = math.isfinite(native_margin) and math.isfinite(expected_margin)
                            candidate_total += 1
                            candidate_finite += int(finite)
                            candidate_hit += int(finite and expected_margin > 0.0)
                            native_margins.append(native_margin)
                        output_values = torch.tensor(
                            native_margins, dtype=torch.float32, device=device
                        ).unsqueeze(-1)
                        output_fields = route_contrasts(output_values, route_type)
                        route_output[route_type] = {
                            key: float(value.squeeze().item()) for key, value in output_fields.items()
                        }

                        event_values = []
                        for event_number, key in enumerate(event_keys):
                            captured = capture.values[key].float()
                            if duplicate_index is not None:
                                identity_maximum = max(
                                    identity_maximum,
                                    float((captured[0] - captured[duplicate_index]).abs().max().item()),
                                )
                            values = captured[:len(state_rows)]
                            fields = route_contrasts(values, route_type)
                            stacked = torch.stack((fields["routing"], fields["selector"]))
                            finite = torch.isfinite(stacked).all(dim=-1)
                            hidden_observations += int(finite.numel())
                            nonfinite_hidden += int((~finite).sum().item())
                            pre_query_maximum = max(
                                pre_query_maximum,
                                float(stacked[:, role_index["facts_end"]].abs().max().item()),
                            )
                            event_values.append(
                                stacked.detach().cpu().numpy().astype(np.float32)
                            )
                            del captured, values, fields, stacked
                        route_hidden[route_type] = event_values
                        del output, logits, output_values, input_ids, attention_mask, lengths, positions
                        capture.values = {}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    pair_events = []
                    for event_number in range(len(events)):
                        semantic = route_hidden["semantic"][event_number]
                        ordinal = route_hidden["ordinal"][event_number]
                        pair_events.append(np.stack((
                            semantic[0], ordinal[0], semantic[1], ordinal[1]
                        )))
                    all_pair_vectors.append(np.stack(pair_events))
                    all_pair_output.append(np.asarray((
                        route_output["semantic"]["routing"],
                        route_output["ordinal"]["routing"],
                        route_output["semantic"]["selector"],
                        route_output["ordinal"]["selector"],
                    ), dtype=np.float32))

                raw = np.stack(all_pair_vectors)
                gram, shared, differential, norms = centered_pair_geometry(raw)
                pair_grams.append(gram)
                shared_energies.append(shared)
                differential_energies.append(differential)
                centered_norms.append(norms)
                output_grams.append(centered_scalar_gram(np.stack(all_pair_output)))
                index_records.append({
                    "superunit_index": superunit_number,
                    "superunit_id": superunit["superunit_id"],
                    "surface": superunit["surface"],
                    "split": superunit["split"],
                    "template": superunit["template"],
                    "item_index": superunit["item_index"],
                    "entity0": superunit["entity0"],
                    "entity1": superunit["entity1"],
                })
                del raw, gram, shared, differential, norms, all_pair_vectors
                completed = superunit_number + 1
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "superunits_complete": completed,
                    "superunits_total": len(superunits),
                }), flush=True)

        capture.close()
        capture = None
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "relation_identity_routing_geometry.npz",
            pair_gram=np.stack(pair_grams),
            shared_energy=np.stack(shared_energies),
            differential_energy=np.stack(differential_energies),
            centered_norm=np.stack(centered_norms),
            output_gram=np.stack(output_grams),
        )
        protocol.write_jsonl(atlas_root / "superunit_index.jsonl", index_records)
        elapsed = time.time() - started
        finite_fraction = 1.0 - nonfinite_hidden / max(hidden_observations, 1)
        summary = {
            "schema_version": "phase1101_model_routing_atlas_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "behavior_formal": bool(
                authorization["models"][model_name]["model_behavior_passed"]
            ),
            "precision": precision,
            "placement": placement,
            "d_model": int(model.get_input_embeddings().weight.shape[1]),
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "relation_pairs": list(protocol.RELATION_PAIRS),
            "surfaces": list(protocol.SURFACES),
            "splits": list(protocol.SPLITS),
            "fields": list(protocol.FIELDS),
            "roles": list(protocol.CAPTURE_ROLES),
            "superunit_count": len(superunits),
            "hidden_finite_fraction": finite_fraction,
            "hidden_observations": hidden_observations,
            "nonfinite_hidden": nonfinite_hidden,
            "identity_maximum_error": identity_maximum,
            "pre_query_maximum_error": pre_query_maximum,
            "candidate_finite_fraction": candidate_finite / max(candidate_total, 1),
            "candidate_accuracy": candidate_hit / max(candidate_finite, 1),
            "candidate_total": candidate_total,
            "elapsed_seconds": elapsed,
            "primary_signature_excludes_output_gram": True,
            "exact_full_d_model_gram": True,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "hidden_finite_fraction": finite_fraction,
            "pre_query_maximum_error": pre_query_maximum,
            "candidate_accuracy": summary["candidate_accuracy"],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
        }), flush=True)
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
