#!/usr/bin/env python3
"""Collect signed eventwise five-relation geometry for Phase1098."""

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
from phase1065_multimode_response_atlas_scan import RoleCapture, event_definitions
import phase1097_conditional_transition_scan as shared
import phase1098_relative_relation_geometry_protocol as protocol


shared.protocol = protocol
EPSILON = 1e-12


def load_superunits(model_name: str) -> list[dict[str, Any]]:
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    units: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit_id = str(row["unit_id"])
        units[unit_id][str(row["state"])] = row
        unit_meta[unit_id] = {
            "superunit_id": str(row["superunit_id"]),
            "relation": str(row["relation"]),
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
            raise RuntimeError(f"incomplete unit {unit_id}")
        meta = unit_meta[unit_id]
        grouped[meta["superunit_id"]][meta["relation"]] = {"unit_id": unit_id, "states": states}
        grouped_meta[meta["superunit_id"]] = {key: value for key, value in meta.items() if key != "relation"}
    result = []
    for superunit_id in sorted(grouped):
        if set(grouped[superunit_id]) != set(protocol.RELATIONS):
            raise RuntimeError(f"incomplete relation set {superunit_id}")
        result.append({
            **grouped_meta[superunit_id],
            "superunit_id": superunit_id,
            "relations": grouped[superunit_id],
        })
    return result


def centered_geometry(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Gram, shared energy, differential energy, and centered norms.

    raw has shape [relation, event, field, role, d_model].
    """
    mean = np.mean(raw.astype(np.float64), axis=0)
    centered = raw.astype(np.float64) - mean[None, ...]
    total = np.mean(np.sum(raw.astype(np.float64) ** 2, axis=-1), axis=0)
    shared = np.sum(mean ** 2, axis=-1) / np.maximum(total, EPSILON)
    differential = np.mean(np.sum(centered ** 2, axis=-1), axis=0) / np.maximum(total, EPSILON)
    norms = np.linalg.norm(centered, axis=-1)
    normalized = np.zeros_like(centered)
    valid = np.isfinite(norms) & (norms > EPSILON)
    normalized[valid] = centered[valid] / norms[valid, None]
    gram = np.einsum("refkd,sefkd->efkrs", normalized, normalized, optimize=True)
    pair_valid = np.einsum("refk,sefk->efkrs", valid.astype(np.int8), valid.astype(np.int8), optimize=True) > 0
    gram[~pair_valid] = np.nan
    return (
        gram.astype(np.float32),
        shared.astype(np.float32),
        differential.astype(np.float32),
        np.moveaxis(norms.astype(np.float32), 0, -1),
    )


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1098 protocol audit failed")
    if not authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1098 hidden scan is not authorized")
    superunits = load_superunits(model_name)
    started = time.time()
    model = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        event_keys = [(str(row["component"]), int(row["depth"])) for row in events]
        fields = list(protocol.FIELDS)
        roles = list(protocol.CAPTURE_ROLES)
        field_index = {value: index for index, value in enumerate(fields)}
        role_index = {value: index for index, value in enumerate(roles)}
        state_order = list(protocol.STATES)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        relation_grams: list[np.ndarray] = []
        shared_energies: list[np.ndarray] = []
        differential_energies: list[np.ndarray] = []
        centered_norms: list[np.ndarray] = []
        output_interactions: list[np.ndarray] = []
        index_records: list[dict[str, Any]] = []
        hidden_observations = nonfinite_hidden = 0
        candidate_total = candidate_finite = candidate_hit = 0
        identity_maximum = 0.0
        pre_task_maximum = 0.0
        capture = RoleCapture(model, layers)
        capture.register()
        with torch.inference_mode():
            for superunit_number, superunit in enumerate(superunits):
                relation_vectors: list[np.ndarray] = []
                relation_output_fields: list[np.ndarray] = []
                for relation_number, relation in enumerate(protocol.RELATIONS):
                    unit = superunit["relations"][relation]
                    state_rows = [unit["states"][state] for state in state_order]
                    forward_rows = list(state_rows)
                    identity_index = None
                    if superunit_number == 0 and relation_number == 0:
                        forward_rows.append(state_rows[0])
                        identity_index = len(forward_rows) - 1
                    input_ids, attention_mask, lengths, positions = shared.pad_rows(forward_rows, int(pad_id), device)
                    capture.begin(positions)
                    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                    capture.validate()
                    batch_axis = torch.arange(len(state_rows), dtype=torch.long, device=output.logits.device)
                    last_positions = lengths[:len(state_rows)].to(output.logits.device) - 1
                    logits = output.logits[batch_axis, last_positions, :].float()
                    native_margins = []
                    for slot, row in enumerate(state_rows):
                        e0_id = int(row["candidate_first_token_ids"]["e0"][0])
                        e1_id = int(row["candidate_first_token_ids"]["e1"][0])
                        e0_score = float(logits[slot, e0_id].item())
                        e1_score = float(logits[slot, e1_id].item())
                        native_margin = e0_score - e1_score
                        expected_margin = native_margin if row["expected_class"] == "e0" else -native_margin
                        finite = math.isfinite(native_margin) and math.isfinite(expected_margin)
                        candidate_total += 1
                        candidate_finite += int(finite)
                        candidate_hit += int(finite and expected_margin > 0.0)
                        native_margins.append(native_margin)
                    margin_tensor = torch.tensor(native_margins, dtype=torch.float32, device=device).unsqueeze(-1)
                    margin_fields = shared.field_contrasts(margin_tensor).squeeze(-1).detach().cpu().numpy()
                    relation_output_fields.append(margin_fields.astype(np.float32))

                    event_values: list[np.ndarray] = []
                    for event_number, key in enumerate(event_keys):
                        captured = capture.values[key].float()
                        if identity_index is not None:
                            identity_maximum = max(identity_maximum, float((captured[0] - captured[identity_index]).abs().max().item()))
                        values = captured[:len(state_rows)]
                        contrast = shared.field_contrasts(values)
                        finite = torch.isfinite(contrast).all(dim=-1)
                        hidden_observations += int(finite.numel())
                        nonfinite_hidden += int((~finite).sum().item())
                        for field_name in ("relational_execution", "lookup_execution"):
                            pre_task_maximum = max(
                                pre_task_maximum,
                                float(contrast[field_index[field_name], role_index["branch_probe"]].abs().max().item()),
                            )
                        event_values.append(contrast.detach().float().cpu().numpy().astype(np.float32))
                        del captured, values, contrast
                    relation_vectors.append(np.stack(event_values))
                    del output, logits, input_ids, attention_mask, lengths, positions, margin_tensor
                    capture.values = {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                raw = np.stack(relation_vectors)
                gram, shared_energy, differential_energy, centered_norm = centered_geometry(raw)
                relation_grams.append(gram)
                shared_energies.append(shared_energy)
                differential_energies.append(differential_energy)
                centered_norms.append(centered_norm)
                output_interactions.append(np.stack(relation_output_fields))
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
                del raw, gram, shared_energy, differential_energy, centered_norm
                completed = superunit_number + 1
                if completed % 4 == 0 or completed == len(superunits):
                    print(json.dumps({"phase": protocol.PHASE, "model": model_name, "superunits_complete": completed, "superunits_total": len(superunits)}), flush=True)

        capture.close()
        capture = None
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "relative_relation_geometry.npz",
            relation_gram=np.stack(relation_grams),
            shared_energy=np.stack(shared_energies),
            differential_energy=np.stack(differential_energies),
            centered_norm=np.stack(centered_norms),
            output_interaction=np.stack(output_interactions),
        )
        protocol.write_jsonl(atlas_root / "superunit_index.jsonl", index_records)
        elapsed = time.time() - started
        finite_fraction = 1.0 - nonfinite_hidden / max(hidden_observations, 1)
        summary = {
            "schema_version": "phase1098_model_relation_geometry_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "behavior_formal": bool(authorization["models"][model_name]["model_behavior_passed"]),
            "precision": precision,
            "placement": placement,
            "d_model": int(model.get_input_embeddings().weight.shape[1]),
            "layer_count": len(layers),
            "event_count": len(events),
            "events": events,
            "relations": list(protocol.RELATIONS),
            "surfaces": list(protocol.SURFACES),
            "splits": list(protocol.SPLITS),
            "fields": fields,
            "roles": roles,
            "superunit_count": len(superunits),
            "hidden_finite_fraction": finite_fraction,
            "hidden_observations": hidden_observations,
            "nonfinite_hidden": nonfinite_hidden,
            "identity_maximum_error": identity_maximum,
            "pre_task_maximum_error": pre_task_maximum,
            "candidate_finite_fraction": candidate_finite / max(candidate_total, 1),
            "candidate_accuracy": candidate_hit / max(candidate_finite, 1),
            "candidate_total": candidate_total,
            "elapsed_seconds": elapsed,
            "primary_signature_excludes_output_interaction": True,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({"phase": protocol.PHASE, "model": model_name, "hidden_finite_fraction": finite_fraction, "candidate_accuracy": summary["candidate_accuracy"], "elapsed_seconds": elapsed, "summary_digest": summary["summary_digest"]}), flush=True)
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
