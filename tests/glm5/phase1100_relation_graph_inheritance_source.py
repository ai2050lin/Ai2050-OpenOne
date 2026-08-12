#!/usr/bin/env python3
"""Extract signed input/output lexical query geometry under audited FP16 loading."""

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
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1099_relation_family_atlas_protocol as source_protocol
import phase1100_relation_graph_inheritance_protocol as protocol


def query_sequences(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[tuple[int, ...]]]:
    grouped: dict[tuple[str, str, str], set[tuple[int, ...]]] = defaultdict(set)
    for row in rows:
        start, end = (int(value) for value in row["role_spans"]["query_end"])
        ids = tuple(int(value) for value in row["input_ids"][start:end + 1])
        if not ids:
            raise RuntimeError(f"empty query span for {row['record_id']}")
        grouped[(str(row["surface"]), str(row["relation"]), str(row["task"]))].add(ids)
    result: dict[tuple[str, str, str], list[tuple[int, ...]]] = {}
    for key, values in grouped.items():
        result[key] = sorted(values)
    return result


def mean_sequence_vector(weight: torch.Tensor, sequences: list[tuple[int, ...]]) -> torch.Tensor:
    vectors = []
    for sequence in sequences:
        ids = torch.tensor(sequence, dtype=torch.long, device=weight.device)
        vectors.append(weight.index_select(0, ids).float().mean(dim=0))
    return torch.stack(vectors).mean(dim=0)


def readable_weight(module: torch.nn.Module) -> tuple[torch.Tensor, str]:
    """Return a real weight tensor without treating an Accelerate meta placeholder as data."""
    weight = module.weight
    if not weight.is_meta:
        return weight, "module_parameter"
    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        raise RuntimeError("offloaded output weight has no readable Accelerate weights_map")
    materialized = weights_map["weight"]
    if materialized.is_meta:
        raise RuntimeError("Accelerate weights_map returned a meta tensor")
    return materialized, "accelerate_offload_map"


def token_form(sequences_max: list[tuple[int, ...]], sequences_min: list[tuple[int, ...]]) -> np.ndarray:
    max_lengths = np.asarray([len(value) for value in sequences_max], dtype=np.float64)
    min_lengths = np.asarray([len(value) for value in sequences_min], dtype=np.float64)
    max_tokens = set().union(*(set(value) for value in sequences_max))
    min_tokens = set().union(*(set(value) for value in sequences_min))
    union = max_tokens | min_tokens
    intersection = max_tokens & min_tokens
    return np.asarray(
        [
            float(max_lengths.mean()),
            float(min_lengths.mean()),
            float(max_lengths.mean() - min_lengths.mean()),
            float(len(max_tokens)),
            float(len(min_tokens)),
            float(len(intersection) / max(len(union), 1)),
            float(len(max_tokens ^ min_tokens)),
        ],
        dtype=np.float32,
    )


def run(model_name: str) -> None:
    preregistration = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1100 protocol audit failed")
    rows = source_protocol.read_jsonl(source_protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    sequences = query_sequences(rows)
    expected = {
        (surface, relation, task)
        for surface in protocol.SURFACES
        for relation in source_protocol.RELATIONS
        for task in source_protocol.TASKS
    }
    if set(sequences) != expected:
        raise RuntimeError("query sequence grid is incomplete")

    started = time.time()
    model = None
    try:
        model, _, _, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        input_weight, input_weight_source = readable_weight(model.get_input_embeddings())
        output_module = model.get_output_embeddings()
        if output_module is None or not hasattr(output_module, "weight"):
            raise RuntimeError("model has no output embedding weight")
        output_weight, output_weight_source = readable_weight(output_module)
        input_vectors = []
        output_vectors = []
        form_vectors = []
        sequence_audit = []
        finite_values = 0
        total_values = 0
        with torch.inference_mode():
            for surface in protocol.SURFACES:
                surface_input = []
                surface_output = []
                surface_form = []
                for relation in source_protocol.RELATIONS:
                    maximum = sequences[(surface, relation, "max")]
                    minimum = sequences[(surface, relation, "min")]
                    in_delta = mean_sequence_vector(input_weight, maximum) - mean_sequence_vector(input_weight, minimum)
                    out_delta = mean_sequence_vector(output_weight, maximum) - mean_sequence_vector(output_weight, minimum)
                    in_array = in_delta.detach().cpu().numpy().astype(np.float32)
                    out_array = out_delta.detach().cpu().numpy().astype(np.float32)
                    surface_input.append(in_array)
                    surface_output.append(out_array)
                    surface_form.append(token_form(maximum, minimum))
                    finite_values += int(np.isfinite(in_array).sum() + np.isfinite(out_array).sum())
                    total_values += int(in_array.size + out_array.size)
                    sequence_audit.append(
                        {
                            "surface": surface,
                            "relation": relation,
                            "max_sequence_count": len(maximum),
                            "min_sequence_count": len(minimum),
                            "max_lengths": sorted({len(value) for value in maximum}),
                            "min_lengths": sorted({len(value) for value in minimum}),
                            "input_norm": float(np.linalg.norm(in_array)),
                            "output_norm": float(np.linalg.norm(out_array)),
                        }
                    )
                input_vectors.append(np.stack(surface_input))
                output_vectors.append(np.stack(surface_output))
                form_vectors.append(np.stack(surface_form))
        source_root = protocol.OUT_ROOT / "source" / model_name
        source_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            source_root / "lexical_source.npz",
            input_query_polarity=np.stack(input_vectors),
            output_query_polarity=np.stack(output_vectors),
            query_token_form=np.stack(form_vectors),
        )
        summary = {
            "schema_version": "phase1100_lexical_source_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": preregistration["protocol_digest"],
            "source_phase_protocol_digest": source_protocol.read_json(source_protocol.OUT_ROOT / "protocol" / "preregistration.json")["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "surfaces": list(protocol.SURFACES),
            "relations": list(source_protocol.RELATIONS),
            "input_dimension": int(input_weight.shape[1]),
            "output_dimension": int(output_weight.shape[1]),
            "input_weight_source": input_weight_source,
            "output_weight_source": output_weight_source,
            "source_finite_fraction": finite_values / max(total_values, 1),
            "source_values": total_values,
            "query_sequence_audit": sequence_audit,
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(source_root / "summary.json", summary)
        print(json.dumps({"phase": protocol.PHASE, "model": model_name, "finite_fraction": summary["source_finite_fraction"], "elapsed_seconds": summary["elapsed_seconds"], "summary_digest": summary["summary_digest"]}, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
