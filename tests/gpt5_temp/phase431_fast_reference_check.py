#!/usr/bin/env python3
"""One-batch equivalence check for the Phase431 vectorized physical collector."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase431_position_time_collect import (  # noqa: E402
    LANGUAGE_MODEL,
    OUT,
    physical_batch,
    physical_batch_reference,
    prepare_physical_rows,
    random_projection_matrix,
)
from phase431_position_time_protocol import write_json  # noqa: E402


def flatten(value: Any, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "created_at":
                continue
            output.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(flatten(child, f"{prefix}[{index}]"))
    else:
        output[prefix] = value
    return output


def main() -> None:
    loaded = load_probe_model(LANGUAGE_MODEL)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(loaded.spec.local_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )
        layers = get_layers(loaded.model)
        hidden_size = int(loaded.model.config.hidden_size)
        projection = random_projection_matrix(hidden_size, loaded.input_device)
        rows = prepare_physical_rows("open")[:2]
        reference = physical_batch_reference(
            loaded, tokenizer, layers, rows, projection, None
        )
        vectorized = physical_batch(loaded, tokenizer, layers, rows, projection, None)
        reference_index = {
            (row["condition_id"], row["layer"]): flatten(row) for row in reference
        }
        vectorized_index = {
            (row["condition_id"], row["layer"]): flatten(row) for row in vectorized
        }
        if set(reference_index) != set(vectorized_index):
            raise RuntimeError("Vectorized collector changed the row denominator")
        maximum = 0.0
        maximum_path = None
        compared = 0
        for key in sorted(reference_index):
            left = reference_index[key]
            right = vectorized_index[key]
            if set(left) != set(right):
                missing = sorted(set(left).symmetric_difference(right))[:20]
                raise RuntimeError(f"Vectorized field mismatch at {key}: {missing}")
            for path in left:
                a, b = left[path], right[path]
                if isinstance(a, (int, float)) and not isinstance(a, bool):
                    difference = abs(float(a) - float(b))
                    if not math.isfinite(difference):
                        raise RuntimeError(f"Non-finite difference at {key} {path}")
                    compared += 1
                    if difference > maximum:
                        maximum = difference
                        maximum_path = f"{key}:{path}"
                elif a != b:
                    raise RuntimeError(f"Vectorized categorical mismatch at {key} {path}")
        tolerance = 1e-4
        payload = {
            "schema_version": "phase431_fast_reference_equivalence.v1",
            "condition_count": len(rows),
            "trace_row_count": len(reference),
            "numeric_field_comparisons": compared,
            "max_absolute_difference": maximum,
            "max_difference_path": maximum_path,
            "frozen_tolerance": tolerance,
            "pass": maximum <= tolerance,
        }
        write_json(OUT / "phase431_fast_reference_equivalence.json", payload)
        print(json.dumps(payload, indent=2))
        if not payload["pass"]:
            raise RuntimeError("Vectorized collector exceeded the frozen tolerance")
    finally:
        release_loaded(loaded)


if __name__ == "__main__":
    main()
