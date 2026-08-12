#!/usr/bin/env python3
"""Aggregate natural read bands and freeze causal confirmation candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

import phase1048_natural_attention_read_protocol as protocol


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model} protocol digest mismatch")

    repeated = []
    for slot in range(1, protocol.NORMALIZED_READ_SLOTS + 1):
        for destination in protocol.DESTINATIONS:
            model_rows = {}
            for model, summary in summaries.items():
                row = next(
                    item for item in summary["analysis"]["bands"]
                    if int(item["normalized_read_slot"]) == slot
                    and item["destination"] == destination
                )
                if (
                    int(row["passing_head_cells"])
                    >= prereg["cross_model_band_gate"][
                        "minimum_passing_heads_per_model"
                    ]
                ):
                    model_rows[model] = row
            if (
                len(model_rows)
                >= prereg["cross_model_band_gate"]["minimum_models"]
            ):
                scores = [
                    float(row["maximum_score"])
                    for row in model_rows.values()
                    if row["maximum_score"] is not None
                ]
                repeated.append({
                    "normalized_read_slot": slot,
                    "destination": destination,
                    "models": sorted(model_rows),
                    "model_count": len(model_rows),
                    "model_rows": model_rows,
                    "median_maximum_score": (
                        float(np.median(scores)) if scores else 0.0
                    ),
                    "passing_head_cells_total": sum(
                        int(row["passing_head_cells"])
                        for row in model_rows.values()
                    ),
                })
    repeated.sort(
        key=lambda row: (
            -int(row["model_count"]),
            -float(row["median_maximum_score"]),
            -int(row["passing_head_cells_total"]),
            int(row["normalized_read_slot"]),
            protocol.DESTINATIONS.index(row["destination"]),
        )
    )
    frozen = repeated[
        :prereg["cross_model_band_gate"]["maximum_frozen_bands"]
    ]

    artifacts = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*") if item.is_file()
    ):
        artifacts.append({
            "path": str(path.relative_to(protocol.OUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    aggregate: dict[str, Any] = {
        "schema_version": "phase1048_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_behavior": {
            model: summary["behavior"]
            for model, summary in summaries.items()
        },
        "model_passing_cell_counts": {
            model: int(summary["analysis"]["passing_cell_count"])
            for model, summary in summaries.items()
        },
        "cross_model_repeated_band_count": len(repeated),
        "cross_model_repeated_bands": repeated,
        "automatic_next_decision": {
            "causal_confirmation_needed": bool(frozen),
            "frozen_bands": frozen,
            "route": (
                "Run the preregistered held-out K/V/KV and destination-Q "
                "reset/replay confirmation."
                if frozen
                else
                "Stop this read-path block; Attention observations alone "
                "do not establish a causal edge."
            ),
        },
        "artifact_manifest": {
            "file_count": len(artifacts),
            "total_bytes": sum(row["bytes"] for row in artifacts),
            "files": artifacts,
        },
        "interpretation": [
            "A repeated band is a natural query-conditioned read candidate.",
            "Attention mass and A-times-V norm remain descriptive until "
            "held-out projection reset/replay confirms causal use.",
            "Physical head IDs are not compared across models.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print({
        "phase": protocol.PHASE,
        "repeated_bands": len(repeated),
        "frozen_bands": [
            (
                row["normalized_read_slot"],
                row["destination"],
                row["models"],
            )
            for row in frozen
        ],
    })


if __name__ == "__main__":
    main()
