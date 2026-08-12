#!/usr/bin/env python3
"""Freeze adaptive fine-observation targets from the Phase1008 atlas."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1008_global_response_atlas_protocol import (
    OUT_ROOT,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
)


MODELS = ("qwen3", "glm4")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def selected_motifs(model_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(
        OUT_ROOT / "final" / model_name / "trajectory_motifs.jsonl"
    )
    return [
        row
        for row in rows
        if row["refinement_eligible"]
        and row["evidence_axes"]["M_cross_model"]["support"] >= 2
        and row["role"] in ("decision_boundary", "answer_boundary")
        and row["operation"] in ("B", "Q", "BQ", "X")
    ]


def build_model_spec(model_name: str) -> dict[str, Any]:
    rows = selected_motifs(model_name)
    attention_sources = [
        row
        for row in rows
        if row["stage"] == "semantic0"
        and row["component"] == "attention_output"
        and row["operation"] in ("B", "Q", "X")
    ]
    attention_peaks = sorted({
        int(row["peak_depth"]) for row in attention_sources
    })
    attention_layers = sorted({
        layer
        for peak in attention_peaks
        for layer in (peak - 1, peak, peak + 1)
        if layer >= 1
    })
    mlp_sources = [
        row
        for row in rows
        if row["component"] == "mlp_output"
        and (
            (
                row["stage"] == "semantic0"
                and row["operation"] in ("B", "Q", "BQ", "X")
            )
            or (
                row["stage"] == "prompt"
                and row["operation"] == "X"
                and row["role"] == "answer_boundary"
            )
        )
    ]
    mlp_targets = sorted({
        (row["stage"], row["role"], int(row["peak_depth"]))
        for row in mlp_sources
    })
    if not attention_layers or not mlp_targets:
        raise RuntimeError(f"{model_name}: underfilled refinement targets")
    return {
        "model": model_name,
        "attention": {
            "stage": "semantic0",
            "role": "decision_boundary",
            "peak_layers": attention_peaks,
            "scan_layers": attention_layers,
            "source_motif_ids": [
                row["motif_id"] for row in attention_sources
            ],
            "decomposition": (
                "real pre-o_proj head slices projected through matching "
                "o_proj columns"
            ),
        },
        "mlp": {
            "targets": [
                {"stage": stage, "role": role, "layer": layer}
                for stage, role, layer in mlp_targets
            ],
            "source_motif_ids": [row["motif_id"] for row in mlp_sources],
            "decomposition": (
                "real down_proj input activation change multiplied by the "
                "matching down_proj column norm"
            ),
        },
    }


def main() -> None:
    atlas = read_json(OUT_ROOT / "final" / "summary.json")
    if atlas["automatic_next_action"] != (
        "targeted_head_and_neuron_observation_warranted"
    ):
        raise RuntimeError("atlas did not authorize fine observation")
    payload = {
        "schema_version": "phase1008_refinement_protocol.v1",
        "phase": PHASE,
        "protocol_revision": 4,
        "parent_protocol_digest": atlas["protocol_digest"],
        "parent_atlas_summary_digest": digest(atlas),
        "models": list(MODELS),
        "excluded_models": {
            "deepseek7b": (
                "no B/Q/BQ trajectory passed the two-split repeated-region "
                "selection and natural rollout evidence was nearly absent"
            )
        },
        "operations": ["B", "Q", "BQ", "X"],
        "model_targets": {
            model_name: build_model_spec(model_name)
            for model_name in MODELS
        },
        "selection_contract": {
            "input": (
                "only refinement-eligible, Qwen3+GLM4 coordinate-free "
                "cross-model motifs from the frozen global atlas"
            ),
            "attention": (
                "semantic0 decision-boundary B/Q/X peaks plus one adjacent "
                "layer on each side"
            ),
            "mlp": (
                "semantic0 B/Q/BQ/X peaks and prompt X answer-boundary peak"
            ),
            "head_ids_cross_model_aligned": False,
            "neuron_ids_cross_model_aligned": False,
            "causal_claim_allowed": False,
        },
        "storage_contract": {
            "persist": (
                "per-unit per-head contribution scalars and per-unit "
                "per-neuron write-magnitude scalars"
            ),
            "do_not_persist": (
                "raw head vectors, raw neuron activations, or full hidden "
                "states"
            ),
        },
        "weight_instrument": {
            "source": (
                "explicit CB+SCB dequantization of the actual runtime 8bit "
                "matrix; generic dequantize() is forbidden"
            ),
            "runtime_comparison": (
                "compare both the explicitly dequantized runtime matrix and "
                "the original local BF16 matrix against the actual 8bit "
                "forward"
            ),
            "maximum_attention_runtime_relative_error": 0.06,
            "maximum_mlp_runtime_relative_error": 0.13,
            "maximum_original_reference_relative_error": 0.15,
            "dual_weight_rank_gate": {
                "attention_top_quartile_median_jaccard": 0.75,
                "attention_top_quartile_minimum_jaccard": 0.45,
                "mlp_top_one_percent_median_jaccard": 0.90,
                "mlp_top_one_percent_minimum_jaccard": 0.75,
                "median_magnitude_correlation": 0.99,
            },
            "revision_reason": (
                "revision 1 exposed unscaled int8 bytes; revision 2 showed "
                "that original BF16 GLM4 MLP weights differ materially from "
                "the quantized runtime; revision 3 calibration showed that "
                "8bit input quantization dominates reconstruction error. "
                "Revision 4 therefore adds independent dual-weight ranking "
                "stability gates. No earlier refinement result is admissible"
            ),
        },
        "recurrence_observer": {
            "head": (
                "top quartile within each selected layer, repeated by split, "
                "template, and name pool"
            ),
            "neuron": (
                "top one percent within each selected layer, repeated by "
                "split, template, and name pool"
            ),
            "meaning": "candidate observation only, never causal necessity",
        },
    }
    payload["preregistration_digest"] = digest(payload)
    output = OUT_ROOT / "refinement" / "protocol.json"
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
