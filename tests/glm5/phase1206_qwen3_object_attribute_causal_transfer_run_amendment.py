#!/usr/bin/env python3
"""Narrow loader-signature compatibility amendment for Phase1206.

The frozen driver passed the already frozen model path to a project loader that
accepts only the model registry name.  No model output existed when this was
detected.  This wrapper changes only the Python call signature and delegates to
the exact same FP16 loader used by Phase1205.
"""

from __future__ import annotations

import torch

import phase1206_qwen3_object_attribute_causal_transfer as phase1206


ORIGINAL_LOAD_FP16 = phase1206.load_fp16


def compatible_load_fp16(model_name, _frozen_model_path):
    return ORIGINAL_LOAD_FP16(model_name)


def main() -> None:
    if phase1206.VECTOR_PATH.exists() or phase1206.RAW_PATH.exists() or phase1206.RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1206 output already exists")
    phase1206.load_fp16 = compatible_load_fp16
    # Forward-only activation patching must not construct autograd graphs.  The
    # frozen driver already does this for capture; this outer guard extends the
    # same semantics to every intervention condition.
    with torch.inference_mode():
        phase1206.run_command()


if __name__ == "__main__":
    main()
