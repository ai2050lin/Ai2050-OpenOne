#!/usr/bin/env python3
"""Resume C412 after a process-level model-transition resource failure."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1933_c399_c414_output_sensitive_language_campaign as campaign


def reconstruct_glm(out: Path) -> dict:
    behavior = campaign.read_rows(out / "raw/glm4_behavior.jsonl")
    by_family = {
        family: float(np.mean([row["correct"] for row in behavior if row["family"] == family]))
        for family in campaign.FAMILIES
    }
    accuracy = float(np.mean([row["correct"] for row in behavior]))
    eligible = accuracy >= 0.80 and min(by_family.values()) >= 0.60
    states = np.load(out / "raw/glm4_role_states.float16.npy", mmap_mode="r")
    shape = list(states.shape)
    campaign.close_memmap(states)
    index = campaign.read_rows(out / "raw/glm4_hidden_index.jsonl")
    if len(index) != shape[0]:
        raise RuntimeError("GLM hidden-state archive/index mismatch")
    return {
        "model": "glm4",
        "rows": len(behavior),
        "placement": {
            "placement": "bf16_cuda_cpu_offload",
            "max_memory": {"gpu0": "11GiB", "cpu": "24GiB"},
            "reconstructed_from_complete_raw": True,
        },
        "accuracy": accuracy,
        "family_accuracy": by_family,
        "eligible": eligible,
        "capture": {"ran": True, "shape": shape, "rows": len(index)},
    }


def main() -> None:
    out = campaign.OUTS["C412"]
    if (out / "analysis/final.json").exists():
        raise RuntimeError("C412 is already closed")
    prereg = campaign.load(out / "protocol/preregistration.json")
    if prereg["producer_sha256"] != campaign.producer_hash():
        raise RuntimeError("Frozen C412 producer hash changed")
    rows = campaign.read_rows(out / "material/cases.jsonl")
    results = [reconstruct_glm(out)]
    try:
        results.append(campaign.run_external_model("deepseek7b", rows, out))
    except Exception as exc:
        error = {
            "model": "deepseek7b",
            "rows": len(rows),
            "status": "execution_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "eligible": False,
            "capture": {"ran": False},
        }
        campaign.save(out / "audit/deepseek7b_execution_error.json", error)
        results.append(error)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    campaign.save(
        out / "audit/process_transition_recovery.json",
        {
            "reason": "Original combined process exited while transitioning from completed GLM4 capture to DeepSeek load.",
            "contract_changed": False,
            "glm_reconstructed_from_complete_raw": True,
            "deepseek_fresh_process": True,
        },
    )
    headline = {
        "status": "cross_model_external_panel_closed",
        "results": results,
        "eligible_models": [row["model"] for row in results if row["eligible"]],
        "strict_interpretation": "Behavior-ineligible models are excluded from internal comparison; eligibility never aligns native coordinates.",
    }
    campaign.close(
        "C412",
        headline,
        {"rows": len(rows) == 480, "sequential": True, "finite": campaign.finite(headline)},
        "C413_abstraction",
    )


if __name__ == "__main__":
    main()
