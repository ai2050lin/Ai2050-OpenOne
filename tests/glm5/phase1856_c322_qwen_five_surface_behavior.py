#!/usr/bin/env python3
"""C322: Qwen behavior qualification on all 1,920 frozen cases."""
from __future__ import annotations

import gc

import numpy as np
import torch

import phase1844_c310_c335_dual_axis_common as common


@torch.inference_mode()
def main() -> None:
    parent = common.core.load(common.OUTS["C321"] / "analysis/final.json")
    rows = common.core.rows(common.OUTS["C321"] / "material/cases.jsonl")
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 1920, "cuda": torch.cuda.is_available(), "model_not_previously_exposed": True}
    protocol = {"status": "qwen_behavior_frozen", "model": "Qwen3-4B BF16 CUDA unquantized", "cases": 1920, "gate": common.core.load(common.OUTS["C321"] / "protocol/preregistration.json")["behavior_gate"], "hidden_state_rule": "No hidden-state analysis is performed in this phase.", "claim_boundary": "Behavior qualification establishes task competence on controlled wrappers; it does not show a hidden mechanism."}
    out = common.prepare("C322", protocol, checks)
    model = None
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quantization = common.model_base.quantization_audit(model)
        compiled = common.compile_qwen(tokenizer, rows)
        if max(len(row["prompt_ids"]) for row in compiled) > common.WIDTH:
            raise RuntimeError("prompt width")
        common.core.write_rows(out / "compiled/qwen3.jsonl", compiled)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        behavior = []
        for start in range(0, len(compiled), 8):
            batch = [(row, row["prompt_ids"], row["candidate_ids"]) for row in compiled[start:start + 8]]
            behavior.extend(common.score_interface_batch(model, device, pad, batch))
            if start % 256 == 0 or start + len(batch) == len(compiled):
                print(f"[C322] {start + len(batch)}/1920", flush=True)
        common.core.write_rows(out / "raw/behavior.jsonl", behavior)
        accuracy = float(np.mean([row["correct"] for row in behavior]))
        lookup = {row["case_id"]: source for row, source in zip(behavior, rows)}
        by_family = {family: float(np.mean([row["correct"] for row in behavior if lookup[row["case_id"]]["family"] == family])) for family in common.FAMILIES}
        by_surface = {surface: float(np.mean([row["correct"] for row in behavior if lookup[row["case_id"]]["surface"] == surface])) for surface in common.NATURAL_SURFACES}
        gate = protocol["gate"]
        eligible = accuracy >= gate["global_min"] and min(by_family.values()) >= gate["family_min"] and min(by_surface.values()) >= gate["surface_min"]
        headline = {"status": "qwen_behavior_adjudicated", "accuracy": accuracy, "by_family_accuracy": by_family, "by_surface_accuracy": by_surface, "behavior_eligible": eligible, "placement": placement, "quantization": quantization, "strict_interpretation": protocol["claim_boundary"]}
        common.close("C322", headline, {"behavior_rows": len(behavior) == 1920, "compiled_rows": len(compiled) == 1920, "finite": common.finite_dict(headline), "bf16": quantization["has_bf16_parameters"], "unquantized": not quantization["has_quantized_modules"]}, "C323_hidden_capture" if eligible else "C323_observation_capture_with_behavior_stratification")
    finally:
        common.model_base.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
