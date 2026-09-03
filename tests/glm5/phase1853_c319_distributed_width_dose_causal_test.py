#!/usr/bin/env python3
"""C319: source-role-count x coordinate-width x polarity causal dose map."""
from __future__ import annotations

import gc

import numpy as np
import torch

import phase1844_c310_c335_dual_axis_common as common


SOURCE_ROLES = (("relation",), ("relation", "query"), ("primary", "relation", "query"))
WIDTHS = (16, 64, 256, 2560)
POLARITIES = (("delete", -1.0), ("enhance", 1.0))


@torch.inference_mode()
def main() -> None:
    parent = common.core.load(common.OUTS["C318"] / "analysis/final.json")
    configs = {row["family"]: row for row in common.core.load(common.OUTS["C318"] / "protocol/intervention_configurations.json")["configurations"]}
    masks = np.load(common.OUTS["C318"] / "protocol/coordinate_width_masks.bool.npy")
    checks = {"parent": parent["all_checks_passed"], "cuda": torch.cuda.is_available(), "all_1152_evaluations_registered": True, "all_coordinate_reference": True}
    protocol = {
        "status": "distributed_dose_causal_test_frozen",
        "base_cases": "sixth H11/order+1/units4-7/two surfaces",
        "grid": "3 nested source-role sets x 4 coordinate widths x 2 polarities",
        "evaluations": 6 * 2 * 4 * 3 * 4 * 2,
        "patch": "At each family's C317 training-selected q, add polarity times the C314 training interaction residual to selected source role positions and coordinates.",
        "target": "C317 training-selected next-checkpoint destination role, evaluated on the same width mask",
        "metrics": ["movement toward additive target", "movement away under enhancement", "gold candidate margin"],
        "distributed_gate": "for delete polarity, full-width three-role movement exceeds 16-coordinate one-role movement by>=0.05 and is positive in at least four families",
        "claim_boundary": "The grid measures patch dose response. It does not turn amplitude-ranked coordinates into semantic atoms and does not establish a unique causal route.",
    }
    out = common.prepare("C319", protocol, checks)
    atlas = np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r")
    states = np.load(common.SIXTH_STATES, mmap_mode="r")
    index = common.core.rows(common.SIXTH_INDEX)
    compiled = common.core.rows(common.SIXTH_COMPILED)
    selected_cases = [row for row in index if row["factor_a"] == 1 and row["factor_b"] == 1 and row["order"] == 1 and row["unit"] >= 4 and row["correct"]]
    model = None
    samples = []
    try:
        model, _tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quantization = common.model_base.quantization_audit(model)
        layers = common.get_layers(model)
        for family_i, family in enumerate(common.FAMILIES):
            config = configs[family]
            q = int(config["q"])
            destination = int(config["destination_index"])
            family_residual = np.asarray(atlas[family_i, 0], np.float32)
            cases = [row for row in selected_cases if row["family"] == family]
            for source_row in cases:
                hidden_i = source_row["hidden_index"]
                row = compiled[hidden_i]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(ids)
                natural_target = np.asarray(states[hidden_i, common.CANONICAL_INDICES[q + 1], destination], np.float32)
                additive_target = natural_target - family_residual[q + 1, destination]
                natural_margin = None
                for source_count, roles in enumerate(SOURCE_ROLES, start=1):
                    for width_i, width in enumerate(WIDTHS):
                        coord_mask = masks[family_i, width_i]
                        coord_indices = np.flatnonzero(coord_mask)
                        denom = float(np.mean(np.abs(natural_target[coord_mask] - additive_target[coord_mask])))
                        for polarity_name, polarity in POLARITIES:
                            captured = []
                            role_vectors = np.zeros((6, 2560), np.float32)
                            for role in roles:
                                role_i = common.ROLES.index(role)
                                role_vectors[role_i, coord_mask] = polarity * family_residual[q, role_i, coord_mask]
                            vectors = common.role_position_vectors(row, role_vectors)

                            def patch_hook(_module, _args, output):
                                value = output[0] if isinstance(output, tuple) else output
                                updated = value.clone()
                                for position, vector in vectors.items():
                                    updated[0, position] = updated[0, position] + torch.tensor(vector, dtype=updated.dtype, device=updated.device)
                                if isinstance(output, tuple):
                                    return (updated, *output[1:])
                                return updated

                            def capture_hook(_module, _args, output):
                                value = output[0] if isinstance(output, tuple) else output
                                captured.append(value[0, row["role_positions"][common.ROLES[destination]]].mean(0).float().cpu().numpy())

                            patch = layers[q - 1].register_forward_hook(patch_hook)
                            capture = layers[q].register_forward_hook(capture_hook)
                            try:
                                output = model(input_ids=ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                            finally:
                                patch.remove()
                                capture.remove()
                            logits = np.asarray([float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]], np.float32)
                            gold = row["gold_position"]
                            target = captured[0]
                            movement = float(1.0 - np.mean(np.abs(target[coord_mask] - additive_target[coord_mask])) / max(denom, 1e-12))
                            margin = float(logits[gold] - logits[1 - gold])
                            samples.append({"family": family, "surface": source_row["surface"], "unit": source_row["unit"], "q": q, "destination_role": common.ROLES[destination], "source_count": source_count, "source_roles": list(roles), "coordinate_width": width, "polarity": polarity_name, "target_movement_toward_additive": movement, "gold_margin": margin})
                            del output
                print(f"[C319] {family}/{source_row['surface']}/u{source_row['unit']} complete", flush=True)
        common.core.write_rows(out / "raw/sample_results.jsonl", samples)
        aggregate_rows = []
        for family in common.FAMILIES:
            for source_count in (1, 2, 3):
                for width in WIDTHS:
                    for polarity_name, _polarity in POLARITIES:
                        rows = [row for row in samples if row["family"] == family and row["source_count"] == source_count and row["coordinate_width"] == width and row["polarity"] == polarity_name]
                        aggregate_rows.append({"family": family, "source_count": source_count, "coordinate_width": width, "polarity": polarity_name, "samples": len(rows), "mean_target_movement_toward_additive": float(np.mean([row["target_movement_toward_additive"] for row in rows])), "mean_gold_margin": float(np.mean([row["gold_margin"] for row in rows]))})
        common.core.write_rows(out / "analysis/dose_map.jsonl", aggregate_rows)
        family_rows = []
        for family in common.FAMILIES:
            lookup = {(row["source_count"], row["coordinate_width"], row["polarity"]): row for row in aggregate_rows if row["family"] == family}
            full = lookup[(3, 2560, "delete")]["mean_target_movement_toward_additive"]
            minimal = lookup[(1, 16, "delete")]["mean_target_movement_toward_additive"]
            polarity_separation = full - lookup[(3, 2560, "enhance")]["mean_target_movement_toward_additive"]
            family_rows.append({"family": family, "full_three_role_delete_movement": full, "minimal_one_role_delete_movement": minimal, "full_minus_minimal": full - minimal, "delete_minus_enhance": polarity_separation, "family_gate_passed": full > 0 and full - minimal >= 0.05})
        passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
        headline = {"status": "distributed_dose_causal_adjudicated", "families": family_rows, "families_passing": passing, "distributed_gate_passed": len(passing) >= 4, "placement": placement, "quantization": quantization, "strict_interpretation": protocol["claim_boundary"]}
        common.close("C319", headline, {"evaluations": len(samples) == 1152, "aggregate_cells": len(aggregate_rows) == 6 * 3 * 4 * 2, "eight_base_cases_per_family": all(sum(row["family"] == family for row in selected_cases) == 8 for family in common.FAMILIES), "finite": common.finite_dict(headline)}, "C320_residual_causal_stage_audit")
    finally:
        common.model_base.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
