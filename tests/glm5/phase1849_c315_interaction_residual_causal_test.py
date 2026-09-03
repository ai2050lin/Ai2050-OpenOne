#!/usr/bin/env python3
"""C315: delete frozen full-coordinate interaction residuals at training-selected layers."""
from __future__ import annotations

import gc

import numpy as np
import torch

import phase1844_c310_c335_dual_axis_common as common


CONDITIONS = ("natural", "correct_delete", "global_delete", "wrong_family_delete", "coordinate_roll_delete", "role_roll_delete")


@torch.inference_mode()
def main() -> None:
    parent = common.core.load(common.OUTS["C314"] / "analysis/final.json")
    strata = common.core.load(common.OUTS["C314"] / "protocol/selected_causal_strata.json")["strata"]
    checks = {"parent": parent["all_checks_passed"], "six_training_selected_strata": len(strata) == 6, "cuda": torch.cuda.is_available(), "all_role_all_coordinate_patch": True}
    protocol = {
        "status": "coarse_residual_causal_test_frozen",
        "samples": "sixth material H11, order +1: 6 families x 2 surfaces x 8 units = 96 base cases",
        "conditions": list(CONDITIONS),
        "evaluations": 96 * len(CONDITIONS),
        "patch": "At the frozen family checkpoint, subtract a role-position-averaged residual over every one of 2560 coordinates. Overlapping semantic roles at one token are averaged, not double-counted.",
        "controls": "RMS-matched global, cyclic wrong-family, coordinate-roll-97, and role-roll-1 residual deletions",
        "primary_metric": "movement of the next checkpoint's six role states toward natural H11 minus the training interaction residual",
        "secondary_metric": "drop in gold candidate margin",
        "family_gate": "correct primary movement>=0.05 and exceeds the best named control by>=0.02",
        "claim_boundary": "Passing would show causal sensitivity to one broad residual-shaped role-state patch, not natural computation of a unique semantic residual. Failure rejects this layer-local patch interface only.",
    }
    out = common.prepare("C315", protocol, checks)
    atlas = np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r")
    family_means = {family: np.asarray(atlas[i, 0], np.float32) for i, family in enumerate(common.FAMILIES)}
    global_mean = np.mean(np.stack(list(family_means.values()), axis=0), axis=0)
    states = np.load(common.SIXTH_STATES, mmap_mode="r")
    index = common.core.rows(common.SIXTH_INDEX)
    compiled = common.core.rows(common.SIXTH_COMPILED)
    selected = [row for row in index if row["factor_a"] == 1 and row["factor_b"] == 1 and row["order"] == 1 and row["correct"]]
    model = None
    samples = []
    try:
        model, _tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quantization = common.model_base.quantization_audit(model)
        layers = get_layers = common.get_layers(model)
        for family_i, family in enumerate(common.FAMILIES):
            q = next(row["q"] for row in strata if row["family"] == family)
            correct = family_means[family][q]
            wrong = family_means[common.FAMILIES[(family_i + 1) % 6]][q]
            controls = {
                "correct_delete": correct,
                "global_delete": common.norm_match(global_mean[q], correct),
                "wrong_family_delete": common.norm_match(wrong, correct),
                "coordinate_roll_delete": np.roll(correct, 97, axis=-1),
                "role_roll_delete": np.roll(correct, 1, axis=0),
            }
            family_cases = [row for row in selected if row["family"] == family]
            for source_row in family_cases:
                hidden_i = source_row["hidden_index"]
                row = compiled[hidden_i]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                natural_next = np.asarray(states[hidden_i, common.CANONICAL_INDICES[q + 1]], np.float32)
                additive_next = natural_next - family_means[family][q + 1]
                denom = float(np.mean(np.abs(natural_next - additive_next)))
                condition_rows = {}
                for condition in CONDITIONS:
                    captured = []
                    vectors = None if condition == "natural" else common.role_position_vectors(row, controls[condition])

                    def patch_hook(_module, _args, output):
                        if vectors is None:
                            return output
                        value = output[0] if isinstance(output, tuple) else output
                        updated = value.clone()
                        for position, vector in vectors.items():
                            updated[0, position] = updated[0, position] - torch.tensor(vector, dtype=updated.dtype, device=updated.device)
                        if isinstance(output, tuple):
                            return (updated, *output[1:])
                        return updated

                    def capture_hook(_module, _args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        captured.append(np.asarray([
                            value[0, row["role_positions"][role]].mean(0).float().cpu().numpy()
                            for role in common.ROLES
                        ], np.float32))

                    patch = layers[q - 1].register_forward_hook(patch_hook)
                    capture = layers[q].register_forward_hook(capture_hook)
                    try:
                        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    finally:
                        patch.remove()
                        capture.remove()
                    logits = np.asarray([float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]], np.float32)
                    gold = row["gold_position"]
                    next_state = captured[0]
                    movement = float(1.0 - np.mean(np.abs(next_state - additive_next)) / max(denom, 1e-12))
                    condition_rows[condition] = {"next_field_movement_toward_additive": movement, "gold_margin": float(logits[gold] - logits[1 - gold]), "prediction": int(logits[1] > logits[0])}
                    del output
                natural_margin = condition_rows["natural"]["gold_margin"]
                correct_movement = condition_rows["correct_delete"]["next_field_movement_toward_additive"]
                best_control_movement = max(condition_rows[name]["next_field_movement_toward_additive"] for name in CONDITIONS[2:])
                samples.append({
                    "family": family,
                    "surface": source_row["surface"],
                    "unit": source_row["unit"],
                    "q": q,
                    "conditions": condition_rows,
                    "correct_primary_movement": correct_movement,
                    "correct_minus_best_control_movement": correct_movement - best_control_movement,
                    "correct_gold_margin_drop": natural_margin - condition_rows["correct_delete"]["gold_margin"],
                })
                print(f"[C315] {family}/{source_row['surface']}/u{source_row['unit']}: move={correct_movement:+.4f} select={correct_movement-best_control_movement:+.4f}", flush=True)
        common.core.write_rows(out / "raw/sample_results.jsonl", samples)
        family_rows = []
        for family in common.FAMILIES:
            rows = [row for row in samples if row["family"] == family]
            movement = float(np.mean([row["correct_primary_movement"] for row in rows]))
            selectivity = float(np.mean([row["correct_minus_best_control_movement"] for row in rows]))
            margin_drop = float(np.mean([row["correct_gold_margin_drop"] for row in rows]))
            family_rows.append({"family": family, "samples": len(rows), "mean_primary_movement": movement, "mean_correct_minus_best_control": selectivity, "mean_gold_margin_drop": margin_drop, "family_gate_passed": movement >= 0.05 and selectivity >= 0.02})
        passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
        headline = {"status": "coarse_residual_causal_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "placement": placement, "quantization": quantization, "strict_interpretation": protocol["claim_boundary"]}
        common.close("C315", headline, {"samples": len(samples) == 96, "sixteen_per_family": all(sum(row["family"] == family for row in samples) == 16 for family in common.FAMILIES), "conditions": all(set(row["conditions"]) == set(CONDITIONS) for row in samples), "finite": common.finite_dict(headline)}, "C316_C320_continue_regardless_of_causal_gate")
    finally:
        common.model_base.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
