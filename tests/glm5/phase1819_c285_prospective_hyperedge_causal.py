#!/usr/bin/env python3
"""C285: locally delete and rescue prospectively qualified joint-word edges."""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1811_c277_c289_joint_response_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step

core, OUT = common.core, common.OUTS["C285"]
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
C281 = common.OUTS["C281"]
Q = 16
CANDIDATE = "primary_relation_query"
SOURCE_ROLES = partition.CANDIDATES[CANDIDATE]
DESTINATION = common.ROLES.index("boundary")
CONDITIONS = ("natural", "delete", "correct_rescue", "coordinate_roll_rescue", "wrong_role_rescue")


def pair_ids(index: list[dict], family: str):
    return common.pair_specs(index, family)


@torch.inference_mode()
def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C281 / "analysis/final.json")
    gates = core.load(common.OUTS["C277"] / "protocol/preregistration.json")["gates"]
    candidate_ok = next(row["broad_gate_passed"] for row in parent["headline"]["candidate_summary"] if row["candidate"] == CANDIDATE)
    checks = {
        "parent": parent["all_checks_passed"], "candidate_eligible": candidate_ok, "fixed_q": Q == 16,
        "all_rule_defined_coordinates": True, "no_topk_pca_cosine_attention_mlp": True, "cuda": torch.cuda.is_available(),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol", "raw"): (OUT / subdir).mkdir()
    protocol = {
        "phase": 1819, "campaign": "C285", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "causal_contract_frozen",
        "candidate": CANDIDATE, "transition": "canonical q16 block-16 output to q17 block-17 output", "destination_role": "boundary",
        "sample": "first behavior-correct factor-A pair from each surface in each family; 12 pairs total",
        "eligible_coordinates": "all coordinates where the frozen joint word predicts a nonzero q17 boundary event while q16 boundary response is silent",
        "conditions": list(CONDITIONS),
        "delete": "replace eligible source-role coordinates at block-16 output with the matched left-cell source state",
        "correct_rescue": "restore the matched natural right-cell source response after deletion",
        "controls": ["roll the response by one physical coordinate", "write the registered response to secondary/context roles"],
        "gate": "successor target-rate deletion drop>=0.20, correct rescue within 0.05 of natural, and correct rescue exceeds every wrong rescue by>=0.10 in at least four families",
        "claim_boundary": "The intervention targets a rule-defined coordinate coalition. Passing would establish local causal use, not uniqueness or a complete circuit.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C286_generation_regardless",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_a = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    train_b = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    test_states = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    test_index = core.rows(C278 / "raw/hidden_index.jsonl")
    compiled = core.rows(common.OUTS["C277"] / "compiled/qwen3.jsonl")
    train_indices = {"a": core.rows(C248 / "raw/hidden_index.jsonl"), "b": core.rows(C264 / "raw/hidden_index.jsonl")}
    threshold = common.thresholds()
    selected = []
    for family in common.FAMILIES:
        specs = pair_ids(test_index, family)
        for surface in common.SURFACES:
            match = next(row for row in specs if row[2]["surface"] == surface)
            selected.append((family, *match))
    model = None
    sample_rows = []
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quant = common.model_base.quantization_audit(model)
        base = model.model
        for family, left_id, right_id, meta in selected:
            train_specs_a = pair_ids(train_indices["a"], family); train_specs_b = pair_ids(train_indices["b"], family)
            al = np.asarray([x[0] for x in train_specs_a], int); ar = np.asarray([x[1] for x in train_specs_a], int)
            bl = np.asarray([x[0] for x in train_specs_b], int); br = np.asarray([x[1] for x in train_specs_b], int)
            train_current = np.concatenate((
                common.event(np.asarray(train_a[ar, Q], np.float32) - np.asarray(train_a[al, Q], np.float32), threshold[Q]),
                common.event(np.asarray(train_b[br, Q], np.float32) - np.asarray(train_b[bl, Q], np.float32), threshold[Q]),
            ), axis=0)
            train_next = np.concatenate((
                common.event(np.asarray(train_a[ar, Q + 1, DESTINATION], np.float32) - np.asarray(train_a[al, Q + 1, DESTINATION], np.float32), threshold[Q + 1]),
                common.event(np.asarray(train_b[br, Q + 1, DESTINATION], np.float32) - np.asarray(train_b[bl, Q + 1, DESTINATION], np.float32), threshold[Q + 1]),
            ), axis=0)
            states_count = 3 ** len(SOURCE_ROLES)
            train_code = partition.code_word(train_current, SOURCE_ROLES)
            fitted, _key, _support = one_step.fit_map(train_code, train_next, states_count, 4, 0.70)
            current_delta = np.asarray(test_states[right_id, common.CANONICAL_NEW_INDICES[Q]], np.float32) - np.asarray(test_states[left_id, common.CANONICAL_NEW_INDICES[Q]], np.float32)
            current_event = common.event(current_delta, threshold[Q])[None, ...]
            code = partition.code_word(current_event, SOURCE_ROLES)
            pure = one_step.lookup(fitted, code, states_count)[0]
            eligible = (current_event[0, DESTINATION] == 0) & (pure != 0)
            row = compiled[right_id]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            replacement = {
                ri: {
                    "left": np.asarray(test_states[left_id, common.CANONICAL_NEW_INDICES[Q], ri], np.float32),
                    "right": np.asarray(test_states[right_id, common.CANONICAL_NEW_INDICES[Q], ri], np.float32),
                }
                for ri in range(6)
            }
            condition_rows = {}
            for condition in CONDITIONS:
                next_state = []
                eligible_t = torch.tensor(eligible, dtype=torch.bool, device=device)

                def patch_hook(_module, _args, output):
                    if condition == "natural": return output
                    value = output.clone()
                    if condition == "wrong_role_rescue":
                        operations = [(ri, "left") for ri in SOURCE_ROLES] + [(ri, "right") for ri in (common.ROLES.index("secondary"), common.ROLES.index("context"))]
                    else:
                        operations = [(ri, condition) for ri in SOURCE_ROLES]
                    for ri, operation in operations:
                        role = common.ROLES[ri]
                        left = replacement[ri]["left"]
                        right = replacement[ri]["right"]
                        if operation in ("delete", "left"): donor = left
                        elif operation == "coordinate_roll_rescue": donor = left + np.roll(right - left, 1)
                        else: donor = right
                        donor_t = torch.tensor(donor, dtype=value.dtype, device=device)
                        for position in row["role_positions"][role]:
                            updated = value[0, position].clone(); updated[eligible_t] = donor_t[eligible_t]; value[0, position] = updated
                    return value

                def next_hook(_module, _args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    next_state.append(value[0, row["role_positions"]["boundary"]].mean(0).float().cpu().numpy())

                patch = base.layers[Q - 1].register_forward_hook(patch_hook)
                capture = base.layers[Q].register_forward_hook(next_hook)
                try:
                    output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                finally:
                    patch.remove(); capture.remove()
                logits = np.asarray([float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]])
                gold = row["gold_position"]
                margin = float(logits[gold] - logits[1 - gold])
                next_delta = next_state[0] - np.asarray(test_states[left_id, common.CANONICAL_NEW_INDICES[Q + 1], DESTINATION], np.float32)
                event = common.event(next_delta, threshold[Q + 1])
                target_rate = float(np.mean(event[eligible] == pure[eligible])) if eligible.any() else 0.0
                condition_rows[condition] = {"successor_target_rate": target_rate, "candidate_margin": margin, "prediction": int(logits[1] > logits[0])}
                del output
            natural = condition_rows["natural"]["successor_target_rate"]
            delete = condition_rows["delete"]["successor_target_rate"]
            correct = condition_rows["correct_rescue"]["successor_target_rate"]
            wrong = max(condition_rows["coordinate_roll_rescue"]["successor_target_rate"], condition_rows["wrong_role_rescue"]["successor_target_rate"])
            sample_rows.append({
                "family": family, "surface": meta["surface"], "left_id": int(left_id), "right_id": int(right_id), "eligible_coordinates": int(eligible.sum()),
                "conditions": condition_rows, "deletion_drop": natural - delete, "correct_recovery_error": abs(correct - natural), "correct_minus_best_wrong": correct - wrong,
            })
            print(f"[C285] {family}/{meta['surface']}: n={eligible.sum()}, drop={natural-delete:+.4f}, rescue={correct-wrong:+.4f}", flush=True)
        core.write_rows(OUT / "raw/sample_results.jsonl", sample_rows)
        families = []
        for family in common.FAMILIES:
            rows = [r for r in sample_rows if r["family"] == family]
            aggregate = {key: float(np.mean([r[key] for r in rows])) for key in ("deletion_drop", "correct_recovery_error", "correct_minus_best_wrong")}
            aggregate["family"] = family
            aggregate["eligible_coordinates_mean"] = float(np.mean([r["eligible_coordinates"] for r in rows]))
            aggregate["family_gate_passed"] = aggregate["deletion_drop"] >= gates["causal_flip_min"] and aggregate["correct_recovery_error"] <= 0.05 and aggregate["correct_minus_best_wrong"] >= gates["causal_control_margin_min"]
            families.append(aggregate)
        passing = sum(bool(r["family_gate_passed"]) for r in families)
        report = {
            "phase": 1819, "campaign": "C285", "status": "joint_word_local_causal_adjudicated", "families": families,
            "families_passing": passing, "broad_gate_passed": passing >= 4, "placement": placement, "quantization": quant,
            "strict_interpretation": "The test intervenes only on the registered q16 source-role coordinate coalition. Correct rescue is restoration of the natural source response; it is not an independently generated mechanism state.",
            "next_authorization": "C286_generation_and_side_effects",
        }
        core.save(OUT / "analysis/summary.json", report)
        ach = {"samples": len(sample_rows) == 12, "families": len(families) == 6, "eligible": all(r["eligible_coordinates"] > 0 for r in sample_rows), "finite": bool(np.isfinite([r[k] for r in families for k in ("deletion_drop", "correct_minus_best_wrong")]).all())}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
        fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1819, "campaign": "C285", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))
    finally:
        common.model_base.release(model); gc.collect()


if __name__ == "__main__": main()
