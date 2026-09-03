#!/usr/bin/env python3
"""C291: causally test every C290-qualified transition-role stratum."""
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

core = common.core
OUT = common.RESULT / "phase1825_c291_training_qualified_joint_word_causal"
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
C290 = common.RESULT / "phase1824_c290_training_supported_causal_strata"
CANDIDATE = "primary_relation_query"
SOURCE_ROLES = partition.CANDIDATES[CANDIDATE]
CONDITIONS = ("natural", "delete", "correct_rescue", "coordinate_roll_rescue", "wrong_role_rescue")


def specs(index: list[dict], family: str):
    return common.pair_specs(index, family)


@torch.inference_mode()
def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C290 / "analysis/final.json")
    passing = tuple(parent["headline"]["passing_families"])
    gates = core.load(common.OUTS["C277"] / "protocol/preregistration.json")["gates"]
    checks = {
        "parent": parent["all_checks_passed"],
        "has_qualified_branches": bool(passing),
        "only_C290_passing_families": True,
        "all_rule_defined_coordinates": True,
        "no_topk_pca_cosine_attention_mlp": True,
        "cuda": torch.cuda.is_available(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol", "raw"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1825,
        "campaign": "C291",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "qualified_local_causal_contract_frozen",
        "candidate": CANDIDATE,
        "families": list(passing),
        "strata": {
            row["family"]: row["selected_stratum"]
            for row in parent["headline"]["families"] if row["family"] in passing
        },
        "samples": "the two fifth-material surface pairs registered by C290 for every passing family",
        "eligible_coordinates": "all coordinates where the frozen joint word predicts a nonzero next destination event while the current destination response is silent",
        "conditions": list(CONDITIONS),
        "delete": "replace eligible coordinates at all three source roles with the matched left-cell state",
        "correct_rescue": "restore the matched natural right-cell source-role state",
        "controls": ["roll source response one physical coordinate", "write source response at secondary/context roles"],
        "family_gate": "mean successor target-rate deletion drop>=0.20, correct rescue within 0.05 of natural, and correct rescue exceeds every wrong rescue by>=0.10",
        "claim_boundary": "Passing establishes local causal use of a rule-defined source-role coordinate coalition. It does not prove uniqueness, minimality, or a complete language circuit.",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_a = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    train_b = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    test_states = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    train_indices = {
        "a": core.rows(C248 / "raw/hidden_index.jsonl"),
        "b": core.rows(C264 / "raw/hidden_index.jsonl"),
    }
    test_index = core.rows(C278 / "raw/hidden_index.jsonl")
    compiled = core.rows(common.OUTS["C277"] / "compiled/qwen3.jsonl")
    thresholds = common.thresholds()
    parent_rows = {row["family"]: row for row in parent["headline"]["families"]}
    model = None
    sample_rows = []
    try:
        model, _tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quantization = common.model_base.quantization_audit(model)
        base = model.model
        for family in passing:
            parent_row = parent_rows[family]
            q = int(parent_row["selected_stratum"]["q"])
            destination = int(parent_row["selected_stratum"]["destination_index"])
            train_a_specs = specs(train_indices["a"], family)
            train_b_specs = specs(train_indices["b"], family)
            al = np.asarray([row[0] for row in train_a_specs], int)
            ar = np.asarray([row[1] for row in train_a_specs], int)
            bl = np.asarray([row[0] for row in train_b_specs], int)
            br = np.asarray([row[1] for row in train_b_specs], int)
            train_current = np.concatenate((
                common.event(np.asarray(train_a[ar, q], np.float32) - np.asarray(train_a[al, q], np.float32), thresholds[q]),
                common.event(np.asarray(train_b[br, q], np.float32) - np.asarray(train_b[bl, q], np.float32), thresholds[q]),
            ), axis=0)
            train_next = np.concatenate((
                common.event(np.asarray(train_a[ar, q + 1, destination], np.float32) - np.asarray(train_a[al, q + 1, destination], np.float32), thresholds[q + 1]),
                common.event(np.asarray(train_b[br, q + 1, destination], np.float32) - np.asarray(train_b[bl, q + 1, destination], np.float32), thresholds[q + 1]),
            ), axis=0)
            states_count = 3 ** len(SOURCE_ROLES)
            fitted, _key, _support = one_step.fit_map(
                partition.code_word(train_current, SOURCE_ROLES), train_next, states_count, 4, 0.70
            )
            for registered in parent_row["fifth_selected_pairs"]:
                left_id = int(registered["left_id"])
                right_id = int(registered["right_id"])
                current_delta = np.asarray(test_states[right_id, common.CANONICAL_NEW_INDICES[q]], np.float32) - np.asarray(
                    test_states[left_id, common.CANONICAL_NEW_INDICES[q]], np.float32
                )
                current_event = common.event(current_delta, thresholds[q])[None, ...]
                pure = one_step.lookup(fitted, partition.code_word(current_event, SOURCE_ROLES), states_count)[0]
                eligible = (current_event[0, destination] == 0) & (pure != 0)
                if int(eligible.sum()) != int(registered["eligible_coordinates"]):
                    raise RuntimeError((family, registered, int(eligible.sum())))
                row = compiled[right_id]
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                replacement = {
                    ri: {
                        "left": np.asarray(test_states[left_id, common.CANONICAL_NEW_INDICES[q], ri], np.float32),
                        "right": np.asarray(test_states[right_id, common.CANONICAL_NEW_INDICES[q], ri], np.float32),
                    }
                    for ri in range(6)
                }
                condition_rows = {}
                for condition in CONDITIONS:
                    captured = []
                    eligible_t = torch.tensor(eligible, dtype=torch.bool, device=device)

                    def patch_hook(_module, _args, output):
                        if condition == "natural":
                            return output
                        value = output.clone()
                        if condition == "wrong_role_rescue":
                            operations = [(ri, "left") for ri in SOURCE_ROLES] + [
                                (common.ROLES.index("secondary"), "right"),
                                (common.ROLES.index("context"), "right"),
                            ]
                        else:
                            operations = [(ri, condition) for ri in SOURCE_ROLES]
                        for ri, operation in operations:
                            if operation in ("delete", "left"):
                                donor = replacement[ri]["left"]
                            elif operation == "coordinate_roll_rescue":
                                donor = replacement[ri]["left"] + np.roll(replacement[ri]["right"] - replacement[ri]["left"], 1)
                            else:
                                donor = replacement[ri]["right"]
                            donor_t = torch.tensor(donor, dtype=value.dtype, device=device)
                            for position in row["role_positions"][common.ROLES[ri]]:
                                updated = value[0, position].clone()
                                updated[eligible_t] = donor_t[eligible_t]
                                value[0, position] = updated
                        return value

                    def next_hook(_module, _args, output):
                        value = output[0] if isinstance(output, tuple) else output
                        positions = row["role_positions"][common.ROLES[destination]]
                        captured.append(value[0, positions].mean(0).float().cpu().numpy())

                    patch = base.layers[q - 1].register_forward_hook(patch_hook)
                    capture = base.layers[q].register_forward_hook(next_hook)
                    try:
                        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    finally:
                        patch.remove()
                        capture.remove()
                    logits = np.asarray([
                        float(output.logits[0, ids.shape[1] - 1, candidate_ids[0]]) for candidate_ids in row["candidate_ids"]
                    ])
                    gold = int(row["gold_position"])
                    next_delta = captured[0] - np.asarray(
                        test_states[left_id, common.CANONICAL_NEW_INDICES[q + 1], destination], np.float32
                    )
                    next_event = common.event(next_delta, thresholds[q + 1])
                    condition_rows[condition] = {
                        "successor_target_rate": float(np.mean(next_event[eligible] == pure[eligible])),
                        "candidate_margin": float(logits[gold] - logits[1 - gold]),
                        "prediction": int(logits[1] > logits[0]),
                    }
                    del output
                natural = condition_rows["natural"]["successor_target_rate"]
                deleted = condition_rows["delete"]["successor_target_rate"]
                correct = condition_rows["correct_rescue"]["successor_target_rate"]
                wrong = max(
                    condition_rows["coordinate_roll_rescue"]["successor_target_rate"],
                    condition_rows["wrong_role_rescue"]["successor_target_rate"],
                )
                sample = {
                    "family": family,
                    "surface": registered["surface"],
                    "q": q,
                    "destination_role": common.ROLES[destination],
                    "left_id": left_id,
                    "right_id": right_id,
                    "eligible_coordinates": int(eligible.sum()),
                    "conditions": condition_rows,
                    "deletion_drop": float(natural - deleted),
                    "correct_recovery_error": float(abs(correct - natural)),
                    "correct_minus_best_wrong": float(correct - wrong),
                }
                sample_rows.append(sample)
                print(f"[C291] {family}/{registered['surface']} q{q}/{common.ROLES[destination]} n={eligible.sum()} drop={sample['deletion_drop']:+.4f} rescue={sample['correct_minus_best_wrong']:+.4f}", flush=True)

        core.write_rows(OUT / "raw/sample_results.jsonl", sample_rows)
        family_results = []
        for family in passing:
            rows = [row for row in sample_rows if row["family"] == family]
            aggregate = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("deletion_drop", "correct_recovery_error", "correct_minus_best_wrong")
            }
            aggregate.update({
                "family": family,
                "samples": len(rows),
                "eligible_coordinates_mean": float(np.mean([row["eligible_coordinates"] for row in rows])),
            })
            aggregate["family_gate_passed"] = bool(
                aggregate["deletion_drop"] >= gates["causal_flip_min"]
                and aggregate["correct_recovery_error"] <= 0.05
                and aggregate["correct_minus_best_wrong"] >= gates["causal_control_margin_min"]
            )
            family_results.append(aggregate)
        passing_causal = [row["family"] for row in family_results if row["family_gate_passed"]]
        report = {
            "phase": 1825,
            "campaign": "C291",
            "status": "training_qualified_local_causal_adjudicated",
            "eligible_families": list(passing),
            "families": family_results,
            "causal_families_passing": passing_causal,
            "causal_families_passing_count": len(passing_causal),
            "placement": placement,
            "quantization": quantization,
            "strict_interpretation": "Only C290-qualified branches were tested. Passing would establish local use of a broad rule-defined coordinate coalition, not uniqueness or a complete circuit; failure rejects this intervention interface for that branch, not the observational event law.",
            "next_authorization": "campaign_adjudication_after_targeted_causal_test",
        }
        core.save(OUT / "analysis/summary.json", report)
        audit_checks = {
            "all_authorized_families_run": set(row["family"] for row in sample_rows) == set(passing),
            "two_surfaces_each": all(sum(row["family"] == family for row in sample_rows) == 2 for family in passing),
            "eligibility_preserved": all(row["eligible_coordinates"] >= 16 for row in sample_rows),
            "finite": bool(np.isfinite([
                row[key] for row in family_results for key in ("deletion_drop", "correct_recovery_error", "correct_minus_best_wrong")
            ]).all()),
        }
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
        final_checks = {
            "contract": all(checks.values()),
            "analysis": all(audit_checks.values()),
            "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        }
        final = {
            "phase": 1825,
            "campaign": "C291",
            "status": "closed",
            "checks": final_checks,
            "all_checks_passed": all(final_checks.values()),
            "headline": report,
            "next_authorization": report["next_authorization"],
        }
        core.save(OUT / "analysis/final.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2))
    finally:
        common.model_base.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
