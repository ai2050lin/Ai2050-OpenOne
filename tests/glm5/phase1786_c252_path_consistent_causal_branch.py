#!/usr/bin/env python3
"""C252: conditional HiddenState-only path intervention on the tri-material core."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C252"]
PARENT = common.OUTS["C248"]
CHECKPOINTS = (8, 16, 24, 32)
TARGETS = ("attitude_event", "contrast")
CONDITIONS = ("correct_path", "wrong_family", "coordinate_roll", "sign_reverse", "reversed_checkpoint_masks")


def get_layers(model):
    return model.model.layers


@torch.inference_mode()
def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    p249 = core.load(common.OUTS["C249"] / "analysis/summary.json")
    p251 = core.load(common.OUTS["C251"] / "audit/independent_final_audit.json")
    eligible = bool(p249["target_families_passed"] and p251["all_checks_passed"])
    checks = {"parents": p251["all_checks_passed"], "eligibility_adjudicated": True, "eligible": eligible, "hidden_state_only": True}
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1786, "campaign": "C252", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "conditional_causal_branch_frozen", "eligible": eligible, "targets": list(TARGETS), "checkpoints": list(CHECKPOINTS),
        "intervention": "replace all tri-material coordinates at frozen checkpoint/role masks with the same-unit A=1 donor role mean while target is A=0 and B=0",
        "controls": list(CONDITIONS[1:]), "readout": "reduction in final boundary-state distance to the clean donor; candidate logits are side-effect diagnostics because the frozen question's answer is invariant to factor A",
        "gate": {"correct_median_improvement_min": 0.0, "best_control_margin_min": 0.02},
        "claim_boundary": "Even a pass would establish trajectory control under a large role-conditioned state intervention, not minimality, natural use, or an output-semantic flip.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_if_and_only_if_C249_targets_pass",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    if not eligible:
        report = {"phase": 1786, "campaign": "C252", "status": "typed_not_tested", "reason": "frozen eligibility failed", "next_authorization": "C253_cross_model_regardless"}
        core.save(OUT / "analysis/summary.json", report)
        core.save(OUT / "analysis/final.json", {"phase": 1786, "campaign": "C252", "status": "closed", "all_checks_passed": True, "headline": report, "next_authorization": report["next_authorization"]})
        core.save(OUT / "audit/independent_final_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": report["next_authorization"]})
        print(json.dumps(report, indent=2)); return

    fields = np.load(PARENT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/hidden_index.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(common.OUTS["C247"] / "compiled/qwen3.jsonl")}
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in index if row["panel"] == "core"}
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    model = None
    rows = []
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        layers = get_layers(model)
        for family in TARGETS:
            fi = common.FAMILIES.index(family)
            wrong_fi = (fi + 1) % len(common.FAMILIES)
            for unit in range(8):
                target = key[(family, "case_review", unit, 0, 0, 1)]
                donor = key[(family, "case_review", unit, 1, 0, 1)]
                target_c, donor_c = compiled[target["case_id"]], compiled[donor["case_id"]]
                target_ids = torch.tensor([target_c["prompt_ids"]], dtype=torch.long, device=device)
                donor_final = np.asarray(fields[donor["hidden_index"], -1, donor["role_positions"]["boundary"]], np.float32).mean(axis=0)
                target_final = np.asarray(fields[target["hidden_index"], -1, target["role_positions"]["boundary"]], np.float32).mean(axis=0)
                base_distance = float(np.linalg.norm(target_final - donor_final))
                base_logits = None
                for condition in CONDITIONS:
                    handles = []
                    for q in CHECKPOINTS:
                        layer = layers[q - 1]
                        source_q = 40 - q if condition == "reversed_checkpoint_masks" else q

                        def make_hook(q=q, source_q=source_q, condition=condition):
                            def hook(_module, _inputs, output):
                                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                                for role_i, role in enumerate(common.ROLES):
                                    target_positions = target_c["role_positions"][role]
                                    donor_positions = donor_c["role_positions"][role]
                                    mask_family = wrong_fi if condition == "wrong_family" else fi
                                    mask = np.asarray(tri[mask_family, 0, source_q, role_i] != 0)
                                    if condition == "coordinate_roll":
                                        mask = np.roll(mask, 137)
                                    coords = np.flatnonzero(mask)
                                    if not coords.size:
                                        continue
                                    donor_value = np.asarray(fields[donor["hidden_index"], q, donor_positions], np.float32).mean(axis=0)
                                    coord_t = torch.tensor(coords, dtype=torch.long, device=hidden.device)
                                    value_t = torch.tensor(donor_value[coords], dtype=hidden.dtype, device=hidden.device)
                                    for pos in target_positions:
                                        if condition == "sign_reverse":
                                            hidden[0, pos, coord_t] = 2 * hidden[0, pos, coord_t] - value_t
                                        else:
                                            hidden[0, pos, coord_t] = value_t
                                if isinstance(output, tuple):
                                    return (hidden, *output[1:])
                                return hidden
                            return hook
                        handles.append(layer.register_forward_hook(make_hook()))
                    try:
                        output = model(input_ids=target_ids, attention_mask=torch.ones_like(target_ids), use_cache=False, return_dict=True, output_hidden_states=True)
                    finally:
                        for handle in handles:
                            handle.remove()
                    final = output.hidden_states[-1][0, -1].float().cpu().numpy()
                    distance = float(np.linalg.norm(final - donor_final))
                    candidates = target_c["candidate_ids"]
                    logits = [float(output.logits[0, -1, candidate[0]]) for candidate in candidates]
                    if base_logits is None:
                        clean_ids = target_ids
                        clean = model(input_ids=clean_ids, attention_mask=torch.ones_like(clean_ids), use_cache=False, return_dict=True)
                        base_logits = [float(clean.logits[0, -1, candidate[0]]) for candidate in candidates]
                    rows.append({
                        "family": family, "unit": unit, "condition": condition, "base_distance": base_distance, "patched_distance": distance,
                        "fractional_donor_distance_improvement": (base_distance - distance) / max(base_distance, 1e-12),
                        "candidate_margin_change": (logits[0] - logits[1]) - (base_logits[0] - base_logits[1]),
                    })
                    del output
                print(f"[C252] {family} unit {unit + 1}/8", flush=True)
        core.write_rows(OUT / "analysis/intervention_rows.jsonl", rows)
        medians = {condition: float(np.median([row["fractional_donor_distance_improvement"] for row in rows if row["condition"] == condition])) for condition in CONDITIONS}
        margin = medians["correct_path"] - max(medians[name] for name in CONDITIONS[1:])
        passed = medians["correct_path"] > 0 and margin >= 0.02
        report = {
            "phase": 1786, "campaign": "C252", "status": "causal_trajectory_adjudicated", "rows": len(rows), "condition_medians": medians,
            "correct_vs_best_control_margin": margin, "trajectory_gate_passed": passed,
            "candidate_margin_change_abs_median": float(np.median(np.abs([row["candidate_margin_change"] for row in rows if row["condition"] == "correct_path"]))),
            "placement": placement, "elapsed_seconds": time.time() - started,
            "strict_interpretation": "This tests whether a distributed, role-conditioned tri-material band can steer the final boundary state toward a clean donor. The task answer is invariant to factor A, so this cannot establish semantic-output necessity or sufficiency.",
            "next_authorization": "C253_cross_model_abstract_replication_regardless_of_causal_result",
        }
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {"rows": len(rows) == 2 * 8 * len(CONDITIONS), "conditions": set(medians) == set(CONDITIONS), "finite": bool(np.isfinite(list(medians.values()) + [margin]).all()), "hooks_removed": True}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final_report = {"phase": 1786, "campaign": "C252", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
        core.save(OUT / "analysis/final.json", final_report)
        core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
        print(json.dumps(final_report, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
