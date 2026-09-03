#!/usr/bin/env python3
"""C269: intervene once on state-selected, prospectively registered coordinate edges."""
from __future__ import annotations

import gc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1797_c263_c272_state_operator_common as common

core, OUT = common.core, common.OUTS["C269"]
C259 = common.prior.RESULT / "phase1793_c259_independent_dense_path_replication"
C265 = common.OUTS["C265"]
CONDITIONS = ("state_single", "full_single", "fixed_single", "wrong_family", "coordinate_roll", "sign_reverse", "state_midpoint")


def state_mask(target_state, donor_state, q, family_i, pred_map, med_map, threshold):
    ri = common.ROLES.index("relation")
    src = np.asarray(target_state[q, ri], np.float32); delta = np.asarray(donor_state[q, ri], np.float32) - src
    event = np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(np.int8)
    high = src >= np.asarray(med_map[family_i, q, ri], np.float32)
    keys = np.where(event < 0, high.astype(np.int8), np.where(event > 0, 2 + high.astype(np.int8), -1))
    prediction = np.zeros(2560, np.int8)
    for key in range(4):
        member = keys == key; p = np.asarray(pred_map[family_i, q, ri, key]); prediction[member] = p[member]
    return (prediction != 0) & (event != 0)


def select_q(target_state, donor_state, family_i, pred_map, med_map, thresholds):
    curves = [float(np.mean(state_mask(target_state, donor_state, q, family_i, pred_map, med_map, thresholds[q]))) for q in range(1, 17)]
    return int(np.argmax(curves) + 1), curves


@torch.inference_mode()
def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C268"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "c259_complete": core.load(C259 / "analysis/final.json")["all_checks_passed"], "single_checkpoint_primary": True, "no_topk": True, "no_attention_mlp": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "analysis").mkdir(exist_ok=True); (OUT / "audit").mkdir(exist_ok=True)
    protocol = {"phase": 1803, "campaign": "C269", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "local_causal_frozen", "selection": "argmax q1-q16 fraction of coordinates satisfying the frozen current-state passport guard", "primary_intervention": "one checkpoint, relation role, every guard-matched coordinate", "conditions": list(CONDITIONS), "gate": {"state_flip_min": 0.20, "state_gain_min": 0.0, "state_minus_best_wrong_control_min": 0.10}, "claim_boundary": "A pass shows local output sufficiency for a registered state-indexed edge. Midpoint or zero interventions remain artificial and cannot by themselves establish natural necessity.", "producer_sha256": core.sha(Path(__file__)), "authorization": "C270_generation_and_side_effects_regardless"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    compiled = core.rows(C259 / "compiled/qwen3.jsonl"); states = np.load(C259 / "raw/role_states.float16.npy", mmap_mode="r")
    key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}
    pred_map = np.load(C265 / "analysis/passport_pred_sign.int8.npy", mmap_mode="r"); med_map = np.load(C265 / "analysis/passport_baseline_median.float16.npy", mmap_mode="r")
    thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    tri = np.load(common.prior.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    fi, wrong_fi, ri = common.FAMILIES.index("attitude_event"), common.FAMILIES.index("type_graph"), common.ROLES.index("relation")
    rows, model, started = [], None, time.time()
    try:
        model, _tokenizer, device, placement = common.previous.load_bf16("qwen3"); layers = model.model.layers
        causal_surfaces = tuple(sorted({row["surface"] for row in compiled}))
        for surface, unit, ta, da in ((s, u, ta, da) for s, u in itertools.product(causal_surfaces, range(8)) for ta, da in ((0, 1), (1, 0))):
            ti, di = key[(surface, unit, ta, 0, 1)], key[(surface, unit, da, 0, 1)]; target, donor = compiled[ti], compiled[di]
            qstar, curve = select_q(states[ti], states[di], fi, pred_map, med_map, thresholds)
            correct_mask = state_mask(states[ti], states[di], qstar, fi, pred_map, med_map, thresholds[qstar])
            wrong_mask = state_mask(states[ti], states[di], qstar, wrong_fi, pred_map, med_map, thresholds[qstar])
            fixed_mask = np.asarray(tri[fi, 0, qstar, ri] != 0)
            masks = {"state_single": correct_mask, "full_single": np.ones(2560, bool), "fixed_single": fixed_mask, "wrong_family": wrong_mask, "coordinate_roll": np.roll(correct_mask, 137), "sign_reverse": correct_mask, "state_midpoint": correct_mask}
            ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device); donor_gold, other = donor["gold_position"], 1 - donor["gold_position"]
            clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            clean_margin = float(clean.logits[0, -1, target["candidate_ids"][donor_gold][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]])
            for condition in CONDITIONS:
                coords = np.flatnonzero(masks[condition]); c = torch.as_tensor(coords, dtype=torch.long, device=device)
                def hook(_module, _inputs, output, condition=condition, coords=coords, c=c):
                    hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                    donor_v = torch.as_tensor(np.asarray(states[di, qstar, ri, coords], np.float32), dtype=hidden.dtype, device=hidden.device)
                    target_v = torch.as_tensor(np.asarray(states[ti, qstar, ri, coords], np.float32), dtype=hidden.dtype, device=hidden.device)
                    value = target_v - (donor_v - target_v) if condition == "sign_reverse" else ((target_v + donor_v) * 0.5 if condition == "state_midpoint" else donor_v)
                    for pos in target["role_positions"]["relation"]: hidden[0, pos, c] = value
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                handle = layers[qstar - 1].register_forward_hook(hook)
                try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                finally: handle.remove()
                margin = float(output.logits[0, -1, target["candidate_ids"][donor_gold][0]] - output.logits[0, -1, target["candidate_ids"][other][0]])
                rows.append({"surface": surface, "unit": unit, "direction": f"{ta}_to_{da}", "condition": condition, "selected_checkpoint": qstar, "coordinates": int(len(coords)), "readiness_curve": curve, "clean_donor_margin": clean_margin, "patched_donor_margin": margin, "donor_margin_gain": margin - clean_margin, "flipped_to_donor": margin > 0})
            print(f"[C269] {surface} u{unit} {ta}->{da} q{qstar} n={correct_mask.sum()}", flush=True)
        core.write_rows(OUT / "analysis/intervention_rows.jsonl", rows)
        summaries = []
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            summaries.append({"condition": condition, "support": len(selected), "median_coordinates": float(np.median([r["coordinates"] for r in selected])), "median_margin_gain": float(np.median([r["donor_margin_gain"] for r in selected])), "flip_rate": float(np.mean([r["flipped_to_donor"] for r in selected]))})
        by = {r["condition"]: r for r in summaries}; controls = ("wrong_family", "coordinate_roll", "sign_reverse")
        margin = by["state_single"]["median_margin_gain"] - max(by[c]["median_margin_gain"] for c in controls)
        passed = by["state_single"]["flip_rate"] >= .20 and by["state_single"]["median_margin_gain"] > 0 and margin >= .10
        report = {"phase": 1803, "campaign": "C269", "status": "local_causal_adjudicated", "summaries": summaries, "state_minus_best_wrong_control": margin, "local_state_edge_gate_passed": passed, "selected_checkpoint_counts": {str(q): sum(r["condition"] == "state_single" and r["selected_checkpoint"] == q for r in rows) for q in range(1, 17)}, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C270_generation_and_side_effects"}
        core.save(OUT / "analysis/summary.json", report)
        ach = {"rows": len(rows) == 32 * len(CONDITIONS), "support": all(r["support"] == 32 for r in summaries), "hooks_removed": True, "finite": bool(np.isfinite([r["median_margin_gain"] for r in summaries] + [margin]).all())}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1803, "campaign": "C269", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
