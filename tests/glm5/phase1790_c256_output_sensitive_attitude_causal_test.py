#!/usr/bin/env python3
"""C256: output-sensitive attitude readout and bidirectional path intervention."""
from __future__ import annotations

import gc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.RESULT / "phase1790_c256_output_sensitive_attitude_causal_test"
CHECKPOINTS = (8, 16, 24, 32)
CONDITIONS = ("correct_path", "wrong_family", "coordinate_roll", "sign_reverse", "reversed_checkpoint_masks")


def material() -> list[dict]:
    rows = []
    for surface, unit, a, b, order in itertools.product(common.SURFACES, range(8), (0, 1), (0, 1), (1, -1)):
        u = common.UNITS[unit]
        relation = "approves of" if a == 0 else "doubts"
        target = f"{u['observer']} {relation} the account that {u['primary']} examined the {u['object']}." if b == 0 else f"{u['observer']} {relation} the account that the {u['object']} was examined by {u['primary']}."
        noise = f"{u['secondary']} moved the {u['other']}."
        question = f"What is {u['observer']}'s stance toward the account?"
        correct, wrong = ("approval", "doubt") if a == 0 else ("doubt", "approval")
        choices, gold = common.options(correct, wrong, order)
        prompt_core = common.wrap(surface, target, noise, question)
        rows.append({
            "case_id": f"c256-{surface}-u{unit}-{a}{b}-{order:+d}", "family": "attitude_event", "surface": surface, "unit": unit,
            "factor_a": a, "factor_b": b, "order": order, "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong,
            "prompt_core": prompt_core, "prompt": f"{prompt_core} {choices}. Reply with only A or B.",
            "role_values": {"primary": u["primary"], "secondary": u["secondary"], "relation": relation, "context": u["object"], "query": u["observer"]},
        })
    return rows


def compile_rows(tokenizer, rows):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    system = "Answer only from the supplied text. Do not use outside knowledge."
    result = []
    for row in rows:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        result.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return result


@torch.inference_mode()
def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C255"] / "audit/independent_final_audit.json")
    tokenizer = common.graph_base.tokenizer()
    rows = material()
    compiled = compile_rows(tokenizer, rows)
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C256"), "rows": len(rows) == 128,
        "balance": sum(r["gold_position"] == 0 for r in rows) == 64, "width": max(len(r["prompt_ids"]) for r in compiled) <= 128,
        "output_sensitive": all(r["correct_answer"] == ("approval" if r["factor_a"] == 0 else "doubt") for r in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1790, "campaign": "C256", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "output_sensitive_causal_test_frozen",
        "object": "same attitude factor now determines the correct approval/doubt answer", "rows": 128,
        "behavior_gate": {"accuracy_min": 0.90}, "event_transfer_gate": {"stable_signed_jaccard_min": 0.08, "best_control_margin_min": 0.02},
        "causal_gate": {"correct_donor_margin_gain_min": 0.0, "best_control_margin_min": 0.10, "correct_flip_rate_min": 0.20},
        "checkpoints": list(CHECKPOINTS), "controls": list(CONDITIONS[1:]),
        "claim_boundary": "A pass is distributed intervention evidence for this controlled readout. It is not minimality, natural-language necessity, or a universal attitude circuit.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "capture_role_states_then_test_frozen_core_then_conditionally_intervene",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    thresholds = np.asarray(core.load(common.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    model = None
    started = time.time()
    try:
        model, _tokenizer, device, placement = common.previous.load_bf16("qwen3")
        states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(128, 37, 6, 2560))
        behavior = []
        for i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
            logits = [float(output.logits[0, -1, candidate[0]]) for candidate in row["candidate_ids"]]
            prediction = int(logits[1] > logits[0])
            behavior.append({"case_id": row["case_id"], "prediction": prediction, "gold_position": row["gold_position"], "correct": prediction == row["gold_position"], "logits": logits})
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES):
                    states[i, q, ri] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            if i % 16 == 0 or i == 127:
                states.flush(); print(f"[C256] clean states {i + 1}/128", flush=True)
        core.write_rows(OUT / "raw/behavior.jsonl", behavior)
        accuracy = float(np.mean([r["correct"] for r in behavior]))
        behavior_eligible = accuracy >= 0.90
        key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}
        event_groups = []
        for surface, unit, order in itertools.product(common.SURFACES, range(8), (1, -1)):
            needed = [(surface, unit, a, b, order) for a, b in itertools.product((0, 1), repeat=2)]
            if not all(behavior[key[item]]["correct"] for item in needed):
                continue
            cells = {(a, b): np.asarray(states[key[(surface, unit, a, b, order)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
            effect = common.factorial_effect(cells)[0]
            event_groups.append(np.where(effect > thresholds[:, None, None], 1, np.where(effect < -thresholds[:, None, None], -1, 0)).astype(np.int8))
        current = np.asarray(event_groups)
        up, down = np.mean(current == 1, axis=0), np.mean(current == -1, axis=0)
        active = up + down
        dominant = np.where(up >= down, 1, -1)
        stable = np.where((active >= .75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= .80), dominant, 0).astype(np.int8)
        frozen = np.asarray(tri[common.FAMILIES.index("attitude_event"), 0])
        observed = common.signed_jaccard(frozen, stable)
        wrong = common.signed_jaccard(np.asarray(tri[common.FAMILIES.index("type_graph"), 0]), stable)
        rolled = max(common.signed_jaccard(np.roll(frozen, shift, axis=-1), stable) for shift in (137, 389, 743, 1291))
        sign_flip = common.signed_jaccard(-frozen, stable)
        event_margin = observed - max(wrong, rolled, sign_flip)
        event_eligible = behavior_eligible and observed >= .08 and event_margin >= .02

        intervention_rows = []
        if event_eligible:
            layers = model.model.layers
            for surface, unit, direction in itertools.product(common.SURFACES, range(8), ((0, 1), (1, 0))):
                target_a, donor_a = direction
                target_i = key[(surface, unit, target_a, 0, 1)]
                donor_i = key[(surface, unit, donor_a, 0, 1)]
                target, donor = compiled[target_i], compiled[donor_i]
                ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device)
                clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                donor_gold = donor["gold_position"]
                other = 1 - donor_gold
                clean_margin = float(clean.logits[0, -1, target["candidate_ids"][donor_gold][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]])
                donor_final = np.asarray(states[donor_i, -1, common.ROLES.index("boundary")], np.float32)
                target_final = np.asarray(states[target_i, -1, common.ROLES.index("boundary")], np.float32)
                base_distance = float(np.linalg.norm(target_final - donor_final))
                for condition in CONDITIONS:
                    handles = []
                    for q in CHECKPOINTS:
                        source_q = 40 - q if condition == "reversed_checkpoint_masks" else q
                        def make_hook(q=q, source_q=source_q, condition=condition):
                            def hook(_module, _inputs, output):
                                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                                for ri, role in enumerate(common.ROLES):
                                    family_i = common.FAMILIES.index("type_graph") if condition == "wrong_family" else common.FAMILIES.index("attitude_event")
                                    mask = np.asarray(tri[family_i, 0, source_q, ri] != 0)
                                    if condition == "coordinate_roll": mask = np.roll(mask, 137)
                                    coords = np.flatnonzero(mask)
                                    if not coords.size: continue
                                    donor_value = np.asarray(states[donor_i, q, ri], np.float32)
                                    coord_t = torch.tensor(coords, dtype=torch.long, device=hidden.device)
                                    value_t = torch.tensor(donor_value[coords], dtype=hidden.dtype, device=hidden.device)
                                    for pos in target["role_positions"][role]:
                                        hidden[0, pos, coord_t] = 2 * hidden[0, pos, coord_t] - value_t if condition == "sign_reverse" else value_t
                                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                            return hook
                        handles.append(layers[q - 1].register_forward_hook(make_hook()))
                    try:
                        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
                    finally:
                        for handle in handles: handle.remove()
                    margin = float(output.logits[0, -1, target["candidate_ids"][donor_gold][0]] - output.logits[0, -1, target["candidate_ids"][other][0]])
                    final = output.hidden_states[-1][0, -1].float().cpu().numpy()
                    distance = float(np.linalg.norm(final - donor_final))
                    intervention_rows.append({"surface": surface, "unit": unit, "direction": f"{target_a}_to_{donor_a}", "condition": condition, "donor_margin_gain": margin - clean_margin, "flipped_to_donor": margin > 0, "trajectory_improvement": (base_distance - distance) / max(base_distance, 1e-12)})
                print(f"[C256] causal {surface} unit {unit + 1}/8 direction {target_a}->{donor_a}", flush=True)
        core.write_rows(OUT / "analysis/intervention_rows.jsonl", intervention_rows)
        medians = {condition: float(np.median([r["donor_margin_gain"] for r in intervention_rows if r["condition"] == condition])) for condition in CONDITIONS} if intervention_rows else {}
        flip_rates = {condition: float(np.mean([r["flipped_to_donor"] for r in intervention_rows if r["condition"] == condition])) for condition in CONDITIONS} if intervention_rows else {}
        causal_margin = medians.get("correct_path", -1e9) - max((medians.get(name, -1e9) for name in CONDITIONS[1:]), default=-1e9)
        causal_passed = bool(intervention_rows) and medians["correct_path"] > 0 and causal_margin >= .10 and flip_rates["correct_path"] >= .20
        report = {"phase": 1790, "campaign": "C256", "status": "adjudicated", "behavior_accuracy": accuracy, "behavior_eligible": behavior_eligible, "event_groups": len(event_groups), "event_transfer_signed_jaccard": observed, "event_controls": {"wrong_family": wrong, "coordinate_roll_max": rolled, "sign_flip": sign_flip}, "event_margin": event_margin, "event_eligible": event_eligible, "intervention_rows": len(intervention_rows), "condition_margin_gain_medians": medians, "condition_flip_rates": flip_rates, "correct_vs_best_control_margin": causal_margin, "output_causal_gate_passed": causal_passed, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": "This is a bidirectional controlled approval/doubt readout. A pass supports distributed output control by the frozen event band; it still does not establish minimality, natural necessity, or a universal attitude representation.", "next_authorization": "C257_independent_output_sensitive_natural_paraphrase_if_pass_else_route_local_redesign"}
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {"behavior_rows": len(behavior) == 128, "event_groups": len(event_groups) >= 28, "intervention_typed": (len(intervention_rows) == 160) if event_eligible else (len(intervention_rows) == 0), "finite": bool(np.isfinite([accuracy, observed, wrong, rolled, sign_flip, event_margin, causal_margin]).all())}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1790, "campaign": "C256", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
        core.save(OUT / "analysis/final.json", final)
        core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
        print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
