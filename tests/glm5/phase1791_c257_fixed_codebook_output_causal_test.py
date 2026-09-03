#!/usr/bin/env python3
"""C257: fix C256's outcome-relative label bug and rerun output-sensitive causality."""
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
OUT = common.RESULT / "phase1791_c257_fixed_codebook_output_causal_test"
CHECKPOINTS = (8, 16, 24, 32)
CONDITIONS = ("correct_path", "wrong_family", "coordinate_roll", "sign_reverse", "reversed_checkpoint_masks")


def material() -> list[dict]:
    rows = []
    for surface, unit, a, b, order in itertools.product(common.SURFACES, range(8), (0, 1), (0, 1), (1, -1)):
        u = common.UNITS[unit]
        relation = "approves of" if a == 0 else "doubts"
        target = f"{u['observer']} {relation} the account that {u['primary']} examined the {u['object']}." if b == 0 else f"{u['observer']} {relation} the account that the {u['object']} was examined by {u['primary']}."
        prompt_core = common.wrap(surface, target, f"{u['secondary']} moved the {u['other']}.", f"What is {u['observer']}'s stance toward the account?")
        if order == 1:
            choices, gold = "(A) approval (B) doubt", (0 if a == 0 else 1)
        else:
            choices, gold = "(A) doubt (B) approval", (1 if a == 0 else 0)
        rows.append({"case_id": f"c257-{surface}-u{unit}-{a}{b}-{order:+d}", "family": "attitude_event", "surface": surface, "unit": unit, "factor_a": a, "factor_b": b, "order": order, "gold_position": gold, "prompt": f"{prompt_core} {choices}. Reply with only A or B.", "role_values": {"primary": u["primary"], "secondary": u["secondary"], "relation": relation, "context": u["object"], "query": u["observer"]}})
    return rows


def compile_rows(tokenizer, rows):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    result = []
    for row in rows:
        ids = core.chat_ids(tokenizer, "Answer only from the supplied text. Do not use outside knowledge.", row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids, value)
            if not spans: raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        result.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return result


@torch.inference_mode()
def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.RESULT / "phase1790_c256_output_sensitive_attitude_causal_test/analysis/final.json")
    old_rows = core.rows(common.RESULT / "phase1790_c256_output_sensitive_attitude_causal_test/material/cases.jsonl")
    old_label_invariant = all(next(r["gold_position"] for r in old_rows if r["surface"] == surface and r["unit"] == unit and r["factor_a"] == 0 and r["factor_b"] == b and r["order"] == order) == next(r["gold_position"] for r in old_rows if r["surface"] == surface and r["unit"] == unit and r["factor_a"] == 1 and r["factor_b"] == b and r["order"] == order) for surface, unit, b, order in itertools.product(common.SURFACES, range(8), (0, 1), (1, -1)))
    rows = material()
    compiled = compile_rows(common.graph_base.tokenizer(), rows)
    fixed_label_changes = all(next(r["gold_position"] for r in rows if r["surface"] == surface and r["unit"] == unit and r["factor_a"] == 0 and r["factor_b"] == b and r["order"] == order) != next(r["gold_position"] for r in rows if r["surface"] == surface and r["unit"] == unit and r["factor_a"] == 1 and r["factor_b"] == b and r["order"] == order) for surface, unit, b, order in itertools.product(common.SURFACES, range(8), (0, 1), (1, -1)))
    checks = {"parent_closed": parent["all_checks_passed"], "c256_label_bug_reproduced": old_label_invariant, "fixed_label_changes_with_factor": fixed_label_changes, "rows": len(rows) == 128, "balance": sum(r["gold_position"] == 0 for r in rows) == 64, "width": max(len(r["prompt_ids"]) for r in compiled) <= 128}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "raw").mkdir(); (OUT / "analysis").mkdir()
    protocol = {"phase": 1791, "campaign": "C257", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "fixed_codebook_causal_test_frozen", "c256_amendment": "C256 changed answer words but kept the correct A/B label invariant within candidate order; its output-causal statistics are invalid and are reclassified as not tested.", "codebook": {"order_+1": "A=approval, B=doubt", "order_-1": "A=doubt, B=approval"}, "behavior_gate": {"accuracy_min": .90}, "event_gate": {"signed_jaccard_min": .08, "control_margin_min": .02}, "causal_gate": {"correct_margin_gain_min": 0.0, "best_control_margin_min": .10, "flip_rate_min": .20}, "checkpoints": list(CHECKPOINTS), "controls": list(CONDITIONS[1:]), "producer_sha256": core.sha(Path(__file__)), "authorization": "run_fixed_codebook_once"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.write_rows(OUT / "material/cases.jsonl", rows); core.write_rows(OUT / "compiled/qwen3.jsonl", compiled); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    thresholds = np.asarray(core.load(common.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    model = None; started = time.time()
    try:
        model, _tokenizer, device, placement = common.previous.load_bf16("qwen3")
        states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(128, 37, 6, 2560)); behavior = []
        for i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device); output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
            logits = [float(output.logits[0, -1, c[0]]) for c in row["candidate_ids"]]; prediction = int(logits[1] > logits[0]); behavior.append({"case_id": row["case_id"], "correct": prediction == row["gold_position"], "prediction": prediction, "gold_position": row["gold_position"], "logits": logits})
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES): states[i, q, ri] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            if i % 16 == 0 or i == 127: states.flush(); print(f"[C257] clean states {i + 1}/128", flush=True)
        core.write_rows(OUT / "raw/behavior.jsonl", behavior); accuracy = float(np.mean([r["correct"] for r in behavior])); behavior_ok = accuracy >= .90
        key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}; event_groups = []
        for surface, unit, order in itertools.product(common.SURFACES, range(8), (1, -1)):
            needed = [(surface, unit, a, b, order) for a, b in itertools.product((0, 1), repeat=2)]
            if not all(behavior[key[item]]["correct"] for item in needed): continue
            cells = {(a, b): np.asarray(states[key[(surface, unit, a, b, order)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}; effect = common.factorial_effect(cells)[0]
            event_groups.append(np.where(effect > thresholds[:, None, None], 1, np.where(effect < -thresholds[:, None, None], -1, 0)).astype(np.int8))
        current = np.asarray(event_groups); up, down = np.mean(current == 1, axis=0), np.mean(current == -1, axis=0); active = up + down; dominant = np.where(up >= down, 1, -1); stable = np.where((active >= .75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= .80), dominant, 0).astype(np.int8)
        fi = common.FAMILIES.index("attitude_event"); frozen = np.asarray(tri[fi, 0]); observed = common.signed_jaccard(frozen, stable); wrong = common.signed_jaccard(np.asarray(tri[common.FAMILIES.index("type_graph"), 0]), stable); rolled = max(common.signed_jaccard(np.roll(frozen, s, axis=-1), stable) for s in (137, 389, 743, 1291)); sign_flip = common.signed_jaccard(-frozen, stable); event_margin = observed - max(wrong, rolled, sign_flip); event_ok = behavior_ok and observed >= .08 and event_margin >= .02
        intervention = []
        if event_ok:
            layers = model.model.layers
            for surface, unit, target_a, donor_a in ((s, u, ta, da) for s, u in itertools.product(common.SURFACES, range(8)) for ta, da in ((0, 1), (1, 0))):
                ti, di = key[(surface, unit, target_a, 0, 1)], key[(surface, unit, donor_a, 0, 1)]; target, donor = compiled[ti], compiled[di]; ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device); donor_gold = donor["gold_position"]; other = 1 - donor_gold
                clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True); clean_margin = float(clean.logits[0, -1, target["candidate_ids"][donor_gold][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]]); donor_final = np.asarray(states[di, -1, 5], np.float32); target_final = np.asarray(states[ti, -1, 5], np.float32); base_distance = float(np.linalg.norm(target_final - donor_final))
                for condition in CONDITIONS:
                    handles = []
                    for q in CHECKPOINTS:
                        source_q = 40 - q if condition == "reversed_checkpoint_masks" else q
                        def make_hook(q=q, source_q=source_q, condition=condition):
                            def hook(_module, _inputs, output):
                                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                                for ri, role in enumerate(common.ROLES):
                                    mfi = common.FAMILIES.index("type_graph") if condition == "wrong_family" else fi; mask = np.asarray(tri[mfi, 0, source_q, ri] != 0)
                                    if condition == "coordinate_roll": mask = np.roll(mask, 137)
                                    coords = np.flatnonzero(mask)
                                    if not coords.size: continue
                                    coord_t = torch.tensor(coords, dtype=torch.long, device=hidden.device); value_t = torch.tensor(np.asarray(states[di, q, ri], np.float32)[coords], dtype=hidden.dtype, device=hidden.device)
                                    for pos in target["role_positions"][role]: hidden[0, pos, coord_t] = 2 * hidden[0, pos, coord_t] - value_t if condition == "sign_reverse" else value_t
                                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                            return hook
                        handles.append(layers[q - 1].register_forward_hook(make_hook()))
                    try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
                    finally:
                        for h in handles: h.remove()
                    margin = float(output.logits[0, -1, target["candidate_ids"][donor_gold][0]] - output.logits[0, -1, target["candidate_ids"][other][0]]); final = output.hidden_states[-1][0, -1].float().cpu().numpy(); distance = float(np.linalg.norm(final - donor_final)); intervention.append({"surface": surface, "unit": unit, "direction": f"{target_a}_to_{donor_a}", "condition": condition, "clean_donor_margin": clean_margin, "patched_donor_margin": margin, "donor_margin_gain": margin - clean_margin, "flipped_to_donor": margin > 0, "trajectory_improvement": (base_distance - distance) / max(base_distance, 1e-12)})
                print(f"[C257] causal {surface} unit {unit + 1}/8 {target_a}->{donor_a}", flush=True)
        core.write_rows(OUT / "analysis/intervention_rows.jsonl", intervention); medians = {c: float(np.median([r["donor_margin_gain"] for r in intervention if r["condition"] == c])) for c in CONDITIONS} if intervention else {}; flips = {c: float(np.mean([r["flipped_to_donor"] for r in intervention if r["condition"] == c])) for c in CONDITIONS} if intervention else {}; causal_margin = medians.get("correct_path", -1e9) - max((medians.get(c, -1e9) for c in CONDITIONS[1:]), default=-1e9); causal_pass = bool(intervention) and medians["correct_path"] > 0 and causal_margin >= .10 and flips["correct_path"] >= .20
        report = {"phase": 1791, "campaign": "C257", "status": "adjudicated", "c256_reclassified": "output causality not tested due outcome-relative candidate labels", "behavior_accuracy": accuracy, "behavior_eligible": behavior_ok, "event_groups": len(event_groups), "event_transfer_signed_jaccard": observed, "event_controls": {"wrong_family": wrong, "coordinate_roll_max": rolled, "sign_flip": sign_flip}, "event_margin": event_margin, "event_eligible": event_ok, "intervention_rows": len(intervention), "condition_margin_gain_medians": medians, "condition_flip_rates": flips, "correct_vs_best_control_margin": causal_margin, "output_causal_gate_passed": causal_pass, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": "The fixed codebook makes factor A change the correct physical output label. Failure is an output-control boundary for this distributed band, not a refutation of its observational recurrence or trajectory effect.", "next_authorization": "C258_output_readout_localization_observation_if_failed; independent_natural_paraphrase_if_passed"}
        core.save(OUT / "analysis/summary.json", report); analysis_checks = {"behavior": len(behavior) == 128, "event_groups": len(event_groups) >= 28, "intervention": len(intervention) == (160 if event_ok else 0), "fixed_codebook": fixed_label_changes, "finite": bool(np.isfinite([accuracy, observed, wrong, rolled, sign_flip, event_margin, causal_margin]).all())}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())}); final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1791, "campaign": "C257", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]}); print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
