#!/usr/bin/env python3
"""C259: independent lexical/surface replication of dense output-path control."""
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
import phase1791_c257_fixed_codebook_output_causal_test as c257

core = common.core
OUT = common.RESULT / "phase1793_c259_independent_dense_path_replication"
SURFACES = ("correspondence", "briefing")
UNITS = (
    ("Jalen", "Keira", "Liora", "nectarine", "telescope"), ("Mason", "Noemi", "Oskar", "broccoli", "tambourine"),
    ("Paxton", "Reina", "Soren", "cranberry", "wheelbarrow"), ("Thalia", "Urban", "Valen", "artichoke", "metronome"),
    ("Willow", "Xavian", "Yvette", "clementine", "typewriter"), ("Zane", "Aurelia", "Bram", "spinach", "hourglass"),
    ("Celine", "Darius", "Estelle", "gooseberry", "microscope"), ("Felix", "Greer", "Hugo", "asparagus", "harmonica"),
)
ROUTES = {"correct_dense": tuple(range(1, 36)), "correct_early": tuple(range(1, 17)), "wrong_family_dense": tuple(range(1, 36)), "coordinate_roll_dense": tuple(range(1, 36)), "reversed_masks_dense": tuple(range(1, 36))}


def wrap(surface, fact1, fact2, question):
    if surface == "correspondence": return f"A correspondence file says: {fact1} An unrelated note says: {fact2} {question}"
    return f"During a briefing, the speaker reported that {fact1} The speaker separately noted that {fact2} {question}"


def material():
    rows = []
    for surface, unit, a, b, order in itertools.product(SURFACES, range(8), (0, 1), (0, 1), (1, -1)):
        p, s, o, obj, other = UNITS[unit]; relation = "welcomes" if a == 0 else "disputes"; target = f"{o} {relation} the report that {p} reviewed the {obj}." if b == 0 else f"{o} {relation} the report that the {obj} was reviewed by {p}."; core_prompt = wrap(surface, target, f"{s} stored the {other}.", f"Does {o} express approval or doubt about the report?")
        if order == 1: choices, gold = "(A) approval (B) doubt", (0 if a == 0 else 1)
        else: choices, gold = "(A) doubt (B) approval", (1 if a == 0 else 0)
        rows.append({"case_id": f"c259-{surface}-u{unit}-{a}{b}-{order:+d}", "family": "attitude_event", "surface": surface, "unit": unit, "factor_a": a, "factor_b": b, "order": order, "gold_position": gold, "prompt": f"{core_prompt} {choices}. Reply with only A or B.", "role_values": {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": o}})
    return rows


@torch.inference_mode()
def main():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.RESULT / "phase1792_c258_output_readout_path_localization/analysis/final.json"); rows = material(); compiled = c257.compile_rows(common.graph_base.tokenizer(), rows); old_primary = {u["primary"] for u in common.UNITS}
    checks = {"parent": parent["all_checks_passed"] and parent["headline"]["output_path_gate_passed"], "rows": len(rows) == 128, "fixed_codebook": all(next(r["gold_position"] for r in rows if r["surface"] == s and r["unit"] == u and r["factor_a"] == 0 and r["factor_b"] == b and r["order"] == o) != next(r["gold_position"] for r in rows if r["surface"] == s and r["unit"] == u and r["factor_a"] == 1 and r["factor_b"] == b and r["order"] == o) for s, u, b, o in itertools.product(SURFACES, range(8), (0, 1), (1, -1))), "new_lexicon": not ({u[0] for u in UNITS} & old_primary), "new_surfaces": not (set(SURFACES) & set(common.SURFACES)), "width": max(len(r["prompt_ids"]) for r in compiled) <= 128}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "raw").mkdir(); (OUT / "analysis").mkdir(); protocol = {"phase": 1793, "campaign": "C259", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "independent_dense_path_replication_frozen", "surfaces": list(SURFACES), "lexical_units": 8, "behavior_gate": .90, "event_gate": {"jaccard_min": .07, "control_margin_min": .02}, "causal_gate": {"correct_dense_flip_rate_min": .80, "correct_vs_best_control_margin_min": 2.0}, "routes": {k: list(v) for k, v in ROUTES.items()}, "producer_sha256": core.sha(Path(__file__)), "authorization": "capture_and_replicate_once"}; core.save(OUT / "protocol/preregistration.json", protocol); core.write_rows(OUT / "material/cases.jsonl", rows); core.write_rows(OUT / "compiled/qwen3.jsonl", compiled); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r"); thresholds = np.asarray(core.load(common.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32); fi = common.FAMILIES.index("attitude_event"); wrong_fi = common.FAMILIES.index("type_graph"); model = None; started = time.time()
    try:
        model, _tok, device, placement = common.previous.load_bf16("qwen3"); states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(128, 37, 6, 2560)); behavior = []
        for i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device); output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True); logits = [float(output.logits[0, -1, c[0]]) for c in row["candidate_ids"]]; pred = int(logits[1] > logits[0]); behavior.append({"case_id": row["case_id"], "correct": pred == row["gold_position"], "prediction": pred, "gold_position": row["gold_position"], "logits": logits})
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES): states[i, q, ri] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            if i % 16 == 0 or i == 127: states.flush(); print(f"[C259] clean {i + 1}/128", flush=True)
        core.write_rows(OUT / "raw/behavior.jsonl", behavior); accuracy = float(np.mean([r["correct"] for r in behavior])); key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}; event_groups = []
        for surface, unit, order in itertools.product(SURFACES, range(8), (1, -1)):
            needed = [(surface, unit, a, b, order) for a, b in itertools.product((0, 1), repeat=2)]
            if not all(behavior[key[x]]["correct"] for x in needed): continue
            cells = {(a, b): np.asarray(states[key[(surface, unit, a, b, order)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}; effect = common.factorial_effect(cells)[0]; event_groups.append(np.where(effect > thresholds[:, None, None], 1, np.where(effect < -thresholds[:, None, None], -1, 0)).astype(np.int8))
        current = np.asarray(event_groups); up, down = np.mean(current == 1, axis=0), np.mean(current == -1, axis=0); active = up + down; stable = np.where((active >= .75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= .80), np.where(up >= down, 1, -1), 0).astype(np.int8); frozen = np.asarray(tri[fi, 0]); observed = common.signed_jaccard(frozen, stable); wrong = common.signed_jaccard(np.asarray(tri[wrong_fi, 0]), stable); rolled = max(common.signed_jaccard(np.roll(frozen, s, axis=-1), stable) for s in (137, 389, 743, 1291)); event_margin = observed - max(wrong, rolled); event_ok = accuracy >= .90 and observed >= .07 and event_margin >= .02; intervention = []
        if event_ok:
            layers = model.model.layers
            for surface, unit, ta, da in ((s, u, ta, da) for s, u in itertools.product(SURFACES, range(8)) for ta, da in ((0, 1), (1, 0))):
                ti, di = key[(surface, unit, ta, 0, 1)], key[(surface, unit, da, 0, 1)]; target, donor = compiled[ti], compiled[di]; ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device); donor_gold = donor["gold_position"]; other = 1 - donor_gold; clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True); clean_margin = float(clean.logits[0, -1, target["candidate_ids"][donor_gold][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]])
                for route, checkpoints in ROUTES.items():
                    handles = []
                    for q in checkpoints:
                        source_q = 36 - q if route == "reversed_masks_dense" else q
                        def make_hook(q=q, source_q=source_q, route=route):
                            def hook(_m, _i, output):
                                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                                for ri, role in enumerate(common.ROLES):
                                    mfi = wrong_fi if route == "wrong_family_dense" else fi; mask = np.asarray(tri[mfi, 0, source_q, ri] != 0)
                                    if route == "coordinate_roll_dense": mask = np.roll(mask, 137)
                                    coords = np.flatnonzero(mask)
                                    if not coords.size: continue
                                    ci = torch.tensor(coords, dtype=torch.long, device=hidden.device); vv = torch.tensor(np.asarray(states[di, q, ri], np.float32)[coords], dtype=hidden.dtype, device=hidden.device)
                                    for pos in target["role_positions"][role]: hidden[0, pos, ci] = vv
                                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                            return hook
                        handles.append(layers[q - 1].register_forward_hook(make_hook()))
                    try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                    finally:
                        for h in handles: h.remove()
                    margin = float(output.logits[0, -1, target["candidate_ids"][donor_gold][0]] - output.logits[0, -1, target["candidate_ids"][other][0]]); intervention.append({"surface": surface, "unit": unit, "direction": f"{ta}_to_{da}", "route": route, "donor_margin_gain": margin - clean_margin, "flipped_to_donor": margin > 0, "patched_margin": margin})
                print(f"[C259] causal {surface} {unit + 1}/8 {ta}->{da}", flush=True)
        core.write_rows(OUT / "analysis/intervention_rows.jsonl", intervention); summaries = [{"route": route, "support": len([r for r in intervention if r["route"] == route]), "median_margin_gain": float(np.median([r["donor_margin_gain"] for r in intervention if r["route"] == route])), "flip_rate": float(np.mean([r["flipped_to_donor"] for r in intervention if r["route"] == route]))} for route in ROUTES] if intervention else []; by = {r["route"]: r for r in summaries}; controls = ("wrong_family_dense", "coordinate_roll_dense", "reversed_masks_dense"); causal_margin = by.get("correct_dense", {}).get("median_margin_gain", -1e9) - max((by.get(c, {}).get("median_margin_gain", -1e9) for c in controls), default=-1e9); causal_pass = bool(intervention) and by["correct_dense"]["flip_rate"] >= .80 and causal_margin >= 2.0
        report = {"phase": 1793, "campaign": "C259", "status": "adjudicated", "behavior_accuracy": accuracy, "behavior_eligible": accuracy >= .90, "event_groups": len(event_groups), "event_signed_jaccard": observed, "event_controls": {"wrong_family": wrong, "coordinate_roll_max": rolled}, "event_margin": event_margin, "event_eligible": event_ok, "route_summaries": summaries, "correct_vs_best_control_margin": causal_margin, "independent_dense_path_gate_passed": causal_pass, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": "A pass is an independent controlled lexical/surface replication of distributed dense-path output control. It remains a large intervention and does not establish a minimal or naturally necessary circuit.", "next_authorization": "C260_minimal_path_ladder_and_natural_generation_side_effects_if_passed; route_local_redesign_if_failed"}; core.save(OUT / "analysis/summary.json", report); ach = {"behavior": len(behavior) == 128, "events": len(event_groups) >= 28, "interventions": len(intervention) == (160 if event_ok else 0), "finite": bool(np.isfinite([accuracy, observed, wrong, rolled, event_margin, causal_margin]).all())}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1793, "campaign": "C259", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); core.save(OUT / "audit/independent_final_audit.json", {"checks": fch, "all_checks_passed": all(fch.values()), "authorization": report["next_authorization"]}); print(json.dumps(final, indent=2))
    finally: common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
