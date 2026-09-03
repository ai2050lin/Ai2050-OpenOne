#!/usr/bin/env python3
"""C260: prefix/role path ladder and direct approval/doubt word readout."""
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
OUT = common.RESULT / "phase1794_c260_path_ladder_natural_word_readout"
C259 = common.RESULT / "phase1793_c259_independent_dense_path_replication"
PREFIXES = (2, 4, 6, 8, 10, 12, 14, 16)
ROLE_ROUTES = {
    "relation_only": ("relation",), "relation_query": ("relation", "query"),
    "relation_query_boundary": ("relation", "query", "boundary"),
    "nonboundary_all": tuple(role for role in common.ROLES if role != "boundary"), "all_roles": common.ROLES,
}


def hook_route(model, tri, states, donor_i, target, family_i, checkpoints, roles, mode="correct"):
    handles = []
    wrong_fi = common.FAMILIES.index("type_graph")
    for q in checkpoints:
        source_q = 36 - q if mode == "reversed" else q
        def make_hook(q=q, source_q=source_q, mode=mode):
            def hook(_m, _i, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                for role in roles:
                    ri = common.ROLES.index(role); fi = wrong_fi if mode == "wrong_family" else family_i; mask = np.asarray(tri[fi, 0, source_q, ri] != 0)
                    if mode == "roll": mask = np.roll(mask, 137)
                    coords = np.flatnonzero(mask)
                    if not coords.size: continue
                    ci = torch.tensor(coords, dtype=torch.long, device=hidden.device); vv = torch.tensor(np.asarray(states[donor_i, q, ri], np.float32)[coords], dtype=hidden.dtype, device=hidden.device)
                    for pos in target["role_positions"][role]: hidden[0, pos, ci] = vv
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            return hook
        handles.append(model.model.layers[q - 1].register_forward_hook(make_hook()))
    return handles


@torch.inference_mode()
def main():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C259 / "analysis/final.json"); compiled = core.rows(C259 / "compiled/qwen3.jsonl"); states = np.load(C259 / "raw/role_states.float16.npy", mmap_mode="r"); tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r"); key = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): i for i, r in enumerate(compiled)}; fi = common.FAMILIES.index("attitude_event")
    checks = {"parent": parent["all_checks_passed"] and parent["headline"]["independent_dense_path_gate_passed"], "prefixes_frozen": PREFIXES == tuple(range(2, 17, 2)), "roles_frozen": set(ROLE_ROUTES) == {"relation_only", "relation_query", "relation_query_boundary", "nonboundary_all", "all_roles"}, "no_coordinate_selection": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); (OUT / "analysis").mkdir(); protocol = {"phase": 1794, "campaign": "C260", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "path_ladder_frozen", "prefix_endpoints": list(PREFIXES), "role_routes": {k: list(v) for k, v in ROLE_ROUTES.items()}, "dense_controls": ["wrong_family", "coordinate_roll", "reversed_masks"], "natural_word_readout": "choice-free prompt scored on direct approval/doubt tokens", "gates": {"prefix_flip_min": .80, "dense_control_margin_min": 2.0, "word_flip_min": .80, "word_control_margin_min": 2.0}, "claim_boundary": "Minimality is only within the registered checkpoint-prefix and role-set ladder; no coordinate subset minimality is claimed.", "producer_sha256": core.sha(Path(__file__)), "authorization": "run_ladder_and_word_readout_once"}; core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    model = None; ladder_rows = []; word_rows = []; started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        routes = {**{f"prefix_{end}": (tuple(range(1, end + 1)), common.ROLES, "correct") for end in PREFIXES}, **{f"roles_{name}": (tuple(range(1, 17)), roles, "correct") for name, roles in ROLE_ROUTES.items()}, "control_wrong_family": (tuple(range(1, 17)), common.ROLES, "wrong_family"), "control_roll": (tuple(range(1, 17)), common.ROLES, "roll"), "control_reversed": (tuple(range(1, 17)), common.ROLES, "reversed")}
        for surface, unit, ta, da in ((s, u, ta, da) for s, u in itertools.product(("correspondence", "briefing"), range(8)) for ta, da in ((0, 1), (1, 0))):
            ti, di = key[(surface, unit, ta, 0, 1)], key[(surface, unit, da, 0, 1)]; target, donor = compiled[ti], compiled[di]; ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device); dg = donor["gold_position"]; other = 1 - dg; clean = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True); clean_margin = float(clean.logits[0, -1, target["candidate_ids"][dg][0]] - clean.logits[0, -1, target["candidate_ids"][other][0]])
            for route, (checkpoints, roles, mode) in routes.items():
                handles = hook_route(model, tri, states, di, target, fi, checkpoints, roles, mode)
                try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                finally:
                    for h in handles: h.remove()
                margin = float(output.logits[0, -1, target["candidate_ids"][dg][0]] - output.logits[0, -1, target["candidate_ids"][other][0]]); ladder_rows.append({"surface": surface, "unit": unit, "direction": f"{ta}_to_{da}", "route": route, "margin_gain": margin - clean_margin, "flipped": margin > 0})
            print(f"[C260] ladder {surface} {unit + 1}/8 {ta}->{da}", flush=True)
        core.write_rows(OUT / "analysis/ladder_rows.jsonl", ladder_rows); ladder_summary = []
        for route in routes:
            subset = [r for r in ladder_rows if r["route"] == route]; ladder_summary.append({"route": route, "support": len(subset), "median_margin_gain": float(np.median([r["margin_gain"] for r in subset])), "flip_rate": float(np.mean([r["flipped"] for r in subset]))})
        by = {r["route"]: r for r in ladder_summary}; passing_prefixes = [end for end in PREFIXES if by[f"prefix_{end}"]["flip_rate"] >= .80 and by[f"prefix_{end}"]["median_margin_gain"] > 0]; earliest = min(passing_prefixes) if passing_prefixes else None; dense_control_margin = by["prefix_16"]["median_margin_gain"] - max(by[name]["median_margin_gain"] for name in ("control_wrong_family", "control_roll", "control_reversed")); ladder_pass = earliest is not None and dense_control_margin >= 2.0

        # Direct word readout uses one surface and all eight units in both directions.
        base_rows = core.rows(C259 / "material/cases.jsonl"); base_by = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): r for r in base_rows}; word_tokens = {"approval": tokenizer.encode(" approval", add_special_tokens=False), "doubt": tokenizer.encode(" doubt", add_special_tokens=False)}
        if any(len(v) != 1 for v in word_tokens.values()): raise RuntimeError(word_tokens)
        word_compiled = []; word_states = []
        for unit, a in itertools.product(range(8), (0, 1)):
            base = base_by[("correspondence", unit, a, 0, 1)]; prompt_core = base["prompt"].split(" (A)")[0]; prompt = f"{prompt_core} Answer with only approval or doubt."; ids_list = core.chat_ids(tokenizer, "Answer only from the supplied text. Do not use outside knowledge.", prompt); positions = {}
            for role, value in base["role_values"].items(): positions[role] = (common.graph_base.name_spans(tokenizer, ids_list, value)[-1] if role == "query" else common.graph_base.name_spans(tokenizer, ids_list, value)[0])
            positions["boundary"] = [len(ids_list) - 1]; ids = torch.tensor([ids_list], dtype=torch.long, device=device); output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True); clean_logits = {name: float(output.logits[0, -1, tok[0]]) for name, tok in word_tokens.items()}; role_state = np.empty((37, 6, 2560), np.float16)
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES): role_state[q, ri] = state[0, positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            word_compiled.append({"unit": unit, "a": a, "prompt_ids": ids_list, "role_positions": positions, "clean_logits": clean_logits}); word_states.append(role_state)
        word_states = np.asarray(word_states); wkey = {(r["unit"], r["a"]): i for i, r in enumerate(word_compiled)}
        for unit, ta, da in ((u, ta, da) for u in range(8) for ta, da in ((0, 1), (1, 0))):
            ti, di = wkey[(unit, ta)], wkey[(unit, da)]; target, donor = word_compiled[ti], word_compiled[di]; donor_word = "approval" if da == 0 else "doubt"; other_word = "doubt" if da == 0 else "approval"; clean_margin = target["clean_logits"][donor_word] - target["clean_logits"][other_word]; ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device)
            for route, mode in (("correct_early", "correct"), ("wrong_family", "wrong_family"), ("coordinate_roll", "roll"), ("reversed_masks", "reversed")):
                handles = hook_route(model, tri, word_states, di, target, fi, tuple(range(1, 17)), common.ROLES, mode)
                try: output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                finally:
                    for h in handles: h.remove()
                margin = float(output.logits[0, -1, word_tokens[donor_word][0]] - output.logits[0, -1, word_tokens[other_word][0]]); word_rows.append({"unit": unit, "direction": f"{ta}_to_{da}", "route": route, "margin_gain": margin - clean_margin, "flipped": margin > 0})
        core.write_rows(OUT / "analysis/natural_word_rows.jsonl", word_rows); word_summary = []
        for route in ("correct_early", "wrong_family", "coordinate_roll", "reversed_masks"):
            subset = [r for r in word_rows if r["route"] == route]; word_summary.append({"route": route, "support": len(subset), "median_margin_gain": float(np.median([r["margin_gain"] for r in subset])), "flip_rate": float(np.mean([r["flipped"] for r in subset]))})
        wb = {r["route"]: r for r in word_summary}; word_margin = wb["correct_early"]["median_margin_gain"] - max(wb[name]["median_margin_gain"] for name in ("wrong_family", "coordinate_roll", "reversed_masks")); word_pass = wb["correct_early"]["flip_rate"] >= .80 and word_margin >= 2.0
        report = {"phase": 1794, "campaign": "C260", "status": "adjudicated", "ladder_summary": ladder_summary, "earliest_passing_prefix_end": earliest, "prefix16_vs_best_control_margin": dense_control_margin, "path_ladder_gate_passed": ladder_pass, "natural_word_summary": word_summary, "natural_word_control_margin": word_margin, "natural_word_gate_passed": word_pass, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": "Minimality is limited to checkpoint-prefix and role-set coverage. Direct approval/doubt logits remove the A/B answer-code dependency but remain a controlled one-token readout, not free generation or natural necessity.", "next_authorization": "new_major_stage_required_for_coordinate_minimality_free_generation_and_unrelated_side_effects"}; core.save(OUT / "analysis/summary.json", report); ach = {"ladder_rows": len(ladder_rows) == 32 * len(routes), "word_rows": len(word_rows) == 16 * 4, "finite": bool(np.isfinite([r[k] for r in ladder_summary + word_summary for k in ("median_margin_gain", "flip_rate")] + [dense_control_margin, word_margin]).all()), "hooks_removed": True}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1794, "campaign": "C260", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); core.save(OUT / "audit/independent_final_audit.json", {"checks": fch, "all_checks_passed": all(fch.values()), "authorization": report["next_authorization"]}); print(json.dumps(final, indent=2))
    finally: common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
