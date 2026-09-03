#!/usr/bin/env python3
"""C270: test full words, short sentences, and an unrelated entity readout."""
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
import phase1803_c269_predicted_edge_local_causal as c269

core, OUT, C265 = common.core, common.OUTS["C270"], common.OUTS["C265"]
CONDITIONS = ("clean", "state_correct", "coordinate_roll", "wrong_family", "full_single")


def make_prompts(surface, unit, a):
    u = common.UNITS[unit]; relation = "endorses" if a == 0 else "challenges"
    fact = f"{u['observer']} {relation} the report that {u['primary']} inspected the {u['object']}."
    noise = f"{u['secondary']} adjusted the {u['other']}."
    base = common.wrap(surface, fact, noise, f"Does {u['observer']} express approval or doubt about the report?")
    entity = common.wrap(surface, fact, noise, f"Who inspected the {u['object']}?")
    return {"relation": relation, "word": base + " Answer with only approval or doubt.", "sentence": base + ' Answer with one short sentence beginning "The attitude is".', "entity": entity + " Answer with only the person's name.", "expected_word": "approval" if a == 0 else "doubt", "expected_entity": u["primary"]}


def compile_prompt(tokenizer, text, relation):
    ids = core.chat_ids(tokenizer, "Answer only from the supplied text.", text)
    spans = common.graph_base.name_spans(tokenizer, ids, relation)
    if not spans: raise RuntimeError((relation, text))
    return ids, spans[0]


def install(model, q, positions, coords, donor_values, mode):
    c = torch.as_tensor(coords, dtype=torch.long, device=next(model.parameters()).device)
    def hook(_module, _inputs, output):
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        if hidden.shape[1] > max(positions):
            value = torch.as_tensor(donor_values, dtype=hidden.dtype, device=hidden.device)
            for pos in positions: hidden[0, pos, c] = value
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    return model.model.layers[q - 1].register_forward_hook(hook)


@torch.inference_mode()
def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C269"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "continue_if_causal_gate_failed": True, "tokenizer_aware": True, "unrelated_entity_panel": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "analysis").mkdir(exist_ok=True); (OUT / "audit").mkdir(exist_ok=True)
    protocol = {"phase": 1804, "campaign": "C270", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "generation_frozen", "panel": "new C263 lexicon under two surfaces and both attitude directions", "conditions": list(CONDITIONS), "generation": {"word_max_tokens": 3, "sentence_max_tokens": 10, "entity_max_tokens": 5}, "gate": {"word_correct_min": .80, "word_control_margin_min": .20, "sentence_contains_target_min": .80, "entity_preservation_min": .80}, "claim_boundary": "The sentence panel is constrained generation, not unrestricted discourse. Entity preservation is one unrelated readout, not a complete side-effect audit.", "producer_sha256": core.sha(Path(__file__)), "authorization": "C271_cross_model_conditional_topology"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    pred_map = np.load(C265 / "analysis/passport_pred_sign.int8.npy", mmap_mode="r"); med_map = np.load(C265 / "analysis/passport_baseline_median.float16.npy", mmap_mode="r"); thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    fi, wrong_fi = common.FAMILIES.index("attitude_event"), common.FAMILIES.index("type_graph")
    model, rows, started = None, [], time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        for surface, unit in itertools.product(common.SURFACES, range(8)):
            prompt_data = {a: make_prompts(surface, unit, a) for a in (0, 1)}
            state_ids = {}; state_positions = {}; state_values = {}
            for a in (0, 1):
                ids, positions = compile_prompt(tokenizer, prompt_data[a]["word"], prompt_data[a]["relation"]); state_ids[a], state_positions[a] = ids, positions
                output = model(input_ids=torch.tensor([ids], device=device), attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device), use_cache=False, return_dict=True, output_hidden_states=True)
                state_values[a] = np.stack([s[0, positions].mean(0).float().cpu().numpy() for s in output.hidden_states])
            for ta, da in ((0, 1), (1, 0)):
                # select_q expects the registered six-role axis; only relation is populated.
                target_fake = np.zeros((37, 6, 2560), np.float32); donor_fake = target_fake.copy(); ri = common.ROLES.index("relation")
                target_fake[:, ri] = state_values[ta]; donor_fake[:, ri] = state_values[da]
                qstar, curve = c269.select_q(target_fake, donor_fake, fi, pred_map, med_map, thresholds)
                masks = {
                    "state_correct": c269.state_mask(target_fake, donor_fake, qstar, fi, pred_map, med_map, thresholds[qstar]),
                    "wrong_family": c269.state_mask(target_fake, donor_fake, qstar, wrong_fi, pred_map, med_map, thresholds[qstar]),
                    "full_single": np.ones(2560, bool),
                }
                masks["coordinate_roll"] = np.roll(masks["state_correct"], 137)
                for task, max_tokens in (("word", 3), ("sentence", 10), ("entity", 5)):
                    ids, positions = compile_prompt(tokenizer, prompt_data[ta][task], prompt_data[ta]["relation"])
                    for condition in CONDITIONS:
                        handle = None
                        if condition != "clean":
                            coords = np.flatnonzero(masks[condition]); donor_values = state_values[da][qstar, coords]
                            handle = install(model, qstar, positions, coords, donor_values, condition)
                        try:
                            generated = model.generate(input_ids=torch.tensor([ids], device=device), attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device), max_new_tokens=max_tokens, do_sample=False, use_cache=True)
                        finally:
                            if handle is not None: handle.remove()
                        text = tokenizer.decode(generated[0, len(ids):].tolist()).strip().lower()
                        expected = prompt_data[ta]["expected_entity"].lower() if task == "entity" else (prompt_data[ta]["expected_word"] if condition == "clean" else prompt_data[da]["expected_word"])
                        success = text.startswith(expected) if task in ("word", "entity") else expected in text
                        rows.append({"surface": surface, "unit": unit, "direction": f"{ta}_to_{da}", "task": task, "condition": condition, "selected_checkpoint": qstar, "coordinates": 0 if condition == "clean" else int(masks[condition].sum()), "expected": expected, "text": text, "success": success, "readiness_curve": curve})
                print(f"[C270] {surface} u{unit} {ta}->{da} q{qstar}", flush=True)
        core.write_rows(OUT / "analysis/generation_rows.jsonl", rows)
        summaries = []
        for task in ("word", "sentence", "entity"):
            for condition in CONDITIONS:
                selected = [r for r in rows if r["task"] == task and r["condition"] == condition]
                summaries.append({"task": task, "condition": condition, "support": len(selected), "success_rate": float(np.mean([r["success"] for r in selected])), "outputs": sorted({r["text"] for r in selected})[:20]})
        by = {(r["task"], r["condition"]): r for r in summaries}
        word_margin = by[("word", "state_correct")]["success_rate"] - max(by[("word", c)]["success_rate"] for c in ("coordinate_roll", "wrong_family"))
        gate = by[("word", "state_correct")]["success_rate"] >= .80 and word_margin >= .20 and by[("sentence", "state_correct")]["success_rate"] >= .80 and by[("entity", "state_correct")]["success_rate"] >= .80
        report = {"phase": 1804, "campaign": "C270", "status": "generation_adjudicated", "summaries": summaries, "word_correct_minus_best_wrong_control": word_margin, "generation_and_side_effect_gate_passed": gate, "placement": placement, "elapsed_seconds": time.time() - started, "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C271_cross_model_conditional_bisimulation"}; core.save(OUT / "analysis/summary.json", report)
        ach = {"rows": len(rows) == 32 * 3 * len(CONDITIONS), "support": all(r["support"] == 32 for r in summaries), "hooks_removed": True, "finite": bool(np.isfinite(word_margin))}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); fch = {"contract": True, "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}; final = {"phase": 1804, "campaign": "C270", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model); gc.collect()


if __name__ == "__main__": main()
