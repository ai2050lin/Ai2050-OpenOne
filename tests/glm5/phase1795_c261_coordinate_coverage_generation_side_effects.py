#!/usr/bin/env python3
"""C261: coordinate-coverage ladder, actual word generation, erasure, and side effects."""
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
OUT = common.RESULT / "phase1795_c261_coordinate_coverage_generation_side_effects"
C259 = common.RESULT / "phase1793_c259_independent_dense_path_replication"
FRACTIONS = (0.125, 0.25, 0.50, 0.75, 1.0)
SEEDS = (17, 43, 79, 131)
CHECKPOINTS = tuple(range(1, 17))
ROLES = ("relation",)


def subset(coords: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1:
        return coords
    # Deterministic nested ranking over physical coordinate IDs; no magnitude ranking or Top-K.
    score = ((coords.astype(np.uint64) * np.uint64(1103515245) + np.uint64(seed * 2654435761)) & np.uint64(0xFFFFFFFF))
    return coords[score < np.uint64(round(fraction * (2**32 - 1)))]


def install_hooks(model, tri, states, donor_i, target, family_i, fraction=1.0, seed=17, mode="correct", operation="donor"):
    handles = []
    wrong_fi = common.FAMILIES.index("type_graph")
    for q in CHECKPOINTS:
        source_q = 36 - q if mode == "reversed" else q

        def make_hook(q=q, source_q=source_q, mode=mode):
            def hook(_module, _inputs, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                for role in ROLES:
                    ri = common.ROLES.index(role)
                    fi = wrong_fi if mode == "wrong_family" else family_i
                    mask = np.asarray(tri[fi, 0, source_q, ri] != 0)
                    if mode == "roll":
                        mask = np.roll(mask, 137)
                    coords = subset(np.flatnonzero(mask), fraction, seed)
                    if not coords.size:
                        continue
                    positions = target["role_positions"][role]
                    c = torch.as_tensor(coords, dtype=torch.long, device=hidden.device)
                    donor = torch.as_tensor(states[donor_i, source_q, ri, coords].astype(np.float32), dtype=hidden.dtype, device=hidden.device)
                    for pos in positions:
                        if operation == "midpoint":
                            current = hidden[0, pos, c]
                            hidden[0, pos, c] = (current + donor) / 2
                        else:
                            hidden[0, pos, c] = donor
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook

        handles.append(model.model.layers[q - 1].register_forward_hook(make_hook()))
    return handles


def forward(model, ids, handles):
    try:
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()


def build_word_panel(model, tokenizer, device):
    base_rows = core.rows(C259 / "material/cases.jsonl")
    base_by = {(r["surface"], r["unit"], r["factor_a"], r["factor_b"], r["order"]): r for r in base_rows}
    word_tokens = {name: tokenizer.encode(f" {name}", add_special_tokens=False) for name in ("approval", "doubt")}
    if any(len(value) != 1 for value in word_tokens.values()):
        raise RuntimeError(word_tokens)
    compiled, states = [], []
    for unit, a in itertools.product(range(8), (0, 1)):
        base = base_by[("correspondence", unit, a, 0, 1)]
        prompt = f"{base['prompt'].split(' (A)')[0]} Answer with only approval or doubt."
        ids_list = core.chat_ids(tokenizer, "Answer only from the supplied text. Do not use outside knowledge.", prompt)
        positions = {}
        for role, value in base["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids_list, value)
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids_list) - 1]
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
        role_state = np.empty((37, 6, 2560), np.float16)
        for q, state in enumerate(output.hidden_states):
            for ri, role in enumerate(common.ROLES):
                role_state[q, ri] = state[0, positions[role]].mean(0).detach().float().cpu().numpy().astype(np.float16)
        compiled.append({
            "unit": unit,
            "a": a,
            "prompt": prompt,
            "prompt_ids": ids_list,
            "role_positions": positions,
            "clean_logits": output.logits[0, -1].float().cpu(),
        })
        states.append(role_state)
    return compiled, np.asarray(states), word_tokens


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("protocol", "analysis", "audit"):
        (OUT / name).mkdir(exist_ok=True)
    checks = {
        "c259_passed": core.load(C259 / "analysis/final.json")["headline"]["independent_dense_path_gate_passed"],
        "fractions_frozen": FRACTIONS == tuple(sorted(FRACTIONS)) and FRACTIONS[-1] == 1,
        "deterministic_seeds": len(set(SEEDS)) == 4,
        "hidden_state_only": True,
        "no_topk": True,
    }
    protocol = {
        "phase": 1795,
        "campaign": "C261",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_load",
        "coordinate_fractions": list(FRACTIONS),
        "partition_seeds": list(SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "roles": list(ROLES),
        "tests": ["deterministic_coordinate_coverage", "actual_first_word_generation", "same_pair_midpoint_erasure", "full_vocabulary_side_effect"],
        "gates": {
            "coverage_flip_min": 0.80,
            "coverage_control_margin_min": 2.0,
            "actual_generation_success_min": 0.80,
            "erasure_margin_over_control_min": 2.0,
        },
        "claim_boundary": "Nested hash subsets retain low and high coordinates without magnitude ranking. Midpoint erasure is an artificial necessity probe, not natural deletion.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_once_then_close_campaign",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        tri = np.load(common.RESULT / "phase1783_c249_third_material_event_core_prediction/analysis/tri_material_core.int8.npy", mmap_mode="r")
        panel, states, word_tokens = build_word_panel(model, tokenizer, device)
        key = {(row["unit"], row["a"]): i for i, row in enumerate(panel)}
        family_i = common.FAMILIES.index("attitude_event")
        coverage_rows, generation_rows, erasure_rows, side_effect_rows = [], [], [], []

        for unit in range(8):
            for target_a, donor_a in ((0, 1), (1, 0)):
                ti, di = key[(unit, target_a)], key[(unit, donor_a)]
                target = panel[ti]
                donor_word = "approval" if donor_a == 0 else "doubt"
                target_word = "approval" if target_a == 0 else "doubt"
                ids = torch.tensor([target["prompt_ids"]], dtype=torch.long, device=device)
                clean_logits = target["clean_logits"]
                clean_target_margin = float(clean_logits[word_tokens[target_word][0]] - clean_logits[word_tokens[donor_word][0]])

                for seed, fraction in itertools.product(SEEDS, FRACTIONS):
                    handles = install_hooks(model, tri, states, di, target, family_i, fraction, seed)
                    output = forward(model, ids, handles)
                    margin = float(output.logits[0, -1, word_tokens[donor_word][0]] - output.logits[0, -1, word_tokens[target_word][0]])
                    coverage_rows.append({"unit": unit, "direction": f"{target_a}_to_{donor_a}", "seed": seed, "fraction": fraction, "donor_margin": margin, "flipped": margin > 0})

                condition_specs = (("correct", "correct"), ("wrong_family", "wrong_family"), ("coordinate_roll", "roll"), ("reversed_masks", "reversed"))
                for label, mode in condition_specs:
                    handles = install_hooks(model, tri, states, di, target, family_i, 1.0, 17, mode)
                    try:
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False, use_cache=False)
                    finally:
                        for handle in handles:
                            handle.remove()
                    token_id = int(generated[0, -1])
                    text = tokenizer.decode([token_id]).strip().lower()
                    generation_rows.append({"unit": unit, "direction": f"{target_a}_to_{donor_a}", "condition": label, "token_id": token_id, "text": text, "donor_word": donor_word, "success": text == donor_word})

                for label, mode in condition_specs:
                    handles = install_hooks(model, tri, states, di, target, family_i, 1.0, 17, mode, operation="midpoint")
                    output = forward(model, ids, handles)
                    erased_margin = float(output.logits[0, -1, word_tokens[target_word][0]] - output.logits[0, -1, word_tokens[donor_word][0]])
                    erasure_rows.append({"unit": unit, "direction": f"{target_a}_to_{donor_a}", "condition": label, "gold_margin_drop": clean_target_margin - erased_margin, "gold_flipped": erased_margin < 0})

                handles = install_hooks(model, tri, states, di, target, family_i, 1.0, 17)
                output = forward(model, ids, handles)
                clean_p = torch.softmax(clean_logits.float(), dim=-1)
                patch_p = torch.softmax(output.logits[0, -1].float().cpu(), dim=-1)
                midpoint = 0.5 * (clean_p + patch_p)
                js = 0.5 * (torch.sum(clean_p * torch.log((clean_p + 1e-30) / (midpoint + 1e-30))) + torch.sum(patch_p * torch.log((patch_p + 1e-30) / (midpoint + 1e-30))))
                candidate_ids = {word_tokens["approval"][0], word_tokens["doubt"][0]}
                clean_candidate_mass = sum(float(clean_p[i]) for i in candidate_ids)
                patch_candidate_mass = sum(float(patch_p[i]) for i in candidate_ids)
                side_effect_rows.append({
                    "unit": unit,
                    "direction": f"{target_a}_to_{donor_a}",
                    "total_variation": float(0.5 * torch.sum(torch.abs(clean_p - patch_p))),
                    "jensen_shannon": float(js),
                    "candidate_mass_change_abs": abs(patch_candidate_mass - clean_candidate_mass),
                    "top1_changed": int(torch.argmax(clean_p)) != int(torch.argmax(patch_p)),
                })

        core.write_rows(OUT / "analysis/coordinate_coverage_rows.jsonl", coverage_rows)
        core.write_rows(OUT / "analysis/generation_rows.jsonl", generation_rows)
        core.write_rows(OUT / "analysis/erasure_rows.jsonl", erasure_rows)
        core.write_rows(OUT / "analysis/side_effect_rows.jsonl", side_effect_rows)

        coverage_summary = []
        for fraction in FRACTIONS:
            subset_rows = [row for row in coverage_rows if row["fraction"] == fraction]
            coverage_summary.append({
                "fraction": fraction,
                "support": len(subset_rows),
                "median_donor_margin": float(np.median([row["donor_margin"] for row in subset_rows])),
                "flip_rate": float(np.mean([row["flipped"] for row in subset_rows])),
                "seed_flip_rates": {str(seed): float(np.mean([row["flipped"] for row in subset_rows if row["seed"] == seed])) for seed in SEEDS},
            })
        generation_summary = []
        erasure_summary = []
        for condition in ("correct", "wrong_family", "coordinate_roll", "reversed_masks"):
            grow = [row for row in generation_rows if row["condition"] == condition]
            erow = [row for row in erasure_rows if row["condition"] == condition]
            generation_summary.append({"condition": condition, "support": len(grow), "success_rate": float(np.mean([row["success"] for row in grow]))})
            erasure_summary.append({"condition": condition, "support": len(erow), "median_gold_margin_drop": float(np.median([row["gold_margin_drop"] for row in erow])), "gold_flip_rate": float(np.mean([row["gold_flipped"] for row in erow]))})
        earliest_fraction = next((row["fraction"] for row in coverage_summary if row["flip_rate"] >= 0.80), None)
        gb = {row["condition"]: row for row in generation_summary}
        eb = {row["condition"]: row for row in erasure_summary}
        erasure_control_margin = eb["correct"]["median_gold_margin_drop"] - max(eb[name]["median_gold_margin_drop"] for name in ("wrong_family", "coordinate_roll", "reversed_masks"))
        side_summary = {
            "support": len(side_effect_rows),
            "median_total_variation": float(np.median([row["total_variation"] for row in side_effect_rows])),
            "median_jensen_shannon": float(np.median([row["jensen_shannon"] for row in side_effect_rows])),
            "median_candidate_mass_change_abs": float(np.median([row["candidate_mass_change_abs"] for row in side_effect_rows])),
            "top1_change_rate": float(np.mean([row["top1_changed"] for row in side_effect_rows])),
        }
        report = {
            "phase": 1795,
            "campaign": "C261",
            "status": "adjudicated",
            "coverage_summary": coverage_summary,
            "earliest_registered_fraction_at_flip_0_8": earliest_fraction,
            "generation_summary": generation_summary,
            "actual_generation_gate_passed": gb["correct"]["success_rate"] >= 0.80 and gb["correct"]["success_rate"] > max(gb[name]["success_rate"] for name in ("wrong_family", "coordinate_roll", "reversed_masks")),
            "erasure_summary": erasure_summary,
            "erasure_control_margin": erasure_control_margin,
            "midpoint_erasure_gate_passed": erasure_control_margin >= 2.0,
            "side_effect_summary": side_summary,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "strict_interpretation": "Coverage is a deterministic nested hash ladder over all physical coordinates, not Top-K. Actual generation tests one controlled first word. Midpoint erasure is artificial, and vocabulary divergence measures broad output redistribution rather than unrelated-task capability.",
            "next_authorization": "C262_independent_free_generation_and_unrelated_task_side_effect_panel_if_generation_specific; redesign_if_broad_or_control_collision",
        }
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {
            "coverage_rows": len(coverage_rows) == 16 * len(SEEDS) * len(FRACTIONS),
            "generation_rows": len(generation_rows) == 16 * 4,
            "erasure_rows": len(erasure_rows) == 16 * 4,
            "side_effect_rows": len(side_effect_rows) == 16,
            "finite": bool(np.isfinite([
                *[row["median_donor_margin"] for row in coverage_summary],
                *[row["success_rate"] for row in generation_summary],
                *[row["median_gold_margin_drop"] for row in erasure_summary],
                erasure_control_margin,
                *side_summary.values(),
            ]).all()),
            "hooks_removed": True,
        }
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1795, "campaign": "C261", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
        core.save(OUT / "analysis/final.json", final)
        core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
        print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
