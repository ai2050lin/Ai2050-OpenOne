#!/usr/bin/env python3
"""C163: call-domain and control adjudication for recipient-only natural graph fields."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1697_c163_natural_graph_call_domain"
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C160 = RESULT / "phase1694_c160_recipient_only_counterfactual_prediction"
C162 = RESULT / "phase1696_c162_linguistic_program_field"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1697, "C163"
DIM, WIDTH, BATCH = 2560, 256, 4
LATE = tuple(range(24, 35))
ROLES = c159.ROLES
CONTROL_MODES = ("baseline", "selected", "exact", "reverse", "wrong_role", "wrong_coordinate", "wrong_relation", "random_same_norm")


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def fresh_rows():
    return [row for row in core.rows(C159 / "analysis/late_half_difference_index.jsonl") if row["partition"] == "fresh"]


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    c160 = core.load(C160 / "audit/independent_final_audit.json")
    c162 = core.load(C162 / "audit/independent_final_audit.json")
    rows = fresh_rows()
    checks = {"authorization": c160["all_checks_passed"] and c162["all_checks_passed"], "recipient_only_pass": c160["scientific_fresh_passed"], "rows": len(rows) == 256, "panels": all(sum(row["panel"] == panel for row in rows) == 128 for panel in c159.PANELS), "relations": all(sum(row["relation_family"] == relation for row in rows) == 64 for relation in ("is_a", "part_of", "located_in", "precedes")), "controls": len(CONTROL_MODES) == 8}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/fresh_interventions.jsonl", rows)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "natural_graph_call_domain_contract_frozen", "model": "Qwen3-4B BF16 CUDA nonquantized", "direction": "2 times C160 selected recipient-only half-difference prediction", "checkpoint_selection": "maximum mean donor-margin gain on isomorphic_nonce panel only", "confirmation": "natural_lexical panel only", "checkpoints": list(LATE), "controls": list(CONTROL_MODES), "natural_gates": {"mean_gain_min": 0.0, "positive_gain_rate_min": 0.60, "donor_choice_increase_min": 0.10, "paired_win_over_each_wrong_control_min": 0.60}, "free_generation_cases": 64, "side_effect": "top-20 noncandidate logit-set overlap and target-gain/noncandidate-change ratio", "claim_boundary": "recipient-only natural-lexical call domain if passed; not training-time formation or a minimal circuit", "forbidden": ["attention", "MLP", "weights", "PCA", "natural-panel checkpoint selection"], "source_hashes": {"C160": core.sha(C160 / "analysis/fresh_selected_predictions.float16.npy"), "C159": core.sha(C159 / "analysis/late_half_difference.float16.npy")}, "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C163_nonce_checkpoint_curve"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps(checks, indent=2))


def patch_forward(model, device, pad, rows, vectors, checkpoint, return_full=False):
    ids, mask, pos, lengths = fixed_base.fixed_batch(rows, pad, device, WIDTH)
    values = torch.from_numpy(np.asarray(vectors)).to(device=device, dtype=torch.float32) if vectors is not None else None

    def patch(_module, _args, output):
        hidden = tensor(output)
        patched = hidden.clone()
        for local, row in enumerate(rows):
            for role_i, role in enumerate(ROLES):
                delta = values[local, role_i].to(dtype=patched.dtype)
                for position in row["role_positions"][role]:
                    patched[local, position] += delta
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = model.model.layers[checkpoint - 1].register_forward_hook(patch) if vectors is not None else None
    try:
        output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
    finally:
        if handle is not None: handle.remove()
    last = torch.stack([output.logits[i, lengths[i] - 1] for i in range(len(rows))])
    scores = np.asarray([[float(last[i, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(rows)], np.float32)
    return scores, last.float().cpu().numpy() if return_full else None


@torch.inference_mode()
def run_curve():
    pairs = fresh_rows()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    rows = [compiled[row["minus_row"]] for row in pairs]
    pred = np.load(C160 / "analysis/fresh_selected_predictions.float16.npy", mmap_mode="r")
    scores = np.zeros((12, 256, 2), np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, 256, BATCH):
            ids = np.arange(start, min(start + BATCH, 256))
            scores[0, ids], _ = patch_forward(model, device, pad, rows[start:start + BATCH], None, 24)
        for qi, q in enumerate(LATE):
            for start in range(0, 256, BATCH):
                ids = np.arange(start, min(start + BATCH, 256))
                scores[qi + 1, ids], _ = patch_forward(model, device, pad, rows[start:start + BATCH], 2.0 * np.asarray(pred[ids, qi], np.float32), q)
            print(f"[C163-curve] q{q}", flush=True)
    finally:
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "raw/checkpoint_curve_logits.float32.npy", scores)
    donor = np.asarray([compiled[row["plus_row"]]["gold_position"] for row in pairs], np.int64)
    margins = np.asarray([[value[i, donor[i]] - value[i, 1 - donor[i]] for i in range(256)] for value in scores])
    gains = margins - margins[0]
    nonce = np.asarray([i for i, row in enumerate(pairs) if row["panel"] == "isomorphic_nonce"])
    curve = [{"q": q, "nonce_mean_gain": float(np.mean(gains[qi + 1, nonce])), "nonce_positive_rate": float(np.mean(gains[qi + 1, nonce] > 0))} for qi, q in enumerate(LATE)]
    selected = max(curve, key=lambda row: (row["nonce_mean_gain"], -row["q"]))["q"]
    lock = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "selection_panel": "isomorphic_nonce", "curve": curve, "selected_checkpoint": selected, "natural_confirmation_unread": True, "authorization": "run_C163_natural_controls"}
    core.save(OUT / "protocol/nonce_checkpoint_selection_lock.json", lock)
    checks = {"shape": list(scores.shape) == [12, 256, 2], "finite": bool(np.isfinite(scores).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/curve_run.json", {"checks": checks, "runtime": placement, "lock": lock})
    core.save(OUT / "audit/internal_curve_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": lock["authorization"]})
    print(json.dumps(lock, indent=2))


def control_vectors(selected_q):
    pairs = fresh_rows()
    qi = LATE.index(selected_q)
    pred = 2.0 * np.asarray(np.load(C160 / "analysis/fresh_selected_predictions.float16.npy", mmap_mode="r")[:, qi], np.float32)
    exact = 2.0 * np.asarray(np.load(C159 / "analysis/late_half_difference.float16.npy", mmap_mode="r")[[row["pair_index"] for row in pairs], qi], np.float32)
    wrong_relation = np.empty_like(pred)
    for panel in c159.PANELS:
        ids = [i for i, row in enumerate(pairs) if row["panel"] == panel]
        wrong_relation[ids] = pred[np.roll(ids, 32)]
    rng = np.random.default_rng(1697)
    random = rng.standard_normal(pred.shape, dtype=np.float32)
    random *= (np.linalg.norm(pred.reshape(256, -1), axis=1) / np.maximum(np.linalg.norm(random.reshape(256, -1), axis=1), 1e-12))[:, None, None]
    return {"selected": pred, "exact": exact, "reverse": -pred, "wrong_role": np.roll(pred, 1, axis=1), "wrong_coordinate": np.roll(pred, 1, axis=2), "wrong_relation": wrong_relation, "random_same_norm": random}


@torch.inference_mode()
def run_controls():
    lock = core.load(OUT / "protocol/nonce_checkpoint_selection_lock.json")
    q = int(lock["selected_checkpoint"])
    pairs = fresh_rows()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    rows = [compiled[row["minus_row"]] for row in pairs]
    vectors = control_vectors(q)
    scores = np.zeros((len(CONTROL_MODES), 256, 2), np.float32)
    collateral = []
    generations = []
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for mi, mode in enumerate(CONTROL_MODES):
            for start in range(0, 256, BATCH):
                ids_ = np.arange(start, min(start + BATCH, 256))
                values = None if mode == "baseline" else vectors[mode][ids_]
                scores[mi, ids_], _ = patch_forward(model, device, pad, rows[start:start + BATCH], values, q)
            print(f"[C163-control] {mode}", flush=True)
        natural_ids = [i for i, row in enumerate(pairs) if row["panel"] == "natural_lexical"]
        for start in range(0, len(natural_ids), BATCH):
            local_ids = natural_ids[start:start + BATCH]
            batch = [rows[i] for i in local_ids]
            base_scores, base_full = patch_forward(model, device, pad, batch, None, q, True)
            selected_scores, selected_full = patch_forward(model, device, pad, batch, vectors["selected"][local_ids], q, True)
            for local, global_i in enumerate(local_ids):
                candidate_ids = {int(value[0]) for value in batch[local]["candidate_ids"]}
                top = [int(value) for value in np.argsort(base_full[local])[-24:][::-1] if int(value) not in candidate_ids][:20]
                selected_top = set(int(value) for value in np.argsort(selected_full[local])[-24:][::-1] if int(value) not in candidate_ids)
                collateral.append({"pair_index": global_i, "top20_overlap": len(set(top) & selected_top) / 20.0, "top20_mean_abs_change": float(np.mean(np.abs(selected_full[local, top] - base_full[local, top])))})
        for global_i in natural_ids[:64]:
            row = rows[global_i]
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            values = torch.from_numpy(vectors["selected"][global_i:global_i + 1]).to(device=device, dtype=torch.float32)
            max_position = max(position for role in ROLES for position in row["role_positions"][role])

            def patch(_module, _args, output):
                hidden = tensor(output)
                if hidden.shape[1] <= max_position:
                    return output
                patched = hidden.clone()
                for role_i, role in enumerate(ROLES):
                    for position in row["role_positions"][role]:
                        patched[0, position] += values[0, role_i].to(dtype=patched.dtype)
                return (patched,) + output[1:] if isinstance(output, tuple) else patched

            handle = model.model.layers[q - 1].register_forward_hook(patch)
            try:
                generated = model.generate(input_ids=ids, attention_mask=mask, position_ids=pos, max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=pad)
            finally:
                handle.remove()
            text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            expected = "A" if compiled[pairs[global_i]["plus_row"]]["gold_position"] == 0 else "B"
            generations.append({"pair_index": global_i, "text": text, "expected": expected, "correct": text.upper().startswith(expected)})
    finally:
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/control_logits.float32.npy", scores)
    core.write_rows(OUT / "raw/natural_collateral.jsonl", collateral)
    core.write_rows(OUT / "raw/natural_free_generation.jsonl", generations)
    checks = {"shape": list(scores.shape) == [8, 256, 2], "finite": bool(np.isfinite(scores).all()), "collateral": len(collateral) == 128, "generation": len(generations) == 64, "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/control_run.json", {"checks": checks, "runtime": placement, "authorization": "analyze_C163"})
    core.save(OUT / "audit/internal_control_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "analyze_C163"})
    print(json.dumps(checks, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/nonce_checkpoint_selection_lock.json")
    pairs = fresh_rows()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    scores = np.load(OUT / "raw/control_logits.float32.npy")
    donor = np.asarray([compiled[row["plus_row"]]["gold_position"] for row in pairs], np.int64)
    margins = np.asarray([[value[i, donor[i]] - value[i, 1 - donor[i]] for i in range(256)] for value in scores])
    gains = margins - margins[0]
    natural = np.asarray([i for i, row in enumerate(pairs) if row["panel"] == "natural_lexical"])
    reports = {}
    baseline_choice = np.argmax(scores[0], axis=1)
    for mi, mode in enumerate(CONTROL_MODES):
        reports[mode] = {"natural_mean_gain": float(np.mean(gains[mi, natural])), "natural_positive_rate": float(np.mean(gains[mi, natural] > 0)), "natural_donor_choice_rate": float(np.mean(np.argmax(scores[mi, natural], axis=1) == donor[natural])), "natural_donor_choice_increase": float(np.mean(np.argmax(scores[mi, natural], axis=1) == donor[natural]) - np.mean(baseline_choice[natural] == donor[natural]))}
    paired = {mode: float(np.mean(gains[CONTROL_MODES.index("selected"), natural] > gains[CONTROL_MODES.index(mode), natural])) for mode in ("reverse", "wrong_role", "wrong_coordinate", "wrong_relation", "random_same_norm")}
    g = protocol["natural_gates"]
    selected = reports["selected"]
    gates = {"gain": selected["natural_mean_gain"] > g["mean_gain_min"], "rate": selected["natural_positive_rate"] >= g["positive_gain_rate_min"], "choice": selected["natural_donor_choice_increase"] >= g["donor_choice_increase_min"], "controls": all(value >= g["paired_win_over_each_wrong_control_min"] for value in paired.values())}
    collateral = core.rows(OUT / "raw/natural_collateral.jsonl")
    generation = core.rows(OUT / "raw/natural_free_generation.jsonl")
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "natural_graph_call_domain_adjudicated", "selected_checkpoint": lock["selected_checkpoint"], "mode_reports": reports, "paired_win_rates": paired, "gates": gates, "natural_call_gate_passed": all(gates.values()), "free_generation_accuracy": float(np.mean([row["correct"] for row in generation])), "side_effect": {"median_top20_overlap": float(np.median([row["top20_overlap"] for row in collateral])), "median_top20_mean_abs_change": float(np.median([row["top20_mean_abs_change"] for row in collateral]))}, "claim_boundary": protocol["claim_boundary"], "next_authorization": "C164 three-model free-interface qualification"}
    core.save(OUT / "analysis/call_domain.json", report)
    checks = {"modes": len(reports) == 8, "controls": len(paired) == 5, "generation": len(generation) == 64, "finite": bool(np.isfinite(scores).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_natural_call_passed": all(gates.values()), "authorization": report["next_authorization"]})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/call_domain.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "curve": core.load(OUT / "audit/internal_curve_run_audit.json")["all_checks_passed"], "controls": core.load(OUT / "audit/internal_control_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"checkpoint": report["selected_checkpoint"], "selected": report["mode_reports"]["selected"], "paired": report["paired_win_rates"], "gates": report["gates"], "free_generation_accuracy": report["free_generation_accuracy"], "side_effect": report["side_effect"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C164"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "run_curve": run_curve, "run_controls": run_controls, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run_curve|run_controls|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
