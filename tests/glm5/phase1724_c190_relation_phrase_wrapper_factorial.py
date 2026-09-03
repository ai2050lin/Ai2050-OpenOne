#!/usr/bin/env python3
"""C190: separate local relation wording from the global prompt wrapper."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1724_c190_relation_phrase_wrapper_factorial"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C186 = RESULT / "phase1720_c186_new_material_response_ecology_prediction"
C187 = RESULT / "phase1721_c187_vocabulary_paraphrase_failure_decomposition"
C189 = RESULT / "phase1723_c189_campaign_synthesis_extended_heatmap"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1720_c186_new_material_response_ecology_prediction as c186

PHASE, CAMPAIGN = 1724, "C190"
DIM, WIDTH, BATCH = 2560, 224, 8
ROLES = c186.ROLES


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def make_case(family, phrase, unit, nodes, phrase_variant, wrapper_variant, order):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    options, gold = ((f"(A) {d} (B) {f}", 0) if order == 1 else (f"(A) {f} (B) {d}", 1))
    if wrapper_variant == 0:
        prompt = f"Facts: {facts} Following only the stated '{phrase}' links, which target is reachable from {a}? {options}. Reply with only A or B."
    else:
        prompt = f"Registry notes: {facts} Begin at {a} and follow only arrows meaning '{phrase}'. Which registered target can be reached? {options}. Reply with only A or B."
    return {
        "case_id": "",
        "family": family,
        "phrase": phrase,
        "phrase_variant": phrase_variant,
        "wrapper_variant": wrapper_variant,
        "unit": unit,
        "partition": c186.split_for(unit),
        "order": order,
        "gold_position": gold,
        "prompt": prompt,
        "nodes": list(nodes),
        "role_values": {"primary": a, "secondary": d, "relation": phrase, "context": b, "query": a},
    }


def material():
    cases = []
    for family, (phrases, units) in c186.RELATIONS.items():
        for unit, nodes in enumerate(units):
            for phrase_variant, wrapper_variant, order in itertools.product((0, 1), (0, 1), (1, -1)):
                row = make_case(family, phrases[phrase_variant], unit, nodes, phrase_variant, wrapper_variant, order)
                row["case_id"] = f"c190-{len(cases):04d}"
                cases.append(row)
    return cases


def compile_rows(tokenizer, cases):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(candidate) != 1 for candidate in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for row in cases:
        ids = core.chat_ids(tokenizer, "Use only the supplied directed links. Answer exactly A or B.", row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C189 / "audit/independent_final_audit.json")
    cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    checks = {
        "authorization": parent["all_checks_passed"] and "C190" in parent["authorization"],
        "cases": len(cases) == 336,
        "families": len(c186.RELATIONS) == 7,
        "candidate_balance": float(np.mean([row["gold_position"] == 0 for row in cases])) == 0.5,
        "phrase_balance": float(np.mean([row["phrase_variant"] == 1 for row in cases])) == 0.5,
        "wrapper_balance": float(np.mean([row["wrapper_variant"] == 1 for row in cases])) == 0.5,
        "cross_cells": sum(row["phrase_variant"] != row["wrapper_variant"] and row["unit"] in (0, 3) and row["order"] == 1 for row in cases) == 28,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "relation_phrase_x_wrapper_factorial_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": "7 families x 6 lexical units x 2 relation phrases x 2 prompt wrappers x 2 candidate orders",
        "behavior_gates": {"global_min": 0.80, "family_phrase_wrapper_min": 0.75},
        "hidden_policy": "only behavior-correct unit0/unit3 off-diagonal phrase-wrapper anchors; diagonal cells reused unchanged from C186-C187",
        "response": "relation q24, frozen 64 source activation coordinates, q25 six-role x 2560 activation field",
        "analysis": "missing-aware total-variation profile similarity for wrapper, relation-phrase, and vocabulary contrasts",
        "factor_label_margin": 0.01,
        "claim_boundary": "separates local relation wording from global prompt wrapper in one explicit graph task; not an abstract semantic invariant",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "imputation", "post-reveal threshold changes"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_behavior_then_lock",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


@torch.inference_mode()
def behavior():
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    logits = np.lib.format.open_memmap(OUT / "raw/behavior_logits.float32.npy", mode="w+", dtype=np.float32, shape=(len(rows), 2))
    index = []
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                scores = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                logits[start + local] = scores
                index.append({
                    "row_index": start + local,
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "unit": row["unit"],
                    "phrase_variant": row["phrase_variant"],
                    "wrapper_variant": row["wrapper_variant"],
                    "order": row["order"],
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                })
        logits.flush()
        core.write_rows(OUT / "raw/behavior_index.jsonl", index)
        checks = {"rows": len(index) == 336, "finite": bool(np.isfinite(logits).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_behavior_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def lock():
    rows = core.rows(OUT / "raw/behavior_index.jsonl")
    gates = core.load(OUT / "protocol/preregistration.json")["behavior_gates"]
    accuracy = lambda selected: float(np.mean([row["correct"] for row in selected]))
    cell_accuracy = {}
    for family in c186.RELATIONS:
        for phrase, wrapper in itertools.product((0, 1), repeat=2):
            selected = [row for row in rows if row["family"] == family and row["phrase_variant"] == phrase and row["wrapper_variant"] == wrapper]
            cell_accuracy[f"{family}/phrase{phrase}/wrapper{wrapper}"] = accuracy(selected)
    global_accuracy = accuracy(rows)
    eligible_families = [family for family in c186.RELATIONS if global_accuracy >= gates["global_min"] and min(value for key, value in cell_accuracy.items() if key.startswith(f"{family}/")) >= gates["family_phrase_wrapper_min"]]
    anchors, missing = [], []
    for family in c186.RELATIONS:
        for unit in (0, 3):
            for phrase, wrapper in ((0, 1), (1, 0)):
                match = [row for row in rows if row["family"] == family and row["unit"] == unit and row["phrase_variant"] == phrase and row["wrapper_variant"] == wrapper and row["order"] == 1]
                if len(match) != 1:
                    raise RuntimeError((family, unit, phrase, wrapper))
                (anchors if match[0]["correct"] and family in eligible_families else missing).append(match[0])
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "behavior_locked", "global_accuracy": global_accuracy, "cell_accuracy": cell_accuracy, "eligible_families": eligible_families, "anchor_rows": [row["row_index"] for row in anchors], "registered_missing": missing, "authorization": "run_off_diagonal_hidden_responses" if anchors else "close_hidden_not_tested"}
    core.save(OUT / "protocol/behavior_eligibility_lock.json", result)
    checks = {"global": global_accuracy >= gates["global_min"], "eligible_nonempty": bool(eligible_families), "accounting": len(anchors) + len(missing) == 28}
    core.save(OUT / "audit/internal_behavior_lock_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_eligible": bool(anchors)})
    print(json.dumps({"global_accuracy": global_accuracy, "eligible_families": eligible_families, "anchors": len(anchors), "missing": len(missing)}, indent=2))


@torch.inference_mode()
def hidden():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    if not eligibility["anchor_rows"]:
        raise RuntimeError("no eligible anchors")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    selected_rows = [compiled[index] for index in eligibility["anchor_rows"]]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "raw/off_diagonal_relation_response.float16.npy", mode="w+", dtype=np.float16, shape=(len(selected_rows), 64, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def perturb(row, selected, sign, epsilon):
            ids, mask, pos, _ = fixed_base.fixed_batch([row] * len(selected), pad, device, WIDTH)
            captured = {}
            def patch(_module, _args, value):
                state = tensor(value); patched = state.clone()
                for local, coordinate in enumerate(selected):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched
            h1 = base.layers[23].register_forward_hook(patch)
            h2 = base.layers[24].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(selected), 6, DIM), np.float32)
            for local in range(len(selected)):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        index = []
        for anchor_i, row in enumerate(selected_rows):
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH)
            captured = {}
            hook = base.layers[23].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            source = captured["state"][0, row["role_positions"]["relation"]].mean(0).float().cpu().numpy()
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            for start in range(0, 64, 16):
                selected = coordinates[start:start + 16]
                plus = perturb(row, selected, 1.0, epsilon)
                minus = perturb(row, selected, -1.0, epsilon)
                raw[anchor_i, start:start + len(selected)] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
            index.append({"anchor_index": anchor_i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "phrase_variant": row["phrase_variant"], "wrapper_variant": row["wrapper_variant"]})
            raw.flush()
            print(f"[C190] {anchor_i + 1}/{len(selected_rows)} {row['family']} u{row['unit']} p{row['phrase_variant']} w{row['wrapper_variant']}", flush=True)
        core.write_rows(OUT / "raw/response_index.jsonl", index)
        checks = {"shape": list(raw.shape) == [len(selected_rows), 64, 6, DIM], "finite": bool(np.isfinite(raw).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/hidden_run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_hidden_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
    return energy / max(energy.sum(), 1e-30)


def similarity(left, right):
    return float(1.0 - 0.5 * np.abs(left - right).sum())


def load_cells():
    cells = {}
    diagonal = np.load(C186 / "raw/new_relation_role_response.float16.npy", mmap_mode="r")
    for row in core.rows(C186 / "raw/response_anchor_index.jsonl"):
        unit = 0 if row["partition"] == "new_confirmation" else 3
        cells[(row["family"], unit, row["phrase_variant"], row["phrase_variant"])] = np.asarray(diagonal[row["anchor_index"]], dtype=np.float32)
    cross = np.load(C187 / "raw/cross_cell_relation_response.float16.npy", mmap_mode="r")
    for row in core.rows(C187 / "raw/response_index.jsonl"):
        cells[(row["family"], row["unit"], row["phrase_variant"], row["phrase_variant"])] = np.asarray(cross[row["anchor_index"]], dtype=np.float32)
    off = np.load(OUT / "raw/off_diagonal_relation_response.float16.npy", mmap_mode="r")
    for row in core.rows(OUT / "raw/response_index.jsonl"):
        cells[(row["family"], row["unit"], row["phrase_variant"], row["wrapper_variant"])] = np.asarray(off[row["anchor_index"]], dtype=np.float32)
    return cells


def analyze():
    cells = load_cells()
    profiles = {key: profile(value) for key, value in cells.items()}
    wrapper_pairs, phrase_pairs, vocabulary_pairs = [], [], []
    for family in c186.RELATIONS:
        for unit in (0, 3):
            for phrase in (0, 1):
                keys = [(family, unit, phrase, wrapper) for wrapper in (0, 1)]
                if all(key in profiles for key in keys):
                    wrapper_pairs.append({"family": family, "unit": unit, "phrase_variant": phrase, "similarity": similarity(profiles[keys[0]], profiles[keys[1]])})
            for wrapper in (0, 1):
                keys = [(family, unit, phrase, wrapper) for phrase in (0, 1)]
                if all(key in profiles for key in keys):
                    phrase_pairs.append({"family": family, "unit": unit, "wrapper_variant": wrapper, "similarity": similarity(profiles[keys[0]], profiles[keys[1]])})
        for phrase, wrapper in itertools.product((0, 1), repeat=2):
            keys = [(family, unit, phrase, wrapper) for unit in (0, 3)]
            if all(key in profiles for key in keys):
                vocabulary_pairs.append({"family": family, "phrase_variant": phrase, "wrapper_variant": wrapper, "similarity": similarity(profiles[keys[0]], profiles[keys[1]])})
    medians = {
        "same_phrase_cross_wrapper": float(np.median([row["similarity"] for row in wrapper_pairs])),
        "same_wrapper_cross_phrase": float(np.median([row["similarity"] for row in phrase_pairs])),
        "same_surface_cross_vocabulary": float(np.median([row["similarity"] for row in vocabulary_pairs])),
    }
    margin = core.load(OUT / "protocol/preregistration.json")["factor_label_margin"]
    delta = medians["same_phrase_cross_wrapper"] - medians["same_wrapper_cross_phrase"]
    label = "relation_phrase_dominant" if delta >= margin else "global_wrapper_dominant" if delta <= -margin else "phrase_wrapper_entangled_or_small"
    registered_missing = core.rows(C187 / "material/registered_missing_cells.jsonl") + core.load(OUT / "protocol/behavior_eligibility_lock.json")["registered_missing"]
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "factorial_response_atlas_complete",
        "observed_cells": len(cells),
        "possible_cells": 56,
        "medians": medians,
        "wrapper_minus_phrase_similarity": delta,
        "factor_label": label,
        "wrapper_pairs": wrapper_pairs,
        "phrase_pairs": phrase_pairs,
        "vocabulary_pairs": vocabulary_pairs,
        "registered_missing": registered_missing,
        "interpretation": "A larger same-phrase cross-wrapper similarity means the local relation phrase explains more target-profile geometry than the global instruction wrapper. This is a conditional response result, not an abstract semantic coordinate.",
        "next_authorization": "C191_synthesize_response_equivalence_classes_without_new_model_runs",
    }
    core.save(OUT / "analysis/factorial_response_atlas.json", report)
    checks = {"nonempty_cells": len(cells) >= 50, "wrapper_pairs": len(wrapper_pairs) >= 20, "phrase_pairs": len(phrase_pairs) >= 20, "vocabulary_pairs": len(vocabulary_pairs) >= 20, "finite": bool(np.isfinite(list(medians.values()) + [delta]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"observed_cells": len(cells), "medians": medians, "delta": delta, "factor_label": label, "checks": checks}, indent=2))


def amend_missing_accounting():
    if (OUT / "analysis/factorial_response_atlas.json").exists():
        raise RuntimeError("analysis already revealed")
    protocol = core.load(OUT / "protocol/preregistration.json")
    protocol["pre_analysis_accounting_correction"] = "Combine the one C187 diagonal behavior-missing cell with the three C190 off-diagonal behavior-missing cells; no raw response, metric, threshold, or scientific branch changed."
    protocol["producer_sha256"] = core.sha(Path(__file__))
    core.save(OUT / "protocol/preregistration.json", protocol)
    print(json.dumps({"status": "missing_accounting_amended_before_analysis", "producer_sha256": protocol["producer_sha256"]}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/factorial_response_atlas.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "behavior": core.load(OUT / "audit/internal_behavior_run_audit.json")["all_checks_passed"],
        "lock": core.load(OUT / "audit/internal_behavior_lock_audit.json")["all_checks_passed"],
        "hidden": core.load(OUT / "audit/internal_hidden_run_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {key: report[key] for key in ("observed_cells", "possible_cells", "medians", "wrapper_minus_phrase_similarity", "factor_label", "registered_missing")}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "amend", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "behavior": behavior, "lock": lock, "hidden": hidden, "amend": amend_missing_accounting, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
