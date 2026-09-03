#!/usr/bin/env python3
"""C192: prospective response-equivalence test across four language programs."""
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
OUT = RESULT / "phase1726_c192_multi_program_response_equivalence"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C191 = RESULT / "phase1725_c191_response_equivalence_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c192_multi_program_response_equivalence.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1720_c186_new_material_response_ecology_prediction as c186

PHASE, CAMPAIGN = 1726, "C192"
DIM, WIDTH, BATCH = 2560, 224, 8
ROLES = c186.ROLES
PROGRAMS = ("forward_endpoint", "reverse_source", "direct_target", "intermediate_node")
ANCHOR_UNITS = (1, 4)


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def ordered_options(correct, wrong, order):
    return ((f"(A) {correct} (B) {wrong}", 0) if order == 1 else (f"(A) {wrong} (B) {correct}", 1))


def make_case(family, phrase, phrase_variant, unit, nodes, program, order):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    if program == "forward_endpoint":
        options, gold = ordered_options(d, f, order)
        question = f"Starting at {a}, which endpoint can be reached by following the stated links? {options}"
        primary, secondary, context, query = a, d, b, a
    elif program == "reverse_source":
        options, gold = ordered_options(a, e, order)
        question = f"Which starting node can reach {d} by following the stated links? {options}"
        primary, secondary, context, query = a, d, c, d
    elif program == "direct_target":
        options, gold = ordered_options(b, f, order)
        question = f"Which node is linked directly from {a} by one stated step? {options}"
        primary, secondary, context, query = a, b, d, a
    elif program == "intermediate_node":
        options, gold = ordered_options(b, f, order)
        question = f"Which node lies on the stated path from {a} to {d}? {options}"
        primary, secondary, context, query = a, d, b, a
    else:
        raise ValueError(program)
    prompt = f"Relation rule: '{phrase}'. Facts: {facts} {question}. Reply with only A or B."
    return {
        "case_id": "",
        "family": family,
        "phrase": phrase,
        "phrase_variant": phrase_variant,
        "unit": unit,
        "program": program,
        "order": order,
        "gold_position": gold,
        "prompt": prompt,
        "nodes": list(nodes),
        "role_values": {"primary": primary, "secondary": secondary, "relation": phrase, "context": context, "query": query},
    }


def material():
    rows = []
    for family, (phrases, units) in c186.RELATIONS.items():
        for unit, nodes in enumerate(units):
            for phrase_variant, program, order in itertools.product((0, 1), PROGRAMS, (1, -1)):
                row = make_case(family, phrases[phrase_variant], phrase_variant, unit, nodes, program, order)
                row["case_id"] = f"c192-{len(rows):04d}"
                rows.append(row)
    return rows


def compile_rows(tokenizer, rows):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(candidate) != 1 for candidate in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for row in rows:
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
    parent = core.load(C191 / "audit/independent_final_audit.json")
    rows = material(); compiled = compile_rows(graph_base.tokenizer(), rows)
    checks = {
        "authorization": parent["all_checks_passed"] and "C192" in parent["authorization"],
        "cases": len(rows) == 672,
        "program_balance": all(sum(row["program"] == program for row in rows) == 168 for program in PROGRAMS),
        "candidate_balance": float(np.mean([row["gold_position"] == 0 for row in rows])) == 0.5,
        "phrase_balance": float(np.mean([row["phrase_variant"] == 1 for row in rows])) == 0.5,
        "anchor_cells": sum(row["unit"] in ANCHOR_UNITS and row["order"] == 1 for row in rows) == 112,
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "multi_program_response_equivalence_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "design": "7 relation families x 6 lexical units x 2 relation phrases x 4 language programs x 2 candidate orders",
        "programs": list(PROGRAMS),
        "behavior_gates": {"global_min": 0.80, "family_program_min": 0.70},
        "hidden_policy": "behavior-correct order-A anchors from untouched units 1 and 4; ineligible cells registered missing without stopping other families",
        "response": "relation q24, frozen 64 source activation coordinates, q25 six-role x 2560 field",
        "primary_prediction": "nearest neighbor among cells with different program, different lexical unit, and different phrase variant retains relation family",
        "primary_gate": {"same_family_rate_min": 0.70, "advantage_over_available_peer_baseline_min": 0.40, "support_min": 60},
        "claim_boundary": "cross-program response-equivalence in one explicit graph micro-language; not natural-language semantic universality",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "imputation", "post-reveal material or gate changes"],
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
                index.append({"row_index": start + local, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "phrase_variant": row["phrase_variant"], "program": row["program"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
        logits.flush(); core.write_rows(OUT / "raw/behavior_index.jsonl", index)
        checks = {"rows": len(index) == 672, "finite": bool(np.isfinite(logits).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_behavior_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def lock():
    rows = core.rows(OUT / "raw/behavior_index.jsonl"); gates = core.load(OUT / "protocol/preregistration.json")["behavior_gates"]
    accuracy = lambda selected: float(np.mean([row["correct"] for row in selected]))
    global_accuracy = accuracy(rows)
    family_program = {family: {program: accuracy([row for row in rows if row["family"] == family and row["program"] == program]) for program in PROGRAMS} for family in c186.RELATIONS}
    eligible = [family for family in c186.RELATIONS if global_accuracy >= gates["global_min"] and min(family_program[family].values()) >= gates["family_program_min"]]
    anchors, missing = [], []
    for family in c186.RELATIONS:
        for unit, phrase, program in itertools.product(ANCHOR_UNITS, (0, 1), PROGRAMS):
            match = [row for row in rows if row["family"] == family and row["unit"] == unit and row["phrase_variant"] == phrase and row["program"] == program and row["order"] == 1]
            if len(match) != 1:
                raise RuntimeError((family, unit, phrase, program))
            (anchors if match[0]["correct"] and family in eligible else missing).append(match[0])
    result = {"phase": PHASE, "campaign": CAMPAIGN, "status": "behavior_locked", "global_accuracy": global_accuracy, "family_program_accuracy": family_program, "eligible_families": eligible, "anchor_rows": [row["row_index"] for row in anchors], "registered_missing": missing, "authorization": "run_multi_program_hidden_responses" if anchors else "close_hidden_not_tested"}
    core.save(OUT / "protocol/behavior_eligibility_lock.json", result)
    checks = {"global": global_accuracy >= gates["global_min"], "eligible_nonempty": bool(eligible), "accounting": len(anchors) + len(missing) == 112}
    core.save(OUT / "audit/internal_behavior_lock_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_eligible": bool(anchors)})
    print(json.dumps({"global_accuracy": global_accuracy, "eligible_families": eligible, "anchors": len(anchors), "missing": len(missing), "family_program_accuracy": family_program}, indent=2))


@torch.inference_mode()
def hidden():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    if not eligibility["anchor_rows"]:
        raise RuntimeError("no eligible anchors")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl"); rows = [compiled[index] for index in eligibility["anchor_rows"]]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    raw = np.lib.format.open_memmap(OUT / "raw/multi_program_relation_response.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), 64, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def perturb(row, selected, sign, epsilon):
            ids, mask, pos, _ = fixed_base.fixed_batch([row] * len(selected), pad, device, WIDTH); captured = {}
            def patch(_module, _args, value):
                state = tensor(value); patched = state.clone()
                for local, coordinate in enumerate(selected):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched
            h1 = base.layers[23].register_forward_hook(patch); h2 = base.layers[24].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
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
        for anchor_i, row in enumerate(rows):
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH); captured = {}
            hook = base.layers[23].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            source = captured["state"][0, row["role_positions"]["relation"]].mean(0).float().cpu().numpy(); epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            for start in range(0, 64, 16):
                selected = coordinates[start:start + 16]
                raw[anchor_i, start:start + len(selected)] = ((perturb(row, selected, 1.0, epsilon) - perturb(row, selected, -1.0, epsilon)) / (2 * epsilon)).astype(np.float16)
            raw.flush(); index.append({"anchor_index": anchor_i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "phrase_variant": row["phrase_variant"], "program": row["program"]})
            print(f"[C192] {anchor_i + 1}/{len(rows)} {row['family']} {row['program']} u{row['unit']} p{row['phrase_variant']}", flush=True)
        core.write_rows(OUT / "raw/response_index.jsonl", index)
        checks = {"shape": list(raw.shape) == [len(rows), 64, 6, DIM], "finite": bool(np.isfinite(raw).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/hidden_run.json", {"checks": checks, "runtime": placement}); core.save(OUT / "audit/internal_hidden_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks}, indent=2))
    finally:
        raw.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1)); return (energy / max(energy.sum(), 1e-30)).astype(np.float32)


def similarity(left, right):
    return float(1.0 - 0.5 * np.abs(left.astype(np.float64) - right.astype(np.float64)).sum())


def analyze():
    raw = np.load(OUT / "raw/multi_program_relation_response.float16.npy", mmap_mode="r"); index = core.rows(OUT / "raw/response_index.jsonl")
    profiles = np.stack([profile(np.asarray(raw[row["anchor_index"]], dtype=np.float32)) for row in index])
    nearest = []
    for i, row in enumerate(index):
        candidates = [j for j, other in enumerate(index) if other["program"] != row["program"] and other["unit"] != row["unit"] and other["phrase_variant"] != row["phrase_variant"]]
        if not candidates:
            continue
        scores = [similarity(profiles[i], profiles[j]) for j in candidates]; best_local = int(np.argmax(scores)); neighbor = candidates[best_local]
        baseline = sum(index[j]["family"] == row["family"] for j in candidates) / len(candidates)
        nearest.append({"cell_index": i, "neighbor_index": neighbor, "similarity": scores[best_local], "same_family": index[neighbor]["family"] == row["family"], "available_peer_baseline": baseline, "candidate_count": len(candidates)})
    rate = float(np.mean([row["same_family"] for row in nearest])); baseline = float(np.mean([row["available_peer_baseline"] for row in nearest])); advantage = rate - baseline
    gate = core.load(OUT / "protocol/preregistration.json")["primary_gate"]
    passed = len(nearest) >= gate["support_min"] and rate >= gate["same_family_rate_min"] and advantage >= gate["advantage_over_available_peer_baseline_min"]
    by_program = {program: {"support": len(selected), "same_family_rate": float(np.mean([row["same_family"] for row in selected])) if selected else None} for program in PROGRAMS for selected in [[row for row in nearest if index[row["cell_index"]]["program"] == program]]}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "multi_program_equivalence_analyzed", "observed_cells": len(index), "possible_anchor_cells": 112, "registered_missing": core.load(OUT / "protocol/behavior_eligibility_lock.json")["registered_missing"], "constrained_nearest_neighbor": {"support": len(nearest), "same_family_rate": rate, "available_peer_baseline": baseline, "advantage": advantage, "passed": passed}, "by_source_program": by_program, "nearest_rows": nearest, "interpretation": "Candidates differ in program, lexical unit, and relation phrase version. Passing supports task-family organization beyond those registered factors, but remains within one explicit graph micro-language.", "next_authorization": "C193_large_natural_sentence_family_observation_if_C192_passes_else_C193_failure_decomposition"}
    core.save(OUT / "analysis/multi_program_equivalence.json", report)
    variation = np.var(profiles, axis=0)
    payload = {"schema": "c192_multi_program_response_equivalence.v1", "result_type": "multi_program_response_equivalence_heatmap", "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B", "title": "C192 Multi-Program Response Equivalence", "dimensions": list(range(DIM)), "default_coordinates": np.argsort(-variation)[:64].astype(int).tolist(), "rows": [{**row, "label": f"{row['family']} / {row['program']} / unit{row['unit']} / phrase{row['phrase_variant']}", "values": profiles[i].tolist()} for i, row in enumerate(index)], "result": report["constrained_nearest_neighbor"], "by_source_program": by_program, "registered_missing": report["registered_missing"], "coordinate_semantics": "Each row is a normalized q25 target-energy response profile over all 2560 physical activation coordinates.", "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"]}
    PUBLIC.parent.mkdir(parents=True, exist_ok=True); PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "rows": len(index), "schema": payload["schema"]}; core.save(OUT / "analysis/public_asset.json", asset)
    checks = {"support": len(nearest) == len(index), "accounting": len(index) + len(report["registered_missing"]) == 112, "all_2560": profiles.shape == (len(index), DIM), "finite": bool(np.isfinite(profiles).all()), "programs": set(by_program) == set(PROGRAMS)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"result": report["constrained_nearest_neighbor"], "by_program": by_program, "asset": asset, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/multi_program_equivalence.json"); asset = core.load(OUT / "analysis/public_asset.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_run_audit.json")["all_checks_passed"], "lock": core.load(OUT / "audit/internal_behavior_lock_audit.json")["all_checks_passed"], "hidden": core.load(OUT / "audit/internal_hidden_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == asset["sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {key: report[key] for key in ("observed_cells", "possible_anchor_cells", "registered_missing", "constrained_nearest_neighbor", "by_source_program")}, "asset": asset, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "analyze", "close")); args = parser.parse_args(); {"contract": contract, "behavior": behavior, "lock": lock, "hidden": hidden, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
