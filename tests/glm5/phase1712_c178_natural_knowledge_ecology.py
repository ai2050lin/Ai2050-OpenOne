#!/usr/bin/env python3
"""C178: behavior-qualified eight-family natural knowledge-ecology response atlas."""
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
OUT = RESULT / "phase1712_c178_natural_knowledge_ecology"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C177 = RESULT / "phase1711_c177_missing_aware_broad_family_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1712, "C178"
DIM, WIDTH, BATCH = 2560, 224, 8
STATES = tuple(range(38))
ROLES = c159.ROLES
PARTITIONS = ("discovery", "confirmation", "fresh")
RELATIONS = {
    "is_a": ("is a kind of", [
        ("apple", "fruit", "food", "item", "wrench", "tool"), ("robin", "bird", "animal", "organism", "salmon", "fish"),
        ("sedan", "vehicle", "machine", "artifact", "spoon", "utensil"), ("oak", "tree", "plant", "living thing", "granite", "rock"),
        ("violin", "instrument", "artifact", "object", "hammer", "tool"), ("tulip", "flower", "plant", "organism", "whale", "mammal")]),
    "part_of": ("is a component of", [
        ("key", "keyboard", "computer", "workstation", "wheel", "bicycle"), ("page", "chapter", "book", "collection", "handle", "door"),
        ("cell", "tissue", "organ", "body", "brick", "wall"), ("blade", "rotor", "turbine", "engine", "seat", "car"),
        ("pixel", "image", "screen", "display", "button", "panel"), ("verse", "stanza", "poem", "anthology", "string", "guitar")]),
    "located_in": ("is located inside", [
        ("coin", "purse", "drawer", "desk", "cup", "cabinet"), ("fish", "tank", "room", "building", "plant", "pot"),
        ("file", "folder", "drive", "computer", "key", "box"), ("seed", "fruit", "basket", "pantry", "book", "shelf"),
        ("village", "county", "state", "country", "boat", "harbor"), ("planet", "system", "galaxy", "universe", "chair", "house")]),
    "precedes": ("comes before", [
        ("dawn", "morning", "noon", "evening", "seed", "tree"), ("draft", "review", "approval", "release", "winter", "spring"),
        ("infancy", "childhood", "adulthood", "retirement", "start", "finish"), ("spark", "flame", "fire", "ash", "cloud", "rain"),
        ("intake", "compression", "combustion", "exhaust", "bud", "flower"), ("premise", "argument", "conclusion", "decision", "question", "answer")]),
    "causes": ("directly causes", [
        ("spark", "ignition", "combustion", "motion", "wind", "rain"), ("pressure", "crack", "leak", "flood", "heat", "melting"),
        ("infection", "fever", "fatigue", "rest", "exercise", "strength"), ("friction", "heat", "expansion", "failure", "cold", "ice"),
        ("impact", "dent", "fracture", "collapse", "light", "shadow"), ("rain", "runoff", "erosion", "sediment", "sunlight", "growth")]),
    "depends_on": ("depends on", [
        ("application", "library", "runtime", "system", "lamp", "switch"), ("report", "dataset", "sensor", "power", "door", "hinge"),
        ("harvest", "rain", "cloud", "ocean", "bridge", "road"), ("service", "database", "storage", "network", "wheel", "axle"),
        ("flight", "engine", "fuel", "refinery", "clock", "spring"), ("analysis", "sample", "instrument", "calibration", "roof", "beam")]),
    "reports_to": ("reports to", [
        ("intern", "analyst", "manager", "director", "pilot", "tower"), ("clerk", "supervisor", "chief", "board", "nurse", "doctor"),
        ("cadet", "officer", "captain", "admiral", "student", "teacher"), ("assistant", "editor", "publisher", "owner", "driver", "station"),
        ("researcher", "leader", "dean", "provost", "actor", "audience"), ("technician", "engineer", "architect", "executive", "chef", "guest")]),
    "derives_from": ("derives from", [
        ("translation", "manuscript", "archive", "collection", "signal", "antenna"), ("theorem", "lemma", "axiom", "framework", "river", "lake"),
        ("plastic", "polymer", "monomer", "feedstock", "bread", "grain"), ("forecast", "measurement", "sensor", "instrument", "song", "guitar"),
        ("summary", "article", "study", "dataset", "shadow", "light"), ("model", "algorithm", "principle", "theory", "painting", "canvas")]),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def partition(unit):
    return PARTITIONS[unit // 2]


def make_case(family, phrase, unit, nodes, truth, surface, codebook):
    a, b, c, d, e, f = nodes
    edges = [(a, b), (b, c), (c, d), (e, f)] if truth == 1 else [(a, b), (c, b), (c, d), (e, f)]
    facts = ". ".join(f"{x} {phrase} {y}" for x, y in edges) + "."
    if surface == 1:
        prompt = f"Facts: {facts} Following only the stated '{phrase}' links, can {a} reach {d}? (A) yes (B) no. Reply with only A or B."
    else:
        prompt = f"Registry notes: {facts} Starting at {a} and following only '{phrase}' arrows, is {d} reachable? (A) yes (B) no. Reply with only A or B."
    yes_position = 0 if codebook == 1 else 1
    gold = yes_position if truth == 1 else 1 - yes_position
    return {
        "case_id": "",
        "family": family,
        "phrase": phrase,
        "unit": unit,
        "unit_id": f"c178-{family}-{unit:02d}",
        "partition": partition(unit),
        "truth": truth,
        "surface": surface,
        "codebook": codebook,
        "gold_position": gold,
        "prompt": prompt,
        "semantic_edges": edges,
        "nodes": list(nodes),
        "role_values": {"primary": a, "secondary": d, "relation": phrase, "context": b, "query": a},
    }


def material():
    cases = []
    for family, (phrase, units) in RELATIONS.items():
        for unit, nodes in enumerate(units):
            for truth, surface, codebook in itertools.product((1, -1), repeat=3):
                row = make_case(family, phrase, unit, nodes, truth, surface, codebook)
                row["case_id"] = f"c178-{len(cases):04d}"
                cases.append(row)
    return cases


def compile_rows(tokenizer, cases):
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(x) != 1 for x in candidates):
        raise RuntimeError(candidates)
    system = "Use only the supplied directed links. Answer exactly A or B."
    compiled = []
    for row in cases:
        ids = core.chat_ids(tokenizer, system, row["prompt"])
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
    parent = core.load(C177 / "audit/independent_final_audit.json")
    cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    cells = {(r["family"], r["unit"], r["truth"], r["surface"], r["codebook"]) for r in cases}
    first_choice = float(np.mean([r["gold_position"] == 0 for r in cases]))
    checks = {
        "authorization": parent["all_checks_passed"],
        "cases": len(cases) == 384,
        "families": len(RELATIONS) == 8,
        "cells": len(cells) == 384,
        "partitions": all(sum(r["partition"] == p for r in cases) == 128 for p in PARTITIONS),
        "candidate_balance": first_choice == 0.5,
        "truth_balance": float(np.mean([r["truth"] == 1 for r in cases])) == 0.5,
        "roles": all(set(r["role_positions"]) == set(ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) < WIDTH,
        "semantic_uniqueness": all((r["truth"] == 1) == ((r["nodes"][1], r["nodes"][2]) in r["semantic_edges"]) for r in cases),
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
        "status": "natural_knowledge_ecology_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "families": list(RELATIONS),
        "cases": len(cases),
        "behavior_gates": {"global_min": 0.80, "family_partition_min": 0.75, "all_frozen_anchors_correct": True},
        "hidden_policy": "no HiddenState capture or perturbation before behavior eligibility lock",
        "eligible_capture": "embedding, 36 block outputs, final norm; six roles for all eligible rows; all tokens for 24 anchors if all families qualify",
        "response": "q24 source-role locked 64 coordinates to q25 six-role full field, symmetric finite difference",
        "source_roles": ["primary", "query", "relation"],
        "naturalness": "human-readable controlled registry English; machine-audited only, not an independent human naturalness claim",
        "claim_boundary": "explicit multi-hop language ecology, not unrestricted world knowledge or spontaneous generation",
        "forbidden": ["attention", "MLP", "weights", "PCA", "hidden capture before behavior lock"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C178_behavior_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "max_width": max(len(r["prompt_ids"]) for r in compiled)}, indent=2))


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
                score = [float(output.logits[local, lengths[local] - 1, c[0]]) for c in row["candidate_ids"]]
                logits[start + local] = score
                pred = int(score[1] > score[0])
                index.append({"row_index": start + local, "case_id": row["case_id"], "family": row["family"], "partition": row["partition"], "unit": row["unit"], "truth": row["truth"], "surface": row["surface"], "codebook": row["codebook"], "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"]})
        logits.flush()
    finally:
        logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/behavior_index.jsonl", index)
    checks = {"rows": len(index) == 384, "finite": bool(np.isfinite(logits).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_behavior_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def lock():
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior_rows = core.rows(OUT / "raw/behavior_index.jsonl")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    accuracy = lambda xs: float(np.mean([x["correct"] for x in xs]))
    by_family_partition = {family: {part: accuracy([x for x in behavior_rows if x["family"] == family and x["partition"] == part]) for part in PARTITIONS} for family in RELATIONS}
    anchor_rows = []
    for part in PARTITIONS:
        for family in RELATIONS:
            choices = [i for i, r in enumerate(rows) if r["partition"] == part and r["family"] == family and r["truth"] == 1 and r["surface"] == 1 and r["codebook"] == 1 and r["unit"] % 2 == 0]
            if len(choices) != 1:
                raise RuntimeError((part, family, len(choices)))
            anchor_rows.append(choices[0])
    anchor_correct = {i: behavior_rows[i]["correct"] for i in anchor_rows}
    eligible = [family for family in RELATIONS if min(by_family_partition[family].values()) >= protocol["behavior_gates"]["family_partition_min"] and all(anchor_correct[i] for i in anchor_rows if rows[i]["family"] == family)]
    global_accuracy = accuracy(behavior_rows)
    global_pass = global_accuracy >= protocol["behavior_gates"]["global_min"]
    if not global_pass:
        eligible = []
    result = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "global_accuracy": global_accuracy, "by_family_partition": by_family_partition, "anchor_rows": anchor_rows, "anchor_correct": anchor_correct, "eligible_families": eligible, "all_families_eligible": len(eligible) == len(RELATIONS), "authorization": "run_C178_hidden_and_response" if eligible else "C178_typed_not_tested_then_C179"}
    core.save(OUT / "protocol/behavior_eligibility_lock.json", result)
    checks = {"global": global_pass, "anchors": len(anchor_rows) == 24, "eligible_nonempty": bool(eligible)}
    core.save(OUT / "audit/internal_behavior_lock_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_eligible": bool(eligible)})
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def hidden():
    protocol = core.load(OUT / "protocol/preregistration.json")
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    if not eligibility["eligible_families"]:
        raise RuntimeError("no eligible family")
    all_rows = core.rows(OUT / "compiled/qwen3.jsonl")
    eligible_indices = [i for i, r in enumerate(all_rows) if r["family"] in eligibility["eligible_families"]]
    rows = [all_rows[i] for i in eligible_indices]
    anchors = [i for i in eligibility["anchor_rows"] if all_rows[i]["family"] in eligibility["eligible_families"]]
    anchor_offsets, offset = {}, 0
    for i in anchors:
        anchor_offsets[i] = (offset, offset + len(all_rows[i]["prompt_ids"])); offset += len(all_rows[i]["prompt_ids"])
    role_raw = np.lib.format.open_memmap(OUT / "raw/eligible_six_role_all_checkpoint.bf16.npy", mode="w+", dtype=np.uint16, shape=(len(rows), 6, 38, DIM))
    token_raw = np.lib.format.open_memmap(OUT / "raw/anchor_all_token_all_checkpoint.bf16.npy", mode="w+", dtype=np.uint16, shape=(38, offset, DIM))
    response = np.lib.format.open_memmap(OUT / "raw/anchor_role_response.float16.npy", mode="w+", dtype=np.float16, shape=(3, len(anchors), 64, 6, DIM))
    epsilons = np.zeros((3, len(anchors)), np.float32)
    lock = core.load(C173 / "protocol/role_specific_coordinate_lock.json")
    coordinate_sets = {"primary": lock["roles"]["primary"]["coordinates"], "query": lock["roles"]["query"]["coordinates"], "relation": core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]}
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def capture_batch(batch):
            cap = {}
            hooks = [base.embed_tokens.register_forward_hook(lambda _m, _a, o: cap.__setitem__(0, tensor(o).detach()))]
            hooks += [layer.register_forward_hook(lambda _m, _a, o, q=i + 1: cap.__setitem__(q, tensor(o).detach())) for i, layer in enumerate(base.layers)]
            hooks += [base.norm.register_forward_hook(lambda _m, _a, o: cap.__setitem__(37, tensor(o).detach()))]
            try:
                ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            return cap

        global_to_local = {g: i for i, g in enumerate(eligible_indices)}
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            cap = capture_batch(batch)
            for local, row in enumerate(batch):
                gi = eligible_indices[start + local]
                for q in STATES:
                    state = cap[q][local]
                    for role_i, role in enumerate(ROLES):
                        role_raw[start + local, role_i, q] = state[row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
                    if gi in anchor_offsets:
                        begin, end = anchor_offsets[gi]
                        token_raw[q, begin:end] = state[:end - begin].contiguous().view(torch.uint16).cpu().numpy()
            if start % 64 == 0: role_raw.flush(); token_raw.flush(); print(f"[C178-hidden] {start + len(batch)}/{len(rows)}", flush=True)

        def perturb(row, source_role, coordinates, sign, epsilon):
            batch = [row] * len(coordinates)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}
            def patch(_m, _a, value):
                h = tensor(value); p = h.clone()
                for local, coordinate in enumerate(coordinates):
                    for position in row["role_positions"][source_role]: p[local, position, int(coordinate)] += sign * epsilon
                return (p,) + value[1:] if isinstance(value, tuple) else p
            h1 = base.layers[23].register_forward_hook(patch)
            h2 = base.layers[24].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(coordinates), 6, DIM), np.float32)
            for local in range(len(coordinates)):
                for role_i, role in enumerate(ROLES): field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for source_i, source_role in enumerate(("primary", "query", "relation")):
            coordinates = np.asarray(coordinate_sets[source_role], int)
            role_i = ROLES.index(source_role)
            for anchor_i, gi in enumerate(anchors):
                row = all_rows[gi]
                local_i = global_to_local[gi]
                source = role_raw[local_i, role_i, 24].view(np.uint16)
                source = torch.from_numpy(source.copy()).view(torch.bfloat16).float().numpy()
                epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
                epsilons[source_i, anchor_i] = epsilon
                for start in range(0, 64, 16):
                    cs = coordinates[start:start + 16]
                    plus = perturb(row, source_role, cs, 1.0, epsilon)
                    minus = perturb(row, source_role, cs, -1.0, epsilon)
                    response[source_i, anchor_i, start:start + len(cs)] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
            response.flush(); print(f"[C178-response] {source_role}", flush=True)
    finally:
        role_raw.flush(); token_raw.flush(); response.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/anchor_epsilons.float32.npy", epsilons)
    core.write_rows(OUT / "raw/eligible_row_index.jsonl", [{"local_index": i, "row_index": gi, "case_id": all_rows[gi]["case_id"]} for i, gi in enumerate(eligible_indices)])
    core.write_rows(OUT / "raw/anchor_index.jsonl", [{"anchor_index": i, "row_index": gi, "case_id": all_rows[gi]["case_id"], "family": all_rows[gi]["family"], "partition": all_rows[gi]["partition"], "token_offset_start": anchor_offsets[gi][0], "token_offset_end": anchor_offsets[gi][1]} for i, gi in enumerate(anchors)])
    checks = {"role_shape": list(role_raw.shape) == [len(rows), 6, 38, DIM], "token_shape": list(token_raw.shape) == [38, offset, DIM], "response_shape": list(response.shape) == [3, len(anchors), 64, 6, DIM], "epsilon": bool(np.all(epsilons > 0)), "finite": bool(np.isfinite(response).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/hidden_run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_hidden_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def response_metrics(pred, actual):
    p = pred.reshape(len(pred), -1).astype(np.float64)
    a = actual.reshape(len(actual), -1).astype(np.float64)
    nrmse = np.linalg.norm(a - p, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    threshold = np.quantile(np.abs(p), 0.95, axis=1)
    sign = [float(np.mean(np.sign(a[i, np.abs(p[i]) >= threshold[i]]) == np.sign(p[i, np.abs(p[i]) >= threshold[i]]))) for i in range(len(p))]
    return nrmse, np.asarray(sign)


def analyze():
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    anchors = core.rows(OUT / "raw/anchor_index.jsonl")
    response = np.load(OUT / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    families = eligibility["eligible_families"]
    anchor_lookup = {(r["partition"], r["family"]): r["anchor_index"] for r in anchors}
    rows = []
    for source_i, source_role in enumerate(("primary", "query", "relation")):
        for family_i, family in enumerate(families):
            pred = np.asarray(response[source_i, anchor_lookup[("discovery", family)]], np.float32)
            for partition in ("confirmation", "fresh"):
                actual = np.asarray(response[source_i, anchor_lookup[(partition, family)]], np.float32)
                nrmse, sign = response_metrics(pred, actual)
                perm_nrmse = response_metrics(np.roll(pred, 1, axis=0), actual)[0]
                wrong_family = families[(family_i + 1) % len(families)]
                wrong = np.asarray(response[source_i, anchor_lookup[("discovery", wrong_family)]], np.float32)
                wrong_nrmse = response_metrics(wrong, actual)[0]
                rows.append({"source_role": source_role, "family": family, "partition": partition, "median_signed_nrmse": float(np.median(nrmse)), "active_sign_agreement": float(np.median(sign)), "source_permutation_advantage": float(np.median(perm_nrmse - nrmse)), "wrong_family_advantage": float(np.median(wrong_nrmse - nrmse))})
    summary = {}
    for source_role in ("primary", "query", "relation"):
        summary[source_role] = {}
        for partition in ("confirmation", "fresh"):
            selected = [r for r in rows if r["source_role"] == source_role and r["partition"] == partition]
            summary[source_role][partition] = {k: float(np.median([r[k] for r in selected])) for k in ("median_signed_nrmse", "active_sign_agreement", "source_permutation_advantage", "wrong_family_advantage")}
        fresh = summary[source_role]["fresh"]
        summary[source_role]["externality_label"] = "replicated" if fresh["active_sign_agreement"] >= 0.55 and fresh["source_permutation_advantage"] >= 0.02 and fresh["wrong_family_advantage"] >= 0.02 else "not_replicated"
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "natural_knowledge_ecology_adjudicated", "behavior": {"global_accuracy": eligibility["global_accuracy"], "eligible_families": families, "by_family_partition": eligibility["by_family_partition"]}, "response_summary": summary, "rows": rows, "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"], "next_authorization": "run_C179_cross_model_functional_eligibility_then_C180_synthesis_heatmap"}
    core.save(OUT / "analysis/natural_ecology_atlas.json", report)
    checks = {"families": len(families) > 0, "rows": len(rows) == 3 * len(families) * 2, "finite": all(np.isfinite([v for k, v in r.items() if isinstance(v, float)]).all() for r in rows)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/natural_ecology_atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_run_audit.json")["all_checks_passed"], "lock": core.load(OUT / "audit/internal_behavior_lock_audit.json")["all_checks_passed"], "hidden": core.load(OUT / "audit/internal_hidden_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"behavior": report["behavior"], "response_summary": report["response_summary"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "behavior", "lock", "hidden", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "behavior": behavior, "lock": lock, "hidden": hidden, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
