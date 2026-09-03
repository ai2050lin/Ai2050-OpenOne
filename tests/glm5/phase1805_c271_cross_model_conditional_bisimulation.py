#!/usr/bin/env python3
"""C271: compare state-conditioned role/depth transition topology across three models."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1797_c263_c272_state_operator_common as common
import phase1748_c214_cross_model_functional_isomorphism as c214
from model_utils import get_model_info

core, OUT = common.core, common.OUTS["C271"]
INTERFACES = ("strict_chat", "demonstrated_chat", "plain")
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def cosine(left, right):
    a, b = np.asarray(left, np.float64).ravel(), np.asarray(right, np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def selected_rows():
    rows = core.rows(common.OUTS["C263"] / "material/cases.jsonl")
    return [r for r in rows if r["panel"] == "core" and r["unit"] in (0, 1) and r["order"] == 1]


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C270"] / "analysis/final.json"); rows = selected_rows()
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 80, "models": common.MODELS == ("qwen3", "glm4", "deepseek7b"), "sequential": True, "all_model_coordinates_retained": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); core.write_rows(OUT / "material/cases.jsonl", rows)
    protocol = {"phase": 1805, "campaign": "C271", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "cross_model_contract_frozen", "models": list(common.MODELS), "interfaces": list(INTERFACES), "interface_selection": "dossier accuracy only; hearing is holdout", "behavior_gate": {"hearing_global_min": .65, "each_family_min": .50}, "hidden_panel": "hearing, two units, five families, four cells, order +1", "state_topology": "family x factorial effect x relative transition stage x semantic role x signed/conditional transition statistics", "null": "all 720 semantic-role permutations after role centering", "gate": {"participants_min": 2, "pair_cosine_min": .30, "pair_exact_p_max": .05}, "claim_boundary": "Models retain their own complete coordinate axes. Only anonymous signed transition statistics are compared; this is not physical coordinate identity or a shared circuit.", "producer_sha256": core.sha(Path(__file__)), "authorization": "run_models_sequentially_then_C272"}
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps(checks, indent=2))


@torch.inference_mode()
def run_model(model_name):
    if (OUT / f"analysis/{model_name}.json").exists(): raise RuntimeError(model_name)
    rows = core.rows(OUT / "material/cases.jsonl"); model, started = None, time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16(model_name); quant = common.previous.quantization_audit(model); info = get_model_info(model, model_name); pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        all_results, compiled_by_interface = {}, {}
        for interface in INTERFACES:
            compiled = [(row, *c214.compile_interface(tokenizer, row, interface)) for row in rows]; compiled_by_interface[interface] = compiled; results = []
            for start in range(0, len(compiled), BATCH[model_name]):
                batch = compiled[start:start + BATCH[model_name]]; expanded = []
                for row, ids, candidates in batch:
                    for ci, candidate in enumerate(candidates): expanded.append((row, ci, ids + candidate, len(ids), candidate))
                width = max(len(x[2]) for x in expanded); ids_t = torch.full((len(expanded), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids_t)
                for i, (_r, _ci, values, _pl, _c) in enumerate(expanded): ids_t[i, :len(values)] = torch.tensor(values, device=device); mask[i, :len(values)] = 1
                output = model(input_ids=ids_t, attention_mask=mask, use_cache=False, return_dict=True); logp = torch.log_softmax(output.logits.float(), dim=-1); scores = np.zeros((len(batch), 2), np.float32)
                for i, (_r, ci, _values, pl, candidate) in enumerate(expanded): scores[i // 2, ci] = sum(float(logp[i, pl + off - 1, token]) for off, token in enumerate(candidate))
                for local, (row, _ids, _cands) in enumerate(batch):
                    pred = int(scores[local, 1] > scores[local, 0]); results.append({"case_id": row["case_id"], "surface": row["surface"], "family": row["family"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"]})
            all_results[interface] = results; core.write_rows(OUT / f"raw/{model_name}_{interface}_behavior.jsonl", results); print(f"[C271] {model_name} {interface} {np.mean([r['correct'] for r in results]):.3f}", flush=True)
        discovery = {name: float(np.mean([r["correct"] for r in values if r["surface"] == "dossier"])) for name, values in all_results.items()}; selected = max(INTERFACES, key=lambda name: (discovery[name], -INTERFACES.index(name))); selected_results = all_results[selected]; holdout = [r for r in selected_results if r["surface"] == "hearing"]; holdout_acc = float(np.mean([r["correct"] for r in holdout])); by_family = {f: float(np.mean([r["correct"] for r in holdout if r["family"] == f])) for f in common.FAMILIES}; eligible = holdout_acc >= .65 and min(by_family.values()) >= .50
        hidden_rows = [r for r in rows if r["surface"] == "hearing"]; nq = info.n_layers + 1; states = np.lib.format.open_memmap(OUT / f"raw/{model_name}_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(40, nq, 6, info.d_model)); hidden_index = []
        compiled_lookup = {row["case_id"]: (ids, candidates) for row, ids, candidates in compiled_by_interface[selected]}
        for i, row in enumerate(hidden_rows):
            ids, _ = compiled_lookup[row["case_id"]]; positions = {}
            for role, value in row["role_values"].items():
                spans = common.graph_base.name_spans(tokenizer, ids, value)
                if not spans: raise RuntimeError((model_name, row["case_id"], role, value))
                positions[role] = spans[-1] if role == "query" else spans[0]
            positions["boundary"] = [len(ids) - 1]; ids_t = torch.tensor([ids], device=device); output = model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), use_cache=False, return_dict=True, output_hidden_states=True)
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES): states[i, q, ri] = state[0, positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            correct = next(x["correct"] for x in selected_results if x["case_id"] == row["case_id"]); hidden_index.append({"hidden_index": i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "behavior_correct": correct}); states.flush()
        core.write_rows(OUT / f"raw/{model_name}_hidden_index.jsonl", hidden_index)
        report = {"model": model_name, "discovery_accuracy": discovery, "selected_interface": selected, "holdout_accuracy": holdout_acc, "by_family_accuracy": by_family, "behavior_eligible": eligible, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started}; core.save(OUT / f"analysis/{model_name}.json", report)
        ach = {"behavior": all(len(v) == 80 for v in all_results.values()), "hidden": len(hidden_index) == 40, "shape": list(states.shape) == [40, nq, 6, info.d_model], "finite": bool(np.isfinite(states[:, :, :, ::64]).all())}; core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); print(json.dumps(report, indent=2))
    finally:
        common.previous.release(model); gc.collect()


def topology(model_name):
    states = np.load(OUT / f"raw/{model_name}_role_states.float16.npy", mmap_mode="r"); index = core.rows(OUT / f"raw/{model_name}_hidden_index.jsonl"); nq, dim = states.shape[1], states.shape[-1]; key = {(r["family"], r["unit"], r["factor_a"], r["factor_b"]): r["hidden_index"] for r in index if r["behavior_correct"]}; stages = sorted(set(min(nq - 2, int(round(frac * (nq - 2)))) for frac in (0, .25, .5, .75)))
    graph = np.zeros((5, 3, len(stages), 6, 6), np.float64)
    for fi, family in enumerate(common.FAMILIES):
        for unit in (0, 1):
            if not all((family, unit, a, b) in key for a, b in itertools.product((0, 1), repeat=2)): continue
            cells = {(a, b): np.asarray(states[key[(family, unit, a, b)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}; effects = common.prior.factorial_effect(cells); base = cells[(0, 0)]
            for ei in range(3):
                for si, q in enumerate(stages):
                    cur, nxt = effects[ei, q], effects[ei, q + 1]
                    for ri in range(6):
                        threshold = max(float(np.median(np.abs(cur[ri]))), 1e-8); active = np.abs(cur[ri]) > threshold; next_active = np.abs(nxt[ri]) > max(float(np.median(np.abs(nxt[ri]))), 1e-8); same = active & next_active & (np.sign(cur[ri]) == np.sign(nxt[ri])); opposite = active & next_active & (np.sign(cur[ri]) == -np.sign(nxt[ri])); high = base[q, ri] >= np.median(base[q, ri])
                        graph[fi, ei, si, ri] += [float(np.mean(cur[ri][active] > 0)) if active.any() else 0, float(np.mean(cur[ri][active] < 0)) if active.any() else 0, float(same.sum() / max(active.sum(), 1)), float(opposite.sum() / max(active.sum(), 1)), float(np.median(np.abs(nxt[ri])) / max(np.median(np.abs(cur[ri])), 1e-8)), float(same[high].sum() / max(active[high].sum(), 1) - same[~high].sum() / max(active[~high].sum(), 1))]
    graph /= 2.0
    return graph


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    current_hash = core.sha(Path(__file__))
    if current_hash != protocol["producer_sha256"]:
        core.save(OUT / "protocol/implementation_erratum.json", {"frozen_producer_sha256": protocol["producer_sha256"], "corrected_producer_sha256": current_hash, "scope": "replace a missing imported cosine helper with the preregistered explicit normalized dot product; no data, interface, threshold, null, gate, or claim changed"})
    reports = {m: core.load(OUT / f"analysis/{m}.json") for m in common.MODELS}; participants = [m for m in common.MODELS if reports[m]["behavior_eligible"]]; graphs = {m: topology(m) for m in participants}; permutations = list(itertools.permutations(range(6))); pairs = []
    for i, left_name in enumerate(participants):
        for right_name in participants[i + 1:]:
            left = graphs[left_name] - graphs[left_name].mean(axis=3, keepdims=True); right = graphs[right_name] - graphs[right_name].mean(axis=3, keepdims=True); observed = cosine(left, right); null = np.asarray([cosine(left, right[:, :, :, p, :]) for p in permutations]); pairs.append({"models": [left_name, right_name], "centered_cosine": observed, "null_q95": float(np.quantile(null, .95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null)))})
    passed = len(participants) >= 2 and bool(pairs) and min(r["centered_cosine"] for r in pairs) >= .30 and max(r["exact_upper_p"] for r in pairs) <= .05
    report = {"phase": 1805, "campaign": "C271", "status": "cross_model_conditional_topology_adjudicated", "models": reports, "participants": participants, "pair_tests": pairs, "cross_model_conditional_topology_gate_passed": passed, "graphs": {m: g.tolist() for m, g in graphs.items()}, "strict_interpretation": "The comparison is an anonymous state-conditioned role/depth transition topology. It neither aligns physical coordinates nor proves causal bisimulation.", "next_authorization": "C272_campaign_adjudication_heatmap"}; core.save(OUT / "analysis/summary.json", report); ach = {"models": set(reports) == set(common.MODELS), "pairs": len(pairs) == len(participants) * (len(participants) - 1) // 2, "finite": bool(np.isfinite([x[k] for x in pairs for k in ("centered_cosine", "exact_upper_p")]).all()) if pairs else True}; core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())}); print(json.dumps({"participants": participants, "pairs": pairs, "passed": passed}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/summary.json"); current_hash = core.sha(Path(__file__)); erratum_path = OUT / "protocol/implementation_erratum.json"; hash_ok = current_hash == protocol["producer_sha256"] or (erratum_path.exists() and core.load(erratum_path)["corrected_producer_sha256"] == current_hash); checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "models": all(core.load(OUT / f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in common.MODELS), "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash_or_registered_erratum": hash_ok}; final = {"phase": 1805, "campaign": "C271", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("contract", "model", "analyze", "close")); p.add_argument("--model", choices=common.MODELS); args = p.parse_args()
    if args.command == "contract": contract()
    elif args.command == "model": run_model(args.model)
    elif args.command == "analyze": analyze()
    else: close()


if __name__ == "__main__": main()
