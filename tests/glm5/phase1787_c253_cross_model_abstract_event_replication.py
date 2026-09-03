#!/usr/bin/env python3
"""C253: sequential third-material replication of the coarse cross-model role-depth graph."""
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
import phase1748_c214_cross_model_functional_isomorphism as c214
from model_utils import get_model_info

core = common.core
OUT = common.OUTS["C253"]
INTERFACES = {"qwen3": "strict_chat", "glm4": "strict_chat", "deepseek7b": "plain"}
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def selected_rows() -> list[dict]:
    rows = core.rows(common.OUTS["C247"] / "material/cases.jsonl")
    return [row for row in rows if row["panel"] == "core" and ((row["surface"] == "case_review" and row["unit"] == 0) or (row["surface"] == "radio_summary" and row["unit"] == 1))]


@torch.inference_mode()
def run_model(model_name: str, rows: list[dict]) -> dict:
    interface = INTERFACES[model_name]
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16(model_name)
        quant = common.previous.quantization_audit(model)
        info = get_model_info(model, model_name)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        compiled = [(row, *c214.compile_interface(tokenizer, row, interface)) for row in rows]
        behavior = []
        for start in range(0, len(compiled), BATCH[model_name]):
            batch = compiled[start:start + BATCH[model_name]]
            expanded = []
            for row, ids, candidates in batch:
                for candidate_i, candidate in enumerate(candidates):
                    expanded.append((row, candidate_i, ids + candidate, len(ids), candidate))
            width = max(len(item[2]) for item in expanded)
            ids_tensor = torch.full((len(expanded), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids_tensor)
            for i, (_row, _ci, values, _pl, _candidate) in enumerate(expanded):
                ids_tensor[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[i, :len(values)] = 1
            output = model(input_ids=ids_tensor, attention_mask=mask, use_cache=False, return_dict=True)
            logp = torch.log_softmax(output.logits.float(), dim=-1)
            scores = np.zeros((len(batch), 2), np.float32)
            for i, (_row, candidate_i, _values, prompt_length, candidate) in enumerate(expanded):
                scores[i // 2, candidate_i] = sum(float(logp[i, prompt_length + offset - 1, token_id]) for offset, token_id in enumerate(candidate))
            for local, (row, _ids, _candidates) in enumerate(batch):
                prediction = int(scores[local, 1] > scores[local, 0])
                behavior.append({"case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
            del output, logp, ids_tensor, mask
        core.write_rows(OUT / f"raw/{model_name}_behavior.jsonl", behavior)
        global_accuracy = float(np.mean([row["correct"] for row in behavior]))
        by_family = {family: float(np.mean([row["correct"] for row in behavior if row["family"] == family])) for family in common.FAMILIES}
        eligible = global_accuracy >= 0.65 and min(by_family.values()) >= 0.50
        hidden_rows = [row for row in rows if row["order"] == 1]
        n_checkpoints = info.n_layers + 1
        states = np.lib.format.open_memmap(OUT / f"raw/{model_name}_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(40, n_checkpoints, 6, info.d_model))
        hidden_index = []
        for case_i, row in enumerate(hidden_rows):
            ids, _candidates = c214.compile_interface(tokenizer, row, interface)
            role_positions = {}
            for role, value in row["role_values"].items():
                spans = common.graph_base.name_spans(tokenizer, ids, value)
                if not spans:
                    raise RuntimeError((model_name, row["case_id"], role, value))
                role_positions[role] = spans[-1] if role == "query" else spans[0]
            role_positions["boundary"] = [len(ids) - 1]
            ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            output = model(input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor), use_cache=False, return_dict=True, output_hidden_states=True)
            for q, state in enumerate(output.hidden_states):
                for role_i, role in enumerate(common.ROLES):
                    states[case_i, q, role_i] = state[0, role_positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            hidden_index.append({"hidden_index": case_i, "case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "behavior_correct": next(item["correct"] for item in behavior if item["case_id"] == row["case_id"])})
            if case_i % 10 == 0 or case_i == 39:
                print(f"[C253] {model_name} hidden {case_i + 1}/40", flush=True)
        states.flush()
        core.write_rows(OUT / f"raw/{model_name}_hidden_index.jsonl", hidden_index)
        report = {"model": model_name, "global_accuracy": global_accuracy, "by_family_accuracy": by_family, "behavior_eligible": eligible, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started}
        core.save(OUT / f"analysis/{model_name}.json", report)
        audit = {"behavior": len(behavior) == 80, "hidden": len(hidden_index) == 40, "shape": states.shape == (40, n_checkpoints, 6, info.d_model), "finite": bool(np.isfinite(states[:, :, :, ::64]).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": audit, "all_checks_passed": all(audit.values())})
        return report
    finally:
        common.previous.release(model)
        gc.collect()


def graph(model_name: str) -> np.ndarray:
    states = np.load(OUT / f"raw/{model_name}_role_states.float16.npy", mmap_mode="r")
    index = core.rows(OUT / f"raw/{model_name}_hidden_index.jsonl")
    key = {(r["family"], r["surface"], r["factor_a"], r["factor_b"]): r["hidden_index"] for r in index if r["behavior_correct"]}
    checkpoints = [int(round(x * (states.shape[1] - 1))) for x in (0, .25, .5, .75, 1.)]
    result = np.zeros((5, 3, 5, 6), np.float64)
    for fi, family in enumerate(common.FAMILIES):
        effects = []
        for surface in common.SURFACES:
            needed = [(family, surface, a, b) for a, b in itertools.product((0, 1), repeat=2)]
            if all(item in key for item in needed):
                cells = {(a, b): np.asarray(states[key[(family, surface, a, b)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
                effects.append(common.factorial_effect(cells))
        if not effects:
            continue
        effect = np.mean(effects, axis=0)[:, checkpoints]
        energy = np.sqrt(np.mean(np.square(effect, dtype=np.float64), axis=-1))
        result[fi] = energy / np.maximum(energy.sum(axis=-1, keepdims=True), 1e-30)
    return result


def cosine(a, b):
    return common.old.cosine(a, b)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C252"] / "audit/independent_final_audit.json")
    rows = selected_rows()
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C253"), "rows": len(rows) == 80, "hidden_rows": sum(r["order"] == 1 for r in rows) == 40, "sequential": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {"phase": 1787, "campaign": "C253", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "cross_model_replication_frozen", "models": list(common.MODELS), "interfaces": INTERFACES, "sequential_loading": True, "graph": "family x factorial effect x five relative depths x six role-span RMS energies, normalized across roles", "gate": {"models_min": 2, "pair_cosine_min": 0.30, "role_permutation_p_max": 0.05}, "claim_boundary": "No physical coordinates, signs, widths, tokenizations, attention, MLP, or weights are aligned.", "producer_sha256": core.sha(Path(__file__)), "authorization": "run_three_models_sequentially"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    reports = {name: run_model(name, rows) for name in common.MODELS}
    participants = [name for name, report in reports.items() if report["behavior_eligible"]]
    graphs = {name: graph(name) for name in participants}
    permutations = list(itertools.permutations(range(6)))
    pairs = []
    for i, left_name in enumerate(participants):
        for right_name in participants[i + 1:]:
            left = graphs[left_name] - graphs[left_name].mean(axis=-1, keepdims=True)
            right = graphs[right_name] - graphs[right_name].mean(axis=-1, keepdims=True)
            observed = cosine(left, right)
            null = np.asarray([cosine(left, right[..., permutation]) for permutation in permutations])
            pairs.append({"models": [left_name, right_name], "centered_cosine": observed, "null_q95": float(np.quantile(null, .95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / 721)})
    old_graphs = core.load(common.OLD["C242"] / "analysis/summary.json")["graphs"]
    within_model = []
    for name in participants:
        if name in old_graphs:
            within_model.append({"model": name, "old_to_third_cosine": cosine(np.asarray(old_graphs[name]), graphs[name])})
    passed = len(participants) >= 2 and bool(pairs) and min(r["centered_cosine"] for r in pairs) >= .30 and max(r["exact_upper_p"] for r in pairs) <= .05
    report = {"phase": 1787, "campaign": "C253", "status": "cross_model_replication_adjudicated", "models": reports, "participants": participants, "graphs": {k: v.tolist() for k, v in graphs.items()}, "pair_tests": pairs, "within_model_old_to_third": within_model, "cross_model_gate_passed": passed, "strict_interpretation": "This is an independent coarse role-depth topology replication. It cannot identify cross-model coordinate code or causal isomorphism.", "next_authorization": "C254_heatmap_and_C255_theory"}
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"models": set(reports) == set(common.MODELS), "pairs": len(pairs) == len(participants) * (len(participants) - 1) // 2, "finite": bool(np.isfinite([r[k] for r in pairs for k in ("centered_cosine", "null_q95", "exact_upper_p")]).all()) if pairs else True}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": True, "models": all(core.load(OUT / f"audit/internal_{name}_audit.json")["all_checks_passed"] for name in common.MODELS), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1787, "campaign": "C253", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
