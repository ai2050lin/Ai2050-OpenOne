#!/usr/bin/env python3
"""C242: sequential cross-model comparison of abstract factorial event graphs."""
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

import phase1768_c234_event_campaign_common as common
import phase1748_c214_cross_model_functional_isomorphism as c214
from model_utils import get_model_info

core = common.core
OUT = common.OUTS["C242"]
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = {"qwen3": "strict_chat", "glm4": "strict_chat", "deepseek7b": "plain"}
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def selected_rows() -> list[dict]:
    rows = core.rows(common.OUTS["C234"] / "material/cases.jsonl")
    return [row for row in rows if (row["surface"] == "chronicle" and row["unit"] == 7) or (row["surface"] == "dispatch" and row["unit"] == 9)]


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C241"] / "audit/independent_final_audit.json")
    rows = selected_rows()
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C242"), "behavior_rows": len(rows) == 80, "hidden_rows": sum(row["order"] == 1 for row in rows) == 40, "families": {row["family"] for row in rows} == set(common.FAMILIES)}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    protocol = {
        "phase": 1776, "campaign": "C242", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cross_model_abstract_event_graph_frozen",
        "models": list(MODELS), "interfaces": INTERFACES, "sequential_loading": True,
        "panel": "five families x two unseen surfaces/lexicons x four factorial cells x two candidate orders",
        "hidden_panel": "order +1 only, 40 rows",
        "behavior_gate": {"global_min": 0.65, "each_family_min": 0.50},
        "graph": "family x factorial effect x five relative-depth checkpoints x six semantic roles; RMS over each model's complete physical coordinate width",
        "null": "all 720 role permutations",
        "cross_model_gate": {"models_min": 2, "all_participant_pairs_cosine_min": 0.30, "all_participant_pairs_permutation_p_max": 0.05},
        "claim_boundary": "No physical coordinate ids, widths, weights, attention, or MLP states are aligned.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_qwen3_then_glm4_then_deepseek7b_sequentially",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


@torch.inference_mode()
def run_model(model_name: str) -> None:
    if (OUT / f"analysis/{model_name}.json").exists():
        raise RuntimeError(f"{model_name} already run")
    rows = core.rows(OUT / "material/cases.jsonl")
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
        complete_groups = {
            family: sum(
                all(next(row["correct"] for row in behavior if row["family"] == family and row["surface"] == surface and row["factor_a"] == a and row["factor_b"] == b and row["order"] == 1) for a, b in itertools.product((0, 1), repeat=2))
                for surface in ("chronicle", "dispatch")
            )
            for family in common.FAMILIES
        }
        eligible = global_accuracy >= 0.65 and min(by_family.values()) >= 0.50 and min(complete_groups.values()) >= 1

        hidden_rows = [row for row in rows if row["order"] == 1]
        n_checkpoints = info.n_layers + 1
        states = np.lib.format.open_memmap(OUT / f"raw/{model_name}_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(40, n_checkpoints, len(common.ROLES), info.d_model))
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
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != n_checkpoints:
                raise RuntimeError((model_name, len(output.hidden_states), n_checkpoints))
            for q, state in enumerate(output.hidden_states):
                for role_i, role in enumerate(common.ROLES):
                    states[case_i, q, role_i] = state[0, role_positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            correct = next(value["correct"] for value in behavior if value["case_id"] == row["case_id"])
            hidden_index.append({"hidden_index": case_i, "case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "behavior_correct": correct})
            if case_i % 10 == 0 or case_i == 39:
                print(f"[C242] {model_name} hidden {case_i + 1}/40", flush=True)
        states.flush()
        core.write_rows(OUT / f"raw/{model_name}_hidden_index.jsonl", hidden_index)
        report = {"model": model_name, "global_accuracy": global_accuracy, "by_family_accuracy": by_family, "complete_factorial_groups": complete_groups, "behavior_eligible": eligible, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started}
        core.save(OUT / f"analysis/{model_name}.json", report)
        checks = {"behavior": len(behavior) == 80, "hidden": len(hidden_index) == 40, "shape": states.shape == (40, n_checkpoints, 6, info.d_model), "finite": bool(np.isfinite(states[:, :, :, ::64]).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"model": model_name, "accuracy": global_accuracy, "by_family": by_family, "complete_groups": complete_groups, "eligible": eligible, "checks": checks}, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


def abstract_graph(model_name: str) -> np.ndarray:
    states = np.load(OUT / f"raw/{model_name}_role_states.float16.npy", mmap_mode="r")
    index = core.rows(OUT / f"raw/{model_name}_hidden_index.jsonl")
    key = {(row["family"], row["surface"], row["factor_a"], row["factor_b"]): row["hidden_index"] for row in index if row["behavior_correct"]}
    checkpoints = sorted(set(int(round(x * (states.shape[1] - 1))) for x in (0, 0.25, 0.5, 0.75, 1.0)))
    graph = np.zeros((5, 3, 5, 6), np.float64)
    for family_i, family in enumerate(common.FAMILIES):
        effect_rows = []
        for surface in ("chronicle", "dispatch"):
            needed = [(family, surface, a, b) for a, b in itertools.product((0, 1), repeat=2)]
            if not all(item in key for item in needed):
                continue
            cells = {(a, b): np.asarray(states[key[(family, surface, a, b)]], np.float32) for a, b in itertools.product((0, 1), repeat=2)}
            effect_rows.append(common.factorial_effect(cells))
        if not effect_rows:
            raise RuntimeError((model_name, family, "no behavior-correct factorial group"))
        effects = np.mean(effect_rows, axis=0)
        sampled = effects[:, checkpoints]
        energy = np.sqrt(np.mean(np.square(sampled, dtype=np.float64), axis=-1))
        energy /= np.maximum(energy.sum(axis=-1, keepdims=True), 1e-30)
        graph[family_i] = energy
    return graph


def analyze() -> None:
    reports = {model: core.load(OUT / f"analysis/{model}.json") for model in MODELS}
    participants = [model for model in MODELS if reports[model]["behavior_eligible"]]
    graphs = {model: abstract_graph(model) for model in participants}
    permutations = list(itertools.permutations(range(6)))
    pairs = []
    for i, left_name in enumerate(participants):
        for right_name in participants[i + 1:]:
            left = graphs[left_name] - graphs[left_name].mean(axis=-1, keepdims=True)
            right = graphs[right_name] - graphs[right_name].mean(axis=-1, keepdims=True)
            observed = common.cosine(left, right)
            null = np.asarray([common.cosine(left, right[:, :, :, permutation]) for permutation in permutations])
            pairs.append({"models": [left_name, right_name], "centered_cosine": observed, "null_median": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null)))})
    gate = core.load(OUT / "protocol/preregistration.json")["cross_model_gate"]
    passed = len(participants) >= gate["models_min"] and bool(pairs) and min(row["centered_cosine"] for row in pairs) >= gate["all_participant_pairs_cosine_min"] and max(row["exact_upper_p"] for row in pairs) <= gate["all_participant_pairs_permutation_p_max"]
    report = {"phase": 1776, "campaign": "C242", "status": "cross_model_abstract_event_graph_adjudicated", "models": reports, "participants": participants, "graphs": {name: value.tolist() for name, value in graphs.items()}, "pair_tests": pairs, "cross_model_gate_passed": passed, "interpretation": "The graph compares relative role energy across factors and depths, not physical coordinate identity.", "next_authorization": "C243_joint_theory_mathematics_heatmap_and_campaign_closure"}
    core.save(OUT / "analysis/summary.json", report)
    checks = {"models": set(reports) == set(MODELS), "pairs": len(pairs) == len(participants) * (len(participants) - 1) // 2, "finite": bool(np.isfinite([row[key] for row in pairs for key in ("centered_cosine", "null_median", "null_q95", "exact_upper_p")]).all()) if pairs else True}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"participants": participants, "pairs": pairs, "passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "models": all(core.load(OUT / f"audit/internal_{model}_audit.json")["all_checks_passed"] for model in MODELS), "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1776, "campaign": "C242", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "model", "analyze", "close"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "contract": contract()
    elif args.command == "model":
        if not args.model:
            raise SystemExit("--model required")
        run_model(args.model)
    elif args.command == "analyze": analyze()
    else: close()


if __name__ == "__main__":
    main()
