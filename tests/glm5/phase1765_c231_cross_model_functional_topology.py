#!/usr/bin/env python3
"""C231: sequential cross-model observation of relative role/checkpoint topology."""
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

import phase1757_c223_surface_transport_common as common
import phase1748_c214_cross_model_functional_isomorphism as c214
from model_utils import get_model_info

core = common.core
OUT = common.OUTS["C231"]
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = {"qwen3": "strict_chat", "glm4": "strict_chat", "deepseek7b": "plain"}
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C230"] / "audit/independent_final_audit.json")
    source_rows = core.rows(common.OUTS["C223"] / "material/cases.jsonl")
    rows = [row for row in source_rows if row["family"] in common.TARGET_FAMILIES and row["unit"] == 6]
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 160, "hidden_rows": sum(row["order"] == 1 for row in rows) == 80, "family_surface_balance": set(sum(row["family"] == family and row["surface"] == surface for row in rows) for family in common.TARGET_FAMILIES for surface in common.SURFACES) == {8}}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    protocol = {
        "phase": 1765, "campaign": "C231", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cross_model_functional_topology_frozen", "models": list(MODELS), "interfaces": INTERFACES,
        "interface_source": "model-specific interfaces selected prospectively in independent C214, not on C231 rows",
        "sequential_loading": True, "behavior_gate": {"global_min": 0.65, "each_family_min": 0.50},
        "hidden_panel": "five target families x four surfaces x four factorial cells at unit 6 and order +1 = 80",
        "topology": "five relative-depth checkpoints x six role means; compare normalized transition energy after role centering",
        "null": "all 720 role permutations", "cross_model_gate": {"models_min": 2, "common_correct_rows_min": 40, "all_pair_centered_cosine_min": 0.30, "all_pair_exact_upper_p_max": 0.05},
        "claim_boundary": "Only relative functional topology is compared. Physical coordinate ids, widths, weights, attention and MLP are not aligned.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_qwen3_then_glm4_then_deepseek7b_sequentially_then_analyze",
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
            for i, (_row, _candidate_i, values, _prompt_length, _candidate) in enumerate(expanded):
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
            del ids_tensor, mask, output, logp
        core.write_rows(OUT / f"raw/{model_name}_behavior.jsonl", behavior)
        global_accuracy = float(np.mean([row["correct"] for row in behavior]))
        by_family = {family: float(np.mean([row["correct"] for row in behavior if row["family"] == family])) for family in common.TARGET_FAMILIES}
        gate = core.load(OUT / "protocol/preregistration.json")["behavior_gate"]
        eligible = global_accuracy >= gate["global_min"] and min(by_family.values()) >= gate["each_family_min"]
        hidden_rows = [row for row in rows if row["order"] == 1]
        states = np.lib.format.open_memmap(OUT / f"raw/{model_name}_topology_states.float16.npy", mode="w+", dtype=np.float16, shape=(80, 5, len(common.ROLES), info.d_model))
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
            checkpoints = sorted(set(int(round(fraction * (len(output.hidden_states) - 1))) for fraction in (0, 0.25, 0.5, 0.75, 1)))
            if len(checkpoints) != 5:
                raise RuntimeError((model_name, checkpoints))
            for checkpoint_i, checkpoint in enumerate(checkpoints):
                for role_i, role in enumerate(common.ROLES):
                    states[case_i, checkpoint_i, role_i] = output.hidden_states[checkpoint][0, role_positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            hidden_index.append({"case_index": case_i, "case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "checkpoints": checkpoints, "behavior_correct": next(value["correct"] for value in behavior if value["case_id"] == row["case_id"])})
            if case_i % 20 == 0 or case_i == 79:
                print(f"[C231] {model_name} hidden {case_i + 1}/80", flush=True)
        states.flush()
        core.write_rows(OUT / f"raw/{model_name}_topology_index.jsonl", hidden_index)
        report = {"phase": 1765, "campaign": "C231", "model": model_name, "interface": interface, "global_accuracy": global_accuracy, "by_family_accuracy": by_family, "behavior_eligible": eligible, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started}
        core.save(OUT / f"analysis/{model_name}.json", report)
        checks = {"behavior_rows": len(behavior) == 160, "hidden_rows": len(hidden_index) == 80, "state_shape": list(states.shape) == [80, 5, 6, info.d_model], "finite": bool(np.isfinite(states).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"model": model_name, "accuracy": global_accuracy, "by_family": by_family, "eligible": eligible, "checks": checks}, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


def analyze() -> None:
    reports = {model: core.load(OUT / f"analysis/{model}.json") for model in MODELS}
    participants = [model for model in MODELS if reports[model]["behavior_eligible"]]
    indexes = {model: core.rows(OUT / f"raw/{model}_topology_index.jsonl") for model in participants}
    common_ids = sorted(set.intersection(*({row["case_id"] for row in indexes[model] if row["behavior_correct"]} for model in participants))) if participants else []
    topologies = {}
    for model in participants:
        states = np.load(OUT / f"raw/{model}_topology_states.float16.npy", mmap_mode="r")
        selected = [row["case_index"] for row in indexes[model] if row["case_id"] in set(common_ids)]
        transition = np.diff(np.asarray(states[selected], np.float32), axis=1)
        energy = np.sqrt(np.mean(np.square(transition, dtype=np.float64), axis=-1)).mean(axis=0)
        energy /= np.maximum(energy.sum(axis=1, keepdims=True), 1e-30)
        topologies[model] = energy
    permutations = list(itertools.permutations(range(6)))
    pairs = []
    for i, left_name in enumerate(participants):
        for right_name in participants[i + 1:]:
            left = topologies[left_name] - topologies[left_name].mean(axis=1, keepdims=True)
            right = topologies[right_name] - topologies[right_name].mean(axis=1, keepdims=True)
            observed = common.previous.cosine(left, right)
            null = np.asarray([common.previous.cosine(left, right[:, permutation]) for permutation in permutations])
            pairs.append({"models": [left_name, right_name], "centered_cosine": observed, "null_median": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": float((1 + np.sum(null >= observed)) / (1 + len(null)))})
    gate = core.load(OUT / "protocol/preregistration.json")["cross_model_gate"]
    passed = len(participants) >= gate["models_min"] and len(common_ids) >= gate["common_correct_rows_min"] and bool(pairs) and min(row["centered_cosine"] for row in pairs) >= gate["all_pair_centered_cosine_min"] and max(row["exact_upper_p"] for row in pairs) <= gate["all_pair_exact_upper_p_max"]
    report = {"phase": 1765, "campaign": "C231", "status": "cross_model_functional_topology_adjudicated", "models": reports, "participants": participants, "common_correct_rows": len(common_ids), "topologies": {model: value.tolist() for model, value in topologies.items()}, "pair_tests": pairs, "cross_model_gate_passed": passed, "interpretation": "The comparison is dimension-free role/checkpoint energy topology. It does not align physical coordinates or prove a shared code.", "next_authorization": "C232_joint_theory_and_mathematics_upgrade_adjudication"}
    core.save(OUT / "analysis/cross_model_summary.json", report)
    checks = {"models": set(reports) == set(MODELS), "participants": set(participants) <= set(MODELS), "common_rows": len(common_ids) <= 80, "pairs": len(pairs) == len(participants) * (len(participants) - 1) // 2, "finite": bool(np.isfinite([row[k] for row in pairs for k in ("centered_cosine", "null_median", "null_q95", "exact_upper_p")]).all()) if pairs else True}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"participants": participants, "common_correct": len(common_ids), "pairs": pairs, "passed": passed}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/cross_model_summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "models": all(core.load(OUT / f"audit/internal_{model}_audit.json")["all_checks_passed"] for model in MODELS), "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1765, "campaign": "C231", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "model", "analyze", "close"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "contract": contract()
    elif args.command == "model":
        if not args.model: raise SystemExit("--model required")
        run_model(args.model)
    elif args.command == "analyze": analyze()
    else: close()


if __name__ == "__main__":
    main()

