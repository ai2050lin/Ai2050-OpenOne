#!/usr/bin/env python3
"""C214: sequential cross-model behavior interfaces and nontrivial role-topology null."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1739_c205_response_ecology_common as common
import phase1571_c098_observation_first_graph_campaign as graph_base
from model_utils import get_model_info

core = common.core
OUT = common.C214
PHASE, CAMPAIGN = 1748, "C214"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("strict_chat", "demonstrated_chat", "plain")
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def render_chat(tokenizer, system: str, user: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            ids = tokenizer.encode("\n".join(f"{row['role'].upper()}: {row['content']}" for row in messages) + "\nASSISTANT:", add_special_tokens=True)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def compile_interface(tokenizer, row: dict, interface: str):
    if interface == "strict_chat":
        ids = render_chat(tokenizer, "Answer from the supplied statement. Reply exactly A or B.", row["prompt"])
    elif interface == "demonstrated_chat":
        ids = render_chat(tokenizer, "Reply exactly A or B. Example: If (A) is correct, reply A. Do not explain.", row["prompt"])
    else:
        ids = tokenizer.encode("Question: " + row["prompt"] + "\nAnswer:", add_special_tokens=True)
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(not candidate for candidate in candidates):
        raise RuntimeError((row["case_id"], interface, candidates))
    return ids, candidates


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C213 / "audit/independent_final_audit.json")
    all_rows = core.rows(common.C212 / "material/cases.jsonl")
    rows = [row for row in all_rows if row["unit"] in (0, 4, 8)]
    checks = {"authorization": parent["all_checks_passed"], "models": MODELS == ("qwen3", "glm4", "deepseek7b"), "cases": len(rows) == 48, "partitions": {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "fresh")} == {"discovery": 16, "confirmation": 16, "fresh": 16}, "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 24}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cross_model_isomorphism_frozen",
        "models": list(MODELS),
        "sequential_loading": True,
        "interfaces": list(INTERFACES),
        "interface_selection": "highest discovery accuracy; ties follow frozen interface order",
        "behavior_qualification": {"global_confirmation_fresh_min": 0.75, "arm_partition_min": 0.65},
        "hidden_panel": "fresh unit 8, order=+1, both arms x four factorial cells = 8 rows",
        "topology": "five relative checkpoints x six role means; physical coordinate ids are never compared",
        "nontrivial_null": "subtract each transition's six-role mean, then compare observed role alignment against all 720 role permutations",
        "topology_gate": {"participating_models_min": 2, "common_rows_min": 8, "all_pair_centered_cosine_min": 0.30, "all_pair_exact_upper_p_max": 0.05},
        "claim_boundary": "cross-model relative role/checkpoint organization only; no same-coordinate code, shared weights or universal language mechanism",
        "forbidden": ["attention", "MLP", "weights", "PCA", "same coordinate ids", "simultaneous model loading", "holdout-selected interface"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_qwen3_then_glm4_then_deepseek7b_sequentially",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "interfaces": list(INTERFACES)}, indent=2))


@torch.inference_mode()
def run_model(model_name: str) -> None:
    if (OUT / f"analysis/{model_name}.json").exists():
        raise RuntimeError(f"{model_name} already run")
    rows = core.rows(OUT / "material/cases.jsonl")
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.load_bf16(model_name)
        quant = common.quantization_audit(model)
        info = get_model_info(model, model_name)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        all_results = {}
        for interface in INTERFACES:
            compiled = [(row, *compile_interface(tokenizer, row, interface)) for row in rows]
            results = []
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
                    results.append({"case_id": row["case_id"], "arm": row["arm"], "unit": row["unit"], "partition": row["partition"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "score0": float(scores[local, 0]), "score1": float(scores[local, 1])})
                del ids_tensor, mask, output, logp
            core.write_rows(OUT / f"raw/{model_name}_{interface}.jsonl", results)
            all_results[interface] = results
            print(f"[C214] {model_name} {interface} accuracy={np.mean([row['correct'] for row in results]):.4f}", flush=True)
        discovery = {interface: float(np.mean([row["correct"] for row in values if row["partition"] == "discovery"])) for interface, values in all_results.items()}
        selected = max(INTERFACES, key=lambda interface: (discovery[interface], -INTERFACES.index(interface)))
        selected_rows = all_results[selected]
        holdout = [row for row in selected_rows if row["partition"] != "discovery"]
        holdout_accuracy = float(np.mean([row["correct"] for row in holdout]))
        by_arm_partition = {arm: {part: float(np.mean([row["correct"] for row in selected_rows if row["arm"] == arm and row["partition"] == part])) for part in ("confirmation", "fresh")} for arm in ("surface_factorial", "path_factorial")}
        qualification = core.load(OUT / "protocol/preregistration.json")["behavior_qualification"]
        eligible = holdout_accuracy >= qualification["global_confirmation_fresh_min"] and min(value for arm in by_arm_partition.values() for value in arm.values()) >= qualification["arm_partition_min"]
        hidden_rows = [row for row in rows if row["partition"] == "fresh" and row["order"] == 1]
        state_path = OUT / f"raw/{model_name}_topology_states.float16.npy"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float16, shape=(len(hidden_rows), 5, len(common.ROLES), info.d_model))
        hidden_index = []
        for case_i, row in enumerate(hidden_rows):
            ids, _ = compile_interface(tokenizer, row, selected)
            role_positions = {}
            for role, value in row["role_values"].items():
                spans = graph_base.name_spans(tokenizer, ids, value)
                if not spans:
                    raise RuntimeError((model_name, row["case_id"], role, value))
                role_positions[role] = spans[-1] if role == "query" else spans[0]
            role_positions["boundary"] = [len(ids) - 1]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True, output_hidden_states=True)
            checkpoints = sorted(set(int(round(fraction * (len(output.hidden_states) - 1))) for fraction in (0, 0.25, 0.5, 0.75, 1)))
            for state_i, checkpoint in enumerate(checkpoints):
                for role_i, role in enumerate(common.ROLES):
                    states[case_i, state_i, role_i] = output.hidden_states[checkpoint][0, role_positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            hidden_index.append({"case_index": case_i, "case_id": row["case_id"], "arm": row["arm"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "checkpoint_indices": checkpoints, "behavior_correct": next(value["correct"] for value in selected_rows if value["case_id"] == row["case_id"])})
            states.flush()
        core.write_rows(OUT / f"raw/{model_name}_topology_index.jsonl", hidden_index)
        report = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "status": "model_observed", "discovery_accuracy": discovery, "selected_interface": selected, "holdout_accuracy": holdout_accuracy, "by_arm_partition": by_arm_partition, "behavior_eligible": eligible, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "topology_cases": len(hidden_rows), "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started}
        core.save(OUT / f"analysis/{model_name}.json", report)
        checks = {"behavior_rows": all(len(value) == 48 for value in all_results.values()), "topology_rows": len(hidden_index) == 8, "state_shape": list(states.shape) == [8, 5, 6, info.d_model], "finite": bool(np.isfinite(states).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps(report, indent=2))
    finally:
        common.release(model)
        gc.collect()


def lock_and_analyze() -> None:
    reports = {model: core.load(OUT / f"analysis/{model}.json") for model in MODELS}
    participants = [model for model, report in reports.items() if report["behavior_eligible"]]
    result_rows = {model: {row["case_id"]: row for row in core.rows(OUT / f"raw/{model}_{reports[model]['selected_interface']}.jsonl")} for model in participants}
    panel = core.rows(OUT / "material/cases.jsonl")
    common_cases = [row for row in panel if row["partition"] == "fresh" and row["order"] == 1 and all(result_rows[model][row["case_id"]]["correct"] for model in participants)]
    topologies = {}
    for model in participants:
        states = np.load(OUT / f"raw/{model}_topology_states.float16.npy", mmap_mode="r")
        index = core.rows(OUT / f"raw/{model}_topology_index.jsonl")
        selected = [row["case_index"] for row in index if row["case_id"] in {case["case_id"] for case in common_cases}]
        transition = np.diff(np.asarray(states[selected], np.float32), axis=1)
        energy = np.sqrt(np.mean(np.square(transition, dtype=np.float64), axis=-1)).mean(axis=0)
        energy /= np.maximum(energy.sum(axis=1, keepdims=True), 1e-30)
        topologies[model] = energy
    pairs = []
    permutations = list(itertools.permutations(range(6)))
    for model_i, left_name in enumerate(participants):
        for right_name in participants[model_i + 1:]:
            left = topologies[left_name] - topologies[left_name].mean(axis=1, keepdims=True)
            right = topologies[right_name] - topologies[right_name].mean(axis=1, keepdims=True)
            observed = common.cosine(left, right)
            null = np.asarray([common.cosine(left, right[:, permutation]) for permutation in permutations])
            p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
            pairs.append({"models": [left_name, right_name], "centered_cosine": observed, "null_median": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)), "exact_upper_p": p, "role_permutations": len(null)})
    gate = core.load(OUT / "protocol/preregistration.json")["topology_gate"]
    passed = len(participants) >= gate["participating_models_min"] and len(common_cases) >= gate["common_rows_min"] and bool(pairs) and min(row["centered_cosine"] for row in pairs) >= gate["all_pair_centered_cosine_min"] and max(row["exact_upper_p"] for row in pairs) <= gate["all_pair_exact_upper_p_max"]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "cross_model_functional_isomorphism_adjudicated", "models": reports, "participants": participants, "common_rows": len(common_cases), "common_case_ids": [row["case_id"] for row in common_cases], "topologies": {model: value.tolist() for model, value in topologies.items()}, "pair_tests": pairs, "cross_model_gate_passed": passed, "interpretation": "Role/checkpoint energy is centered against a frozen permutation null. Even a pass would identify only this functional topology, never equal physical coordinates or a shared complete language code.", "next_authorization": "C215_campaign_synthesis_theory_gate_and_heatmap"}
    core.save(OUT / "analysis/cross_model_isomorphism.json", report)
    checks = {"models": set(reports) == set(MODELS), "participants_typed": set(participants) <= set(MODELS), "common_accounting": len(common_cases) <= 8, "pairs": len(pairs) == len(participants) * (len(participants) - 1) // 2, "finite": bool(np.isfinite([row[key] for row in pairs for key in ("centered_cosine", "null_median", "null_q95", "exact_upper_p")]).all()) if pairs else True}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"participants": participants, "common_rows": len(common_cases), "pairs": pairs, "passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/cross_model_isomorphism.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "models": all(core.load(OUT / f"audit/internal_{model}_audit.json")["all_checks_passed"] for model in MODELS), "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "model", "analyze", "close"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "contract":
        contract()
    elif args.command == "model":
        if not args.model:
            raise SystemExit("--model required")
        run_model(args.model)
    elif args.command == "analyze":
        lock_and_analyze()
    else:
        close()


if __name__ == "__main__":
    main()
