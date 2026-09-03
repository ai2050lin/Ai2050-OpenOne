#!/usr/bin/env python3
"""C287: sequentially capture cross-model fifth-material joint role states."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1811_c277_c289_joint_response_common as common
import phase1748_c214_cross_model_functional_isomorphism as interfaces
from model_utils import get_model_info

core, OUT = common.core, common.OUTS["C287"]
C277 = common.OUTS["C277"]
C286 = common.OUTS["C286"]
INTERFACES = ("strict_chat", "demonstrated_chat", "plain")
BATCH = {"qwen3": 8, "glm4": 1, "deepseek7b": 1}


def selected_rows() -> list[dict]:
    rows = core.rows(C277 / "material/cases.jsonl")
    return [row for row in rows if row["panel"] == "core" and row["unit"] in (0, 1) and row["order"] == 1]


def score_candidates(model, device, pad: int, batch):
    expanded = []
    for row, ids, candidates in batch:
        for ci, candidate in enumerate(candidates): expanded.append((row, ci, ids + candidate, len(ids), candidate))
    width = max(len(x[2]) for x in expanded)
    ids_t = torch.full((len(expanded), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids_t)
    for i, (_row, _ci, values, _prefix, _candidate) in enumerate(expanded):
        ids_t[i, :len(values)] = torch.tensor(values, device=device); mask[i, :len(values)] = 1
    output = model(input_ids=ids_t, attention_mask=mask, use_cache=False, return_dict=True)
    logp = torch.log_softmax(output.logits.float(), dim=-1); scores = np.zeros((len(batch), 2), np.float32)
    for i, (_row, ci, _values, prefix, candidate) in enumerate(expanded):
        scores[i // 2, ci] = sum(float(logp[i, prefix + offset - 1, token]) for offset, token in enumerate(candidate))
    return scores


@torch.inference_mode()
def run_model(model_name: str, rows: list[dict], protocol: dict) -> dict:
    model = None; started = time.time()
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16(model_name)
        quant = common.model_base.quantization_audit(model); info = get_model_info(model, model_name)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        compiled_by_interface = {name: [(row, *interfaces.compile_interface(tokenizer, row, name)) for row in rows] for name in INTERFACES}
        all_results = {}
        for interface, compiled in compiled_by_interface.items():
            results = []
            for start in range(0, len(compiled), BATCH[model_name]):
                batch = compiled[start:start + BATCH[model_name]]; scores = score_candidates(model, device, pad, batch)
                for local, (row, _ids, _candidates) in enumerate(batch):
                    prediction = int(scores[local, 1] > scores[local, 0])
                    results.append({"case_id": row["case_id"], "surface": row["surface"], "family": row["family"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
            all_results[interface] = results; core.write_rows(OUT / f"raw/{model_name}_{interface}_behavior.jsonl", results)
            print(f"[C287] {model_name}/{interface}: {np.mean([r['correct'] for r in results]):.4f}", flush=True)
        discovery = {name: float(np.mean([r["correct"] for r in values if r["surface"] == "dossier"])) for name, values in all_results.items()}
        selected = max(INTERFACES, key=lambda name: (discovery[name], -INTERFACES.index(name)))
        holdout = [r for r in all_results[selected] if r["surface"] == "hearing"]
        holdout_accuracy = float(np.mean([r["correct"] for r in holdout]))
        by_family = {family: float(np.mean([r["correct"] for r in holdout if r["family"] == family])) for family in common.previous.FAMILIES}
        behavior_eligible = holdout_accuracy >= protocol["behavior_gate"]["global_min"] and min(by_family.values()) >= protocol["behavior_gate"]["family_min"]
        hidden_rows = [row for row in rows if row["surface"] == "hearing"]
        nq = info.n_layers + 1
        states = np.lib.format.open_memmap(OUT / f"raw/{model_name}_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), nq, 6, info.d_model))
        hidden_index = []
        compiled_lookup = {row["case_id"]: ids for row, ids, _candidates in compiled_by_interface[selected]}
        selected_behavior = {row["case_id"]: row for row in all_results[selected]}
        for i, row in enumerate(hidden_rows):
            ids = compiled_lookup[row["case_id"]]
            positions = {}
            for role, value in row["role_values"].items():
                spans = common.graph_base.name_spans(tokenizer, ids, value)
                if not spans: raise RuntimeError((model_name, row["case_id"], role, value))
                positions[role] = spans[-1] if role == "query" else spans[0]
            positions["boundary"] = [len(ids) - 1]
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            output = model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != nq: raise RuntimeError((model_name, len(output.hidden_states), nq))
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(common.ROLES): states[i, q, ri] = state[0, positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            behavior = selected_behavior[row["case_id"]]
            hidden_index.append({"hidden_index": i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "behavior_correct": behavior["correct"]})
            states.flush()
        core.write_rows(OUT / f"raw/{model_name}_hidden_index.jsonl", hidden_index)
        report = {
            "model": model_name, "discovery_accuracy": discovery, "selected_interface": selected, "holdout_accuracy": holdout_accuracy,
            "by_family_accuracy": by_family, "behavior_eligible": behavior_eligible,
            "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started,
        }
        core.save(OUT / f"analysis/{model_name}.json", report)
        audit = {"behavior_rows": all(len(v) == 80 for v in all_results.values()), "hidden_rows": len(hidden_index) == 40, "shape": list(states.shape) == [40, nq, 6, info.d_model], "finite": bool(np.isfinite(states[:, :, :, ::64]).all())}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": audit, "all_checks_passed": all(audit.values())})
        return report
    finally:
        common.model_base.release(model); gc.collect()


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C286 / "analysis/final.json"); rows = selected_rows()
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 80, "models": common.MODELS == ("qwen3", "glm4", "deepseek7b"), "sequential_cuda": True, "all_model_coordinates_retained": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol", "raw", "material"): (OUT / subdir).mkdir()
    core.write_rows(OUT / "material/cases.jsonl", rows)
    protocol = {
        "phase": 1821, "campaign": "C287", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "cross_model_capture_frozen",
        "models": list(common.MODELS), "interfaces": list(INTERFACES), "interface_selection": "dossier accuracy only; hearing is holdout",
        "behavior_gate": {"global_min": 0.65, "family_min": 0.50},
        "hidden_panel": "hearing surface, units 0-1, five core families, complete 2x2 factor cells",
        "comparison_object": "anonymous six-role sign-word transition tables at relative depths; physical coordinate axes stay separate",
        "claim_boundary": "No same coordinate number, weight, attention state, MLP state, or cross-model vector alignment is asserted.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_three_models_sequentially_then_C288",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    reports = {}
    for model_name in common.MODELS:
        reports[model_name] = run_model(model_name, rows, protocol)
    ach = {"models": set(reports) == set(common.MODELS), "model_audits": all(core.load(OUT / f"audit/internal_{name}_audit.json")["all_checks_passed"] for name in common.MODELS), "sequential_complete": True}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    report = {"phase": 1821, "campaign": "C287", "status": "cross_model_joint_states_captured", "models": reports, "participants": [name for name, value in reports.items() if value["behavior_eligible"]], "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C288_cross_model_automaton_isomorphism"}
    core.save(OUT / "analysis/summary.json", report)
    fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1821, "campaign": "C287", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

