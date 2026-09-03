#!/usr/bin/env python3
"""C146: sequential three-model output-interface sweep for directed paths."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
OUT = RESULT / "phase1680_c146_cross_model_interface_sweep"
C145 = RESULT / "phase1679_c145_correct_error_depth_trajectory_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from model_utils import MODEL_CONFIGS
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE, CAMPAIGN = 1680, "C146"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("yes_no", "true_false", "A_B", "correct_incorrect")
LABELS = {"yes_no": ("yes", "no"), "true_false": ("true", "false"), "A_B": ("A", "B"), "correct_incorrect": ("correct", "incorrect")}
ROLES = ("source", "bridge", "target", "boundary")
WIDTH, BATCH = 224, 4
SYL = ("zaf", "yud", "xir", "wep", "voq", "utn", "sim", "ral", "qek", "poj", "nuv", "mox", "lir", "keg", "jaf", "huz")


def now() -> str: return datetime.now(timezone.utc).isoformat()


def tok_for(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)


def value(i: int, j: int) -> str:
    return f"Iface{SYL[(i + j * 3) % 16]}{SYL[(i * 5 + j + 7) % 16]}{i:02d}{chr(97+j)}"


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit in range(32):
        names = tuple(value(unit, j) for j in range(4))
        partition = "discovery" if unit < 16 else "confirmation"
        units.append({"unit_id": f"c146-{unit:02d}", "partition": partition, "values": list(names)})
        for depth, truth, surface, interface in itertools.product(("direct", "two_hop"), (1, -1), (1, -1), INTERFACES):
            if depth == "direct":
                edges = [(0, 2), (1, 3)] if truth == 1 else [(2, 0), (1, 3)]
            else:
                edges = [(0, 1), (1, 2), (3, 1)] if truth == 1 else [(1, 0), (1, 2), (3, 1)]
            edge_text = ("; " if surface == 1 else " | ").join(f"{names[a]} -> {names[b]}" for a, b in (edges if surface == 1 else list(reversed(edges))))
            stem = f"A path must follow arrow direction. Nodes: {', '.join(names)}. Directed links: {edge_text}."
            if interface == "yes_no": ending = f" Is there a directed path from {names[0]} to {names[2]}? Answer yes if reachable, otherwise no. Reply exactly yes or no."
            elif interface == "true_false": ending = f" Claim: a directed path exists from {names[0]} to {names[2]}. Is the claim true or false? Reply exactly true or false."
            elif interface == "A_B": ending = f" Is {names[2]} reachable from {names[0]}? A means reachable; B means not reachable. Reply exactly A or B."
            else: ending = f" Claim: {names[2]} is reachable from {names[0]} by following arrows. Reply correct if the claim holds, otherwise incorrect."
            cases.append({"case_id": f"c146-{len(cases):05d}", "unit_id": f"c146-{unit:02d}", "partition": partition, "values": list(names), "depth": depth, "truth_factor": truth, "surface_factor": surface, "interface": interface, "gold_position": 0 if truth == 1 else 1, "prompt": stem + ending, "role_values": {"source": names[0], "bridge": names[1], "target": names[2]}})
    return units, cases


def compile_rows(name: str, cases: list[dict]) -> list[dict]:
    tok = tok_for(name); cache = {}
    for interface in INTERFACES:
        ids = [tok.encode(" " + label, add_special_tokens=False) for label in LABELS[interface]]
        if any(len(x) != 1 for x in ids): raise RuntimeError((name, interface, ids))
        cache[interface] = ids
    out = []
    for row in cases:
        ids = core.chat_ids(tok, "Follow directed arrows only and emit exactly the requested label.", row["prompt"])
        positions = {}
        for role, val in row["role_values"].items():
            spans = graph_base.name_spans(tok, ids, val)
            if not spans: raise RuntimeError((name, row["case_id"], role, val))
            positions[role] = spans[-1] if role in {"source", "target"} else spans[0]
        positions["boundary"] = [len(ids) - 1]
        out.append({**row, "prompt_ids": ids, "candidate_ids": cache[row["interface"]], "role_positions": positions})
    return out


def contract() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C145 / "audit/independent_closure_audit.json")
    units, cases = material(); compiled = {m: compile_rows(m, cases) for m in MODELS}
    cells = defaultdict(int)
    for row in cases: cells[(row["partition"], row["depth"], row["truth_factor"], row["surface_factor"], row["interface"])] += 1
    zero = {"truth": float(np.mean([r["truth_factor"] == 1 for r in cases])), "surface": float(np.mean([(r["surface_factor"] == 1) == (r["truth_factor"] == 1) for r in cases]))}
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "start_C146",
        "units": len(units) == 32, "cases": len(cases) == 1024,
        "unique": len({r["prompt"] for r in cases}) == 1024,
        "factorial_cells": len(cells) == 2 * 2 * 2 * 2 * 4 and set(cells.values()) == {16},
        "zero": zero == {"truth": .5, "surface": .5},
        "compiled": all(len(rows) == 1024 for rows in compiled.values()),
        "roles": all(set(r["role_positions"]) == set(ROLES) for rows in compiled.values() for r in rows),
        "width": max(len(r["prompt_ids"]) for rows in compiled.values() for r in rows) < WIDTH,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units); core.write_rows(OUT / "material/cases.jsonl", cases)
    for model, rows in compiled.items(): core.write_rows(OUT / f"compiled/{model}.jsonl", rows)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "cross_model_interface_contract_frozen",
        "models": list(MODELS), "interfaces": list(INTERFACES), "cases_per_model": 1024,
        "selection": "discovery only; maximize qualified model count, then minimum model global accuracy, then frozen interface order",
        "discovery_gate": {"global_min": .80, "truth_min": .75, "models_required": 2},
        "confirmation_gate": {"global_min": .80, "truth_min": .75, "models_required": 2},
        "model_policy": "local BF16 nonquantized CUDA, strictly sequential load and release",
        "forbidden": ["attention", "MLP", "weights", "post-reveal interface or gate changes"],
        "claim_boundary": "behavior-interface qualification only; no internal cross-model mechanism",
        "source_hashes": {"C145": core.sha(C145 / "audit/independent_closure_audit.json")},
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_qwen3_then_glm4_then_deepseek7b",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "zero": zero, "max_width": {m: max(len(r["prompt_ids"]) for r in v) for m, v in compiled.items()}}, indent=2))


@torch.inference_mode()
def run_model(name: str) -> None:
    rows = core.rows(OUT / f"compiled/{name}.jsonl")
    path = OUT / f"raw/{name}_all_interface_logits.float32.npy"; path.parent.mkdir(parents=True, exist_ok=True)
    saved = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    result, model, repeat = [], None, None
    try:
        model, tok, device, placement = load_bf16(name); quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start+BATCH]
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            local = np.asarray([[float(output.logits[i, lengths[i]-1, c[0]]) for c in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
            saved[start:start+len(batch)] = local
            for i, row in enumerate(batch):
                pred = int(local[i, 1] > local[i, 0])
                result.append({"row_index": start+i, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "interface": row["interface"], "depth": row["depth"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"], "gold_margin": float(local[i, row["gold_position"]] - local[i, 1-row["gold_position"]])})
            if (start // BATCH + 1) % 64 == 0: saved.flush(); print(f"[C146 {name}] {start+len(batch)}/{len(rows)}", flush=True)
            del output, ids, mask, positions
        saved.flush()
        batch = rows[:BATCH]; ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
        check = np.asarray([[float(output.logits[i, lengths[i]-1, c[0]]) for c in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
        repeat = float(np.max(np.abs(check - np.asarray(saved[:BATCH]))))
    finally:
        saved.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / f"raw/{name}_behavior_index.jsonl", result)
    checks = {"rows": len(result) == 1024, "finite": bool(np.isfinite(saved).all()), "repeat": repeat == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "model": name, "status": "all_interfaces_captured", "checks": checks, "repeat_logits_max_abs": repeat, "logits_sha256": core.sha(path), "runtime": placement, "authorization": "continue_next_model_then_select"}
    core.save(OUT / f"analysis/{name}_capture.json", report); core.save(OUT / f"audit/{name}_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps({k:v for k,v in report.items() if k != "runtime"}, indent=2))


def acc(rows: list[dict]) -> float: return float(np.mean([r["correct"] for r in rows])) if rows else 0.0


def summarize(rows: list[dict], interface: str, partition: str) -> dict:
    local = [r for r in rows if r["interface"] == interface and r["partition"] == partition]
    return {"n": len(local), "global": acc(local), "truth": {str(v): acc([r for r in local if r["truth_factor"] == v]) for v in (1, -1)}, "depth": {v: acc([r for r in local if r["depth"] == v]) for v in ("direct", "two_hop")}, "surface": {str(v): acc([r for r in local if r["surface_factor"] == v]) for v in (1, -1)}}


def select() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); gate = protocol["discovery_gate"]
    data = {m: core.rows(OUT / f"raw/{m}_behavior_index.jsonl") for m in MODELS}
    table, ranked = {}, []
    for interface in INTERFACES:
        table[interface] = {m: summarize(data[m], interface, "discovery") for m in MODELS}
        qualified = [m for m in MODELS if table[interface][m]["global"] >= gate["global_min"] and min(table[interface][m]["truth"].values()) >= gate["truth_min"]]
        min_global = min([table[interface][m]["global"] for m in qualified], default=0.0)
        ranked.append((len(qualified), min_global, -INTERFACES.index(interface), interface, qualified))
    count, minimum, _, winner, qualified = max(ranked)
    if count < gate["models_required"]: winner, qualified = None, []
    freeze = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "read_partition": "discovery", "table": table, "winner": winner, "discovery_qualified_models": qualified, "gate": gate, "confirmation_unread_by_selection": True, "authorization": "validate_confirmation" if winner else "close_no_common_interface_continue_C147_typed_missing"}
    core.save(OUT / "protocol/frozen_interface.json", freeze)
    checks = {"models": len(data) == 3, "interfaces": len(table) == 4, "rows": all(len(v) == 1024 for v in data.values()), "typed_winner": winner is None or winner in INTERFACES}
    core.save(OUT / "audit/internal_selection_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps({"winner": winner, "discovery_qualified_models": qualified, "table": table}, indent=2))


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_interface.json"); protocol = core.load(OUT / "protocol/preregistration.json"); gate = protocol["confirmation_gate"]
    if freeze["winner"] is None:
        report = {"phase":PHASE,"campaign":CAMPAIGN,"status":"no_common_discovery_interface","winner":None,"confirmation_qualified_models":[],"common_interface_gate_passed":False,"authorization":"close_C146_continue_C147_typed_missing"}
    else:
        table = {m: summarize(core.rows(OUT / f"raw/{m}_behavior_index.jsonl"), freeze["winner"], "confirmation") for m in MODELS}
        qualified = [m for m in MODELS if table[m]["global"] >= gate["global_min"] and min(table[m]["truth"].values()) >= gate["truth_min"]]
        report = {"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"common_interface_confirmation_adjudicated","winner":freeze["winner"],"confirmation_table":table,"confirmation_qualified_models":qualified,"common_interface_gate_passed":len(qualified)>=gate["models_required"],"authorization":"close_C146_start_C147"}
    core.save(OUT / "analysis/confirmation.json", report)
    checks = {"frozen": report["winner"] == freeze["winner"], "typed": isinstance(report["confirmation_qualified_models"], list), "models": all(m in MODELS for m in report["confirmation_qualified_models"])}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks":checks,"all_checks_passed":all(checks.values()),"scientific_common_interface_gate_passed":report["common_interface_gate_passed"],"authorization":report["authorization"]})
    print(json.dumps(report, indent=2))


def close() -> None:
    result=core.load(OUT/"analysis/confirmation.json")
    checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"models":all(core.load(OUT/f"audit/{m}_capture_audit.json")["all_checks_passed"] for m in MODELS),"selection":core.load(OUT/"audit/internal_selection_audit.json")["all_checks_passed"],"confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"]}
    closure={"phase":PHASE,"campaign":CAMPAIGN,"status":"cross_model_interface_sweep_closed","headline":{"winner":result["winner"],"qualified_models":result["confirmation_qualified_models"],"gate_passed":result["common_interface_gate_passed"]},"claim_boundary":"behavior interface only; C147 may inspect qualified models but cannot infer coordinate identity","next_authorization":"C147 relative topology for confirmation-qualified models; typed not-tested if fewer than two"}
    core.save(OUT/"analysis/closure.json",closure);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":"independent_final_then_C147"});print(json.dumps(closure,indent=2))


def main() -> None:
    if len(sys.argv)<2: raise SystemExit("contract|run_model NAME|select|validate|close")
    mode=sys.argv[1]
    if mode=="contract":contract()
    elif mode=="run_model":run_model(sys.argv[2])
    elif mode=="select":select()
    elif mode=="validate":validate()
    elif mode=="close":close()
    else:raise SystemExit(mode)


if __name__=="__main__":main()
