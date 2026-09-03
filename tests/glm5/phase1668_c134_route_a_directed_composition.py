#!/usr/bin/env python3
"""C134 route A: directed-link composition behavior and prospective typed trajectories."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1668_c134_route_a_directed_composition"
C133 = RESULT / "phase1667_c133_multiroute_campaign_contract"
C129 = RESULT / "phase1663_c129_direct_precedence_typed_transition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE = 1668
CAMPAIGN = "C134"
PARTITIONS = ("discovery", "confirmation")
ROUTE_TYPES = ("direct", "two_hop", "alternative_paths", "direct_shortcut", "reverse_listing", "irrelevant_dense")
ROLES = ("source", "bridge", "target", "distractor", "boundary")
CHECKPOINTS = c127.CHECKPOINTS
DIM = 2560
WIDTH = 224
BATCH = 8
SYSTEM = "Use only the directed route links. Answer only yes or no."
SYLLABLES = ("baf", "cud", "dix", "fom", "gup", "hez", "jol", "kav", "lum", "neq", "piv", "qor", "rax", "syt", "tuv", "wim")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def values_for(index: int) -> tuple[str, str, str, str, str]:
    a = SYLLABLES[index % 16]
    b = SYLLABLES[(index * 13 + 4) % 16]
    return tuple(f"Graph{a}{b}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d", "e"))


def edges_for(route_type: str, truth: int, unit_index: int) -> list[tuple[int, int]]:
    a, b, c, d, e = range(5)
    if route_type == "direct":
        return [(a, c), (b, d), (d, e)] if truth == 1 else [(c, a), (b, d), (d, e)]
    if route_type == "two_hop":
        if truth == 1:
            return [(a, b), (b, c), (d, e)]
        return [(b, a), (b, c), (d, e)] if unit_index % 2 == 0 else [(a, b), (c, b), (d, e)]
    if route_type == "alternative_paths":
        return [(a, b), (b, c), (a, d), (d, c)] if truth == 1 else [(a, b), (c, b), (a, d), (c, d)]
    if route_type == "direct_shortcut":
        return [(a, b), (b, c), (a, c), (d, e)] if truth == 1 else [(a, b), (c, b), (c, a), (d, e)]
    if route_type == "reverse_listing":
        return [(b, c), (a, b), (d, e)] if truth == 1 else [(c, b), (a, b), (d, e)]
    if route_type == "irrelevant_dense":
        return [(a, b), (b, c), (d, e), (e, d)] if truth == 1 else [(a, b), (c, b), (d, e), (e, d)]
    raise KeyError(route_type)


def render_prompt(values: tuple[str, ...], edges: list[tuple[int, int]], surface: int, family: str = "route") -> str:
    separator = "; " if surface == 1 else " | "
    label = "Directed route links" if surface == 1 else "Route graph"
    edge_text = separator.join(f"{values[left]} -> {values[right]}" for left, right in edges)
    nodes = ", ".join(values)
    return f"Path rule: yes means at least one directed path follows the arrows from source to target. Graph family: {family}. Nodes: {nodes}. {label}: {edge_text}. Query: Is there a directed path from {values[0]} to {values[2]}? Reply exactly yes or no."


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = values_for(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c134-{unit_index:02d}", "partition": partition, "values": list(values), "world": "controlled_directed_link_graph"}
        units.append(unit)
        for route_type, truth, surface in itertools.product(ROUTE_TYPES, (1, -1), (1, -1)):
            edges = edges_for(route_type, truth, unit_index)
            cases.append({**unit, "case_id": f"c134-{len(cases):04d}", "route_type": route_type, "truth_factor": truth, "surface_factor": surface, "edge1_factor": None, "edge2_factor": None, "edges": edges, "truth": truth == 1, "gold_position": 0 if truth == 1 else 1, "prompt": render_prompt(values, edges, surface)})
        for edge1, edge2, surface in itertools.product((1, -1), repeat=3):
            a, b, c, d, e = range(5)
            first = (a, b) if edge1 == 1 else (a, d)
            second = (b, c) if edge2 == 1 else (d, c)
            edges = [first, second, (d, e)]
            truth = 1 if edge1 == edge2 else -1
            cases.append({**unit, "case_id": f"c134-{len(cases):04d}", "route_type": "edge_factorial", "truth_factor": truth, "surface_factor": surface, "edge1_factor": edge1, "edge2_factor": edge2, "edges": edges, "truth": truth == 1, "gold_position": 0 if truth == 1 else 1, "prompt": render_prompt(values, edges, surface, "edge-factorial")})
    return units, cases


def compile_rows(tokenizer, cases: list[dict]) -> list[dict]:
    candidates = [[int(value) for value in tokenizer.encode(" " + label, add_special_tokens=False)] for label in ("yes", "no")]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for row in cases:
        ids = core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        spans = [graph_base.name_spans(tokenizer, ids, value) for value in row["values"]]
        if any(not value for value in spans[:4]):
            raise RuntimeError((row["case_id"], spans))
        roles = {"source": spans[0][-1], "bridge": spans[1][0], "target": spans[2][-1], "distractor": spans[3][0], "boundary": [len(ids) - 1]}
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": roles})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C134 exists: {OUT}")
    parent = core.load(C133 / "protocol/preregistration.json")
    parent_audit = core.load(C133 / "audit/independent_contract_audit.json")
    units, cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    route_cells = Counter((row["partition"], row["route_type"], row["truth_factor"], row["surface_factor"]) for row in cases if row["route_type"] != "edge_factorial")
    factorial_cells = Counter((row["partition"], row["edge1_factor"], row["edge2_factor"], row["surface_factor"]) for row in cases if row["route_type"] == "edge_factorial")
    zero = {"always_yes": float(np.mean([row["truth"] for row in cases])), "always_no": float(np.mean([not row["truth"] for row in cases])), "surface_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases]))}
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "start_route_A_C134" and parent["authorization"] == "start_route_A_C134",
        "units": len(units) == 32,
        "cases": len(cases) == 1024,
        "route_cells": route_cells == {(partition, route, truth, surface): 16 for partition in PARTITIONS for route in ROUTE_TYPES for truth in (1, -1) for surface in (1, -1)},
        "factorial_cells": factorial_cells == {(partition, e1, e2, surface): 16 for partition in PARTITIONS for e1 in (1, -1) for e2 in (1, -1) for surface in (1, -1)},
        "unique_prompts": len({row["prompt"] for row in cases}) == 1024,
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "zero_models": all(value <= 0.5 for value in zero.values()),
        "semantic_path_truth": True,
    }
    # Recompute path truth without relying on a helper that may not exist.
    def reachable(edges):
        seen, frontier = {0}, [0]
        while frontier:
            node = frontier.pop()
            for left, right in edges:
                if left == node and right not in seen:
                    seen.add(right); frontier.append(right)
        return 2 in seen
    checks["semantic_path_truth"] = all(reachable(row["edges"]) == row["truth"] for row in cases)
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    source_paths = {"c133_protocol": C133 / "protocol/preregistration.json", "c133_audit": C133 / "audit/independent_contract_audit.json", "c129_vector": C129 / "analysis/discovery_nominee_increment.float32.npy"}
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "route_A_directed_composition_contract_frozen",
        "object": "explicit directed-link direct/composed path family with edge-factorial interaction",
        "model": "Qwen3-4B local BF16 CUDA nonquantized", "units": 32, "cases": 1024, "partitions": list(PARTITIONS), "route_types": list(ROUTE_TYPES), "roles": list(ROLES), "checkpoints": list(CHECKPOINTS), "coordinates": DIM,
        "behavior_gate": {"global_min": 0.95, "partition_min": 0.90, "truth_min": 0.90, "surface_min": 0.90, "route_min": 0.90, "factorial_cell_min": 0.85},
        "prediction_gate": {"median_checkpoint_cosine_min": 0.90, "checkpoint_cosine_ge_0_8_fraction_min": 0.75, "relative_trajectory_error_max": 0.50},
        "interaction_gate": {"median_checkpoint_cosine_min": 0.90, "checkpoint_cosine_ge_0_8_fraction_min": 0.75},
        "c129_fixed_candidate": {"role": "boundary", "transition_index": 35, "vector_sha256": core.sha(C129 / "analysis/discovery_nominee_increment.float32.npy")},
        "confirmation_policy": "capture may occur together, but discovery code reads only discovery rows and freezes checkpointwise scalar maps before confirmation code is run",
        "observation_policy": "registered roles, exact 38 checkpoints, all 2560 coordinates; no PCA/SVD, attention, MLP, or weight inspection",
        "stop_conditions": ["behavior failure forbids route A HiddenState", "prediction failure closes causal authorization but does not block routes B-E"],
        "claim_boundary": "controlled explicit graph language; trajectory prediction is not a universal language operator or unique causal circuit",
        "source_paths": {name: str(path) for name, path in source_paths.items()}, "source_hashes": {name: core.sha(path) for name, path in source_paths.items()}, "producer_sha256": core.sha(Path(__file__)), "authorization": "run_c134_behavior",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


def accuracy(rows: list[dict]) -> float:
    return float(np.mean([row["correct"] for row in rows]))


@torch.inference_mode()
def behavior() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "run_c134_behavior" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C134 behavior unauthorized")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"; logits_path.parent.mkdir(parents=True, exist_ok=True)
    logits_array = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    results, model, repeat = [], None, 0.0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def run(batch):
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
            logits = model.lm_head(boundary).float()
            scores = np.asarray([[float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]] for local, row in enumerate(batch)], dtype=np.float32)
            return scores, output, boundary, logits, ids, mask, positions
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start+BATCH]; scores, output, boundary, logits, ids, mask, positions = run(batch)
            logits_array[start:start+len(batch)] = scores
            for local, row in enumerate(batch):
                pred = int(scores[local, 1] > scores[local, 0])
                results.append({"row_index": start+local, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "route_type": row["route_type"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "edge1_factor": row["edge1_factor"], "edge2_factor": row["edge2_factor"], "gold_position": row["gold_position"], "prediction": pred, "correct": pred == row["gold_position"], "yes_minus_no": float(scores[local,0]-scores[local,1])})
            del output, boundary, logits, ids, mask, positions
        logits_array.flush(); scores, output, boundary, logits, ids, mask, positions = run(rows[:BATCH]); repeat = float(np.max(np.abs(scores-np.asarray(logits_array[:BATCH]))))
    finally:
        logits_array.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", results)
    factorial = [row for row in results if row["route_type"] == "edge_factorial"]
    summary = {
        "global_accuracy": accuracy(results),
        "by_partition": {key: accuracy([row for row in results if row["partition"] == key]) for key in PARTITIONS},
        "by_truth": {str(key): accuracy([row for row in results if row["truth_factor"] == key]) for key in (1,-1)},
        "by_surface": {str(key): accuracy([row for row in results if row["surface_factor"] == key]) for key in (1,-1)},
        "by_route": {key: accuracy([row for row in results if row["route_type"] == key]) for key in (*ROUTE_TYPES,"edge_factorial")},
        "factorial_cells": {f"{e1},{e2}": accuracy([row for row in factorial if row["edge1_factor"] == e1 and row["edge2_factor"] == e2]) for e1 in (1,-1) for e2 in (1,-1)},
    }
    gates = protocol["behavior_gate"]
    gate = summary["global_accuracy"] >= gates["global_min"] and min(summary["by_partition"].values()) >= gates["partition_min"] and min(summary["by_truth"].values()) >= gates["truth_min"] and min(summary["by_surface"].values()) >= gates["surface_min"] and min(summary["by_route"].values()) >= gates["route_min"] and min(summary["factorial_cells"].values()) >= gates["factorial_cell_min"]
    checks = {"rows": len(results)==1024, "finite": bool(np.isfinite(logits_array).all()), "repeat": repeat==0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    report = {"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"behavior_qualified" if gate else "behavior_failed","checks":checks,"summary":summary,"gate_passed":gate,"repeat_logits_max_abs":repeat,"runtime":{"placement":placement,"quantization":quant},"authorization":"capture_c134_typed_roles" if gate else "close_A_continue_B"}
    core.save(OUT / "analysis/behavior.json", report); core.save(OUT / "audit/internal_behavior_audit.json", {"checks":checks,"all_integrity_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":report["authorization"]})
    print(json.dumps({k:v for k,v in report.items() if k!="runtime"}, indent=2))


def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def capture() -> None:
    behavior_report = core.load(OUT / "analysis/behavior.json")
    if behavior_report["authorization"] != "capture_c134_typed_roles": raise RuntimeError("C134 capture unauthorized")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw_path = OUT / "raw/qwen3_role_typed_checkpoints.bf16.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint16, shape=(len(rows),len(ROLES),len(CHECKPOINTS),DIM))
    model, repeat = None, 0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant=quantization_audit(model)
        if len(model.model.layers)!=36: raise RuntimeError(len(model.model.layers))
        pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def run(batch):
            captured={}; handles=[model.model.embed_tokens.register_forward_hook(lambda _m,_a,o: captured.__setitem__("embedding",tensor_output(o).detach()))]
            for li,layer in enumerate(model.model.layers): handles.append(layer.register_forward_hook(lambda _m,_a,o,index=li: captured.__setitem__(f"block_{index}",tensor_output(o).detach())))
            handles.append(model.model.norm.register_forward_hook(lambda _m,_a,o: captured.__setitem__("norm",tensor_output(o).detach())))
            try:
                ids,mask,positions,lengths=fixed_base.fixed_batch(batch,pad,device,WIDTH); output=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,output_hidden_states=False,return_dict=True)
            finally:
                for handle in handles: handle.remove()
            return [captured["embedding"],*[captured[f"block_{li}"] for li in range(36)],captured["norm"]],output,ids,mask,positions,lengths
        for start in range(0,len(rows),BATCH):
            batch=rows[start:start+BATCH]; tensors,output,ids,mask,positions,lengths=run(batch)
            for local,row in enumerate(batch):
                for ri,role in enumerate(ROLES):
                    for si,tensor in enumerate(tensors): raw[start+local,ri,si]=tensor[local,row["role_positions"][role]].mean(dim=0).contiguous().view(torch.uint16).cpu().numpy()
            if (start//BATCH+1)%32==0: raw.flush(); print(f"[C134] {start+len(batch)}/{len(rows)}",flush=True)
            del tensors,output,ids,mask,positions
        raw.flush(); tensors,output,ids,mask,positions,lengths=run(rows[:BATCH])
        for local,row in enumerate(rows[:BATCH]):
            for ri,role in enumerate(ROLES):
                for si,tensor in enumerate(tensors):
                    bits=tensor[local,row["role_positions"][role]].mean(dim=0).contiguous().view(torch.uint16).cpu().numpy(); repeat=max(repeat,int(np.max(np.abs(bits.astype(np.int64)-raw[local,ri,si].astype(np.int64)))))
    finally:
        raw.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    checks={"shape":list(raw.shape)==[1024,5,38,DIM],"finite":bool(np.isfinite(c127.decode(raw[:2])).all()),"repeat_bits":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"capture_complete","checks":checks,"shape":list(raw.shape),"sha256":core.sha(raw_path),"runtime":{"placement":placement,"quantization":quant},"authorization":"discover_c134_predictions"}
    core.save(OUT / "analysis/capture.json",report);core.save(OUT / "audit/internal_capture_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":report["authorization"]});print(json.dumps({k:v for k,v in report.items() if k!="runtime"},indent=2))


def partition_fields(partition: str) -> tuple[np.ndarray,np.ndarray]:
    rows=core.rows(OUT / "compiled/qwen3.jsonl"); units=core.rows(OUT / "material/units.jsonl"); raw=np.load(OUT / "raw/qwen3_role_typed_checkpoints.bf16.npy",mmap_mode="r")
    selected=[unit for unit in units if unit["partition"]==partition]; lookup={unit["unit_id"]:i for i,unit in enumerate(selected)}
    route=np.zeros((len(selected),len(ROUTE_TYPES),len(ROLES),len(CHECKPOINTS),DIM),dtype=np.float32); interaction=np.zeros((len(selected),len(ROLES),len(CHECKPOINTS),DIM),dtype=np.float32)
    for index,row in enumerate(rows):
        if row["partition"]!=partition: continue
        value=c127.decode(raw[index]); ui=lookup[row["unit_id"]]
        if row["route_type"] in ROUTE_TYPES:
            route[ui,ROUTE_TYPES.index(row["route_type"])]+=float(row["truth_factor"])/4.0*value
        else:
            interaction[ui]+=float(row["edge1_factor"]*row["edge2_factor"])/2.0*value
    return route,interaction


def discover() -> None:
    if core.load(OUT / "analysis/capture.json")["authorization"]!="discover_c134_predictions":raise RuntimeError("unauthorized")
    route,interaction=partition_fields("discovery"); np.save(OUT / "analysis/discovery_route_truth_fields.float32.npy",route);np.save(OUT / "analysis/discovery_edge_interaction.float32.npy",interaction)
    means=np.mean(route,axis=0,dtype=np.float32); direct=means[ROUTE_TYPES.index("direct"),ROLES.index("boundary")]
    maps={};
    for route_type in ROUTE_TYPES:
        target=means[ROUTE_TYPES.index(route_type),ROLES.index("boundary")]; alpha=[]
        for state in range(len(CHECKPOINTS)):
            denom=float(np.dot(direct[state],direct[state]));alpha.append(0.0 if denom<=1e-12 else float(np.dot(target[state],direct[state])/denom))
        maps[route_type]={"alpha_by_checkpoint":alpha,"discovery_cosine_by_checkpoint":[cosine(direct[state],target[state]) for state in range(len(CHECKPOINTS))]}
    interaction_mean=np.mean(interaction[:,ROLES.index("boundary")],axis=0,dtype=np.float32)
    np.save(OUT / "analysis/discovery_direct_boundary_trajectory.float32.npy",direct);np.save(OUT / "analysis/discovery_interaction_boundary_trajectory.float32.npy",interaction_mean)
    freeze={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"route_A_predictions_frozen","maps":maps,"interaction_reference_sha256":core.sha(OUT / "analysis/discovery_interaction_boundary_trajectory.float32.npy"),"confirmation_unread":True,"authorization":"validate_c134_confirmation"}
    discovery_checks={"route_shape":list(route.shape)==[16,6,5,38,DIM],"interaction_shape":list(interaction.shape)==[16,5,38,DIM],"finite":bool(np.isfinite(route).all() and np.isfinite(interaction).all()),"maps":len(maps)==6}
    core.save(OUT / "protocol/frozen_predictions.json",freeze);core.save(OUT / "audit/internal_discovery_audit.json",{"checks":discovery_checks,"all_checks_passed":all(discovery_checks.values()),"authorization":freeze["authorization"]});print(json.dumps({"status":freeze["status"],"maps":{k:{"median_discovery_cosine":float(np.median(v["discovery_cosine_by_checkpoint"]))} for k,v in maps.items()}},indent=2))


def trajectory_metrics(predicted: np.ndarray,target: np.ndarray) -> dict:
    active=[i for i in range(len(CHECKPOINTS)) if np.linalg.norm(predicted[i])>1e-8 and np.linalg.norm(target[i])>1e-8]
    cos=[cosine(predicted[i],target[i]) for i in active]; error=float(np.linalg.norm(predicted-target)/max(np.linalg.norm(target),1e-12))
    return {"active_checkpoints":len(active),"cosine_by_checkpoint":cos,"median_checkpoint_cosine":float(np.median(cos)) if cos else 0.0,"checkpoint_cosine_ge_0_8_fraction":float(np.mean(np.asarray(cos)>=0.8)) if cos else 0.0,"relative_trajectory_error":error}


def validate() -> None:
    protocol=core.load(OUT / "protocol/preregistration.json");freeze=core.load(OUT / "protocol/frozen_predictions.json")
    if freeze["authorization"]!="validate_c134_confirmation":raise RuntimeError("unauthorized")
    route,interaction=partition_fields("confirmation");np.save(OUT / "analysis/confirmation_route_truth_fields.float32.npy",route);np.save(OUT / "analysis/confirmation_edge_interaction.float32.npy",interaction)
    means=np.mean(route,axis=0,dtype=np.float32);direct=means[ROUTE_TYPES.index("direct"),ROLES.index("boundary")];results={};gates=protocol["prediction_gate"]
    for route_type in ROUTE_TYPES:
        alpha=np.asarray(freeze["maps"][route_type]["alpha_by_checkpoint"],dtype=np.float32);predicted=alpha[:,None]*direct;target=means[ROUTE_TYPES.index(route_type),ROLES.index("boundary")];metrics=trajectory_metrics(predicted,target);passed=metrics["median_checkpoint_cosine"]>=gates["median_checkpoint_cosine_min"] and metrics["checkpoint_cosine_ge_0_8_fraction"]>=gates["checkpoint_cosine_ge_0_8_fraction_min"] and metrics["relative_trajectory_error"]<=gates["relative_trajectory_error_max"];results[route_type]={"metrics":metrics,"passed":passed}
    interaction_reference=np.load(OUT / "analysis/discovery_interaction_boundary_trajectory.float32.npy");interaction_target=np.mean(interaction[:,ROLES.index("boundary")],axis=0,dtype=np.float32);interaction_metrics=trajectory_metrics(interaction_reference,interaction_target);ig=protocol["interaction_gate"];interaction_pass=interaction_metrics["median_checkpoint_cosine"]>=ig["median_checkpoint_cosine_min"] and interaction_metrics["checkpoint_cosine_ge_0_8_fraction"]>=ig["checkpoint_cosine_ge_0_8_fraction_min"]
    c129=np.load(C129 / "analysis/discovery_nominee_increment.float32.npy");k319={route_type:cosine(c129,means[ROUTE_TYPES.index(route_type),ROLES.index("boundary"),36]-means[ROUTE_TYPES.index(route_type),ROLES.index("boundary"),35]) for route_type in ROUTE_TYPES}
    report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"route_A_confirmation_adjudicated","route_predictions":results,"edge_interaction":{"metrics":interaction_metrics,"passed":interaction_pass},"c129_fixed_transition_cosines":k319,"causal_gate_passed":results["two_hop"]["passed"] and interaction_pass,"authorization":"close_c134_continue_B"}
    confirmation_checks={"confirmation_shape":list(route.shape)==[16,6,5,38,DIM],"finite":bool(np.isfinite(route).all()),"six_routes":len(results)==6}
    core.save(OUT / "analysis/confirmation.json",report);core.save(OUT / "audit/internal_confirmation_audit.json",{"checks":confirmation_checks,"all_checks_passed":all(confirmation_checks.values()),"scientific_gates":{"two_hop_prediction":results["two_hop"]["passed"],"interaction":interaction_pass,"causal":report["causal_gate_passed"]},"authorization":report["authorization"]});print(json.dumps(report,indent=2))


def close() -> None:
    behavior_report=core.load(OUT / "analysis/behavior.json")
    if not behavior_report["gate_passed"]:
        closure={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"route_A_behavior_failed","headline":behavior_report["summary"],"claim_boundary":"behavior only; no HiddenState route A evidence","next_authorization":"continue route B independently from existing qualified assets"}
    else:
        confirmation=core.load(OUT / "analysis/confirmation.json");closure={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"route_A_closed","headline":{"behavior":behavior_report["summary"],"two_hop_prediction":confirmation["route_predictions"]["two_hop"],"edge_interaction":confirmation["edge_interaction"],"causal_gate_passed":confirmation["causal_gate_passed"]},"results":confirmation,"theory_update":"Direct-to-composed trajectory prediction and the two-edge nonadditive interaction are separately adjudicated; K319 similarity remains output-adjacent evidence.","problems":["explicit graph metalanguage","Qwen3 only","registered roles rather than all tokens","truth aligned with yes/no"],"claim_boundary":"typed activation trajectory evidence, not a universal natural-language operator or unique causal circuit","next_authorization":"continue route B; causal intervention only if the C133 four-part causal gate remains satisfied after route B"}
    core.save(OUT / "analysis/closure.json",closure);checks={"contract":core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],"behavior":core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],"branch_files":(not behavior_report["gate_passed"]) or all((OUT/p).exists() for p in ("analysis/capture.json","protocol/frozen_predictions.json","analysis/confirmation.json"))};core.save(OUT / "audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":behavior_report["gate_passed"] and core.load(OUT / "analysis/confirmation.json")["causal_gate_passed"] if behavior_report["gate_passed"] else False,"authorization":"independent_audit_then_route_B"});print(json.dumps(closure,indent=2))


def main() -> None:
    modes={"contract":contract,"behavior":behavior,"capture":capture,"discover":discover,"validate":validate,"close":close}
    if len(sys.argv)!=2 or sys.argv[1] not in modes:raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__=="__main__":main()
