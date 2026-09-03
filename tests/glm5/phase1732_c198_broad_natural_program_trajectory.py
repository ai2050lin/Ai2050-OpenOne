#!/usr/bin/env python3
"""C198: broad natural-program behavior and signed q23-q25 trajectory observation."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1732_c198_broad_natural_program_trajectory"
C194 = RESULT / "phase1728_c194_signed_operator_campaign_contract"
C195 = RESULT / "phase1729_c195_signed_role_checkpoint_trajectory"
C197 = RESULT / "phase1731_c197_structure_model_tournament"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1726_c192_multi_program_response_equivalence as c192
import phase1728_c194_signed_operator_campaign_contract as c194

PHASE, CAMPAIGN = 1732, "C198"
DIM, WIDTH, BATCH = 2560, 256, 8
ROLES = c192.ROLES
VALID_PROGRAMS = tuple(p for p in c194.NATURAL_PROGRAMS if p != "location")
ANCHOR_UNITS = c194.ANCHOR_UNITS


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C197 / "audit/independent_final_audit.json"); rows = core.rows(C194 / "material/natural_cases.jsonl"); compiled = core.rows(C194 / "compiled/qwen3_natural.jsonl")
    bad = [r for r in rows if r["program"] == "location"]
    kept_ids = {r["case_id"] for r in rows if r["program"] in VALID_PROGRAMS}; kept = [r for r in rows if r["case_id"] in kept_ids]; kept_compiled = [r for r in compiled if r["case_id"] in kept_ids]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C198_broad_natural_program_behavior_and_signed_trajectory",
        "pre_run_naturalness_failure_registered": len(bad) == 32 and all("The rests inside" in r["prompt"] or "Inside the" in r["prompt"] for r in bad),
        "nine_programs": len(VALID_PROGRAMS) == 9 and len(kept) == 288, "balanced": sum(r["gold_position"] == 0 for r in kept) == 144,
        "partitions": {p: sum(r["partition"] == p for r in kept) for p in ("discovery", "confirmation", "fresh")} == {"discovery": 108, "confirmation": 108, "fresh": 72},
        "roles": all(set(r["role_positions"]) == set(ROLES) for r in kept_compiled),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); core.write_rows(OUT / "material/cases.jsonl", kept); core.write_rows(OUT / "material/registered_invalid_location.jsonl", bad); core.write_rows(OUT / "compiled/qwen3.jsonl", kept_compiled)
    coordinates = core.load(C195 / "protocol/source_coordinates.json")["coordinates"][:32]; core.save(OUT / "protocol/source_coordinates.json", {"coordinates": coordinates})
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "broad_natural_program_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized", "programs": list(VALID_PROGRAMS), "cases": 288,
        "registered_material_missing": {"program": "location", "cases": 32, "reason": "C194 variable-order bug produced unnatural/nonsemantic surface forms; detected before model loading; no repair or replacement in this phase"},
        "behavior_gates": {"global_min": 0.80, "program_partition_min": 0.65},
        "hidden_policy": "capture all preregistered order-A anchors for observation; semantic labels only on behavior-correct anchors; incorrect anchors remain execution-interface observations",
        "anchors": "nine programs x four frozen units x two surfaces = 72 order-A rows",
        "source": "q23 relation role, first 32 of C195 frozen source coordinates", "targets": "q24/q25 six roles x all 2560 coordinates",
        "external_transform": "apply C197 role_coordinate_gain learned only on graph discovery data; compare against identity",
        "external_gate": {"identity_nrmse_improvement_min": 0.03, "nrmse_max": 0.80},
        "claim_boundary": "controlled natural English micro-programs; machine-audited naturalness only; descriptive observations on behavior failures are not semantic evidence",
        "forbidden": ["attention", "MLP", "weights", "PCA", "repairing location after freeze", "dropping incorrect anchors after hidden reveal"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_behavior_lock_and_signed_trajectory_then_C199_unseen_composition_prediction",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks, "programs": list(VALID_PROGRAMS), "invalid_location": len(bad)}, indent=2))


@torch.inference_mode()
def behavior():
    rows = core.rows(OUT / "compiled/qwen3.jsonl"); (OUT / "raw").mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(OUT / "raw/behavior_logits.float32.npy", mode="w+", dtype=np.float32, shape=(len(rows), 2)); index = []; model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start+BATCH]; ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH); output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                value = [float(output.logits[local, lengths[local]-1, candidate[0]]) for candidate in row["candidate_ids"]]; prediction = int(value[1] > value[0]); scores[start+local] = value
                index.append({"row_index": start+local, "case_id": row["case_id"], "program": row["program"], "unit": row["unit"], "partition": row["partition"], "surface": row["surface"], "order": row["order"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "margin": float(value[row["gold_position"]] - value[1-row["gold_position"]])})
        scores.flush(); core.write_rows(OUT / "raw/behavior_index.jsonl", index)
        checks = {"rows": len(index) == 288, "finite": bool(np.isfinite(scores).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}; core.save(OUT / "analysis/behavior_run.json", {"checks": checks, "runtime": placement}); core.save(OUT / "audit/internal_behavior_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks}, indent=2))
    finally:
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def lock():
    rows = core.rows(OUT / "raw/behavior_index.jsonl"); gates = core.load(OUT / "protocol/preregistration.json")["behavior_gates"]
    acc = lambda values: float(np.mean([r["correct"] for r in values]))
    global_accuracy = acc(rows); by_program_partition = {program: {part: acc([r for r in rows if r["program"] == program and r["partition"] == part]) for part in ("discovery", "confirmation", "fresh")} for program in VALID_PROGRAMS}
    eligible = [p for p in VALID_PROGRAMS if global_accuracy >= gates["global_min"] and min(by_program_partition[p].values()) >= gates["program_partition_min"]]
    anchor_rows = [r["row_index"] for r in rows if r["unit"] in ANCHOR_UNITS and r["order"] == 1]
    lock = {"phase": PHASE, "campaign": CAMPAIGN, "status": "behavior_locked", "global_accuracy": global_accuracy, "by_program_partition": by_program_partition, "eligible_programs": eligible, "anchor_rows": anchor_rows, "correct_anchor_rows": [i for i in anchor_rows if rows[i]["correct"]], "incorrect_anchor_rows": [i for i in anchor_rows if not rows[i]["correct"]], "authorization": "run_all_registered_hidden_observations"}
    core.save(OUT / "protocol/behavior_lock.json", lock); checks = {"accounting": len(anchor_rows) == 72, "programs": len(by_program_partition) == 9, "typed": set(eligible) <= set(VALID_PROGRAMS)}; core.save(OUT / "audit/internal_lock_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps(lock, indent=2))


@torch.inference_mode()
def hidden():
    compiled = core.rows(OUT / "compiled/qwen3.jsonl"); lock_data = core.load(OUT / "protocol/behavior_lock.json"); rows = [compiled[i] for i in lock_data["anchor_rows"]]; behavior_rows = core.rows(OUT / "raw/behavior_index.jsonl")
    coords = core.load(OUT / "protocol/source_coordinates.json")["coordinates"]; (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "raw/natural_signed_trajectory.float16.npy", mode="w+", dtype=np.float16, shape=(72, 32, 2, 6, DIM)); baseline = np.lib.format.open_memmap(OUT / "raw/natural_baseline_states.float16.npy", mode="w+", dtype=np.float16, shape=(72, 4, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); base = model.model; pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def observe(row, selected, sign, epsilon):
            ids, mask, pos, _ = fixed_base.fixed_batch([row]*len(selected), pad, device, WIDTH); caught = {}
            def patch(_m, _a, value):
                state = tensor(value); changed = state.clone()
                for local, coordinate in enumerate(selected):
                    for position in row["role_positions"]["relation"]: changed[local, position, int(coordinate)] += sign*epsilon
                return (changed,) + value[1:] if isinstance(value, tuple) else changed
            hooks = [base.layers[22].register_forward_hook(patch), base.layers[23].register_forward_hook(lambda _m,_a,v: caught.__setitem__("q24", tensor(v).detach())), base.layers[24].register_forward_hook(lambda _m,_a,v: caught.__setitem__("q25", tensor(v).detach()))]
            try: model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for h in hooks: h.remove()
            result = np.empty((len(selected),2,6,DIM),np.float32)
            for local in range(len(selected)):
                for si,name in enumerate(("q24","q25")):
                    for ri,role in enumerate(ROLES): result[local,si,ri] = caught[name][local,row["role_positions"][role]].mean(0).float().cpu().numpy()
            return result
        hidden_index=[]
        for local_i, row in enumerate(rows):
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH); output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True, output_hidden_states=True); states=(output.hidden_states[0],output.hidden_states[23],output.hidden_states[24],output.hidden_states[25])
            for si,state in enumerate(states):
                for ri,role in enumerate(ROLES): baseline[local_i,si,ri]=state[0,row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            source=np.asarray(baseline[local_i,1,ROLES.index("relation")],dtype=np.float32); epsilon=0.5*float(np.sqrt(np.mean(np.square(source),dtype=np.float64)))
            for start in range(0,32,16):
                selected=coords[start:start+16]; raw[local_i,start:start+len(selected)]=((observe(row,selected,1.0,epsilon)-observe(row,selected,-1.0,epsilon))/(2*epsilon)).astype(np.float16)
            raw.flush(); baseline.flush(); b=behavior_rows[lock_data["anchor_rows"][local_i]]; hidden_index.append({"anchor_index":local_i,"row_index":b["row_index"],"case_id":row["case_id"],"program":row["program"],"unit":row["unit"],"partition":row["partition"],"surface":row["surface"],"behavior_correct":b["correct"],"behavior_margin":b["margin"],"epsilon":epsilon}); print(f"[C198] {local_i+1}/72 {row['program']} u{row['unit']} s{row['surface']} correct={b['correct']}",flush=True)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index); checks={"raw_shape":list(raw.shape)==[72,32,2,6,DIM],"baseline_shape":list(baseline.shape)==[72,4,6,DIM],"finite":bool(np.isfinite(raw).all()) and bool(np.isfinite(baseline).all()),"bf16":quant["has_bf16_parameters"],"unquantized":not quant["has_quantized_modules"]}; core.save(OUT / "analysis/hidden_run.json",{"checks":checks,"runtime":placement}); core.save(OUT / "audit/internal_hidden_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks},indent=2))
    finally:
        raw.flush(); baseline.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def metrics(pred, truth):
    e2=np.square(pred-truth,dtype=np.float64).sum(); t2=np.square(truth,dtype=np.float64).sum(); weight=np.minimum(np.abs(pred),np.abs(truth)).astype(np.float64); return {"nrmse":float(np.sqrt(e2/max(t2,1e-30))),"weighted_sign_agreement":float((weight*(np.signbit(pred)==np.signbit(truth))).sum()/max(weight.sum(),1e-30))}


def analyze():
    raw=np.load(OUT/"raw/natural_signed_trajectory.float16.npy",mmap_mode="r"); index=core.rows(OUT/"raw/hidden_index.jsonl"); gain=np.load(C197/"analysis/operators/role_coordinate_gain.float32.npy")
    identity_rows=[]; gain_rows=[]; by_program={}
    for program in VALID_PROGRAMS:
        selected=[r["anchor_index"] for r in index if r["program"]==program and r["behavior_correct"]];
        if not selected: by_program[program]={"support":0,"identity":None,"graph_gain":None}; continue
        u=np.asarray(raw[selected,:,0],dtype=np.float32); v=np.asarray(raw[selected,:,1],dtype=np.float32); im=metrics(u,v); gm=metrics(u*gain[None,None,:,:],v); identity_rows.append((im,len(selected))); gain_rows.append((gm,len(selected))); by_program[program]={"support":len(selected),"identity":im,"graph_gain":gm}
    def pooled(rows,key): return sum(m[key]*n for m,n in rows)/max(sum(n for _,n in rows),1)
    identity_nrmse=pooled(identity_rows,"nrmse"); graph_nrmse=pooled(gain_rows,"nrmse"); improvement=identity_nrmse-graph_nrmse; gate=core.load(OUT/"protocol/preregistration.json")["external_gate"]; passed=improvement>=gate["identity_nrmse_improvement_min"] and graph_nrmse<=gate["nrmse_max"]
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"broad_natural_trajectory_analyzed","behavior":{k:v for k,v in core.load(OUT/"protocol/behavior_lock.json").items() if k not in ("anchor_rows","correct_anchor_rows","incorrect_anchor_rows")},"hidden_anchors":len(index),"behavior_correct_hidden_anchors":sum(r["behavior_correct"] for r in index),"by_program":by_program,"graph_role_coordinate_gain_external":{"identity_nrmse":identity_nrmse,"graph_gain_nrmse":graph_nrmse,"improvement":improvement,"passed":passed},"interpretation":"Natural-program trajectories are descriptive unless behavior-correct. Transfer of a graph-fitted checkpoint gain tests generic local dynamics, not semantic equivalence.","next_authorization":"C199_unseen_composition_holdout_prediction"}; core.save(OUT/"analysis/natural_trajectory.json",report)
    checks={"nine_programs":len(by_program)==9,"anchors":len(index)==72,"accounting":sum(r["behavior_correct"] for r in index)+sum(not r["behavior_correct"] for r in index)==72,"finite":bool(np.isfinite([identity_nrmse,graph_nrmse,improvement]).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"behavior":report["behavior"],"external":report["graph_role_coordinate_gain_external"],"by_program":by_program,"checks":checks},indent=2))


def close():
    protocol=core.load(OUT/"protocol/preregistration.json"); report=core.load(OUT/"analysis/natural_trajectory.json"); checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"behavior":core.load(OUT/"audit/internal_behavior_audit.json")["all_checks_passed"],"lock":core.load(OUT/"audit/internal_lock_audit.json")["all_checks_passed"],"hidden":core.load(OUT/"audit/internal_hidden_audit.json")["all_checks_passed"],"analysis":core.load(OUT/"audit/internal_analysis_audit.json")["all_checks_passed"],"hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("contract","behavior","lock","hidden","analyze","close")); args=parser.parse_args(); {"contract":contract,"behavior":behavior,"lock":lock,"hidden":hidden,"analyze":analyze,"close":close}[args.command]()


if __name__=="__main__": main()
