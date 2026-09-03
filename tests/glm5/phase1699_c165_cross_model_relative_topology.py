#!/usr/bin/env python3
"""C165: coordinate-free, role-aligned relative-depth HiddenState topology across qualified models."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1699_c165_cross_model_relative_topology"
C164 = RESULT / "phase1698_c164_three_model_free_interface"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from model_utils import get_model_info
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1698_c164_three_model_free_interface as c164

PHASE, CAMPAIGN = 1699, "C165"
ROLES = ("source_record", "relation_record", "target_record", "query_source", "boundary")
RELATIVE = ("embedding", "quarter", "half", "three_quarter", "final")
BATCH = {"qwen3": 4, "glm4": 1, "deepseek7b": 1}


def now(): return datetime.now(timezone.utc).isoformat()


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C164 / "audit/independent_final_audit.json")
    summary = core.load(C164 / "analysis/summary.json")
    interface = summary["preferred_common_interface"]
    eligible = summary["common_interface_models"].get(interface, []) if interface else []
    cases = [r for r in core.rows(C164 / "material/cases.jsonl") if r["partition"] in ("confirmation", "fresh")]
    checks = {
        "authorization": parent["all_checks_passed"],
        "cases": len(cases) == 64,
        "interface_typed": (interface is None and len(eligible) == 0) or (interface in c164.INTERFACES and len(eligible) >= 2),
        "partitions": all(sum(r["partition"] == p for r in cases) == 32 for p in ("confirmation", "fresh")),
        "roles": len(ROLES) == 5,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "cross_model_relative_topology_contract_frozen" if eligible else "typed_not_tested_no_common_interface",
        "interface": interface, "eligible_models": eligible, "cases": len(cases), "roles": list(ROLES),
        "relative_checkpoints": list(RELATIVE),
        "capture": "all five semantic roles x embedding/quarter/half/three-quarter/final x every activation coordinate",
        "topology": "off-diagonal entries of the five-role cosine Gram matrix; coordinates never compared between models",
        "gates": {"within_model_partition_cosine_min": 0.50, "cross_model_matched_cosine_min": 0.50, "role_permutation_advantage_min": 0.05, "relation_permutation_advantage_min": 0.05},
        "claim_boundary": "functional role topology, not shared physical coordinates, parameter identity, or a unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cross-model coordinate equality", "post-unblind threshold changes"],
        "source_hashes": {"C164": core.sha(C164 / "analysis/final.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_eligible_models_sequentially" if eligible else "close_typed_not_tested",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


def role_positions(tokenizer, ids, row):
    values = {
        "source_record": row["source"], "relation_record": row["relation_phrase"],
        "target_record": row["intended"], "query_source": row["source"],
    }
    positions = {}
    for role, value in values.items():
        spans = graph_base.name_spans(tokenizer, ids, value)
        if not spans: raise RuntimeError((row["case_id"], role, value))
        positions[role] = spans[-1] if role == "query_source" else spans[0]
    positions["boundary"] = [len(ids) - 1]
    return positions


def checkpoint_indices(layers):
    return [0, max(1, round(layers * .25)), max(1, round(layers * .50)), max(1, round(layers * .75)), layers]


def batches(rows, size):
    for start in range(0, len(rows), size): yield rows[start:start + size]


@torch.inference_mode()
def run_model(model_name):
    protocol = core.load(OUT / "protocol/preregistration.json")
    if model_name not in protocol["eligible_models"]: raise RuntimeError((model_name, protocol["eligible_models"]))
    if (OUT / f"analysis/{model_name}.json").exists(): raise RuntimeError("already run")
    cases = core.rows(OUT / "material/cases.jsonl")
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_bf16_parameters"]: raise RuntimeError(qa)
        info = get_model_info(model, model_name)
        checkpoints = checkpoint_indices(info.n_layers)
        compiled = []
        for row in cases:
            ids = c164.render_ids(tokenizer, row, protocol["interface"])
            compiled.append({**row, "input_ids": ids, "role_positions": role_positions(tokenizer, ids, row)})
        core.write_rows(OUT / f"compiled/{model_name}.jsonl", compiled)
        state = np.zeros((len(compiled), len(checkpoints), len(ROLES), info.d_model), dtype=np.float16)
        for start, batch in enumerate(batches(compiled, BATCH[model_name])):
            width = max(len(r["input_ids"]) for r in batch)
            pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            input_ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(input_ids)
            for i, row in enumerate(batch):
                values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
                input_ids[i, width-len(values):] = values; mask[i, width-len(values):] = 1
            output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
            if len(output.hidden_states) < info.n_layers + 1: raise RuntimeError((len(output.hidden_states), info.n_layers))
            base = start * BATCH[model_name]
            for i, row in enumerate(batch):
                shift = width - len(row["input_ids"])
                for qi, q in enumerate(checkpoints):
                    h = output.hidden_states[q][i]
                    for ri, role in enumerate(ROLES):
                        pos = torch.tensor([shift + p for p in row["role_positions"][role]], device=h.device)
                        state[base+i, qi, ri] = h.index_select(0, pos).float().mean(0).cpu().numpy().astype(np.float16)
            del output, input_ids, mask
            if (start + 1) % 16 == 0: print(f"[C165] {model_name} {min((start+1)*BATCH[model_name],len(compiled))}/{len(compiled)}", flush=True)
        np.save(OUT / f"raw/{model_name}_role_states.float16.npy", state)
        report = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "status": "relative_role_field_captured",
            "shape": list(state.shape), "layers": info.n_layers, "d_model": info.d_model,
            "checkpoints": dict(zip(RELATIVE, checkpoints)), "placement": placement, "quantization_audit": qa,
            "elapsed_seconds": time.time()-started,
        }
        core.save(OUT / f"analysis/{model_name}.json", report)
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": {"shape": state.shape[:3] == (64,5,5), "finite": bool(np.isfinite(state).all()), "bf16": qa["has_bf16_parameters"], "unquantized": not qa["has_quantized_modules"]}, "all_checks_passed": bool(state.shape[:3] == (64,5,5) and np.isfinite(state).all() and qa["has_bf16_parameters"] and not qa["has_quantized_modules"])})
        print(json.dumps(report, indent=2))
    finally:
        if model is not None: release_bf16(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def gram_signature(vectors):
    x = vectors.astype(np.float64)
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    gram = x @ x.T
    return gram[np.triu_indices(len(ROLES), 1)]


def topology(model):
    rows = core.rows(OUT / "material/cases.jsonl")
    state = np.load(OUT / f"raw/{model}_role_states.float16.npy", mmap_mode="r")
    lookup = {(r["unit"],r["path_factor"],r["surface_factor"],r["target_factor"]):i for i,r in enumerate(rows)}
    signatures = {}
    for part in ("confirmation","fresh"):
        for relation in sorted({r["relation_family"] for r in rows}):
            unit_rows = [r for r in rows if r["partition"]==part and r["relation_family"]==relation and r["target_factor"]==1]
            for qi, relative in enumerate(RELATIVE):
                values=[]
                for row in unit_rows:
                    plus = state[lookup[(row["unit"],row["path_factor"],row["surface_factor"],1)],qi].astype(np.float32)
                    minus = state[lookup[(row["unit"],row["path_factor"],row["surface_factor"],-1)],qi].astype(np.float32)
                    values.append(gram_signature((plus-minus)/2))
                signatures[(part,relation,relative)] = np.mean(values,axis=0)
    return signatures


def permute_signature(signature, permutation):
    gram = np.eye(len(ROLES))
    tri = np.triu_indices(len(ROLES),1)
    gram[tri]=signature; gram[(tri[1],tri[0])]=signature
    p=gram[np.ix_(permutation,permutation)]
    return p[tri]


def analyze():
    protocol=core.load(OUT/"protocol/preregistration.json")
    models=protocol["eligible_models"]
    if len(models)<2:
        report={"phase":PHASE,"campaign":CAMPAIGN,"status":"typed_not_tested","reason":"fewer than two models share a qualified free interface","topology_gate_passed":False,"next_authorization":"C166 synthesis"}
        core.save(OUT/"analysis/summary.json",report); core.save(OUT/"audit/internal_analysis_audit.json",{"checks":{"typed":True},"all_checks_passed":True}); print(json.dumps(report,indent=2)); return
    tops={m:topology(m) for m in models}
    relations=sorted({k[1] for k in tops[models[0]]})
    within={}
    for m in models:
        vals=[cosine(tops[m][("confirmation",rel,q)],tops[m][("fresh",rel,q)]) for rel in relations for q in RELATIVE]
        within[m]={"median":float(np.median(vals)),"values":vals}
    pair_reports=[]
    role_perms=[np.roll(np.arange(len(ROLES)),shift) for shift in range(1,len(ROLES))]
    for a,b in itertools.combinations(models,2):
        matched=[]; wrong_role=[]; wrong_relation=[]
        for part in ("confirmation","fresh"):
            for ri,rel in enumerate(relations):
                wrong_rel=relations[(ri+1)%len(relations)]
                for q in RELATIVE:
                    sa,sb=tops[a][(part,rel,q)],tops[b][(part,rel,q)]
                    matched.append(cosine(sa,sb))
                    wrong_role.append(max(cosine(sa,permute_signature(sb,p)) for p in role_perms))
                    wrong_relation.append(cosine(sa,tops[b][(part,wrong_rel,q)]))
        med=float(np.median(matched)); role_adv=med-float(np.median(wrong_role)); rel_adv=med-float(np.median(wrong_relation))
        q=protocol["gates"]
        passed=med>=q["cross_model_matched_cosine_min"] and role_adv>=q["role_permutation_advantage_min"] and rel_adv>=q["relation_permutation_advantage_min"] and min(within[a]["median"],within[b]["median"])>=q["within_model_partition_cosine_min"]
        pair_reports.append({"models":[a,b],"matched_median":med,"wrong_role_median":float(np.median(wrong_role)),"role_advantage":role_adv,"wrong_relation_median":float(np.median(wrong_relation)),"relation_advantage":rel_adv,"passed":bool(passed)})
    serializable={m:{"|".join(k):v.tolist() for k,v in tops[m].items()} for m in models}
    core.save(OUT/"analysis/topology_signatures.json",serializable)
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"cross_model_relative_topology_adjudicated","interface":protocol["interface"],"models":models,"within_model":within,"cross_model_pairs":pair_reports,"topology_gate_passed":any(r["passed"] for r in pair_reports),"claim_boundary":protocol["claim_boundary"],"next_authorization":"C166 synthesis and coordinate heatmap"}
    core.save(OUT/"analysis/summary.json",report)
    checks={"models":all((OUT/f"analysis/{m}.json").exists() for m in models),"finite":all(np.isfinite(r["matched_median"]) for r in pair_reports),"pairs":len(pair_reports)==len(list(itertools.combinations(models,2)))}
    core.save(OUT/"audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())})
    print(json.dumps(report,indent=2))


def close():
    p=core.load(OUT/"protocol/preregistration.json"); s=core.load(OUT/"analysis/summary.json")
    checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"models":all(core.load(OUT/f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in p["eligible_models"]),"analysis":core.load(OUT/"audit/internal_analysis_audit.json")["all_checks_passed"]}
    final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"scientific_topology_passed":s["topology_gate_passed"],"next_authorization":s["next_authorization"]}
    core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("contract","run","analyze","close")); p.add_argument("--model",choices=("qwen3","glm4","deepseek7b")); a=p.parse_args()
    if a.command=="contract":contract()
    elif a.command=="run":
        if not a.model: raise SystemExit("--model required")
        run_model(a.model)
    elif a.command=="analyze":analyze()
    else:close()


if __name__=="__main__":main()
