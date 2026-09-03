#!/usr/bin/env python3
"""C201: model-specific behavior interfaces and cross-model relative HiddenState topology."""
from __future__ import annotations
import argparse,gc,json,sys,time
from collections.abc import Mapping
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; OUT=RESULT/"phase1735_c201_cross_model_functional_topology"; C198=RESULT/"phase1732_c198_broad_natural_program_trajectory"; C200=RESULT/"phase1734_c200_natural_deletion_rescue_adjudication"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import MODELS,load_bf16,quantization_audit,release_bf16
from model_utils import get_model_info
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE,CAMPAIGN=1735,"C201"; INTERFACES=("direct_chat","ab_chat","plain_direct"); ROLES=("primary","secondary","relation","context","query","boundary"); BATCH={"qwen3":8,"glm4":1,"deepseek7b":1}


def render_chat(tokenizer,system,user):
    messages=[{"role":"system","content":system},{"role":"user","content":user}]; kwargs={"tokenize":True,"add_generation_prompt":True}
    try: ids=tokenizer.apply_chat_template(messages,enable_thinking=False,**kwargs)
    except (TypeError,ValueError):
        try: ids=tokenizer.apply_chat_template(messages,**kwargs)
        except Exception: ids=tokenizer.encode("\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)+"\nASSISTANT:",add_special_tokens=True)
    if isinstance(ids,Mapping): ids=ids["input_ids"]
    if isinstance(ids,torch.Tensor): ids=ids.tolist()
    if ids and isinstance(ids[0],list): ids=ids[0]
    return [int(x) for x in ids]


def direct_prompt(row): return row["prompt"].split("(A)",1)[0].strip()+" Answer with the exact answer phrase."


def compile_interface(tokenizer,row,interface):
    if interface=="ab_chat":
        ids=render_chat(tokenizer,"Answer from the supplied statement. Reply exactly A or B.",row["prompt"]); candidates=[tokenizer.encode(" A",add_special_tokens=False),tokenizer.encode(" B",add_special_tokens=False)]
    elif interface=="direct_chat":
        ids=render_chat(tokenizer,"Answer from the supplied statement using only the exact requested phrase.",direct_prompt(row)); candidates=[tokenizer.encode(" "+x,add_special_tokens=False) for x in row["answer_candidates"]]
    else:
        ids=tokenizer.encode("Question: "+direct_prompt(row)+"\nAnswer:",add_special_tokens=True); candidates=[tokenizer.encode(" "+x,add_special_tokens=False) for x in row["answer_candidates"]]
    if any(not x for x in candidates): raise RuntimeError((row["case_id"],interface,candidates))
    return ids,candidates


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(C200/"audit/independent_final_audit.json"); rows=core.rows(C198/"material/cases.jsonl")
    checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="C201_cross_model_model_specific_interfaces_and_relative_topology","models":tuple(MODELS)==("qwen3","glm4","deepseek7b"),"cases":len(rows)==288,"interfaces":len(INTERFACES)==3,"programs":len({r["program"] for r in rows})==9}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); core.write_rows(OUT/"material/cases.jsonl",rows)
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"cross_model_functional_topology_frozen","models":list(MODELS),"sequential_loading":True,"interfaces":list(INTERFACES),"interface_selection":"highest discovery accuracy, ties follow direct_chat then ab_chat then plain_direct","holdout_qualification":{"global_confirmation_fresh_min":0.75,"program_partition_min":0.60},"common_rule":"program qualifies on at least two models; topology rows must be correct for every participating model","topology":"five relative HiddenState checkpoints x six semantic roles; compare normalized transition-energy topology and signed balance, never physical coordinate ids","topology_gate":{"common_programs_min":4,"common_rows_min":8,"median_pair_similarity_min":0.65},"raw_states":"per-model full physical activation coordinates saved separately for selected cases","claim_boundary":"functional topology only; model-specific coordinates, tokenizers and depths are not identified with one another","forbidden":["attention","MLP","weights","PCA","same coordinate ids","simultaneous model loading","selecting interface on holdout"],"producer_sha256":core.sha(Path(__file__)),"authorization":"run_three_models_sequentially_then_C202_theory_adjudication"}; core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks,"interfaces":list(INTERFACES)},indent=2))


@torch.inference_mode()
def run_behavior(model_name):
    if (OUT/f"analysis/{model_name}_behavior.json").exists(): raise RuntimeError("already run")
    rows=core.rows(OUT/"material/cases.jsonl"); model=tokenizer=None; started=time.time()
    try:
        model,tokenizer,device,placement=load_bf16(model_name); quant=quantization_audit(model); info=get_model_info(model,model_name); all_results={}
        pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for interface in INTERFACES:
            compiled=[]
            for row in rows:
                ids,candidates=compile_interface(tokenizer,row,interface); compiled.append((row,ids,candidates))
            results=[]
            for start in range(0,len(compiled),BATCH[model_name]):
                batch=compiled[start:start+BATCH[model_name]]; expanded=[]
                for row,ids,candidates in batch:
                    for ci,candidate in enumerate(candidates): expanded.append((row,ci,ids+candidate,len(ids),candidate))
                width=max(len(x[2]) for x in expanded); ids_tensor=torch.full((len(expanded),width),pad,dtype=torch.long,device=device); mask=torch.zeros_like(ids_tensor)
                for i,(_r,_ci,values,_pl,_c) in enumerate(expanded): ids_tensor[i,:len(values)]=torch.tensor(values,dtype=torch.long,device=device); mask[i,:len(values)]=1
                output=model(input_ids=ids_tensor,attention_mask=mask,use_cache=False,return_dict=True); logp=torch.log_softmax(output.logits.float(),dim=-1); batch_scores=np.zeros((len(batch),2),np.float32)
                for i,(_r,ci,values,prompt_length,candidate) in enumerate(expanded):
                    value=0.0
                    for k,token_id in enumerate(candidate): value+=float(logp[i,prompt_length+k-1,token_id])
                    batch_scores[i//2,ci]=value
                for local,(row,_ids,_cand) in enumerate(batch):
                    pred=int(batch_scores[local,1]>batch_scores[local,0]); results.append({"case_id":row["case_id"],"program":row["program"],"unit":row["unit"],"partition":row["partition"],"surface":row["surface"],"gold_position":row["gold_position"],"prediction":pred,"correct":pred==row["gold_position"],"score0":float(batch_scores[local,0]),"score1":float(batch_scores[local,1])})
                del ids_tensor,mask,output,logp
            core.write_rows(OUT/f"raw/{model_name}_{interface}.jsonl",results); all_results[interface]=results; print(f"[C201] {model_name} {interface} accuracy={np.mean([r['correct'] for r in results]):.4f}",flush=True)
        discovery={interface:float(np.mean([r["correct"] for r in values if r["partition"]=="discovery"])) for interface,values in all_results.items()}; selected=max(INTERFACES,key=lambda x:(discovery[x],-INTERFACES.index(x))); selected_rows=all_results[selected]; holdout=[r for r in selected_rows if r["partition"]!="discovery"]; global_holdout=float(np.mean([r["correct"] for r in holdout])); by_program={p:{part:float(np.mean([r["correct"] for r in selected_rows if r["program"]==p and r["partition"]==part])) for part in ("confirmation","fresh")} for p in sorted({r["program"] for r in selected_rows})}; q=core.load(OUT/"protocol/preregistration.json")["holdout_qualification"]; eligible=[p for p,v in by_program.items() if global_holdout>=q["global_confirmation_fresh_min"] and min(v.values())>=q["program_partition_min"]]
        report={"phase":PHASE,"campaign":CAMPAIGN,"model":model_name,"status":"behavior_interface_locked","discovery_accuracy":discovery,"selected_interface":selected,"holdout_accuracy":global_holdout,"by_program_holdout":by_program,"eligible_programs":eligible,"model_info":{"layers":info.n_layers,"d_model":info.d_model,"class":info.model_class},"placement":placement,"quantization":quant,"elapsed_seconds":time.time()-started}; core.save(OUT/f"analysis/{model_name}_behavior.json",report); checks={"rows":all(len(v)==288 for v in all_results.values()),"finite":all(np.isfinite([[r["score0"],r["score1"]] for r in values]).all() for values in all_results.values()),"bf16":quant["has_bf16_parameters"],"unquantized":not quant["has_quantized_modules"]}; core.save(OUT/f"audit/internal_{model_name}_behavior_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps(report,indent=2))
    finally:
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def lock():
    reports={m:core.load(OUT/f"analysis/{m}_behavior.json") for m in MODELS}; programs=sorted({p for r in reports.values() for p in r["eligible_programs"]}); common={p:[m for m in MODELS if p in reports[m]["eligible_programs"]] for p in programs}; common_programs=[p for p,m in common.items() if len(m)>=2]
    selected_results={m:{r["case_id"]:r for r in core.rows(OUT/f"raw/{m}_{reports[m]['selected_interface']}.jsonl")} for m in MODELS}; material=core.rows(OUT/"material/cases.jsonl"); selected=[]
    for program in common_programs:
        participants=common[program]; candidates=[r for r in material if r["program"]==program and r["partition"]!="discovery" and all(selected_results[m][r["case_id"]]["correct"] for m in participants)]
        selected.extend(candidates[:2])
    core.write_rows(OUT/"protocol/topology_cases.jsonl",selected)
    result={"phase":PHASE,"campaign":CAMPAIGN,"status":"cross_model_behavior_locked","reports":reports,"common_program_models":common,"common_programs":common_programs,"topology_case_ids":[r["case_id"] for r in selected],"topology_authorized":len(common_programs)>=4 and len(selected)>=8,"authorization":"run_relative_hidden_topology" if len(common_programs)>=4 and len(selected)>=8 else "topology_typed_not_tested"}; core.save(OUT/"protocol/cross_model_lock.json",result); checks={"models":set(reports)==set(MODELS),"program_accounting":all(set(v)<=set(MODELS) for v in common.values()),"selected_unique":len(selected)==len({r["case_id"] for r in selected})}; core.save(OUT/"audit/internal_lock_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps(result,indent=2))


@torch.inference_mode()
def run_hidden(model_name):
    lock_data=core.load(OUT/"protocol/cross_model_lock.json"); report=core.load(OUT/f"analysis/{model_name}_behavior.json")
    if not lock_data["topology_authorized"] or not report["eligible_programs"]:
        core.save(OUT/f"analysis/{model_name}_topology.json",{"model":model_name,"status":"typed_not_tested","reason":"cross-model topology not authorized or model has no eligible programs"}); core.save(OUT/f"audit/internal_{model_name}_topology_audit.json",{"checks":{"typed":True},"all_checks_passed":True}); return
    cases=core.rows(OUT/"protocol/topology_cases.jsonl"); selected_results={r["case_id"]:r for r in core.rows(OUT/f"raw/{model_name}_{report['selected_interface']}.jsonl")}; cases=[r for r in cases if r["program"] in report["eligible_programs"] and selected_results[r["case_id"]]["correct"]]
    model=tokenizer=None
    try:
        model,tokenizer,device,placement=load_bf16(model_name); quant=quantization_audit(model); info=get_model_info(model,model_name); state_path=OUT/f"raw/{model_name}_topology_states.float16.npy"; state_path.parent.mkdir(parents=True,exist_ok=True); states=np.lib.format.open_memmap(state_path,mode="w+",dtype=np.float16,shape=(len(cases),5,6,info.d_model)); index=[]
        for ci,row in enumerate(cases):
            ids,_candidates=compile_interface(tokenizer,row,report["selected_interface"]); positions={}
            for role,value in row["role_values"].items():
                spans=graph_base.name_spans(tokenizer,ids,value)
                if not spans: raise RuntimeError((model_name,row["case_id"],role,value,report["selected_interface"]))
                positions[role]=spans[-1] if role=="query" else spans[0]
            positions["boundary"]=[len(ids)-1]; input_ids=torch.tensor([ids],dtype=torch.long,device=device); mask=torch.ones_like(input_ids); output=model(input_ids=input_ids,attention_mask=mask,use_cache=False,return_dict=True,output_hidden_states=True); hidden=output.hidden_states; checkpoints=sorted(set(int(round(f*(len(hidden)-1))) for f in (0,.25,.5,.75,1)))
            if len(checkpoints)!=5: raise RuntimeError(checkpoints)
            for si,hidx in enumerate(checkpoints):
                for ri,role in enumerate(ROLES): states[ci,si,ri]=hidden[hidx][0,positions[role]].mean(0).float().cpu().numpy().astype(np.float16)
            index.append({"case_index":ci,"case_id":row["case_id"],"program":row["program"],"interface":report["selected_interface"],"checkpoint_indices":checkpoints}); states.flush(); print(f"[C201] hidden {model_name} {ci+1}/{len(cases)} {row['program']}",flush=True)
        core.write_rows(OUT/f"raw/{model_name}_topology_index.jsonl",index); transition=np.diff(np.asarray(states,dtype=np.float32),axis=1); rms=np.sqrt(np.mean(np.square(transition,dtype=np.float64),axis=-1)); topology=rms.mean(axis=0); topology=topology/np.maximum(topology.sum(axis=1,keepdims=True),1e-30); positive=(transition>0).mean(axis=(0,3)); top_report={"model":model_name,"status":"relative_topology_observed","cases":len(cases),"d_model":info.d_model,"layers":info.n_layers,"interface":report["selected_interface"],"transition_role_energy":topology.tolist(),"transition_positive_fraction":positive.tolist(),"placement":placement,"quantization":quant}; core.save(OUT/f"analysis/{model_name}_topology.json",top_report); checks={"cases":len(cases)==len(index),"shape":list(states.shape)==[len(cases),5,6,info.d_model],"finite":bool(np.isfinite(states).all()),"normalized":bool(np.allclose(topology.sum(axis=1),1.0))}; core.save(OUT/f"audit/internal_{model_name}_topology_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps(top_report,indent=2))
    finally:
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def analyze():
    lock_data=core.load(OUT/"protocol/cross_model_lock.json"); topologies={m:core.load(OUT/f"analysis/{m}_topology.json") for m in MODELS}; observed=[m for m,v in topologies.items() if v["status"]=="relative_topology_observed"]; pairs=[]
    for i,m1 in enumerate(observed):
        for m2 in observed[i+1:]:
            a=np.asarray(topologies[m1]["transition_role_energy"],dtype=np.float64); b=np.asarray(topologies[m2]["transition_role_energy"],dtype=np.float64); pairs.append({"models":[m1,m2],"similarity":float(np.mean(1-.5*np.abs(a-b).sum(axis=1)))})
    median=float(np.median([p["similarity"] for p in pairs])) if pairs else None; gate=core.load(OUT/"protocol/preregistration.json")["topology_gate"]; passed=lock_data["topology_authorized"] and len(observed)>=2 and len(lock_data["common_programs"])>=gate["common_programs_min"] and len(lock_data["topology_case_ids"])>=gate["common_rows_min"] and median is not None and median>=gate["median_pair_similarity_min"]
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"cross_model_functional_topology_analyzed" if pairs else "cross_model_topology_typed_not_tested","behavior":{"selected_interface":{m:lock_data["reports"][m]["selected_interface"] for m in MODELS},"holdout_accuracy":{m:lock_data["reports"][m]["holdout_accuracy"] for m in MODELS},"eligible_programs":{m:lock_data["reports"][m]["eligible_programs"] for m in MODELS},"common_programs":lock_data["common_programs"],"topology_cases":len(lock_data["topology_case_ids"])},"topologies":topologies,"pair_similarity":pairs,"median_pair_similarity":median,"topology_gate_passed":passed,"claim_boundary":"Relative role/checkpoint topology only; no coordinate identity or common internal code is inferred.","next_authorization":"C202_campaign_theory_adjudication_and_heatmap"}; core.save(OUT/"analysis/cross_model_topology.json",report); checks={"models":set(topologies)==set(MODELS),"pairs_typed":all(0<=p["similarity"]<=1 for p in pairs),"finite":median is None or bool(np.isfinite(median))}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps(report,indent=2))


def close():
    protocol=core.load(OUT/"protocol/preregistration.json"); report=core.load(OUT/"analysis/cross_model_topology.json"); checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"behavior":all(core.load(OUT/f"audit/internal_{m}_behavior_audit.json")["all_checks_passed"] for m in MODELS),"lock":core.load(OUT/"audit/internal_lock_audit.json")["all_checks_passed"],"topology":all(core.load(OUT/f"audit/internal_{m}_topology_audit.json")["all_checks_passed"] for m in MODELS),"analysis":core.load(OUT/"audit/internal_analysis_audit.json")["all_checks_passed"],"hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("contract","behavior","lock","hidden","analyze","close")); p.add_argument("--model",choices=MODELS); a=p.parse_args()
    if a.command=="contract":contract()
    elif a.command=="behavior":
        if not a.model: raise SystemExit("--model required")
        run_behavior(a.model)
    elif a.command=="lock":lock()
    elif a.command=="hidden":
        if not a.model: raise SystemExit("--model required")
        run_hidden(a.model)
    elif a.command=="analyze":analyze()
    else:close()
if __name__=="__main__":main()
