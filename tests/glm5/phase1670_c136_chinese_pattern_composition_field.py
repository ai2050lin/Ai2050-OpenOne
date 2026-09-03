#!/usr/bin/env python3
"""C136 route C: Chinese compositional pattern field, observation first."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1670_c136_chinese_pattern_composition_field"
C135 = RESULT / "phase1669_c135_all_token_coordinate_transmission"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127

PHASE, CAMPAIGN = 1670, "C136"
TASKS = ("agent", "patient", "action", "attitude_event", "outer_negation", "inner_negation", "coreference", "nesting")
ROLES = ("experiencer", "agent", "action", "patient", "boundary")
CHECKPOINTS = c127.CHECKPOINTS
DIM, WIDTH, BATCH = 2560, 176, 8
NAMES = ("林岚", "周澈", "沈禾", "唐宁", "顾川", "许遥", "陆青", "叶安", "苏言", "韩溪", "程木", "秦舟", "白露", "江月", "宋野", "温桥", "夏柯", "罗星", "孟竹", "陶然")
OBJECTS = ("红苹果", "蓝盒子", "旧地图", "银钥匙", "玻璃杯", "纸风筝", "木雕像", "黄雨伞", "铜铃铛", "黑帽子", "绿皮球", "白信封", "小花瓶", "紫背包", "灰围巾", "橙皮书", "短铅笔", "长木尺", "圆镜子", "方托盘")
ACTIONS = ("吃", "搬运", "检查", "收藏", "清洗", "修理", "描画", "触碰")


def now(): return datetime.now(timezone.utc).isoformat()


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left.ravel(), right.ravel()) / denominator)


def sentence(task, e, a, v, p, nested, surface):
    active = f"{e}喜欢看到{a}{v}{p}"
    passive = f"{e}喜欢看到{p}被{a}{v}"
    if task == "outer_negation": return (f"档案记载，{e}并不喜欢看到{a}{v}{p}。" if surface == 1 else f"根据档案，{e}不喜欢看到{p}被{a}{v}。")
    if task == "inner_negation": return (f"档案记载，{e}喜欢看到{a}没有{v}{p}。" if surface == 1 else f"根据档案，{e}喜欢的是{p}没有被{a}{v}这件事。")
    if task == "coreference": return (f"{e}告诉{a}：“我喜欢看到我自己{v}{p}。”" if surface == 1 else f"{e}对{a}说：“我喜欢看我自己{v}{p}。”")
    if task == "nesting": return (f"{e}说，{a}相信{nested}{v}{p}。" if surface == 1 else f"据{e}所说，{a}相信{p}被{nested}{v}。")
    return f"档案记载，{active}。" if surface == 1 else f"根据档案，{passive}。"


def question(task, truth, e, a, v, p, nested, foil_e, foil_a, foil_v, foil_p):
    if task == "agent": return f"执行“{v}{p}”动作的人是{a if truth == 1 else foil_a}吗？"
    if task == "patient": return f"{a}{v}的对象是{p if truth == 1 else foil_p}吗？"
    if task == "action": return f"{a}对{p}执行的动作是{v if truth == 1 else foil_v}吗？"
    if task == "attitude_event": return f"喜欢“{a}{v}{p}”这件事的人是{e if truth == 1 else foil_e}吗？"
    if task == "outer_negation": return f"{e}{'不喜欢' if truth == 1 else '喜欢'}看到{a}{v}{p}吗？"
    if task == "inner_negation": return f"{e}喜欢的事件中，{a}{'没有' if truth == 1 else ''}{v}{p}吗？"
    if task == "coreference": return f"引号内两个第一人称表达都指{e if truth == 1 else a}吗？"
    if task == "nesting": return f"{e}说的内容中，相信“{nested}{v}{p}”的人是{a if truth == 1 else foil_a}吗？"
    raise KeyError(task)


def material():
    units, cases = [], []
    for i in range(16):
        e, a, nested = NAMES[i], NAMES[(i + 5) % 20], NAMES[(i + 11) % 20]
        p, v = OBJECTS[i], ACTIONS[i % len(ACTIONS)]
        partition = "discovery" if i < 8 else "confirmation"
        unit = {"unit_id": f"c136-{i:02d}", "partition": partition, "experiencer": e, "agent_value": a, "nested_agent": nested, "patient_value": p, "action_value": v}
        units.append(unit)
        for task, truth, surface in itertools.product(TASKS, (1, -1), (1, -1)):
            prompt = sentence(task, e, a, v, p, nested, surface) + question(task, truth, e, a, v, p, nested, NAMES[(i+1)%20], NAMES[(i+6)%20], ACTIONS[(i+1)%len(ACTIONS)], OBJECTS[(i+1)%20]) + "请只回答 yes 或 no。"
            role_values = {"experiencer": e, "agent": nested if task == "nesting" else (e if task == "coreference" else a), "action": v, "patient": p}
            cases.append({**unit, "case_id": f"c136-{len(cases):04d}", "task": task, "truth_factor": truth, "surface_factor": surface, "truth": truth == 1, "gold_position": 0 if truth == 1 else 1, "prompt": prompt, "role_values": role_values})
    return units, cases


def compile_rows(tokenizer, cases):
    candidate_ids = [[int(value) for value in tokenizer.encode(" " + label, add_special_tokens=False)] for label in ("yes", "no")]
    if any(len(value) != 1 for value in candidate_ids): raise RuntimeError(candidate_ids)
    rows = []
    for row in cases:
        ids = core.chat_ids(tokenizer, "请根据给定句子的字面结构回答。只输出 yes 或 no。", row["prompt"])
        role_positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans: raise RuntimeError((row["case_id"], role, value))
            role_positions[role] = spans[-1] if role in {"agent", "patient"} else spans[0]
        role_positions["boundary"] = [len(ids)-1]
        rows.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": role_positions})
    return rows


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C135 / "audit/independent_closure_audit.json")
    units, cases = material(); compiled = compile_rows(graph_base.tokenizer(), cases)
    cell = {(partition, task, truth, surface): 0 for partition in ("discovery","confirmation") for task in TASKS for truth in (1,-1) for surface in (1,-1)}
    for row in cases: cell[(row["partition"],row["task"],row["truth_factor"],row["surface_factor"])]+=1
    zero={"always_yes":float(np.mean([row["truth"] for row in cases])),"always_no":float(np.mean([not row["truth"] for row in cases])),"surface":float(np.mean([(row["surface_factor"]==1)==row["truth"] for row in cases]))}
    checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="start_route_C_C136","units":len(units)==16,"cases":len(cases)==512,"cells":set(cell.values())=={8},"unique":len({row["prompt"] for row in cases})==512,"zero":all(value==0.5 for value in zero.values()),"roles":all(set(row["role_positions"])==set(ROLES) for row in compiled),"width":max(len(row["prompt_ids"]) for row in compiled)<WIDTH}
    if not all(checks.values()):raise RuntimeError(checks)
    core.write_rows(OUT/"material/units.jsonl",units);core.write_rows(OUT/"material/cases.jsonl",cases);core.write_rows(OUT/"compiled/qwen3.jsonl",compiled)
    paths={"c135_audit":C135/"audit/independent_closure_audit.json"}
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"route_C_chinese_pattern_contract_frozen","object":"Chinese experiencer-agent-action-patient, negation, coreference, and nesting pattern field","model":"Qwen3-4B local BF16 CUDA nonquantized","tasks":list(TASKS),"roles":list(ROLES),"units":16,"cases":512,"behavior_gate":{"global_min":0.90,"partition_min":0.85,"truth_min":0.85,"surface_min":0.85,"task_min":0.80},"confirmation_gate":{"task_cosine_min":0.70,"top256_overlap_min":0.30,"passing_tasks_min":6,"gram_cosine_min":0.75},"observation":"full 2560 coordinates at 38 strict checkpoints; no isolated-direction assumption","naturalness_boundary":"controlled but grammatical Chinese templates; machine audit only, no independent human blind naturalness rating","forbidden":["PCA","SVD","attention","MLP","weights"],"source_paths":{k:str(v) for k,v in paths.items()},"source_hashes":{k:core.sha(v) for k,v in paths.items()},"producer_sha256":core.sha(Path(__file__)),"authorization":"run_c136_behavior"}
    core.save(OUT/"protocol/preregistration.json",protocol);core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":protocol["authorization"]});print(json.dumps({"checks":checks,"zero_models":zero,"max_width":max(len(row["prompt_ids"]) for row in compiled)},ensure_ascii=False,indent=2))


def accuracy(rows):return float(np.mean([row["correct"] for row in rows]))


@torch.inference_mode()
def behavior():
    protocol=core.load(OUT/"protocol/preregistration.json");rows=core.rows(OUT/"compiled/qwen3.jsonl")
    if protocol["authorization"]!="run_c136_behavior":raise RuntimeError("unauthorized")
    path=OUT/"raw/qwen3_candidate_logits.float32.npy";path.parent.mkdir(parents=True,exist_ok=True);raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(rows),2));results=[];model=None;repeat=0.0
    try:
        model,tokenizer,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def run(batch):
            ids,mask,positions,lengths=fixed_base.fixed_batch(batch,pad,device,WIDTH);out=model.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True);boundary=torch.stack([out.last_hidden_state[i,n-1] for i,n in enumerate(lengths)]);logits=model.lm_head(boundary).float();scores=np.asarray([[float(logits[i,c[0]]) for c in row["candidate_ids"]] for i,row in enumerate(batch)],dtype=np.float32);return scores,out,ids,mask,positions
        for start in range(0,len(rows),BATCH):
            batch=rows[start:start+BATCH];scores,out,ids,mask,positions=run(batch);raw[start:start+len(batch)]=scores
            for i,row in enumerate(batch):
                pred=int(scores[i,1]>scores[i,0]);results.append({"row_index":start+i,"case_id":row["case_id"],"unit_id":row["unit_id"],"partition":row["partition"],"task":row["task"],"truth_factor":row["truth_factor"],"surface_factor":row["surface_factor"],"prediction":pred,"gold_position":row["gold_position"],"correct":pred==row["gold_position"]})
            del out,ids,mask,positions
        raw.flush();scores,out,ids,mask,positions=run(rows[:BATCH]);repeat=float(np.max(np.abs(scores-np.asarray(raw[:BATCH]))))
    finally:
        raw.flush()
        if model is not None:release_bf16(model)
        gc.collect();torch.cuda.empty_cache()
    core.write_rows(OUT/"raw/qwen3_behavior_index.jsonl",results)
    summary={"global_accuracy":accuracy(results),"by_partition":{k:accuracy([r for r in results if r["partition"]==k]) for k in ("discovery","confirmation")},"by_truth":{str(k):accuracy([r for r in results if r["truth_factor"]==k]) for k in (1,-1)},"by_surface":{str(k):accuracy([r for r in results if r["surface_factor"]==k]) for k in (1,-1)},"by_task":{k:accuracy([r for r in results if r["task"]==k]) for k in TASKS}}
    g=protocol["behavior_gate"];gate=summary["global_accuracy"]>=g["global_min"] and min(summary["by_partition"].values())>=g["partition_min"] and min(summary["by_truth"].values())>=g["truth_min"] and min(summary["by_surface"].values())>=g["surface_min"] and min(summary["by_task"].values())>=g["task_min"]
    checks={"rows":len(results)==512,"finite":bool(np.isfinite(raw).all()),"repeat":repeat==0.0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"behavior_qualified" if gate else "behavior_failed","summary":summary,"checks":checks,"gate_passed":gate,"repeat_logits_max_abs":repeat,"authorization":"capture_c136_pattern_field" if gate else "close_c136_continue_D"};core.save(OUT/"analysis/behavior.json",report);core.save(OUT/"audit/internal_behavior_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":report["authorization"]});print(json.dumps(report,ensure_ascii=False,indent=2))


def tensor_output(value):return value[0] if isinstance(value,tuple) else value


@torch.inference_mode()
def capture():
    if core.load(OUT/"analysis/behavior.json")["authorization"]!="capture_c136_pattern_field":raise RuntimeError("unauthorized")
    rows=core.rows(OUT/"compiled/qwen3.jsonl");path=OUT/"raw/qwen3_pattern_role_field.bf16.npy";raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint16,shape=(512,5,38,DIM));model=None;repeat=0
    try:
        model,tokenizer,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        def run(batch):
            cap={};handles=[model.model.embed_tokens.register_forward_hook(lambda _m,_a,o:cap.__setitem__("e",tensor_output(o).detach()))]
            for li,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(lambda _m,_a,o,j=li:cap.__setitem__(f"b{j}",tensor_output(o).detach())))
            handles.append(model.model.norm.register_forward_hook(lambda _m,_a,o:cap.__setitem__("n",tensor_output(o).detach())))
            try:ids,mask,pos,lengths=fixed_base.fixed_batch(batch,pad,device,WIDTH);out=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
            finally:
                for h in handles:h.remove()
            return [cap["e"],*[cap[f"b{i}"] for i in range(36)],cap["n"]],out,ids,mask,pos
        for start in range(0,512,BATCH):
            batch=rows[start:start+BATCH];tensors,out,ids,mask,pos=run(batch)
            for i,row in enumerate(batch):
                for ri,role in enumerate(ROLES):
                    for qi,tensor in enumerate(tensors):raw[start+i,ri,qi]=tensor[i,row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
            if (start//BATCH+1)%16==0:raw.flush();print(f"[C136] {start+len(batch)}/512",flush=True)
            del tensors,out,ids,mask,pos
        raw.flush();tensors,out,ids,mask,pos=run(rows[:BATCH])
        for i,row in enumerate(rows[:BATCH]):
            for ri,role in enumerate(ROLES):
                for qi,tensor in enumerate(tensors):
                    bits=tensor[i,row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy();repeat=max(repeat,int(np.max(np.abs(bits.astype(np.int64)-raw[i,ri,qi].astype(np.int64)))))
    finally:
        raw.flush()
        if model is not None:release_bf16(model)
        gc.collect();torch.cuda.empty_cache()
    checks={"shape":list(raw.shape)==[512,5,38,DIM],"finite":bool(np.isfinite(c127.decode(raw[:2])).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};report={"phase":PHASE,"campaign":CAMPAIGN,"status":"capture_complete","checks":checks,"shape":list(raw.shape),"sha256":core.sha(path),"authorization":"discover_c136_patterns"};core.save(OUT/"analysis/capture.json",report);core.save(OUT/"audit/internal_capture_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":report["authorization"]});print(json.dumps(report,indent=2))


def fields(partition):
    rows=core.rows(OUT/"compiled/qwen3.jsonl");units=core.rows(OUT/"material/units.jsonl");selected=[u for u in units if u["partition"]==partition];lookup={u["unit_id"]:i for i,u in enumerate(selected)};raw=np.load(OUT/"raw/qwen3_pattern_role_field.bf16.npy",mmap_mode="r");out=np.zeros((len(selected),len(TASKS),len(ROLES),38,DIM),np.float32)
    for i,row in enumerate(rows):
        if row["partition"]==partition:out[lookup[row["unit_id"]],TASKS.index(row["task"])]+=float(row["truth_factor"])/4*c127.decode(raw[i])
    return out


def top256(v):return set(np.argpartition(np.abs(v),-256)[-256:].tolist())


def discover():
    f=fields("discovery");np.save(OUT/"analysis/discovery_pattern_fields.float32.npy",f);left=f[:4].mean(0);right=f[4:].mean(0);nominees={}
    for ti,task in enumerate(TASKS):
        candidates=[]
        for ri,role in enumerate(ROLES):
            for q in range(38):
                c=cosine(left[ti,ri,q],right[ti,ri,q]);n=min(float(np.linalg.norm(left[ti,ri,q])),float(np.linalg.norm(right[ti,ri,q])));candidates.append((max(c,0)*n,c,ri,q))
        _,stability,ri,q=max(candidates);vector=f[:,ti,ri,q].mean(0);path=OUT/f"protocol/discovery_{task}.float32.npy";path.parent.mkdir(parents=True,exist_ok=True);np.save(path,vector);nominees[task]={"role":ROLES[ri],"role_index":ri,"checkpoint":CHECKPOINTS[q],"checkpoint_index":q,"split_half_cosine":stability,"vector_sha256":core.sha(path),"support":sorted(top256(vector))}
    freeze={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"pattern_nominees_frozen","nominees":nominees,"confirmation_unread":True,"authorization":"validate_c136_confirmation"};core.save(OUT/"protocol/frozen_patterns.json",freeze);checks={"shape":list(f.shape)==[8,8,5,38,DIM],"finite":bool(np.isfinite(f).all()),"tasks":len(nominees)==8};core.save(OUT/"audit/internal_discovery_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":freeze["authorization"]});print(json.dumps(freeze,ensure_ascii=False,indent=2))


def validate():
    protocol=core.load(OUT/"protocol/preregistration.json");freeze=core.load(OUT/"protocol/frozen_patterns.json");f=fields("confirmation");np.save(OUT/"analysis/confirmation_pattern_fields.float32.npy",f);results={};disc_vectors=[];conf_vectors=[]
    for ti,task in enumerate(TASKS):
        n=freeze["nominees"][task];d=np.load(OUT/f"protocol/discovery_{task}.float32.npy");c=f[:,ti,n["role_index"],n["checkpoint_index"]].mean(0);overlap=len(set(n["support"])&top256(c))/256;co=cosine(d,c);passed=co>=protocol["confirmation_gate"]["task_cosine_min"] and overlap>=protocol["confirmation_gate"]["top256_overlap_min"];results[task]={"cosine":co,"top256_overlap":overlap,"passed":passed};disc_vectors.append(d);conf_vectors.append(c)
    dg=np.asarray([[cosine(a,b) for b in disc_vectors] for a in disc_vectors]);cg=np.asarray([[cosine(a,b) for b in conf_vectors] for a in conf_vectors]);gram=cosine(dg[np.triu_indices(8,1)],cg[np.triu_indices(8,1)]);passing=sum(r["passed"] for r in results.values());gate=passing>=protocol["confirmation_gate"]["passing_tasks_min"] and gram>=protocol["confirmation_gate"]["gram_cosine_min"];report={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_C_confirmation_adjudicated","tasks":results,"passing_tasks":passing,"cross_task_gram_cosine":gram,"prediction_gate_passed":gate,"authorization":"close_c136_continue_D"};core.save(OUT/"analysis/confirmation.json",report);checks={"shape":list(f.shape)==[8,8,5,38,DIM],"finite":bool(np.isfinite(f).all()),"tasks":len(results)==8};core.save(OUT/"audit/internal_confirmation_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":report["authorization"]});print(json.dumps(report,ensure_ascii=False,indent=2))


def close():
    b=core.load(OUT/"analysis/behavior.json")
    if not b["gate_passed"]:closure={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_C_behavior_failed","headline":b["summary"],"claim_boundary":"behavior only; no Chinese pattern HiddenState","next_authorization":"continue route D"}
    else:
        c=core.load(OUT/"analysis/confirmation.json");closure={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_C_closed","headline":{"behavior":b["summary"],"confirmation":c},"theory_update":"typed multi-role pattern response graph adjudicated without assuming isolated directions","problems":["controlled templates","machine-only naturalness","Qwen3 only","role spans registered from text","readability is not causality"],"claim_boundary":"Chinese pattern response observation, not a complete grammar mechanism","next_authorization":"continue route D"}
    core.save(OUT/"analysis/closure.json",closure);checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"behavior":core.load(OUT/"audit/internal_behavior_audit.json")["all_checks_passed"],"branch":(not b["gate_passed"]) or all((OUT/p).exists() for p in ("analysis/capture.json","protocol/frozen_patterns.json","analysis/confirmation.json"))};core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":b["gate_passed"] and core.load(OUT/"analysis/confirmation.json")["prediction_gate_passed"] if b["gate_passed"] else False,"authorization":"independent_audit_then_route_D"});print(json.dumps(closure,ensure_ascii=False,indent=2))


def main():
    modes={"contract":contract,"behavior":behavior,"capture":capture,"discover":discover,"validate":validate,"close":close};modes[sys.argv[1]]()
if __name__=="__main__":main()
