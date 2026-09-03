#!/usr/bin/env python3
"""Full-token, full-coordinate Q/K/V edge ledger on the token-atomic atlas."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2538 = RESULT / "phase2538_c117505_c121600_token_atomic_hypergraph_behavior"
OUT = RESULT / "phase2539_c121601_c125696_full_token_qkv_edge_ledger"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2539, "C121601-C125696"
LATE = tuple(range(20, 36))
REGIONS = ("frame", "facts_entity", "facts_relation", "facts_value", "question_context", "query_property", "candidate", "instruction", "answer_boundary")
EXTERNAL_REGIONS = tuple(range(8))
sys.path.insert(0, str(TESTS)); import model_utils  # noqa: E402


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8-sig"))
def read(path: Path) -> list[dict]: return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
    return h.hexdigest()
def allocate(path: Path, shape: tuple[int,...], dtype=np.float16):
    path.parent.mkdir(parents=True,exist_ok=True); x=np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=shape); x[:]=np.nan if np.issubdtype(dtype,np.floating) else -1; return x
def pad(seqs:list[list[int]],pad_id:int,device):
    width=max(map(len,seqs));ids=torch.full((len(seqs),width),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
    return ids,mask
def rel_rms(a:torch.Tensor,b:torch.Tensor)->float:
    a=a.detach().float();b=b.detach().float()
    return float(torch.sqrt(torch.mean((a-b)**2))/(torch.sqrt(torch.mean(b**2))+1e-12))


def collect(model, tokenizer, rows:list[dict])->tuple[list[dict],dict,dict]:
    rows=sorted(rows,key=lambda r:(r["unit"],r["family_id"],r["language"],r["meaning_swap"],r["query_property"]))
    layers=model_utils.get_layers(model);cfg=model.config;n=len(rows);max_seq=max(len(r["prompt_ids"])for r in rows)
    nh=int(cfg.num_attention_heads);nkv=int(cfg.num_key_value_heads);hd=int(cfg.head_dim);dim=int(cfg.hidden_size);nr=len(REGIONS)
    paths={
        "embedding":OUT/"fields/token_embedding.float16.npy",
        "hidden":OUT/"fields/full_token_residual_q20_q36.float16.npy",
        "query":OUT/"fields/answer_query_post_rope.float16.npy",
        "key":OUT/"fields/source_key_post_rope.float16.npy",
        "value":OUT/"fields/source_value.float16.npy",
        "qk_logit":OUT/"fields/answer_source_qk_logit.float16.npy",
        "attention":OUT/"fields/answer_source_softmax.float16.npy",
        "weighted_value":OUT/"fields/answer_source_weighted_value.float16.npy",
        "region_head":OUT/"fields/region_head_weighted_value.float16.npy",
        "region_residual":OUT/"fields/region_wo_residual.float16.npy",
    }
    arrays={
        "embedding":allocate(paths["embedding"],(n,max_seq,dim)),
        "hidden":allocate(paths["hidden"],(n,17,max_seq,dim)),
        "query":allocate(paths["query"],(n,16,nh,hd)),
        "key":allocate(paths["key"],(n,16,nkv,max_seq,hd)),
        "value":allocate(paths["value"],(n,16,nkv,max_seq,hd)),
        "qk_logit":allocate(paths["qk_logit"],(n,16,nh,max_seq)),
        "attention":allocate(paths["attention"],(n,16,nh,max_seq)),
        "weighted_value":allocate(paths["weighted_value"],(n,16,nh,max_seq,hd)),
        "region_head":allocate(paths["region_head"],(n,16,nh,nr,hd)),
        "region_residual":allocate(paths["region_residual"],(n,16,nr,dim)),
    }
    residual_in={};norm_input={};pos_emb={};o_input={};attn_out={};layer_out={};handles=[]
    for li in LATE:
        def layer_pre(_m,args,li=li):residual_in[li]=args[0].detach()
        handles.append(layers[li].register_forward_pre_hook(layer_pre))
        def attn_pre(_m,args,kwargs,li=li):
            norm_input[li]=(args[0] if args else kwargs["hidden_states"]).detach();pos_emb[li]=tuple(x.detach() for x in kwargs["position_embeddings"])
        handles.append(layers[li].self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True))
        def opre(_m,args,li=li):o_input[li]=args[0].detach()
        handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(opre))
        def apost(_m,_a,out,li=li):attn_out[li]=out[0].detach()
        handles.append(layers[li].self_attn.register_forward_hook(apost))
        def lpost(_m,_a,out,li=li):layer_out[li]=(out[0] if isinstance(out,tuple) else out).detach()
        handles.append(layers[li].register_forward_hook(lpost))
    route_sum={"head_rms":np.zeros((16,nh)),"external_mass":np.zeros((16,nh)),"entropy":np.zeros((16,nh))};count=0
    max_qk_error=max_pre_error=max_res_error=0.0;index_rows=[];device=model.get_input_embeddings().weight.device
    try:
        for start in range(0,n,2):
            batch=rows[start:start+2];ids,mask=pad([r["prompt_ids"]for r in batch],tokenizer.pad_token_id,device)
            residual_in.clear();norm_input.clear();pos_emb.clear();o_input.clear();attn_out.clear();layer_out.clear()
            with torch.inference_mode():
                out=model.model(input_ids=ids,attention_mask=mask,use_cache=False,output_attentions=True,return_dict=True)
                emb=model.model.embed_tokens(ids)
            for bi,row in enumerate(batch):
                ri=start+bi;seq=len(row["prompt_ids"]);boundary=seq-1
                arrays["embedding"][ri,:seq]=emb[bi,:seq].float().cpu().numpy().astype(np.float16)
                arrays["hidden"][ri,0,:seq]=residual_in[20][bi,:seq].float().cpu().numpy().astype(np.float16)
                for lli,li in enumerate(LATE):
                    arrays["hidden"][ri,lli+1,:seq]=layer_out[li][bi,:seq].float().cpu().numpy().astype(np.float16)
                    sa=layers[li].self_attn;x=norm_input[li];shape=(*x.shape[:-1],-1,hd)
                    with torch.inference_mode():
                        q=sa.q_norm(sa.q_proj(x).view(shape)).transpose(1,2)
                        k=sa.k_norm(sa.k_proj(x).view(shape)).transpose(1,2)
                        v=sa.v_proj(x).view(shape).transpose(1,2)
                        cos,sin=pos_emb[li];q,k=apply_rotary_pos_emb(q,k,cos,sin)
                    qr_native=q[bi,:,boundary];kr_native=k[bi,:,:seq];vr_native=v[bi,:,:seq]
                    repk_native=kr_native.repeat_interleave(nh//nkv,dim=0)
                    logits_native=torch.einsum("hd,hpd->hp",qr_native,repk_native)*float(getattr(sa,"scaling",hd**-0.5))
                    calc=torch.softmax(logits_native,dim=-1,dtype=torch.float32).to(q.dtype)
                    qr=qr_native.float();kr=kr_native.float();vr=vr_native.float();repv=vr.repeat_interleave(nh//nkv,dim=0)
                    logits=logits_native.float();line=out.attentions[li][bi,:,boundary,:seq].float()
                    max_qk_error=max(max_qk_error,float(torch.max(torch.abs(calc-line))))
                    weighted=line[:,:,None]*repv
                    arrays["query"][ri,lli]=qr.cpu().numpy().astype(np.float16)
                    arrays["key"][ri,lli,:,:seq]=kr.cpu().numpy().astype(np.float16)
                    arrays["value"][ri,lli,:,:seq]=vr.cpu().numpy().astype(np.float16)
                    arrays["qk_logit"][ri,lli,:,:seq]=logits.cpu().numpy().astype(np.float16)
                    arrays["attention"][ri,lli,:,:seq]=line.cpu().numpy().astype(np.float16)
                    arrays["weighted_value"][ri,lli,:,:seq]=weighted.cpu().numpy().astype(np.float16)
                    groups=[]
                    for name in REGIONS:
                        pos=row["regions"].get(name,[])
                        groups.append(weighted[:,pos].sum(dim=1))
                    region=torch.stack(groups,dim=1)
                    arrays["region_head"][ri,lli]=region.cpu().numpy().astype(np.float16)
                    projected=torch.stack([F.linear(region[:,r].reshape(-1),sa.o_proj.weight.float(),None)for r in range(nr)])
                    arrays["region_residual"][ri,lli]=projected.detach().cpu().numpy().astype(np.float16)
                    max_pre_error=max(max_pre_error,rel_rms(region.sum(1).reshape(-1),o_input[li][bi,boundary]))
                    expected=projected.sum(0)+(sa.o_proj.bias.float() if sa.o_proj.bias is not None else 0)
                    max_res_error=max(max_res_error,rel_rms(expected,attn_out[li][bi,boundary]))
                    route_sum["head_rms"][lli]+=torch.sqrt(torch.mean(o_input[li][bi,boundary].float().view(nh,hd)**2,dim=-1)).cpu().numpy()
                    external=sorted({p for name in REGIONS[:-1] for p in row["regions"].get(name,[])})
                    route_sum["external_mass"][lli]+=line[:,external].sum(-1).cpu().numpy()
                    route_sum["entropy"][lli]+=(-(line*torch.log(line+1e-12)).sum(-1)).cpu().numpy()
                index_rows.append({"field_row":ri,"case_id":row["case_id"],"unit":row["unit"],"family_id":row["family_id"],
                                   "family":row["family"],"language":row["language"],"meaning_swap":row["meaning_swap"],
                                   "query_property":row["query_property"],"prompt_length":seq,"answer_boundary_token":boundary,
                                   "regions":row["regions"],"prompt_ids":row["prompt_ids"]})
                count+=1
            if (start+len(batch))%32==0:
                for x in arrays.values():x.flush()
                print(f"[phase2539] {start+len(batch)}/{n}",flush=True)
    finally:
        for h in handles:h.remove()
        for x in arrays.values():x.flush()
    del arrays
    wo_norm=np.zeros((16,nh),np.float64)
    for lli,li in enumerate(LATE):
        w=layers[li].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for h in range(nh):wo_norm[lli,h]=np.linalg.norm(w[:,h*hd:(h+1)*hd])
    features={k:v/count for k,v in route_sum.items()};features["wo_norm"]=wo_norm
    meta={"model":{"layers":len(layers),"late_layers":list(LATE),"hidden_size":dim,"attention_heads":nh,"kv_heads":nkv,"head_dim":hd,"max_sequence":max_seq},
          "conservation":{"maximum_qk_softmax_absolute_error":max_qk_error,"maximum_head_sum_relative_rms":max_pre_error,"maximum_wo_sum_relative_rms":max_res_error},
          "fields":{k:{"path":str(p),"shape":list(np.load(p,mmap_mode="r").shape),"dtype":"float16","bytes":p.stat().st_size,"sha256":sha(p)}for k,p in paths.items()}}
    return index_rows,meta,features


def select_routes(index_rows:list[dict],features:dict,nh:int)->dict:
    arr=np.load(OUT/"fields/region_head_weighted_value.float16.npy",mmap_mode="r")
    idx={(r["unit"],r["family_id"],r["language"],r["meaning_swap"],r["query_property"]):r["field_row"]for r in index_rows}
    families=sorted({r["family_id"]for r in index_rows});inter=[]
    for unit in (34,35):
        u=[]
        for fi in families:
            langs=[]
            for lang in ("en","zh"):
                cells={(m,q):np.asarray(arr[idx[(unit,fi,lang,m,q)]],np.float32)for m in (0,1)for q in(0,1)}
                langs.append((cells[(0,0)]-cells[(0,1)]-cells[(1,0)]+cells[(1,1)])/4)
            u.append(langs)
        inter.append(u)
    inter=np.asarray(inter,np.float32);ipath=OUT/"derived/region_head_walsh.float16.npy";ipath.parent.mkdir(parents=True,exist_ok=True);np.save(ipath,inter.astype(np.float16))
    energy=np.square(inter[0][...,list(EXTERNAL_REGIONS),:]).sum(axis=(0,1,4,5));pairs=[]
    flat_features=[]
    for lli,li in enumerate(LATE):
        for h in range(nh):
            f={k:float(v[lli,h])for k,v in features.items()};pairs.append((li,h,float(energy[lli,h]),f));flat_features.append([f["head_rms"],f["external_mass"],f["entropy"],f["wo_norm"]])
    pairs.sort(key=lambda x:(-x[2],x[0],x[1]));top=pairs[:32];topkeys={(x[0],x[1])for x in top};pool=[x for x in pairs if(x[0],x[1])not in topkeys]
    allfeat=np.asarray(flat_features);mu=allfeat.mean(0);sd=allfeat.std(0)+1e-9
    rng=np.random.default_rng(2539);matched=[]
    for rep in range(5):
        used=set();chosen=[]
        for li,h,_e,f in top:
            candidates=[x for x in pool if x[0]==li and(x[0],x[1])not in used]
            if not candidates:candidates=[x for x in pool if(x[0],x[1])not in used]
            target=(np.asarray([f[k]for k in("head_rms","external_mass","entropy","wo_norm")])-mu)/sd
            scored=[]
            for x in candidates:
                z=(np.asarray([x[3][k]for k in("head_rms","external_mass","entropy","wo_norm")])-mu)/sd
                scored.append((float(np.sum((z-target)**2))+float(rng.uniform(0,0.02)),x))
            x=min(scored,key=lambda z:z[0])[1];used.add((x[0],x[1]));chosen.append({"layer":x[0],"head":x[1],"features":x[3]})
        matched.append(chosen)
    energy35=np.square(inter[1][...,list(EXTERNAL_REGIONS),:]).sum(axis=(0,1,4,5));f34=energy.reshape(-1);f35=energy35.reshape(-1)
    ranks34=np.argsort(np.argsort(f34));ranks35=np.argsort(np.argsort(f35));overlap=len(set(np.argsort(-f34)[:32])&set(np.argsort(-f35)[:32]))
    table=[{"layer":li,"head":h,"selection_energy":e,**f}for li,h,e,f in pairs];tpath=OUT/"analysis/all_route_features.jsonl";write(tpath,table)
    return {"selection_unit":34,"all_routes":len(pairs),"top":[{"layer":x[0],"head":x[1],"energy":x[2],"features":x[3]}for x in top],
            "matched_random_sets":matched,"unit34_unit35_spearman":float(np.corrcoef(ranks34,ranks35)[0,1]),"top32_overlap":overlap,
            "interaction":{"path":str(ipath),"shape":list(inter.shape),"bytes":ipath.stat().st_size,"sha256":sha(ipath)},
            "feature_table":{"path":str(tpath),"bytes":tpath.stat().st_size,"sha256":sha(tpath)}}


def append_memo(r:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");compact={k:{q:v[q]for q in("path","shape","bytes","sha256")}for k,v in r["fields"].items()}
    text=rf"""


## Phase {PHASE}: 全token Q/K/V地址—内容—写入物理账本（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化CUDA上，对Phase2538全部32个双unit合格族的surface0四格共512条token-atomic提示，覆盖layer20–35全部512条query-head路线、全部8个KV heads、全部source token和全部128/2560物理坐标。保存q20–q36全token residual、答案位置post-RoPE Q、全token post-RoPE K与V、QK logit、softmax质量、每条answer→source边的128维加权V，以及九个严格区域经真实$W_O$写入的2560维residual。top32只由unit34完整坐标Walsh能量冻结供后续干预，另构造五组同层位并按head RMS、source mass、entropy和$W_O$范数最近匹配的对照。

$$s_{{lhaj}}=q_{{lah}}^\top k_{{ljg}}/\sqrt d,\quad \alpha_{{lhaj}}=\operatorname{{softmax}}_j(s_{{lhaj}}),\quad e_{{lhaj}}=\alpha_{{lhaj}}v_{{ljg}},$$
$$g_{{lr}}=W_{{O,l}}\operatorname{{concat}}_h\sum_{{j\in S_r}}e_{{lhaj}}.$$

**结果汇总。** 范围 `{json.dumps(r['scope'],ensure_ascii=False)}`；数值守恒 `{json.dumps(r['conservation'],ensure_ascii=False)}`；冻结路线 `{json.dumps({k:r['routes'][k] for k in ('selection_unit','all_routes','unit34_unit35_spearman','top32_overlap')},ensure_ascii=False)}`；字段 `{json.dumps(compact,ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2539_c121601_c125696_full_token_qkv_edge_ledger.py`；全场、逐token索引、全部route特征、五组匹配对照、Walsh交互和final位于`{OUT}`。

**分析与理论进展。** 该账本第一次把“Q/K地址匹配”和“V内容搬运”在观测量上分开，并保留每条source边的低值128维内容；QK重算、softmax、head求和和$W_O$分区写入均有独立守恒门。它仍只证明数值分解可核算，不证明Q表达查询、K表达角色或V表达语义；这些职责必须由下一Phase的分离干预裁决。五组匹配对照修正旧random在层位和一般物理强度上的混杂。

**问题硬伤与结论。** GQA中四个query heads共享一个KV head，K/V干预天然影响一个query-head组；float16落盘有舍入；结构化字段标签可能放大地址规则；Walsh只消去合同中的一阶项；匹配只基于四个可测协变量且不能保证交换性。全坐标账本是基础设施，不称语言编码机制闭合。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)


def main()->None:
    prior=load(P2538/"analysis/final.json");qualified=set(prior["behavior"]["qualified_family_ids"])
    material=[r for r in read(P2538/"material/token_atomic_rows.jsonl")if r["family_id"]in qualified and r["surface"]==0]
    model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
    try:index,meta,features=collect(model,tokenizer,material);routes=select_routes(index,features,int(model.config.num_attention_heads))
    finally:model_utils.release_model(model);gc.collect()
    ipath=OUT/"material/field_rows.jsonl";write(ipath,index);rpath=OUT/"analysis/frozen_routes.json";save(rpath,routes)
    fields=meta["fields"]|{"index":{"path":str(ipath),"shape":[len(index)],"bytes":ipath.stat().st_size,"sha256":sha(ipath)}}
    checks={"source_passed":prior["all_checks_passed"],"rows_512":len(index)==512,"all_512_routes":routes["all_routes"]==512,
            "full_token_hidden":fields["hidden"]["shape"][:3]==[512,17,meta["model"]["max_sequence"]],
            "full_qkv_coordinates":fields["query"]["shape"][-1]==128 and fields["key"]["shape"][-1]==128 and fields["value"]["shape"][-1]==128,
            "qk_softmax_conservation":meta["conservation"]["maximum_qk_softmax_absolute_error"]<.02,
            "head_conservation":meta["conservation"]["maximum_head_sum_relative_rms"]<.01,
            "wo_conservation":meta["conservation"]["maximum_wo_sum_relative_rms"]<.01,
            "five_matched_sets":len(routes["matched_random_sets"])==5 and all(len(x)==32 for x in routes["matched_random_sets"]),
            "all_hashed":all(len(v["sha256"])==64 for v in fields.values()),"claim_boundary":True}
    result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","scope":{"rows":len(index),"families":len(qualified),**meta["model"]},
            "conservation":meta["conservation"],"routes":routes,"fields":fields,"checks":checks,"all_checks_passed":all(checks.values())}
    save(OUT/"analysis/final.json",result)
    if result["all_checks_passed"]:append_memo(result)
    print(json.dumps({"phase":PHASE,"scope":result["scope"],"conservation":result["conservation"],"routes":{k:routes[k]for k in("all_routes","unit34_unit35_spearman","top32_overlap")},"field_bytes":{k:v["bytes"]for k,v in fields.items()},"checks":checks,"all_checks_passed":result["all_checks_passed"]},ensure_ascii=False,indent=2))
    if not result["all_checks_passed"]:raise RuntimeError(checks)


if __name__=="__main__":main()
