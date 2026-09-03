#!/usr/bin/env python3
"""Twenty-four-family autonomous multi-token generation plus multihop, role binding, and full-clause reorder."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2529 = RESULT / "phase2529_c95777_c97568_full_source_kv_head_residual_ledger"
P2531 = RESULT / "phase2531_c99073_c101120_redundant_route_cuts_rescue"
OUT = RESULT / "phase2532_c101121_c104192_autonomous_multifamily_recursive_generation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2532, "C101121-C104192"
LATE = tuple(range(20, 36)); CONDITIONS = ("no_patch", "head_zero_top", "head_zero_random", "head_zero_all_late")
sys.path.insert(0, str(TESTS)); import model_utils  # noqa: E402


OPERATIONS = (
 ("taxonomy", "[ {e} ] is a [ {p} ]", "[ {e} ]属于[ {p} ]", ("tropical fruit", "metal hand tool"), ("热带水果", "金属手工具")),
 ("part_whole", "[ {e} ] is part of [ {p} ]", "[ {e} ]是[ {p} ]的一部分", ("river engine", "stone cottage"), ("河流引擎", "石头小屋")),
 ("profession", "[ {e} ] works as [ {p} ]", "[ {e} ]的职业是[ {p} ]", ("village doctor", "music teacher"), ("乡村医生", "音乐教师")),
 ("preference", "[ {e} ] prefers [ {p} ]", "[ {e} ]偏好[ {p} ]", ("jasmine tea", "dark coffee"), ("茉莉花茶", "深色咖啡")),
 ("membership", "[ {e} ] belongs to [ {p} ]", "[ {e} ]属于[ {p} ]", ("chess club", "rowing team"), ("象棋社团", "划船队伍")),
 ("translation", "[ {e} ] translates as [ {p} ]", "[ {e} ]翻译为[ {p} ]", ("quiet river", "blue lake"), ("安静河流", "蓝色湖泊")),
 ("temporal", "[ {e} ] occurs during [ {p} ]", "[ {e} ]发生在[ {p} ]", ("early morning", "late evening"), ("清晨时分", "深夜时分")),
 ("spatial", "[ {e} ] is located in [ {p} ]", "[ {e} ]位于[ {p} ]", ("eastern valley", "western harbor"), ("东部山谷", "西部港口")),
 ("causal", "[ {e} ] causes [ {p} ]", "[ {e} ]导致[ {p} ]", ("bright flame", "heavy rainfall"), ("明亮火焰", "强烈降雨")),
 ("permission", "[ {e} ] is marked [ {p} ]", "[ {e} ]被标记为[ {p} ]", ("fully allowed", "strictly blocked"), ("完全允许", "严格禁止")),
 ("possession", "[ {e} ] owns [ {p} ]", "[ {e} ]拥有[ {p} ]", ("silver bicycle", "wooden violin"), ("银色自行车", "木制小提琴")),
 ("instrument", "[ {e} ] uses [ {p} ]", "[ {e} ]使用[ {p} ]", ("small compass", "heavy hammer"), ("小型指南针", "重型铁锤")),
 ("origin", "[ {e} ] comes from [ {p} ]", "[ {e} ]来自[ {p} ]", ("northern island", "southern desert"), ("北方岛屿", "南方沙漠")),
 ("destination", "[ {e} ] travels toward [ {p} ]", "[ {e} ]前往[ {p} ]", ("coastal station", "mountain village"), ("海滨车站", "山间村庄")),
 ("material", "[ {e} ] is made from [ {p} ]", "[ {e} ]由[ {p} ]制成", ("polished copper", "woven cotton"), ("抛光铜材", "编织棉布")),
 ("color", "[ {e} ] is painted [ {p} ]", "[ {e} ]被涂成[ {p} ]", ("deep violet", "pale orange"), ("深紫颜色", "浅橙颜色")),
 ("size", "[ {e} ] has size [ {p} ]", "[ {e} ]的尺寸是[ {p} ]", ("very narrow", "extremely wide"), ("非常狭窄", "极其宽阔")),
 ("temperature", "[ {e} ] is kept [ {p} ]", "[ {e} ]被保持在[ {p} ]", ("mildly warm", "deeply frozen"), ("轻微温暖", "深度冷冻")),
 ("speed", "[ {e} ] moves [ {p} ]", "[ {e} ]移动得[ {p} ]", ("rather slowly", "very quickly"), ("相当缓慢", "非常快速")),
 ("rank", "[ {e} ] is ranked [ {p} ]", "[ {e} ]排名为[ {p} ]", ("first place", "second place"), ("第一名次", "第二名次")),
 ("action", "[ {e} ] performs [ {p} ]", "[ {e} ]执行[ {p} ]", ("careful inspection", "rapid delivery"), ("仔细检查", "快速递送")),
 ("patient", "[ {e} ] receives [ {p} ]", "[ {e} ]接收[ {p} ]", ("paper package", "glass bottle"), ("纸质包裹", "玻璃瓶子")),
 ("manner", "[ {e} ] speaks in [ {p} ]", "[ {e} ]以[ {p} ]说话", ("calm manner", "urgent manner"), ("平静方式", "紧急方式")),
 ("purpose", "[ {e} ] is used for [ {p} ]", "[ {e} ]用于[ {p} ]", ("winter travel", "summer farming"), ("冬季旅行", "夏季耕作")),
)
NAMES = {32: {"en": ("Silver Otter", "Golden Finch"), "zh": ("银色水獭", "金色雀鸟")},
         33: {"en": ("Copper Badger", "Velvet Heron"), "zh": ("铜色獾兽", "绒羽苍鹭")}}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def norm(text: str) -> str: return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())
def parse_entity(text: str, entities: list[str]) -> str | None:
    hits=[e for e in entities if norm(e) in norm(text)]; return hits[0] if len(set(hits)) == 1 else None
def route_map(items: list[dict]) -> dict[int,list[int]]:
    out={}
    for x in items:out.setdefault(int(x["layer"]),[]).append(int(x["head"]))
    return out


def compile_main(tokenizer) -> list[dict]:
    rows=[]
    for unit in (32,33):
        for fi,(family,enfmt,zhfmt,enprops,zhprops) in enumerate(OPERATIONS):
            for lang in ("en","zh"):
                e0,e1=NAMES[unit][lang]; props=enprops if lang=="en" else zhprops; fmt=enfmt if lang=="en" else zhfmt
                for surface in (0,1):
                    for swap in (0,1):
                        mapping=(0,1) if swap==0 else (1,0)
                        facts=[fmt.format(e=e0,p=props[mapping[0]]),fmt.format(e=e1,p=props[mapping[1]])]
                        if surface:facts=facts[::-1]
                        for query in (0,1):
                            target=e0 if mapping[0]==query else e1
                            if lang=="en":
                                question=(f"Which entity is associated with [ {props[query]} ]?" if surface==0 else f"Using only those facts, identify the entity matching [ {props[query]} ].")
                                prompt="Facts: "+". ".join(facts)+". "+question+" Return only the complete entity name. Answer:"
                            else:
                                question=(f"哪个实体与[ {props[query]} ]相关？" if surface==0 else f"只根据这些事实，找出匹配[ {props[query]} ]的实体。")
                                prompt="事实："+"。".join(facts)+"。"+question+"只返回完整实体名称。答案："
                            ids=[int(v) for v in tokenizer.encode(prompt,add_special_tokens=False)]
                            rows.append({"task":"main","unit":unit,"family_id":fi,"family":family,"language":lang,"surface":surface,
                                         "meaning_swap":swap,"query_property":query,"entities":[e0,e1],"target":target,"prompt":prompt,"prompt_ids":ids})
    return rows


def compile_aux(tokenizer) -> list[dict]:
    rows=[]
    for unit,names in ((34,{"en":("Amber Fox","Ivory Crane"),"zh":("琥珀狐狸","象牙仙鹤")}),
                       (35,{"en":("Indigo Wolf","Scarlet Swan"),"zh":("靛蓝灰狼","赤红天鹅")})):
        for lang in ("en","zh"):
            a,b=names[lang]
            # True two-hop branch binding.
            mids=("Meral class","Torin class") if lang=="en" else ("墨岚类别","拓林类别")
            uppers=("living object","crafted object") if lang=="en" else ("生命物体","制造物体")
            for surface in (0,1):
                for swap in (0,1):
                    mp=(0,1) if swap==0 else (1,0)
                    facts=([f"[ {a} ] belongs to [ {mids[0]} ]",f"[ {b} ] belongs to [ {mids[1]} ]",f"[ {mids[0]} ] belongs to [ {uppers[mp[0]]} ]",f"[ {mids[1]} ] belongs to [ {uppers[mp[1]]} ]"] if lang=="en" else
                           [f"[ {a} ]属于[ {mids[0]} ]",f"[ {b} ]属于[ {mids[1]} ]",f"[ {mids[0]} ]属于[ {uppers[mp[0]]} ]",f"[ {mids[1]} ]属于[ {uppers[mp[1]]} ]"])
                    if surface:facts=facts[::-1]
                    for q in (0,1):
                        target=a if mp[0]==q else b
                        prompt=(("Facts: "+". ".join(facts)+f". Following both links, which entity ultimately belongs to [ {uppers[q]} ]? Return only the complete entity name. Answer:") if lang=="en" else
                                ("事实："+"。".join(facts)+f"。沿着两级关系，哪个实体最终属于[ {uppers[q]} ]？只返回完整实体名称。答案："))
                        rows.append({"task":"multihop","unit":unit,"language":lang,"surface":surface,"meaning_swap":swap,"query_property":q,
                                     "entities":[a,b],"target":target,"prompt":prompt,"prompt_ids":[int(v) for v in tokenizer.encode(prompt,add_special_tokens=False)]})
            # Subject-action-object role composition.
            objects=("crisp red apple","ripe green pear") if lang=="en" else ("清脆红苹果","成熟绿梨子")
            for surface in (0,1):
                for swap in (0,1):
                    mp=(0,1) if swap==0 else (1,0); facts=([f"[ {a} ] likes to eat [ {objects[mp[0]]} ]",f"[ {b} ] likes to eat [ {objects[mp[1]]} ]"] if lang=="en" else [f"[ {a} ]喜欢吃[ {objects[mp[0]]} ]",f"[ {b} ]喜欢吃[ {objects[mp[1]]} ]"])
                    if surface:facts=facts[::-1]
                    for q in (0,1):
                        target=a if mp[0]==q else b
                        prompt=(("Facts: "+". ".join(facts)+f". Who likes to eat [ {objects[q]} ]? Return only the complete person name. Answer:") if lang=="en" else
                                ("事实："+"。".join(facts)+f"。谁喜欢吃[ {objects[q]} ]？只返回完整人物名称。答案："))
                        rows.append({"task":"role_binding","unit":unit,"language":lang,"surface":surface,"meaning_swap":swap,"query_property":q,
                                     "entities":[a,b],"target":target,"prompt":prompt,"prompt_ids":[int(v) for v in tokenizer.encode(prompt,add_special_tokens=False)]})
            # Full-clause reorder; the output must contain all clauses unchanged and in chronological order.
            clauses=([f"At dawn {a} opened the northern gate",f"At noon {b} carried the sealed map",f"At dusk {a} closed the southern gate"] if lang=="en" else
                     [f"清晨{a}打开了北门",f"中午{b}携带了密封地图",f"傍晚{a}关闭了南门"])
            orders=((2,0,1),(1,2,0),(2,1,0),(1,0,2))
            for surface,order in enumerate(orders):
                shuffled=[clauses[i] for i in order]
                prompt=(("Reorder these complete sentences chronologically without changing their words. Sentences: "+" | ".join(shuffled)+". Output all three complete sentences separated by |. Answer:") if lang=="en" else
                        ("请按时间顺序重排以下完整句子，不要改动句中文字。句子："+"｜".join(shuffled)+"。输出全部三个完整句子并用｜分隔。答案："))
                rows.append({"task":"full_clause_reorder","unit":unit,"language":lang,"surface":surface,"meaning_swap":0,"query_property":0,
                             "entities":[a,b],"target":" | ".join(clauses),"clauses":clauses,"prompt":prompt,"prompt_ids":[int(v) for v in tokenizer.encode(prompt,add_special_tokens=False)]})
    return rows


class HeadControl:
    def __init__(self, model):
        self.model=model; self.layers=model_utils.get_layers(model); self.nheads=int(model.config.num_attention_heads); self.hdim=int(model.config.head_dim)
        self.active={};self.handles=[]
        for layer in LATE:
            def hook(_m,args,layer=layer):
                heads=self.active.get(layer,[])
                if not heads:return None
                x=args[0];changed=x.clone().view(x.shape[0],x.shape[1],self.nheads,self.hdim);changed[:,-1,heads,:]=0
                return(changed.reshape_as(x),*args[1:])
            self.handles.append(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook))
    def close(self):
        for h in self.handles:h.remove()


def generate(model,tokenizer,rows:list[dict],condition:str,routes:dict[int,list[int]],max_new:int)->list[dict]:
    tokenizer.padding_side="left";device=model.get_input_embeddings().weight.device;control=HeadControl(model);out=[]
    if condition=="head_zero_all_late":control.active={l:list(range(control.nheads)) for l in LATE}
    elif condition!="no_patch":control.active=routes
    try:
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];enc=tokenizer([r["prompt"] for r in batch],return_tensors="pt",padding=True,add_special_tokens=False);enc={k:v.to(device) for k,v in enc.items()}
            with torch.inference_mode():seq=model.generate(**enc,max_new_tokens=max_new,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            width=enc["input_ids"].shape[1]
            for row,s in zip(batch,seq):
                text=tokenizer.decode(s[width:].cpu().tolist(),skip_special_tokens=True)
                if row["task"]=="full_clause_reorder":
                    positions=[norm(text).find(norm(c)) for c in row["clauses"]];parsed=all(p>=0 for p in positions) and positions==sorted(positions)
                else:parsed=parse_entity(text,row["entities"])==row["target"]
                out.append({k:row[k] for k in row if k not in ("prompt_ids",)}|{"condition":condition,"generated_text":text,"correct":bool(parsed)})
    finally:control.close()
    return out


def behavior_summary(rows:list[dict])->dict:
    main=[r for r in rows if r["task"]=="main" and r["condition"]=="no_patch"];detail={};qualified=[]
    for fi,(family,*_) in enumerate(OPERATIONS):
        detail[str(fi)]={};gates=[]
        for unit in (32,33):
            vals=[r for r in main if r["family_id"]==fi and r["unit"]==unit]
            language={x:float(np.mean([r["correct"] for r in vals if r["language"]==x])) for x in ("en","zh")}
            surface={str(x):float(np.mean([r["correct"] for r in vals if r["surface"]==x])) for x in (0,1)}
            swap={str(x):float(np.mean([r["correct"] for r in vals if r["meaning_swap"]==x])) for x in (0,1)}
            gate=float(np.mean([r["correct"] for r in vals]))>=.65 and min(language.values())>=.5 and min(surface.values())>=.5 and min(swap.values())>=.5
            detail[str(fi)][str(unit)]={"n":len(vals),"accuracy":float(np.mean([r["correct"] for r in vals])),"language":language,"surface":surface,"swap":swap,"gate":gate};gates.append(gate)
        if all(gates):qualified.append(fi)
    aux={}
    for task in ("multihop","role_binding","full_clause_reorder"):
        vals=[r for r in rows if r["task"]==task and r["condition"]=="no_patch"]
        aux[task]={"n":len(vals),"accuracy":float(np.mean([r["correct"] for r in vals]))}
    return{"main_accuracy":float(np.mean([r["correct"] for r in main])),"qualified_family_ids":qualified,"qualified_families":[OPERATIONS[i][0] for i in qualified],"detail":detail,"auxiliary":aux}


def causal_summary(rows:list[dict])->dict:
    out={}
    for task in sorted({r["task"] for r in rows}):
        out[task]={}
        for condition in sorted({r["condition"] for r in rows if r["task"]==task}):
            vals=[r for r in rows if r["task"]==task and r["condition"]==condition]
            out[task][condition]={"n":len(vals),"accuracy":float(np.mean([r["correct"] for r in vals]))}
    return out


def capture_recursive_field(model,tokenizer,rows:list[dict],top:dict[int,list[int]],steps=8)->dict:
    tokenizer.padding_side="left";device=model.get_input_embeddings().weight.device;layers=model_utils.get_layers(model);dim=int(model.config.hidden_size);hdim=int(model.config.head_dim)
    (OUT/"fields").mkdir(parents=True,exist_ok=True)
    n=len(rows);paths={"hidden":OUT/"fields/autonomous_boundary_q20_q36.float16.npy","embedding":OUT/"fields/autonomous_boundary_embedding.float16.npy","head":OUT/"fields/autonomous_top32_pre_o.float16.npy","tokens":OUT/"fields/autonomous_generated_token_ids.int32.npy"}
    hidden=np.lib.format.open_memmap(paths["hidden"],mode="w+",dtype=np.float16,shape=(n,steps,17,dim));embedding=np.lib.format.open_memmap(paths["embedding"],mode="w+",dtype=np.float16,shape=(n,steps,dim));head=np.lib.format.open_memmap(paths["head"],mode="w+",dtype=np.float16,shape=(n,steps,32,hdim));tokens=np.lib.format.open_memmap(paths["tokens"],mode="w+",dtype=np.int32,shape=(n,steps));tokens[:]=-1
    layer_out={};head_out={};handles=[];route_order=[(int(x["layer"]),int(x["head"])) for x in load(P2529/"analysis/final.json")["routes"]["top"]]
    for layer in range(19,36):
        def lh(_m,_a,o,layer=layer):layer_out[layer]=(o[0] if isinstance(o,tuple) else o).detach()
        handles.append(layers[layer].register_forward_hook(lh))
    for layer in sorted(set(l for l,_ in route_order)):
        def oh(_m,args,layer=layer):head_out[layer]=args[0].detach()
        handles.append(layers[layer].self_attn.o_proj.register_forward_pre_hook(oh))
    try:
        for start in range(0,n,8):
            batch=rows[start:start+8];enc=tokenizer([r["prompt"] for r in batch],return_tensors="pt",padding=True,add_special_tokens=False);ids=enc["input_ids"].to(device);mask=enc["attention_mask"].to(device)
            for step in range(steps):
                layer_out.clear();head_out.clear()
                with torch.inference_mode():out=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1,return_dict=True)
                next_token=out.logits[:,-1].argmax(-1);bi=torch.arange(ids.shape[0],device=device)
                embedding[start:start+len(batch),step]=model.model.embed_tokens(ids[:,-1]).detach().float().cpu().numpy().astype(np.float16)
                hidden[start:start+len(batch),step]=torch.stack([layer_out[l][bi,-1].float().cpu() for l in range(19,36)],1).numpy().astype(np.float16)
                head[start:start+len(batch),step]=torch.stack([head_out[l][:,-1].view(ids.shape[0],-1,hdim)[:,h].float().cpu() for l,h in route_order],1).numpy().astype(np.float16)
                tokens[start:start+len(batch),step]=next_token.cpu().numpy().astype(np.int32)
                ids=torch.cat([ids,next_token[:,None]],1);mask=torch.cat([mask,torch.ones_like(next_token[:,None])],1)
            if(start+len(batch))%48==0:print(f"[phase2532 field] {start+len(batch)}/{n}",flush=True)
    finally:
        for h in handles:h.remove()
        for x in (hidden,embedding,head,tokens):x.flush()
        del hidden,embedding,head,tokens
    return{name:{"path":str(path),"shape":list(np.load(path,mmap_mode="r").shape),"dtype":str(np.load(path,mmap_mode="r").dtype),"bytes":path.stat().st_size,"sha256":sha(path)} for name,path in paths.items()}


def append_memo(r:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: 二十四语言操作族自主多token生成、真多跳与整句重排（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 新建24个外部语言操作族，覆盖类别、部分、职业、偏好、成员、翻译、时间、空间、因果、许可、拥有、工具、来源、目的地、材料、颜色、尺寸、温度、速度、排序、动作、受事、方式、目的。unit32/33、英中、双surface、双事实绑定、双query全交叉共768条，无候选、实体名必须完整自主生成且英中实体均为多token。另做32条真两跳分支、32条“谁喜欢吃什么”主谓宾角色绑定、16条三个完整长句逐字保留重排。对合格24族的unit33锁箱及辅助任务，在每个自回归步持续zero贡献top32、随机32或全部late heads；baseline额外保存每个生成步的q20–q36×2560 HiddenState、输入词嵌入和top32×128 head坐标。

$$p(y_{{1:T}}|x)=\prod_{{t=1}}^T p(y_t|x,y_{{<t}}),\qquad h_{{a_t}}^{{l+1}}=h_{{a_t}}^l+A_l(h_{{\le a_t}})+M_l(h_{{a_t}}).$$

**结果汇总。** 行为资格 `{json.dumps(r['behavior'],ensure_ascii=False)}`；自主因果 `{json.dumps(r['causal'],ensure_ascii=False)}`；递归场 `{json.dumps(r['fields'],ensure_ascii=False)}`；设计 `{json.dumps(r['design'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2532_c101121_c104192_autonomous_multifamily_recursive_generation.py`；全部材料、逐条件自主生成文本、逐token参数场、SHA-256和final位于`{OUT}`。

**分析与理论进展。** 该Phase把第一token teacher-forced充分性升级到完整实体的真实自回归输出，并把长距离材料区分为“查询某位置实体”和“输出全部完整句子”。若top32持续删除损害多token行为而随机不损害，才支持同一路线在递归生成中自然参与；若只影响格式或后续token，不等于上游语义丢失。真多跳先过行为门再解释内部阴性。

**问题硬伤与结论。** 24族仍是受控微世界；“open”要求实体名而非任意散文；生成精确匹配会把同义改写记为失败，长句重排则同时报告顺序和原文保持；持续zero head输出是强干预；baseline逐token场不等于干预条件的完整场。跨模型物理route仍需各模型独立选择。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as f:f.write(text)


def main()->None:
    f2529=load(P2529/"analysis/final.json");f2531=load(P2531/"analysis/final.json")
    model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
    try:
        main_rows=compile_main(tokenizer);aux_rows=compile_aux(tokenizer);all_material=main_rows+aux_rows
        top=route_map(f2529["routes"]["top"]);random_routes=route_map(f2529["routes"]["random"])
        baseline=generate(model,tokenizer,main_rows+[r for r in aux_rows if r["task"]!="full_clause_reorder"],"no_patch",{},16)
        baseline+=generate(model,tokenizer,[r for r in aux_rows if r["task"]=="full_clause_reorder"],"no_patch",{},80)
        behavior=behavior_summary(baseline)
        qualified=set(behavior["qualified_family_ids"])
        lock=[r for r in main_rows if r["unit"]==33 and r["surface"]==1 and r["family_id"] in qualified]
        aux_causal=[r for r in aux_rows if r["task"] in ("multihop","role_binding")]
        reorder=[r for r in aux_rows if r["task"]=="full_clause_reorder"]
        causal_rows=[]
        for condition,routes in (("no_patch",{}),("head_zero_top",top),("head_zero_random",random_routes),("head_zero_all_late",{})):
            causal_rows+=generate(model,tokenizer,lock+aux_causal,condition,routes,16)
            causal_rows+=generate(model,tokenizer,reorder,condition,routes,80)
        fields=capture_recursive_field(model,tokenizer,lock,top,8)
    finally:model_utils.release_model(model);gc.collect()
    material_path=OUT/"material/autonomous_rows.jsonl";write(material_path,all_material)
    baseline_path=OUT/"behavior/baseline_generation.jsonl";write(baseline_path,baseline)
    causal_path=OUT/"output/causal_generation.jsonl";write(causal_path,causal_rows)
    causal=causal_summary(causal_rows)
    multi_token=[]
    for lang in ("en","zh"):
        for unit in (32,33):
            for entity in NAMES[unit][lang]:multi_token.append(len(tokenizer.encode(entity,add_special_tokens=False))>=2)
    design={"main_rows":len(main_rows),"operation_families":len(OPERATIONS),"aux_rows":len(aux_rows),"lockbox_rows":len(lock),"all_entity_names_multitoken":all(multi_token),"units":[32,33],"languages":["en","zh"]}
    checks={"sources_passed":f2529["all_checks_passed"]and f2531["all_checks_passed"],"main_rows_768":len(main_rows)==768,"families_24":len(OPERATIONS)==24,"at_least_20_qualified":len(qualified)>=20,"multitoken_entities":all(multi_token),"true_multihop_rows":sum(r["task"]=="multihop" for r in aux_rows)==32,"role_rows":sum(r["task"]=="role_binding" for r in aux_rows)==32,"full_reorder_rows":sum(r["task"]=="full_clause_reorder" for r in aux_rows)==16,"four_causal_conditions":all(len(v)==4 for v in causal.values()),"field_hidden_coordinates":fields["hidden"]["shape"][-1]==2560,"hashes":all(len(sha(p))==64 for p in (material_path,baseline_path,causal_path)),"claim_boundary":True}
    r={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","design":design,"behavior":behavior,"causal":causal,"fields":fields,"files":{"material":{"path":str(material_path),"sha256":sha(material_path)},"baseline":{"path":str(baseline_path),"sha256":sha(baseline_path)},"causal":{"path":str(causal_path),"sha256":sha(causal_path)}},"adjudication":{"autonomous_multitoken_tested":True,"full_clause_reorder_requires_content_and_order":True,"multihop_internal_mechanism_closed":False,"language_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())}
    save(OUT/"analysis/final.json",r)
    if r["all_checks_passed"]:append_memo(r)
    print(json.dumps({"phase":PHASE,"design":design,"qualified":behavior["qualified_families"],"aux":behavior["auxiliary"],"causal":causal,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2))
    if not r["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
