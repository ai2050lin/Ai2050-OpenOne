"""Local real BF16 joint branch/style study; no production registration.

Full native coordinates at three declared readouts, G/U/A at two readouts.
Related prompt constructions are not a demonstrated training genealogy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from shared.dist_protocol import grammar
from shared.dist_protocol.mechanism_trace import GpuLease, unbits, file_sha
from shared.dist_protocol.natural_register_runner import snapshot
from tools.power_analysis import check_phase
from tools import provenance as PV

MODEL = ROOT/'models/hf/qwen3-4b'
OUT = ROOT/'tests/glm5/result/branch_style_joint_v1'
SEED = 2026110801
FAMILY = 'BRANCH-STYLE-JOINT-v1'
NAMES = [('苹果','apple'),('梨','pear'),('香蕉','banana'),('桃子','peach'),('橙子','orange'),('柠檬','lemon'),('芒果','mango'),('西瓜','watermelon'),('杯子','cup'),('盘子','plate'),('碗','bowl'),('瓶子','bottle'),('盒子','box'),('箱子','chest'),('椅子','chair'),('桌子','table'),('书包','backpack'),('帽子','hat'),('围巾','scarf'),('外套','coat'),('雨伞','umbrella'),('手套','glove'),('气球','balloon'),('玩具','toy'),('汽车','car'),('卡车','truck'),('自行车','bicycle'),('小船','boat'),('火车','train'),('飞机','airplane'),('铅笔','pencil'),('信封','envelope'),('台灯','lamp'),('时钟','clock')]
COLORS = [('蓝色','blue'),('绿色','green'),('红色','red'),('黄色','yellow'),('紫色','purple'),('白色','white'),('黑色','black'),('橙色','orange')]
STYLE = ['自然聊天','严谨论辩']
TASK = ['按资料用中文回答被询问对象的颜色。','仅把被询问对象的颜色描述译成一个英文句子。']
PROV = None

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def identity():
    global PROV
    if PROV is None:
        def git(*args):
            return subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,capture_output=True,text=True,timeout=20).stdout.strip()
        PV._git = git
        PROV = PV.provenance(MODEL,SEED,'branch_style_joint.v1')
    return PROV

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    obj={'provenance':identity(),**obj}
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=1,allow_nan=False)+'\n',encoding='utf-8')
    q.replace(p)

def log(stage,done,total,start,**extra):
    import torch
    elapsed=time.time()-start
    item={'stage':stage,'done':done,'total':total,'seconds':round(elapsed,1),'eta_seconds':round(elapsed/max(done,1)*(total-done),1),**extra}
    if torch.cuda.is_initialized():
        item.update(cuda_allocated_MiB=round(torch.cuda.memory_allocated()/2**20),cuda_reserved_MiB=round(torch.cuda.memory_reserved()/2**20),cuda_peak_MiB=round(torch.cuda.max_memory_allocated()/2**20))
    if os.name=='nt' and (done==0 or done==total or done%64==0):
        try:
            cmd=f"(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object {{ $_.InstanceName -like 'pid_{os.getpid()}_*' }} | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            r=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=12)
            item['shared_gpu_MiB']=round(float(r.stdout.strip())/2**20,2) if r.returncode==0 and r.stdout.strip() else None
            item['shared_gpu_measurement']='Windows PDH' if item['shared_gpu_MiB'] is not None else 'unavailable; not assumed zero'
        except Exception as e: item['shared_gpu_measurement']=type(e).__name__
    line=json.dumps(item,ensure_ascii=False)
    print(line,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n');f.flush()

@contextmanager
def gpu_lease():
    start=time.time();lease=GpuLease()
    while True:
        try:lease.acquire();break
        except RuntimeError:
            if time.time()-start>2700:raise
            log('waiting_for_existing_gpu_lease',0,1,start)
            time.sleep(20)
    try:yield lease
    finally:lease.release()

def build_spec():
    pairs=list(itertools.combinations(range(len(NAMES)),2));random.Random(SEED).shuffle(pairs)
    fillers=[]
    for i,(a,b) in enumerate(pairs[:512]):
        rng=random.Random(SEED+i)
        ci,di=rng.sample(range(len(COLORS)),2)
        if i%2:a,b=b,a
        slots={'X':NAMES[a][0],'Y':NAMES[b][0],'EN':NAMES[a][1],'C':COLORS[ci][0],'D':COLORS[di][0],'EC':COLORS[ci][1],'ED':COLORS[di][1],'TAG':f'R{i:03}'}
        fillers.append({'id':f'pair-{min(a,b):02}-{max(a,b):02}','slots':slots})
    frames=[]
    bodies=['记录{TAG}：{X}是{C}的，{Y}是{D}的。','记录{TAG}：{X}呈现{C}，{Y}呈现{D}。','记录{TAG}：甲号对应{X}，乙号对应{Y}；甲号的颜色为{C}，乙号的颜色为{D}。']
    for branch,style,task in itertools.product(range(3),range(2),range(2)):
        text=(f'请以{STYLE[style]}的口吻表达。只依据记录，不增添事实，不改变立场。'+TASK[task]+'\n'+bodies[branch]+'\n被询问对象：{X}。\n回应：')
        answer='{C}' if task==0 else '{EC}'
        contrast='{D}' if task==0 else '{ED}'
        frames.append({'id':f'b{branch}-s{style}-t{task}','levels':{'branch':str(branch),'style':str(style),'task':str(task)},'text':text,'target':'X','target_occurrence':'last','probe':{'prompt':text,'answer':answer,'answers':[answer],'contrast':[contrast]}})
    return {'schema_version':'template_grammar.v1','family_id':FAMILY,'grammar_version':1,'module':'BRANCH','factors':['branch','style','task'],'layers':'all','readouts':['target','final','probe_answer'],'frames':frames,'fillers':fillers,'splits':{'salt':str(SEED),'fractions':{'discovery':.5,'confirm':.25,'lockbox':.25}},'max_items':10000,'prereg':{'min_behavior_acc':.85,'min_cell_n':8,'min_lockbox_cos':.5}}

def prepare():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    spec=build_spec();assert not grammar.validate(spec),grammar.validate(spec)
    corpus=grammar.generate(spec,include_canary=False)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    slots={x['id']:x['slots'] for x in spec['fillers']}
    rows=[]
    for item in corpus['items']:
        f=slots[item['filler_id']];task=int(item['cond']['task'])
        chat=tok.apply_chat_template([{'role':'user','content':item['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
        body_start=chat.index('记录'+f['TAG'])
        body_end=chat.index('\n被询问对象')-1
        def pos(c):return next(i for i,(a,b) in enumerate(enc.offset_mapping) if a<=c<b)
        candidates=[]
        for col,ec in [(f['C'],f['EC']),(f['D'],f['ED'])]:
            candidates.extend([f'嗯，{f["X"]}是{col}的。',f'依据资料，{f["X"]}的颜色是{col}。'] if task==0 else [f'Well, the {f["EN"]} is {ec}.',f'The {f["EN"]} is {ec}.'])
        rows.append({'id':item['id'],'family':item['filler_id'],'split':item['split'],'branch':int(item['cond']['branch']),'style':int(item['cond']['style']),'task':task,'text':item['text'],'chat_text':chat,'tokens':enc.input_ids,'readouts':[pos(body_start),pos(body_end),len(enc.input_ids)-1],'candidates':candidates,'candidate_tokens':[tok.encode(x,add_special_tokens=False) for x in candidates],'slots':f})
    rows.sort(key=lambda r:(r['family'],r['branch'],r['task'],r['style']))
    disc=sorted({r['family'] for r in rows if r['split']=='discovery'},key=lambda x:digest([SEED,x]))
    pilot=set(disc[:8])
    for r in rows:
        if r['family'] in pilot:r['split']='pilot'
    existing=OUT/'inputs.json'
    data={'schema':'branch_style_inputs.v1','kind':'real_design','grammar_sha256':grammar.spec_sha(spec),'rows':rows,'pilot_families':sorted(pilot),'source':'Author-written fictional records; pair-grouped split; individual vocabulary reused across splits.'}
    if existing.exists():assert read(existing)['rows']==rows,'immutable inputs differ'
    save(OUT/'grammar.json',spec);save(existing,data)
    save(OUT/'grammar_preview.json',grammar.preview(spec,sample=3))
    model_snapshot=snapshot(MODEL)
    plan={'schema':'branch_style_prereg.v1','seed':SEED,'kind':'real','model_snapshot':model_snapshot,'input_sha256':file_sha(existing),'script_sha256':file_sha(__file__),'precision':'bf16','attention':'sdpa','batch':4,'capture':'all 37 hs entries at body-start/body-end/output-prefix; G/U/A all 36 blocks at body-end/output-prefix; full vocabulary final logits. BF16 bits. Other positions absent. hs[-1] post final norm.','disk_cap_GiB':36,'reserve_GiB':12,'max_gpu_hours':3,'behavior_gate':{'content_pooled':.85,'min_cell':.75,'paired_style_per_task':.65,'scoring':'max full logP(correct chat/debate) > max full logP(wrong chat/debate); paired style shift = (LPchat-LPdebate)_chat - same_debate. Separate free generation.'},'split_counts':{s:len({r['family'] for r in rows if r['split']==s}) for s in ['pilot','discovery','confirm','lockbox']},'primary':'confirm family-mean standardized MLP A prediction NMSE: interaction vs additive >=5% reduction and paired bootstrap 97.5% lower difference >0; style-difference prediction correlation >0 on each construction and task; group patch KL improvement vs two matched random groups. No per-unit significance claim.','models':['mean','identity/length ridge','additive content+condition','content-by-condition ridge with low-rank budget controlled by validation','shuffled content-by-condition negative control'],'selection':'Only discovery. Three construction patterns are related templates, not a learned genealogy. Select 32 native units by joint heldout style effect across all branches and both tasks; max 8/block.','power':check_phase(.10,128,p_d=.10,alpha=.025),'continuous_power':{'standardized_paired_effect':.30,'n':128,'alpha_two_sided':.025,'normal_approx_power':None},'causal':'Confirm, one construction chosen deterministically per family, both tasks/directions. Replace final-prefix A with opposite-style same-family donor for selected group; two same-block-count random groups, and wrong-family donor. Compare full vocabulary KL toward donor and correct-vs-wrong full-candidate margins. Conditional patching, not standalone graph prediction.'}
    from scipy.stats import norm
    plan['continuous_power']['normal_approx_power']=float(norm.cdf(.30*np.sqrt(128)-norm.ppf(1-.025/2)))
    assert plan['power']['pass'] and plan['continuous_power']['normal_approx_power']>=.8
    if (OUT/'prereg.json').exists():assert read(OUT/'prereg.json')['model_snapshot']==model_snapshot
    save(OUT/'prereg.json',plan)
    PV.RESULTS=OUT/'provenance_receipts'
    PV.write_result('branch_style_prereg',plan,MODEL,SEED,'branch_style_joint')
    print(json.dumps({'prepared':len(rows),'splits':plan['split_counts'],'snapshot':model_snapshot,'power':plan['power']},ensure_ascii=False),flush=True)

def bits(x):return x.detach().contiguous().view(__import__('torch').uint16).cpu().numpy()

class Engine:
    def __init__(self,capture=False):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(4);torch.manual_seed(SEED)
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.active=False;self.patch=None;self.pos=None;self.data={};self.handles=[]
        for l,block in enumerate(self.model.model.layers):
            if capture:
                for name,module in [('G',block.mlp.gate_proj),('U',block.mlp.up_proj)]:
                    def hook(m,args,value,l=l,name=name):
                        if self.active:self.data[name][l]=value[self.bi,self.pos].detach().clone()
                    self.handles.append(module.register_forward_hook(hook))
            def down(m,args,l=l):
                value=args[0]
                if self.active:self.data['A'][l]=value[self.bi,self.pos].detach().clone()
                if self.patch is not None and self.in_prefill and l in self.patch:
                    units,donor=self.patch[l]
                    value=value.clone();value[:,-1,units]=donor.to(value.device,value.dtype)
                    return (value,)
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(down))

    def run(self,rows,capture=False,score=True):
        import torch
        b=len(rows);length=max(len(r['tokens']) for r in rows)
        ids=torch.full((b,length),self.tok.pad_token_id or self.tok.eos_token_id,dtype=torch.long,device='cuda')
        mask=torch.zeros_like(ids);positions=[]
        for i,r in enumerate(rows):
            n=len(r['tokens']);ids[i,-n:]=torch.tensor(r['tokens'],device='cuda');mask[i,-n:]=1
            positions.append([length-n+p for p in r['readouts']])
        p=torch.tensor(positions,device='cuda');self.bi=torch.arange(b,device='cuda')[:,None];self.pos=p[:,1:]
        self.active=capture;self.data={k:{} for k in ['G','U','A']}
        self.in_prefill=True
        with torch.inference_mode():
            out=self.model(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(-1)-1,use_cache=score,output_hidden_states=capture,logits_to_keep=1)
            self.active=False
            self.in_prefill=False
            logits=out.logits[:,-1].float()
            data={}
            if capture:
                data['hs']=bits(torch.stack([h[self.bi,p] for h in out.hidden_states],dim=1))
                for k in ['G','U','A']:data[k]=bits(torch.stack([self.data[k][l] for l in range(36)],dim=1))
                expected=torch.nn.functional.silu(torch.stack([self.data['G'][l] for l in range(36)],1))*torch.stack([self.data['U'][l] for l in range(36)],1)
                actual=torch.stack([self.data['A'][l] for l in range(36)],1)
                data['gva_max_abs_error']=float((expected.float()-actual.float()).abs().max())
                hidden=out.hidden_states[-1][:,-1]
                # Match the model's [batch, 1, hidden] GEMM shape; a 2-D GEMM
                # has different BF16 rounding despite identical input values.
                rec=self.model.get_output_embeddings()(hidden[:,None])[:,0].float()
                data['logit_max_abs_error']=float((rec-logits).abs().max())
                data['logits']=bits(out.logits[:,-1])
            scores=None
            if score:
                cand=[c for r in rows for c in r['candidate_tokens']]
                ll=max(len(c) for c in cand)
                ci=torch.full((b*4,ll),self.tok.pad_token_id or self.tok.eos_token_id,dtype=torch.long,device='cuda')
                cm=torch.zeros_like(ci)
                for i,c in enumerate(cand):ci[i,:len(c)]=torch.tensor(c,device='cuda');cm[i,:len(c)]=1
                first=logits.log_softmax(-1).repeat_interleave(4,0).gather(1,ci[:,:1]).squeeze(1)
                cache=out.past_key_values;cache.batch_repeat_interleave(4)
                am=torch.cat([mask.repeat_interleave(4,0),cm[:,:-1]],1)
                pos=mask.sum(-1).repeat_interleave(4)[:,None]+torch.arange(ll-1,device='cuda')[None]
                nxt=self.model(input_ids=ci[:,:-1],attention_mask=am,position_ids=pos,past_key_values=cache,use_cache=False).logits.float()
                logp=nxt.log_softmax(-1).gather(2,ci[:,1:,None]).squeeze(-1)
                scores=(first+(logp*cm[:,1:]).sum(-1)).reshape(b,4).cpu().numpy()
        return logits.cpu().numpy(),scores,data

def selected_rows(part):
    rows=read(OUT/'inputs.json')['rows']
    return [r for r in rows if r['split']==part]

def behavior(rows,scores):
    cells={};pairs=defaultdict(dict)
    for i,r in enumerate(rows):
        key=f"b{r['branch']}-s{r['style']}-t{r['task']}"
        cells.setdefault(key,[]).append(bool(max(scores[i,:2])>max(scores[i,2:])))
        pairs[(r['family'],r['branch'],r['task'])][r['style']]=float(scores[i,0]-scores[i,1])
    shifts={str(t):[v[0]-v[1] for (f,b,tt),v in pairs.items() if tt==t and len(v)==2] for t in [0,1]}
    return {'content_accuracy':float(np.mean([max(s[:2])>max(s[2:]) for s in scores])),'cells':{k:{'n':len(v),'accuracy':float(np.mean(v))} for k,v in cells.items()},'style':{t:{'pairs':len(v),'positive_shift_fraction':float(np.mean(np.array(v)>0)),'mean_shift':float(np.mean(v))} for t,v in shifts.items()}}

def collect(part):
    import torch
    rows=selected_rows(part);plan=read(OUT/'prereg.json')
    if part!='pilot':assert read(OUT/'pilot'/'summary.json')['gate_passed'],'pilot behavior failed'
    if part!='pilot':assert read(OUT/'calibration_controls.json')['passed'],'numeric/causal calibration failed'
    if part=='confirm':assert (OUT/'frozen.json').exists(),'freeze method first'
    if part=='lockbox':raise RuntimeError('lockbox remains sealed in this phase')
    folder=OUT/part;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():print('already complete',part);return
    start=time.time();log(part,0,len(rows),start)
    shapes={'hs':(len(rows),37,3,2560),'G':(len(rows),36,2,9728),'U':(len(rows),36,2,9728),'A':(len(rows),36,2,9728),'logits':(len(rows),151936)}
    need=sum(np.prod(s)*2 for s in shapes.values())
    assert shutil.disk_usage(OUT).free>need+12*2**30,'native fields exceed disk reserve'
    state=folder/'state.json';done=read(state)['done'] if state.exists() else 0
    arrays={k:np.lib.format.open_memmap(folder/f'{k}.npy',mode='r+' if done else 'w+',dtype=np.uint16,shape=s) for k,s in shapes.items()}
    scores=np.lib.format.open_memmap(folder/'scores.npy',mode='r+' if done else 'w+',dtype=np.float32,shape=(len(rows),4))
    checks=[]
    with gpu_lease(),torch.inference_mode():
        engine=Engine(capture=True)
        for off in range(done,len(rows),4):
            batch=rows[off:off+4];_,sc,data=engine.run(batch,capture=True)
            for k,a in arrays.items():a[off:off+len(batch)]=data[k]
            scores[off:off+len(batch)]=sc
            checks.append({'gva':data['gva_max_abs_error'],'logits':data['logit_max_abs_error']})
            assert data['gva_max_abs_error']<=.05 and data['logit_max_abs_error']<=.125,checks[-1]
            for a in list(arrays.values())+[scores]:a.flush()
            save(state,{'done':off+len(batch),'model_snapshot':plan['model_snapshot'],'script_sha256':file_sha(__file__)})
            if (off+len(batch))%16==0 or off+len(batch)==len(rows):log(part,off+len(batch),len(rows),start)
    result=behavior(rows,scores)
    gates=plan['behavior_gate']
    passed=result['content_accuracy']>=gates['content_pooled'] and min(c['accuracy'] for c in result['cells'].values())>=gates['min_cell'] and all(s['positive_shift_fraction']>=gates['paired_style_per_task'] for s in result['style'].values())
    save(folder/'summary.json',{'kind':'real','rows':len(rows),'families':len({r['family'] for r in rows}),'behavior':result,'gate_passed':passed,'native_checks':checks,'model_snapshot':plan['model_snapshot'],'input_sha256':plan['input_sha256'],'script_sha256':file_sha(__file__),'seconds':time.time()-start})
    print(json.dumps({'part':part,'gate_passed':passed,'behavior':result},ensure_ascii=False),flush=True)

def controls():
    import torch
    assert snapshot(MODEL)==read(OUT/'prereg.json')['model_snapshot']
    start=time.time();log('calibration_controls',0,1,start)
    with gpu_lease(),torch.inference_mode():
        engine=Engine(True);pilot=selected_rows('pilot');late=[]
        root=pilot[0]['slots']
        body=f'记录：{root["X"]}是{root["C"]}的，{root["Y"]}是{root["D"]}的。'
        for s in range(2):
            user=body+'\n稍后的表达要求：请以'+STYLE[s]+'的口吻回答。\n回应：'
            text=engine.tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=engine.tok(text,add_special_tokens=False,return_offsets_mapping=True)
            char=text.index(body)+len(body)-1
            early=next(i for i,(a,b) in enumerate(enc.offset_mapping) if a<=char<b)
            late.append({'tokens':enc.input_ids,'readouts':[early,early,len(enc.input_ids)-1],'candidate_tokens':pilot[0]['candidate_tokens']})
        # Equal execution shape and identical early token positions. Padding only
        # after both completed prompts cannot influence the earlier causal field.
        width=max(len(r['tokens']) for r in late)
        for r in late:r['tokens']=r['tokens']+[engine.tok.pad_token_id]*(width-len(r['tokens']))
        assert late[0]['tokens'][:late[0]['readouts'][0]+1]==late[1]['tokens'][:late[1]['readouts'][0]+1]
        logits,_,captured=engine.run(late,capture=True,score=False)
        field=unbits(captured['hs']);early_error=float(np.max(abs(field[0,:,0]-field[1,:,0])))
        save(OUT/'causal_calibration_diagnostic.json',{'earlier_prefix_error_by_layer':[float(np.max(abs(field[0,l,0]-field[1,l,0]))) for l in range(37)],'execution_shape':[2,width],'note':'Original check used unequal left padding, mixing numerical execution shape with future-cue comparison. Equal early positions and common shape are the causal control.'})
        assert early_error==0.,'future style cue appeared in earlier causal prefix'
        choices=[]
        for b,s,t in itertools.product(range(3),range(2),range(2)):
            candidates=[r for r in pilot if (r['branch'],r['style'],r['task'])==(b,s,t)]
            choices.extend(candidates[:2])
        generations=[]
        engine.active=False;engine.patch=None
        for off in range(0,len(choices),4):
            batch=choices[off:off+4];width=max(len(r['tokens']) for r in batch)
            ids=torch.full((len(batch),width),engine.tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,r in enumerate(batch):n=len(r['tokens']);ids[i,-n:]=torch.tensor(r['tokens'],device='cuda');mask[i,-n:]=1
            tokens=engine.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=48,do_sample=False,pad_token_id=engine.tok.pad_token_id)
            for r,output in zip(batch,tokens[:,width:]):
                text=engine.tok.decode(output,skip_special_tokens=True)
                expected=r['slots']['C'] if r['task']==0 else r['slots']['EC']
                other=r['slots']['D'] if r['task']==0 else r['slots']['ED']
                generations.append({'id':r['id'],'branch':r['branch'],'style':r['style'],'task':r['task'],'text':text,'expected':expected,'other':other,'lexical_fidelity':expected in text and other not in text})
            log('free_generation_preview',min(off+4,len(choices)),len(choices),start)
    passed=early_error==0 and np.mean([r['lexical_fidelity'] for r in generations])>=.8
    save(OUT/'calibration_controls.json',{'kind':'real','passed':bool(passed),'late_cue_prefix_max_error':early_error,'gva_error':captured['gva_max_abs_error'],'logit_error':captured['logit_max_abs_error'],'free_preview':generations,'preview_lexical_fidelity':float(np.mean([r['lexical_fidelity'] for r in generations])),'scope':'24 discovery-pilot greedy previews, lexical screen plus readable texts; not a semantic benchmark. Earlier identical causal prefixes must be bitwise identical.','model_snapshot':read(OUT/'prereg.json')['model_snapshot'],'script_sha256':file_sha(__file__)})
    assert passed,'calibration controls failed'
    save(OUT/'formal_execution.json',{'prereg_sha256':file_sha(OUT/'prereg.json'),'collector_sha256':file_sha(__file__),'pilot_sha256':file_sha(OUT/'pilot'/'summary.json'),'controls_sha256':file_sha(OUT/'calibration_controls.json'),'note':'Engineering additions after initial pilot: bounded lease waiting, Unicode-safe resource logging, prefill-only patches, and calibration controls. Frozen gates unchanged; no confirm exposed.'})
    print(json.dumps({'controls_passed':bool(passed),'early_error':early_error,'free_lexical':float(np.mean([r['lexical_fidelity'] for r in generations]))}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','pilot','controls','discovery','confirm']);args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='controls':controls()
    else:collect(args.command)

if __name__=='__main__':main()
